//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/scheduling/vf_allocate_linear.hpp"

#include "vpux/compiler/utils/linear_scan.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"
#include "vpux/utils/core/developer_build_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include <algorithm>
#include <utility>

using namespace vpux;

namespace {
using ExcludedRangeOffsetAndSize = std::pair<vpux::AddressType, vpux::AddressType>;

struct PersistentBufferPlacement {
    vpux::AddressType alignedOffset;
    vpux::AddressType consumedSpan;
};

PersistentBufferPlacement packPersistentBuffer(vpux::AddressType currentOffset, const BufferRecord& rec) {
    const auto alignedOffset = vpux::alignValUp(currentOffset, rec.offsetAlignment);
    return {alignedOffset, (alignedOffset - currentOffset) + rec.size};
}

// Walk the built schedule's alloc/dealloc entries in order and compute the true peak
// concurrently-live byte count (ideal address packing, no fragmentation). Unlike a live
// set tracked only at tryAllocTemporary() call sites, this also captures PERSISTENT_CANDIDATE
// buffers evicted mid-schedule: their retroactive allocation is recorded directly as an
// entry in iterationSchedule (see findAndSpillPersistentBuffer), so it is picked up here too.
//
// Ordering matches verifyLocalSchedule(): deallocations of an entry are applied BEFORE its
// allocations, since eviction always frees a buffer before the retry that needs the space
// succeeds (same real-time ordering the entry's alloc/dealloc lists were recorded in).
//
// Diagnostic-only: also flags genuine CMX address overlaps among buffers considered live at
// the same instant (would indicate a real correctness bug, not just a fragmentation gap) versus
// aliased root values that legitimately share one address (not a bug, just double-counted here).
vpux::AddressType computeTruePeakConcurrentBytes(const IterationSchedule& iterationSchedule,
                                                 const ClassifierVFResult& classifier, Logger log) {
    if (!log.isActive(LogLevel::Trace) || !isDeveloperBuild()) {
        return 0;
    }

    struct LiveBuf {
        vpux::AddressType addr;
        vpux::AddressType size;
    };
    struct AddressRange {
        vpux::AddressType begin;
        vpux::AddressType end;
        mlir::Value value;
    };
    llvm::DenseMap<mlir::Value, LiveBuf> live;
    vpux::AddressType sum = 0;
    vpux::AddressType peak = 0;
    for (const auto& entry : iterationSchedule) {
        for (auto buf : entry.deallocations) {
            const auto it = live.find(buf);
            if (it == live.end()) {
                continue;
            }
            sum -= it->second.size;
            live.erase(it);
        }
        for (const auto& allocPair : entry.allocations) {
            const auto& buf = allocPair.first;
            if (live.count(buf)) {
                continue;
            }
            const auto it = classifier.valueToRecord.find(buf);
            if (it == classifier.valueToRecord.end()) {
                continue;
            }
            const auto size = classifier.buffers[it->second].size;
            live[buf] = {allocPair.second, size};
            sum += size;
            if (sum > peak) {
                peak = sum;
                SmallVector<AddressRange> ranges;
                ranges.reserve(live.size());
                for (const auto& [v, lb] : live) {
                    ranges.push_back({lb.addr, lb.addr + lb.size, v});
                }
                std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) {
                    return lhs.begin < rhs.begin;
                });
                for (size_t i = 1; i < ranges.size(); ++i) {
                    if (ranges[i].begin < ranges[i - 1].end) {
                        log.trace("computeTruePeakConcurrentBytes: ADDRESS OVERLAP at peak sum={0}: [{1},{2}) "
                                  "value={3} overlaps [{4},{5}) value={6}",
                                  sum, ranges[i - 1].begin, ranges[i - 1].end, ranges[i - 1].value, ranges[i].begin,
                                  ranges[i].end, ranges[i].value);
                    }
                }
            }
        }
    }
    return peak;
}

// Clip excluded ranges to the intersection with currently free gaps.
// allocWithExcludedRegion routes each range through Partitioner::allocFixed,
// which requires the range to lie entirely inside a free gap. Parts of an
// excluded range that overlap already-allocated memory must be dropped;
// otherwise allocFixed's precondition is violated. Multiple sub-ranges may
// be produced for a single input if it straddles allocated regions.
SmallVector<ExcludedRangeOffsetAndSize> intersectExcludedWithGaps(ArrayRef<ExcludedRangeOffsetAndSize> excluded,
                                                                  ArrayRef<vpux::Partitioner::Gap> gaps) {
    SmallVector<ExcludedRangeOffsetAndSize> trimmed;
    for (const auto& [addr, size] : excluded) {
        const auto excEnd = addr + size;
        for (const auto& gap : gaps) {
            if (gap.begin >= excEnd) {
                break;  // gaps are sorted by begin; no further overlap possible
            }
            if (gap.end <= addr) {
                continue;
            }
            const auto interBegin = std::max(gap.begin, addr);
            const auto interEnd = std::min(gap.end, excEnd);
            trimmed.emplace_back(interBegin, interEnd - interBegin);
        }
    }
    return trimmed;
}
}  // namespace

vpux::VfAllocateLinear::VfAllocateLinear(const ComputeRegion& region, const ClassifierVFResult& classifier,
                                         AddressType memoryLimit, const VfSchedStrategyDescriptor& params, Logger log)
        : _region(region),
          _classifier(classifier),
          _memoryLimit(memoryLimit),
          _params(params),
          _log(std::move(log)),
          _linearScan(/*size=*/memoryLimit,
                      /*reservedVec=*/{},
                      /*alignment=*/static_cast<uint64_t>(vpux::DEFAULT_CMX_ALIGNMENT)) {
    _log.setName("vf-allocate-linear");
}

// Spilling cost is determined by the size of buffer
size_t vpux::VfAllocateLinear::getSpillCost(size_t bufferSize) {
    return bufferSize;
}

// Highest temporary address touched so far.
//
// Split in two parts, because a prefetch reservation is not final: it can still
// be handed back by cancelPendingPrefetch(), in which case the buffer is never
// placed at that address and must leave no trace in the reported peak.
//   * `committedHighWater` - monotonic mark of allocations that are part of the
//     schedule for good (regular allocations, downgraded persistents and
//     prefetches whose allocation entry was already emitted)
//   * still-pending prefetches - accounted for only as long as they are pending,
//     so a cancelled one drops out of the mark automatically without having to
//     restore any snapshot
//
// A snapshot-and-restore scheme would not work here: committed allocations can
// raise the mark after a prefetch was reserved (a DATA_IN op of the same window
// allocating a non-prefetched buffer, or a persistent eviction), and restoring
// the pre-prefetch value on cancellation would then drop those legitimate
// contributions too.
vpux::AddressType vpux::VfAllocateLinear::effectiveTemporaryHighWater(AddressType committedHighWater,
                                                                      ArrayRef<PendingPrefetch> pendingPrefetch) {
    auto highWater = committedHighWater;
    for (const auto& pending : pendingPrefetch) {
        highWater = std::max(highWater, pending.topAddr);
    }
    return highWater;
}

// Evict the cheapest still-admitted persistent. Returns true if
// an eviction happened (the caller should retry the failing alloc).
// Spill of a persistent buffer is done differently from a temporary spill:
// besides marking the buffer as spilled it also reduces the reserved
// persistent range so the linear scan can reuse the freed memory. The evicted
// buffer is retroactively recorded as if it had been allocated at its first
// use inside the persistent zone, then deallocated at the current op.
bool vpux::VfAllocateLinear::findAndSpillPersistentBuffer(VfAllocateResult& result, ComputeExplicitSchedule& entry,
                                                          size_t currentOpPos, ArrayRef<mlir::Value> inOpBuffers,
                                                          ArrayRef<mlir::Value> outOpBuffers,
                                                          SmallVector<BufferRecord>& admittedPersistentCand,
                                                          AddressType& reservedSize, AddressType& persistentFloor,
                                                          AddressType& committedHighWater) {
    if (admittedPersistentCand.empty()) {
        _log.trace("No persistent candidate left to evict");
        return false;
    }

    // Pick smallest-cost admitted persistent (admission order is DESC by cost,
    // so back-to-front traversal visits candidates cheapest first). Skip the
    // current op's own inputs/outputs — evicting them would defeat the retry.
    BufferRecord persistedBuffer;
    persistedBuffer.rootValue = nullptr;
    for (auto it = admittedPersistentCand.rbegin(); it != admittedPersistentCand.rend(); ++it) {
        const auto& buffer = it->rootValue;
        if (llvm::is_contained(inOpBuffers, buffer) || llvm::is_contained(outOpBuffers, buffer)) {
            continue;
        }

        // Skipping current-op buffers can change the modeled back-to-front
        // packing order. Repack the selected candidate first, followed by all
        // remaining persistents in their normal eviction order, and accept the
        // candidate only when that complete layout still fits in the currently
        // reserved zone. The zone cannot be enlarged here because temporary
        // allocations may already occupy the memory below persistentFloor.
        auto candidatePlacement = packPersistentBuffer(persistentFloor, *it);
        auto packedEnd = candidatePlacement.alignedOffset + it->size;
        for (auto remainingIt = admittedPersistentCand.rbegin(); remainingIt != admittedPersistentCand.rend();
             ++remainingIt) {
            if (remainingIt == it) {
                continue;
            }
            const auto remainingPlacement = packPersistentBuffer(packedEnd, *remainingIt);
            packedEnd = remainingPlacement.alignedOffset + remainingIt->size;
        }
        if (packedEnd > persistentFloor + reservedSize) {
            _log.trace("Cannot evict persistent value={0}: reordered packed end {1} exceeds reserved-zone end {2}",
                       buffer, packedEnd, persistentFloor + reservedSize);
            continue;
        }

        persistedBuffer = *it;
        admittedPersistentCand.erase(std::next(it).base());
        break;
    }
    if (persistedBuffer.rootValue == nullptr) {
        _log.trace("No persistent candidate left to evict");
        return false;
    }

    // Address inside the persistent zone the evicted buffer occupies. Alignment
    // applies to the OFFSET, so raise `persistentFloor` to a
    // `persistedBuffer.offsetAlignment`-aligned address; the gap between the old
    // floor and the aligned offset is unusable and counts as freed persistent
    // space alongside the buffer size itself.
    const auto placement = packPersistentBuffer(persistentFloor, persistedBuffer);
    const auto alignedOffset = placement.alignedOffset;

    // Retroactive alloc entry at the buffer's first template use so downstream
    // consumers see the persistent slot as a TEMPORARY at `alignedOffset` up to
    // the point of eviction. If the buffer has not been used yet in the
    // template, its future first use will go through the normal tryAlloc path
    // (guided by `wasEverSpilled` below).
    const auto firstUseOp = persistedBuffer.firstUseOp;
    const auto noTemplateUse = std::numeric_limits<size_t>::max();
    if (firstUseOp != noTemplateUse && firstUseOp < currentOpPos) {
        VPUX_THROW_UNLESS(firstUseOp < result.iterationSchedule.size(),
                          "Spilled persistent firstUseOp {0} outside iteration schedule (size={1})", firstUseOp,
                          result.iterationSchedule.size());
        result.iterationSchedule[firstUseOp].allocations.push_back({persistedBuffer.rootValue, alignedOffset});
        // Since non longer persistent buffer is now allocated as temporary update
        // the committed high-water mark
        committedHighWater = std::max(committedHighWater, alignedOffset + persistedBuffer.size);
        _log.trace("Downgraded persistent, insert alloc at {0}, value={1} addr={2}, committedHighWater={3}", firstUseOp,
                   persistedBuffer.rootValue, alignedOffset, committedHighWater);
    }

    _linearScan.handler().spilled(persistedBuffer.rootValue);
    entry.deallocations.push_back(persistedBuffer.rootValue);
    result.spilledBuffers.push_back(persistedBuffer.rootValue);

    // Persistent zone shrinks by the alignment gap plus the buffer size.
    const auto freed = admittedPersistentCand.empty() ? reservedSize : placement.consumedSpan;
    VPUX_THROW_WHEN(freed > reservedSize, "Evicted persistent buffer would free {0} bytes but reservedSize is only {1}",
                    freed, reservedSize);
    reservedSize -= freed;
    persistentFloor += freed;

    _log.trace("Op {0} dealloc (Spill) PERSISTENT value={1} (persistentFloor now {2})", currentOpPos,
               persistedBuffer.rootValue, persistentFloor);

    return true;
}

// Find and spill temporary buffer. Deallocate range in linear scan
// and record deallocation in compute schedule entry
bool vpux::VfAllocateLinear::findAndSpillTemporaryBuffer(
        ComputeExplicitSchedule& entry, size_t currentOpPos, ArrayRef<mlir::Value> inOpBuffers,
        ArrayRef<mlir::Value> outOpBuffers, ValueOrderedSet& liveTemporary, ValueOrderedSet& speculativeReservations,
        ArrayRef<PendingPrefetch> pendingPrefetch, VfAllocateResult& result) {
    // Pick a TEMPORARY buffer for DDR spill. Choice rule: largest
    // size first, then longest remaining use window (lastUseOp - currentOpPos).
    // Buffers that are inputs or outputs of the operation being scheduled are
    // currently excluded
    // TODO: Add support for relocation spilling - E#236129
    // retry (the very next attempt needs them resident in CMX).
    // TODO: Update the logic to consider required space schedule operation
    // instead of just looking for largest buffer to spill. Other factors
    // like distance to next use may also give benefits
    mlir::Value bufferToSpill;
    vpux::AddressType bestSize = 0;
    size_t bestRemainingWindow = 0;
    for (auto buffer : liveTemporary) {
        if (llvm::is_contained(inOpBuffers, buffer) || llvm::is_contained(outOpBuffers, buffer)) {
            continue;
        }
        // A buffer placed ahead of its producing DATA_IN op holds no data yet, so
        // there is nothing to spill — the downstream scheduler cannot emit a
        // spill write for a buffer whose producer has not run. Such buffers are
        // given back by cancelPendingPrefetch(), which the eviction cascade runs
        // before reaching this point.
        const bool isPendingPrefetch = llvm::any_of(pendingPrefetch, [&](const PendingPrefetch& pending) {
            return pending.buffer == buffer;
        });
        if (isPendingPrefetch) {
            continue;
        }
        const auto it = _classifier.valueToRecord.find(buffer);
        if (it == _classifier.valueToRecord.end()) {
            continue;
        }
        const auto& bufferRec = _classifier.buffers[it->second];
        const size_t remaining = (bufferRec.lastUseOp > currentOpPos) ? (bufferRec.lastUseOp - currentOpPos) : 0u;
        const auto size = std::max<vpux::AddressType>(bufferRec.size, 1);
        const bool better = (size > bestSize) || (size == bestSize && remaining > bestRemainingWindow);
        if (better) {
            bufferToSpill = buffer;
            bestSize = size;
            bestRemainingWindow = remaining;
        }
    }

    if (!bufferToSpill) {
        _log.trace("No live temporary to spill at op {0}", currentOpPos);
        return false;
    }
    liveTemporary.erase(bufferToSpill);
    _linearScan.handler().markAsDead(bufferToSpill);
    _linearScan.freeDeadRanges();

    // Last-compute overlap can reserve a future output before its producer runs.
    //
    //                         Schedule order --->
    //                  Earlier op       Current op       Future producer
    //                  --------------   --------------   ----------------
    // Before:          reserve B ------> spill B (!) ---> produce B
    // After:           [alloc removed]   reuse space      allocate/produce B
    //
    // (!) B has no contents yet, so there is nothing to spill. Remove its early
    // allocation from the schedule; its producer will allocate it when reached.
    // Track those reservations explicitly. Absence of an already-scheduled
    // producer is not sufficient: a downgraded persistent external input also
    // has no in-loop producer, but its allocation contains valid data and must
    // follow the normal spill path.
    if (speculativeReservations.erase(bufferToSpill)) {
        for (auto& scheduledEntry : result.iterationSchedule) {
            llvm::erase_if(scheduledEntry.allocations, [&](const auto& allocation) {
                return allocation.first == bufferToSpill;
            });
        }
        _log.trace("Op {0} cancel early reservation for TEMP value={1}", currentOpPos, bufferToSpill);
        return true;
    }

    _linearScan.handler().spilled(bufferToSpill);
    // Record the spill in the current entry's deallocations so
    // the verifier and downstream consumers see the victim's
    // CMX bytes as freed at this opPos. Without this, a
    // subsequent alloc at the same offset (this very iter)
    // overlaps the still-recorded live range.
    entry.deallocations.push_back(bufferToSpill);
    result.spilledBuffers.push_back(bufferToSpill);
    _log.trace("Op {0} dealloc (Spill) TEMP value={1}", currentOpPos, bufferToSpill);
    return true;
}

// Free dead buffers that are not used anymore and create deallocation entry
// in compute schedule
void vpux::VfAllocateLinear::freeDeadAtOp(size_t opPos, ComputeExplicitSchedule& entry,
                                          ValueOrderedSet& liveTemporary) {
    SmallVector<mlir::Value, 8> bufferNotUsedAnymore;
    for (auto buffer : liveTemporary) {
        const auto it = _classifier.valueToRecord.find(buffer);
        if (it == _classifier.valueToRecord.end()) {
            continue;
        }
        const auto& bufferRec = _classifier.buffers[it->second];
        if (bufferRec.lastUseOp < opPos) {
            bufferNotUsedAnymore.push_back(buffer);
        }
    }
    for (auto buffer : bufferNotUsedAnymore) {
        liveTemporary.erase(buffer);
        _linearScan.handler().markAsDead(buffer);
        entry.deallocations.push_back(buffer);
        _log.trace("Op {0} dealloc TEMP value={1}", opPos, buffer);
    }
    if (!bufferNotUsedAnymore.empty()) {
        _linearScan.freeDeadRanges();
        _log.trace("Op {0} freed dead ranges (count={1})", opPos, bufferNotUsedAnymore.size());
    }
}

// Try one TEMPORARY allocation; returns the address on success or
// InvalidAddress on failure. `extraExcludedCandidates` reserves additional
// (addr, size) ranges. Attempts are made from the largest suffix down: first
// with the full set, then dropping the last entry on each retry until the
// set is empty. The persistent-zone exclusion (baseExcluded) is preserved
// on every attempt. Later entries (added last) are the first to be
// sacrificed, so callers should push lower-priority ranges to the tail.
// Prefetches require the full exclusion set and fail instead of relaxing it.
vpux::AddressType vpux::VfAllocateLinear::tryAllocTemporary(
        mlir::Value out, AddressType persistentFloor,
        ArrayRef<std::pair<AddressType, AddressType>> extraExcludedCandidates, bool requireExtraExcluded) {
    auto& scanHandler = _linearScan.handler();
    scanHandler.markAsAlive(out);

    // Persistent zone occupies [persistentFloor, memoryLimit). Skip the
    // excluded-region argument entirely when the persistent zone is empty
    // — Partitioner::allocFixed asserts size>0 on the excluded range.
    SmallVector<ExcludedRangeOffsetAndSize> baseExcluded;
    if (persistentFloor < _memoryLimit) {
        baseExcluded.push_back({persistentFloor, _memoryLimit - persistentFloor});
    }

    // Trim to free gaps: allocWithExcludedRegion routes each range through
    // Partitioner::allocFixed and portions overlapping already-allocated memory
    // (e.g. extraExcludedCandidates that refer to buffers currently live in CMX)
    // would violate that precondition. Buffers already live are naturally avoided
    // by alloc(), so dropping their addresses from the exclusion set is safe.
    auto tryAlloc = [&](ArrayRef<ExcludedRangeOffsetAndSize> excluded) {
        const auto trimmed = intersectExcludedWithGaps(excluded, _linearScan.gaps());
        if (trimmed.empty()) {
            return _linearScan.alloc(std::initializer_list<mlir::Value>{out},
                                     /*allowSpills=*/false, DirectionT::Up);
        }
        return _linearScan.allocWithExcludedRegion(trimmed, std::initializer_list<mlir::Value>{out},
                                                   /*allowSpills=*/false, DirectionT::Up);
    };

    bool allocStatus = false;
    // Iteratively shrink the extras from the tail down to zero. baseExcluded
    // is preserved on every attempt — only extras are sacrificed. Later
    // extras (pushed last) are lowest priority and dropped first.
    for (size_t i = extraExcludedCandidates.size() + 1; i > 0 && !allocStatus; --i) {
        const size_t numOfExcludeRanges = i - 1;
        SmallVector<ExcludedRangeOffsetAndSize> excludedRegion(extraExcludedCandidates.begin(),
                                                               extraExcludedCandidates.begin() + numOfExcludeRanges);
        excludedRegion.append(baseExcluded.begin(), baseExcluded.end());
        _log.trace("Try TEMP alloc with {0} extra excluded range(s) - [offset, size]: {1}", numOfExcludeRanges,
                   excludedRegion);
        allocStatus = tryAlloc(excludedRegion);
        if (!allocStatus) {
            _log.nest().trace("Allocation failed");
        }
        if (requireExtraExcluded) {
            break;
        }
    }

    if (!allocStatus) {
        scanHandler.markAsDead(out);
        _log.trace("TEMP alloc failed value={0} (persistentFloor={1})", out, persistentFloor);
        return InvalidAddress;
    }
    const auto addr = scanHandler.getAddress(out);
    _log.trace("TEMP alloc success at addr={0}", addr);
    return addr;
}

// Collect, per template position, the buffers loaded by the DATA_IN ops that
// feed the NEXT COMPUTE op. Only COMPUTE positions get a non-empty list.
//
// The window of a COMPUTE op ends at the next COMPUTE op, so the DATA_IN ops
// inside it are exactly the ones loading the next COMPUTE op's inputs:
//
//   0_DATA_IN 1_DATA_IN -> 2_COMPUTE -> [3_DATA_IN 4_DATA_IN] -> 5_COMPUTE
//                                        \__ window of 2_COMPUTE __/
//
// Every buffer loaded by a DATA_IN op of the window is a candidate. Buffers
// without a classifier record are skipped - the allocator does not track them.
vpux::VfAllocateLinear::PrefetchCandidates vpux::VfAllocateLinear::collectPrefetchCandidates(
        const LoopBody& templateBody) const {
    PrefetchCandidates candidates(templateBody.size());

    for (size_t opPos = 0; opPos < templateBody.size(); ++opPos) {
        if (templateBody[opPos].allocationType != AllocationType::COMPUTE) {
            continue;
        }
        for (size_t nextPos = opPos + 1; nextPos < templateBody.size(); ++nextPos) {
            const auto& nextInfo = templateBody[nextPos];
            if (nextInfo.allocationType == AllocationType::COMPUTE) {
                break;
            }
            if (nextInfo.allocationType != AllocationType::DATA_IN) {
                continue;
            }
            for (auto buf : nextInfo.outBuffers) {
                const auto recIt = _classifier.valueToRecord.find(buf);
                if (recIt == _classifier.valueToRecord.end()) {
                    continue;
                }
                candidates[opPos].push_back({buf, nextInfo.opIdx});
            }
        }
        for (const auto& candidate : candidates[opPos]) {
            _log.trace("Op {0} prefetch candidate for opIdx={1}, value={2}", opPos, candidate.producerOpIdx,
                       candidate.buffer);
        }
    }
    return candidates;
}

// Reserve ranges for the next COMPUTE op's DATA_IN buffers while the buffers of
// the COMPUTE op at `opPos` are still live.
//
// The linear scan hands out only ranges that no live buffer occupies, so every
// range reserved here is disjoint from all input and output buffers of the
// current COMPUTE op. That is exactly the condition which lets those DATA_IN
// DMAs execute in parallel with the current COMPUTE op instead of waiting for
// its memory to be released.
//
// Only the address is decided here. The allocation entry itself is emitted at
// the DATA_IN op's own schedule entry by emitPrefetchedAllocations(), see the
// comment there.
//
// Best effort: nothing is spilled or evicted to make room. A candidate that
// does not fit is left to the regular allocation path at its own template
// position, where freeDeadAtOp() has already released the COMPUTE op's dead
// buffers — "if there is not enough free memory to prefetch then first free and
// then allocate". The remaining candidates are still attempted, so a partial
// prefetch is possible.
//
// Candidates are kept in program order rather than reordered by size: DMAs of
// one queue execute in issue order, so a candidate left behind stalls every
// later DMA of that queue anyway. Prefetching a prefix is what buys overlap,
// maximizing prefetched bytes is not.
void vpux::VfAllocateLinear::prefetchNextComputeInputs(
        size_t opPos, ArrayRef<PrefetchCandidate> candidates, AddressType persistentFloor,
        const llvm::DenseSet<mlir::Value>& overlapSensitiveBufs,
        ArrayRef<std::pair<AddressType, AddressType>> firstInputExcludedRanges, ValueOrderedSet& liveTemporary,
        SmallVector<PendingPrefetch>& pendingPrefetch, AddressType committedHighWater) {
    auto& scanHandler = _linearScan.handler();

    for (const auto& candidate : candidates) {
        const auto buf = candidate.buffer;
        const auto producerOpIdx = candidate.producerOpIdx;
        const auto recIt = _classifier.valueToRecord.find(buf);
        VPUX_THROW_WHEN(recIt == _classifier.valueToRecord.end(),
                        "Prefetch candidate has no classifier record (op={0})", opPos);
        if (liveTemporary.count(buf) != 0) {
            // Already placed, e.g. by the last-compute overlap co-allocation.
            _log.trace("Op {0} prefetch for opIdx={1} skip, buffer already alive value={2}", opPos, producerOpIdx, buf);
            continue;
        }
        if (scanHandler.getCategory(buf) == BufferCategory::PERSISTENT_CANDIDATE && !scanHandler.wasEverSpilled(buf)) {
            // Persistent: address is owned by the outer scheduler, it is resident
            // for the whole loop and needs no prefetch.
            _log.trace("Op {0} prefetch for opIdx={1} skip persistent value={2}", opPos, producerOpIdx, buf);
            continue;
        }

        // Last-compute overlap: a prefetched buffer consumed by the last COMPUTE
        // op must keep avoiding the first-compute input ranges, otherwise the
        // next iteration's DATA_IN DMAs could no longer overlap with it.
        const ArrayRef<ExcludedRangeOffsetAndSize> extraExcludeRanges =
                overlapSensitiveBufs.contains(buf) ? firstInputExcludedRanges : ArrayRef<ExcludedRangeOffsetAndSize>{};

        const auto addr = tryAllocTemporary(buf, persistentFloor, extraExcludeRanges, /*requireExtraExcluded=*/true);
        if (addr == InvalidAddress) {
            _log.trace("Op {0} prefetch for opIdx={1} not possible, no free range for value={2}", opPos, producerOpIdx,
                       buf);
            continue;
        }

        liveTemporary.insert(buf);
        const auto recSize = _classifier.buffers[recIt->second].size;
        // The reserved range raises the effective high-water mark only while the
        // prefetch is pending - it is folded into the committed mark by
        // emitPrefetchedAllocations() and dropped by cancelPendingPrefetch().
        pendingPrefetch.push_back({buf, producerOpIdx, opPos, addr, addr + recSize});
        _log.trace("Op {0} prefetch range reserved for opIdx={1} value={2} addr={3} size={4} highWater={5}", opPos,
                   producerOpIdx, buf, addr, recSize, effectiveTemporaryHighWater(committedHighWater, pendingPrefetch));
    }
}

// Emit the allocation entries of buffers this op produces and the prefetch step
// already placed, and drop them from the pending list.
//
// The entry is deliberately emitted here and not at the COMPUTE op the prefetch
// was done for: every schedule entry must allocate exactly the buffers its own
// op produces. `verifyLocalSchedule` checks that invariant, and the pipelined
// pairwise rewrite (rewriteVFPipelinedScheduleIntoPairwiseSteps) moves whole
// compute stages together with their input DMA entries across iterations -
// an allocation left behind in another op's entry would end up in a different
// iteration than the DMA producing the buffer. The overlap the prefetch buys
// comes from the ADDRESS, which was already chosen while the COMPUTE op's
// buffers were live, not from where the entry is recorded.
//
// From this point the buffer is an ordinary live temporary: its producer is part
// of the schedule, so it may also be spilled if memory runs out later.
void vpux::VfAllocateLinear::emitPrefetchedAllocations(size_t opPos, ArrayRef<mlir::Value> outOpBuffers,
                                                       ComputeExplicitSchedule& entry,
                                                       SmallVector<PendingPrefetch>& pendingPrefetch,
                                                       AddressType& committedHighWater) {
    llvm::erase_if(pendingPrefetch, [&](const PendingPrefetch& pending) {
        if (!llvm::is_contained(outOpBuffers, pending.buffer)) {
            return false;
        }
        entry.allocations.push_back({pending.buffer, pending.addr});
        // The reservation became a real allocation entry, so it can no longer be
        // cancelled - its range now belongs to the committed high-water mark.
        committedHighWater = std::max(committedHighWater, pending.topAddr);
        _log.trace("Op {0} emit prefetched allocation for opIdx={1} value={2} addr={3} (prefetched at op {4}), "
                   "committedHighWater={5}",
                   opPos, pending.producerOpIdx, pending.buffer, pending.addr, pending.prefetchOpPos,
                   committedHighWater);
        return true;
    });
}

// Give back the range reserved by the most recent still-pending prefetch.
// Returns true if a prefetch was cancelled (the caller should retry the
// allocation that failed).
//
// Cancelling costs nothing at runtime: the buffer's DATA_IN DMA has not been
// scheduled yet, so no data is lost and nothing was recorded in the schedule -
// the buffer is simply placed by the regular path at its own template position
// and loses only the overlap with the current COMPUTE op. That makes this the
// cheapest step of the eviction cascade, ahead of persistent eviction and
// temporary spilling which both add DMA traffic.
//
// Cancellation is LIFO: the last prefetched buffer is the one whose DMA is
// furthest in the future, so it is the least valuable one to keep.
//
// The cancelled buffer also stops contributing to the temporary high-water mark:
// popping it from `pendingPrefetch` is all that takes, because a pending
// reservation is never folded into the committed mark - see
// effectiveTemporaryHighWater(). Leaving it in would report a peak for a range
// that no op of the final schedule ever occupies. The buffer is placed again by
// the regular path at its own template position, where it contributes to the
// mark with the address it really gets.
bool vpux::VfAllocateLinear::cancelPendingPrefetch(SmallVector<PendingPrefetch>& pendingPrefetch,
                                                   ValueOrderedSet& liveTemporary, size_t currentOpPos,
                                                   AddressType committedHighWater) {
    if (pendingPrefetch.empty()) {
        return false;
    }
    const auto pending = pendingPrefetch.pop_back_val();

    liveTemporary.erase(pending.buffer);
    _linearScan.handler().markAsDead(pending.buffer);
    _linearScan.freeDeadRanges();

    _log.trace("Op {0} cancel prefetch for opIdx={1} of value={2} (was prefetched at op {3}, addr={4}), highWater={5}",
               currentOpPos, pending.producerOpIdx, pending.buffer, pending.prefetchOpPos, pending.addr,
               effectiveTemporaryHighWater(committedHighWater, pendingPrefetch));
    return true;
}

// Perform allocation on provided compute region with linear scan algorithm
//
// Design notes:
//   * Single-pass linear scan with no backtracking. One forward iteration over
//     the template iteration's op stream - first loop body. Update in IterationSchedule
//     entries with allocation and deallocation points of all buffers
//   * Take buffer data from classifier result
//   * Persistent zone: reserved as ONE block. Not handled directly by
//     the linear scan and treated as reserved range
//   * In case of insufficient memory to allocate a new temporary buffer, try to evict
//     Eviction cascade:
//     - Step 1: Cancel pending prefetches (cheapest - no data is lost, see cancelPendingPrefetch)
//     - Step 2: Evict persistent candidates until enough space to allocated
//               new temporary buffer. Spilled persistent buffer is no longer treated as persistent.
//               Shrink persistent reserved range
//     - Step 3: If eviction of all persistent candidates does not free enough space, spill TEMPORARY buffers
//   * LastComputeOverlapWithNextIterInputs=ENABLED heuristic: outBuffers of the LAST op in template iteration occupy
//     different range than iteration input buffer to allow execution overlap of with iteration
//   * IntraIterFetch=ENABLED heuristic (prefetching within an iteration): right after a COMPUTE op is scheduled -
//     and before the next ops release its dead buffers - reserve ranges for the DATA_IN buffers of the NEXT
//     COMPUTE op. Those ranges are then guaranteed to be disjoint from all buffers of the current COMPUTE
//     op, which is what allows those DMAs to run in parallel with it. The allocation entry is still emitted
//     at the DATA_IN op's own schedule entry, so every entry keeps allocating exactly its own outputs.
//     See collectPrefetchCandidates(), prefetchNextComputeInputs() and emitPrefetchedAllocations()
//
vpux::VfAllocateResult vpux::VfAllocateLinear::performAllocation() {
    _log.trace("Start (memoryLimit={0}, iterations={1}, opsPerIter={2}, strategy(IntraIterFetch={3}, "
               "LastComputeOverlapWithNextIterInputs={4})",
               _memoryLimit, _classifier.numIterations, _classifier.opsPerIteration,
               static_cast<int>(_params.intraIterFetch), static_cast<int>(_params.lastCompOverlapNextIterIn));

    VfAllocateResult result;

    // ---- Perform initial checks -----------------------------------------------------
    if (_region.schedulingLoop == nullptr) {
        _log.trace("Region has no scheduling loop");
        return result;
    }
    const auto& loop = *_region.schedulingLoop;
    if (loop.type != LoopType::VF || loop.loopBodies.empty() || loop.loopBodies.front().empty()) {
        _log.trace("Empty VF loop");
        return result;
    }
    if (!_classifier.classifierSupportedRegion) {
        _log.trace("Classifier unsupported region, refusing");
        return result;
    }
    if (_classifier.numIterations == 0 || _classifier.opsPerIteration == 0) {
        _log.trace("Empty classification");
        return result;
    }
    if (_memoryLimit == 0) {
        _log.trace("Memory limit is 0");
        return result;
    }

    const auto& templateBody = loop.loopBodies.front();
    auto& scanHandler = _linearScan.handler();

    const bool isLastCompOverlapNextIterInEn = _params.lastCompOverlapNextIterIn ==
                                               VfSchedStrategyDescriptor::LastComputeOverlapWithNextIterInputs::ENABLED;

    // ENABLED last-compute overlap: pipelined-loop execution requires that the region
    // outputs (drained by DATA_OUT ops after the last compute op) and the region
    // inputs loaded by DATA_IN ops preceding the first compute op occupy disjoint
    // CMX ranges. Collect both sets up front for downstream use. The last-compute
    // op's own input buffers are tracked separately, so that any inputs still live
    // at that point are steered away from first-compute input ranges as well.
    SmallVector<mlir::Value> lastComputeOpOutBufsRevOrder;
    SmallVector<mlir::Value> lastComputeOpInBufsOrder;
    SmallVector<mlir::Value> firstInputBufsOrdered;
    if (isLastCompOverlapNextIterInEn) {
        for (auto it = templateBody.rbegin(); it != templateBody.rend(); ++it) {
            if (it->allocationType == AllocationType::COMPUTE) {
                lastComputeOpInBufsOrder.append(it->inBuffers.begin(), it->inBuffers.end());
                break;
            }
            if (it->allocationType == AllocationType::DATA_OUT) {
                lastComputeOpOutBufsRevOrder.append(it->inBuffers.begin(), it->inBuffers.end());
            }
        }
        for (const auto& info : templateBody) {
            if (info.allocationType == AllocationType::COMPUTE) {
                break;
            }
            if (info.allocationType == AllocationType::DATA_IN) {
                firstInputBufsOrdered.append(info.outBuffers.begin(), info.outBuffers.end());
            }
        }
        // Remove buffers that appear both as first-compute inputs and as last-compute
        // in/out buffers — such a buffer cannot be steered away from itself, so keeping
        // it in both sets would be meaningless (and could confuse the overlap heuristic).
        // Such situation can happen in case of sequence of in-place compute ops where each one
        // reads and writes to same root buffer, probably to different portions of it
        {
            llvm::DenseSet<mlir::Value> firstInputBufsSetTmp(firstInputBufsOrdered.begin(),
                                                             firstInputBufsOrdered.end());
            llvm::DenseSet<mlir::Value> commonBufs;
            for (auto v : lastComputeOpOutBufsRevOrder) {
                if (firstInputBufsSetTmp.contains(v)) {
                    commonBufs.insert(v);
                }
            }
            for (auto v : lastComputeOpInBufsOrder) {
                if (firstInputBufsSetTmp.contains(v)) {
                    commonBufs.insert(v);
                }
            }
            if (!commonBufs.empty()) {
                llvm::erase_if(lastComputeOpOutBufsRevOrder, [&](mlir::Value v) {
                    return commonBufs.contains(v);
                });
                llvm::erase_if(lastComputeOpInBufsOrder, [&](mlir::Value v) {
                    return commonBufs.contains(v);
                });
                llvm::erase_if(firstInputBufsOrdered, [&](mlir::Value v) {
                    return commonBufs.contains(v);
                });
                _log.trace("Removed {0} buffer(s) common to last-compute and first-input sets", commonBufs.size());
            }
        }

        if (!firstInputBufsOrdered.empty() &&
            (!lastComputeOpOutBufsRevOrder.empty() || !lastComputeOpInBufsOrder.empty())) {
            _log.trace("Last-compute overlap candidates: lastComputeOpOutBufs={0}, lastComputeOpInBufs={1}, "
                       "firstInputBufs={2}",
                       lastComputeOpOutBufsRevOrder, lastComputeOpInBufsOrder, firstInputBufsOrdered);
        }
    }
    // Fast-lookup sets used inside the allocation loop to steer overlap-aware
    // placement of last-compute buffers away from first-compute input ranges.
    const llvm::DenseSet<mlir::Value> lastComputeOpOutBufs(lastComputeOpOutBufsRevOrder.begin(),
                                                           lastComputeOpOutBufsRevOrder.end());
    const llvm::DenseSet<mlir::Value> lastComputeOpInBufs(lastComputeOpInBufsOrder.begin(),
                                                          lastComputeOpInBufsOrder.end());
    const llvm::DenseSet<mlir::Value> firstInputBufsSet(firstInputBufsOrdered.begin(), firstInputBufsOrdered.end());
    // (addr, size) ranges of first-compute input buffers as they were allocated
    // — used as extra excluded ranges when placing last-compute outputs.
    SmallVector<ExcludedRangeOffsetAndSize> firstInputExcludedRanges;
    // Buffers whose placement the last-compute overlap heuristic cares about. A
    // prefetch of any of them must respect `firstInputExcludedRanges` too. Both
    // source sets stay empty when the heuristic is disabled.
    const llvm::DenseSet<mlir::Value> overlapSensitiveBufs = [&] {
        llvm::DenseSet<mlir::Value> bufs(lastComputeOpInBufs.begin(), lastComputeOpInBufs.end());
        bufs.insert(lastComputeOpOutBufs.begin(), lastComputeOpOutBufs.end());
        return bufs;
    }();

    // Prefetching within an iteration: DATA_IN buffers of the next COMPUTE op are
    // placed while the current COMPUTE op still owns its buffers, so they end up
    // in non-conflicting ranges and their DMAs can overlap with it.
    const bool isPrefetchEnabled = _params.intraIterFetch == VfSchedStrategyDescriptor::IntraIterFetch::ENABLED;
    const auto prefetchCandidates = isPrefetchEnabled ? collectPrefetchCandidates(templateBody) : PrefetchCandidates{};
    // Buffers placed ahead of their producing DATA_IN op, in prefetch order.
    SmallVector<PendingPrefetch> pendingPrefetch;

    // Register every classifier buffer with the handler so that
    // allocation has knowledge of buffer details like size of VF specific attributes
    for (const auto& bufferRec : _classifier.buffers) {
        if (!bufferRec.rootValue) {
            // Ignore non root SSA values as those are not what allocator tracks and represent only aliases
            continue;
        }

        const size_t spillCost = getSpillCost(bufferRec.size);
        scanHandler.addBufferData(bufferRec.rootValue, bufferRec.category, bufferRec.size, bufferRec.offsetAlignment,
                                  spillCost);
    }

    // Admit candidates greedily (by descending cost) until the next one would
    // not fit alongside the volatile working set's optimistic estimate.
    SmallVector<BufferRecord> admittedPersistentCand;  // sorted by cost DESC (admission order).
    AddressType reservedSize = 0;
    AddressType maxPersistentAlignment = 1;
    for (size_t i = 0; i < _classifier.persistentFitOrder.size(); ++i) {
        const size_t idx = _classifier.persistentFitOrder[i];
        const auto& rec = _classifier.buffers[idx];

        // Persistents are evicted from admittedPersistentCand back-to-front.
        // Simulate that exact order for admission as well: the new candidate is
        // evicted first, followed by the already-admitted candidates in reverse.
        // Packing in forward admission order is not equivalent for mixed
        // alignments and can underestimate the bytes later consumed by eviction.
        auto tentativePlacement = packPersistentBuffer(/*currentOffset=*/0, rec);
        auto tentativePackedSize = tentativePlacement.alignedOffset + rec.size;
        for (auto admittedIt = admittedPersistentCand.rbegin(); admittedIt != admittedPersistentCand.rend();
             ++admittedIt) {
            tentativePlacement = packPersistentBuffer(tentativePackedSize, *admittedIt);
            tentativePackedSize = tentativePlacement.alignedOffset + admittedIt->size;
        }
        if (tentativePackedSize > _memoryLimit) {
            _log.trace("Persistent candidate (size={0}) exceeds memoryLimit, dropped", rec.size);
            continue;
        }
        const auto tentativeMaxAlignment = std::max(maxPersistentAlignment, rec.offsetAlignment);
        const auto unalignedFloor = _memoryLimit - tentativePackedSize;
        const auto tentativeFloor = vpux::alignValDown(unalignedFloor, tentativeMaxAlignment);
        const auto tentativeReservedSize = _memoryLimit - tentativeFloor;
        // Reject candidates that would crowd out the volatile working set.
        if (tentativeReservedSize + _classifier.temporaryPeakBytes > _memoryLimit) {
            _log.trace("Persistent candidate (size={0}) would crowd out volatile peak {1}", rec.size,
                       _classifier.temporaryPeakBytes);
            continue;
        }
        maxPersistentAlignment = tentativeMaxAlignment;
        reservedSize = tentativeReservedSize;
        admittedPersistentCand.push_back(rec);
        _log.trace("AdmittedPersistentBuf={0} category={1} (size={2}, align={3}, spillCost={4}, floorAfter={5})",
                   rec.rootValue, stringifyBufferCategory(rec.category), rec.size, rec.offsetAlignment,
                   getSpillCost(rec.size), _memoryLimit - reservedSize);
    }

    // Downgrade all persistent buffers to TEMPORARY if they are not part
    // of admitted set. This is needed to ensure that those buffers are allocated
    // as normal temporary buffers
    llvm::DenseSet<mlir::Value> persistentChangedToTemporary;
    {
        llvm::DenseSet<mlir::Value> admittedPersistentBufsSet;
        for (const auto& rec : admittedPersistentCand) {
            admittedPersistentBufsSet.insert(rec.rootValue);
        }
        for (const auto& rec : _classifier.buffers) {
            if (rec.category != BufferCategory::PERSISTENT_CANDIDATE || !rec.rootValue) {
                continue;
            }
            if (admittedPersistentBufsSet.contains(rec.rootValue)) {
                continue;
            }

            // Downgrade to TEMPORARY. The buffer is loop-invariant
            // (no cross-iteration dependency), so each iteration re-DMAs
            // its own copy into its own per-iter TEMPORARY slot — same
            // semantics as a per-iteration region input. No shared CMX
            // slot, no pre-region restore — just a normal local
            // allocation
            scanHandler.addBufferData(rec.rootValue, BufferCategory::TEMPORARY, std::max<AddressType>(rec.size, 1),
                                      rec.offsetAlignment, /*spillCost=*/0);
            persistentChangedToTemporary.insert(rec.rootValue);
            _log.trace("Downgrade loop-invariant PC to TEMPORARY (size={0})", rec.size);
        }
        _log.trace("Unadmitted PC processing done (downgraded={0})", persistentChangedToTemporary.size());
    }

    AddressType persistentFloor = _memoryLimit - reservedSize;
    _log.trace("Persistent zone initialized (reservedSize={0}, floor={1})", reservedSize, persistentFloor);

    // Track information about live temporary buffers. This is needed for
    // proper deallocation and eviction handling.
    ValueOrderedSet liveTemporary;
    // Buffers allocated early solely by the last-compute overlap extension.
    // Their CMX ranges are reservations until the actual producer is reached.
    ValueOrderedSet speculativeReservations;
    // Outputs whose producer has already been scheduled. The overlap extension
    // may reload one of these after a spill; that allocation contains real data
    // and must not be treated as a reservation for a future producer.
    llvm::DenseSet<mlir::Value> producedBuffers;

    // High-water mark of the allocations that are final. Ranges reserved by a
    // still-pending prefetch are excluded here (they may be cancelled) and added
    // on top by effectiveTemporaryHighWater().
    AddressType committedHighWater = 0;

    // Iterate over loops template body and schedule operations one by one
    for (size_t opPos = 0; opPos < templateBody.size(); ++opPos) {
        const auto& info = templateBody[opPos];
        ComputeExplicitSchedule entry;
        entry.allocInfo = info;
        entry.templatePos = opPos;
        _log.trace("Op {0} begin (opIdx={1}, inBuffers={2}, outBuffers={3}, liveTemporary={4})", opPos, info.opIdx,
                   info.inBuffers.size(), info.outBuffers.size(), liveTemporary.size());
        _log = _log.nest();

        // Record the allocations of buffers this op produces and an earlier
        // COMPUTE op already prefetched a range for.
        emitPrefetchedAllocations(opPos, info.outBuffers, entry, pendingPrefetch, committedHighWater);

        // Free dead TEMPORARY buffers.
        freeDeadAtOp(opPos, entry, liveTemporary);

        // Last-compute overlap co-allocation heuristic: when the op produces any of the last-compute
        // op's input buffers, first try to co-allocate all last-compute in/out buffers
        // alongside this op's own outputs. If that tentative attempt cannot fit
        // without spilling (or is otherwise infeasible), roll back and retry with
        // only the op's own outputs.
        const bool opProducesLastComputeInput =
                isLastCompOverlapNextIterInEn && !lastComputeOpInBufs.empty() &&
                std::any_of(info.outBuffers.begin(), info.outBuffers.end(), [&](mlir::Value v) {
                    return lastComputeOpInBufs.contains(v);
                });
        const bool isLastComputeOp = isLastCompOverlapNextIterInEn && !lastComputeOpOutBufs.empty() &&
                                     std::any_of(info.outBuffers.begin(), info.outBuffers.end(), [&](mlir::Value v) {
                                         return lastComputeOpOutBufs.contains(v);
                                     });

        bool isSchedOpWithLastComputeOverlapEnabled =
                isLastCompOverlapNextIterInEn && (!lastComputeOpOutBufs.empty() || !lastComputeOpInBufs.empty());
        // Schedule an operation. Retry is possible in case of:
        // - attempting to co-allocate last-compute input buffers to not overlap with first-compute input buffers
        //   (last-compute overlap enabled)
        while (true) {
            // Snapshot the append-only mutable state so a failed scheduling attempt
            // can be rolled back without touching allocations from previous ops.
            const auto allocsSize = entry.allocations.size();
            const auto excludedRangesSize = firstInputExcludedRanges.size();
            const AddressType committedHighWaterSnapshot = committedHighWater;

            SmallVector<mlir::Value> opBuffers = info.outBuffers;
            llvm::DenseSet<mlir::Value> speculativeAttemptBuffers;

            // When last-compute overlap is enabled and scheduling operation which produces inputs for last compute
            // operation try to co-allocate all last-compute input and output buffers alongside this op's own outputs so
            // that in the end next iteration DATA_IN DMAs can overlap with this iteration last compute op and its
            // output DATA_OUT DMAs
            if (isSchedOpWithLastComputeOverlapEnabled && opProducesLastComputeInput) {
                llvm::DenseSet<mlir::Value> alreadyPresent(opBuffers.begin(), opBuffers.end());
                for (auto v : lastComputeOpOutBufsRevOrder) {
                    if (alreadyPresent.insert(v).second) {
                        opBuffers.push_back(v);
                        speculativeAttemptBuffers.insert(v);
                    }
                }
                for (auto v : lastComputeOpInBufsOrder) {
                    if (alreadyPresent.insert(v).second) {
                        opBuffers.push_back(v);
                        speculativeAttemptBuffers.insert(v);
                    }
                }
                _log.trace("Op {0} last-compute overlap extended attempt with last-compute buffers ({1} extra)", opPos,
                           opBuffers.size() - info.outBuffers.size());
            }

            if (isSchedOpWithLastComputeOverlapEnabled && (opProducesLastComputeInput || isLastComputeOp)) {
                // Order buffers based on latest use first to improve last-compute overlap
                // and enable overlapping of last compute op outputs with first compute op inputs.
                // If allocation is done first for latest buffers then it is more likely that they will be allocated
                // in non overlapping way with first compute op inputs
                std::sort(opBuffers.begin(), opBuffers.end(), [&](mlir::Value a, mlir::Value b) {
                    const auto aIt = _classifier.valueToRecord.find(a);
                    const auto bIt = _classifier.valueToRecord.find(b);
                    VPUX_THROW_WHEN(aIt == _classifier.valueToRecord.end() || bIt == _classifier.valueToRecord.end(),
                                    "Buffer has no classifier record (op={0})", opPos);
                    return _classifier.buffers[aIt->second].lastUseOp > _classifier.buffers[bIt->second].lastUseOp;
                });
            }
            // Add input buffer only if it was previously spilled. Input buffers which are
            // not produced as part of a loop body are allocated outside of this allocator
            // This is done after previous sorting step as priority is for having latest buffers
            // to be allocated first to improve likelihood of no-overlap of last buffers with body inputs
            for (auto inBuf : info.inBuffers) {
                if (scanHandler.wasEverSpilled(inBuf) || persistentChangedToTemporary.contains(inBuf)) {
                    opBuffers.push_back(inBuf);
                }
            }

            // In case of last-compute overlap enabled and scheduling operation which produces inputs for last compute
            // operation there is a possibility to rerun scheduling of this op in case first attempt to co-allocate
            // last-compute input and output buffers alongside this op's own outputs fails and requires spilling
            bool retrySchedulingThisOp = false;

            // Allocate input and output buffers. Check if there were already allocated if they were output
            // of previous operations or spill read them if they were evicted before
            for (auto opBuf : opBuffers) {
                const auto recIt = _classifier.valueToRecord.find(opBuf);
                VPUX_THROW_WHEN(recIt == _classifier.valueToRecord.end(), "Buffer has no classifier record (op={0})",
                                opPos);
                const auto category = scanHandler.getCategory(opBuf);
                if (category == BufferCategory::PERSISTENT_CANDIDATE && !scanHandler.wasEverSpilled(opBuf)) {
                    // Persistent: address is owned by the outer scheduler,
                    // no per-op allocation entry needed.
                    _log.trace("Op {0} skip persistent value={1}", opPos, opBuf);
                    continue;
                }
                if (liveTemporary.count(opBuf) != 0) {
                    _log.trace("Op {0} buffer already alive, value={1}", opPos, opBuf);
                    continue;
                }

                // Last-compute overlap: for last-compute buffers pass
                // first-compute input ranges as excluded so pipelined iterations
                // can overlap. tryAllocTemporary internally falls back to a plain
                // allocation if the overlap-aware attempt fails.
                const ArrayRef<ExcludedRangeOffsetAndSize> extraExcludeRanges =
                        isSchedOpWithLastComputeOverlapEnabled &&
                                        (lastComputeOpInBufs.contains(opBuf) || lastComputeOpOutBufs.contains(opBuf))
                                ? ArrayRef<ExcludedRangeOffsetAndSize>(firstInputExcludedRanges)
                                : ArrayRef<ExcludedRangeOffsetAndSize>{};

                AddressType addr = tryAllocTemporary(opBuf, persistentFloor, extraExcludeRanges);

                // Tentative extended attempt bails on any spill need — the whole
                // attempt is redone without the extension.
                if (addr == InvalidAddress && isSchedOpWithLastComputeOverlapEnabled) {
                    _log.trace("Op {0} tentative attempt cannot place value={1} without spill — retry scheduling",
                               opPos, opBuf);
                    retrySchedulingThisOp = true;
                    break;
                }

                // If allocation failed give back the memory reserved by prefetches first. Those
                // buffers carry no data yet, so handing their range back is free — unlike a
                // persistent eviction or a temporary spill, which both cost extra DMA traffic.
                while (addr == InvalidAddress &&
                       cancelPendingPrefetch(pendingPrefetch, liveTemporary, opPos, committedHighWater)) {
                    addr = tryAllocTemporary(opBuf, persistentFloor);
                }

                // If allocation failed try to spill to get more available memory and retry
                // TODO: Unify logic for spilling temporary and persistent buffers. There are possible
                // improvements here as it is not guaranteed that spilling persistend buffers will be better than
                // spilling temporary ones, especially if its cost is small or can be parallelized with compute
                while (addr == InvalidAddress &&
                       findAndSpillPersistentBuffer(result, entry, opPos, info.inBuffers, info.outBuffers,
                                                    admittedPersistentCand, reservedSize, persistentFloor,
                                                    committedHighWater)) {
                    addr = tryAllocTemporary(opBuf, persistentFloor);
                }

                if (addr == InvalidAddress) {
                    while (addr == InvalidAddress &&
                           findAndSpillTemporaryBuffer(entry, opPos, info.inBuffers, info.outBuffers, liveTemporary,
                                                       speculativeReservations, pendingPrefetch, result)) {
                        addr = tryAllocTemporary(opBuf, persistentFloor);
                    }
                }

                if (addr == InvalidAddress) {
                    // After spilling of all buffers no new op could be scheduled — infeasible under current strategy
                    // parameters.
                    result.persistentReservedBytes = reservedSize;
                    result.peakUsedBytes =
                            reservedSize + effectiveTemporaryHighWater(committedHighWater, pendingPrefetch);
                    if (_log.isActive(LogLevel::Trace) && isDeveloperBuild()) {
                        result.peakLiveTemporaryBytes =
                                computeTruePeakConcurrentBytes(result.iterationSchedule, _classifier, _log);
                    }
                    _log.trace("Infeasible at op {0} (peakUsed={1}, limit={2})", opPos, result.peakUsedBytes,
                               _memoryLimit);
                    return result;  // feasible=false
                }
                // After buffer was allocated successfully update schedule entry allocation data
                liveTemporary.insert(opBuf);
                if (speculativeAttemptBuffers.contains(opBuf) && !producedBuffers.contains(opBuf)) {
                    speculativeReservations.insert(opBuf);
                }
                entry.allocations.push_back({opBuf, addr});
                const auto recSize = _classifier.buffers[recIt->second].size;
                committedHighWater = std::max(committedHighWater, addr + recSize);
                if (firstInputBufsSet.contains(opBuf)) {
                    firstInputExcludedRanges.push_back({addr, recSize});
                }
                _log.trace("Op {0} emit allocation value={1} addr={2} size={3} highWater={4}", opPos, opBuf, addr,
                           recSize, effectiveTemporaryHighWater(committedHighWater, pendingPrefetch));
            }

            if (!retrySchedulingThisOp) {
                break;
            }

            // Undo tentative-attempt allocations in case first attempt failed
            // so that retry is possible
            for (size_t i = allocsSize; i < entry.allocations.size(); ++i) {
                const auto val = entry.allocations[i].first;
                scanHandler.markAsDead(val);
                liveTemporary.erase(val);
                speculativeReservations.erase(val);
            }
            _linearScan.freeDeadRanges();
            entry.allocations.resize(allocsSize);
            firstInputExcludedRanges.resize(excludedRangesSize);
            committedHighWater = committedHighWaterSnapshot;
            // On next iteration try again without the overlap-aware extension. The op's own outputs will be allocated,
            // but the last-compute in/out buffers will be skipped. This is a fallback attempt.
            isSchedOpWithLastComputeOverlapEnabled = false;
        }

        // An early reservation becomes a real allocation once its producer is
        // scheduled; subsequent eviction must therefore spill its contents.
        for (auto outBuf : info.outBuffers) {
            speculativeReservations.erase(outBuf);
            producedBuffers.insert(outBuf);
        }

        // Prefetch: this op is fully scheduled and its buffers are still live, so
        // any range reserved now cannot conflict with them. Reserving ranges for
        // the next COMPUTE op's DATA_IN buffers here is what lets their DMAs
        // overlap with this op instead of reusing the ranges it releases.
        if (isPrefetchEnabled && !prefetchCandidates[opPos].empty()) {
            prefetchNextComputeInputs(opPos, prefetchCandidates[opPos], persistentFloor, overlapSensitiveBufs,
                                      firstInputExcludedRanges, liveTemporary, pendingPrefetch, committedHighWater);
        }

        _log = _log.unnest();

        _log.trace("Op {0} schedule entry ready (allocs={1}, deallocs={2}, liveTemporary={3})", opPos,
                   entry.allocations.size(), entry.deallocations.size(), liveTemporary.size());
        result.iterationSchedule.push_back(std::move(entry));
    }

    // Every prefetch targets a DATA_IN op of the same iteration, so the loop above
    // must have reached each producer and emitted (or cancelled) its allocation.
    VPUX_THROW_UNLESS(pendingPrefetch.empty(), "{0} prefetched buffer(s) left without an allocation entry",
                      pendingPrefetch.size());

    // Confirmed persistents flow into sharedExternalBuffers in deterministic
    // (admitted-order) sequence
    for (const auto& rec : admittedPersistentCand) {
        VPUX_THROW_WHEN(scanHandler.wasEverSpilled(rec.rootValue), "Admitted persistent was spilled (value={0})",
                        rec.rootValue);
        result.sharedExternalBuffers.insert(rec.rootValue);

        // If this persistent is an explicit-external template anchor, also
        // forward all iteration-equivalent SSA values. This preserves group
        // information for downstream schedulers that operate on fully
        // replicated per-iteration values.
        const auto recIt = _classifier.valueToRecord.find(rec.rootValue);
        if (recIt != _classifier.valueToRecord.end()) {
            const auto templateIdx = recIt->second;

            const auto groupIt = llvm::find_if(_classifier.explicitExternalGroups, [&](const auto& group) {
                return group.templateBufferIdx == templateIdx;
            });

            if (groupIt != _classifier.explicitExternalGroups.end()) {
                for (auto iterVal : groupIt->iterationValues) {
                    result.sharedExternalBuffers.insert(iterVal);
                }
            }
        }

        _log.trace("Final sharedExternal add persistent value={0}", rec.rootValue);
    }

    result.feasible = true;
    result.persistentReservedBytes = reservedSize;
    // No prefetch is pending at this point, so the committed mark is the final one
    result.peakUsedBytes = reservedSize + committedHighWater;
    if (_log.isActive(LogLevel::Trace) && isDeveloperBuild()) {
        result.peakLiveTemporaryBytes = computeTruePeakConcurrentBytes(result.iterationSchedule, _classifier, _log);
    }

    VPUX_THROW_WHEN(result.peakUsedBytes > _memoryLimit,
                    "VfAllocateLinear: peakUsed {0} exceeds memoryLimit {1} — invariant violation",
                    result.peakUsedBytes, _memoryLimit);

    _log.trace("Done (feasible={0}, iterationEntries={1}, reserved={2}, peakUsed={3}, sharedExt={4}, "
               "spilledBuffers={5}, temporaryHighWater={6} vs truePeakConcurrentBytes={7} (fragmentation={8}))",
               result.feasible, result.iterationSchedule.size(), result.persistentReservedBytes, result.peakUsedBytes,
               result.sharedExternalBuffers.size(), result.spilledBuffers.size(), committedHighWater,
               result.peakLiveTemporaryBytes,
               static_cast<int64_t>(committedHighWater) - static_cast<int64_t>(result.peakLiveTemporaryBytes));

    return result;
}
