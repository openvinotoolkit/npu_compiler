//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/scheduling/vf_allocate_linear.hpp"

#include "vpux/compiler/utils/linear_scan.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>

#include <algorithm>
#include <utility>

using namespace vpux;

namespace {
using ExcludedRangeOffsetAndSize = std::pair<vpux::AddressType, vpux::AddressType>;

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
                                                          AddressType& temporaryHighWater) {
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
    // `persistedBuffer.alignment`-aligned address; the gap between the old
    // floor and the aligned offset is unusable and counts as freed persistent
    // space alongside the buffer size itself.
    const auto alignedOffset = vpux::alignValUp(persistentFloor, persistedBuffer.alignment);

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
        // Since non longer persistent buffer is now allocated as temporary update temporaryHighWater
        temporaryHighWater = std::max(temporaryHighWater, alignedOffset + persistedBuffer.size);
        _log.trace("Downgraded persistent, insert alloc at {0}, value={1} addr={2}, highWater={3}", firstUseOp,
                   persistedBuffer.rootValue, alignedOffset, temporaryHighWater);
    }

    _linearScan.handler().spilled(persistedBuffer.rootValue);
    entry.deallocations.push_back(persistedBuffer.rootValue);
    result.spilledBuffers.push_back(persistedBuffer.rootValue);

    // Persistent zone shrinks by the alignment gap plus the buffer size.
    const auto freed = (alignedOffset - persistentFloor) + persistedBuffer.size;
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
bool vpux::VfAllocateLinear::findAndSpillTemporaryBuffer(ComputeExplicitSchedule& entry, size_t currentOpPos,
                                                         ArrayRef<mlir::Value> inOpBuffers,
                                                         ArrayRef<mlir::Value> outOpBuffers,
                                                         ValueOrderedSet& liveTemporary, VfAllocateResult& result) {
    // Pick a TEMPORARY buffer for DDR spill. Choice rule: largest
    // size first, then longest remaining use window (lastUseOp - currentOpPos).
    // Buffers that are inputs or outputs of the operation being scheduled are
    // excluded — spilling them would defeat the purpose of the allocation
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
    _linearScan.handler().spilled(bufferToSpill);
    _linearScan.freeDeadRanges();
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
vpux::AddressType vpux::VfAllocateLinear::tryAllocTemporary(
        mlir::Value out, AddressType persistentFloor,
        ArrayRef<std::pair<AddressType, AddressType>> extraExcludedCandidates) {
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
//     - Step 1: Evict persistent candidates until enough space to allocated
//               new temporary buffer. Spilled persistent buffer is no longer treated as persistent.
//               Shrink persistent reserved range
//     - Step 2: If eviction of all persistent candidates does not free enough space, spill TEMPORARY buffers
//   * LastComputeOverlapWithNextIterInputs=ENABLED heuristic: outBuffers of the LAST op in template iteration occupy
//     different range than iteration input buffer to allow execution overlap of with iteration
//
vpux::VfAllocateResult vpux::VfAllocateLinear::performAllocation() {
    _log.trace("Start (memoryLimit={0}, iterations={1}, opsPerIter={2}, strategy(FetchScope={3}, "
               "LastComputeOverlapWithNextIterInputs={4})",
               _memoryLimit, _classifier.numIterations, _classifier.opsPerIteration, static_cast<int>(_params.fetchScp),
               static_cast<int>(_params.lastCompOverlapNextIterIn));

    VfAllocateResult result;

    if (_params.fetchScp != VfSchedStrategyDescriptor::FetchScope::NONE) {
        _log.trace("Non-NONE fetch scope {0} not yet supported by linear allocator",
                   static_cast<int>(_params.fetchScp));
        return result;
    }

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

    // Register every classifier buffer with the handler so that
    // allocation has knowledge of buffer details like size of VF specific attributes
    for (const auto& bufferRec : _classifier.buffers) {
        if (!bufferRec.rootValue) {
            // Ignore non root SSA values as those are not what allocator tracks and represent only aliases
            continue;
        }

        const size_t spillCost = getSpillCost(bufferRec.size);
        scanHandler.addBufferData(bufferRec.rootValue, bufferRec.category, bufferRec.size, bufferRec.alignment,
                                  spillCost);
    }

    // Admit candidates greedily (by descending cost) until the next one would
    // not fit alongside the volatile working set's optimistic estimate.
    SmallVector<BufferRecord> admittedPersistentCand;  // sorted by cost DESC (admission order).
    AddressType reservedSize = 0;
    for (size_t i = 0; i < _classifier.persistentFitOrder.size(); ++i) {
        const size_t idx = _classifier.persistentFitOrder[i];
        const auto& rec = _classifier.buffers[idx];
        const auto pad = vpux::alignValUp(rec.size, rec.alignment);
        if (reservedSize + pad > _memoryLimit) {
            _log.trace("Persistent candidate (size={0}) exceeds memoryLimit, dropped", rec.size);
            continue;
        }
        // Reject candidates that would crowd out the volatile working set.
        if (reservedSize + pad + _classifier.temporaryPeakBytes > _memoryLimit) {
            _log.trace("Persistent candidate (size={0}) would crowd out volatile peak {1}", rec.size,
                       _classifier.temporaryPeakBytes);
            continue;
        }
        reservedSize += pad;
        admittedPersistentCand.push_back(rec);
        _log.trace("AdmittedPersistentBuf={0} category={1} (size={2}, align={3}, spillCost={4}, floorAfter={5})",
                   rec.rootValue, stringifyBufferCategory(rec.category), rec.size, rec.alignment,
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
                                      rec.alignment, /*spillCost=*/0);
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

    AddressType temporaryHighWater = 0;

    // Iterate over loops template body and schedule operations one by one
    for (size_t opPos = 0; opPos < templateBody.size(); ++opPos) {
        const auto& info = templateBody[opPos];
        ComputeExplicitSchedule entry;
        entry.allocInfo = info;
        entry.templatePos = opPos;
        _log.trace("Op {0} begin (opIdx={1}, inBuffers={2}, outBuffers={3}, liveTemporary={4})", opPos, info.opIdx,
                   info.inBuffers.size(), info.outBuffers.size(), liveTemporary.size());
        _log = _log.nest();

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
            const AddressType highWaterSnapshot = temporaryHighWater;

            SmallVector<mlir::Value> opBuffers = info.outBuffers;

            // When last-compute overlap is enabled and scheduling operation which produces inputs for last compute
            // operation try to co-allocate all last-compute input and output buffers alongside this op's own outputs so
            // that in the end next iteration DATA_IN DMAs can overlap with this iteration last compute op and its
            // output DATA_OUT DMAs
            if (isSchedOpWithLastComputeOverlapEnabled && opProducesLastComputeInput) {
                llvm::DenseSet<mlir::Value> alreadyPresent(opBuffers.begin(), opBuffers.end());
                for (auto v : lastComputeOpOutBufsRevOrder) {
                    if (alreadyPresent.insert(v).second) {
                        opBuffers.push_back(v);
                    }
                }
                for (auto v : lastComputeOpInBufsOrder) {
                    if (alreadyPresent.insert(v).second) {
                        opBuffers.push_back(v);
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

                // If allocation failed try to spill to get more available memory and retry
                // TODO: Unify logic for spilling temporary and persistent buffers. There are possible
                // improvements here as it is not guaranteed that spilling persistend buffers will be better than
                // spilling temporary ones, especially if its cost is small or can be parallelized with compute
                while (addr == InvalidAddress &&
                       findAndSpillPersistentBuffer(result, entry, opPos, info.inBuffers, info.outBuffers,
                                                    admittedPersistentCand, reservedSize, persistentFloor,
                                                    temporaryHighWater)) {
                    addr = tryAllocTemporary(opBuf, persistentFloor);
                }

                if (addr == InvalidAddress) {
                    while (addr == InvalidAddress &&
                           findAndSpillTemporaryBuffer(entry, opPos, info.inBuffers, info.outBuffers, liveTemporary,
                                                       result)) {
                        addr = tryAllocTemporary(opBuf, persistentFloor);
                    }
                }

                if (addr == InvalidAddress) {
                    // After spilling of all buffers no new op could be scheduled — infeasible under current strategy
                    // parameters.
                    result.persistentReservedBytes = reservedSize;
                    result.peakUsedBytes = reservedSize + temporaryHighWater;
                    _log.trace("Infeasible at op {0} (peakUsed={1}, limit={2})", opPos, result.peakUsedBytes,
                               _memoryLimit);
                    return result;  // feasible=false
                }
                // After buffer was allocated successfully update schedule entry allocation data
                liveTemporary.insert(opBuf);
                entry.allocations.push_back({opBuf, addr});
                const auto recSize = _classifier.buffers[recIt->second].size;
                temporaryHighWater = std::max(temporaryHighWater, addr + recSize);
                if (firstInputBufsSet.contains(opBuf)) {
                    firstInputExcludedRanges.push_back({addr, recSize});
                }
                _log.trace("Op {0} emit allocation value={1} addr={2} size={3} highWater={4}", opPos, opBuf, addr,
                           recSize, temporaryHighWater);
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
            }
            _linearScan.freeDeadRanges();
            entry.allocations.resize(allocsSize);
            firstInputExcludedRanges.resize(excludedRangesSize);
            temporaryHighWater = highWaterSnapshot;
            // On next iteration try again without the overlap-aware extension. The op's own outputs will be allocated,
            // but the last-compute in/out buffers will be skipped. This is a fallback attempt.
            isSchedOpWithLastComputeOverlapEnabled = false;
        }

        _log = _log.unnest();

        _log.trace("Op {0} schedule entry ready (allocs={1}, deallocs={2}, liveTemporary={3})", opPos,
                   entry.allocations.size(), entry.deallocations.size(), liveTemporary.size());
        result.iterationSchedule.push_back(std::move(entry));
    }

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
    result.peakUsedBytes = reservedSize + temporaryHighWater;

    VPUX_THROW_WHEN(result.peakUsedBytes > _memoryLimit,
                    "VfAllocateLinear: peakUsed {0} exceeds memoryLimit {1} — invariant violation",
                    result.peakUsedBytes, _memoryLimit);

    _log.trace("Done (feasible={0}, iterationEntries={1}, reserved={2}, peakUsed={3}, sharedExt={4}, "
               "spilledBuffers={5})",
               result.feasible, result.iterationSchedule.size(), result.persistentReservedBytes, result.peakUsedBytes,
               result.sharedExternalBuffers.size(), result.spilledBuffers.size());

    return result;
}
