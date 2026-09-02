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
using ExcludedRange = std::pair<vpux::AddressType, vpux::AddressType>;
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
// InvalidAddress on failure.
vpux::AddressType vpux::VfAllocateLinear::tryAllocTemporary(mlir::Value out, AddressType persistentFloor) {
    auto& scanHandler = _linearScan.handler();
    scanHandler.markAsAlive(out);
    // Persistent zone occupies [persistentFloor, memoryLimit). Skip the
    // excluded-region argument entirely when the persistent zone is empty
    // — Partitioner::allocFixed asserts size>0 on the excluded range.
    bool allocStatus = false;
    if (persistentFloor < _memoryLimit) {
        SmallVector<ExcludedRange> excludedRegion{{persistentFloor, _memoryLimit - persistentFloor}};
        allocStatus = _linearScan.allocWithExcludedRegion(excludedRegion, std::initializer_list<mlir::Value>{out},
                                                          /*allowSpills=*/false, DirectionT::Up);
    } else {
        allocStatus = _linearScan.alloc(std::initializer_list<mlir::Value>{out},
                                        /*allowSpills=*/false, DirectionT::Up);
    }
    if (!allocStatus) {
        scanHandler.markAsDead(out);
        _log.trace("TEMP alloc failed value={0} (persistentFloor={1})", out, persistentFloor);
        return InvalidAddress;
    }
    const auto addr = scanHandler.getAddress(out);
    _log.trace("TEMP alloc success value={0} addr={1}", out, addr);
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
//   * OutputResidency=KEEP heuristic: outBuffers of the LAST op in template iteration occupy different range than
//     iteration input buffer to allow execution overlap of with iteration
//
vpux::VfAllocateResult vpux::VfAllocateLinear::performAllocation() {
    _log.trace("Start (memoryLimit={0}, iterations={1}, opsPerIter={2}, strategy(FetchScope={3}, OutputResidency={4})",
               _memoryLimit, _classifier.numIterations, _classifier.opsPerIteration, static_cast<int>(_params.fetchScp),
               static_cast<int>(_params.outputRes));

    VfAllocateResult result;

    if (_params.fetchScp != VfSchedStrategyDescriptor::FetchScope::NONE) {
        _log.trace("Non-NONE fetch scope {0} not yet supported by linear allocator",
                   static_cast<int>(_params.fetchScp));
        return result;
    }

    if (_params.outputRes != VfSchedStrategyDescriptor::OutputResidency::DROP) {
        _log.trace("Non-DROP output residency {0} not yet supported by linear allocator",
                   static_cast<int>(_params.outputRes));
        return result;  // OutputResidency=KEEP is not yet supported by linear allocator, E#222157
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

    // If not all persistent candidates could be admitted then early return as
    // this is not yet supported until E#222690 is done
    if (admittedPersistentCand.size() < _classifier.persistentFitInitial.size()) {
        result.persistentReservedBytes = reservedSize;
        _log.trace("Not all persistent candidates admitted (admitted={0}, initial={1}), refusing",
                   admittedPersistentCand.size(), _classifier.persistentFitInitial.size());
        return result;  // feasible=false
    }

    // Downgrade all persistent buffers to TEMPORARY if they are not part
    // of admitted set. This is needed to ensure that those buffers are allocated
    // as normal temporary buffers
    {
        llvm::DenseSet<mlir::Value> admittedPersistentBufsSet;
        for (const auto& rec : admittedPersistentCand) {
            admittedPersistentBufsSet.insert(rec.rootValue);
        }
        size_t persistentChangedToTemporaryCount = 0;
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
            ++persistentChangedToTemporaryCount;
            _log.trace("Downgrade loop-invariant PC to TEMPORARY (size={0})", rec.size);
        }
        _log.trace("Unadmitted PC processing done (downgraded={0})", persistentChangedToTemporaryCount);
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

        SmallVector<mlir::Value> opBuffers = info.outBuffers;
        // Add input buffer only if it was previously spilled. Input buffers which are
        // not produced as part of a loop body are allocated outside of this allocator
        for (auto inBuf : info.inBuffers) {
            if (scanHandler.wasEverSpilled(inBuf)) {
                opBuffers.push_back(inBuf);
            }
        }

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

            AddressType addr = tryAllocTemporary(opBuf, persistentFloor);

            // If allocation failed try to spill to get more available memory and retry
            // TODO: E#222690 Persistent buffer spilling is not fully supported yet. For now
            // handle only spill of temporary buffers. Need to update the allocations/deallocations for persistent
            // buffers in compute schedule, decrement reservedSize range and increase persistentFloor to reflect the
            // eviction of the persistent buffer. Remove this buffer from admittedPersistentCand
            // TODO: Unify logic for spilling temporary and persistent buffers. There are possible
            // improvements here as it is not guaranteed that spilling persistend buffers will be better than
            // spilling temporary ones, especially if its cost is small or can be parallelized with compute
            // TODO: Analyze if persistent buffer spill is ever needed as persistent buffer candidates will be rejected
            // already if their persistent allocation together with classifier temporary peak bytes calculation exceed
            // memory limit
            while (addr == InvalidAddress &&
                   findAndSpillTemporaryBuffer(entry, opPos, info.inBuffers, info.outBuffers, liveTemporary, result)) {
                addr = tryAllocTemporary(opBuf, persistentFloor);
            }

            if (addr == InvalidAddress) {
                // After spilling of all buffers no new op could be scheduled — infeasible under current strategy
                // parameters.
                result.persistentReservedBytes = reservedSize;
                result.peakUsedBytes = reservedSize + temporaryHighWater;
                _log.trace("Infeasible at op {0} (peakUsed={1}, limit={2})", opPos, result.peakUsedBytes, _memoryLimit);
                return result;  // feasible=false
            }
            // After buffer was allocated successfully update schedule entry allocation data
            liveTemporary.insert(opBuf);
            entry.allocations.push_back({opBuf, addr});
            const auto recSize = _classifier.buffers[recIt->second].size;
            temporaryHighWater = std::max(temporaryHighWater, addr + recSize);
            _log.trace("Op {0} emit allocation value={1} addr={2} size={3} highWater={4}", opPos, opBuf, addr, recSize,
                       temporaryHighWater);
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
