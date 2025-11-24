//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/core/format.hpp"

using namespace vpux;

mlir::Operation* getExecInnerOp(mlir::async::ExecuteOp execOp) {
    mlir::Operation* innerOp = nullptr;
    auto* bodyBlock = execOp.getBody();
    for (auto& op : bodyBlock->getOperations()) {
        if (VPUIP::isPureViewOp(&op)) {
            continue;
        }
        innerOp = &op;
        break;
    }
    VPUX_THROW_UNLESS(innerOp, "No inner op located for '{0}'", execOp.getLoc());
    return innerOp;
}

//
// Feasible Memory Scheduler Spilling support
//

FeasibleMemorySchedulerSpilling::FeasibleMemorySchedulerSpilling(VPU::MemoryKind memKind,
                                                                 VPU::MemoryKind secondLvlMemKind,
                                                                 AsyncDepsInfo& depsInfo, AliasesInfo& aliasInfo,
                                                                 Logger log,
                                                                 LinearScan<mlir::Value, LinearScanHandler>& scan)
        : _log(log),
          _allocOpInsertionPoint(nullptr),
          _memKind(memKind),
          _secondLvlMemKind(secondLvlMemKind),
          _depsInfo(depsInfo),
          _aliasInfo(aliasInfo),
          _scan(scan) {
    _log.setName("feasible-memory-scheduler-spilling");
}

// Optimize immediate spilling of computeOps output that happens as a buffer relocation
// Example sequence:
//  1. t = 0: ORIGINAL (computeOp)
//  .. t = 0: .....
//  .. t = 1:
//  2. t = 1: SPILL_WRITE
//  .. t = 1:
//  3. t = 2: SPILL_READ
//
// COMPUTEOP -> SPILL_WRITE -> SPILL_READ sequence must happen immediately one after the other with respect
// to timeline as only then removal of SPILL_WRITE/READ sequence is possible. In other cases there might be
// a clash in terms of buffer ranges for operation happeninig in between.
// During optimization SPILL_WRITE -> SPILL_READ pair is removed from schedule and ORIGINAL (computeOp) is
// assigned final memory range - result of SPILL_READ
void FeasibleMemorySchedulerSpilling::removeComputeOpRelocationSpills(
        FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps) {
    _log.trace("Optimize spills resulting from compute op immediate relocation");

    std::unordered_map<size_t, size_t> opIndexToSchedId;

    const auto findScheduledOpsVecIndexByOpIndex = [&](size_t opIndex) {
        if (opIndexToSchedId.empty()) {
            opIndexToSchedId.reserve(scheduledOps.size());
            for (size_t schedIndex = 0; schedIndex < scheduledOps.size(); schedIndex++) {
                if (scheduledOps[schedIndex].isOriginalOp()) {
                    opIndexToSchedId[scheduledOps[schedIndex].op_] = schedIndex;
                }
            }
        }

        auto it = opIndexToSchedId.find(opIndex);
        if (it != opIndexToSchedId.end()) {
            return it->second;
        }
        VPUX_THROW("No schedule index for op index '{0}'", opIndex);
    };

    SmallVector<size_t> operationIndexesToRemove;
    for (size_t opIndex = 0; opIndex < scheduledOps.size(); opIndex++) {
        if (scheduledOps[opIndex].isSpillWrite() && !scheduledOps[opIndex].isDataOp()) {
            // Check if related computeOp has single output. If not
            // then skip this optimization
            if (_depsInfo.getExecuteOpAtIndex(scheduledOps[opIndex].op_).getBodyResults().size() > 1) {
                continue;
            }

            // Located SPILL_WRITE for compute op
            size_t spillWriteIndex = opIndex;
            std::optional<size_t> origOpIndex;
            std::optional<int64_t> prevTime;
            size_t prevTimeFirstOpIndex = 0;
            std::optional<size_t> spillReadIndex;
            std::optional<int64_t> nextTime;

            // Search for corresponding ORIGINAL op and check
            // if it is at closest previous time
            auto prevOpIdx = static_cast<int64_t>(opIndex) - 1;
            for (; prevOpIdx >= 0 && prevOpIdx < static_cast<int64_t>(scheduledOps.size()); --prevOpIdx) {
                if (!prevTime.has_value() &&
                    scheduledOps[prevOpIdx].cycleBegin_ < scheduledOps[spillWriteIndex].cycleBegin_) {
                    // Identified previous time value
                    prevTime = scheduledOps[prevOpIdx].cycleBegin_;
                }

                if (prevTime.has_value()) {
                    if (scheduledOps[prevOpIdx].cycleBegin_ < prevTime) {
                        // Time of op in given iteration is smaller than closest previous time
                        // In such case break early as that means that COMPUTEOP -> SPILL_WRITE
                        // sequence does not appear immediately one after the other
                        prevTimeFirstOpIndex = static_cast<size_t>(prevOpIdx) + 1;
                        break;
                    }

                    if (scheduledOps[prevOpIdx].op_ == scheduledOps[spillWriteIndex].op_ &&
                        scheduledOps[prevOpIdx].isOriginalOp()) {
                        // Identified original COMPUTEOP that appears just before SPILL_WRITE
                        origOpIndex = static_cast<size_t>(prevOpIdx);
                    }
                }
            }

            // Original operation just before SPILL_WRITE was no located. Stop analysis for given operation index
            if (!origOpIndex.has_value()) {
                continue;
            }

            // Search for matching SPILL_READ op and check
            // if it is at closest next time
            for (size_t i = opIndex + 1; i < scheduledOps.size(); i++) {
                if (!nextTime.has_value() && scheduledOps[i].cycleBegin_ > scheduledOps[spillWriteIndex].cycleBegin_) {
                    nextTime = scheduledOps[i].cycleBegin_;
                }

                if (nextTime.has_value()) {
                    if (scheduledOps[i].cycleBegin_ > nextTime) {
                        // Time of op in given iteration is larger than closest next time
                        // In such case break early as that means that SPILL_WRITE -> SPILL_READ
                        // sequence does not appear immediately one after the other
                        break;
                    }

                    if (scheduledOps[i].op_ == scheduledOps[spillWriteIndex].op_) {
                        // Identified SPILL_READ that appears just after SPILL_WRITE
                        spillReadIndex = i;
                        break;
                    }
                }
            }

            // SPILL_READ does not appear right after SPILL_WRITE. Stop analysis for given op
            if (!spillReadIndex.has_value()) {
                continue;
            }

            // Check if no operation depends on range assigned later to SPILL_READ output between
            // scheduled ops at index starting from prevTimeFirstOpIndex to spillReadIndex
            auto resBegin = scheduledOps[spillReadIndex.value()].beginOutputResource(0);
            auto resEnd = scheduledOps[spillReadIndex.value()].endOutputResource(0) - 1;
            bool rangeUsed = false;
            for (size_t i = prevTimeFirstOpIndex; i < spillReadIndex.value(); i++) {
                // Skip check for SPILL_WRITE
                if (i == spillWriteIndex) {
                    continue;
                }

                if (scheduledOps[i].hasActiveInputResource()) {
                    for (size_t r = 0; r < scheduledOps[i].numOfInputResources(); r++) {
                        auto beg = scheduledOps[i].beginInputResource(r);
                        auto end = scheduledOps[i].endInputResource(r) - 1;

                        if ((beg <= resBegin && resEnd <= end) || (resBegin <= beg && beg <= resEnd) ||
                            (resBegin <= end && end <= resEnd)) {
                            rangeUsed = true;
                            break;
                        }
                    }
                }

                if (rangeUsed) {
                    break;
                }

                if (scheduledOps[i].hasActiveOutputResource()) {
                    for (size_t r = 0; r < scheduledOps[i].numOfOutputResources(); r++) {
                        if (i == origOpIndex.value() &&
                            scheduledOps[origOpIndex.value()].getOutputBuffer(r) ==
                                    scheduledOps[spillReadIndex.value()].getOutputBuffer(0)) {
                            // Skip check for buffer that is planned for relocation
                            continue;
                        }

                        auto beg = scheduledOps[i].beginOutputResource(r);
                        auto end = scheduledOps[i].endOutputResource(r) - 1;

                        if ((beg <= resBegin && resEnd <= end) || (resBegin <= beg && beg <= resEnd) ||
                            (resBegin <= end && end <= resEnd)) {
                            rangeUsed = true;
                            break;
                        }
                    }
                }
                if (rangeUsed) {
                    break;
                }
            }

            // Range is used by other operation. Optimization cannot be performed
            if (rangeUsed) {
                _log.trace("Range is used, cannot relocate output of op - '{0}'",
                           scheduledOps[origOpIndex.value()].op_);
                continue;
            }

            auto& origOp = scheduledOps[origOpIndex.value()];
            auto& spillWriteOp = scheduledOps[spillWriteIndex];
            auto& spillReadOp = scheduledOps[spillReadIndex.value()];
            auto spillBuf = spillReadOp.getOutputBuffer(0);

            // Check if operation writes just to part of buffer. In that case optimization cannot
            // be performed as operation is not a sole owner of full buffer
            bool operationOwnsBuffer = true;
            auto* bodyBlock = _depsInfo.getExecuteOpAtIndex(origOp.op_).getBody();
            for (auto& op : bodyBlock->getOperations()) {
                if (mlir::isa<VPUIP::LayerOpInterface>(op)) {
                    auto layerOp = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);

                    for (auto output : layerOp.getOutputs()) {
                        const auto type = mlir::dyn_cast<vpux::NDTypeInterface>(output.getType());
                        if (type == nullptr || type.getMemoryKind() != _memKind) {
                            continue;
                        }

                        const auto rootBuffer = _aliasInfo.getRoot(output);

                        if (rootBuffer != spillBuf) {
                            // This is not an output that is going to be spilled. Move
                            // to next one
                            continue;
                        }
                        // Found an output corresponding to a root buffer that is marked for spilling
                        // Check if size is the same. If not then optimization cannot be performed
                        // because current operation only produces part of buffer and relocation
                        // will break data for whole buffer
                        if (getCompactSize(rootBuffer) != getCompactSize(output)) {
                            operationOwnsBuffer = false;
                        }
                        break;
                    }
                }
            }

            // Operation is not an only producer of this buffer. Optimization cannot be performed
            if (!operationOwnsBuffer) {
                _log.trace("Operation is not an only producer of this buffer, cannot relocate output of op - '{0}'",
                           origOp.op_);
                continue;
            }

            // Identified COMPUTEOP -> SPILL_WRITE -> SPILL_READ sequence that can be optimized
            _log.trace("Identified COMPUTEOP -> SPILL_WRITE -> SPILL_READ sequence that can be optimized");

            _log.nest().trace("op = '{0}'\t type = '{1}'\t time = '{2}'", origOp.op_, origOp.opTypeName(),
                              origOp.cycleBegin_);
            _log.nest().trace("op = '{0}'\t type = '{1}'\t time = '{2}'", spillWriteOp.op_, spillWriteOp.opTypeName(),
                              spillWriteOp.cycleBegin_);
            _log.nest().trace("op = '{0}'\t type = '{1}'\t time = '{2}'", spillReadOp.op_, spillReadOp.opTypeName(),
                              spillReadOp.cycleBegin_);

            // The spill buffer may have multiple write users. If yes, skip the optimization.
            bool hasSharedOutputBuffer = false;

            const auto spillRootBuffer = _aliasInfo.getRoot(spillBuf);
            auto origExecOp = _depsInfo.getExecuteOpAtIndex(origOp.op_);
            const auto allAlias = _aliasInfo.getAllAliases(spillRootBuffer);

            for (const auto alias : allAlias) {
                auto inputExecOp = alias.getDefiningOp<mlir::async::ExecuteOp>();
                if (inputExecOp == nullptr || origExecOp == inputExecOp) {
                    continue;
                }

                auto inputInnerOp = getExecInnerOp(inputExecOp);
                for (const auto output : inputInnerOp->getResults()) {
                    if (spillRootBuffer == _aliasInfo.getRoot(output)) {
                        // Get the dependency index and convert to scheduled operation index
                        const auto inputOpId = _depsInfo.getIndex(inputExecOp);
                        const auto schedOpId = findScheduledOpsVecIndexByOpIndex(inputOpId);
                        const auto inputEndCycle = scheduledOps[schedOpId].cycleEnd_;
                        if (origOp.cycleBegin_ >= inputEndCycle) {
                            _log.nest().trace("Spill buffer is shared with op index {0} at {1}", inputOpId,
                                              inputInnerOp->getLoc());
                            hasSharedOutputBuffer = true;
                            break;
                        }
                    }
                }

                if (hasSharedOutputBuffer) {
                    break;
                }
            }

            // ComputeOp output buffer range can be assigned resulting range of SPILL_READ
            // First find matching buffer that was spilled
            bool foundMatchingBuffer = false;
            bool removeSpillOps = false;
            for (size_t i = 0; i < origOp.numOfOutputResources(); i++) {
                if (origOp.getOutputBuffer(i) == spillBuf) {
                    // Check if same buffer is also used as operation input. This is the case
                    // for eltwise inplace op
                    bool isInplaceOp = false;
                    for (size_t j = 0; j < origOp.numOfInputResources(); j++) {
                        if (origOp.getInputBuffer(j) == spillBuf) {
                            isInplaceOp = true;
                            break;
                        }
                    }

                    if (isInplaceOp || !hasSharedOutputBuffer) {
                        removeSpillOps = true;
                        // Found matching resource index. Update assigned range
                        origOp.outputResourceInfo_[i].begin_ = spillReadOp.outputResourceInfo_[0].begin_;
                        origOp.outputResourceInfo_[i].end_ = spillReadOp.outputResourceInfo_[0].end_;
                    }

                    // In case of spilling inplace operation need to create new buffer for output
                    // and remove inplace attribute
                    if (isInplaceOp) {
                        auto allocOpInsertionPoint =
                                _depsInfo.getExecuteOpAtIndex(scheduledOps.begin()->op_).getOperation();

                        auto origExecInnerOp = getExecInnerOp(origExecOp);
                        if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(origExecInnerOp)) {
                            nceOp.removeIsInplaceAttr();
                        }

                        // Create allocation operation for new output buffer as the old one is also used
                        // as operation input and this code should not modify this allocation
                        mlir::OpBuilder builder(allocOpInsertionPoint);
                        auto newBufferOp = builder.clone(*spillBuf.getDefiningOp());
                        // Add location suffix to identify this as a spill replacement buffer
                        newBufferOp->setLoc(appendLoc(newBufferOp->getLoc(), "replace_spill"));
                        auto newBufferResult = newBufferOp->getResult(0);

                        // Update buffer data in scheduledOps
                        origOp.outputResourceInfo_[i].buffer_ = newBufferResult;

                        // Assign new buffer to operand of operation
                        for (auto operand : origExecInnerOp->getOperands() | indexed) {
                            if (operand.value() == spillBuf) {
                                origExecInnerOp->setOperand(operand.index(), newBufferResult);
                            }
                        }

                        // Update aliases info for newly created root buffer
                        _aliasInfo.removeAlias(origExecInnerOp->getResult(0));
                        _aliasInfo.removeAlias(origExecOp.getBodyResults()[0]);

                        _aliasInfo.addAlias(newBufferResult, newBufferResult);
                        _aliasInfo.addAlias(newBufferResult, origExecOp.getBodyResults()[0]);
                        _aliasInfo.addAlias(newBufferResult, origExecInnerOp->getResult(0));

                        // Configure address as prepared by scheduler.
                        // Since it is a new buffer it was not assigned before
                        _scan.handler().setAddress(newBufferResult, origOp.outputResourceInfo_[i].begin_);

                        // Update addresses for all other operations that share the same spill buffer
                        // Context: The spill buffer has multiple users and may be allocated-deallocated-allocated due
                        // to spilling. The final memory offset is determined by the last allocation operation. Since
                        // this spill buffer is being replaced by a new buffer, we need to restore the original
                        // allocation offset for other users to maintain memory consistency
                        for (const auto alias : _aliasInfo.getAllAliases(spillBuf)) {
                            auto otherUserExecOp = alias.getDefiningOp<mlir::async::ExecuteOp>();
                            if (otherUserExecOp == nullptr || origExecOp == otherUserExecOp) {
                                continue;
                            }

                            // Find the scheduled operation corresponding to this input operation
                            const auto schedOpId =
                                    findScheduledOpsVecIndexByOpIndex(_depsInfo.getIndex(otherUserExecOp));
                            const auto& schedOp = scheduledOps[schedOpId];

                            // Update address for each output resource that uses the spill buffer
                            for (size_t resourceIdx = 0; resourceIdx < schedOp.numOfOutputResources(); resourceIdx++) {
                                if (schedOp.isActiveOutputResource(resourceIdx) &&
                                    schedOp.getOutputBuffer(resourceIdx) == spillBuf) {
                                    // Restore the original allocation offset for this shared buffer user
                                    _scan.handler().setAddress(spillBuf, schedOp.beginOutputResource(resourceIdx));
                                }
                            }
                        }
                    }

                    foundMatchingBuffer = true;
                    break;
                }
            }
            VPUX_THROW_UNLESS(foundMatchingBuffer, "Matching buffer not found for relocation spilling optimization");

            if (removeSpillOps) {
                // SPILL_WRITE and SPILL_READ operations can be removed
                operationIndexesToRemove.push_back(spillWriteIndex);
                operationIndexesToRemove.push_back(spillReadIndex.value());
            }
        }
    }

    if (operationIndexesToRemove.empty()) {
        _log.trace("No compute ops immediate relocation spilling identified");
        return;
    }

    // Remove in reverse order to have indexes valid after erasing entries in scheduledOp
    for (auto opIt = operationIndexesToRemove.rbegin(); opIt != operationIndexesToRemove.rend(); opIt++) {
        scheduledOps.erase(scheduledOps.begin() + *opIt);
    }

    _log.trace("Operations that have been removed - '{0}'", operationIndexesToRemove.size());
}

/*
    Verify the below optimization since <CMX> resource is not guaranteed
    to be reserved through {other ops} and may be overridden.

          <DDR>                                <DDR>
            |     <CMX>                        /          <CMX>
            |     /                           /             .
        GatherDMA (isDataOp = true)          /              .
            |                               /               .
          <CMX>                            /                .
            |                             /                 .
        SpillWriteDMA                    /                  .
            |                       =>  |                   .
          <DDR>                          \                  .
           ...                            \     ...        .
        {other ops}                        \  {other ops} .
           ...                              \    ...     .
          <DDR>                              \          .
            |                                 \        .
        SpillReadDMA                          GatherDMA
            |                                    |
            <CMX>                                 <CMX>
*/
llvm::DenseSet<size_t> FeasibleMemorySchedulerSpilling::identifyDataOpsWithInputOverwrite(
        FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps) {
    llvm::DenseMap<size_t, FeasibleMemoryScheduler::ScheduledOpInfo> dataOpsWithInputResources;
    for (unsigned opIndex = 0; opIndex < scheduledOps.size(); opIndex++) {
        auto& op = scheduledOps[opIndex];
        if (op.isOriginalOp() && op.isDataOp() && op.hasActiveInputResource()) {
            dataOpsWithInputResources[op.op_] = op;
        }
    }

    if (dataOpsWithInputResources.empty()) {
        return {};
    }

    // store legal and illegal data op indices to avoid re-checking for
    // a repeating data op spill
    llvm::DenseSet<size_t> legalDataOpsWithInputResources;
    llvm::DenseSet<size_t> illegalDataOpsWithInputResources;
    for (unsigned opIndex = scheduledOps.size(); opIndex-- > 0;) {
        auto& op = scheduledOps[opIndex];
        if ((op.isSpillWrite() || op.isSpillRead()) && op.isDataOp()) {
            if (legalDataOpsWithInputResources.contains(op.op_) || illegalDataOpsWithInputResources.contains(op.op_)) {
                continue;
            }
            if (!dataOpsWithInputResources.contains(op.op_)) {
                continue;
            }

            auto parentOp = dataOpsWithInputResources[op.op_];

            // check if legal or illegal
            auto legalOptimization = true;
            // Note: spill read can also overwrite the input
            for (unsigned nestedOpIndex = opIndex + 1; nestedOpIndex-- > 0;) {
                auto nestedOp = scheduledOps[nestedOpIndex];
                if (nestedOp.isOriginalOp() && nestedOp.op_ == op.op_) {
                    break;
                }

                for (size_t inResource = 0; inResource < parentOp.numOfInputResources(); inResource++) {
                    auto resBegin = parentOp.beginInputResource(inResource);
                    auto resEnd = parentOp.endInputResource(inResource);
                    auto rangeUsed = false;

                    if (nestedOp.hasActiveInputResource()) {
                        for (size_t r = 0; r < nestedOp.numOfInputResources(); r++) {
                            auto beg = nestedOp.beginInputResource(r);
                            auto end = nestedOp.endInputResource(r) - 1;

                            if ((beg <= resBegin && resEnd <= end) || (resBegin <= beg && beg <= resEnd) ||
                                (resBegin <= end && end <= resEnd)) {
                                rangeUsed = true;
                                break;
                            }
                        }
                    }

                    if (!rangeUsed && nestedOp.hasActiveOutputResource()) {
                        for (size_t r = 0; r < nestedOp.numOfOutputResources(); r++) {
                            auto beg = nestedOp.beginOutputResource(r);
                            auto end = nestedOp.endOutputResource(r) - 1;

                            if ((beg <= resBegin && resEnd <= end) || (resBegin <= beg && beg <= resEnd) ||
                                (resBegin <= end && end <= resEnd)) {
                                rangeUsed = true;
                                break;
                            }
                        }
                    }

                    if (rangeUsed) {
                        legalOptimization = false;
                        break;
                    }
                }

                if (!legalOptimization) {
                    break;
                }
            }

            if (legalOptimization) {
                legalDataOpsWithInputResources.insert(op.op_);
            } else {
                illegalDataOpsWithInputResources.insert(op.op_);
            }
        }
    }

    return illegalDataOpsWithInputResources;
}

// Optimize spilling of dataOps. This function will check scheduledOps list and analyze spilling sequence of dataOps.
// Example sequence:
//  1. ORIGINAL (dataOp)
//  ...
//  2. SPILL_WRITE
//  ...
//  3. SPILL_READ
//
// If between ORIGINAL and SPILL_WRITE buffer is not used (e.g. such scenario can happen during prefetching) then
// both 1. and 2. can be removed and SPILL_READ (3.) will be changed to ORIGINAL type.
// If ORIGINAL has active input resources skip optimization verify no memory overlap.
// TODO:
// If between ORIGINAL and SPILL_WRITE buffer is used only as an input then SPILL_WRITE (2.) can be removed
// and SPILL_READ (3.) turned into equivalent ORIGINAL dataOp
void FeasibleMemorySchedulerSpilling::optimizeDataOpsSpills(FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps) {
    _log.trace("Optimize data ops spills");
    // Find data ops which can not be optimized due to possible memory override:
    // - data ops with input resources which may not be reserved throughout the spill
    auto illegalDataOpsWithInputResources = identifyDataOpsWithInputOverwrite(scheduledOps);
    // Collect information about all data ops that have been spilled
    // For each such dataOp store a sequence of indexes for scheduleOps array
    // where each entry corresponds to related spillWrite/Read operation. First entry
    // (index 0) is the index for original dataOp
    std::unordered_map<FeasibleMemoryScheduler::operationIdxType, SmallVector<size_t>> dataOpSpillTree;
    for (unsigned opIndex = 0; opIndex < scheduledOps.size(); opIndex++) {
        auto& op = scheduledOps[opIndex];
        // Check if this is spillRead/Write of data op
        if ((op.isSpillWrite() || op.isSpillRead()) && op.isDataOp()) {
            // Check if related dataOp has single output. If not
            // then skip this optimization
            if (_depsInfo.getExecuteOpAtIndex(op.op_).getBodyResults().size() > 1) {
                continue;
            }
            // Skip ops with input resources overlap
            if (illegalDataOpsWithInputResources.contains(op.op_)) {
                continue;
            }
            // Find if this is spilling of already encountered dataOp
            auto dataOpIt = dataOpSpillTree.find(op.op_);
            if (dataOpIt != dataOpSpillTree.end()) {
                // If dataOp was already identified store index of related spill operation
                dataOpIt->second.push_back(opIndex);
            } else {
                // If this is spilling of new op, find source op and check if this is dataOp
                auto origOpIndex = static_cast<int>(opIndex) - 1;
                for (; origOpIndex >= 0; origOpIndex--) {
                    auto schedOrigOp = scheduledOps[origOpIndex];
                    if (schedOrigOp.isOriginalOp() && schedOrigOp.op_ == op.op_) {
                        // As a first element store index to original operation
                        dataOpSpillTree[schedOrigOp.op_].push_back(origOpIndex);
                        // Store index to identified spillWrite/Read operation
                        dataOpSpillTree[schedOrigOp.op_].push_back(opIndex);
                        break;
                    }
                }
                VPUX_THROW_UNLESS(origOpIndex >= 0,
                                  "Unable to find in scheduled ops original operation for a given spill op '{0}'",
                                  op.op_);
            }
        }
    }

    if (dataOpSpillTree.empty()) {
        _log.trace("No data ops spilling identified");
        return;
    }

    // Dump data ops spilling information
    _log.trace("Data operation spilling sequence:");
    for (auto& dataOp : dataOpSpillTree) {
        _log.nest(1).trace("Operation - '{0}'", dataOp.first);
        for (auto& i : dataOp.second) {
            auto& op = scheduledOps[i];
            _log.nest(2).trace("['{0}']: op = '{1}'\t type = '{2}'\t time = '{3}'", i, op.op_, op.opTypeName(),
                               op.cycleBegin_);
        }
    }

    SmallVector<size_t> operationIndexesToRemove;
    // Check if between original op / spillRead and spillWrite buffer
    // from dataOp is used by any operation
    _log.trace("Check on possible removal of spills of data operations:");
    for (auto& dataOp : dataOpSpillTree) {
        _log.nest(1).trace("Operation - '{0}'", dataOp.first);
        auto dataOpSpillIndexes = dataOp.second;
        for (size_t i = 0; i < dataOpSpillIndexes.size() - 1; i++) {
            // Check for a sequence origOp/SpillRead -> SpillWrite
            if (scheduledOps[dataOpSpillIndexes[i]].isSpillWrite()) {
                continue;
            }
            if (!scheduledOps[dataOpSpillIndexes[i + 1]].isSpillWrite()) {
                continue;
            }
            auto& origOrSpillReadOpIndex = dataOpSpillIndexes[i];
            auto nextSpillWriteOpIndex = dataOpSpillIndexes[i + 1];

            VPUX_THROW_UNLESS(origOrSpillReadOpIndex < nextSpillWriteOpIndex,
                              "Incorrect order of indexes of spill read and next spill write ops for scheduledOps");

            bool isBufferUsedAsArgument = false;
            bool isBufferUsedAsResult = false;
            auto buffer = scheduledOps[origOrSpillReadOpIndex].getOutputBuffer(0);
            for (size_t schedOpIdx = origOrSpillReadOpIndex + 1; schedOpIdx < nextSpillWriteOpIndex; schedOpIdx++) {
                // TODO: Maybe it would make sense to create a geenric utility function
                // to check if a given buffer is used as a operation input or output
                auto execOp = _depsInfo.getExecuteOpAtIndex(scheduledOps[schedOpIdx].op_);
                // Check if buffer is used for an operation input
                for (auto operand : execOp->getOperands()) {
                    if (const auto asyncType = mlir::dyn_cast<mlir::async::ValueType>(operand.getType())) {
                        const auto type = mlir::dyn_cast<vpux::NDTypeInterface>(asyncType.getValueType());
                        if (type == nullptr || type.getMemoryKind() != _memKind) {
                            continue;
                        }

                        if (_aliasInfo.getRoot(operand) == buffer) {
                            isBufferUsedAsArgument = true;
                            break;
                        }
                    }
                }
                // Check if buffer is used for an operation output
                for (auto res : execOp.getBodyResults()) {
                    auto resType = mlir::dyn_cast<vpux::NDTypeInterface>(res.getType());
                    if (const auto asyncType = mlir::dyn_cast<mlir::async::ValueType>(res.getType())) {
                        resType = mlir::dyn_cast<vpux::NDTypeInterface>(asyncType.getValueType());
                    }

                    if (resType == nullptr || resType.getMemoryKind() != _memKind) {
                        continue;
                    }

                    if (_aliasInfo.getRoot(res) == buffer) {
                        isBufferUsedAsResult = true;
                        break;
                    }
                }

                if (isBufferUsedAsArgument && isBufferUsedAsResult) {
                    break;
                }
            }

            auto isBufferUsed = isBufferUsedAsArgument || isBufferUsedAsResult;

            // If buffer was not used by any operation in between then given read-write pair is not needed
            // This can happen if scheduler prefetched dataOp which got immediately spilled
            if (!isBufferUsed) {
                _log.nest(2).trace("Buffer not used at all between spillRead/OrigOp '{0}' and next spillWrite op '{1}'",
                                   origOrSpillReadOpIndex, nextSpillWriteOpIndex);
                _log.nest(2).trace("Remove spillRead/OrigOp  - '{0}'", origOrSpillReadOpIndex);
                _log.nest(2).trace("Remove next spillWriteOp - '{0}'", nextSpillWriteOpIndex);

                // Ops can be removed
                operationIndexesToRemove.push_back(origOrSpillReadOpIndex);
                operationIndexesToRemove.push_back(nextSpillWriteOpIndex);

                // If read operation was origOp then change next corresponding operation
                if (scheduledOps[origOrSpillReadOpIndex].isOriginalOp()) {
                    // In such case update next read operation to be original operation
                    for (size_t j = i + 2; j < dataOpSpillIndexes.size(); j++) {
                        auto nextSpillReadIndex = dataOpSpillIndexes[j];
                        if (scheduledOps[nextSpillReadIndex].isSpillRead()) {
                            _log.nest(2).trace("Change next spillRead to origOp - '{0}'", nextSpillReadIndex);
                            scheduledOps[nextSpillReadIndex].opType_ = scheduledOps[origOrSpillReadOpIndex].opType_;
                            // move any active resources from original op to optimized spill op
                            if (scheduledOps[origOrSpillReadOpIndex].hasActiveInputResource()) {
                                VPUX_THROW_WHEN(scheduledOps[nextSpillReadIndex].hasActiveInputResource(),
                                                "Spill read '{0}' expected to have no active resources",
                                                nextSpillReadIndex);
                                scheduledOps[nextSpillReadIndex].inputResourceInfo_ =
                                        scheduledOps[origOrSpillReadOpIndex].inputResourceInfo_;
                            }
                            break;
                        }
                    }
                }
            } else if (isBufferUsedAsArgument && !isBufferUsedAsResult) {
                // If buffer was used just as an argument then next spillWrite can be removed
                // as buffer state has not changed. Following spillRead needs to be turned
                // to OrigOp
                _log.nest(2).trace("Buffer used as input between spillRead/OrigOp '{0}' and next spillWrite op '{1}'",
                                   origOrSpillReadOpIndex, nextSpillWriteOpIndex);
                // TODO:
                // Besides removing spillWrite need to update next spillRead
                // operationIndexesToRemove.push_back(nextSpillWriteOpIndex);
            }
        }
    }

    // Sort operation indexes
    std::sort(operationIndexesToRemove.begin(), operationIndexesToRemove.end());

    // Remove in reverse order to have indexes valid after erasing entries in scheduledOp
    for (auto opIt = operationIndexesToRemove.rbegin(); opIt != operationIndexesToRemove.rend(); opIt++) {
        scheduledOps.erase(scheduledOps.begin() + *opIt);
    }
    _log.trace("Operations that have been removed - '{0}'", operationIndexesToRemove.size());
}

// This function tries to eliminate redundant spill write operations if exactly the same
// buffer was already spilled before and resides in DDR. In such case subsequent
// spill write can be removed leaving just the needed spill read that will refer
// to first DDR location of spilled buffer
void FeasibleMemorySchedulerSpilling::removeRedundantSpillWrites(
        FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps) {
    _log.trace("Remove redundant Spill Writes");

    SmallVector<size_t> spillWriteIndexes;
    SmallVector<bool> isSpillWriteDuplicate(scheduledOps.size(), false);
    size_t numSpillWritesToRemove = 0;

    // Traverse whole scheduled ops structure and check each spill write/read op
    for (size_t index = 0; index < scheduledOps.size(); index++) {
        auto& op = scheduledOps[index];
        if (op.isSpillWrite()) {
            _log.trace("SPILL WRITE for op '{0}', idx - '{1}'", op.op_, index);
            // For each spill write op check if this is a duplicate by comparing op number and buffer of each
            // previously encountered spill writes
            for (auto spillWriteIndexIt = spillWriteIndexes.rbegin(); spillWriteIndexIt != spillWriteIndexes.rend();
                 spillWriteIndexIt++) {
                if (scheduledOps[*spillWriteIndexIt].op_ == op.op_ &&
                    scheduledOps[*spillWriteIndexIt].getInputBuffer(0) == op.getInputBuffer(0)) {
                    // If op and buffer match then duplicate was detected
                    isSpillWriteDuplicate[index] = true;
                    numSpillWritesToRemove++;
                    _log.nest().trace(
                            "Duplicate spill for op '{0}': SPILL WRITE idx - '{1}', previous SPILL WRITE idx - '{2}'",
                            op.op_, index, *spillWriteIndexIt);
                    break;
                }
            }
            // Store information about position of each spill write for reference
            spillWriteIndexes.push_back(index);
        }
    }

    _log.trace("Spills detected - '{0}', spill writes to remove - '{1}'", spillWriteIndexes.size(),
               numSpillWritesToRemove);

    // Copy operations without spill write duplicates.
    FeasibleMemoryScheduler::ScheduledOpInfoVec scheduledOpsOptimized;
    for (auto op : scheduledOps | indexed) {
        if (!isSpillWriteDuplicate[op.index()]) {
            scheduledOpsOptimized.push_back(op.value());
        }
    }
    scheduledOps = std::move(scheduledOpsOptimized);
}

SmallVector<mlir::Value> FeasibleMemorySchedulerSpilling::getAsyncResultsForBuffer(
        mlir::async::ExecuteOp opThatWasSpilled, mlir::Value buffer) {
    SmallVector<mlir::Value> buffersToCheck = {buffer};
    SmallVector<mlir::Value> asyncResults;

    // Search if this buffer is a replacement for some original buffer which got spilled
    // If such original buffer is located use it for aliases as this information is not updated
    // with new buffers in dependent operations
    for (auto& replacementPairs : _bufferReplacementAfterSpillRead) {
        if (replacementPairs.second == buffer) {
            buffersToCheck.push_back(replacementPairs.first);
        }
    }
    for (auto& bufferToCheck : buffersToCheck) {
        for (auto bufferAlias : _aliasInfo.getAllAliases(bufferToCheck)) {
            if (mlir::isa<mlir::async::ValueType>(bufferAlias.getType()) &&
                bufferAlias.getDefiningOp() == opThatWasSpilled.getOperation()) {
                asyncResults.push_back(bufferAlias);
            }
        }
    }

    VPUX_THROW_WHEN(asyncResults.empty(),
                    "No async result matched for a given buffer\n buffer - {0}\n op that was spilled - {1}", buffer,
                    opThatWasSpilled);

    return asyncResults;
}

mlir::Value FeasibleMemorySchedulerSpilling::getBufferFromAsyncResult(mlir::Value asyncResult) {
    const auto resultType = asyncResult.getType();
    VPUX_THROW_UNLESS(mlir::isa<mlir::async::ValueType>(resultType), "This is not async result. Got: '{0}'",
                      resultType);
    return _aliasInfo.getRoot(asyncResult);
}

mlir::async::ExecuteOp FeasibleMemorySchedulerSpilling::insertSpillWriteDmaOp(mlir::async::ExecuteOp opThatWasSpilled,
                                                                              mlir::async::ExecuteOp insertAfterExecOp,
                                                                              mlir::Value bufferToSpill,
                                                                              size_t allocatedAddress, int spillId) {
    auto spillWriteNameLoc = appendLoc(opThatWasSpilled->getLoc(), "{0}{1}", SPILL_WRITE_OP_NAME_SUFFIX,
                                       _depsInfo.getIndex(opThatWasSpilled));
    _log.trace("Insert Spill Write dmaOp - '{0}'", spillWriteNameLoc);

    // Get spill destination buffer type (memref) from the provided
    // type of source buffer that is to be spilled
    auto getSpillBufferType = [&](vpux::NDTypeInterface type) -> mlir::Type {
        auto spillType = type;
        if (auto distBuffType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(type)) {
            spillType = distBuffType.getCompactType();
        }

        auto secondLvlMemAttr = IndexedSymbolAttr::get(spillType.getContext(), stringifyEnum(_secondLvlMemKind));
        return spillType.changeMemSpace(secondLvlMemAttr);
    };
    auto spillBufferType = getSpillBufferType(bufferToSpill.getType());

    // Update address of the buffer that is to be spilled as spillWrite source buffer
    // is not correctly configured during scheduler memory allocation
    _scan.handler().setAddress(bufferToSpill, allocatedAddress);

    // Create buffer in second level memory
    mlir::OpBuilder builder(_allocOpInsertionPoint);
    builder.setInsertionPoint(_allocOpInsertionPoint);

    mlir::IntegerAttr swizzlingKeyAttr;
    if (auto allocOp = bufferToSpill.getDefiningOp<VPURT::Alloc>()) {
        swizzlingKeyAttr = allocOp.getSwizzlingKeyAttr();
    } else if (auto distAllocOp = bufferToSpill.getDefiningOp<VPURT::AllocDistributed>()) {
        swizzlingKeyAttr = distAllocOp.getSwizzlingKeyAttr();
    }

    mlir::Operation* newBufferOp;
    if (swizzlingKeyAttr) {
        newBufferOp = builder.create<VPURT::Alloc>(spillWriteNameLoc, spillBufferType, nullptr, swizzlingKeyAttr);
    } else {
        newBufferOp =
                builder.create<mlir::memref::AllocOp>(spillWriteNameLoc, mlir::cast<mlir::MemRefType>(spillBufferType));
    }
    auto newBufferResult = newBufferOp->getResult(0);

    // Update aliases info for newly created root buffer
    _aliasInfo.addAlias(newBufferResult, newBufferResult);

    // Create new AsyncExecOp
    builder.setInsertionPointAfter(insertAfterExecOp);
    auto spillWriteExecOp = builder.create<mlir::async::ExecuteOp>(spillWriteNameLoc, newBufferResult.getType(),
                                                                   std::nullopt, std::nullopt);

    VPUIP::NNDMAOp spillWriteDmaOp;

    auto bodyBlock = spillWriteExecOp.getBody();
    builder.setInsertionPointToStart(bodyBlock);

    // Build body of spill write async exec op
    // Create DmaOp in the body of new AsyncExecOp
    spillWriteDmaOp = builder.create<VPUIP::NNDMAOp>(spillWriteNameLoc, bufferToSpill, newBufferResult);
    spillWriteDmaOp.setSpillId(spillId);
    builder.create<mlir::async::YieldOp>(spillWriteNameLoc, spillWriteDmaOp->getResults());

    // Update aliases for spillWrite result
    _aliasInfo.addAlias(newBufferResult, spillWriteDmaOp.getOutput());
    _aliasInfo.addAlias(newBufferResult, spillWriteExecOp.getBodyResults()[0]);

    // Update executor attributes of new AsyncExecOp
    auto dmaOpExecutor = mlir::dyn_cast_or_null<VPUIP::AsyncLayerOpInterface>(spillWriteDmaOp.getOperation());
    auto executor = dmaOpExecutor.getExecutor();
    if (executor != nullptr) {
        VPUIP::VPUIPDialect::setExecutor(spillWriteExecOp, executor);
    }

    // Update dependencies map and get new operation index
    _depsInfo.insertNewExecOpToDepsMap(spillWriteExecOp);

    // Update dependency
    _depsInfo.addDependency(opThatWasSpilled, spillWriteExecOp);

    return spillWriteExecOp;
}

mlir::async::ExecuteOp FeasibleMemorySchedulerSpilling::insertSpillReadDmaOp(mlir::async::ExecuteOp opThatWasSpilled,
                                                                             mlir::Value bufferToSpill,
                                                                             mlir::async::ExecuteOp spillWriteExecOp,
                                                                             mlir::async::ExecuteOp insertAfterExecOp,
                                                                             size_t allocatedAddress, int spillId) {
    auto spillReadNameLoc = appendLoc(opThatWasSpilled->getLoc(), "{0}{1}", SPILL_READ_OP_NAME_SUFFIX,
                                      _depsInfo.getIndex(opThatWasSpilled));
    _log.trace("Insert Spill Read dmaOp - '{0}'", spillReadNameLoc);

    // Get information about spill write returned memref type and prepare new one with proper memory location
    auto spillWriteResult = spillWriteExecOp.getBodyResults()[0];
    auto spillWriteAsyncType = mlir::dyn_cast<mlir::async::ValueType>(spillWriteResult.getType());

    // Create buffer in first level memory to bring back spilled buffer
    mlir::OpBuilder builder(_allocOpInsertionPoint);
    builder.setInsertionPoint(_allocOpInsertionPoint);
    mlir::Operation* newBufferOp;
    if (auto distAllocOp = bufferToSpill.getDefiningOp<VPURT::AllocDistributed>()) {
        newBufferOp = builder.create<VPURT::AllocDistributed>(spillReadNameLoc, bufferToSpill.getType(),
                                                              distAllocOp.getAlignmentAttr(),
                                                              distAllocOp.getSwizzlingKeyAttr());
    } else if (auto allocOp = bufferToSpill.getDefiningOp<VPURT::Alloc>()) {
        // In case original spilled buffer had swizzling attribute, populate it to newly allocated buffer
        newBufferOp = builder.create<VPURT::Alloc>(spillReadNameLoc, bufferToSpill.getType(),
                                                   allocOp.getAlignmentAttr(), allocOp.getSwizzlingKeyAttr());
    } else {
        newBufferOp = builder.create<mlir::memref::AllocOp>(spillReadNameLoc,
                                                            mlir::cast<mlir::MemRefType>(bufferToSpill.getType()));
    }
    auto newBufferResult = newBufferOp->getResult(0);

    // Update aliases info for newly created root buffer
    _aliasInfo.addAlias(newBufferResult, newBufferResult);

    // Configure address as prepared by scheduler.
    // Since it is a new buffer it was not assigned before
    _scan.handler().setAddress(newBufferResult, allocatedAddress);

    // Store information about what buffer replaces original buffer that was marked for spilling
    // If such replacement pair already exists, then update it with a new buffer
    bool replacementPairFound = false;
    for (auto& replacementPairs : _bufferReplacementAfterSpillRead) {
        if (replacementPairs.second == bufferToSpill) {
            replacementPairs.second = newBufferResult;
            replacementPairFound = true;
            break;
        }
    }
    // If this buffer didn't correspond to any existing buffer replacement pair then insert a new one
    if (!replacementPairFound) {
        _bufferReplacementAfterSpillRead.insert({bufferToSpill, newBufferResult});
    }

    // Create new AsyncExecOp in correct place
    builder.setInsertionPointAfter(insertAfterExecOp);
    auto spillReadExecOp = builder.create<mlir::async::ExecuteOp>(spillReadNameLoc, newBufferOp->getResultTypes(),
                                                                  std::nullopt, std::nullopt);

    // Update operands of new AsyncExecOp to contain result of AsyncExecOp of spillWrite
    spillReadExecOp.getBodyOperandsMutable().append(spillWriteResult);
    auto innerAsyncArgForSpill =
            spillReadExecOp.getBody()->addArgument(spillWriteAsyncType.getValueType(), spillWriteResult.getLoc());

    VPUIP::NNDMAOp spillReadDmaOp;

    auto bodyBlock = spillReadExecOp.getBody();
    builder.setInsertionPointToStart(bodyBlock);

    // Build body of spill read async exec op
    // Create DmaOp in the body of new AsyncExecOp
    spillReadDmaOp = builder.create<VPUIP::NNDMAOp>(spillReadNameLoc, innerAsyncArgForSpill, newBufferResult);
    spillReadDmaOp.setSpillId(spillId);
    builder.create<mlir::async::YieldOp>(spillReadNameLoc, spillReadDmaOp->getResults());

    // Update alias for spillRead result
    _aliasInfo.addAlias(newBufferResult, spillReadDmaOp.getOutput());
    _aliasInfo.addAlias(newBufferResult, spillReadExecOp.getBodyResults()[0]);

    // Update executor attributes of new AsyncExecOp
    auto dmaOpExecutor = mlir::dyn_cast_or_null<VPUIP::AsyncLayerOpInterface>(spillReadDmaOp.getOperation());
    auto executor = dmaOpExecutor.getExecutor();
    if (executor != nullptr) {
        VPUIP::VPUIPDialect::setExecutor(spillReadExecOp, executor);
    }

    // Update dependencies map and get new operation index
    _depsInfo.insertNewExecOpToDepsMap(spillReadExecOp);

    // Update dependency
    _depsInfo.addDependency(spillWriteExecOp, spillReadExecOp);

    return spillReadExecOp;
}

mlir::Operation* FeasibleMemorySchedulerSpilling::SpillUsersUpdate::getViewOpForMasterBuffer(mlir::Value asyncResult) {
    // Identify pure viewOp for master buffer that is related to result of
    // asyncExecOp that is spilled
    mlir::Operation* viewOpForMasterBuffer = nullptr;
    mlir::Value sourceAlias = asyncResult;

    while ((sourceAlias = *(_spillingParentClass._aliasInfo.getSources(sourceAlias).begin()))) {
        auto sourceOp = sourceAlias.getDefiningOp();
        if (sourceOp == nullptr) {
            continue;
        }
        if (VPUIP::isPureViewOp(sourceOp)) {
            if (auto viewOp = mlir::dyn_cast<mlir::ViewLikeOpInterface>(sourceOp)) {
                auto source = viewOp.getViewSource();
                if (mlir::isa<mlir::BlockArgument>(source)) {
                    source = _spillingParentClass._aliasInfo.getRoot(source);
                }

                if (source == _bufferToSpill) {
                    VPUX_THROW_UNLESS(
                            viewOpForMasterBuffer == nullptr,
                            "Chain of pure-view ops is not supported for the same buffer in single async.ExecuteOp. "
                            "Pure-view has already been identified for given asyncResult - {0}",
                            asyncResult);
                    viewOpForMasterBuffer = viewOp.getOperation();
                }
            }
        }
    }

    return viewOpForMasterBuffer;
}

SmallVector<mlir::async::ExecuteOp>
FeasibleMemorySchedulerSpilling::SpillUsersUpdate::getUsersOfSpilledOpThatNeedUpdate(
        mlir::Value opThatWasSpilledResult) {
    // Get all asyncExecOps that are users of result of spilled op and appear in
    // IR after spillRead. Those users would need to be updated to refer to result
    // of spillRead
    SmallVector<mlir::async::ExecuteOp> usersOfSpilledOpThatNeedUpdate;
    for (auto* user : opThatWasSpilledResult.getUsers()) {
        if (mlir::isa_and_nonnull<mlir::async::ExecuteOp>(user) && !user->isBeforeInBlock(_spillReadExecOp)) {
            usersOfSpilledOpThatNeedUpdate.push_back(mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(user));
        }
    }
    return usersOfSpilledOpThatNeedUpdate;
}

unsigned int FeasibleMemorySchedulerSpilling::SpillUsersUpdate::getOperandIndexForSpillResultUser(
        mlir::async::ExecuteOp spillResultUser, mlir::Value spilledAsyncResult) {
    // For a given user of result of spilled operation identify the
    // operand index for this dependency
    unsigned int operandIndex = 0;
    bool operandFound = false;
    for (const auto& operand : spillResultUser.getOperands()) {
        if (mlir::isa<mlir::async::ValueType>(operand.getType())) {
            if (operand == spilledAsyncResult) {
                operandFound = true;
                break;
            }
            operandIndex++;
        }
    }
    VPUX_THROW_UNLESS(operandFound, "Unable to find async.ExecOp operand index matching result of op that was spilled");
    return operandIndex;
}

void FeasibleMemorySchedulerSpilling::SpillUsersUpdate::updateSpillResultUsers(mlir::Value oldResult,
                                                                               mlir::Value newResult) {
    // Find operations which should be excluded from operand update to result of spillRead.
    // Those are all operations which appear in IR before spillRead
    llvm::SmallPtrSet<mlir::Operation*, 1> excludedUsersFromOperandsUpdate;
    for (auto* user : oldResult.getUsers()) {
        if (mlir::isa_and_nonnull<mlir::async::ExecuteOp>(user) &&
            user->isBeforeInBlock(_spillReadExecOp.getOperation())) {
            excludedUsersFromOperandsUpdate.insert(user);
        }
    }

    // Update connections opThatWasSpilled -> SpillWrite -> SpillRead -> UserOfSpilledBuffer
    oldResult.replaceAllUsesExcept(newResult, excludedUsersFromOperandsUpdate);
}

void FeasibleMemorySchedulerSpilling::SpillUsersUpdate::updateSpillBufferUsers(mlir::Value oldBuffer,
                                                                               mlir::Value newBuffer) {
    // Get information about the users of original output buffer that should still refer to it
    // (e.g. operations that appear in IR before)
    llvm::SmallPtrSet<mlir::Operation*, 1> excludedUsersFromOrigBufferUpdate;
    for (auto* user : oldBuffer.getUsers()) {
        if (user != nullptr) {
            if (user->getParentOp()->isBeforeInBlock(_spillReadExecOp)) {
                excludedUsersFromOrigBufferUpdate.insert(user);
            }
        }
    }

    // Update all users of original output buffer with the new buffer from spillRead except
    // the operations which were identified to refer to old output buffer
    oldBuffer.replaceAllUsesExcept(newBuffer, excludedUsersFromOrigBufferUpdate);

    // Handle inplace op also, add aliases to new buffer since inplace operation also writes there
    for (auto* user : newBuffer.getUsers()) {
        if (user != nullptr && !user->getParentOp()->isBeforeInBlock(_spillReadExecOp)) {
            VPUIP::NCEClusterTaskOp nceClusterTask;
            mlir::OpResult possibleInplaceOpResult;

            // Inplace operation may use buffer through view like op that means same buffer usage.
            // Take first operation whcih is not: ViewOp, DistributedCastOp.
            while (mlir::isa<VPUIP::ViewOp, VPUIP::DistributedCastOp>(user)) {
                user = *user->getUsers().begin();
            }

            if (mlir::isa<VPUIP::NCEClusterTaskOp>(user)) {
                nceClusterTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(user);
                possibleInplaceOpResult = nceClusterTask.getResults()[0];
            }

            if (nceClusterTask != nullptr && nceClusterTask.getIsInplace().value_or(false)) {
                auto userOutputRootBuf =
                        *_spillingParentClass._aliasInfo.getRoots(nceClusterTask.getOutputBuff()).begin();
                /* For long term refer to #E70663.
                   Check if inplace operation writes to this buffer.
                   If it does then inplace operation output should be added as alias.
                   Two cases are possible, inplace op writes:
                   1. directly to input. Then match its output with new buffer, root buffer is same as for new.
                   2. via view op. Root is not updated for viewOp result yet,
                      root buffer is root of spilled buffer and it is compared with the old buffer. */
                if (userOutputRootBuf == *_spillingParentClass._aliasInfo.getRoots(oldBuffer).begin() ||
                    userOutputRootBuf == *_spillingParentClass._aliasInfo.getRoots(newBuffer).begin()) {
                    _spillingParentClass._aliasInfo.addAlias(newBuffer, possibleInplaceOpResult);
                    _spillingParentClass._aliasInfo.addAlias(
                            newBuffer, nceClusterTask->getParentOfType<mlir::async::ExecuteOp>().getBodyResults()[0]);
                }
            }
        }
    }
}

void FeasibleMemorySchedulerSpilling::SpillUsersUpdate::resolveSpillBufferUsage() {
    auto opThatWasSpilledResults = _spillingParentClass.getAsyncResultsForBuffer(_opThatWasSpilled, _bufferToSpill);

    auto spillReadExecOpResult = _spillReadExecOp.getBodyResults()[0];

    // Users referring to spilled buffer need to be properly updated to now refer to result of spillRead
    for (auto& opThatWasSpilledResult : opThatWasSpilledResults) {
        auto usersOfSpilledOpThatNeedUpdate = getUsersOfSpilledOpThatNeedUpdate(opThatWasSpilledResult);

        // Identify pure viewOp for master buffer that is related to result of asyncExecOp
        // that is spilled. If such operation is located then users need to have
        // similar operation injected to properly refer to replacement of spilled buffer
        auto* viewOpForMasterBuffer = getViewOpForMasterBuffer(opThatWasSpilledResult);
        if (viewOpForMasterBuffer && !usersOfSpilledOpThatNeedUpdate.empty()) {
            for (auto userOfSpilledOpThatNeedUpdate : usersOfSpilledOpThatNeedUpdate) {
                auto userOfSpilledOpBodyBlock = userOfSpilledOpThatNeedUpdate.getBody();
                // Insert view Op defining relation between spilled buffer and user of
                // asyncExecOp result referring to this buffer
                mlir::OpBuilder builder(userOfSpilledOpThatNeedUpdate);
                builder.setInsertionPointToStart(userOfSpilledOpBodyBlock);
                auto newViewOp = builder.clone(*viewOpForMasterBuffer);

                // Get asyncExecOp argument index related to result of spilled asyncExecOp
                auto operandIndex =
                        getOperandIndexForSpillResultUser(userOfSpilledOpThatNeedUpdate, opThatWasSpilledResult);

                // Get argument of asyncExecOp block that would need to be updated
                // to be used in the body through newly inserted view op
                auto arg = userOfSpilledOpBodyBlock->getArgument(operandIndex);
                arg.replaceAllUsesWith(newViewOp->getOpResult(0));

                auto finalType = newViewOp->getOpOperand(0).get().getType();
                newViewOp->setOperand(0, arg);
                newViewOp->getOpOperand(0).get().setType(finalType);
            }
        }
        updateSpillResultUsers(opThatWasSpilledResult, spillReadExecOpResult);
    }

    // Get new output buffer that is the result of spillRead
    auto newOutputBuffer = _spillingParentClass.getBufferFromAsyncResult(spillReadExecOpResult);

    // If there are operations which were referring directly to output buffer that was spilled
    // they should be updated to refer to result of spillRead if they appear in the IR
    // after the op whose result was spilled
    updateSpillBufferUsers(_bufferToSpill, newOutputBuffer);
}

// This function will update operands of users of spilled buffer
// and make proper connections
void FeasibleMemorySchedulerSpilling::updateSpillWriteReadUsers(mlir::Value bufferToSpill,
                                                                mlir::async::ExecuteOp spillWriteExecOp,
                                                                mlir::async::ExecuteOp spillReadExecOp) {
    _log.trace("Update users of Spill Write-Read pair: '{0}' -> '{1}'", spillWriteExecOp->getLoc(),
               spillReadExecOp->getLoc());

    // Find asyncExecOps which have result corresponding to buffer that got spilled
    SmallVector<mlir::async::ExecuteOp> opsThatWereSpilled;
    for (auto bufferAlias : _aliasInfo.getAllAliases(bufferToSpill)) {
        if (mlir::isa<mlir::async::ValueType>(bufferAlias.getType())) {
            if (auto execOpWithSpilledResult = mlir::dyn_cast<mlir::async::ExecuteOp>(bufferAlias.getDefiningOp())) {
                if (execOpWithSpilledResult->isBeforeInBlock(spillReadExecOp)) {
                    opsThatWereSpilled.push_back(execOpWithSpilledResult);
                }
            }
        }
    }

    std::sort(opsThatWereSpilled.begin(), opsThatWereSpilled.end(),
              [](mlir::async::ExecuteOp execOp1, mlir::async::ExecuteOp execOp2) {
                  return execOp2.getOperation()->isBeforeInBlock(execOp1.getOperation());
              });

    for (auto& opThatWasSpilled : opsThatWereSpilled) {
        _log.trace("Resolve users of operation: '{0}'", opThatWasSpilled->getLoc());
        SpillUsersUpdate spillUsersUpdateHandler(*this, opThatWasSpilled, spillReadExecOp, bufferToSpill);
        spillUsersUpdateHandler.resolveSpillBufferUsage();
    }
}

// Create Spill Write operation based on data from feasible scheduler
void FeasibleMemorySchedulerSpilling::createSpillWrite(FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps,
                                                       size_t schedOpIndex) {
    auto& schedOp = scheduledOps[schedOpIndex];
    auto schedOpBuffer = schedOp.getInputBuffer(0);
    _log = _log.nest();
    _log.trace("Create Spill Write for buffer - '{0}', spillId - '{1}'", schedOpBuffer, ++_spillId);

    // Get the insertion point. Pick first non-implicit previous op
    // SpillWrite operation will be inserted just after it
    mlir::async::ExecuteOp spillWriteInsertionPoint = nullptr;
    auto insertionPointIndex = schedOpIndex;
    while (insertionPointIndex > 0) {
        if (scheduledOps[insertionPointIndex].isOriginalOp()) {
            spillWriteInsertionPoint = _depsInfo.getExecuteOpAtIndex(scheduledOps[insertionPointIndex].op_);
            break;
        }
        insertionPointIndex--;
    }
    VPUX_THROW_UNLESS(spillWriteInsertionPoint != nullptr, "No location to insert Spill Write was identified");

    // In scheduledOpInfo structure op_ identifier for a spillWrite operation contains id
    // of the original operation which result had to be spilled
    auto opThatWasSpilled = _depsInfo.getExecuteOpAtIndex(schedOp.op_);

    auto spillBuffer = schedOpBuffer;
    if (_bufferReplacementAfterSpillRead.find(schedOpBuffer) != _bufferReplacementAfterSpillRead.end()) {
        spillBuffer = _bufferReplacementAfterSpillRead[schedOpBuffer];
        _log.trace("Actual buffer for Spill Write - '{0}'", spillBuffer);
    }

    auto spillWriteExecOp = insertSpillWriteDmaOp(opThatWasSpilled, spillWriteInsertionPoint, spillBuffer,
                                                  schedOp.beginInputResource(0), _spillId);
    _spillWriteInfoVec.push_back({schedOpBuffer, spillWriteExecOp, _spillId});

    size_t spillWriteIndex = _depsInfo.getIndex(spillWriteExecOp);
    _log.trace("Spill Write new opId - '{0}'", spillWriteIndex);

    // After implicit spill write operation has been replaced with a proper dma op task then update
    // scheduled ops structure
    schedOp.opType_ = FeasibleMemoryScheduler::EOpType::ORIGINAL_SPILL_WRITE_OP;
    schedOp.op_ = spillWriteIndex;
    _log = _log.unnest();
}

// Create Spill Read operation based on data from feasible scheduler
void FeasibleMemorySchedulerSpilling::createSpillRead(FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps,
                                                      size_t schedOpIndex) {
    auto& schedOp = scheduledOps[schedOpIndex];
    auto schedOpBuffer = schedOp.getOutputBuffer(0);
    _log = _log.nest();
    // Get most recent spillWrite operation for the given spillRead to properly
    // connect both operations
    auto spillWriteInfo =
            std::find_if(_spillWriteInfoVec.rbegin(), _spillWriteInfoVec.rend(), [&](SpillWriteInfo info) {
                return (info.spilledBuffer == schedOpBuffer);
            });
    VPUX_THROW_UNLESS(spillWriteInfo != _spillWriteInfoVec.rend(),
                      "No matching spill write operation identified for a given Spill Read (opIdx '{0}')", schedOp.op_);

    _log.trace("Create Spill Read for buffer - '{0}', spillId - '{1}'", schedOpBuffer, spillWriteInfo->spillId);
    auto spillWriteExecOp = spillWriteInfo->execOp;

    // Get the insertion point. Pick first non-implicit previous op
    // SpillRead operation will be inserted just after it
    mlir::async::ExecuteOp spillReadInsertionPoint = nullptr;
    auto insertionPointIndex = schedOpIndex;
    while (insertionPointIndex > 0) {
        if (scheduledOps[insertionPointIndex].isOriginalOp()) {
            spillReadInsertionPoint = _depsInfo.getExecuteOpAtIndex(scheduledOps[insertionPointIndex].op_);
            break;
        }
        insertionPointIndex--;
    }
    VPUX_THROW_UNLESS(spillReadInsertionPoint != nullptr, "No location to insert Spill Read was identified");

    // In scheduledOpInfo structure op_ identifier for a spillRead operation contains id
    // of the original operation which result had to be spilled
    auto opThatWasSpilled = _depsInfo.getExecuteOpAtIndex(schedOp.op_);

    auto spillBuffer = schedOpBuffer;
    if (_bufferReplacementAfterSpillRead.find(schedOpBuffer) != _bufferReplacementAfterSpillRead.end()) {
        spillBuffer = _bufferReplacementAfterSpillRead[schedOpBuffer];
        _log.trace("Actual buffer for Spill Read - '{0}'", spillBuffer);
    }
    auto spillReadExecOp =
            insertSpillReadDmaOp(opThatWasSpilled, spillBuffer, spillWriteExecOp, spillReadInsertionPoint,
                                 schedOp.beginOutputResource(0), spillWriteInfo->spillId);

    // After both SpillWrite and SpillRead are inserted update connections
    updateSpillWriteReadUsers(spillBuffer, spillWriteExecOp, spillReadExecOp);

    size_t spillReadIndex = _depsInfo.getIndex(spillReadExecOp);
    _log.trace("Spill Read new opId - '{0}'", spillReadIndex);

    // If there are any other spill operations referring to the same op,
    // update them to refer to new spillRead operation
    for (size_t i = schedOpIndex + 1; i < scheduledOps.size(); i++) {
        auto& otherSchedOp = scheduledOps[i];
        if (otherSchedOp.op_ == schedOp.op_ &&
            ((otherSchedOp.isSpillWrite() && otherSchedOp.getInputBuffer(0) == schedOpBuffer) ||
             (otherSchedOp.isSpillRead() && otherSchedOp.getOutputBuffer(0) == schedOpBuffer))) {
            otherSchedOp.op_ = spillReadIndex;
        }
    }
    // After implicit spillRead operation has been replaced with a proper dma op task then update
    // scheduled ops structure
    schedOp.opType_ = FeasibleMemoryScheduler::EOpType::ORIGINAL_SPILL_READ_OP;
    schedOp.op_ = spillReadIndex;
    _log = _log.unnest();
}

// This method will go through all scheduled ops and when spill
// operation is identified it will translate it to required DmaOp
void FeasibleMemorySchedulerSpilling::insertSpillDmaOps(FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps) {
    _log.trace("Insert Spill DmaOps if needed");
    _log = _log.nest();

    // Locate first async-exec-op that will be used to determine insertion point for
    // new allocation operations
    _allocOpInsertionPoint = _depsInfo.getExecuteOpAtIndex(scheduledOps.begin()->op_).getOperation();
    VPUX_THROW_UNLESS(_allocOpInsertionPoint != nullptr,
                      "Unable to find insertion point for new allocation operations");

    /* Calc number of spills and allocate space in advance
       otherwise resizing 2D vectors N times where N is number of spills reads and writes
       is an expensive operation. */
    int numberOfSpillOps = std::count_if(scheduledOps.begin(), scheduledOps.end(),
                                         [](vpux::FeasibleMemoryScheduler::ScheduledOpInfo& schedOp) {
                                             return schedOp.isSpillWrite() || schedOp.isSpillRead();
                                         });
    _depsInfo.preAllocateForNewOps(numberOfSpillOps);

    for (size_t i = 0; i < scheduledOps.size(); i++) {
        auto& schedOp = scheduledOps[i];
        if (schedOp.isSpillWrite()) {
            _log.trace("Spill Write needed for opId - '{0}'", scheduledOps[i].op_);
            createSpillWrite(scheduledOps, i);
        } else if (schedOp.isSpillRead()) {
            _log.trace("Spill Read needed for opId - '{0}'", scheduledOps[i].op_);
            createSpillRead(scheduledOps, i);
        }
    }
    _log = _log.unnest();
    _log.trace("Spill dmaOps resolved");
}
