//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_control_edges.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/core/prefetch_data_ops.hpp"
#include "vpux/compiler/core/schedule_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
#include "vpux/compiler/core/developer_build_utils.hpp"
#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

#include <vpu_cost_model.h>

#include <queue>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_FEASIBLEALLOCATION
#define GEN_PASS_DEF_FEASIBLEALLOCATION
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// MemRefAllocRewrite
//

class MemRefAllocRewrite final : public mlir::OpRewritePattern<mlir::memref::AllocOp> {
public:
    MemRefAllocRewrite(LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::memref::AllocOp>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    LinearScanHandler& _allocInfo;
    Logger _log;
};

mlir::LogicalResult MemRefAllocRewrite::matchAndRewrite(mlir::memref::AllocOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp->getLoc());

    const auto val = origOp.getMemref();

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));
    _log.trace("Replace with statically allocated VPURT.DeclareBufferOp (offset = {0})", offset);

    auto valType = mlir::cast<vpux::NDTypeInterface>(val.getType());
    auto section = VPURT::getBufferSection(valType.getMemoryKind());
    auto sectionIndex = valType.getMemSpace().getIndex();

    if (sectionIndex.has_value()) {
        auto sectionIndexArrayAttr = getIntArrayAttr(getContext(), ArrayRef(sectionIndex.value()));
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, sectionIndexArrayAttr,
                                                            offset, nullptr);
    } else {
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, offset);
    }

    return mlir::success();
}

//
// AllocRewrite
//

class AllocRewrite final : public mlir::OpRewritePattern<VPURT::Alloc> {
public:
    AllocRewrite(LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURT::Alloc>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPURT::Alloc origOp, mlir::PatternRewriter& rewriter) const final;

private:
    LinearScanHandler& _allocInfo;
    Logger _log;
};

mlir::LogicalResult AllocRewrite::matchAndRewrite(VPURT::Alloc origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp->getLoc());

    const auto val = origOp.getBuffer();

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));
    _log.trace("Replace with statically allocated VPURT.DeclareBufferOp (offset = {0})", offset);

    auto valType = mlir::cast<vpux::NDTypeInterface>(val.getType());
    auto section = VPURT::getBufferSection(valType.getMemoryKind());
    auto sectionIndex = valType.getMemSpace().getIndex();

    auto swizzlingKey = origOp.getSwizzlingKeyAttr();

    if (sectionIndex.has_value()) {
        auto sectionIndexArrayAttr = getIntArrayAttr(getContext(), ArrayRef(sectionIndex.value()));
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, sectionIndexArrayAttr,
                                                            offset, swizzlingKey);
    } else {
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, nullptr, offset,
                                                            swizzlingKey);
    }

    return mlir::success();
}

//
// AllocDistributedRewrite
//

class AllocDistributedRewrite final : public mlir::OpRewritePattern<VPURT::AllocDistributed> {
public:
    AllocDistributedRewrite(LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURT::AllocDistributed>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPURT::AllocDistributed origOp, mlir::PatternRewriter& rewriter) const final;

private:
    LinearScanHandler& _allocInfo;
    Logger _log;
};

mlir::LogicalResult AllocDistributedRewrite::matchAndRewrite(VPURT::AllocDistributed origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp->getLoc());

    const auto val = origOp.getBuffer();

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));
    _log.trace("Replace with statically allocated VPURT.DeclareBufferOp (offset = {0})", offset);

    auto section = VPURT::getBufferSection(mlir::cast<vpux::NDTypeInterface>(val.getType()).getMemoryKind());

    auto swizzlingKey = origOp.getSwizzlingKeyAttr();

    rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, nullptr, offset, swizzlingKey);

    return mlir::success();
}

//
// FeasibleAllocationPass
//

class FeasibleAllocationPass final : public VPUIP::impl::FeasibleAllocationBase<FeasibleAllocationPass> {
public:
    FeasibleAllocationPass(VPUIP::MemKindCreateFunc memKindCb, VPUIP::MemKindCreateFunc secondLevelmemKindCb,
                           bool linearizeSchedule, bool enablePipelining, bool enablePrefetching,
                           bool optimizeFragmentation, bool optimizeDynamicSpilling, Logger log);

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    void updateAsyncExecuteOpPosition(mlir::func::FuncOp& netFunc, AsyncDepsInfo& depsInfo,
                                      llvm::ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps);
    void updateAsyncExecuteOpPositionOfSpillOps(AsyncDepsInfo& depsInfo,
                                                FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps);
    void assignCyclesToExecOps(AsyncDepsInfo& depsInfo, FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps);
    void linearizeComputeOps(bool linearizeSchedule, bool enablePipelining, mlir::func::FuncOp& netFunc,
                             AsyncDepsInfo& depsInfo);
    void updateCycleAfterAllScheduleOptimizations(AsyncDepsInfo& depsInfo,
                                                  FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps);

private:
    VPUIP::MemKindCreateFunc _memKindCb;
    VPUIP::MemKindCreateFunc _secondLvlMemKindCb;
    VPU::MemoryKind _memKind{vpux::VPU::MemoryKind::CMX_NN};
    VPU::MemoryKind _secondLvlMemKind{vpux::VPU::MemoryKind::DDR};
    mlir::SymbolRefAttr _memKindAttr;
    bool _linearizeSchedule{false};
    bool _enablePipelining{true};
    bool _enablePrefetching{true};
    bool _enableScheduleStatistics{false};
    bool _optimizeFragmentation{true};
    bool _optimizeDynamicSpilling{true};
};

FeasibleAllocationPass::FeasibleAllocationPass(VPUIP::MemKindCreateFunc memKindCb,
                                               VPUIP::MemKindCreateFunc secondLvlmemKindCb, bool linearizeSchedule,
                                               bool enablePipelining, bool enablePrefetching,
                                               bool optimizeFragmentation, bool optimizeDynamicSpilling, Logger log)
        : _memKindCb(std::move(memKindCb)),
          _secondLvlMemKindCb(std::move(secondLvlmemKindCb)),
          _linearizeSchedule(linearizeSchedule),
          _enablePipelining(enablePipelining),
          _enablePrefetching(enablePrefetching),
          _optimizeFragmentation(optimizeFragmentation),
          _optimizeDynamicSpilling(optimizeDynamicSpilling) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult FeasibleAllocationPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    auto maybeMemKind = _memKindCb(memSpaceName.getValue());
    if (!maybeMemKind.has_value()) {
        return mlir::failure();
    }

    _memKind = maybeMemKind.value();
    _memKindAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(_memKind));

    auto maybeSecondLvlMemKind = _secondLvlMemKindCb(secondLvlMemSpaceName.getValue());
    if (!maybeSecondLvlMemKind.has_value()) {
        return mlir::failure();
    }

    _secondLvlMemKind = maybeSecondLvlMemKind.value();

    if (enablePipelining.hasValue()) {
        _enablePipelining = enablePipelining.getValue();
    }

    if (enablePrefetching.hasValue()) {
        _enablePrefetching = enablePrefetching.getValue();
    }

    if (optimizeDynamicSpilling.hasValue()) {
        _optimizeDynamicSpilling = optimizeDynamicSpilling.getValue();
    }

    if (linearizeSchedule.hasValue()) {
        _linearizeSchedule = linearizeSchedule.getValue();
    }

    return mlir::success();
}

// This method will update all AsyncExecOp position in the block so that their
// order is aligned with order generated by list-scheduler. All operations will
// appear in non-descending order of start time. Such reordering is needed as
// execution order has more constraints than topological order that IR is
// aligned with. Without such sorting insertion of token dependency might hit
// an error.
void FeasibleAllocationPass::updateAsyncExecuteOpPosition(
        mlir::func::FuncOp& netFunc, AsyncDepsInfo& depsInfo,
        llvm::ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps) {
    // Update placement of AsyncExecuteOps
    mlir::Operation* prevAsyncOp = nullptr;
    for (auto& schedOp : scheduledOps) {
        if (!schedOp.isOriginalOp()) {
            continue;
        }
        mlir::Operation* asyncOp = depsInfo.getExecuteOpAtIndex(schedOp.op_);
        VPUX_THROW_UNLESS(asyncOp != nullptr, "AsyncOp not located based on index");
        if (prevAsyncOp != nullptr) {
            asyncOp->moveAfter(prevAsyncOp);
        } else {
            // For the first element place it before current first async exec op
            auto firstAsyncExecOp = *(netFunc.getOps<mlir::async::ExecuteOp>().begin());
            asyncOp->moveBefore(firstAsyncExecOp);
        }
        prevAsyncOp = asyncOp;
    }

    VPUIP::moveDeclarationsToTop(netFunc);
}

// This method will insert defined not scheduled operations after current operation and update reScheduled ops
// recursion required for cases where spill will depend on spill e.g.
//                  | SPILL-DPU0 | SPILL-DPU1 |
// DPU0 | DPU1 |
//             | SW |
// where SW uses DPU0 and DPU1 output, SPILL-DPU0 will depend on SW and SPILL-DPU1 will depend on SPILL-DPU0
// recursion required to correctly insert SPILL-DPU1 before SPILL-DPU0
void insertSpillOps(size_t opInd, std::map<size_t, std::set<size_t>>& insertAfterOp,
                    std::map<size_t, FeasibleMemoryScheduler::ScheduledOpInfo*>& spillToOpInfoMap,
                    std::set<size_t>& reScheduledOps, FeasibleMemoryScheduler::ScheduledOpInfoVec& schedule) {
    auto opsToInsert = insertAfterOp[opInd];
    for (auto op : opsToInsert) {
        if (reScheduledOps.find(op) != reScheduledOps.end()) {
            continue;
        }
        insertSpillOps(op, insertAfterOp, spillToOpInfoMap, reScheduledOps, schedule);
        schedule.push_back(*spillToOpInfoMap[op]);
        reScheduledOps.insert(op);
    }
    insertAfterOp.erase(opInd);
}

// This method will update all AsyncExecOp position in the block related to
// spill ops. The goal is to move them close to their dependency which can
// be some data or control flow source or an operation that executes on same queue type
// This is done to get rid of potential long dependencies across IR as scheduler
// initially places spill ops as late as they are needed but in fact based on their
// dependencies and executor type they will be executed earlier
void FeasibleAllocationPass::updateAsyncExecuteOpPositionOfSpillOps(
        AsyncDepsInfo& depsInfo, FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps) {
    // TODO: E#132048 use std::unordered_map with custom hash
    std::map<std::pair<FeasibleMemoryScheduler::QueueType, size_t>, size_t> lastExecOpOnGivenQueueInstance;
    SmallVector<std::pair<size_t, size_t>> pairOfSpillOpAndItsNewPlaceInSchedVec;

    std::map<size_t, std::set<size_t>> spillDepMap;
    std::map<size_t, FeasibleMemoryScheduler::ScheduledOpInfo*> spillToOpInfoMap;
    for (auto schedOp = scheduledOps.begin(); schedOp != scheduledOps.end(); ++schedOp) {
        if (schedOp->queueType.execKind != VPU::ExecutorKind::DMA_NN) {
            continue;
        }

        if (!schedOp->isOriginalSpillWriteOp() && !schedOp->isOriginalSpillReadOp()) {
            for (auto instIndex : schedOp->executorInstanceMask.set_bits()) {
                auto queueKey = std::make_pair(schedOp->queueType, instIndex);
                lastExecOpOnGivenQueueInstance[queueKey] = schedOp->op_;
            }
            continue;
        }

        // store operation dependencies
        const auto opDeps = depsInfo.getOpDeps(schedOp->op_);
        spillDepMap[schedOp->op_].insert(opDeps.begin(), opDeps.end());

        // store previous queue dependency
        for (auto instIndex : schedOp->executorInstanceMask.set_bits()) {
            auto queueKey = std::make_pair(schedOp->queueType, instIndex);
            if (lastExecOpOnGivenQueueInstance.find(queueKey) != lastExecOpOnGivenQueueInstance.end()) {
                spillDepMap[schedOp->op_].insert(lastExecOpOnGivenQueueInstance[queueKey]);
            }
            // update queue dependency for next op
            lastExecOpOnGivenQueueInstance[queueKey] = schedOp->op_;
        }

        // store spill operation
        spillToOpInfoMap[schedOp->op_] = schedOp;
    }

    FeasibleMemoryScheduler::ScheduledOpInfoVec newOrder;
    std::set<size_t> reScheduledOps;
    std::map<size_t, std::set<size_t>> insertAfterOp;
    for (auto schedOp = scheduledOps.rbegin(); schedOp != scheduledOps.rend(); ++schedOp) {
        if (!insertAfterOp[schedOp->op_].empty()) {
            // re-schedule operations after current op
            insertSpillOps(schedOp->op_, insertAfterOp, spillToOpInfoMap, reScheduledOps, newOrder);
        }

        if (schedOp->queueType.execKind != VPU::ExecutorKind::DMA_NN) {
            newOrder.push_back(*schedOp);
            continue;
        }

        if (!schedOp->isOriginalSpillWriteOp() && !schedOp->isOriginalSpillReadOp()) {
            newOrder.push_back(*schedOp);
            continue;
        }

        for (auto dep : spillDepMap[schedOp->op_]) {
            insertAfterOp[dep].insert(schedOp->op_);
        }
    }

    // use reversed new order
    std::reverse(newOrder.begin(), newOrder.end());
    scheduledOps = std::move(newOrder);
}

// This method will assign cycle intervals to 'async.execute' ops based on ScheduledOpInfoVec, and assign
// instance mask (port) to DMA operations.
void FeasibleAllocationPass::assignCyclesToExecOps(AsyncDepsInfo& depsInfo,
                                                   FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps) {
    for (auto& schedOp : scheduledOps) {
        // transfer cycle attributes to IR
        auto execOp = depsInfo.getExecuteOpAtIndex(schedOp.op_);
        execOp->setAttr(cycleBegin, getIntAttr(execOp->getContext(), schedOp.cycleBegin_));
        execOp->setAttr(cycleEnd, getIntAttr(execOp->getContext(), schedOp.cycleEnd_));
        // update instance mask for DMAs
        if (schedOp.queueType.execKind == VPU::ExecutorKind::DMA_NN) {
            SmallVector<uint64_t> executorInstanceMaskVec;
            for (auto portIdx : schedOp.executorInstanceMask.set_bits()) {
                executorInstanceMaskVec.push_back(portIdx);
            }
            VPUIP::VPUIPDialect::setExecutorInstanceMask(
                    execOp, getIntArrayAttr(execOp->getContext(), executorInstanceMaskVec));
        }
    }
}

// This method will re-create execution flow - schedule, after scheduling optimizations which can
// modify order of operations and impact cycles of operations. New cycles will be assigned to
// 'async.execute' operations and ScheduledOpInfoVec will be updated with new execution order.
void FeasibleAllocationPass::updateCycleAfterAllScheduleOptimizations(
        AsyncDepsInfo& depsInfo, FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps) {
    // distribute ops into their execution queues
    std::map<std::pair<FeasibleMemoryScheduler::QueueType, size_t>,
             std::queue<FeasibleMemoryScheduler::ScheduledOpInfo*>>
            perQueueOps;
    for (auto opIt = scheduledOps.begin(); opIt != scheduledOps.end(); opIt++) {
        for (auto instIndex : opIt->executorInstanceMask.set_bits()) {
            auto queueKey = std::make_pair(opIt->queueType, instIndex);
            perQueueOps[queueKey].push(opIt);
        }
    }

    // check if given queue has operations left
    auto opsOnQueue = [&](const std::pair<FeasibleMemoryScheduler::QueueType, size_t> queueKey) {
        return !perQueueOps[queueKey].empty();
    };

    // check if all queues are empty
    auto allQueuesFinished = [&]() {
        for (const auto& entry : perQueueOps) {
            if (opsOnQueue(entry.first)) {
                return false;
            }
        }
        return true;
    };

    // check if given operation is next on all instances used by the operation
    auto nextOnAllUsedInstances = [&](FeasibleMemoryScheduler::ScheduledOpInfo* opInfo) {
        for (auto instIndex : opInfo->executorInstanceMask.set_bits()) {
            auto queueKey = std::make_pair(opInfo->queueType, instIndex);
            if (perQueueOps[queueKey].front()->op_ != opInfo->op_) {
                return false;
            }
        }
        return true;
    };

    // store cycle end of operations used for dependency tracking and schedule cycle
    std::unordered_map<FeasibleMemoryScheduler::operationIdxType, int64_t> opCycleEndMap;

    // check if all dependencies were scheduled
    auto isOperationReady = [&](size_t opIndex, int64_t& depCycleEnd) {
        mlir::async::ExecuteOp execOp = depsInfo.getExecuteOpAtIndex(opIndex);
        VPUX_THROW_UNLESS(execOp != nullptr, "Async ExecuteOp not located based on index");
        for (const auto& dep : execOp.getDependencies()) {
            const auto execDep = dep.getDefiningOp<mlir::async::ExecuteOp>();
            const auto indexDep = depsInfo.getIndex(execDep);

            // could dependencies be not scheduled ?
            auto cycleEndDep = opCycleEndMap.find(indexDep);
            if (cycleEndDep == opCycleEndMap.end()) {
                return false;
            }

            depCycleEnd = std::max(depCycleEnd, cycleEndDep->second);
        }
        return true;
    };

    // store cycle state per queue to find next free cycles on given queue
    std::map<std::pair<FeasibleMemoryScheduler::QueueType, size_t>, int64_t> queueCycleState;

    // find next ready operations from parallel queues, and update their execution cycles
    auto getNextReadyOps = [&]() {
        FeasibleMemoryScheduler::ScheduledOpInfoVec nextReadyOps;
        for (const auto& entry : perQueueOps) {
            const auto queueKey = entry.first;
            if (!opsOnQueue(queueKey)) {
                continue;
            }

            // ensure op next on all instances
            auto currOp = entry.second.front();
            if (!nextOnAllUsedInstances(currOp)) {
                continue;
            }

            // ensure dependencies scheduled
            int64_t depCycleEnd = 0;
            if (!isOperationReady(currOp->op_, depCycleEnd)) {
                continue;
            }

            auto scheduleCycle = depCycleEnd;
            for (auto instIndex : currOp->executorInstanceMask.set_bits()) {
                auto queueKey = std::make_pair(currOp->queueType, instIndex);
                const auto queueFreeCycle = queueCycleState[queueKey];
                scheduleCycle = std::max(scheduleCycle, queueFreeCycle);
            }

            // update cycle info and store
            const auto opCycleCost = currOp->cycleEnd_ - currOp->cycleBegin_;
            currOp->cycleBegin_ = scheduleCycle;
            currOp->cycleEnd_ = scheduleCycle + opCycleCost;
            nextReadyOps.push_back(*currOp);
        }

        VPUX_THROW_WHEN(nextReadyOps.empty(), "No ready operations found");
        return nextReadyOps;
    };

    // find next operation with earliest scheduling cycle
    auto chooseNextOpToSchedule = [](FeasibleMemoryScheduler::ScheduledOpInfoVec& readyOps) {
        return std::min_element(readyOps.begin(), readyOps.end(), [](const auto& op1, const auto& op2) {
            // first cycle begin
            if (op1.cycleBegin_ != op2.cycleBegin_) {
                return op1.cycleBegin_ < op2.cycleBegin_;
            }
            // second smaller tasks first
            if (op1.cycleEnd_ != op2.cycleEnd_) {
                return op1.cycleEnd_ < op2.cycleEnd_;
            }
            // operation index
            if (op1.op_ != op2.op_) {
                return op1.op_ < op2.op_;
            }
            // allow self comparison
            return false;
        });
    };

    // define new order for execution
    FeasibleMemoryScheduler::ScheduledOpInfoVec newOrder;
    while (!allQueuesFinished()) {
        // find next operations from parallel queues which are ready
        auto nextReadyOps = getNextReadyOps();

        // find operation with earliest scheduling cycle
        auto scheduleOp = chooseNextOpToSchedule(nextReadyOps);

        // update IR cycle information
        mlir::async::ExecuteOp targetExecOp = depsInfo.getExecuteOpAtIndex(scheduleOp->op_);
        targetExecOp->setAttr(cycleBegin, getIntAttr(targetExecOp->getContext(), scheduleOp->cycleBegin_));
        targetExecOp->setAttr(cycleEnd, getIntAttr(targetExecOp->getContext(), scheduleOp->cycleEnd_));

        // candidate operation scheduled, update structures
        newOrder.push_back(*scheduleOp);
        if (opCycleEndMap.find(scheduleOp->op_) != opCycleEndMap.end()) {
            // for ops on multiple queues
            opCycleEndMap[scheduleOp->op_] = std::max(opCycleEndMap[scheduleOp->op_], scheduleOp->cycleEnd_);
        } else {
            opCycleEndMap[scheduleOp->op_] = scheduleOp->cycleEnd_;
        }
        for (auto instIndex : scheduleOp->executorInstanceMask.set_bits()) {
            auto queueKey = std::make_pair(scheduleOp->queueType, instIndex);
            queueCycleState[queueKey] = scheduleOp->cycleEnd_;
            VPUX_THROW_UNLESS(perQueueOps[queueKey].front()->op_ == scheduleOp->op_,
                              "Attempt to remove invalid front operation '{0}', should be '{1}'",
                              perQueueOps[queueKey].front()->op_, scheduleOp->op_);
            perQueueOps[queueKey].pop();
        }
    }

    VPUX_THROW_WHEN(scheduledOps.size() != newOrder.size(),
                    "Failed to update IR with optimized schedule, state can be invalid");
    scheduledOps = std::move(newOrder);
}

// This method will inject dependencies between operations based on linearization option
void FeasibleAllocationPass::linearizeComputeOps(bool linearizeSchedule, bool enablePipelining,
                                                 mlir::func::FuncOp& netFunc, AsyncDepsInfo& depsInfo) {
    // various linearization options
    mlir::async::ExecuteOp prevExecOp = nullptr;
    llvm::DenseMap<VPU::ExecutorKind, mlir::async::ExecuteOp> prevExecOpMap;
    netFunc->walk([&](mlir::async::ExecuteOp execOp) {
        const auto currExecutor = VPUIP::VPUIPDialect::getExecutorKind(execOp);
        if (!linearizeSchedule && !VPUIP::VPUIPDialect::isComputeExecutorKind(currExecutor)) {
            return;
        }

        if (linearizeSchedule || !enablePipelining) {
            if (prevExecOp != nullptr) {
                depsInfo.addDependency(prevExecOp, execOp);
            }
            prevExecOp = execOp;
        } else {
            if (prevExecOpMap.find(currExecutor) != prevExecOpMap.end()) {
                depsInfo.addDependency(prevExecOpMap[currExecutor], execOp);
            }
            prevExecOpMap[currExecutor] = execOp;
        }
    });
}

void FeasibleAllocationPass::safeRunOnFunc() {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    parseEnv("IE_NPU_ENABLE_SCHEDULE_STATISTICS", _enableScheduleStatistics);

#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    auto& ctx = getContext();
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    // cluster information
    auto tileOp = config::getTileExecutor(module);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Failed to get NCE_Cluster information");
    auto tileCount = tileOp.getCount();

    auto dmaPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    VPUX_THROW_UNLESS(dmaPorts != nullptr, "Failed to get DMA information");
    auto dmaCount = dmaPorts.getCount();

    // linear scan
    auto available = config::getAvailableMemory(module, _memKindAttr);
    const auto maxSize = available.size();
    uint64_t alignment = vpux::DEFAULT_CMX_ALIGNMENT;

    // Check for reserved memory which memory scheduler should take into account
    // so that they not overlap with other buffers
    auto reservedMemVec = config::getReservedMemOffsetAndSizeVec(module, _memKindAttr);

    LinearScan<mlir::Value, LinearScanHandler> scan(maxSize.count(), reservedMemVec, alignment);
    auto& aliasesInfo = getAnalysis<AliasesInfoMemType<VPU::MemoryKind::CMX_NN>>();
    auto& liveRangeInfo = getAnalysis<MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>>();
    auto& depsInfo = getAnalysis<AsyncDepsInfo>();

    linearizeComputeOps(_linearizeSchedule, _enablePipelining, func, depsInfo);

    // VPUNN cost model
    const auto arch = config::getArch(module);
    auto maybeCostModelAnalysis = getCachedParentAnalysis<VPU::CostModelAnalysis>(module);
    auto costModel = VPU::CostModelAnalysis::getOrCreateCostModel(maybeCostModelAnalysis, arch, _log);

    // If schedule analysis is enabled dynamic spilling stats will be gathered
    vpux::SpillStats dynamicSpillingBeforePrefetching;
    vpux::SpillStats dynamicSpillingAfterPrefetching;
    vpux::SpillStats dynamicSpillingAfterSpillOptimizations;

    // feasible memory scheduler - list scheduler
    FeasibleMemoryScheduler scheduler(_memKind, _secondLvlMemKind, liveRangeInfo, depsInfo, _log, scan, arch, costModel,
                                      tileCount, dmaCount, _enableScheduleStatistics, _optimizeFragmentation);

    // 1. initial schedule
    auto scheduledOps = scheduler.generateSchedule();

    if (_enableScheduleStatistics) {
        dynamicSpillingBeforePrefetching = vpux::getDynamicSpillingStats(scheduledOps);
    }

    // 2. prefetching
    if (_enablePrefetching && !_linearizeSchedule) {
        PrefetchDataOps prefetching(scheduledOps, depsInfo);
        if (prefetching.enableDataOpPrefetching()) {
            VPUIP::moveDeclarationsToTop(func);
            LinearScan<mlir::Value, LinearScanHandler> prefetchScan(maxSize.count(), reservedMemVec, alignment);
            auto prefetchLiveRangeInfo = MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>{func, aliasesInfo};
            // prefetching logic has reordered IR, depsInfo needs to be regenerated since
            // scheduling depends on incrementing value of async-deps-info along IR
            depsInfo = AsyncDepsInfo{func};
            if (!_enablePipelining) {
                linearizeComputeOps(_linearizeSchedule, _enablePipelining, func, depsInfo);
            }

            FeasibleMemoryScheduler schedulerWithPrefetch(_memKind, _secondLvlMemKind, prefetchLiveRangeInfo, depsInfo,
                                                          _log, prefetchScan, arch, costModel, tileCount, dmaCount,
                                                          _enableScheduleStatistics, _optimizeFragmentation);
            scheduledOps = schedulerWithPrefetch.generateSchedule();
            scan = std::move(prefetchScan);
            scheduler.cleanUpAndLogSchedule(scheduledOps);
        }
    }

    // TODO: recurse to strategy with useful info

    if (_enableScheduleStatistics) {
        dynamicSpillingAfterPrefetching = vpux::getDynamicSpillingStats(scheduledOps);
    }

    // 3. optimize spills
    FeasibleMemorySchedulerSpilling spilling(_memKind, _secondLvlMemKind, depsInfo, aliasesInfo, _log, scan);
    if (_optimizeDynamicSpilling) {
        spilling.optimizeDataOpsSpills(scheduledOps);
        spilling.removeComputeOpRelocationSpills(scheduledOps);
        spilling.removeRedundantSpillWrites(scheduledOps);
    }

    if (_enableScheduleStatistics) {
        dynamicSpillingAfterSpillOptimizations = vpux::getDynamicSpillingStats(scheduledOps);
    }

    // 3. re-order IR for spill insertion
    updateAsyncExecuteOpPosition(func, depsInfo, scheduledOps);

    // 4. insert spill dmas
    spilling.insertSpillDmaOps(scheduledOps);

    // 5. add cycle info to async.execute
    assignCyclesToExecOps(depsInfo, scheduledOps);

    // 6. update dependencies
    // Recreate aliasesInfo after spill insertion to get updated information about
    // root buffers of affected spill result users.
    aliasesInfo = AliasesInfoMemType<VPU::MemoryKind::CMX_NN>(func);
    FeasibleMemorySchedulerControlEdges controlEdges(_memKind, depsInfo, aliasesInfo, _log, scan);
    // controlEdges.insertDependenciesBasic(scheduledOps); // Old method, maintained only for debug
    controlEdges.insertMemoryControlEdges(scheduledOps);
    // Linearize DMA tasks before unrolling will introduce additional dependency across different DMA engines.
    // But it's fine for single DMA engine. So insert dependency to simplify barrier scheduling.
    if (dmaCount == 1) {
        controlEdges.insertScheduleOrderDepsForExecutor(scheduledOps, VPU::ExecutorKind::DMA_NN);
    }
    linearizeComputeOps(_linearizeSchedule, _enablePipelining, func, depsInfo);
    controlEdges.updateDependenciesInIR();

    // 7. After dependencies are determined, location of spill operations should be updated
    // to be close to their dependencies
    updateAsyncExecuteOpPositionOfSpillOps(depsInfo, scheduledOps);
    updateAsyncExecuteOpPosition(func, depsInfo, scheduledOps);

    // 8. After all scheduling optimization, as a final step:
    //  8.1. Recreate schedule order and operation cycles after above optimizations
    updateCycleAfterAllScheduleOptimizations(depsInfo, scheduledOps);
    //  8.2. Clean temporary attributes and log schedule
    scheduler.cleanUpAndLogSchedule(scheduledOps);
    //  8.3. Use final order
    updateAsyncExecuteOpPosition(func, depsInfo, scheduledOps);

    if (_enableScheduleStatistics) {
        // verify all dependencies preserved for correct analysis
        verifyDependenciesPreservedInCycles(depsInfo, scheduledOps);

        // schedule statistics
        printScheduleStatistics(func, depsInfo, _log, scheduledOps);

        // dynamic spilling statistics
        printSpillingStatistics(_log, dynamicSpillingBeforePrefetching, dynamicSpillingAfterPrefetching,
                                dynamicSpillingAfterSpillOptimizations);

        // create a tracing JSON
        aliasesInfo = AliasesInfoMemType<VPU::MemoryKind::CMX_NN>(func);
        auto traceLiveRangeInfo = MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>{func, aliasesInfo};
        createTracingJSON(func, traceLiveRangeInfo, scan, maxSize.count());
    }

    // 7. convert to allocated ops
    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addDynamicallyLegalOp<mlir::memref::AllocOp>([&](mlir::memref::AllocOp op) {
        const auto type = mlir::cast<vpux::NDTypeInterface>(op.getMemref().getType());
        return type == nullptr || type.getMemoryKind() != _memKind;
    });
    target.addDynamicallyLegalOp<VPURT::Alloc>([&](VPURT::Alloc op) {
        const auto type = mlir::cast<vpux::NDTypeInterface>(op.getBuffer().getType());
        return type == nullptr || type.getMemoryKind() != _memKind;
    });
    target.addDynamicallyLegalOp<VPURT::AllocDistributed>([&](VPURT::AllocDistributed op) {
        const auto type = mlir::cast<vpux::NDTypeInterface>(op.getBuffer().getType());
        return type == nullptr || type.getMemoryKind() != _memKind;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MemRefAllocRewrite>(scan.handler(), &ctx, _log);
    patterns.add<AllocRewrite>(scan.handler(), &ctx, _log);
    patterns.add<AllocDistributedRewrite>(scan.handler(), &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        _log.error("Failed to replace Alloc/Dealloc Operations");
        signalPassFailure();
        return;
    }

    _log.info("[FeasibleAllocation phase]");
    _log.info("[NN Cache statistics]  {0}", costModel->getPreloadedCacheCounter().printString());
}

}  // namespace

//
// createFeasibleAllocationPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createFeasibleAllocationPass(
        MemKindCreateFunc memKindCb, MemKindCreateFunc secondLvlmemKindCb, const bool linearizeSchedule,
        const bool enablePrefetching, const bool enablePipelining, const bool optimizeFragmentation,
        const bool optimizeDynamicSpilling, Logger log) {
    return std::make_unique<FeasibleAllocationPass>(std::move(memKindCb), std::move(secondLvlmemKindCb),
                                                    linearizeSchedule, enablePipelining, enablePrefetching,
                                                    optimizeFragmentation, optimizeDynamicSpilling, log);
}
