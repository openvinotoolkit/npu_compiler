//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/core/cycle_cost_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/inference_execution_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <llvm/ADT/SetOperations.h>

namespace vpux::VPURT {
#define GEN_PASS_DECL_SIMPLIFYSCHEDULE
#define GEN_PASS_DEF_SIMPLIFYSCHEDULE
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;
namespace {

// resolve out of order dependencies for tasks in the same FIFO by linking task's update barrier(s)
// to any next task's (in the same FIFO) update barrier(s) that is/are earlier than current update barrier(s)
// re-link DMAs in FIFO: DMA-1 -> DMA-2 -> DMA-3 with following barriers:
/*
           DMA-1                 DMA-1
    DMA-2    |            DMA-2   |
      |      |                \  /
    Bar0     |                Bar0
      |      |       =>        |
    DMA-3    |               DMA-3
         \  /                  |
         Bar1                 Bar1
*/
void resolveOutOfOrderDependencies(const std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& dmaTaskOpQueues,
                                   BarrierInfo& barrierInfo) {
    for (const auto& entry : dmaTaskOpQueues) {
        // operate per queue
        const auto& dmaQueue = entry.second;

        std::set<size_t> nextUpdateBarriers;
        for (const auto& task : dmaQueue | reversed) {
            const auto& updateBarriers = barrierInfo.getUpdateBarriers(task);
            if (updateBarriers.empty()) {
                continue;
            }
            if (!barrierInfo.getWaitBarriers(task).empty()) {
                nextUpdateBarriers.insert(updateBarriers.begin(), updateBarriers.end());
                continue;
            }

            // link earlier next update barriers
            const auto minUpdateBarrier = VPURT::getMinEntry(updateBarriers);
            for (const auto& nextUpdate : nextUpdateBarriers) {
                if (nextUpdate >= minUpdateBarrier) {
                    break;
                }

                barrierInfo.addProducer(barrierInfo.getBarrierOpAtIndex(nextUpdate), task);
            }

            nextUpdateBarriers.insert(updateBarriers.begin(), updateBarriers.end());
        }
    }

    barrierInfo.optimizeBarriers();
}

// check if barrier is waiting for / updating tasks only on the same queue
bool taskQueueIsBottleneck(size_t taskInd, size_t barrierInd, bool producers, BarrierInfo& barrierInfo) {
    BarrierInfo::TaskSet tasks;
    if (producers) {
        tasks = barrierInfo.getBarrierProducers(barrierInd);
    } else {
        tasks = barrierInfo.getBarrierConsumers(barrierInd);
    }

    const auto taskQueueType = VPURT::getTaskQueueType(barrierInfo.getTaskOpAtIndex(taskInd), false);
    for (const auto& task : tasks) {
        if (taskQueueType != VPURT::getTaskQueueType(barrierInfo.getTaskOpAtIndex(task), false)) {
            return false;
        }

        // would indicate a cycle
        if ((producers && task >= taskInd) || (!producers && taskInd >= task)) {
            return false;
        }
    }

    return true;
}

// find barriers between max and min barrier
BarrierInfo::TaskSet findValidBarrierCandidates(size_t maxWaitBarrier, size_t minUpdateBarrier,
                                                const BarrierInfo::TaskSet& maybeNewBarriers) {
    BarrierInfo::TaskSet intermediateBarriers;
    for (const auto& barrierIdn : maybeNewBarriers) {
        if (barrierIdn <= maxWaitBarrier || barrierIdn >= minUpdateBarrier) {
            continue;
        }

        intermediateBarriers.insert(barrierIdn);
    }

    return intermediateBarriers;
}

void linkNewBarriers(size_t taskInd, const BarrierInfo::TaskSet& newBarriers, bool producer, BarrierInfo& barrierInfo) {
    for (const auto& newBarrierInd : newBarriers) {
        if (producer) {
            barrierInfo.addProducer(barrierInfo.getBarrierOpAtIndex(newBarrierInd), taskInd);
        } else {
            barrierInfo.addConsumer(barrierInfo.getBarrierOpAtIndex(newBarrierInd), taskInd);
        }
    }
}

bool taskCanBeLinkedToNewUpdateBarrier(size_t taskInd, size_t barrierInd,
                                       std::map<size_t, VPURT::TaskConfig>& taskIndMap, BarrierInfo& barrierInfo) {
    // check if only consumers on same queue
    if (taskQueueIsBottleneck(taskInd, barrierInd, false, barrierInfo)) {
        return true;
    }

    for (const auto& producerInd : barrierInfo.getBarrierProducers(barrierInd)) {
        // avoid comparing invalid costs
        if (taskIndMap[producerInd].cycleCost == 1) {
            return false;
        }
    }

    const auto taskCycleEnd = taskIndMap[taskInd].cycleStart + taskIndMap[taskInd].cycleCost;
    for (const auto& consumerInd : barrierInfo.getBarrierConsumers(barrierInd)) {
        if (taskCycleEnd > taskIndMap[consumerInd].cycleStart) {
            return false;
        }
    }

    return true;
}

bool taskCanBeLinkedToNewWaitBarrier(size_t taskInd, size_t barrierInd, std::map<size_t, VPURT::TaskConfig>& taskIndMap,
                                     BarrierInfo& barrierInfo) {
    // check if only consumers on same queue
    if (taskQueueIsBottleneck(taskInd, barrierInd, true, barrierInfo)) {
        return true;
    }

    const auto taskCycleBegin = taskIndMap[taskInd].cycleStart;
    for (const auto& producerInd : barrierInfo.getBarrierProducers(barrierInd)) {
        // avoid comparing invalid costs
        if (taskIndMap[producerInd].cycleCost == 1) {
            return false;
        }
        if (taskCycleBegin <= taskIndMap[producerInd].cycleStart + taskIndMap[producerInd].cycleCost) {
            return false;
        }
    }

    return true;
}

// reduce parallel overlapping control flows for FIFO/Queue
// 1) update barriers: for any task's update barrier, find next task with earlier update barriers
// try to link earlier update barriers to current task.
// 2) wait barriers: for any task's wait barrier, find previous task with later wait barriers
// try to link later wait barriers to current task.
// When linking tasks check if performance will not be affected by:
// - barrier is fully dependent on tasks from the same FIFO/Queue
// - barrier producer/consumer simulated cycles will not cause a performance regression
/*
            Bar0                         Bar0
             |   \                        |  \
            ...    \                     ...  DMA-0
             |       \                    |  /
            Bar1   DMA-0                 Bar1
          /  |       /                 /  |
        /   ...    /                 /   ...
      /      |   /         =>      /      |
    DMA-1   Bar2                 DMA-1   Bar2
      \      |                     \      |
        \   ...                      \   ...
          \  |                         \  |
            Bar3                         Bar3
*/
void reduceParallelControlFlowsForQueue(VPURT::TaskConfigVec& ops, std::map<size_t, VPURT::TaskConfig>& taskIndMap,
                                        BarrierInfo& barrierInfo) {
    auto frontTask = ops.begin();

    // handle update barriers
    while (frontTask != ops.end()) {
        const auto frontTaskInd = barrierInfo.getIndex(frontTask->taskOp);
        const auto minUpdateBarrier = VPURT::getMinEntry(barrierInfo.getUpdateBarriers(frontTaskInd));
        if (minUpdateBarrier == std::numeric_limits<size_t>::min()) {
            ++frontTask;
            continue;
        }

        const auto maxWaitBarrier = VPURT::getMaxEntry(barrierInfo.getWaitBarriers(frontTaskInd));
        if (maxWaitBarrier + 1 == minUpdateBarrier) {
            ++frontTask;
            continue;
        }

        // find next tasks wait barrier
        BarrierInfo::TaskSet intermediateBarriers;
        auto nextTask = frontTask;
        ++nextTask;
        while (nextTask != ops.end()) {
            const auto nextTaskInd = barrierInfo.getIndex(nextTask->taskOp);
            const auto& nextWaitBarriers = barrierInfo.getWaitBarriers(nextTaskInd);
            // stop when task max wait barrier > minUpdateBarrier
            if (VPURT::getMaxEntry(nextWaitBarriers) > minUpdateBarrier) {
                break;
            }
            llvm::set_union(intermediateBarriers,
                            findValidBarrierCandidates(maxWaitBarrier, minUpdateBarrier, nextWaitBarriers));
            ++nextTask;
        }

        BarrierInfo::TaskSet newUpdateBarriers;
        for (const auto& barrier : intermediateBarriers) {
            // check if task can be linked
            if (!taskCanBeLinkedToNewUpdateBarrier(frontTaskInd, barrier, taskIndMap, barrierInfo)) {
                continue;
            }
            newUpdateBarriers.insert(barrier);
        }

        // add new barriers
        linkNewBarriers(frontTaskInd, newUpdateBarriers, true, barrierInfo);
        ++frontTask;
    }

    // handle wait barriers
    auto frontTaskRev = ops.begin();
    ++frontTaskRev;
    while (frontTaskRev != ops.end()) {
        const auto frontTaskRevInd = barrierInfo.getIndex(frontTaskRev->taskOp);
        const auto maxWaitBarrier = VPURT::getMaxEntry(barrierInfo.getWaitBarriers(frontTaskRevInd));
        if (maxWaitBarrier == std::numeric_limits<size_t>::max()) {
            ++frontTaskRev;
            continue;
        }

        const auto minUpdateBarrier = VPURT::getMinEntry(barrierInfo.getUpdateBarriers(frontTaskRevInd));
        if (minUpdateBarrier == maxWaitBarrier + 1) {
            ++frontTaskRev;
            continue;
        }

        // find prev tasks wait barrier
        BarrierInfo::TaskSet intermediateBarriers;
        auto prevTask = frontTaskRev;
        do {
            --prevTask;
            const auto prevTaskInd = barrierInfo.getIndex(prevTask->taskOp);
            const auto& prevUpdateBarriers = barrierInfo.getUpdateBarriers(prevTaskInd);
            // stop when task min update barrier < maxWaitBarrier
            if (VPURT::getMinEntry(prevUpdateBarriers) < maxWaitBarrier) {
                break;
            }
            llvm::set_union(intermediateBarriers,
                            findValidBarrierCandidates(maxWaitBarrier, minUpdateBarrier, prevUpdateBarriers));
        } while (prevTask != ops.begin());

        BarrierInfo::TaskSet newWaitBarriers;
        for (const auto& barrier : intermediateBarriers) {
            // check if task can be linked
            if (!taskCanBeLinkedToNewWaitBarrier(frontTaskRevInd, barrier, taskIndMap, barrierInfo)) {
                continue;
            }
            newWaitBarriers.insert(barrier);
        }

        // add new barriers
        linkNewBarriers(frontTaskRevInd, newWaitBarriers, false, barrierInfo);

        ++frontTaskRev;
    }
}

void reduceParallelControlFlows(std::map<VPURT::TaskQueueType, VPURT::TaskConfigVec> taskQueueMap,
                                BarrierInfo& barrierInfo) {
    // create BarrierInfo::index -> TaskConfig mapping
    std::map<size_t, VPURT::TaskConfig> taskIndMap;
    for (auto& entry : taskQueueMap) {
        for (auto& taskConfig : entry.second) {
            taskIndMap[barrierInfo.getIndex(taskConfig.taskOp)] = taskConfig;
        }
    }

    for (auto& entry : taskQueueMap) {
        if (entry.first.type != VPU::ExecutorKind::DMA_NN) {
            continue;
        }

        // reduce per FIFO/Queue
        reduceParallelControlFlowsForQueue(entry.second, taskIndMap, barrierInfo);
    }
}

//
//  SimplifySchedulePass
//

class SimplifySchedulePass final : public VPURT::impl::SimplifyScheduleBase<SimplifySchedulePass> {
public:
    explicit SimplifySchedulePass(const bool reduceParallelControlFlowsFlag,
                                  std::optional<WorkloadManagementMode> workloadManagementMode, Logger log)
            : _reduceParallelControlFlows(reduceParallelControlFlowsFlag),
              _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool _shareWaitAndUpdateBarriers{false};
    bool _reduceParallelControlFlows{true};
    std::optional<WorkloadManagementMode> _workloadManagementMode = std::nullopt;
};

void SimplifySchedulePass::safeRunOnFunc() {
    auto funcOp = getOperation();
    _shareWaitAndUpdateBarriers = VPURT::isShareWaitAndUpdateBarriersNeeded(_workloadManagementMode);

    auto shareWaitAndUpdateBarriers = shareWaitAndUpdateBarriersOpt.hasValue()
                                              ? static_cast<bool>(shareWaitAndUpdateBarriersOpt.getValue())
                                              : _shareWaitAndUpdateBarriers;
    _log.trace("Sharing wait and update barriers: {0}", shareWaitAndUpdateBarriers);

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    const auto arch = config::getArch(module);
    auto maybeCostModelAnalysis = getCachedParentAnalysis<VPU::CostModelAnalysis>(module);
    auto costModel = VPU::CostModelAnalysis::getOrCreateCostModel(maybeCostModelAnalysis, arch, _log);
    CycleCostInfo cycleCostInfo(std::move(costModel), funcOp);

    // If barrier order by producer, some task's wait barrier number is bigger than update barrier, but some functions
    // in this pass like resolveOutOfOrderDependencies and reduceParallelControlFlows need task's wait barrier number
    // smaller than update barrier, so here we will reorder the barrier by consumption to guarantee the requirement.
    // order tasks and barriers
    VPURT::orderExecutionTasksAndBarriers(funcOp, barrierInfo, _log, true);

    // 1. avoid out of order dependencies
    auto dmaTaskOpQueues = VPURT::getTaskOpQueues(funcOp, barrierInfo, VPU::ExecutorKind::DMA_NN);
    // inject dependencies where needed
    resolveOutOfOrderDependencies(dmaTaskOpQueues, barrierInfo);
    // re-order execution
    VPURT::orderExecutionTasksAndBarriers(funcOp, barrierInfo, _log);

    // 2. make execution fully controlled by barriers
    if (shareWaitAndUpdateBarriers) {
        const auto legalVariantCount = VPUIP::getBarrierMaxVariantCount(funcOp) / 2;
        barrierInfo.buildTaskQueueTypeMap();
        barrierInfo.shareWaitAndUpdateBarriers(legalVariantCount);
        // re-order execution
        VPURT::orderExecutionTasksAndBarriers(funcOp, barrierInfo, _log);
    }

    if (_reduceParallelControlFlows) {
        // run simulation to get operation cycles
        VPURT::InferenceExecutionSimulator infSim(_log, funcOp, cycleCostInfo);
        infSim.runSim();

        // Before BarrierScheduling barriers must be ordered not by consumption order but this is done in following
        // passes
        VPURT::orderExecutionTasksAndBarriers(funcOp, barrierInfo, _log, true);

        // 3. reduce parallel control flows based on cycles
        reduceParallelControlFlows(infSim.getQueueTaskMap(), barrierInfo);

        // re-order execution
        VPURT::orderExecutionTasksAndBarriers(funcOp, barrierInfo, _log);

        // optimize
        barrierInfo.optimizeBarriers(/* checkValidSlotCount */ false, /* considerTaskFifoDependency */ false);

        // order to by producer after optimization
        VPURT::orderExecutionTasksAndBarriers(funcOp, barrierInfo, _log);
    }

    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(funcOp);

    _log.info("[SimplifySchedule phase]");
    cycleCostInfo.printNNCacheStatistics(_log);
}

}  // namespace

//
// createSimplifySchedulePass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createSimplifySchedulePass(
        const bool reduceParallelControlFlowsFlag, std::optional<WorkloadManagementMode> workloadManagementMode,
        Logger log) {
    return std::make_unique<SimplifySchedulePass>(reduceParallelControlFlowsFlag, workloadManagementMode, log);
}
