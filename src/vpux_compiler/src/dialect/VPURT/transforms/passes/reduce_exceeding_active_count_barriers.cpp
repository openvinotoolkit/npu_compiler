//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/core/execution_group_analysis.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_REDUCEEXCEEDINGACTIVECOUNTBARRIERS
#define GEN_PASS_DEF_REDUCEEXCEEDINGACTIVECOUNTBARRIERS
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class ReduceExceedingActiveCountBarriersPass final :
        public VPURT::impl::ReduceExceedingActiveCountBarriersBase<ReduceExceedingActiveCountBarriersPass> {
public:
    explicit ReduceExceedingActiveCountBarriersPass(std::optional<int> virtualBarrierThresholdForWlm,
                                                    std::optional<WorkloadManagementMode> workloadManagementMode,
                                                    const bool unevenVariantSplit, Logger log)
            : _virtualBarrierThresholdForWlm(virtualBarrierThresholdForWlm),
              _workloadManagementMode(workloadManagementMode),
              _unevenVariantSplitFlag(unevenVariantSplit) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool linearizeTasks(std::set<size_t>& linearizationTasks, BarrierInfo& barrierInfo);
    void linearizeBarriers(mlir::DenseSet<VPURT::DeclareVirtualBarrierOp>& barrierOps, BarrierInfo& barrierInfo);

    // When linearizing tasks barriers are being inserted to represent new control flow. During
    // this process already existing dependency might be checked to prevent from barrier insertion.
    // This is optional feature as nevertheless unnecessary dependencies are being removed in optimizeBarriers step
    // The benefit of having this disabled is smaller memory footprint
    bool _checkDependencyWhenLinearizing = true;
    bool _rebuildControlMap = true;
    const bool _considerTaskFifoDependency = false;
    bool _mergeWaitBarriersIteratively = false;
    bool _considerTaskExecutorType = false;
    bool _shareWaitAndUpdateBarriers = true;

    std::optional<int> _virtualBarrierThresholdForWlm = std::nullopt;
    std::optional<WorkloadManagementMode> _workloadManagementMode = std::nullopt;
    bool _unevenVariantSplitFlag = false;
    SmallVector<llvm::BitVector> _taskControlMap;
    size_t _controlMapOffset = 0;
    size_t _controlMapBlockIdx = 0;
    size_t _availableSlots = 0;

    // Remove checks of slot count limits in barrier optimizations during linearization in order to merge more
    // barriers from parallel branches that wouldn't be merged otherwise. Subsequent analysis verifies the slot
    // count and if necessary, enforce barrier split.
    const bool _checkSlotCountWhenOptimizing = false;
};

// check if taskA and taskB use the same barriers
bool useSameBarriers(size_t taskA, size_t taskB, BarrierInfo& barrierInfo) {
    if (barrierInfo.getWaitBarriers(taskA) != barrierInfo.getWaitBarriers(taskB)) {
        return false;
    }

    return barrierInfo.getUpdateBarriers(taskA) == barrierInfo.getUpdateBarriers(taskB);
}

// linearize tasks to execute sequentially
/*
                                   TaskOp1
                                      |
                                     Bar1
                                      |
    TaskOp1...TaskOpM     =>         ...
                                      |
                                     BarM
                                      |
                                   TaskOpM
*/
bool ReduceExceedingActiveCountBarriersPass::linearizeTasks(std::set<size_t>& linearizationTasks,
                                                            BarrierInfo& barrierInfo) {
    _log.trace("Linearizing tasks");

    if (linearizationTasks.size() < 2) {
        _log.trace("Can not linearize '{0}' tasks", linearizationTasks.size());
        return false;
    }

    // account for parallel tasks with same wait & update barrier(s) which do not produce new barriers
    const auto findEndItr = [&](std::set<size_t>::iterator currItr) {
        auto slotCount = barrierInfo.getNumOfSlotsUsed(barrierInfo.getTaskOpAtIndex(*currItr));
        auto endItr = currItr;
        ++endItr;
        while (endItr != linearizationTasks.end() && useSameBarriers(*currItr, *endItr, barrierInfo)) {
            slotCount += barrierInfo.getNumOfSlotsUsed(barrierInfo.getTaskOpAtIndex(*endItr));
            if (slotCount > _availableSlots) {
                return endItr;
            }
            ++endItr;
        }
        return endItr;
    };

    auto currTask = linearizationTasks.begin();
    auto nextTask = currTask;
    ++nextTask;

    auto earliestConsumer = barrierInfo.getTaskOpAtIndex(*currTask);
    mlir::OpBuilder builder(earliestConsumer);
    builder.setInsertionPoint(earliestConsumer);

    auto getBlockIndexesForTasksBatch = [&](std::set<size_t>& linearizationTasks) {
        std::unordered_map<size_t, size_t> blockIndexes;
        for (auto task : linearizationTasks) {
            blockIndexes[task] = barrierInfo.getControlGraphBlockIndex(task);
        }
        return blockIndexes;
    };

    auto linearizationTasksBlockIndexes = getBlockIndexesForTasksBatch(linearizationTasks);
    if (_checkDependencyWhenLinearizing && _rebuildControlMap) {
        _controlMapBlockIdx = barrierInfo.isSyncPoint(*currTask) ? linearizationTasksBlockIndexes[*nextTask]
                                                                 : linearizationTasksBlockIndexes[*currTask];
        std::tie(_taskControlMap, _controlMapOffset) =
                barrierInfo.buildTaskControlMap(_controlMapBlockIdx, _considerTaskFifoDependency);
    }
    _rebuildControlMap = false;
    nextTask = currTask;

    // linearize all tasks
    bool linearized = false;
    while (currTask != linearizationTasks.end() && nextTask != linearizationTasks.end()) {
        // find sections of tasks using same barriers
        auto producersEnd = findEndItr(currTask);
        nextTask = producersEnd;
        if (nextTask == linearizationTasks.end()) {
            break;
        }
        auto consumersEnd = findEndItr(nextTask);

        if (_checkDependencyWhenLinearizing) {
            unsigned currTaskBlockIdx = linearizationTasksBlockIndexes[*currTask];
            unsigned nextTaskBlockIdx = linearizationTasksBlockIndexes[*nextTask];
            if (nextTaskBlockIdx > _controlMapBlockIdx) {
                _controlMapBlockIdx = nextTaskBlockIdx;
                std::tie(_taskControlMap, _controlMapOffset) =
                        barrierInfo.buildTaskControlMap(_controlMapBlockIdx, _considerTaskFifoDependency);
            }

            if (currTaskBlockIdx != nextTaskBlockIdx) {
                currTask = nextTask;
                ++nextTask;
                continue;
            } else if (currTaskBlockIdx == _controlMapBlockIdx) {
                // skip if barrier already exists
                // TODO: E#80600 also check FIFO dependency
                if (barrierInfo.controlPathExistsBetweenTasksInSameBlock(_taskControlMap, *currTask - _controlMapOffset,
                                                                         *nextTask - _controlMapOffset)) {
                    currTask = nextTask;
                    ++nextTask;
                    continue;
                }
            }
        }

        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(earliestConsumer->getLoc());
        _log.trace("Created new barrier '{0}'", newBarrier);
        barrierInfo.addNewBarrier(newBarrier);

        while (currTask != producersEnd) {
            _log.nest().trace("Add producer '{0}' to new barrier", *currTask);
            barrierInfo.addProducer(newBarrier, *currTask);
            ++currTask;
        }
        _log.trace("Producer slots number - {0}", barrierInfo.getProducerSlotCount(newBarrier));

        while (nextTask != consumersEnd) {
            _log.nest().trace("Add consumer '{0}' to new barrier", *nextTask);
            barrierInfo.addConsumer(newBarrier, *nextTask);
            ++nextTask;
        }
        _log.trace("Consumer slots number - {0}", barrierInfo.getConsumerSlotCount(newBarrier));

        linearized = true;
    }

    return linearized;
}

// linearize barriers such that tasks execute sequentially
/*
    TaskOp1...TaskOpK       TaskOp1 - Bar - ... - Bar - TaskOpK
            |                                |
       Bar1...BarN      =>              Bar1...BarN
            |                                |
    TaskOp1...TaskOpM       TaskOp1 - Bar - ... - Bar - TaskOpm
*/
void ReduceExceedingActiveCountBarriersPass::linearizeBarriers(
        mlir::DenseSet<VPURT::DeclareVirtualBarrierOp>& barrierOps, BarrierInfo& barrierInfo) {
    _log.trace("Linearizing barriers");

    // store barrier producers and consumers to linearize
    std::set<size_t> tasksToLinearize;
    for (const auto& barrierOp : barrierOps) {
        _log.nest().trace("Barrier '{0}'", barrierOp);

        const auto& barrierProducers = barrierInfo.getBarrierProducers(barrierOp);
        tasksToLinearize.insert(barrierProducers.begin(), barrierProducers.end());

        const auto& barrierConsumers = barrierInfo.getBarrierConsumers(barrierOp);
        tasksToLinearize.insert(barrierConsumers.begin(), barrierConsumers.end());
    }

    // TODO: try to efficiently linearize producers or consumers only
    // Note: might increase compilation time

    // linearize producers and consumers
    auto linearized = linearizeTasks(tasksToLinearize, barrierInfo);
    _log.trace("Linearized = '{0}' producers and consumers", linearized);
}

void ReduceExceedingActiveCountBarriersPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto arch = VPU::getArch(module);

    const auto numBarriersToUse = numBarriers.hasValue() ? checked_cast<size_t>(numBarriers.getValue())
                                                         : checked_cast<size_t>(VPUIP::getNumAvailableBarriers(func));
    const auto maxAvailableSlots = maxVariantCount.hasValue() ? checked_cast<size_t>(maxVariantCount.getValue())
                                                              : VPUIP::getBarrierMaxVariantCount(func);
    const auto maxSlotsSum = VPUIP::getBarrierMaxVariantSum(func);
    _log.trace("There are {0} physical barriers and {1} slots for each barrier", numBarriersToUse, maxAvailableSlots);
    _log.trace("Run-time limit for sum of producers and consumers per barrier: {0}", maxSlotsSum);

    // divide slots equally between producers and consumers and split barriers if needed
    _availableSlots = VPUIP::getAvailableSlots(maxSlotsSum, maxAvailableSlots);

    VPUX_THROW_UNLESS(numBarriersToUse > 1, "Not possible to satisfy barrier requirement numBarriersToUse '{0}'",
                      numBarriersToUse);

    auto wlmFlag = (vpux::VPUIP::getWlmStatus(module) == vpux::VPUIP::WlmStatus::ENABLED) && !isArchVPUX3XXX(arch);
    auto shareWaitAndUpdateBarriers = shareWaitAndUpdateBarriersOpt.hasValue()
                                              ? static_cast<bool>(shareWaitAndUpdateBarriersOpt.getValue())
                                              : _shareWaitAndUpdateBarriers;

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    if (barrierInfo.getNumOfBarrierOps() <= numBarriersToUse) {
        _log.trace("Fewer barriers '{0}', than physical barriers.", barrierInfo.getNumOfBarrierOps());
        barrierInfo.clearAttributes();
        return;
    }

    if (_unevenVariantSplitFlag) {
        barrierInfo.enableUnevenVariantSplit();
    }

    if (wlmFlag && _virtualBarrierThresholdForWlm.has_value() &&
        barrierInfo.getNumOfBarrierOps() > static_cast<size_t>(_virtualBarrierThresholdForWlm.value())) {
        _log.trace("WLM flag turned off because number of barrier is above threshold {0} > {1}",
                   barrierInfo.getNumOfBarrierOps(), _virtualBarrierThresholdForWlm.value());
        wlmFlag = false;
        vpux::VPUIP::setWlmStatus(module, vpux::VPUIP::WlmStatus::FAILED);
    }

    if (wlmFlag) {
        // In case of WLM all tasks need to be driven by single barrier as this is one of the constraints
        // to make each schedule feasible for WLM enabling
        _mergeWaitBarriersIteratively = true;
        // For some models, strictly enforcing 1-wait barrier per task can lead to performance regression when tasks
        // executor type is not taken into account when batches of tasks must be linearized. Taking into account tasks
        // executor type can help avoid placing tasks from same engine under different barriers, thus not preventing
        // them from running in parallel.
        _considerTaskExecutorType = true;
    }

    VPURT::BarrierSimulator barrierSim(func, wlmFlag, barrierInfo);

    VPUX_THROW_UNLESS(barrierSim.isDynamicBarriers(), "Barrier generated by barrier scheduler must be dynamic");
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");

    if (_considerTaskFifoDependency) {
        barrierInfo.buildTaskQueueTypeMap();
    }

    auto barSimLog = _log.nest();
    barSimLog.setName("barrier-schedule-sim");
    if (mlir::succeeded(barrierSim.simulateBarriers(barSimLog, numBarriersToUse))) {
        _log.trace("Barrier simulation passed with '{0}' barriers, no isses with exceeding barriers count",
                   numBarriersToUse);
        barrierInfo.clearAttributes();
        return;
    }

    SmallVector<mlir::DenseSet<VPURT::DeclareVirtualBarrierOp>> barrierBatchesToLegalize;
    auto barrierBatchesFromPrevIteration = barrierBatchesToLegalize;

    const auto updateAnalysis = [&]() {
        barrierInfo.optimizeBarriers(_checkSlotCountWhenOptimizing, /* considerTaskFifoDependency */ true);
        if (shareWaitAndUpdateBarriers) {
            // Assert tasks are driven by barriers. This will not be required for WLM runs once #E130177 is implemented.
            // Currently, enabling this for WLM can trigger rollback and performance regressions.
            barrierInfo.shareWaitAndUpdateBarriers(_availableSlots);
        }
        VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log);

        barrierSim = VPURT::BarrierSimulator{func, wlmFlag, barrierInfo};
        if (!mlir::succeeded(barrierSim.checkProducerAndConsumerCount(_log))) {
            _log.trace("Active barrier slot count is not valid, will split barriers");
            // split barriers that violate the slot count limits
            barrierInfo.splitBarriersWithExceedingVariantCount(_availableSlots, maxSlotsSum, maxAvailableSlots);
            VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log);
        }

        // merge parallel wait barriers respecting limits of slot count per barrier
        if (barrierInfo.ensureTasksDrivenBySingleBarrier(_availableSlots, _mergeWaitBarriersIteratively,
                                                         _considerTaskExecutorType)) {
            // IR was modified
            VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log);
        }

        barrierSim = VPURT::BarrierSimulator{func, wlmFlag, barrierInfo};
        if (mlir::succeeded(barrierSim.simulateBarriers(barSimLog, numBarriersToUse, true))) {
            _log.trace("Barrier simulation passed with '{0}' barriers, no isses with exceeding barriers count",
                       numBarriersToUse);
            VPUX_THROW_UNLESS(barrierSim.getBarrierBatchesToLegalize().empty(),
                              "Simulation passed, but '{0}' batches to legalize exist",
                              barrierSim.getBarrierBatchesToLegalize().size());
        }
        barrierBatchesToLegalize = barrierSim.getBarrierBatchesToLegalize();

        if (!barrierBatchesToLegalize.empty() && barrierBatchesToLegalize == barrierBatchesFromPrevIteration) {
            // If the simulation cannot proceed even after linearizing all tasks associated with active barriers it will
            // return a set of barriers encountered in the previous iteration. This indicates that more tasks need to be
            // linearized in order to reduce the number of active barriers. Here we calculate barriers associated with
            // tasks that can run in parallel to those indicated by the simulator and add them to the batch for
            // linearization.
            _log.trace("Encountered identical barrier batches as in previous iteration. Linearization will try to "
                       "include barriers associated with parallel tasks.");
            barrierSim.calculateBarrierBatchesForParallelTasks();
            barrierBatchesToLegalize = barrierSim.getBarrierBatchesToLegalize();
        }
    };

    // optimize current barrier state and perform simulation
    updateAnalysis();

    // iterate through barrier batches to legalize and reduce active barrier count in each batch
    for (size_t it = 0; it < barrierInfo.getNumOfBarrierOps() && !barrierBatchesToLegalize.empty(); ++it) {
        _log.trace("Iteration '{0}', there are '{1}' batches", it, barrierBatchesToLegalize.size());

        _rebuildControlMap = true;  // rebuild task control map on each new iteration
        for (auto& activeBarriers : barrierBatchesToLegalize) {
            _log.trace("There are '{0}' active barriers, reduce active barrier count", activeBarriers.size());

            VPUX_THROW_UNLESS(activeBarriers.size() > 0,
                              "Failed to retrieve active barriers from barrier simulation, got '{0}' active barriers",
                              activeBarriers.size());

            // TODO: E#71194 try to merge active barriers
            // TODO: E#71585 use task cycle info
            // Note: currently not merging barriers due to worse performance

            // linearize execution
            if (_considerTaskFifoDependency) {
                barrierInfo.buildTaskQueueTypeMap();
            }
            linearizeBarriers(activeBarriers, barrierInfo);
        }

        updateAnalysis();
        VPUX_THROW_UNLESS(
                barrierBatchesToLegalize.empty() || barrierBatchesToLegalize != barrierBatchesFromPrevIteration,
                "Encountered identical barrier batches as in previous iteration. Cannot reduce active barriers count.");
        barrierBatchesFromPrevIteration = barrierBatchesToLegalize;
    }

    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    auto execGroupAnalysis = ExecutionGroupAnalysis(func);
    auto validateEachTileSeparately = true;
    if (_workloadManagementMode.has_value() && _workloadManagementMode.value() == WorkloadManagementMode::PWLM_V0_LCA) {
        validateEachTileSeparately = false;
    }
    auto execGroups = validateEachTileSeparately ? execGroupAnalysis.getExecutionGroups()
                                                 : execGroupAnalysis.getExecutionGroupsForTile(0);
    VPUX_THROW_UNLESS(barrierInfo.verifyBarriersForTaskDescriptorFetch(execGroups, wlmFlag),
                      "Encountered execution group without required barrier for task descriptor fetch.");

    // remove attributes before removing barriers
    barrierInfo.clearAttributes();

    VPURT::postProcessBarrierOps(func);

    VPUX_THROW_UNLESS(VPURT::verifyBarrierSlots(func, _log), "Barrier slot count check failed");
    auto hasOneWaitBarrierPerTask = VPURT::verifyOneWaitBarrierPerTask(func, _log);
    if (_mergeWaitBarriersIteratively) {
        VPUX_THROW_UNLESS(hasOneWaitBarrierPerTask, "Encountered task with more than one wait barrier");
    }
}

}  // namespace

//
// createReduceExceedingActiveCountBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createReduceExceedingActiveCountBarriersPass(
        std::optional<int> virtualBarrierThresholdForWlm, std::optional<WorkloadManagementMode> workloadManagementMode,
        const bool unevenVariantSplitFlag, Logger log) {
    return std::make_unique<ReduceExceedingActiveCountBarriersPass>(
            virtualBarrierThresholdForWlm, workloadManagementMode, unevenVariantSplitFlag, log);
}
