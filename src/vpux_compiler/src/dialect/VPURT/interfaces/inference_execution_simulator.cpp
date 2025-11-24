//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/VPURT/interfaces/inference_execution_simulator.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/shave.hpp"

#include <vpu_cost_model.h>

using namespace vpux;

namespace {

// Get an ID that will identify given execution queue
// This ID does not correspond to any value in HW. It is just used
// by compiler to differentiate queues by some abstract value that
// can be composed of cluster index in case of NCE or ActShave or
// port and channel in case of DMA
int64_t getQueueId(VPURT::TaskOp taskOp) {
    auto* op = taskOp.getInnerTaskOp();
    if (auto dmaTask = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(op)) {
        const auto port = dmaTask.getPortVal();
        VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
        const auto portValue = port.value();

        return getDMAQueueIdEncoding(portValue, dmaTask.getChannelType());
    } else if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
        const auto& dpuTasks = nceOp.getVariants().getOps<VPUIP::DPUTaskOp>();
        VPUX_THROW_UNLESS(!dpuTasks.empty(), "Encountered op '{0}' with empty dpu list", op->getLoc());
        auto dpuTask = *(dpuTasks.begin());
        return dpuTask.getClusterId().value_or(0);
    } else if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
        auto numTiles = VPU::getNumTiles(swKernelOp);
        auto tileIndex = swKernelOp.getTileIndex().value_or(0);
        auto listIndex = swKernelOp.getListIndex().value_or(0);
        return getShaveQueueIdEncoding(numTiles, tileIndex, listIndex);
    }

    return 0;
}

// Helper function used by logger to create clear string about executor instance
// that a given task has executed on. Examples:
// - DMA_NN[port = 0, channel = DDR]
// - NCE[cluster = 0]
// - SHAVE_ACT[cluster = 0]
std::string getTaskQueueInfoString(VPURT::TaskQueueType taskType, VPURT::TaskOp taskOp,
                                   bool supportsDedicatedActShaveQueues) {
    std::string infoStr = stringifyEnum(taskType.type).data();

    auto* op = taskOp.getInnerTaskOp();

    if (auto dmaTask = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(op)) {
        const auto port = dmaTask.getPortVal();
        VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
        const auto portValue = port.value();

        infoStr += "[port = " + std::to_string(portValue);
        auto channel = dmaTask.getChannelType();
        if (channel.has_value()) {
            std::string channelName = stringifyEnum(channel.value()).data();
            infoStr += ", channel = " + channelName;
        }
        infoStr += "]";
    } else if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
        auto getTileStr = [&]() {
            // NOTE: use of cluster, tile and list indexes is not consistent. When tile is used in combination with
            // cluster it is understood that cluster identifies NCE unit and tile identifies its SHAVE sub unit
            // (typically used in profiling). When tile index is used in combination with list index it is the tile that
            // identifies NCE unit and list index identifies its SHAVE sub-unit (typically used in HAS and in backend).
            if (!supportsDedicatedActShaveQueues) {
                return std::string("");
            }
            auto listIndex = swKernelOp.getListIndex().value_or(0);
            return std::string(", tile = ") + std::to_string(listIndex);
        };
        auto tileIndex = swKernelOp.getTileIndex().value_or(0);
        infoStr += "[cluster = " + std::to_string(tileIndex) + getTileStr() + "]";
    } else if (mlir::isa<VPUIP::NCEClusterTaskOp>(op)) {
        infoStr += "[cluster = " + std::to_string(taskType.id) + "]";
    }

    return infoStr;
}

}  // namespace

SmallVector<size_t> vpux::VPURT::getSubTasksStartTime(const SmallVector<size_t>& subTasksCost, size_t startTime,
                                                      size_t queueCount) {
    SmallVector<size_t> queuesCycleTime(queueCount, startTime);
    SmallVector<size_t> subTasksStartTime;

    for (auto& subTaskCost : subTasksCost) {
        auto queueCycleTimeItr = std::min_element(queuesCycleTime.begin(), queuesCycleTime.end());

        subTasksStartTime.push_back(*queueCycleTimeItr);
        *queueCycleTimeItr += subTaskCost;
    }

    return subTasksStartTime;
}

vpux::VPURT::TaskConfig::TaskConfig(VPURT::TaskOp op, SmallVector<int64_t>& virtBarrierWaitVec,
                                    SmallVector<int64_t>& virtBarrierUpdateVec, size_t cost,
                                    SmallVector<size_t>& subTasksCost)
        : taskOp(op),
          virtBarrierWaits(virtBarrierWaitVec),
          virtBarrierUpdates(virtBarrierUpdateVec),
          cycleCost(cost),
          subTasksCycleCost(subTasksCost) {
}

vpux::VPURT::InferenceExecutionSimulator::InferenceExecutionSimulator(Logger log, mlir::func::FuncOp funcOp,
                                                                      CycleCostInfo& cycleCostInfo)
        : _log(log), _funcOp(funcOp), _cycleCostInfo(cycleCostInfo) {
    auto module = funcOp->getParentOfType<mlir::ModuleOp>();

    _supportsDedicatedActShaveQueues = config::isFifoPerShaveEngineEnabled(funcOp);

    if (auto tileOp = config::getTileExecutor(module)) {
        // In case of ActShave tasks on a single cluster compiler does not assign it to a dedicated engine instance
        // (unless dedicated FIFOs for each SHAVE engine are enabled) as it is dispatched only at inference based on
        // engine availability. Nevertheless simulator needs to know this to correctly model those queues and track
        // cycles
        if (auto shaveActExec = tileOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT)) {
            _numOfExecutorQueuesForWhichAssignmentIsAtInference[VPU::ExecutorKind::SHAVE_ACT] =
                    _supportsDedicatedActShaveQueues ? 1 : shaveActExec.getCount();
        }
        _dpuCount = tileOp.getSubExecutor(VPU::ExecutorKind::DPU).getCount();
    }
    // Parse model and gather information about barrier, tasks and their cycle cost
    parseFunc();
}

void vpux::VPURT::InferenceExecutionSimulator::parseFunc() {
    DenseMap<mlir::Operation*, int64_t> barrierOpToVIdMap;
    // Assign virtual IDs to each barrier operation. Whole simulation will
    // use those VIDs when tracking dependencies
    int64_t vid = 0;
    _funcOp->walk([&](mlir::Operation* op) {
        if (mlir::isa<VPURT::DeclareVirtualBarrierOp, VPURT::ConfigureBarrierOp>(op)) {
            VPUX_THROW_WHEN(vid >= std::numeric_limits<uint32_t>::max(), "Barrier virtual id '{0}' is too large ", vid);

            _VIdToBarrierOpMap[vid] = op;
            barrierOpToVIdMap[op] = vid++;
        }
    });

    auto getVirtualBarrierDeps = [&](mlir::ValueRange barriers) {
        SmallVector<int64_t> vIds;
        for (const auto bar : barriers) {
            const auto virtBarIt = barrierOpToVIdMap.find(bar.getDefiningOp());
            VPUX_THROW_WHEN(virtBarIt == barrierOpToVIdMap.end(), "Barrier at '{0}' was not assigned virtual ID",
                            bar.getLoc());

            const auto vid = virtBarIt->second;
            vIds.push_back(vid);
        }
        return vIds;
    };

    // Scan whole model and assign tasks to their queues
    _funcOp->walk([&](VPURT::TaskOp taskOp) {
        auto virtBarrierWaits = getVirtualBarrierDeps(taskOp.getWaitBarriers());
        auto virtBarrierUpdates = getVirtualBarrierDeps(taskOp.getUpdateBarriers());

        size_t cost = 0;
        SmallVector<size_t> subTasksCost;

        // For better visualization, get per variant cost as opposed to cost of whole nceOp
        if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp())) {
            std::vector<size_t> costPerVariantVec;
            auto costModel = _cycleCostInfo.getCostModel();
            for (auto&& dpuTaskOp : nceOp.getVariants().getOps<VPUIP::DPUTaskOp>()) {
                if (auto cycleCostInterface = mlir::dyn_cast<VPUIP::CycleCostInterface>(dpuTaskOp.getOperation())) {
                    auto subTaskCost = cycleCostInterface.getOperationCycleCost(costModel);
                    costPerVariantVec.push_back(subTaskCost);
                }
            }
            cost = VPUNN::dpu_schedule(static_cast<size_t>(_dpuCount), costPerVariantVec);
            _cycleCostInfo.updateAndStoreInvalidCostCycles(cost, nceOp);
            if (std::all_of(costPerVariantVec.begin(), costPerVariantVec.end(), [](size_t cost) {
                    return cost < VPU::INVALID_COST_BASE;
                })) {
                subTasksCost = SmallVector<size_t>(costPerVariantVec.begin(), costPerVariantVec.end());
            }

        } else {
            cost = _cycleCostInfo.getCycleCost(taskOp.getInnerTaskOp());
        }
        TaskConfig taskCfg(taskOp, virtBarrierWaits, virtBarrierUpdates, cost, subTasksCost);

        // If a task produces any barriers update
        // barrier information to determine final value of producer count
        for (auto virtBarrierId : taskCfg.virtBarrierUpdates) {
            _virtBarriers[virtBarrierId].addProducer();
        }

        VPURT::TaskQueueType queueType;
        queueType.type = taskOp.getExecutorKind();
        queueType.id = getQueueId(taskOp);

        // Add a task to a container for a given queue type
        _queueTasksMap[queueType].push_back(taskCfg);
    });
}

// Class for maintaining information about a queue of execution
// To be used by InferenceExecutionSimulator::runSim() function to track
// the progress of execution
class QueueState {
public:
    QueueState(): _taskIdx(0) {
    }

    QueueState(size_t numOfRuntimeDispatchedExecutors): _taskIdx(0) {
        VPUX_THROW_UNLESS(numOfRuntimeDispatchedExecutors > 0,
                          "Number of executors need to be larger then 0, got '{0}'", numOfRuntimeDispatchedExecutors);
        _cycle.resize(numOfRuntimeDispatchedExecutors);
    }

    size_t getCurrentTaskIdx() {
        return _taskIdx;
    }

    size_t getCycle() {
        // Get minimal cycle value. This aligs with HW behavior which will
        // pick a task from a queue to an executor currently being free
        return *std::min_element(_cycle.begin(), _cycle.end());
    }

    void progressQueueToCycle(size_t newCycle) {
        // Once task from a queue gets executed update cycle state of a queue
        // and increment task index
        auto cycleItr = std::min_element(_cycle.begin(), _cycle.end());
        VPUX_THROW_WHEN(newCycle < *cycleItr, "New cycle '{0}' is smaller then exisitng '{1}'", newCycle, *cycleItr);
        *cycleItr = newCycle;
        _taskIdx++;
    }

private:
    // Vector maintaining cycle state of given queue
    // When task is executed cycle is being updated to cycleEnd
    // of this task. This member is a vector because there can be mutliple
    // executors of the same type that are not assigned or distinguishable by compiler
    // and are dispatched at runtime.
    // Example: 2 ActShave engines on 1 cluster on VPU2.7
    SmallVector<size_t> _cycle;

    // Index of next tast to be executed on given queue
    size_t _taskIdx;
};

void vpux::VPURT::InferenceExecutionSimulator::runSim() {
    // Create a map of all encountered queue types and initialize queue state
    // based on information on how many executors there are of exactly the same type from
    // compiler point of view that are dispatched at runtime
    std::map<VPURT::TaskQueueType, QueueState> queueStateMap;
    for (auto& queue : _queueTasksMap) {
        auto numOfCycleQueuesToTrack = 1;
        if (_numOfExecutorQueuesForWhichAssignmentIsAtInference.find(queue.first.type) !=
            _numOfExecutorQueuesForWhichAssignmentIsAtInference.end()) {
            numOfCycleQueuesToTrack = _numOfExecutorQueuesForWhichAssignmentIsAtInference[queue.first.type];
        }
        queueStateMap[queue.first] = QueueState(numOfCycleQueuesToTrack);
    }

    // Run simulation to update cycleBegin/End of each task
    // Iterate over all queues and check next task on queue. If dependencies are satisfied
    // task can be executed and cycle state on this queue needs to be updated. If there are any
    // update barriers their counter will be decremented
    // If no task was executed on all queues that stop the simulation
    bool progressed = true;
    while (progressed) {
        progressed = false;
        for (auto& queueTypeTasks : _queueTasksMap) {
            auto& queueType = queueTypeTasks.first;
            auto& queueTasks = queueTypeTasks.second;

            auto index = queueStateMap[queueType].getCurrentTaskIdx();
            if (index >= queueTasks.size()) {
                continue;
            }

            auto cost = queueTasks[index].cycleCost;

            bool waitingForDependency = false;
            size_t cycleBegin = queueStateMap[queueType].getCycle();

            // Check all wait barriers. If all are satisfied task can be executed
            // CycleBegin value needs to take into account at what cycle last barrier
            // (from cycle point of view) was released
            for (auto waitVirtBarrierId : queueTasks[index].virtBarrierWaits) {
                if (!_virtBarriers[waitVirtBarrierId].isReleased()) {
                    waitingForDependency = true;
                    break;
                }
                cycleBegin = std::max(cycleBegin, _virtBarriers[waitVirtBarrierId].getReleaseCycle());
            }

            if (waitingForDependency) {
                continue;
            }

            // All dependencies satisfied - task ready to execute
            size_t cycleEnd = cycleBegin + cost;

            _log.trace("Run {0}[{1}]: cost: {2} cycleBegin: {3} cycleEnd: {4}",
                       getTaskQueueInfoString(queueType, queueTasks[index].taskOp, _supportsDedicatedActShaveQueues),
                       index, cost, cycleBegin, cycleEnd);

            // Update all update barriers of this task. Decrement their counter
            // and pass information at what cycle this update has happened. This is later needed
            // to understand at what cycle barrier was released
            for (auto updateVirtBarrierId : queueTasks[index].virtBarrierUpdates) {
                VPUX_THROW_WHEN(_virtBarriers[updateVirtBarrierId].isReleased(), "Barrier {0} was already released",
                                updateVirtBarrierId);

                _virtBarriers[updateVirtBarrierId].decrementAtCycle(cycleEnd);

                _log.nest().trace("Decrement virt barrier {0}{1}", updateVirtBarrierId,
                                  _virtBarriers[updateVirtBarrierId].isReleased() ? " - barrier released" : "");
            }

            // Task has executed on this queue. Update queue state with new cycle
            queueStateMap[queueType].progressQueueToCycle(cycleEnd);
            queueTasks[index].cycleStart = cycleBegin;

            if (queueType.type == VPU::ExecutorKind::DPU && !queueTasks[index].subTasksCycleCost.empty()) {
                queueTasks[index].subTasksCycleStart =
                        VPURT::getSubTasksStartTime(queueTasks[index].subTasksCycleCost, cycleBegin, _dpuCount);
            }

            progressed = true;
        }
    }

    // Check if all operations were processed - each queue state index should correspond to
    // the number of tasks in IR that were to be processed by this queue
    // If this is not the case then simulation of execution most likely encountered incorrect
    // dependencies setting which caused a hang
    for (auto& queueTypeTasks : _queueTasksMap) {
        auto& queueType = queueTypeTasks.first;
        auto& queueTasks = queueTypeTasks.second;

        auto index = queueStateMap[queueType].getCurrentTaskIdx();
        VPUX_THROW_WHEN(index != queueTasks.size(),
                        "Not all operations were processed for {0}, index - {1}, queue size - {2}", queueType.type,
                        queueType.id, queueTasks.size());
    }
}

double vpux::VPURT::InferenceExecutionSimulator::getDPUTotalEnergy() {
    double dpuTotalEnergy = 0;
    _funcOp->walk([&](VPUIP::DPUTaskOp dpuTaskOp) {
        auto dpuWorkload = vpux::getDPUWorkload(dpuTaskOp, _cycleCostInfo.getArchKind());
        auto dpuEnergy = _cycleCostInfo.getCostModel()->DPUEnergy(dpuWorkload);
        dpuTotalEnergy += dpuEnergy;
        _log.trace("[Energy] dpuTask - {0}, energy - {1}", dpuTaskOp->getLoc(), dpuEnergy);
    });
    return dpuTotalEnergy;
}

double vpux::VPURT::InferenceExecutionSimulator::getSHAVETotalEnergy() {
    double shaveTotalEnergy = 0;
    std::set<std::string> unsupportedOps;
    _funcOp->walk([&](VPUIP::SwKernelOp swKernelOp) {
        double shaveEnergy = 0;
        auto vpunnSwOp = getVPUNNSWKernelOp(swKernelOp, _cycleCostInfo.getCostModel()->isShave2ApiUsed());
        if (vpunnSwOp != nullptr) {
            shaveEnergy = _cycleCostInfo.getCostModel()->SHAVEEnergy(*vpunnSwOp);
        } else {
            _log.debug("[Energy] an unsupported SwKernel op in VPUNN found - {0}", swKernelOp->getLoc());
            auto kernelName = swKernelOp.getKernelFunction().getLeafReference().str();
            unsupportedOps.insert(std::move(kernelName));
        }
        shaveTotalEnergy += shaveEnergy;
        _log.trace("[Energy] SwKernelOp - {0}, energy - {1}", swKernelOp->getLoc(), shaveEnergy);
    });
    if (!unsupportedOps.empty()) {
        _log.warning("[Energy] The following SHAVE operations are unsupported by VPUNN, which can influence the "
                     "estimation: {0}",
                     unsupportedOps);
    }
    return shaveTotalEnergy;
}

std::map<VPURT::TaskQueueType, VPURT::TaskConfigVec> vpux::VPURT::InferenceExecutionSimulator::getQueueTaskMap() {
    VPUX_THROW_WHEN(_queueTasksMap.empty(), "Queue task map not initialized");

    return _queueTasksMap;
}

SmallVector<VPURT::TaskConfig, 1> vpux::VPURT::InferenceExecutionSimulator::getTaskCycleConfig() {
    VPUX_THROW_WHEN(_queueTasksMap.empty(), "Queue task map not initialized");

    TaskConfigVec allQueueTaskConfig;

    for (auto& queueTypeTasks : _queueTasksMap) {
        auto& queueTasks = queueTypeTasks.second;

        allQueueTaskConfig.insert(allQueueTaskConfig.end(), queueTasks.begin(), queueTasks.end());
    }

    return allQueueTaskConfig;
}

SmallVector<VPURT::TaskConfig, 1> vpux::VPURT::InferenceExecutionSimulator::getTaskCycleConfig(
        VPU::ExecutorKind execKind) {
    VPUX_THROW_WHEN(_queueTasksMap.empty(), "Queue task map not initialized");

    TaskConfigVec allQueueTaskConfig;

    for (auto& queueTypeTasks : _queueTasksMap) {
        if (queueTypeTasks.first.type != execKind) {
            continue;
        }

        auto& queueTasks = queueTypeTasks.second;

        allQueueTaskConfig.insert(allQueueTaskConfig.end(), queueTasks.begin(), queueTasks.end());
    }

    return allQueueTaskConfig;
}

int64_t vpux::VPURT::InferenceExecutionSimulator::getInferenceLatencyInCycles() {
    int64_t latency = 0;
    for (auto& queueTypeTasks : _queueTasksMap) {
        auto& queueTasks = queueTypeTasks.second;

        auto lastTaskCycleEnd = queueTasks.back().cycleStart + queueTasks.back().cycleCost;
        if (lastTaskCycleEnd > latency) {
            latency = lastTaskCycleEnd;
        }
    }
    return latency;
}

void vpux::VPURT::InferenceExecutionSimulator::updateCyclesInIR() {
    for (auto& queueTypeTasks : _queueTasksMap) {
        auto& queueTasks = queueTypeTasks.second;
        for (auto& task : queueTasks) {
            task.taskOp->setAttr(cycleBegin, getIntAttr(task.taskOp->getContext(), task.cycleStart));
            task.taskOp->setAttr(cycleEnd, getIntAttr(task.taskOp->getContext(), task.cycleStart + task.cycleCost));
        }
    }
}

VPURT::BarrierConfigInfo vpux::VPURT::InferenceExecutionSimulator::getVirtBarrierConfig(
        uint32_t virtualBarrierId) const {
    return _virtBarriers.at(virtualBarrierId);
}

mlir::Operation* vpux::VPURT::InferenceExecutionSimulator::getDeclareBarrierOp(uint32_t virtualBarrierId) {
    return _VIdToBarrierOpMap[virtualBarrierId];
}
