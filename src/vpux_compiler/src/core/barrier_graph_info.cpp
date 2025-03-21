//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_graph_info.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {
// The Hop stands for the amount of barriers visited from one task to next task with same TaskQueueType. When there are
// parallel paths between two tasks, hop stands for the longest path
struct Hop {
    size_t value;
    VPURT::TaskQueueType type;
    bool operator<(const Hop& another) const {
        return std::make_pair(value, type.type) < std::make_pair(another.value, another.type.type);
    }
};

BarrierGraphInfo::BarrierSet convertBitVectorToSet(const llvm::BitVector& vector) {
    BarrierGraphInfo::BarrierSet result;
    for (auto elem : vector.set_bits()) {
        result.insert(elem);
    }
    return result;
}
}  // namespace

BarrierGraphInfo::BarrierGraphInfo()
        : _log(Logger::global().nest("barrier-graph-info", 0)),
          _func(nullptr),
          _barrierInfo(nullptr)

{
}

BarrierGraphInfo::BarrierGraphInfo(mlir::func::FuncOp func)
        : _log(Logger::global().nest("barrier-graph-info", 0)),
          _func(func),
          _barrierInfo(std::make_shared<BarrierInfo>(func)) {
    // Categorize tasks into task queue list
    _taskQueueTypeMap = VPURT::getTaskOpQueues(_func, *_barrierInfo);
    buildTaskFifo();
    // Categorize tasks and barriers into different execution steps.
    calculateTaskAndBarrierExecutionSteps();
    // Build dependency between each barrier
    buildBarrierDependenceMap();
    // Calculate the longest hop queue type for each barrier
    calculateBarrierLongestHopQueueType();
    // Correct the longest hop queue type for the final barrier
    correctFinalBarrierLongestHopQueueType();
}

void BarrierGraphInfo::buildTaskFifo() {
    size_t taskCount = std::accumulate(
            _taskQueueTypeMap.begin(), _taskQueueTypeMap.end(), size_t(0),
            [](size_t count, std::pair<const VPURT::TaskQueueType, SmallVector<uint32_t>> taskQueueTypeVecPair) {
                return count + taskQueueTypeVecPair.second.size();
            });

    _nextTaskInSameQueue.resize(taskCount);
    _prevTaskInSameQueue.resize(taskCount);

    for (auto& [_, tasksVec] : _taskQueueTypeMap) {
        std::optional<size_t> prevTask = std::nullopt;
        for (auto taskInd : tasksVec) {
            _prevTaskInSameQueue[taskInd] = prevTask;
            if (prevTask.has_value()) {
                _nextTaskInSameQueue[prevTask.value()] = taskInd;
            }
            prevTask = taskInd;
        }
    }
}

BarrierGraphInfo::BarrierSet BarrierGraphInfo::getParentBarrier(size_t barrierInd) {
    return convertBitVectorToSet(_barrierParentData[barrierInd]);
}

BarrierGraphInfo::BarrierSet BarrierGraphInfo::getChildrenBarrier(size_t barrierInd) {
    return convertBitVectorToSet(_barrierChildrenData[barrierInd]);
}

std::optional<size_t> BarrierGraphInfo::getNextTaskInFifo(size_t taskInd) const {
    VPUX_THROW_UNLESS(taskInd < _nextTaskInSameQueue.size(), "Can not find next task for {0}", taskInd);
    // TODO: Optimize once E#119383 is implemented
    return _nextTaskInSameQueue[taskInd];
}

std::optional<size_t> BarrierGraphInfo::getPreTaskInFifo(size_t taskInd) const {
    VPUX_THROW_UNLESS(taskInd < _prevTaskInSameQueue.size(), "Can not find prev task for {0}", taskInd);
    // TODO: Optimize once E#119383 is implemented
    return _prevTaskInSameQueue[taskInd];
}

VPURT::TaskQueueType BarrierGraphInfo::getTaskQueueType(size_t taskInd) const {
    auto iter = llvm::find_if(_taskQueueTypeMap, [&](const auto& item) {
        return llvm::find(item.second, taskInd) != item.second.end();
    });
    VPUX_THROW_WHEN(iter == _taskQueueTypeMap.end(), "Can not find task {0} from the mapping", taskInd);
    return iter->first;
}

size_t BarrierGraphInfo::getTaskExecutionStep(size_t taskInd) const {
    VPUX_THROW_UNLESS(taskInd < _taskExecutionStep.size(), "Can not find execution step for task {0}", taskInd);
    return _taskExecutionStep[taskInd];
}

std::map<size_t, SmallVector<size_t>> BarrierGraphInfo::getExecutionStepTaskBatch() const {
    return _executeStepTaskBatch;
}

SmallVector<VPURT::TaskQueueType> BarrierGraphInfo::getBarrierLongestQueueType() const {
    return _barrierLongestHopQueueType;
}

SmallVector<size_t> BarrierGraphInfo::getBarrierFirstExecutionStep() const {
    return _barrierFirstExecutionStep;
}
SmallVector<size_t> BarrierGraphInfo::getBarrierLastExecutionStep() const {
    return _barrierLastExecutionStep;
}

void BarrierGraphInfo::calculateBarrierLongestHopQueueType() {
    _barrierLongestHopQueueType.resize(_barrierInfo->getNumOfBarrierOps());

    // find the last op that has no wait barriers but only implicit dependence on the consumer op in the same task queue
    // or the consumer op itself if there is no implicit op in the same queue
    auto findLastImplicitConsumerOp = [&](const size_t& consumerOp) {
        auto lastImplicitConsumerOp = consumerOp;
        auto nextOp = getNextTaskInFifo(consumerOp);
        // If the next op with same task queue has no wait barriers, it means the task will execute once the previous op
        // are finished. In that case, we think the next op has the same execution step
        while (nextOp.has_value() && _barrierInfo->getWaitBarriers(nextOp.value()).empty()) {
            lastImplicitConsumerOp = nextOp.value();
            nextOp = getNextTaskInFifo(nextOp.value());
        }
        return lastImplicitConsumerOp;
    };

    const VPURT::TaskQueueType defaultHopQueueType{VPU::ExecutorKind::DMA_NN,
                                                   getDMAQueueIdEncoding(VPUIP::DmaChannelType::DDR)};

    for (auto barrierInd : irange(_barrierInfo->getNumOfBarrierOps())) {
        // Get the task type with the longest barrier hop

        // The first operation along the chain of ops on given queue from consumer ops of
        // this barrier.
        std::map<VPURT::TaskQueueType, size_t> firstConsumerWithTaskQueueType;
        // The last operation along the chain of ops on given queue without a wait barrier after given consumer op of
        // this barrier.
        std::map<VPURT::TaskQueueType, size_t> lastConsumerWithTaskQueueType;
        auto consumerOps = _barrierInfo->getBarrierConsumers(barrierInd);
        for (const auto& taskInd : consumerOps) {
            auto taskQueueType = getTaskQueueType(taskInd);
            if (firstConsumerWithTaskQueueType.find(taskQueueType) == firstConsumerWithTaskQueueType.end()) {
                firstConsumerWithTaskQueueType[taskQueueType] = taskInd;
            } else if (firstConsumerWithTaskQueueType[taskQueueType] > taskInd) {
                firstConsumerWithTaskQueueType[taskQueueType] = taskInd;
            }
            // last consumer need to take implicit dependence op into consideration
            auto lastConsumer = findLastImplicitConsumerOp(taskInd);
            if (lastConsumerWithTaskQueueType.find(taskQueueType) == lastConsumerWithTaskQueueType.end()) {
                lastConsumerWithTaskQueueType[taskQueueType] = lastConsumer;
            } else if (lastConsumer > lastConsumerWithTaskQueueType[taskQueueType]) {
                lastConsumerWithTaskQueueType[taskQueueType] = lastConsumer;
            }
        }

        std::optional<Hop> longestHop;
        for (const auto& [taskQueueType, consumer] : lastConsumerWithTaskQueueType) {
            auto currentExecutionStep = getTaskExecutionStep(consumer);
            // Calculate the hop from current task to next task
            auto nextTask = getNextTaskInFifo(consumer);
            if (nextTask.has_value()) {
                auto nextTaskExecutionStep = getTaskExecutionStep(nextTask.value());
                VPUX_THROW_WHEN(
                        nextTaskExecutionStep < currentExecutionStep,
                        "Next task {0} execution step is supposed to greater than current task {1} execution step",
                        nextTask.value(), consumer);
                Hop currentHop{nextTaskExecutionStep - currentExecutionStep, taskQueueType};
                if (!longestHop.has_value() || longestHop.value() < currentHop) {
                    longestHop = currentHop;
                }
            }
        }

        for (const auto& [taskQueueType, consumer] : firstConsumerWithTaskQueueType) {
            auto currentExecutionStep = getTaskExecutionStep(consumer);
            // Calculate the hop from pre task to current task
            auto preTask = getPreTaskInFifo(consumer);
            // If consumer doesn't have previous task, while means it's the first task in the FIFO
            Hop currentHop{std::numeric_limits<size_t>::max(), taskQueueType};
            if (preTask.has_value()) {
                auto preTaskExecutionStep = getTaskExecutionStep(preTask.value());
                VPUX_THROW_WHEN(
                        preTaskExecutionStep > currentExecutionStep,
                        "Pre task {0} execution step is supposed to smaller than current task {1} execution step",
                        preTask.value(), consumer);
                currentHop.value = currentExecutionStep - preTaskExecutionStep;
            }

            if (!longestHop.has_value() || longestHop.value() < currentHop) {
                longestHop = currentHop;
            }
        }
        if (longestHop.has_value()) {
            _barrierLongestHopQueueType[barrierInd] = longestHop.value().type;
        } else {
            // For barrier without consumer ops, assign default hop type
            _barrierLongestHopQueueType[barrierInd] = defaultHopQueueType;
        }
    }
}

/*
Update the longest hop queue type for the final barrier.
If no other barriers has same hop queue type as the final barrier, it means when doing color bin barrier assignment,
the final barrier will own a physical barrier alone, which may cause runtime failures. So we need to change the
type for the final barrier in that case.
*/
void BarrierGraphInfo::correctFinalBarrierLongestHopQueueType() {
    std::optional<size_t> finalBarrier = std::nullopt;
    for (auto barrierInd : irange(_barrierInfo->getNumOfBarrierOps())) {
        auto barrierOp = _barrierInfo->getBarrierOpAtIndex(barrierInd);
        if (barrierOp.getIsFinalBarrier()) {
            VPUX_THROW_WHEN(finalBarrier.has_value(), "More than one final barriers are found");
            finalBarrier = barrierInd;
        }
    }
    if (!finalBarrier.has_value()) {
        return;
    }
    VPUX_THROW_UNLESS(finalBarrier.value() < _barrierLongestHopQueueType.size(),
                      "BarrierQueueType size {0} is smaller then final barrier index {1}",
                      _barrierLongestHopQueueType.size(), finalBarrier.value());
    auto taskType = _barrierLongestHopQueueType[finalBarrier.value()];
    auto count = std::count_if(_barrierLongestHopQueueType.begin(), _barrierLongestHopQueueType.end(),
                               [&taskType](const auto& item) {
                                   return item == taskType;
                               });
    if (count == 1) {
        // When final barrier is the only one with task type, it may result in runtime failure after applying color
        // bin barrier assignment. So need to move final barrier to another DMA type
        auto finalBarrierTaskType = _barrierLongestHopQueueType[finalBarrier.value()];
        auto iter = std::find_if(_barrierLongestHopQueueType.begin(), _barrierLongestHopQueueType.end(),
                                 [&finalBarrierTaskType](const auto& item) {
                                     return item.type == VPU::ExecutorKind::DMA_NN && item != finalBarrierTaskType;
                                 });
        if (iter != _barrierLongestHopQueueType.end()) {
            _barrierLongestHopQueueType[finalBarrier.value()] = *iter;
        }
    }
}

BarrierInfo& BarrierGraphInfo::getBarrierInfo() {
    return *_barrierInfo;
}

void BarrierGraphInfo::clearAttributes() {
    auto taskIndexAttr = mlir::StringAttr::get(_func.getContext(), "task-index");
    _func.walk([&](VPURT::TaskOp taskOp) {
        taskOp->removeAttr(taskIndexAttr);
    });
}

// Categorize tasks into different execution steps. For tasks in same execution step, they are supposed to be
// executed after same barrier set. Perform schedule simulation with step tracking and update also
// barrier first (when it is used for the first time) and last execution step (when it is fully consumed)
void BarrierGraphInfo::calculateTaskAndBarrierExecutionSteps() {
    _barrierFirstExecutionStep.resize(_barrierInfo->getNumOfBarrierOps(), std::numeric_limits<size_t>::max());
    _barrierLastExecutionStep.resize(_barrierInfo->getNumOfBarrierOps(), 0);

    size_t taskCount = std::accumulate(
            _taskQueueTypeMap.begin(), _taskQueueTypeMap.end(), size_t(0),
            [](size_t count, std::pair<const VPURT::TaskQueueType, SmallVector<uint32_t>> taskQueueTypeVecPair) {
                return count + taskQueueTypeVecPair.second.size();
            });

    _taskExecutionStep.resize(taskCount);

    // Need to copy barrierInfo because the function will run barrier schedule simulation and modifies barrier
    // information along the way
    auto tmpBarrierInfo = *_barrierInfo;
    auto frontTasks = VPURT::initializeTaskOpQueueIterators(_taskQueueTypeMap);

    auto updateBarriers = [&](size_t readyOp, size_t executionStep) {
        const auto updateBarriers = tmpBarrierInfo.getUpdateBarriers(readyOp);
        for (const auto& updateBarInd : updateBarriers) {
            // For each update barrier if it is produced for the first time set
            // its first execute step
            _barrierFirstExecutionStep[updateBarInd] =
                    std::min(_barrierFirstExecutionStep[updateBarInd], executionStep);

            tmpBarrierInfo.removeProducer(updateBarInd, readyOp);
            if (tmpBarrierInfo.getBarrierProducers(updateBarInd).empty()) {
                tmpBarrierInfo.resetBarrier(updateBarInd);
            }
        }

        // For executed task update its wait barrier last step if it is bigger
        // then exisiting one
        const auto waitBarriers = _barrierInfo->getWaitBarriers(readyOp);
        for (const auto& waitBarInd : waitBarriers) {
            _barrierLastExecutionStep[waitBarInd] = std::max(_barrierLastExecutionStep[waitBarInd], executionStep);
        }
    };

    size_t executionStep = 0;
    while (!VPURT::allQueuesReachedEnd(frontTasks, _taskQueueTypeMap)) {
        auto readyOps = VPURT::findReadyOpsFromTaskOpQueues(frontTasks, _taskQueueTypeMap, tmpBarrierInfo);
        for (auto taskInd : readyOps) {
            _executeStepTaskBatch[executionStep].push_back(taskInd);
            _taskExecutionStep[taskInd] = executionStep;
            updateBarriers(taskInd, executionStep);
        }
        executionStep++;
    }
}

void BarrierGraphInfo::buildBarrierDependenceMap() {
    // init _barrierParentData and _barrierChildrenData
    _barrierParentData.resize(_barrierInfo->getNumOfBarrierOps());
    _barrierChildrenData.resize(_barrierInfo->getNumOfBarrierOps());

    const auto totalBarrierNum = _barrierInfo->getNumOfBarrierOps();
    for (auto barInd : irange(totalBarrierNum)) {
        _barrierParentData[barInd] = llvm::BitVector(totalBarrierNum);
        _barrierChildrenData[barInd] = llvm::BitVector(totalBarrierNum);
    }

    // Since tasks might not be guarded by barriers and their execution is controlled by FIFO order
    // it is needed to track last task on given queue with barrier to make proper parent/child
    // connection for subsequent task which has barriers.
    // For example:
    // Bar0 -> Op1 -> Op2 -> Bar1
    // Bar1 parent barrier is Bar0 even though Bar0 is not directly controlling Op2 task
    DenseMap<VPURT::TaskQueueType, size_t> taskQueueLastOpWithWaitBarMap;

    for (auto taskInd : irange(_barrierInfo->getNumOfTasks())) {
        const auto taskQueueType = getTaskQueueType(taskInd);

        // For now do this only for DMA as other tasks are nevertheless always guarded by barriers.
        // TODO: This would need to be extended once following tasks are resolved: E#124900 and E#126579
        if (taskQueueType.type == VPU::ExecutorKind::DMA_NN) {
            if (!_barrierInfo->getWaitBarriers(taskInd).empty()) {
                taskQueueLastOpWithWaitBarMap[taskQueueType] = taskInd;
            }
        }

        if (_barrierInfo->getUpdateBarriers(taskInd).empty()) {
            continue;
        }

        auto waitBars = _barrierInfo->getWaitBarriers(taskInd);
        if (taskQueueType.type == VPU::ExecutorKind::DMA_NN &&
            taskQueueLastOpWithWaitBarMap.find(taskQueueType) != taskQueueLastOpWithWaitBarMap.end()) {
            auto prevTaskWithBarsOnFifo = taskQueueLastOpWithWaitBarMap[taskQueueType];
            waitBars.insert(_barrierInfo->getWaitBarriers(prevTaskWithBarsOnFifo).begin(),
                            _barrierInfo->getWaitBarriers(prevTaskWithBarsOnFifo).end());
        }

        for (auto waitBar : waitBars) {
            for (auto updateBar : _barrierInfo->getUpdateBarriers(taskInd)) {
                _barrierParentData[updateBar].set(waitBar);
                _barrierChildrenData[waitBar].set(updateBar);
            }
        }
    }
}

BarrierGraphInfoTest::BarrierGraphInfoTest(std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& taskQueueMaps,
                                           BarrierInfoMaps& barrierMaps) {
    _barrierInfo = std::make_shared<BarrierInfoTest>(barrierMaps);
    _taskQueueTypeMap = taskQueueMaps;
    buildTaskFifo();
    calculateTaskAndBarrierExecutionSteps();
    buildBarrierDependenceMap();
    calculateBarrierLongestHopQueueType();
}
