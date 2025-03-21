//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPURT/interfaces/enqueue_barrier.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

using namespace vpux;

vpux::VPURT::EnqueueBarrierHandler::EnqueueBarrierHandler(mlir::func::FuncOp func, BarrierInfo& barrierInfo, Logger log)
        : _barrierInfo(barrierInfo),
          _barrierDepthCheck(BARRIER_FIFO_SIZE > 1 ? BARRIER_FIFO_SIZE - 1 : 1),
          _dmaFifoDepth(DMA_OUTSTANDING_TRANSACTIONS),
          _optimizeAndMergeEnqFlag(true),
          _log(log) {
    _taskQueueTypeMap = VPURT::getTaskOpQueues(func, _barrierInfo);
    initPrevPhysBarrierData(func);
    _startBarrierIndex = getStartBarrierIndex(func);
    _barrierInfo.buildTaskQueueTypeMap();
}

vpux::VPURT::EnqueueBarrierHandler::EnqueueBarrierHandler(
        BarrierInfoTest& barrierInfoTest, std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& taskQueueTypeMap,
        SmallVector<size_t>& barrierToPidVec, size_t barrierFifoDepth, size_t dmaFifoDepth,
        bool optimizeAndMergeEnqFlag, Logger log)
        : _barrierInfo(barrierInfoTest),
          _taskQueueTypeMap(taskQueueTypeMap),
          _barrierDepthCheck(barrierFifoDepth > 1 ? barrierFifoDepth - 1 : 1),
          _dmaFifoDepth(dmaFifoDepth),
          _optimizeAndMergeEnqFlag(optimizeAndMergeEnqFlag),
          _log(log) {
    size_t highestPid = 0;
    for (auto pid : barrierToPidVec) {
        highestPid = std::max(highestPid, pid);
    }

    initPrevPhysBarrierData(barrierToPidVec, highestPid + 1);
    // For test scenario use first barrier as start barrier
    _startBarrierIndex = 0;
    // Task queue type is provided as part of barrierInfoTest
}

// For each barrier find index of previous barrier using same PID
void vpux::VPURT::EnqueueBarrierHandler::initPrevPhysBarrierData(SmallVector<size_t>& barrierToPidVec,
                                                                 size_t nPhysBars) {
    size_t numOfBarriers = _barrierInfo.getNumOfBarrierOps();

    VPUX_THROW_UNLESS(numOfBarriers == barrierToPidVec.size(), "Not matching number of barriers {0} != {1}",
                      numOfBarriers, barrierToPidVec.size());

    _barrierPidPrevUsageVec.resize(numOfBarriers);
    SmallVector<std::optional<size_t>> lastBarUsingPid(nPhysBars);

    for (size_t vid = 0; vid < numOfBarriers; vid++) {
        auto pid = barrierToPidVec[vid];
        _barrierPidPrevUsageVec[vid] = lastBarUsingPid[pid];
        lastBarUsingPid[pid] = vid;
    }
}

std::optional<size_t> vpux::VPURT::EnqueueBarrierHandler::getStartBarrierIndex(mlir::func::FuncOp func) {
    std::optional<size_t> startBarrierIndex;
    func.walk([&](VPURT::ConfigureBarrierOp barOp) {
        if (barOp.getIsStartBarrier()) {
            startBarrierIndex = _barrierInfo.getIndex(barOp);
            return mlir::WalkResult::interrupt();
        }

        return mlir::WalkResult::advance();
    });
    return startBarrierIndex;
}

// For each barrier find index of previous barrier using same PID
void vpux::VPURT::EnqueueBarrierHandler::initPrevPhysBarrierData(mlir::func::FuncOp func) {
    // Get information for each barrier about previous barrier that used same PID
    size_t nPhysBars = VPUIP::getNumAvailableBarriers(func);
    size_t numOfBarriers = _barrierInfo.getNumOfBarrierOps();

    SmallVector<size_t> barrierToPidVec(numOfBarriers);

    func.walk([&](VPURT::ConfigureBarrierOp barOp) {
        size_t pid = barOp.getId();
        auto vid = _barrierInfo.getIndex(barOp);
        barrierToPidVec[vid] = pid;
    });

    initPrevPhysBarrierData(barrierToPidVec, nPhysBars);
}

// For a given task get initial proposal of enqueue barrier
// Use previous barrier using same PID as wait barrier
std::optional<size_t> vpux::VPURT::EnqueueBarrierHandler::getInitialEnqueueBarrier(size_t taskInd) {
    auto waitBarriers = _barrierInfo.getWaitBarriers(taskInd);
    VPUX_THROW_WHEN(waitBarriers.size() > 1, "Task index {0} has more than 1 wait barrier", taskInd);

    if (waitBarriers.empty()) {
        return std::nullopt;
    }

    auto vid = *waitBarriers.begin();

    return _barrierPidPrevUsageVec[vid];
}

// Check if barA is guaranteed to be consumed before barB, that means
// that all consumers of barA based on schedule will run before consumers of barB
bool vpux::VPURT::EnqueueBarrierHandler::isBarrierAConsumedBeforeBarrierB(size_t barA, size_t barB) {
    auto barAConsumers = _barrierInfo.getBarrierConsumers(barA);
    auto barBConsumers = _barrierInfo.getBarrierConsumers(barB);

    // Check if there is a path from each barA consumer to any of BarB producers
    // and if each barB consumer is dependant on any of barA consumers
    // If this is satisfied barB is guaranteed to have consumption event after barA
    //        |-> barAConsumer0 ->  ..             | -> barBConsumer0
    //  barA->|-> barAConsumer1 ->  ..      barB ->| -> barBConsumer1
    //                                             | -> barBConsumer2

    // Sore information about barB consumers which where not covered
    // by first loop check
    auto notCoveredBarBConsumers = barBConsumers;

    auto isPathFromTask1ToTask2 = [&](size_t task1, size_t task1Block, size_t task2, size_t task2Block) {
        if (task1Block < task2Block) {
            // If tasks are from different blocks then they are guaranteed to have a path
            // through a sync task
            return true;
        }

        auto& [taskControlMap, taskControlMapOffset] =
                _taskControlMapCache.getTaskControlMapAndOffset(_barrierInfo, task1Block);
        return _barrierInfo.controlPathExistsBetweenTasksInSameBlock(taskControlMap, task1 - taskControlMapOffset,
                                                                     task2 - taskControlMapOffset, false);
    };

    // First check paths from all barA consumers to any of barB consumers
    for (auto& barAConsumer : barAConsumers) {
        bool isPath = false;
        auto barAConsumerBlock = _barrierInfo.getControlGraphBlockIndex(barAConsumer);
        for (auto& barBConsumer : barBConsumers) {
            auto barBConsumerBlock = _barrierInfo.getControlGraphBlockIndex(barBConsumer);
            if (barAConsumerBlock > barBConsumerBlock) {
                // If barA consumer is on later block than barB then for sure
                // barA is not consuemd before barB
                return false;
            }

            isPath = isPathFromTask1ToTask2(barAConsumer, barAConsumerBlock, barBConsumer, barBConsumerBlock);
            if (isPath) {
                notCoveredBarBConsumers.erase(barBConsumer);
                break;
            }
        }
        if (!isPath) {
            return false;
        }
    }

    // Check remaining not covered barB consumers as previous loop guarantees to verify dependency
    // only from each barA consumer
    for (auto& barBConsumer : notCoveredBarBConsumers) {
        bool isPath = false;
        auto barBConsumerBlock = _barrierInfo.getControlGraphBlockIndex(barBConsumer);
        for (auto& barAConsumer : barAConsumers) {
            auto barAConsumerBlock = _barrierInfo.getControlGraphBlockIndex(barAConsumer);
            if (barAConsumerBlock > barBConsumerBlock) {
                // If barA consumer is on later block than barB then for sure
                // barA is not consuemd before barB
                return false;
            }
            isPath = isPathFromTask1ToTask2(barAConsumer, barAConsumerBlock, barBConsumer, barBConsumerBlock);
            if (isPath) {
                break;
            }
        }
        if (!isPath) {
            return false;
        }
    }

    return true;
}

// Check if tasks in set A are all guaranteed to run before tasks in set B
bool vpux::VPURT::EnqueueBarrierHandler::areTasksABeforeTasksB(const BarrierInfo::TaskSet& tasksA,
                                                               const BarrierInfo::TaskSet& tasksB) {
    // Currently this function supports scenario when either tasksA or tasksB sets contain only 1 element
    // as only such scenario is what is currently needed
    VPUX_THROW_UNLESS(tasksA.size() == 1 || tasksB.size() == 1,
                      "At least one tasks sets need to have size of 1, taskA size {0}, taskB size {1}", tasksA.size(),
                      tasksB.size());

    for (const auto taskA : tasksA) {
        auto taskABlock = _barrierInfo.getControlGraphBlockIndex(taskA);
        for (const auto taskB : tasksB) {
            auto taskBBlock = _barrierInfo.getControlGraphBlockIndex(taskB);
            if (taskABlock > taskBBlock) {
                // If taskA is on later block than taskB then for sure
                // taskA is after
                return false;
            }

            if (taskABlock < taskBBlock) {
                continue;
            }

            auto& [taskControlMap, taskControlMapOffset] =
                    _taskControlMapCache.getTaskControlMapAndOffset(_barrierInfo, taskABlock);
            if (!_barrierInfo.controlPathExistsBetweenTasksInSameBlock(taskControlMap, taskA - taskControlMapOffset,
                                                                       taskB - taskControlMapOffset, false)) {
                return false;
            }
        }
    }

    return true;
}

// Check if barrier is consumed before given task
bool vpux::VPURT::EnqueueBarrierHandler::isBarrierConsumedBeforeTask(size_t bar, size_t taskInd) {
    auto barConsumers = _barrierInfo.getBarrierConsumers(bar);

    BarrierInfo::TaskSet taskIndSet;
    taskIndSet.insert(taskInd);
    return areTasksABeforeTasksB(barConsumers, taskIndSet);
}

// Check if barrier consumption depends on task, what means that barrier will not be
// fully consumed until task runs before
bool vpux::VPURT::EnqueueBarrierHandler::isBarrierConsumptionDependantOnTask(size_t bar, size_t taskInd) {
    auto barConsumers = _barrierInfo.getBarrierConsumers(bar);

    auto taskBlock = _barrierInfo.getControlGraphBlockIndex(taskInd);
    for (const auto barConsTask : barConsumers) {
        if (barConsTask == taskInd) {
            return true;
        }
        auto barConsTaskBlock = _barrierInfo.getControlGraphBlockIndex(barConsTask);
        if (taskBlock > barConsTaskBlock) {
            // If task is on later block than barrier consumer then for sure
            // task is after
            continue;
        }

        if (taskBlock < barConsTaskBlock) {
            return true;
        }

        auto& [taskControlMap, taskControlMapOffset] =
                _taskControlMapCache.getTaskControlMapAndOffset(_barrierInfo, taskBlock);
        if (_barrierInfo.controlPathExistsBetweenTasksInSameBlock(taskControlMap, taskInd - taskControlMapOffset,
                                                                  barConsTask - taskControlMapOffset, false)) {
            return true;
        }
    }

    return false;
}

// If some previously enqueued op is enqueued at later barrier adjust the enqueue target
// to be equal to that of that previous enqueue if possible
void vpux::VPURT::EnqueueBarrierHandler::delayEnqIfNeededBasedOnPrevEnq(std::optional<size_t>& enqBarOpt,
                                                                        std::optional<size_t> previousEnqBarOpt) {
    auto prevEnqBar = previousEnqBarOpt.value();
    auto enqBar = enqBarOpt.value();
    if (enqBar != prevEnqBar && !isBarrierAConsumedBeforeBarrierB(prevEnqBar, enqBar)) {
        // prevEnqBar is not guaranted to be consumed before enqBar.
        // Check if there is enqBar -> prevEnqBar dependency
        if (isBarrierAConsumedBeforeBarrierB(enqBar, prevEnqBar)) {
            // there is enqBar -> prevEnqBar. It is safe to enqueue at prevEnqBar
            // because at prevEnqBar required barrier reprograming restriction will be
            // satisfied as for enqBar
            _log.nest().trace("Delay enqueue from {0} to {1}", enqBar, prevEnqBar);
            enqBarOpt = previousEnqBarOpt;
            return;
        }
        // Not hitting this case will require physical barrier assignment to
        // be update with an additional restriction of prev DMA wait barrier
        // to be dependant on DMA prev instance of wait barrier if it is before in
        // schedule
        // On the other hand runtime nevertheless would handle this correctly
        // as when task enqueue barrier would be consumed earlier then previous
        // tasks gets enqueued it would just mark the work item ready and enqueue
        // would happen together with previous one. This way there is no deadlock
        _log.nest().trace("Could not reliably delay task enqueue");
    }

    return;
}

// If enqueue of task can safely happen earlier at previous ops enqueue then
// update enqueue barrier. Tasks using same barrier for enqueue will have single
// WorkItem task to be processed by runtime
void vpux::VPURT::EnqueueBarrierHandler::optimizeEnqueueIfPossible(std::optional<size_t>& enqBarOpt,
                                                                   BarrierInfo::TaskSet& waitBarriers,
                                                                   BarrierInfo::TaskSet& updateBarriers,
                                                                   std::optional<size_t> previousTaskIndOpt,
                                                                   std::optional<size_t> previousEnqBarOpt) {
    auto enqBar = enqBarOpt.value();
    auto prevTaskInd = previousTaskIndOpt.value();
    auto prevEnqBar = previousEnqBarOpt.value();

    // First check if previous task runs after previous instance of wait barrier for currently
    // enqueued task is consumed. This way we know that when prev task runs, task wait barrier
    // will get remapped to instance expected by task itself and it will not start prematurely
    //
    //             |-> .... ->|
    // waitBarPrev |-> .... ->|-> prevTaskInd -> ..... waitBar -> taskInd
    //             |-> .... ->|
    //
    if (!waitBarriers.empty()) {
        auto waitBar = *waitBarriers.begin();
        auto waitBarPrev = getNthPrevBarInstance(waitBar, 1);
        if (waitBarPrev.has_value()) {
            if (!isBarrierConsumedBeforeTask(waitBarPrev.value(), prevTaskInd)) {
                return;
            }
        }
    }

    // Check if at barrier FIFO depth prev instances of wait and update barriers are guaranteed
    // to be reprogrammed before previous task enqueue barrier
    //
    // waitBarPrevN    -> .... |
    // updateBar0PrevN -> .... |-> prevTaskEnqBar
    // updateBar1PrevN -> .... |

    auto taskBarriers = updateBarriers;
    taskBarriers.insert(waitBarriers.begin(), waitBarriers.end());

    SmallVector<size_t> prevVids;

    // For update barriers get N-th prev bar instance, where N is depth of check which correspond to ready barrier
    // configs stored in barrier FIFO
    for (auto& vid : taskBarriers) {
        auto prevVidOpt = getNthPrevBarInstance(vid, _barrierDepthCheck);
        if (prevVidOpt.has_value()) {
            prevVids.push_back(prevVidOpt.value());
        }
    }

    // Perform dependency checks of task barriers at depth N and previous task enqueue barrier
    bool optimizationCanBePerformed = true;
    for (auto prevVid : prevVids) {
        if (prevVid == prevEnqBar) {
            continue;
        }
        optimizationCanBePerformed = isBarrierAConsumedBeforeBarrierB(prevVid, prevEnqBar);
        if (!optimizationCanBePerformed) {
            break;
        }
    }

    if (optimizationCanBePerformed) {
        _log.nest().trace("Enqueue can be optimized and merged with previous, orig enqueue: {0}, new enqueue: {1}",
                          enqBar, prevEnqBar);
        enqBarOpt = previousEnqBarOpt;
    }

    return;
}

// If some previously enqueued op is enqueued at later barrier adjust the enqueue target
// to be equal to that of previous enqueue
mlir::LogicalResult vpux::VPURT::EnqueueBarrierHandler::delayEnqIfNeededBasedOnFifoState(
        std::optional<size_t>& enqBarOpt, std::vector<std::optional<size_t>>& outstandingEnqueuesTaskIndexVec,
        std::vector<std::optional<size_t>>& outstandingEnqueuesTaskWaitBarIndexVec, size_t outstandingEnquOpsCounter) {
    auto oldestTaskIndexOpt = outstandingEnqueuesTaskIndexVec[outstandingEnquOpsCounter];
    if (!oldestTaskIndexOpt.has_value()) {
        return mlir::success();
    }

    // Make sure enqueue target will be later then the moment of latest DMA in FIFO
    // has started to guarantee FIFO is never larger than its limit.
    // If original enqueue target is earlier delay it to wait barrier of related DMA

    auto oldestTaskIndex = oldestTaskIndexOpt.value();
    auto enqBar = enqBarOpt.value();
    auto enqBarConsumers = _barrierInfo.getBarrierConsumers(enqBar);

    // Check if oldest task is guaranteed to happen before consumption of enqueue barrier
    //                    |->..............|-> enqBarUser0
    // oldetsTaskIndex -> ...........enqBar|-> enqBarUser1
    //                         |->.........|-> enqBarUser2
    //
    // In that case no need to delay
    BarrierInfo::TaskSet oldestTaskIndexSet;
    oldestTaskIndexSet.insert(oldestTaskIndex);
    if (areTasksABeforeTasksB(oldestTaskIndexSet, enqBarConsumers)) {
        _log.nest().trace("No need to delay enqueue due to FIFO limit as oldest task ({0}) will complete in time",
                          oldestTaskIndex);
        return mlir::success();
    }

    // Check whole FIFO starting from the oldest till newest one
    // and see if this given task is guaranteed to execeute when
    // all users of enque barrier have completed, meaning enqueue barrier
    // consumption event happens. If there is such dependency
    // delay enqueue to that of wait barrier of this task
    //
    //         |-> enqBarUser0 ..->|
    // enqBar->|-> enqBarUser1 .......| -> bar -> somePrevTaskInFifo
    //         |-> enqBarUser2 .....->|
    //
    auto fifoSize = outstandingEnqueuesTaskIndexVec.size();
    for (size_t fifoEntryToCheck = 0; fifoEntryToCheck < fifoSize; fifoEntryToCheck++) {
        auto outstandingEnqueuesTaskIndexVecEntryIndex = (outstandingEnquOpsCounter + fifoEntryToCheck) % fifoSize;
        auto taskIndexInFifo = outstandingEnqueuesTaskIndexVec[outstandingEnqueuesTaskIndexVecEntryIndex];

        VPUX_THROW_UNLESS(taskIndexInFifo.has_value(),
                          "Expected to have valid task in outstanding enqueue FIFO at index {0}",
                          outstandingEnqueuesTaskIndexVecEntryIndex);

        BarrierInfo::TaskSet taskIndexInFifoSet;
        taskIndexInFifoSet.insert(taskIndexInFifo.value());
        if (areTasksABeforeTasksB(enqBarConsumers, taskIndexInFifoSet)) {
            VPUX_THROW_UNLESS(
                    outstandingEnqueuesTaskWaitBarIndexVec[outstandingEnqueuesTaskIndexVecEntryIndex].has_value(),
                    "Task {0} used to delay enqueue has no wait barrier", taskIndexInFifo.value());
            _log.nest().trace(
                    "Delay enqueue due to FIFO limit, orig enqueue: {0}, new enqueue: {1}", enqBar,
                    outstandingEnqueuesTaskWaitBarIndexVec[outstandingEnqueuesTaskIndexVecEntryIndex].value());
            enqBarOpt = outstandingEnqueuesTaskWaitBarIndexVec[outstandingEnqueuesTaskIndexVecEntryIndex];
            return mlir::success();
        }
    }

    // Could not reliably set and delay enqueue barrier due to fifo size of task
    // because of lack of dependencies between enqueue barrier and wait barriers of
    // tasks already in FIFO. Delaying in such case might not guarantee that when task
    // is enqueued its barrier is ready (reprogrammed)
    return mlir::failure();
}

std::optional<size_t> vpux::VPURT::EnqueueBarrierHandler::getNthPrevBarInstance(size_t vid, size_t n) {
    std::optional<size_t> prevVid = vid;
    for (size_t i = 0; i < n; i++) {
        if (!prevVid.has_value()) {
            return std::nullopt;
        }

        prevVid = _barrierPidPrevUsageVec[prevVid.value()];
    }

    return prevVid;
}

mlir::LogicalResult vpux::VPURT::EnqueueBarrierHandler::findInitialEnqWithLcaForGivenBarriers(
        std::optional<size_t>& enqBarOpt, BarrierInfo::TaskSet& waitBarriers, BarrierInfo::TaskSet& updateBarriers) {
    SmallVector<size_t> prevVids;

    // Enqueue barrier can be searched within a range of largest barrier index from prev
    // bar instances and smallest index of task barrier.
    // Note. Barrier indexes where assigned following barrier consumption order
    // TODO: To be analyzed and rechecked if uppear limit could be defined
    // on same larger barrier, for example based on next barrier with same PID as theoretically
    // task could be enqueued on barrier with larer index then task barriers if they
    // are on parallel branches. This is less common but possible scenario.
    size_t enqVidRangeMin = std::numeric_limits<size_t>::min();
    size_t enqVidRangeMax = std::numeric_limits<size_t>::max();

    // For wait barriers get prev bar instance
    for (auto& vid : waitBarriers) {
        auto prevVidOpt = getNthPrevBarInstance(vid, 1);
        if (prevVidOpt.has_value()) {
            prevVids.push_back(prevVidOpt.value());
            enqVidRangeMin = std::max(enqVidRangeMin, prevVidOpt.value());
        }

        enqVidRangeMax = std::min(enqVidRangeMax, vid);
    }

    // For update barriers get N-th prev bar instance, where N is depth of check which correspond to ready barrier
    // configs stored in barrier FIFO
    for (auto& vid : updateBarriers) {
        auto prevVidOpt = getNthPrevBarInstance(vid, _barrierDepthCheck);
        if (prevVidOpt.has_value()) {
            prevVids.push_back(prevVidOpt.value());
            enqVidRangeMin = std::max(enqVidRangeMin, prevVidOpt.value());
        }

        enqVidRangeMax = std::min(enqVidRangeMax, vid);
    }

    if (prevVids.empty()) {
        enqBarOpt = std::nullopt;
        return mlir::success();
    }

    if (prevVids.size() == 1) {
        enqBarOpt = prevVids[0];
        return mlir::success();
    }

    // First check if the latest barrier from prev vid range is not already dependant on other barriers in prevVids
    // In such case it is an LCA candidate
    bool isEnqVidRangeMinValidCandidate = true;
    for (auto& prevVid : prevVids) {
        if (prevVid == enqVidRangeMin) {
            continue;
        }
        isEnqVidRangeMinValidCandidate = isBarrierAConsumedBeforeBarrierB(prevVid, enqVidRangeMin);
        if (!isEnqVidRangeMinValidCandidate) {
            break;
        }
    }

    if (isEnqVidRangeMinValidCandidate) {
        enqBarOpt = enqVidRangeMin;
        return mlir::success();
    }

    // Search for an LCA of barriers prevVids vector within range [enqVidRangeMin, enqVidRangeMax]
    // Below code checks each barrier within a given range because underlying structure allows for
    // a quick check on barrier dependency.
    // TODO: Analyze possible improvement to go through chain of barrier dependencies if existing
    // approach turns out to be a compile time bottleneck
    for (size_t vid = enqVidRangeMin + 1; vid < enqVidRangeMax; vid++) {
        bool isVidValidCandidate = true;
        for (auto& prevVid : prevVids) {
            isVidValidCandidate = isBarrierAConsumedBeforeBarrierB(prevVid, vid);
            if (!isVidValidCandidate) {
                break;
            }
        }
        if (isVidValidCandidate) {
            enqBarOpt = vid;
            return mlir::success();
        }
    }

    return mlir::failure();
}

mlir::LogicalResult vpux::VPURT::EnqueueBarrierHandler::calculateEnqueueBarriers() {
    _tasksEnqBar.resize(_barrierInfo.getNumOfTasks());

    // Processed queue types
    llvm::DenseSet<VPURT::TaskQueueType> processedQueues;

    for (auto& [queueType, taskVec] : _taskQueueTypeMap) {
        _log.trace("Enqueue tasks for {0}:{1}", VPU::stringifyExecutorKind(queueType.type), queueType.id);
        std::optional<size_t> previousTaskWaitBarOpt;
        std::optional<size_t> previousEnqBarOpt;
        std::optional<size_t> previousTaskIndOpt;

        size_t outstandingEnqueueLimit = 0;
        std::vector<std::optional<size_t>> outstandingEnqueuesTaskWaitBarIndexVec;
        std::vector<std::optional<size_t>> outstandingEnqueuesTaskIndexVec;

        bool supportEnqAtBootstrap = false;

        // Opimizing consecutive enqueues allows to reduce number of WorkItem tasks
        // what has impact on performance as runtime has less overhead processing them
        bool optimizeAndMergeEnq = _optimizeAndMergeEnqFlag;

        // Configure based on difference in handling different engines
        if (queueType.type == VPU::ExecutorKind::DMA_NN) {
            outstandingEnqueueLimit = _dmaFifoDepth;
            outstandingEnqueuesTaskWaitBarIndexVec.resize(outstandingEnqueueLimit);
            outstandingEnqueuesTaskIndexVec.resize(outstandingEnqueueLimit);

            // Only DMA tasks can be enqueued at bootstrap as they do not require
            // descriptor fetching
            supportEnqAtBootstrap = true;
        } else if (queueType.type == VPU::ExecutorKind::SHAVE_ACT) {
            // This optimization is not supported for SHV tasks as it does not yet support linked lists
            // Also there are 2 SHV engines in single cluster and compiler has no explicit control
            // over which engine is going to be used as there is single FIFO. In that case
            // algorithm cannot assume that ShaveTaskN blocks execution of ShaveTaskN+1
            // Optimization enabling should be revisitied when E#111941 is done
            optimizeAndMergeEnq = false;
        }

        size_t outstandingEnquOpsCounter = 0;

        for (auto taskInd : taskVec) {
            _log.trace("Find enqueue for task {0}", taskInd);
            _log = _log.nest();
            auto waitBarriers = _barrierInfo.getWaitBarriers(taskInd);
            VPUX_THROW_WHEN(waitBarriers.size() > 1, "Task index {0} has more than 1 wait barrier", taskInd);

            auto updateBarriers = _barrierInfo.getUpdateBarriers(taskInd);

            auto taskBarriers = updateBarriers;
            taskBarriers.insert(waitBarriers.begin(), waitBarriers.end());

            std::optional<size_t> enqBarOpt;
            size_t waitBarVid;

            // Initial enqueue proposal is previous wait barrier usage
            // or null in case task doesn't have any barriers
            if (taskBarriers.empty()) {
                enqBarOpt = std::nullopt;
            } else {
                if (!waitBarriers.empty()) {
                    waitBarVid = *waitBarriers.begin();
                    _log.trace("Task wait barrier: {0}", waitBarVid);
                }
                if (mlir::failed(findInitialEnqWithLcaForGivenBarriers(enqBarOpt, waitBarriers, updateBarriers))) {
                    _log.trace("Failed to find enqueue barrier using LCA for task {0}", taskInd);
                    return mlir::failure();
                }
            }

            if (!enqBarOpt.has_value() && !supportEnqAtBootstrap) {
                // earliest barrier consumption event is at start barrier
                if (!_startBarrierIndex.has_value()) {
                    _log.trace("Failed to set enqueue barrier for task {0} because of lack of start barrier", taskInd);
                    return mlir::failure();
                }
                enqBarOpt = _startBarrierIndex.value();
            }

            _log.trace("Initial enqueue proposal: {0}",
                       (enqBarOpt.has_value() ? std::to_string(enqBarOpt.value()) : "BOOTSTRAP"));

            // If current enqueue has no value, meaning enqueue at BOOTSTRAP and previous task
            // was enqueued at some barrier then change enqueue to that barrier as execution of this
            // task is nevertheless blocked and in that case they can be linked in single WorkItem task
            if (previousEnqBarOpt.has_value() && !enqBarOpt.has_value()) {
                _log.nest().trace("Delay enqueue from BOOTSTRAP to that of previous task - {0}",
                                  previousEnqBarOpt.value());
                enqBarOpt = previousEnqBarOpt;
            }

            if (previousEnqBarOpt.has_value() && enqBarOpt.has_value()) {
                // Check if enqueue needs to be delayed because of previous enqueue happening later
                if (optimizeAndMergeEnq) {
                    delayEnqIfNeededBasedOnPrevEnq(enqBarOpt, previousEnqBarOpt);
                }

                // If enqueue barrier is different than previous then perform checks if
                // consecutive enqueue optimization can be applied
                if (optimizeAndMergeEnq && enqBarOpt.value() != previousEnqBarOpt.value()) {
                    optimizeEnqueueIfPossible(enqBarOpt, waitBarriers, updateBarriers, previousTaskIndOpt,
                                              previousEnqBarOpt);
                }
            }

            // Check if there is any outstanding enqueue limit. If yes then in case
            // there is any enqueue barrier check if it is different then prviously enqueued task
            // If yes then that would be a different enqueue operation and would occupy FIFO
            // This needs to be tracked and enqueued tasks might require to have enqueue delayed
            // to not overflow the FIFO
            if (outstandingEnqueueLimit > 0 && enqBarOpt.has_value()) {
                if (!previousEnqBarOpt.has_value() || enqBarOpt.value() != previousEnqBarOpt.value()) {
                    const auto res = delayEnqIfNeededBasedOnFifoState(enqBarOpt, outstandingEnqueuesTaskIndexVec,
                                                                      outstandingEnqueuesTaskWaitBarIndexVec,
                                                                      outstandingEnquOpsCounter);
                    if (mlir::failed(res)) {
                        _log.warning("Could not reliably set and delay enqueue barrier due to fifo size of task: {0}, "
                                     "enqBar: {1}",
                                     taskInd, enqBarOpt.value());
                        return mlir::failure();
                    }

                    // If task has no wait barrier use previous task wait barrier as effectively this task
                    // due to FIFO dep would wait also on that barrier. previousTaskWaitBarOpt will contain
                    // last tasks wait barrier on this FIFO. It doesn't have to be last predecessor.
                    if (waitBarriers.empty()) {
                        outstandingEnqueuesTaskWaitBarIndexVec[outstandingEnquOpsCounter] = previousTaskWaitBarOpt;
                    } else {
                        outstandingEnqueuesTaskWaitBarIndexVec[outstandingEnquOpsCounter] = *waitBarriers.begin();
                    }
                    outstandingEnqueuesTaskIndexVec[outstandingEnquOpsCounter] = taskInd;
                    outstandingEnquOpsCounter = (outstandingEnquOpsCounter + 1) % outstandingEnqueueLimit;
                }
            }

            _tasksEnqBar[taskInd] = enqBarOpt;
            _log.trace("Enqueue task at {0}",
                       (enqBarOpt.has_value() ? std::to_string(enqBarOpt.value()) : "BOOTSTRAP"));

            // Check if enqueue barrier depends on the task to be enqueued. If yes then this is a deadlock
            if (enqBarOpt.has_value() && isBarrierConsumptionDependantOnTask(enqBarOpt.value(), taskInd)) {
                _log.warning("Enqueue barrier {0} depends on the task to be enqueued {1}", enqBarOpt.value(), taskInd);
                return mlir::failure();
            }

            previousEnqBarOpt = enqBarOpt;
            previousTaskIndOpt = taskInd;

            if (!waitBarriers.empty()) {
                previousTaskWaitBarOpt = waitBarVid;
            }

            _log = _log.unnest();
        }
        processedQueues.insert(queueType);
    }

    return mlir::success();
}

// Get barrier index that given task should be enqueued at. No value means enqueue at bootstrap
std::optional<size_t> vpux::VPURT::EnqueueBarrierHandler::getEnqueueBarrier(size_t taskInd) {
    VPUX_THROW_UNLESS(taskInd < _tasksEnqBar.size(), "Task has unexpected index {0}", taskInd);

    return _tasksEnqBar[taskInd];
}

// Get barrier that given taskOp should be enqueued at. Null value means enqueue at bootstrap
mlir::Value vpux::VPURT::EnqueueBarrierHandler::getEnqueueBarrier(VPURT::TaskOp taskOp) {
    auto taskInd = _barrierInfo.getIndex(taskOp);

    VPUX_THROW_UNLESS(taskInd < _tasksEnqBar.size(), "Task has unexpected index {0} - {1}", taskInd, taskOp);

    auto enqBarIndOpt = _tasksEnqBar[taskInd];

    if (!enqBarIndOpt.has_value()) {
        return nullptr;
    }

    return _barrierInfo.getBarrierOpAtIndex(enqBarIndOpt.value()).getBarrier();
}
