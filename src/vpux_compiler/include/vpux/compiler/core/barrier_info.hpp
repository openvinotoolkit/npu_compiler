//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/SmallSet.h>

namespace vpux {

// Declare BarrierInfo utils used for testing purposes
// BarrierInfoMaps is intended to be used where code (e.g. test) wants to retrieve
// low level barrier and task data structures.
using BarrierMap = SmallVector<SmallVector<size_t>>;
struct BarrierInfoMaps {
    BarrierMap barrierProducerMap = {};
    BarrierMap barrierConsumerMap = {};
    BarrierMap taskUpdateBarriers = {};
    BarrierMap taskWaitBarriers = {};
    SmallVector<size_t> syncTasksIds = {};
    std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> taskQueueTypeMap = {};
};

class BarrierInfo;
BarrierInfoMaps getBarrierMaps(BarrierInfo& barrierInfo);

// BarrierInfo class is used to store information about barriers and tasks in the control graph
// with useful methods to modify, legalize and optimize dependencies
class BarrierInfo {
public:
    // TaskSet is used to store barrier's producer/consumer task op index as well as task op's
    // wait/update barrier index, which is supposed to have better performance than BitVector when the data size is
    // small.
    using TaskSet = llvm::SmallSet<size_t, 16>;
    explicit BarrierInfo();
    explicit BarrierInfo(mlir::func::FuncOp func);
    friend class BarrierInfoTest;
    virtual ~BarrierInfo() = default;

private:
    friend BarrierInfoMaps getBarrierMaps(BarrierInfo& barrierInfo);

public:
    void updateIR();
    void clearAttributes();
    TaskSet& getWaitBarriers(size_t taskInd);
    TaskSet& getUpdateBarriers(size_t taskInd);
    uint32_t getIndex(VPURT::TaskOp taskOp) const;
    uint32_t getIndex(VPURT::BarrierOpInterface barrierOp) const;
    virtual VPURT::TaskOp getTaskOpAtIndex(size_t opIdx) const;
    VPURT::TaskQueueType getTaskQueueType(size_t taskInd) const;
    std::optional<size_t> getPrevTaskOnFifo(size_t taskInd) const;
    // Return the closest previous task on the same queue that has a wait barrier
    std::optional<size_t> getPrevTaskOnFifoWithWaitBar(size_t taskInd) const;
    VPURT::BarrierOpInterface getBarrierOpAtIndex(size_t opIdx) const;
    void enableUnevenVariantSplit();

private:
    void addTaskOp(VPURT::TaskOp taskOp);
    void buildBarrierMaps(mlir::func::FuncOp func);
    /**
     * @brief Creates LUT storing control graph block index for every task
     */
    void buildTaskBlockMap();
    void setWaitBarriers(size_t taskIdn, const TaskSet& barriers);
    void setUpdateBarriers(size_t taskIdn, const TaskSet& barriers);
    void resizeBitMap(SmallVector<llvm::BitVector>& bitMap, size_t length, uint32_t bits);
    void resetBitMap(SmallVector<llvm::BitVector>& bitMap);
    bool producersControlsAllConsumers(const TaskSet& origProducers, const TaskSet& newConsumers,
                                       const TaskSet& origConsumers, ArrayRef<TaskSet> origWaitBarriersMap);
    bool inImplicitQueueTypeDependencyList(const TaskSet& taskList);

    void optimizeBarrierProducers(size_t blockIdx);
    void optimizeBarrierConsumers(size_t blockIdx);
    void optimizeBarriersWithSameProducers(size_t blockIdx, bool checkValidSlotCount = true);

    bool inRange(const size_t low, const size_t high, const size_t val) const;
    void setBarrierMask(llvm::BitVector& mask, const BarrierInfo::TaskSet& barriers, size_t offset = 0);
    void splitBarrierProducers(VPURT::DeclareVirtualBarrierOp barrierOp, size_t availableSlots);
    void splitBarrierConsumers(VPURT::DeclareVirtualBarrierOp barrierOp, size_t availableSlots);
    SmallVector<BarrierInfo::TaskSet> createProducerBatches(const BarrierInfo::TaskSet& waitBarriers,
                                                            size_t availableSlots, bool considerTaskExecutorType);
    void linearizeLegalParallelProducers(size_t taskInd, const BarrierInfo::TaskSet& parallelProducers,
                                         const BarrierInfo::TaskSet& parallelConsumers, size_t availableSlots,
                                         bool considerTaskExecutorType);

    /**
     * @brief check if total slot count in provided set of tasks is smaller or equal to availableSlots
     *
     * @param producers - set of tasks which slots should be checked
     * @param availableSlots - number of slots available for provided tasks (producers)
     * @return true when total slots count in producers <= availableSlots
     * @return false otherwise
     *
     */
    bool canMergeBarriersForTasks(const BarrierInfo::TaskSet& producers, size_t availableSlots);
    bool eliminateParallelWaitBarriers(size_t taskInd, size_t availableSlots, bool considerTaskExecutorType);
    void mergeLegalParallelProducers(size_t taskInd, const BarrierInfo::TaskSet& parallelProducers,
                                     const BarrierInfo::TaskSet& parallelConsumers);
    void linkTasksToBarriers(const TaskSet& tasksToAdd, const TaskSet& newBarriers, bool waitBarriers,
                             size_t availableSlots);

    /**
     * @brief Get index of barrier's latest producer
     *
     * @param barInd - barrier index
     * @return size_t - largest index of among barrier producers
     */
    size_t getBarrierLatestProducer(size_t barInd);

public:
    void logBarrierInfo();
    void optimizeBarriers(bool checkValidSlotCount = true, bool considerTaskFifoDependency = false);

    /**
     * @brief Eliminate tasks not controlled by barriers
     *
     * Eliminate tasks not controlled by barriers, by sharing wait / update barriers of parent / child DMA to create a
     * schedule fully managed by barriers  which simplifies runtime handling 1) update barriers: find task(s) without
     * update barrier, find next task (on the same FIFO) with update barrier(s) link update barrier(s) to all tasks that
     * don't have update barrier 2) wait barriers: find task(s) without wait barrier, find previous task (on the same
     * FIFO) with wait barrier(s) link wait barrier(s) to all tasks that don't have wait barrier
     *
     *    Bar0
     *    |             Bar0
     *    DMA-0            |
     *    |     =>  DMA-0 DMA-1
     *    DMA-1            |
     *    |             Bar1
     *    Bar1
     *
     */
    void shareWaitAndUpdateBarriers(size_t availableSlots);

    /**
     * @brief initialize task queue type map structure for the queues that should be used when building the map.
     *
     * @param executorKind - set of executors for which the map should be initialized
     */
    void initializeTaskQueueTypeMap(const mlir::DenseSet<vpux::VPU::ExecutorKind>& executorKind);

    /**
     * @brief build task queue type map for the initialized executor kinds.
     *
     * If the map was previously not initialized, by default the map will be initialized for DMA executor.
     * If the map is to be built for a different executor than previously initialized, clearTaskQueueTypeMap()
     * should be called first.
     */
    void buildTaskQueueTypeMap();

    /**
     * @brief Remove all entries from task queue task map
     */
    void clearTaskQueueTypeMap();

    /**
     * @brief build task control map for given task block
     *
     * @param blockIdx block index
     * @param considerTaskFifoDependency
     * @param ignoreOutOfBlockDependencies - when calculating task control map ignore out of block dependencies.
     * @return std::pair<SmallVector<llvm::BitVector>, size_t> representing
     * taskControlMap - a 2-d array suitable for use with controlPathExistsBetweenTasksInSameBlock()
     * and
     * task index offset that should be subtracted from task indexes when testing if path exists between tasks.
     * eg. controlPathExistsBetweenTasksInSameBlock(taskControlMap, taskA - offset, taskB - offset);
     *
     */
    std::pair<SmallVector<llvm::BitVector>, size_t> buildTaskControlMap(size_t blockIdx,
                                                                        bool considerTaskFifoDependency = true,
                                                                        bool ignoreOutOfBlockDependencies = false);
    virtual size_t getNumOfTasks() const;
    size_t getNumOfBarrierOps() const;
    virtual size_t getBarrierMaxVariantSum() const;
    static size_t getNumOfSlotsUsed(VPURT::TaskOp op);
    virtual size_t getNumOfSlotsUsedByTask(VPURT::TaskOp op) const;
    void resetBarrier(VPURT::BarrierOpInterface barrierOp);
    void resetBarrier(size_t barrierInd);
    size_t addNewBarrier(VPURT::BarrierOpInterface barrierOp);
    size_t addNewTaskOp(VPURT::TaskOp taskOp);
    bool controlPathExistsBetweenTasksInSameBlock(const SmallVector<llvm::BitVector>& taskControlMap, size_t taskAInd,
                                                  size_t taskBInd, bool biDirection = true) const;
    size_t getProducerSlotCount(VPURT::BarrierOpInterface barrierOp);
    size_t getConsumerSlotCount(VPURT::BarrierOpInterface barrierOp);
    void addProducer(size_t barrierInd, size_t taskInd);
    void addProducer(VPURT::BarrierOpInterface barrierOp, size_t taskInd);
    void addProducers(size_t barrierInd, const TaskSet& taskInds);
    void addConsumer(size_t barrierInd, size_t taskInd);
    void addConsumer(VPURT::BarrierOpInterface barrierOp, size_t taskInd);
    void addConsumers(size_t barrierInd, const TaskSet& taskInds);
    void removeProducer(size_t barrierInd, size_t taskInd);
    void removeConsumer(size_t barrierInd, size_t taskInd);

    void removeProducer(VPURT::BarrierOpInterface barrierOp, size_t taskInd);
    void removeConsumer(VPURT::BarrierOpInterface barrierOp, size_t taskInd);
    void removeProducers(size_t barrierInd, const TaskSet& taskInds);
    void removeConsumers(size_t barrierInd, const TaskSet& taskInds);
    void removeProducers(VPURT::BarrierOpInterface barrierOp, const TaskSet& taskInds);
    void removeConsumers(VPURT::BarrierOpInterface barrierOp, const TaskSet& taskInds);
    TaskSet& getBarrierProducers(VPURT::BarrierOpInterface barrierOp);
    TaskSet& getBarrierConsumers(VPURT::BarrierOpInterface barrierOp);
    TaskSet& getBarrierProducers(size_t barrierIdn);
    TaskSet& getBarrierConsumers(size_t barrierIdn);
    TaskSet getBarriersUsers(const std::set<int64_t>& barrierInds);
    SmallVector<TaskSet> createLegalVariantBatches(const TaskSet& tasks, size_t availableSlots,
                                                   bool considerTaskExecutorType = false);
    std::optional<VPURT::TaskQueueType> haveSameImplicitDependencyTaskQueueType(const TaskSet& taskInds);
    bool canBarriersBeMerged(const TaskSet& barrierProducersA, const TaskSet& barrierConsumersA,
                             const TaskSet& barrierProducersB, const TaskSet& barrierConsumersB,
                             ArrayRef<TaskSet> origWaitBarriersMap);
    SmallVector<TaskSet> getWaitBarriersMap();
    void dumpBarrierDependencies(const SmallVector<BarrierInfo::TaskSet>& depsMap, std::string fileName);
    void dumpQueues(const std::map<VPURT::TaskQueueType, llvm::BitVector>& queueMap, std::string fileName);
    void dumpSlots(std::string fileName);
    void dumpBarriers(std::string fileNamePrefix);

    void splitControlGraphToBlocks(const size_t blockSize);
    bool verifyControlGraphSplit();

    /**
     * @brief Verify barriers required for task descriptor fetch
     *
     * @param wlmEnabled if true: verify schedule for WLM. If false, verify schedule for non-WLM.
     * @return Returns `true` when required barriers exist. Returns `false` otherwise.
     */
    bool verifyBarriersForTaskDescriptorFetch(const ExecutionGroupListMap& executionGroups, bool wlmEnabled = true);

    /**
     * @brief Adjust dependencies for the provided tasks if they are connected to other tasks in a way that violates
     * constrains of existing control graph split.
     * @param tasks - list of tasks whose connections via update and wait barriers be checked and corrected, if needed.
     * For the below examples the graph would be corrected if task 7 is provided in the argument.
     */
    bool adjustTasksDependenciesToGraphSplitConstraints(const TaskSet& producers);

    /**
     * @brief Get index of sync-task from the block preceding the block of the provided taskInd
     *
     * @param taskInd - task index
     * @return std::optional<size_t> preceding block sync-task index if exists. Tasks from 0'th block
     * do not have sync-point.
     */
    std::optional<size_t> getPreviousBlockSyncPoint(size_t taskInd) const;
    void splitBarriersWithExceedingVariantCount(size_t availableSlots, size_t maxSlotsSum, size_t maxAvailableSlots);
    void splitBarrierProducers(size_t availableSlots, size_t maxSlotsSum, bool maxSlotsSumLimitEnabled);
    void splitBarrierConsumers(size_t availableSlots, size_t maxSlotsSum, bool maxSlotsSumLimitEnabled);
    bool ensureTasksDrivenBySingleBarrier(size_t availableSlots, bool mergeWaitBarriersIteratively = false,
                                          bool considerTaskExecutorType = false);
    void removeSyncTaskAttributes();
    bool hasBarrierDependency(size_t taskOneIdx, size_t taskTwoIdx, size_t& commonBarrier);
    bool isSyncPoint(size_t taskIdx);

    /**
     * @brief Get control graph block count
     * The number of control graph blocks is by 1 larger than the number of sync points.
     * Eg. if control graph split is not done (no sync points present) it retuns 1.
     *
     * @return Number of control graph blocks
     */
    size_t getControlGraphBlockCount() const;

    /**
     * @brief Get task indexes range for given control graph tasks block.
     *
     * If present, the returned tasks range include sync-points on both ends. For first and last block
     * the range includes first and last task indexes for lower and upper bound respectively.
     *
     * @param blockInd control graph block index
     * @param blockStartSyncPoint if false the lower bound sync-point is not included.
     * @param blockEndSyncPoint if false the upper bound sync-point is not included.
     * @return Lower and upper bound range for given task block
     */
    std::pair<size_t, size_t> getControlGraphBlockTaskRange(size_t blockInd, bool blockStartSyncPoint = true,
                                                            bool blockEndSyncPoint = true) const;

    /**
     * @brief For each of the provided tasks, find previous tasks that can execute in parallel
     *
     * @param tasks - set of task indexes for which parallel task candidates are to be sought
     * @param maxCount - maximal number of parallel task candidates for each provided task
     * @return SmallVector<std::set<size_t>> vector of size of the provided tasks set, containing
     * indexes of tasks such that there's no topological connection (either via barrier or FIFO)
     * between the given task and the found task.
     * For given task index (idx) from provided tasks set, the search is limited to:
     * (i) the task block in which a given task (idx) resides in,
     * (ii) tasks that have any barrier dependence (i.e either wait or update any barrier)
     */
    SmallVector<std::set<size_t>> findParallelTasksWithBarrierDependence(const BarrierInfo::TaskSet& tasks,
                                                                         size_t maxCount = 1);

    /**
     * @brief Get vector of barriers associated with given control graph tasks block.
     *
     * If present, the returned tasks range include sync-points on both ends. For first and last block
     * the range includes first and last task indexes for lower and upper bound respectively.
     *
     * @param blockInd control graph block index
     * @param blockStartSyncPoint if false the lower bound sync-point is not included.
     * @param blockEndSyncPoint if false the upper bound sync-point is not included.
     * @param updateBarriers - if true return update barriers, otherwise return wait barriers for given blockInd
     * @return Barriers associated with given task block
     */
    SmallVector<size_t> getBarriersForTaskBlock(size_t blockInd, bool blockStartSyncPoint = false,
                                                bool blockEndSyncPoint = false, bool updateBarriers = true) const;
    /**
     * @brief Get control graph block index for given task index
     *
     * @param taskInd - task index
     * @return block index
     */
    size_t getControlGraphBlockIndex(size_t taskInd) const;

    /**
     * @brief Block index of barrier's last producer.
     *
     * If the barrier is sync point update barrier (i.e. its last producer is sync point) then the barrier is considered
     * to belong to the block following the block of the sync point.
     *
     * @param barInd barrier index
     * @return return the block index of barrier's latest producer.
     */
    size_t getBarrierBlockIndex(size_t barInd);

    /**
     * @brief Get index of sync-task for the block to which the task taskInd belongs to.
     *
     * @param taskInd - task for which sync-task should be calculated
     * @return  return index of the sync point from the taskInd's block, if the sync-task exists and return std::nullopt
     * otherwise. (Tasks from last block do not have a sync-point).
     *
     */
    std::optional<size_t> getControlGraphSyncPoint(size_t taskInd) const;

    /**
     * @brief Create barrier representation of dependencies implied FIFOs execution order.
     * The newly created dependencies are stored internally and can be removed by calling
     * @param blockIdx - task block index for which the dependencies should be generated
     * @param executorKind - set of FIFO executors that should be taken into account. By default all FIFOs are
     * considered.
     *
     * @see removeBarrierDependenciesImpliedByFIFO()
     *
     * @return Number of newly created connections between tasks (producers and consumers) and barriers
     */
    unsigned createBarrierDependenciesImpliedByFIFO(
            size_t blockIdx, std::optional<mlir::DenseSet<vpux::VPU::ExecutorKind>> executorKind = std::nullopt);

    /**
     * @brief Create barrier dependencies between task execution groups
     *
     * @param blockIdx - task block index for which the dependencies should be generated
     * @param executorKind - set of FIFO executors that should be taken into account. By default all FIFOs are
     * @param execGroups - groups of tasks used to split schedule by barriers
     *
     * @return Number of newly created barriers
     */
    unsigned createBarrierDependenciesBetweenExecutionGroups(size_t blockIdx, vpux::VPU::ExecutorKind executorKind,
                                                             ExecutionGroupListMap& executionGroups);

    /**
     * @brief Remove barrier representation of dependencies implied FIFOs execution order created by
     * @see createBarrierDependenciesImpliedByFIFO(size_t blockIdx)
     *
     * @return Number of removed connections between tasks (producers and consumers) and barriers
     */
    unsigned removeBarrierDependenciesImpliedByFIFO();

private:
    /**
     * @brief check if task group has update barrier required for fetching task descriptors from the subsequent
     * execution groups.
     *
     *                          |->BAR->
     *                          |
     *           |  1 --> 2 --> 3   |  4 --> 5 --> 6   |  7 --> 8 --> 9   |
     *           |------------------|------------------|------------------|
     *           |                  |                  |                  |
     *           | CurrentGroup     |   ChildGroup     | GrandChildGroup  |
     *
     * @param grpIdx - index of task group to check (CurrentGroup)
     * @param blockIdx - control graph split block index
     * @param fifoExecGroups - list of execution groups for given FIFO
     * @param taskControlMap - task control map calculated for blockIdx (should account for all FIFO dependencies)
     * @param controlMapOffset - task control map offset
     * @param wlmEnabled - WLM flag
     * @return for wlmEnabled=true case return true, if GrandChildGroup does not exist or if GrandChildGroup exists and
     * the last task from task execution group grpIdx has update barrier.
     * @return for wlmEnabled=false case return true, if GrandChildGroup does not exist or if GrandChildGroup exists and
     * the last task from task execution group grpIdx has update barrier and consumption of this barrier does not depend
     * on any task from GrandChildGroup or later.
     * @return false, otherwise
     */
    bool hasBarrierDependencyRequiredForDescriptorFetch(int grpIdx, size_t blockIdx,
                                                        const ExecutionGroupList& fifoExecGroups,
                                                        SmallVector<llvm::BitVector>& taskControlMap,
                                                        size_t controlMapOffset, bool wlmEnabled);

    Logger _log;
    mlir::func::FuncOp _func;

    mlir::StringAttr _taskIndexAttrName;
    mlir::StringAttr _barrierIndexAttrName;
    mlir::StringAttr _syncTaskAttrName;

    SmallVector<VPURT::TaskOp> _allTaskOps;
    SmallVector<VPURT::BarrierOpInterface> _allBarrierOps;

    bool _enableUnevenVariantSplit{false};

    // Control graph split variables
    // After graph split is performed in splitControlGraphToBlocks method
    // the variables store information about the resulting graph blocks
    //
    // Example:  number of all task: 1200, split size: 500
    // Graph is split into 3 blocks with tasks indexes
    // being placed in blocks in following way
    //  block 0:    0 -  499
    //  block 1:  500 -  999
    //  block 2: 1000 - 1199
    // If control graph is split some methods can use it to reduce overhead of processing barriers (smaller memory)
    // Ids of tasks that are synchronization points between blocks. From example above that would be 499 and 999
    // If control graph split is not done, this vector should be empty.
    SmallVector<size_t> _syncTasksIds;
    // _taskToBlockMap stores indexes of tasks blocks for every task in the graph. This is initialized based on
    // _syncTasksIds. The map is built in order to improve performance.
    SmallVector<size_t> _taskToBlockMap;
    // Note:
    //  - task produces its update barriers
    //  - task consumes its wait barriers

    // indexOf(VPURT::BarrierOpInterface) 'is produced by' [ indexOf(VPURT::TaskOp)... ].
    SmallVector<TaskSet> _barrierProducerMap;
    // indexOf(VPURT::BarrierOpInterface) 'is consumed by' [ indexOf(VPURT::TaskOp)... ].
    SmallVector<TaskSet> _barrierConsumerMap;

    // indexOf(VPURT::TaskOp) 'waits for' [ indexOf(VPURT::BarrierOpInterface)... ].
    SmallVector<TaskSet> _taskWaitBarriers;
    // indexOf(VPURT::TaskOp) 'updates' [ indexOf(VPURT::BarrierOpInterface)... ].
    SmallVector<TaskSet> _taskUpdateBarriers;

    // If optimization is to be done taking into account FIFO dependencies, these dependencies are temporarily stored.
    // The tuple contains barrier index and parent and child task indexes.
    SmallVector<std::tuple<size_t, size_t, size_t>> _fifoDependencies;

    // Initialize below structure with buildTaskQueueTypeMap()
    // indexOf(VPURT::TaskQueueType) 'contains' [ indexOf(VPURT::TaskOp)... ].
    std::map<VPURT::TaskQueueType, llvm::BitVector> _taskQueueTypeMap;
};

// BarrierInfoTest is a test class that inherits from BarrierInfo and provides additional methods and
// overrides to test low level functioning of BarrierInfo class
class BarrierInfoTest : public BarrierInfo {
public:
    explicit BarrierInfoTest(mlir::func::FuncOp func);
    explicit BarrierInfoTest(BarrierInfoMaps& barrierMaps);
    void initializeBarrierMaps(BarrierInfoMaps& barrierMaps);
    void setTaskQueueTypeMap(const std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& taskQueueMaps);
    void setMaxVariantCountPerBarrier(size_t variantCount);
    size_t getNumOfTasks() const override;
    size_t getBarrierMaxVariantSum() const override;
    size_t getNumOfSlotsUsedByTask(VPURT::TaskOp op) const override;
    VPURT::TaskOp getTaskOpAtIndex(size_t opIdx) const override;
    BarrierInfoMaps optimizeBarrierProducers(size_t blockIdx);
    BarrierInfoMaps optimizeBarriersWithSameProducers(size_t blockIdx, bool checkValidSlotCount = true);
    BarrierInfoMaps optimizeBarrierConsumers(size_t blockIdx);
    BarrierInfoMaps getBarrierMaps();
    BarrierInfoMaps optimizeBarriers(bool checkValidSlotCount = true, bool considerTaskFifoDependency = false);

private:
    size_t _maxVariantCountPerBarrier = 0;
};

//
// Helper routines to work with BarrierInfoTest
//
SmallVector<BarrierInfo::TaskSet> toTaskSet(const SmallVector<SmallVector<size_t>>& map);
SmallVector<SmallVector<size_t>> toTaskVec(const SmallVector<BarrierInfo::TaskSet>& map);
void fillProducersAndConsumers(BarrierInfoMaps& barrierMaps);

}  // namespace vpux
