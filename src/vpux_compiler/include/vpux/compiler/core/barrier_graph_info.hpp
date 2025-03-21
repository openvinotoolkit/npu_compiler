//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/barrier_info.hpp"

namespace vpux {

/**
 * This class is responsible for building and storing information about the graph of barriers like below.
 * For example, the subgraph:
 *                 op0
 *                  |
 *                 bar0
 *                  |
 *                 op1
 *               /     \
 *             bar1    bar2
 *              |       |
 *             op2     op3
 *              |     /    \
 *             bar3 bar4   bar5
 *              |     |     |
 *             op4   op5   op6
 *                 \ /   /
 *                  bar6
 *
 * The class will build barrier graph like below:
 *                 bar0
 *               /     \
 *             bar1    bar2
 *              |     /    \
 *             bar3 bar4   bar5
 *                 \ /   /
 *                  bar6
 *
 * From the view of barrier, the tasks can be separated into different execution steps according to their waiting
 * barriers. For example, the execution steps for the subgraph is:
 * step 0: op0
 * step 1: op1
 * step 2: op2, op3
 * step 3: op4, op5, op6
 *
 * Besides that, the longest hop info can be provided by this class.
 * For each barrier, there may be more than one consumer. And for each consumer, the hop stands for how many
 * barriers will be passed from current consumer to the next task on the same HW queue. This is calculated independently
 * for both direction - hop distance to previous one and hop distance to next one, largest one is picked. The task queue
 * type with largest hop value is treated as the longest hop queue type for the barrier. For example, the subgraph
 * below,
 *
 *  DMA0     DMA1
 *    \        /
 *       bar0
 *    /        \
 *  DPU0      DMA2
 *    \        /
 *       bar1
 *    /    |    \
 *  SHV0  DPU1   DMA3
 *    \    |    /
 *       bar2
 *    /        \
 *  DPU2      DMA4
 *    \        /
 *       bar3
 *    /        \
 *  DPU3      DMA5
 *    \        /
 *       bar4
 *    /        \
 *  SHV1       DMA6
 *
 *  For bar1, the hops for its consumers are
 *  SHV0: 3([bar2, bar3, bar4] to SHV1)
 *  DPU1: 1([bar2] to DPU2)
 *  DMA3: 1([bar2] to DMA4)
 *  so the bar1's longest hop queue type is SHV
 *
 * And this class is used to help build the barrier graph and calculate the execution step and longest hop infomation.
 */
class BarrierGraphInfo {
public:
    using BarrierSet = BarrierInfo::TaskSet;
    BarrierGraphInfo();
    explicit BarrierGraphInfo(mlir::func::FuncOp func);

    // get the parent barriers from the barrier graph
    BarrierSet getParentBarrier(size_t barrierInd);
    // get the child barriers from the barrier graph
    BarrierSet getChildrenBarrier(size_t barrierInd);
    // get the task's execution step
    size_t getTaskExecutionStep(size_t taskInd) const;
    // get all task execution batches (key: execution step, value: vector of task indexes)
    std::map<size_t, SmallVector<size_t>> getExecutionStepTaskBatch() const;
    // get the longest hop task queue type for each barrier.
    SmallVector<VPURT::TaskQueueType> getBarrierLongestQueueType() const;
    // get the first execution step when the barrier has no dependence on any other
    SmallVector<size_t> getBarrierFirstExecutionStep() const;
    // get the last execution step when the barrier is no longer required by other barriers.
    SmallVector<size_t> getBarrierLastExecutionStep() const;

    BarrierInfo& getBarrierInfo();
    void clearAttributes();

protected:
    std::optional<size_t> getNextTaskInFifo(size_t taskInd) const;
    std::optional<size_t> getPreTaskInFifo(size_t taskInd) const;
    VPURT::TaskQueueType getTaskQueueType(size_t taskInd) const;

    void buildTaskFifo();
    void calculateTaskAndBarrierExecutionSteps();
    void buildBarrierDependenceMap();
    void calculateBarrierLongestHopQueueType();
    void correctFinalBarrierLongestHopQueueType();

    Logger _log;
    mlir::func::FuncOp _func;
    std::shared_ptr<BarrierInfo> _barrierInfo;
    mlir::StringAttr _taskExecutionStepAttrName;

    // indexOf(VPURT::TaskQueueType) 'contains' [ indexOf(VPURT::TaskOp)... ].
    std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> _taskQueueTypeMap;
    // indexOf(VPURT::TaskQueueType) 'contains' indexOf(VPURT::TaskOp).
    SmallVector<std::optional<size_t>> _nextTaskInSameQueue;
    // indexOf(VPURT::TaskQueueType) 'contains' indexOf(VPURT::TaskOp).
    SmallVector<std::optional<size_t>> _prevTaskInSameQueue;
    // indexOf(Execution step) 'contains' [ indexOf(VPURT::TaskOp)... ].
    std::map<size_t, SmallVector<size_t>> _executeStepTaskBatch;
    // indexOf(indexOf(VPURT::TaskOp)) 'contains' execution step.
    SmallVector<size_t> _taskExecutionStep;
    // indexOf(indexOf(VPURT::BarrierOp)) 'contains' [ indexOf(VPURT::BarrierOp parents)... ].
    SmallVector<llvm::BitVector> _barrierParentData;
    // indexOf(indexOf(VPURT::BarrierOp)) 'contains' [ indexOf(VPURT::BarrierOp children)... ].
    SmallVector<llvm::BitVector> _barrierChildrenData;
    // indexOf(indexOf(VPURT::BarrierOp)) 'contains' [ LongestHopTaskQueueType... ].
    SmallVector<VPURT::TaskQueueType> _barrierLongestHopQueueType;
    // indexOf(indexOf(VPURT::BarrierOp)) 'contains' [ first execution step... ].
    SmallVector<size_t> _barrierFirstExecutionStep;
    // indexOf(indexOf(VPURT::BarrierOp)) 'contains' [ last execution step... ].
    SmallVector<size_t> _barrierLastExecutionStep;
};

class BarrierGraphInfoTest : public BarrierGraphInfo {
public:
    BarrierGraphInfoTest(std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& taskQueueMaps,
                         BarrierInfoMaps& barrierMaps);
};
}  // namespace vpux
