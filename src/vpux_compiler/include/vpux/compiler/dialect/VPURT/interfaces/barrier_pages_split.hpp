//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/core/execution_group_analysis.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

namespace vpux {
namespace VPURT {

constexpr StringLiteral wlmPageAttrName = "wlmPage";
constexpr StringLiteral barProgDmaAtPageAttrName = "BarProgDmaAtPage";

//
// BarrierPagesSplitHandler
// Module is responsible for splitting graph into subgraph (pages) based on barrier usage
//
// Input requirements:
//   Incoming IR is expected to have barriers ordered by consumption
//
// Page assignment:
//   Each barrier is assigned to a page based on the formula:
//     BarPage = BarInd / PageSize
//   PageSize is half of physical barriers
//
// Physical Barrier allocation:
//   Pages are divided into even and odd indices:
//   Even-numbered pages (Page 0, 2, 4, ...) use physical barriers from the first half:
//     [0, (numPhysBars/2)−1]
//   Odd-numbered pages (Page 1, 3, 5, ...) use physical barriers from the second half:
//     [numPhysBars/2, numPhysBars ​−1]
//   Physical barrier assignment following page split is done by AssignPhysicalBarriers pass
//
// Task dependencies constraints:
//   Each task must only use barriers from neighboring pages (e.g., Page N and Page N+1),
//   because those pages will use exclusive physical barriers
//   Pages using the same set of physical barriers (e.g., Page N and Page N+2) cannot overlap in execution.
//   Tasks accessing Page N+2 can only start after all tasks in Page N have finished.
//
// Legalization Process:
//   If a task violates task constraints, the barrier dependencies are modified to ensure correctness
//
class BarrierPagesSplitHandler {
public:
    BarrierPagesSplitHandler() = delete;
    BarrierPagesSplitHandler(mlir::func::FuncOp func, BarrierInfo& barrierInfo, size_t numPhysBarriers,
                             Logger log = Logger::global());
    // Below constructor is meant to be used only for unit testing purpose
    BarrierPagesSplitHandler(BarrierInfoTest& barrierInfoTest,
                             std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& taskQueueTypeMap, size_t pageSize,
                             size_t barrierFifoDepth = 1, size_t numClusters = 1,
                             const SmallVector<size_t>& shvTasksWithDpu = {},
                             const SmallVector<size_t>& shvTasksWithDma = {}, Logger log = Logger::global());

    // Reconfigure barrier FIFO depth. Meant to be used in case of LIT tests
    void reconfigureBarrierFifoDepth(size_t barrierFifoDepth);
    void initializeForAssignment(mlir::func::FuncOp func);
    void initializeForLegalization();
    void initializeForEnqueue(mlir::func::FuncOp func);
    void initializeForVerification(mlir::func::FuncOp func);
    void initializeTaskQueueTypeMap(mlir::func::FuncOp func);

    SmallVector<size_t> getFirstAndLastBoundaryTasksForPage(size_t pageInd);

    bool areNoDepsGoingBeyondNeighborPage();
    SmallVector<size_t> getTasksWithNonAdjacentPageDependencyToLegalize();

    void legalizeWaitBarrierDependency(size_t taskInd, VPURT::TaskQueueType& taskQueueType, size_t barInd);
    void legalizeUpdateBarrierDependency(size_t taskInd, VPURT::TaskQueueType& taskQueueType, size_t barInd);
    void legalizeNonAdjacentPageDependencies(SmallVector<size_t>& tasksWithDepsToLegalize);

    bool areBoundaryTasksFromNeighborPagesDependent();
    SmallVector<std::pair<size_t, size_t>> getBoundaryTaskPairsMissingDepInBetween();
    void legalizeDepsForBoundaryTasks(SmallVector<std::pair<size_t, size_t>>& boundaryTaskPairsMissingDepInBetween);

    bool isSplitToPagesValid();

    void assignPagesToBarriersInIr();
    void assignPagesToTasksInIr();
    void readTaskPageAssignmentFromIr();

    void updateTaskPageAssignmentForShvInCaseOfNoDedicatedShvFifos();

    struct DummyDmaDataForPagesWithNoTasks {
        size_t pageInd;
        size_t waitBar;
        size_t updateBar;
        size_t insertBefore;
    };

    SmallVector<DummyDmaDataForPagesWithNoTasks> getDummyDmaDataForPagesWithNoTasks();

    void legalizeLongDependenciesForBarrierPagesSplit();
    void legalizeBoundaryTasksForBarrierPagesSplit();

    void verifyTaskBarrierPagesAreValid();
    void verifyTaskPagesAreNeverDecreasingOnQueue();
    void verifyNoCyclicDeps();
    void verifyPhysicalBarsDependencies();
    void verifyBarProgDmaDependencies(mlir::func::FuncOp func);
    void verifyEnqueueDmas(mlir::func::FuncOp func);
    void verifyEnqueueOfDmas(mlir::func::FuncOp func);

    bool cleanupRedundantBarriers();
    void ensureBarrierHasProducer();

    BarrierInfo getUpdatedBarrierInfo();

    // To be used for testing.
    BarrierInfoMaps getBarrierMaps();

    struct DmaProgrammingBarrierPosition {
        bool valid = false;
        SmallVector<size_t> waitBars;
        SmallVector<size_t> updateBars;
        size_t insertAfter;
    };

    DmaProgrammingBarrierPosition getDmaProgrammingBarrierPosition(size_t pageInd);
    SmallVector<DmaProgrammingBarrierPosition> getDmaProgrammingBarrierPositions();
    void legalizeForDmaProgrammingBarriers();
    void updateTaskPageAssignmentForQueue(size_t startTaskIndex, size_t newPageIndex, VPURT::TaskQueueType queueType,
                                          VPURT::TaskOp moveBeforeOp);

    SmallVector<size_t> getBarrierOfSingleBarrierPages();

    struct DummyDmaInsertionData {
        size_t pageInd;
        VPURT::TaskQueueType queueType;
        SmallVector<size_t> waitBars;
        SmallVector<size_t> updateBars;
        size_t insertAfter;
    };
    SmallVector<DummyDmaInsertionData> getAndLegalizeDummyDmaInsertionData();

    struct DummyBarrierData {
        size_t pageInd;
        size_t producer;
        size_t consumer;
        size_t insertAfter;
    };
    SmallVector<DummyBarrierData> getDummyBarriersInsertionData();

    SmallVector<size_t> getLastTasksOnFifoPerPageWithNoUpdBar();
    void addUpdateBarriersForLastTaskOnFifoInPage(SmallVector<size_t>& lastTaskPerTypePerPageThatNeedsUpdateBarrier);

    void initPrevPhysBarrierData(SmallVector<size_t>& barrierToPidVec);
    void initPrevPhysBarrierData(mlir::func::FuncOp func);

    struct EnqueueDmaData {
        size_t pageInd;
        VPURT::TaskQueueType queueType;
        size_t startTaskIdx;
        size_t endTaskIdx;
        SmallVector<size_t> waitBars;
        size_t insertBefore;
    };

    SmallVector<EnqueueDmaData> getEnqueueDmaData(
            const ExecutionGroupAnalysis& execGroupAnalysis,
            const mlir::DenseSet<vpux::config::ExecutorKind>& executorEnqAtBootstrap);
    VPURT::BarrierPagesSplitHandler::EnqueueDmaData getDataForNewEnqueueDmaForTask(
            size_t taskInd, size_t taskWorkloadStartIdx, size_t taskWorkloadEndIdx, VPURT::TaskQueueType queueType,
            SmallVector<mlir::DenseMap<VPURT::TaskQueueType, std::pair<size_t, size_t>>>& firstAndLastDmaOfTypePerPage);

    // For each task retrieve enqueue barrier index
    SmallVector<std::optional<size_t>> getAndLegalizeEnqueueBarrierData();

private:
    void initializePageSplitData();
    size_t getBarrierPage(size_t barInd);
    SmallVector<size_t> getFirstBoundaryTasksForPage(size_t pageInd);
    SmallVector<size_t> getLastBoundaryTasksForPage(size_t pageInd);

    void initializeBarrierToPageAssignment();
    void initializeTaskToPageAssignment();

    void readBarrierPageAssignmentFromIr();

    VPURT::TaskQueueType getTaskQueueType(size_t taskInd);
    VPURT::TaskQueueType getTaskQueueTypeWithExplicitFifo(size_t taskInd);

    void updateBoundaryTasksDataForTask(size_t taskInd);
    void enforceBoundaryTaskHasUpdateBarrier(size_t pageInd);
    void initializeBoundaryTasksData();
    void updateBoundaryTasksDataForPage(size_t pageInd);
    bool isTaskWithNonAdjacentPageDependency(size_t taskInd);

    bool isDepFromTaskAToTaskB(size_t taskA, size_t taskB);
    bool isDepFromTaskToBarrier(size_t taskInd, size_t barInd);
    bool isDepFromBarrierToTask(size_t barInd, size_t taskInd);
    bool isDepFromBarAToBarB(size_t barA, size_t barB);

    std::optional<size_t> getPrevTaskOnSameQueueIfQueueOrderEnabled(size_t taskInd) const;
    std::optional<size_t> getNextTaskOnSameQueueIfQueueOrderSupported(size_t taskInd) const;
    // Return the closest previous task on the same queue that has a wait barrier
    std::optional<size_t> getPrevTaskOnSameQueueWithWaitBarIfQueueOrderSupported(size_t taskInd) const;
    // Return the closest next task on the same queue that has an update barrier
    std::optional<size_t> getNextTaskOnSameQueueWithUpdateBarIfQueueOrderSupported(size_t taskInd) const;

    void getPageStartTasksAndBars(size_t pageInd, BarrierInfo::TaskSet& pageStartTasks,
                                  BarrierInfo::TaskSet& pageStartBars);
    void getPageEndTasksAndBars(size_t pageInd, BarrierInfo::TaskSet& pageEndTasks, BarrierInfo::TaskSet& pageEndBars,
                                BarrierInfo::TaskSet& pageEndAllBars);

    std::optional<size_t> getInsertionPointForDmaProgrammingBarriers(const BarrierInfo::TaskSet& pageStartBars,
                                                                     const BarrierInfo::TaskSet& pageEndBars);
    void adjustPageStartAndEndPointsIfOnBlockBoundary(BarrierInfo::TaskSet& pageStartBars,
                                                      BarrierInfo::TaskSet& pageStartTasks,
                                                      BarrierInfo::TaskSet& pageEndBars,
                                                      BarrierInfo::TaskSet& pageEndAllBars,
                                                      BarrierInfo::TaskSet& pageEndTasks);
    void legalizePageStartBarsDependingOnPageEndBars(BarrierInfo::TaskSet& pageStartTasks,
                                                     BarrierInfo::TaskSet& pageStartBars,
                                                     BarrierInfo::TaskSet& pageStartBarsToLegalize);
    BarrierInfo::TaskSet getPageStartBarsDependingOnPageEndBars(BarrierInfo::TaskSet& pageStartBars,
                                                                BarrierInfo::TaskSet& pageEndBars);

    bool isDummyDmaNeeded(size_t pageInd, VPURT::TaskQueueType dmaQueueType,
                          std::optional<size_t> lastDmaTaskOnSameQueueInPageOpt);

    std::pair<vpux::BarrierInfo::TaskSet, vpux::BarrierInfo::TaskSet> getDummyDmaBars(size_t pageInd);
    size_t getDummyDmaInsertionPoint(
            size_t pageInd, VPURT::TaskQueueType dmaQueueType,
            SmallVector<mlir::DenseMap<VPURT::TaskQueueType, std::pair<size_t, size_t>>>& firstAndLastDmaOfTypePerPage);
    BarrierInfo::TaskSet getDummyDmaUpdateBars(size_t pageInd, size_t insertionPoint,
                                               SmallVector<std::pair<size_t, size_t>>& firstAndLastBarIndPerPage);

    SmallVector<mlir::DenseMap<VPURT::TaskQueueType, std::pair<size_t, size_t>>> getFirstAndLastDmaOfTypePerPage();

    void findShvTasksWithDpuOrDma(mlir::func::FuncOp func);

    SmallVector<size_t> getBarrierPidPrevUsageVec(BarrierInfo::TaskSet& barriers);

    // Get number of workloads under single processed task
    // In case of DPU it is number of DPU variant
    // In case of SHV it is number of SHV kernel run ops
    size_t getNumberOfWorkloads(size_t taskInd);

    bool isDpuEnqueueDelayNeededAfterShvWithDpu(
            size_t taskInd, VPURT::TaskQueueType queueType,
            DenseMap<VPURT::TaskQueueType, size_t>& shvTasksWithDpuVecStartIndPerQueue, size_t shvTasksWithDpuVecInd,
            std::map<VPURT::TaskQueueType, SmallVector<EnqueueDmaData>>& enqDmaDataVecPerQueue);

    void createInitialDelayDpuEnqueueAfterShvWithDpu(
            size_t taskInd, VPURT::TaskQueueType queueType, bool wasDpuAlreadyDelayed, size_t taskWorkloadStartIdx,
            size_t taskWorkloadEndIdx, size_t shvTasksWithDpuVecInd,
            std::map<VPURT::TaskQueueType, SmallVector<EnqueueDmaData>>& enqDmaDataVecPerQueue);

    void updateDelayedDpuEnqPositionToNotImpactEnqOfPrevDpu(
            size_t taskInd, VPURT::TaskQueueType queueType, const VPURT::TaskQueueType& enqueueDmaQueueType,
            const DenseMap<VPURT::TaskQueueType, std::optional<size_t>>& previousTaskIndOptPerQueue,
            std::map<VPURT::TaskQueueType, SmallVector<EnqueueDmaData>>& enqDmaDataVecPerQueue);

    void updateDpuEnqPositionToAccountForSyncTask(
            size_t taskInd, VPURT::TaskQueueType queueType, const VPURT::TaskQueueType& enqueueDmaQueueType,
            std::map<VPURT::TaskQueueType, SmallVector<EnqueueDmaData>>& enqDmaDataVecPerQueue);

    SmallVector<size_t> _barrierToPageAssignment;

    // Vector where index is pageInd and value is first barrier in that page
    SmallVector<size_t> _firstBarrierInPage;

    // Store information at which page given task start to execute
    SmallVector<size_t> _taskPageAssignment;

    // Store information about first and last task index on each page
    SmallVector<std::optional<std::pair<size_t, size_t>>> _firstAndLastTaskPerPage;

    // For each page index store per HW FIFO boundary task data.
    // Since for each HW FIFO there can be a sequence of boundary tasks store index of first and last one
    // Depending on legalization either first task is used, when checking deps to earlier pages
    // or last task when checking deps to next pages
    SmallVector<std::map<VPURT::TaskQueueType, std::pair<size_t, size_t>>> _firstAndLastBoundaryTaskForEachPagePerFifo;

    BarrierInfo _barrierInfo;

    size_t _pageSize;
    size_t _pageCount = 0;

    bool _fifoPerShaveEngineEnabled = true;

    // For each queue store a vector of task indexes on this queue
    std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> _taskQueueTypeMap;

    // Default size of BARRIER FIFO depth, which corresponds to
    // number of virtual barriers using same physical barrier that will be pushed
    // to the Barrier FIFO using compiler schedule DMA
    // 4 is used because in a single register write we can push 4 barrier descriptors
    // Use 1 in case barrier FIFO is not supported
    static constexpr int64_t BARRIER_FIFO_SIZE = 4;
    size_t _barrierFifoDepth;

    SmallVector<DmaProgrammingBarrierPosition> _barProgDmaPosVec;

    std::optional<size_t> _startBarrierIndex;

    // For each barrier index store index of barrier with same PID
    // If there is no value set it means barrier is the first one to use given PID
    SmallVector<std::optional<size_t>> _barrierPidPrevUsageVec;

    // Store information about SHV tasks which can submit DPU ops as
    // subsequent DPUs might need to have their enqueue delayed to guarantee
    // DPU submitted by SHV is not blocked in the DPU FIFO
    mlir::DenseMap<size_t, SmallVector<size_t>> _shvTasksWithDpuPerTile;

    // Store information about SHV tasks which can submit DMAs as this needs
    // to be taken into account
    mlir::DenseMap<size_t, mlir::DenseSet<size_t>> _shvTasksWithDmaPerTile;

    // Store information about task control map that will be initialized through
    // barrierInfo for given control graph split block index
    std::optional<size_t> _blockIdxOfTaskControlMap;
    std::pair<SmallVector<llvm::BitVector>, size_t> _taskControlMapAndOffset;

    // Flag indicating if number of workloads should be retrieved from
    // TaskOp. In case of UnitTest to be set to "false" to remove dependency on IR
    bool _getNumberOfWorkloadsFromTaskOpFlag = true;

    bool _legalizationMode = false;

    Logger _log;
};

}  // namespace VPURT
}  // namespace vpux
