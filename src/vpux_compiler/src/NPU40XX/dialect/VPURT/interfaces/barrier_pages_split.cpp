//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPURT/interfaces/barrier_pages_split.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/shave.hpp"
#include "vpux/compiler/utils/wlm_legalization_utils.hpp"

#include <llvm/ADT/SetOperations.h>

using namespace vpux;

vpux::VPURT::BarrierPagesSplitHandler::BarrierPagesSplitHandler(mlir::func::FuncOp func, BarrierInfo& barrierInfo,
                                                                size_t numPhysBarriers, Logger log)
        : _barrierInfo(barrierInfo), _log(log) {
    _barrierFifoDepth = config::getConstraint(func, config::BARRIER_FIFO_DEPTH);
    VPUX_THROW_UNLESS(numPhysBarriers % 2 == 0, "Number of physical barriers must be even, numPhysBarriers - {0}",
                      numPhysBarriers);
    _pageSize = numPhysBarriers / 2;
}

// Below constructor is meant to be used only for unit testing purpose
vpux::VPURT::BarrierPagesSplitHandler::BarrierPagesSplitHandler(
        BarrierInfoTest& barrierInfoTest, std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& taskQueueTypeMap,
        size_t pageSize, size_t barrierFifoDepth, size_t numClusters, const SmallVector<size_t>& shvTasksWithDpu,
        Logger log)
        : _barrierInfo(barrierInfoTest),
          _pageSize(pageSize),
          _taskQueueTypeMap(taskQueueTypeMap),
          _barrierFifoDepth(barrierFifoDepth),
          _log(log) {
    _startBarrierIndex = 0;
    _getNumberOfWorkloadsFromTaskOpFlag = false;

    initializeBarrierToPageAssignment();
    initializeTaskToPageAssignment();
    initializeBoundaryTasksData();

    // Initialize data for SHV tasks with DPU
    for (auto shvTaskInd : shvTasksWithDpu) {
        auto shvQueueIt = llvm::find_if(_taskQueueTypeMap, [&](const auto& item) {
            return llvm::find(item.second, shvTaskInd) != item.second.end();
        });
        VPUX_THROW_WHEN(shvQueueIt == _taskQueueTypeMap.end(), "Can not find task {0} in task queue map", shvTaskInd);

        auto tileIndex = vpux::getShaveTileIndexFromEncodedId(shvQueueIt->first.id, numClusters);

        _shvTasksWithDpuPerTile[tileIndex].push_back(shvTaskInd);
    }
}

void vpux::VPURT::BarrierPagesSplitHandler::reconfigureBarrierFifoDepth(size_t barrierFifoDepth) {
    _barrierFifoDepth = barrierFifoDepth;
}

// Configure the barrier page split handler for assignment of barriers and tasks to pages
void vpux::VPURT::BarrierPagesSplitHandler::initializeForAssignment(mlir::func::FuncOp func) {
    _taskQueueTypeMap = VPURT::getTaskOpQueues(func, _barrierInfo);

    mlir::DenseSet<vpux::VPU::ExecutorKind> executorKind = {
            VPU::ExecutorKind::DPU,
            VPU::ExecutorKind::DMA_NN,
            VPU::ExecutorKind::SHAVE_ACT,
    };
    _barrierInfo.initializeTaskQueueTypeMap(executorKind);
    _barrierInfo.buildTaskQueueTypeMap();

    initializeBarrierToPageAssignment();
    initializeTaskToPageAssignment();
}

// Configure the barrier page split handler for legalization of the schedule for split into pages
void vpux::VPURT::BarrierPagesSplitHandler::initializeForLegalization() {
    mlir::DenseSet<vpux::VPU::ExecutorKind> executorKind = {
            VPU::ExecutorKind::DPU,
            VPU::ExecutorKind::DMA_NN,
            VPU::ExecutorKind::SHAVE_ACT,
    };
    _barrierInfo.initializeTaskQueueTypeMap(executorKind);
    _barrierInfo.buildTaskQueueTypeMap();

    // Get number of pages based on information in IR. Read page assignment from last barrier in IR
    auto lastBarOp = _barrierInfo.getBarrierOpAtIndex(_barrierInfo.getNumOfBarrierOps() - 1);
    auto lastPageOpt = lastBarOp.getWlmPage();
    VPUX_THROW_UNLESS(lastPageOpt.has_value(), "Barrier {0} does not have page assignment", lastBarOp);
    _pageCount = lastPageOpt.value() + 1;

    readTaskPageAssignmentFromIr();
    readBarrierPageAssignmentFromIr();

    initializeBoundaryTasksData();
}

void vpux::VPURT::BarrierPagesSplitHandler::findShvTasksWithDpu(size_t numClusters) {
    for (auto& [queueType, taskVec] : _taskQueueTypeMap) {
        if (queueType.type != VPU::ExecutorKind::SHAVE_ACT) {
            continue;
        }

        for (auto taskInd : taskVec) {
            if (isDpuShaveKernelType(_barrierInfo.getTaskOpAtIndex(taskInd))) {
                auto tileIndex = vpux::getShaveTileIndexFromEncodedId(queueType.id, numClusters);
                _shvTasksWithDpuPerTile[tileIndex].push_back(taskInd);
            }
        }
    }
    // Sort as later indexes are being compared to minimize the range of checks and reduce
    // compile time
    for (auto& [_, shvTasksVec] : _shvTasksWithDpuPerTile) {
        llvm::sort(shvTasksVec);
    }
}

// Configure the barrier page split handler for finding enqueue DMA data
void vpux::VPURT::BarrierPagesSplitHandler::initializeForEnqueue(mlir::func::FuncOp func) {
    initializeForLegalization();
    _taskQueueTypeMap = VPURT::getTaskOpQueues(func, _barrierInfo);

    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto numClusters = config::getTileExecutor(module).getCount();

    findShvTasksWithDpu(numClusters);
    initPrevPhysBarrierData(func);

    // Locate start barrier
    func.walk([&](VPURT::ConfigureBarrierOp barOp) {
        if (barOp.getIsStartBarrier()) {
            _startBarrierIndex = _barrierInfo.getIndex(barOp);
            return mlir::WalkResult::interrupt();
        }

        return mlir::WalkResult::advance();
    });
}

// Configure the barrier page split handler for performing final verification
void vpux::VPURT::BarrierPagesSplitHandler::initializeForVerification(mlir::func::FuncOp func) {
    // Requires same initialization as for enqueue search
    initializeForEnqueue(func);
}

void vpux::VPURT::BarrierPagesSplitHandler::readBarrierPageAssignmentFromIr() {
    _barrierToPageAssignment.resize(_barrierInfo.getNumOfBarrierOps());
    _firstBarrierInPage.resize(_pageCount);

    for (size_t barInd = 0; barInd < _barrierInfo.getNumOfBarrierOps(); barInd++) {
        auto barOp = _barrierInfo.getBarrierOpAtIndex(barInd);
        auto pageOpt = barOp.getWlmPage();
        VPUX_THROW_UNLESS(pageOpt.has_value(), "Barrier {0} does not have page assignment", barInd);
        auto pageInd = pageOpt.value();
        _barrierToPageAssignment[barInd] = pageOpt.value();

        if (pageInd > 0 && _firstBarrierInPage[pageInd] == 0) {
            _firstBarrierInPage[pageInd] = barInd;
        }
    }
}

size_t vpux::VPURT::BarrierPagesSplitHandler::getBarrierPage(size_t barInd) {
    VPUX_THROW_UNLESS(barInd < _barrierToPageAssignment.size(), "Barrier index {0} out of range {1}", barInd,
                      _barrierToPageAssignment.size());
    return _barrierToPageAssignment[barInd];
}

void vpux::VPURT::BarrierPagesSplitHandler::initializeBarrierToPageAssignment() {
    _log.trace("Initializing barrier to page assignment");
    _barrierToPageAssignment.resize(_barrierInfo.getNumOfBarrierOps());

    size_t pageInd = 0;
    size_t barriersInPage = 0;
    BarrierInfo::TaskSet processedSyncPoints;

    // First barrier is always in page 0
    _firstBarrierInPage.push_back(0);

    for (size_t barInd = 0; barInd < _barrierInfo.getNumOfBarrierOps(); barInd++) {
        // If there are already max number of barriers assigned to page
        // then switch to next page for current barrier
        if (barriersInPage == _pageSize) {
            barriersInPage = 0;
            pageInd++;
            _firstBarrierInPage.push_back(barInd);
        } else {
            // Perform a check if barrier is produced by a sync task
            // In such case increment page index for next barrier
            // so that sync task is also a boundary task of page split
            auto barrierProducers = _barrierInfo.getBarrierProducers(barInd);
            for (auto producer : barrierProducers) {
                if (!_barrierInfo.isSyncPoint(producer)) {
                    continue;
                }
                if (processedSyncPoints.count(producer) > 0) {
                    // If this sync point was already processed, then no need to
                    // increment page index
                    continue;
                }

                pageInd++;
                barriersInPage = 0;
                processedSyncPoints.insert(producer);
                _firstBarrierInPage.push_back(barInd);
            }
        }
        _barrierToPageAssignment[barInd] = pageInd;
        barriersInPage++;
    }

    _pageCount = pageInd + 1;
}

// Calculate in which page each task starts. Configure this based on task wait barrier and
// previous task (on same FIFO) page assignment.
// taskPage = max(waitBarriersPage, prevTaskPage)
void vpux::VPURT::BarrierPagesSplitHandler::initializeTaskToPageAssignment() {
    _taskPageAssignment.resize(_barrierInfo.getNumOfTasks());
    _firstAndLastTaskPerPage.resize(_pageCount);
    _log.trace("Initializing task to page assignment");

    for (auto& [_, taskVec] : _taskQueueTypeMap) {
        for (size_t i = 0; i < taskVec.size(); i++) {
            auto taskInd = taskVec[i];

            size_t taskPage = 0;

            // Get page assignment of wait barriers
            auto waitBars = _barrierInfo.getWaitBarriers(taskInd);

            if (!waitBars.empty()) {
                taskPage = getBarrierPage(*std::max_element(waitBars.begin(), waitBars.end()));
            }

            if (i > 0) {
                // Task page assignment cannot be smaller then previous task on same FIFO
                auto prevTaskInd = taskVec[i - 1];
                taskPage = std::max(taskPage, _taskPageAssignment[prevTaskInd]);
            }

            _taskPageAssignment[taskInd] = taskPage;
            _log.nest().trace("Task {0}: page {1}", taskInd, taskPage);
        }
    }

    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        auto taskPage = _taskPageAssignment[taskInd];

        if (!_firstAndLastTaskPerPage[taskPage].has_value()) {
            _firstAndLastTaskPerPage[taskPage] = std::make_pair(taskInd, taskInd);
        } else {
            _firstAndLastTaskPerPage[taskPage].value().second = taskInd;
        }
    }
}

// Update boundary task data for provided task
void vpux::VPURT::BarrierPagesSplitHandler::updateBoundaryTasksDataForTask(size_t taskInd) {
    auto taskPage = _taskPageAssignment[taskInd];
    auto queueType = _barrierInfo.getTaskQueueType(taskInd);
    auto updateBars = _barrierInfo.getUpdateBarriers(taskInd);
    auto waitBars = _barrierInfo.getWaitBarriers(taskInd);

    if (waitBars.empty() && updateBars.empty()) {
        // If there are no wait or update barriers, then no need to consider this task as
        // boundary task as such task has no tight affiliation to any page
        return;
    }

    // Check if there was already a boundary task of this type for this page identified
    if (_firstAndLastBoundaryTaskForEachPagePerFifo[taskPage].find(queueType) ==
        _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage].end()) {
        // If not check if all update barriers belong to the same page as the task.
        // If all update barriers are within taskPage, then this is NOT a boundary task
        if (llvm::all_of(updateBars, [&](size_t barInd) {
                return getBarrierPage(barInd) == taskPage;
            })) {
            return;
        }

        // If there is an update barrier from next page then this is a first boundary
        // task on this HW FIFO
        _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage][queueType] = std::make_pair(taskInd, taskInd);
    }

    if (taskInd < _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage][queueType].second) {
        return;
    }

    // If there is already a boundary task of this type for this page, then update the data
    // so that after traversing all tasks there is also information about the last boundary task
    _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage][queueType].second = taskInd;

    _log.nest().trace("Task {0} is boundary task for page {1}, queue type {2}:{3}", taskInd, taskPage,
                      stringifyEnum(queueType.type).data(), queueType.id);
}

// Make sure last boundary task on each HW FIFO updates barrier from next page.
// Tasks which are on the HW FIFO after a boundary task but they themselve do not have any
// update barrier from next page should have this legalized as rest of code expects this
// based on agreed definition of boundary task
void vpux::VPURT::BarrierPagesSplitHandler::enforceBoundaryTaskHasUpdateBarrier(size_t pageInd) {
    for (auto boundaryTask : getLastBoundaryTasksForPage(pageInd)) {
        auto taskUpdateBarsVec = to_small_vector(_barrierInfo.getUpdateBarriers(boundaryTask));

        // For analysis do not account for barriers from current page. Boundary task on PageN
        // needs to have update barriers from next page
        taskUpdateBarsVec.erase(llvm::remove_if(taskUpdateBarsVec,
                                                [&](auto barInd) {
                                                    return getBarrierPage(barInd) == pageInd;
                                                }),
                                taskUpdateBarsVec.end());
        if (!taskUpdateBarsVec.empty()) {
            // If task updates barriers from next page then no need to add any additional update barrier
            continue;
        }

        // If there are no update barriers of this page then need to search for some other update
        // barrier that can be used and which boundary task can update
        _log.nest().trace("Boundary task {0} from page {1} does not have update barrier on next pages", boundaryTask,
                          pageInd);

        // Check for update barrier of some previous task on the same FIFO
        auto currentTaskOpt = _barrierInfo.getPrevTaskOnSameQueue(boundaryTask);
        while (taskUpdateBarsVec.empty() && currentTaskOpt.has_value() &&
               _taskPageAssignment[currentTaskOpt.value()] == pageInd) {
            taskUpdateBarsVec = to_small_vector(_barrierInfo.getUpdateBarriers(currentTaskOpt.value()));

            taskUpdateBarsVec.erase(llvm::remove_if(taskUpdateBarsVec,
                                                    [&](auto barInd) {
                                                        return getBarrierPage(barInd) == pageInd;
                                                    }),
                                    taskUpdateBarsVec.end());

            currentTaskOpt = _barrierInfo.getPrevTaskOnSameQueue(currentTaskOpt.value());
        }

        VPUX_THROW_UNLESS(!taskUpdateBarsVec.empty(),
                          "Boundary task {0} from page {1} has no update barriers on next pages", boundaryTask,
                          pageInd);

        // If task updates multiple barriers, pick only one with smallest index and
        // update barrier dependencies so that boundary also updates this barrier
        // No need to use more barriers as 1 barrier is enough to know that boundary task has finished
        auto newUpdateBar = *std::min_element(taskUpdateBarsVec.begin(), taskUpdateBarsVec.end());

        _barrierInfo.addProducer(newUpdateBar, boundaryTask);
        _log.nest().trace("Add producer {0} to barrier {1}", boundaryTask, newUpdateBar);
    }
}

// Check all tasks and identify boundary tasks.
// A boundary task is one where at least one update barrier belongs to a page
// that is greater than taskPage (indicating cross-page dependency).
// Boundary tasks are later used for legalization purposes.
void vpux::VPURT::BarrierPagesSplitHandler::initializeBoundaryTasksData() {
    _firstAndLastBoundaryTaskForEachPagePerFifo.resize(_pageCount);
    _firstAndLastTaskPerPage.resize(_pageCount);
    _log.trace("Initializing boundary tasks data");

    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        auto taskPage = _taskPageAssignment[taskInd];

        if (!_firstAndLastTaskPerPage[taskPage].has_value()) {
            _firstAndLastTaskPerPage[taskPage] = std::make_pair(taskInd, taskInd);
        } else {
            _firstAndLastTaskPerPage[taskPage].value().second = taskInd;
        }

        updateBoundaryTasksDataForTask(taskInd);
    }

    for (size_t pageInd = 0; pageInd < _pageCount - 1; pageInd++) {
        auto pageBoundaryTasks = getFirstAndLastBoundaryTasksForPage(pageInd);
        if (pageBoundaryTasks.empty()) {
            _log.trace("No boundary tasks for page {0}", pageInd);
            // If in rare case there is no boundary task for page - task that starts at PageN and updates
            // barrier from PageN+1 then need to create one artificially. Pick lates task in PageN and find
            // first wait barrier from PageN+1 and create dependency.
            VPUX_THROW_WHEN(!_firstAndLastTaskPerPage[pageInd].has_value(), "No first and last task set for page {0}",
                            pageInd);
            auto lastTaskOnPage = _firstAndLastTaskPerPage[pageInd].value().second;
            auto nextPageBarrier = _firstBarrierInPage[pageInd + 1];

            _barrierInfo.addProducer(nextPageBarrier, lastTaskOnPage);
            _log.nest(2).trace("Add producer {0}(page {1}) to barrier {2}(page {3})", lastTaskOnPage, pageInd,
                               nextPageBarrier, getBarrierPage(nextPageBarrier));

            _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd][_barrierInfo.getTaskQueueType(lastTaskOnPage)] =
                    std::make_pair(lastTaskOnPage, lastTaskOnPage);
            pageBoundaryTasks.push_back(lastTaskOnPage);
        }

        _log.trace("Page {0} boundary tasks: {1}", pageInd, pageBoundaryTasks);
        enforceBoundaryTaskHasUpdateBarrier(pageInd);
    }
}

// Update boundary task data for provided page
void vpux::VPURT::BarrierPagesSplitHandler::updateBoundaryTasksDataForPage(size_t pageInd) {
    _log.trace("Update boundary tasks data for page {0}", pageInd);
    VPUX_THROW_WHEN(pageInd >= _pageCount, "Page index {0} out of range {1}", pageInd, _pageCount);

    // No need to update data for last page
    if (pageInd == _pageCount - 1) {
        return;
    }

    VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].empty(), "No boundary tasks set for page {0}",
                    pageInd);
    VPUX_THROW_WHEN(!_firstAndLastTaskPerPage[pageInd].has_value(), "No first and last task set for page {0}", pageInd);

    _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].clear();
    for (size_t taskInd = _firstAndLastTaskPerPage[pageInd].value().first;
         taskInd <= _firstAndLastTaskPerPage[pageInd].value().second; taskInd++) {
        updateBoundaryTasksDataForTask(taskInd);
    }

    _log.trace("Page {0} boundary tasks: {1}", pageInd, getFirstAndLastBoundaryTasksForPage(pageInd));
    enforceBoundaryTaskHasUpdateBarrier(pageInd);
}

// For given page get first and last boundary tasks for each HW FIFO
SmallVector<size_t> vpux::VPURT::BarrierPagesSplitHandler::getFirstAndLastBoundaryTasksForPage(size_t pageInd) {
    auto boundaryTasks = getFirstBoundaryTasksForPage(pageInd);
    auto lastBoundaryTasks = getLastBoundaryTasksForPage(pageInd);
    boundaryTasks.insert(boundaryTasks.end(), lastBoundaryTasks.begin(), lastBoundaryTasks.end());

    llvm::sort(boundaryTasks);
    boundaryTasks.erase(std::unique(boundaryTasks.begin(), boundaryTasks.end()), boundaryTasks.end());
    return boundaryTasks;
}

// For given page get first boundary tasks for each HW FIFO
SmallVector<size_t> vpux::VPURT::BarrierPagesSplitHandler::getFirstBoundaryTasksForPage(size_t pageInd) {
    VPUX_THROW_UNLESS(pageInd < _firstAndLastBoundaryTaskForEachPagePerFifo.size(),
                      "Page index {0} not within limit {1}", pageInd,
                      _firstAndLastBoundaryTaskForEachPagePerFifo.size() - 1);

    SmallVector<size_t> boundaryTasks;

    for (auto& [_, firstLastTaskIndPair] : _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd]) {
        boundaryTasks.push_back(firstLastTaskIndPair.first);
    }

    llvm::sort(boundaryTasks);
    return boundaryTasks;
}

// For given page get last boundary tasks for each HW FIFO
SmallVector<size_t> vpux::VPURT::BarrierPagesSplitHandler::getLastBoundaryTasksForPage(size_t pageInd) {
    VPUX_THROW_UNLESS(pageInd < _firstAndLastBoundaryTaskForEachPagePerFifo.size(),
                      "Page index {0} not within limit {1}", pageInd,
                      _firstAndLastBoundaryTaskForEachPagePerFifo.size() - 1);

    SmallVector<size_t> boundaryTasks;

    for (auto& [_, firstLastTaskIndPair] : _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd]) {
        boundaryTasks.push_back(firstLastTaskIndPair.second);
    }

    llvm::sort(boundaryTasks);
    return boundaryTasks;
}

// Set barrier page assignment to barrier ops in IR
// This information is needed by physical barrier assignment pass or for later
// passes for logging and debug purpose
void vpux::VPURT::BarrierPagesSplitHandler::assignPagesToBarriersInIr() {
    // Assign pages to barriers
    _log.trace("Assigning pages to barriers");
    for (size_t barInd = 0; barInd < _barrierInfo.getNumOfBarrierOps(); barInd++) {
        auto barOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(_barrierInfo.getBarrierOpAtIndex(barInd));
        auto barPage = getBarrierPage(barInd);
        barOp.setWlmPage(barPage);
        _log.nest().trace("Barrier {0} assigned to page {1}", barInd, barPage);
    }
}

// Set barrier page assignment to task ops in IR
// This information is used by page split passes
void vpux::VPURT::BarrierPagesSplitHandler::assignPagesToTasksInIr() {
    _log.trace("Assigning pages to tasks");
    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        auto taskOp = _barrierInfo.getTaskOpAtIndex(taskInd);
        taskOp.setWlmPage(_taskPageAssignment[taskInd]);
    }
}

// Read barrier page assignment from IR. After first WLM page split runs, next passes
// are expected to read this data from IR
void vpux::VPURT::BarrierPagesSplitHandler::readTaskPageAssignmentFromIr() {
    // Assign pages to barriers
    _log.trace("Retrieve task to pages assignment from IR");
    _taskPageAssignment.resize(_barrierInfo.getNumOfTasks());
    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        auto taskOp = _barrierInfo.getTaskOpAtIndex(taskInd);
        VPUX_THROW_UNLESS(taskOp.getWlmPageAttr() != nullptr,
                          "Get: attribute '{0}' was not set for '{1}' operation at '{2}'", VPURT::wlmPageAttrName,
                          taskOp->getName(), taskOp->getLoc());
        _taskPageAssignment[taskInd] = checked_cast<uint32_t>(taskOp.getWlmPage().value());
    }
}

// Identify pages that do not have any tasks assigned to this page
// All legalization steps expect each page to have at least one task so during page
// split it is expected that pages without tasks will have dummy tasks inserted using information
// returned from this method.
SmallVector<VPURT::BarrierPagesSplitHandler::DummyDmaDataForPagesWithNoTasks>
vpux::VPURT::BarrierPagesSplitHandler::getDummyDmaDataForPagesWithNoTasks() {
    SmallVector<DummyDmaDataForPagesWithNoTasks> dummyDmaDataForPagesWithNoTasks;

    // Traverse all pages and identify which do not have any tasks assigned
    // Skip last page as it is OK if it doesn't have tasks
    for (size_t pageInd = 0; pageInd < _pageCount - 1; pageInd++) {
        if (_firstAndLastTaskPerPage[pageInd].has_value()) {
            continue;
        }
        auto nextPageFirstBar = _firstBarrierInPage[pageInd + 1];
        auto pageLastBar = nextPageFirstBar - 1;
        auto insertBefore = _barrierInfo.getBarrierLatestProducer(pageLastBar) + 1;
        dummyDmaDataForPagesWithNoTasks.push_back({pageInd, pageLastBar, nextPageFirstBar, insertBefore});
    }

    return dummyDmaDataForPagesWithNoTasks;
}

// Check if given task has illegal dependencies from page split point of view
// If task is in pageN then:
// - if wait barrier is in earlier page (pageX, and X < N)
// - if update barrier page is later then next page (PageX, and X > N + 1)
// then task is considered to have too long dependency and this needs to be legalized
bool vpux::VPURT::BarrierPagesSplitHandler::isTaskWithNonAdjacentPageDependency(size_t taskInd) {
    auto taskPage = _taskPageAssignment[taskInd];

    // Check if any wait barrier is smaller than task page
    auto waitBars = _barrierInfo.getWaitBarriers(taskInd);
    if (llvm::any_of(waitBars, [&](const auto bar) {
            return getBarrierPage(bar) < taskPage;
        })) {
        return true;
    }

    // Check if any update barrier is greater than task page by more than 1
    auto updateBars = _barrierInfo.getUpdateBarriers(taskInd);
    if (llvm::any_of(updateBars, [&](const auto bar) {
            return getBarrierPage(bar) > taskPage + 1;
        })) {
        return true;
    }

    return false;
}
// Check for each task if their barrier dependencies are not too long. If any task has
// illegal dependencies then split is not valid and needs to be legalized
bool vpux::VPURT::BarrierPagesSplitHandler::areNoDepsGoingBeyondNeighborPage() {
    _log.trace("Checking if no deps going beyond neighbor page");
    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        if (isTaskWithNonAdjacentPageDependency(taskInd)) {
            _log.trace("Task {0} has deps going beyond neighbor page", taskInd);
            return false;
        }
    }
    _log.trace("All deps are within neighbor pages");
    return true;
}

// Get all tasks which have illegal - too long dependencies. These tasks will be legalized
// so that barrier dependencies satisfy barrier pages split constraint - task can use barriers
// only from adjacent pages
SmallVector<size_t> vpux::VPURT::BarrierPagesSplitHandler::getTasksWithNonAdjacentPageDependencyToLegalize() {
    _log.trace("Getting tasks with long bar deps to legalize");
    SmallVector<size_t> tasksWithDepsToLegalize;
    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        if (isTaskWithNonAdjacentPageDependency(taskInd)) {
            tasksWithDepsToLegalize.push_back(taskInd);
        }
    }
    _log.trace("Number of tasks with long bar deps: {0}", tasksWithDepsToLegalize.size());
    return tasksWithDepsToLegalize;
}

// Legalize long dependency on wait barrier for a given task. If task on PageN
// has wait barrier on PageX, where X < N, then this dependency needs to be legalized
void vpux::VPURT::BarrierPagesSplitHandler::legalizeWaitBarrierDependency(size_t taskInd,
                                                                          VPURT::TaskQueueType& taskQueueType,
                                                                          size_t barInd) {
    size_t taskPage = _taskPageAssignment[taskInd];

    auto barPage = getBarrierPage(barInd);
    VPUX_THROW_WHEN(barPage > taskPage, "Task {0} from page {1} waits on barrier {2} from page {3}", taskInd, taskPage,
                    barInd, barPage);
    if (taskPage == barPage) {
        // Barrier is within accepted range
        // No need to legalize
        return;
    }
    _log.trace("Wait bar {0}(page {1}) needs to be legalized", barInd, barPage);

    // Find wait barrier of previous tasks on same FIFO. This barrier will be used for legalization
    // purpose as existing wait barrier will be disconnected from this task
    std::optional<size_t> waitBarOnSamePage = std::nullopt;
    std::optional<size_t> taskWithWaitBarOnSamePageOpt = taskInd;
    do {
        auto taskWaitBars = _barrierInfo.getWaitBarriers(taskWithWaitBarOnSamePageOpt.value());
        for (auto taskWaitBar : taskWaitBars) {
            if (getBarrierPage(taskWaitBar) == taskPage) {
                waitBarOnSamePage = taskWaitBar;
                break;
            }
        }
        if (waitBarOnSamePage.has_value()) {
            break;
        }

        taskWithWaitBarOnSamePageOpt =
                _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(taskWithWaitBarOnSamePageOpt.value());
    } while (!waitBarOnSamePage.has_value() && taskWithWaitBarOnSamePageOpt.has_value());

    VPUX_THROW_WHEN(!waitBarOnSamePage.has_value(), "No wait bar on page {0} for task {1}(page {2})", taskPage, taskInd,
                    taskPage);
    VPUX_THROW_WHEN(!taskWithWaitBarOnSamePageOpt.has_value(),
                    "No task with wait bar on same page for task {0}(page {1})", taskInd, taskPage);

    auto waitBarOnSamePageInd = waitBarOnSamePage.value();
    auto taskWithWaitBarOnSamePage = taskWithWaitBarOnSamePageOpt.value();
    _log.trace("Prev task {0} on same FIFO has wait bar {1}(page {2})", taskWithWaitBarOnSamePage, waitBarOnSamePageInd,
               getBarrierPage(waitBarOnSamePageInd));

    // TODO: E#160461 Some improvement may be done by having some heuristic on which boundary
    // task should be used. Maybe take into account timing information or queue type?

    // Get producers of barInd to update boundaryTaskWaitBarInd
    auto barProdTasks = _barrierInfo.getBarrierProducers(barInd);
    for (auto barProdTask : barProdTasks) {
        auto barProdTaskPage = _taskPageAssignment[barProdTask];
        _log.nest().trace("Bar {0} producer: {1}(page {2})", barInd, barProdTask, barProdTaskPage);
        auto barProdQueueType = _barrierInfo.getTaskQueueType(barProdTask);

        if (taskQueueType == barProdQueueType) {
            _log.nest().trace("Task and producer are on the same FIFO {0}:{1}. No need to insert dependency "
                              "from barProdTask",
                              stringifyEnum(taskQueueType.type).data(), taskQueueType.id);
            continue;
        }

        if (taskPage - barPage == 1 && barProdTaskPage == barPage) {
            // Task and producer are on previous page. Connect task directly to closest wait barrier
            // of task that is to be legalized
            //
            // Example:
            // Before:
            // Task(PageN-1) -> BarToLegalize(PageN-1) ------------------|
            //                  Bar(PageN) ->Task(PageN) -> TaskToLegalize(PageN)
            //
            // After:
            // Task(PageN-1) -> BarToLegalize(PageN-1)
            //          |-----> Bar(PageN) ->Task(PageN) -> TaskToLegalize(PageN)

            // Connect barProdTask to identified barrier but only if such dependency does not exist
            // Do not perform dependency check only if taskWithWaitBarOnSamePage is the same as taskInd, which is the
            // case when task has multiple wait barriers, as the dependency check method will give false information
            // because it will still have dependency chain through barrier that is to be legalized.
            if (taskWithWaitBarOnSamePage == taskInd ||
                !isDepFromTaskAToTaskB(barProdTask, taskWithWaitBarOnSamePage)) {
                _barrierInfo.addProducer(waitBarOnSamePageInd, barProdTask);
                _log.nest(2).trace("Add producer {0}(page {1}) to barrier {2}(page {3}) which is wait barrier of "
                                   "task {4}(page {5}) on same FIFO as task {6}(page {7})",
                                   barProdTask, barProdTaskPage, waitBarOnSamePageInd, taskPage,
                                   taskWithWaitBarOnSamePage, _taskPageAssignment[taskWithWaitBarOnSamePage], taskInd,
                                   taskPage);
            }

        } else {
            // Barrier producer task is on earlier page than wait barrier that is to be legalized
            // or wait barrier is on earlier page than previous page where task is
            // Example 1:
            //
            //   Before:
            //   Task(PageN-2) -> BarToLegalize(PageN-1) ----------------------------|
            //                                        Bar(PageN) ->Task(PageN) -> TaskToLegalize(PageN)
            //
            //   After:
            //   Task(PageN-2) -> BarToLegalize(PageN-1)
            //       |---------> BoundaryTask(N-1) -> Bar(PageN) ->Task(PageN) -> TaskToLegalize(PageN)
            //
            // Example 2:
            //
            //   Before:
            //   Task(PageN-2) -> BarToLegalize(PageN-2) ----------------------------|
            //                                        Bar(PageN) ->Task(PageN) -> TaskToLegalize(PageN)
            //
            //   After:
            //   Task(PageN-2) -> BarToLegalize(PageN-2)
            //       |---------> BoundaryTask(N-1) -> Bar(PageN) ->Task(PageN) -> TaskToLegalize(PageN)
            //

            // TODO: E#160461 Analyze if this could be improved by picking different boundary task
            auto pageBoundaryTasks = getLastBoundaryTasksForPage(barProdTaskPage + 1);
            VPUX_THROW_WHEN(pageBoundaryTasks.empty(), "No boundary tasks set for page {0}", barProdTaskPage + 1);
            auto pageBoundaryTask = pageBoundaryTasks[0];
            _log.nest(2).trace("Page boundary task: {0}(page {1})", pageBoundaryTask,
                               _taskPageAssignment[pageBoundaryTask]);
            auto pageBoundaryTasksOnSameFifoIt =
                    _firstAndLastBoundaryTaskForEachPagePerFifo[barProdTaskPage + 1].find(barProdQueueType);
            if (pageBoundaryTasksOnSameFifoIt !=
                _firstAndLastBoundaryTaskForEachPagePerFifo[barProdTaskPage + 1].end()) {
                // There is a boundary task on same FIFO. No need to insert dependency from barProdTask
                pageBoundaryTask = pageBoundaryTasksOnSameFifoIt->second.second;
                _log.nest(2).trace("Page boundary task exists on same HW FIFO. No need to insert dependency "
                                   "from barProdTask. Use {0}(page {1}) as boundary task",
                                   pageBoundaryTask, _taskPageAssignment[pageBoundaryTask]);
            } else {
                auto pageBoundaryTaskWaitBars = _barrierInfo.getWaitBarriers(pageBoundaryTask);
                auto pageBoundaryTaskWithWaitBar = pageBoundaryTask;
                if (pageBoundaryTaskWaitBars.empty()) {
                    auto prevTaskOpt = _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(pageBoundaryTask);
                    if (prevTaskOpt.has_value()) {
                        pageBoundaryTaskWaitBars = _barrierInfo.getWaitBarriers(prevTaskOpt.value());
                        pageBoundaryTaskWithWaitBar = prevTaskOpt.value();
                    }
                }

                auto boundaryTaskWaitBarInd = *pageBoundaryTaskWaitBars.begin();
                _log.nest(2).trace("Boundary task {0} closest wait bar: {1}", pageBoundaryTask, boundaryTaskWaitBarInd);

                // Add dependency from task producing into barrier to be legalized to a wait barrier
                // of boundary task. Check if such dependency already exists
                if (!isDepFromTaskAToTaskB(barProdTask, pageBoundaryTaskWithWaitBar)) {
                    _barrierInfo.addProducer(boundaryTaskWaitBarInd, barProdTask);

                    _log.nest(2).trace("Add producer {0}(page {1}) to barrier {2}(page {3})", barProdTask,
                                       _taskPageAssignment[barProdTask], boundaryTaskWaitBarInd,
                                       getBarrierPage(boundaryTaskWaitBarInd));
                }
            }
            // Add dependency from pageBoundaryTask to taskInd but only if this pageBoundaryTask is on previous
            // page and there is no such dependency in place
            if (_taskPageAssignment[pageBoundaryTask] + 1 == taskPage &&
                !isDepFromTaskAToTaskB(pageBoundaryTask, taskInd)) {
                _barrierInfo.addProducer(waitBarOnSamePageInd, pageBoundaryTask);
                _log.nest(2).trace("Add dependency from boundary task {0}(page {1}) to task wait barrier {2}(page {3})",
                                   pageBoundaryTask, _taskPageAssignment[pageBoundaryTask], waitBarOnSamePageInd,
                                   getBarrierPage(waitBarOnSamePageInd));
            }
        }
    }
    _barrierInfo.removeConsumer(barInd, taskInd);
    _log.trace("Remove consumer {0}(page {1}) from barrier {2}(page {3})", taskInd, taskPage, barInd, barPage);
}

// Legalize long dependency on update barrier for a given task. If task on PageN
// has update barrier on PageX, where X > N+1, then this dependency needs to be legalized
void vpux::VPURT::BarrierPagesSplitHandler::legalizeUpdateBarrierDependency(size_t taskInd,
                                                                            VPURT::TaskQueueType& taskQueueType,
                                                                            size_t barInd) {
    size_t taskPage = _taskPageAssignment[taskInd];

    auto barPage = getBarrierPage(barInd);
    VPUX_THROW_WHEN(barPage < taskPage, "Task {0} from page {1} updates barrier {2} from page {3}", taskInd, taskPage,
                    barInd, barPage);
    if (barPage - taskPage <= 1) {
        // Barrier is within accepted range
        // No need to legalize
        return;
    }
    _log.trace("Update bar {0}(page {1}) needs to be legalized", barInd, barPage);

    // Task on PageN produces into barrier from PageX where X > N + 1
    // Remove this dependency and make task update wait barrier of boundary task from PageN+1
    //
    // Example:
    // Before:
    // Task(PageN) --------------------------------------------> BarToLegalize(PageN+2)
    //
    // After:
    // Task(PageN)                                               BarToLegalize(PageN+2)
    //          |-----> Bar(PageN+1) -> BoundaryTask(PageN+1) ---->|

    // TODO: E#160461 Some improvement may be done by having some heuristic on which boundary
    // task should be used. Maybe take into account timing information or queue type?
    auto pageBoundaryTasks = getLastBoundaryTasksForPage(taskPage + 1);
    VPUX_THROW_WHEN(pageBoundaryTasks.empty(), "No boundary tasks set for page {0}", taskPage + 1);
    auto pageBoundaryTask = pageBoundaryTasks[0];

    auto pageBoundaryTasksOnSameFifoIt = _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage + 1].find(taskQueueType);
    if (pageBoundaryTasksOnSameFifoIt != _firstAndLastBoundaryTaskForEachPagePerFifo[taskPage + 1].end()) {
        // There is a boundary task on same FIFO. No need to insert dependency from barProdTask
        // Before adding the dependency, check for existing control path will prevent from adding this unnecessary
        // dependency
        pageBoundaryTask = pageBoundaryTasksOnSameFifoIt->second.second;
    }

    std::optional<size_t> pageBoundaryTaskWithWaitBarOpt = pageBoundaryTask;
    std::optional<size_t> pageBoundaryTaskWaitBarIndOpt = std::nullopt;

    // Start searching for a wait barrier from taskPage + 1 that is the one in this page that blocks
    // boundary task in this page. It doesn't have to be a wait barrier directly attached to this task
    // as it can have no wait barriers. It might be some previous task on same FIFO
    _log.trace("Start check from page boundary task {0}(page {1})", pageBoundaryTaskWithWaitBarOpt.value(),
               taskPage + 1);
    do {
        auto pageBoundaryTaskWaitBars = _barrierInfo.getWaitBarriers(pageBoundaryTaskWithWaitBarOpt.value());
        for (auto taskWaitBar : pageBoundaryTaskWaitBars) {
            if (getBarrierPage(taskWaitBar) == taskPage + 1) {
                pageBoundaryTaskWaitBarIndOpt = taskWaitBar;
                break;
            }
        }
        if (pageBoundaryTaskWaitBarIndOpt.has_value()) {
            break;
        }

        pageBoundaryTaskWithWaitBarOpt =
                _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(pageBoundaryTaskWithWaitBarOpt.value());

    } while (pageBoundaryTaskWithWaitBarOpt.has_value());

    VPUX_THROW_WHEN(!pageBoundaryTaskWithWaitBarOpt.has_value(), "No boundary task with valid barrier on page {0}",
                    taskPage + 1);

    auto pageBoundaryTaskWithWaitBar = pageBoundaryTaskWithWaitBarOpt.value();

    VPUX_THROW_WHEN(!pageBoundaryTaskWaitBarIndOpt.has_value(), "No wait barrier found for boundary task {0}(page {1})",
                    pageBoundaryTaskWithWaitBar, taskPage + 1);

    auto boundaryTaskWaitBarInd = pageBoundaryTaskWaitBarIndOpt.value();

    _log.trace("Use boundary task {0}(page {1}) and its wait barrier {2}(page {3})", pageBoundaryTaskWithWaitBar,
               _taskPageAssignment[pageBoundaryTaskWithWaitBar], boundaryTaskWaitBarInd,
               getBarrierPage(boundaryTaskWaitBarInd));

    // Before adding taskInd to boundary task dep, check if such dependency does not already exist
    if (!isDepFromTaskAToTaskB(taskInd, pageBoundaryTaskWithWaitBar)) {
        _barrierInfo.addProducer(boundaryTaskWaitBarInd, taskInd);
        _log.trace("Add producer {0}(page {1}) to boundary task {2}(page {3}) wait barrier {4}(page {5})", taskInd,
                   taskPage, pageBoundaryTaskWithWaitBar, _taskPageAssignment[pageBoundaryTaskWithWaitBar],
                   boundaryTaskWaitBarInd, getBarrierPage(boundaryTaskWaitBarInd));
    }

    // Before adding a dependency from boundary task to barInd, check if such dependency does not already
    // exist by checking if there is a control path between boundary task and any producer of barInd
    if (barPage == _taskPageAssignment[pageBoundaryTask] + 1 && !isDepFromTaskToBarrier(pageBoundaryTask, barInd)) {
        _barrierInfo.addProducer(barInd, pageBoundaryTask);
        _log.trace("Add producer {0}(page {1}) to barrier {2}(page {3})", pageBoundaryTask,
                   _taskPageAssignment[pageBoundaryTask], barInd, barPage);
    }

    _barrierInfo.removeProducer(barInd, taskInd);
    _log.trace("Remove producer {0}(page {1}) from barrier {2}(page {3})", taskInd, taskPage, barInd, barPage);
}

// Legalize all too long dependencies for tasks identified by getTasksWithNonAdjacentPageDependencyToLegalize
// This is core legalization method and it will modify task-barrier dependencies so that task
// updates only barriers from its and next page
// This method does not create any new barriers
void vpux::VPURT::BarrierPagesSplitHandler::legalizeNonAdjacentPageDependencies(
        SmallVector<size_t>& tasksWithDepsToLegalize) {
    _log.trace("Legalizing long deps beyond next page");
    _log = _log.nest();
    for (auto taskInd : tasksWithDepsToLegalize) {
        size_t taskPage = _taskPageAssignment[taskInd];
        auto taskQueueType = _barrierInfo.getTaskQueueType(taskInd);
        _log.trace("Task {0}(page {1}) with long dep needs to be legalized", taskInd, taskPage);
        _log = _log.nest();

        // Check if there are any wait barriers in the task that represent long dep.
        // Wait barrier page needs to match task page assignment
        auto waitBars = _barrierInfo.getWaitBarriers(taskInd);

        _log.trace("Checking wait barriers");
        for (auto barInd : waitBars) {
            legalizeWaitBarrierDependency(taskInd, taskQueueType, barInd);
        }

        // Check if there are any update barriers in the task that represent long dep.
        // Update barrier page needs to match task page (PageN) or next page (PageN+1)
        auto updateBars = _barrierInfo.getUpdateBarriers(taskInd);

        _log.trace("Checking update barriers");
        for (auto barInd : updateBars) {
            legalizeUpdateBarrierDependency(taskInd, taskQueueType, barInd);
        }
        _log = _log.unnest();
    }
    _log = _log.unnest();

    // Invalidate task control map after barrier deps have changed
    _blockIdxOfTaskControlMap = std::nullopt;
}

// Check if there is a dependency from taskA to taskB taking into account barrier dependency
// and HW FIFOs
bool vpux::VPURT::BarrierPagesSplitHandler::isDepFromTaskAToTaskB(size_t taskA, size_t taskB) {
    return _barrierInfo.isDepFromTaskAToTaskB(taskA, taskB, _taskControlMapAndOffset, _blockIdxOfTaskControlMap);
}

// Check if there is a dependency from task to barrier by checking if there is any dependency
// from this task to any producer of this barrier. If yes then there is a guarantee that task
// needs to execute before barrier is produced - there is topological dependency
bool vpux::VPURT::BarrierPagesSplitHandler::isDepFromTaskToBarrier(size_t taskInd, size_t barInd) {
    return _barrierInfo.isDepFromTaskToBarrier(taskInd, barInd, _taskControlMapAndOffset, _blockIdxOfTaskControlMap);
}

// Check if there is a dependency from barrier to task by checking if there is a dependency
// from any consumer of this barrier to this task. If yes then there is a guarantee that task
// can execute only after barrier is produced - there is topological dependency.
bool vpux::VPURT::BarrierPagesSplitHandler::isDepFromBarrierToTask(size_t barInd, size_t taskInd) {
    return _barrierInfo.isDepFromBarrierToTask(barInd, taskInd, _taskControlMapAndOffset, _blockIdxOfTaskControlMap);
}

// Check if there is a dependency from barA to barB by checking if there is any dependency
// from barA consumer to barB producer
bool vpux::VPURT::BarrierPagesSplitHandler::isDepFromBarAToBarB(size_t barA, size_t barB) {
    return _barrierInfo.isDepFromBarAToBarB(barA, barB, _taskControlMapAndOffset, _blockIdxOfTaskControlMap);
}

// Check if boundary task of consecutive pages depend on each other
// Example:
//   PageN boundary tasks:   taskA, taskB
//   PageN+1 boundary tasks: taskC, taskD
// There needs to be dependency from taskA to taskC&D and taskB to taskC&D
// This way when any of PageN+1 boundary tasks executes it is guaranteed that
// all barriers from PageN have been consumed and they are configured for PageN+2 and
// boundary tasks of PageN+1 are first tasks which produce into barriers from PageN+1
bool vpux::VPURT::BarrierPagesSplitHandler::areBoundaryTasksFromNeighborPagesDependent() {
    _log.trace("Checking if boundary tasks from neighbor pages are dependent");
    if (_pageCount <= 2) {
        return true;
    }

    // Check deps from PageN to PageN+1 boundary tasks
    for (size_t pageInd = 0; pageInd < _pageCount - 2; pageInd++) {
        VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].empty(),
                        "No boundary tasks set for page {0}", pageInd);
        VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd + 1].empty(),
                        "No boundary tasks set for page {0}", pageInd + 1);
        auto pageBoundaryTaskPerFifo = _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd];
        auto nextPageBoundaryTaskPerFifo = _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd + 1];

        for (auto [_, pageFirstLastBoundaryTask] : pageBoundaryTaskPerFifo) {
            for (auto [_, nextPageFirstLastBoundaryTask] : nextPageBoundaryTaskPerFifo) {
                auto task1 = pageFirstLastBoundaryTask.second;
                auto task2 = nextPageFirstLastBoundaryTask.first;
                if (!isDepFromTaskAToTaskB(task1, task2)) {
                    _log.trace("Boundary tasks {0} and {1} are not dependent", task1, task2);
                    return false;
                }
            }
        }
    }

    return true;
}

// Get all boundary task pairs from neighbor pages which are missing dependency in between
// If boundary task from PageN+1 misses dependency from boundary task from PageN then
// return such pair
SmallVector<std::pair<size_t, size_t>>
vpux::VPURT::BarrierPagesSplitHandler::getBoundaryTaskPairsMissingDepInBetween() {
    SmallVector<std::pair<size_t, size_t>> boundaryTaskPairsMissingDepInBetween;

    _log.trace("Getting boundary task pairs missing dep in between");
    if (_pageCount <= 2) {
        return boundaryTaskPairsMissingDepInBetween;
    }

    // Check deps from PageN to PageN+1 boundary tasks
    for (size_t pageInd = 0; pageInd < _pageCount - 2; pageInd++) {
        VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].empty(),
                        "No boundary tasks set for page {0}", pageInd);
        VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd + 1].empty(),
                        "No boundary tasks set for page {0}", pageInd + 1);
        auto pageBoundaryTaskPerFifo = _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd];
        auto nextPageBoundaryTaskPerFifo = _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd + 1];

        for (auto [_, pageFirstLastBoundaryTask] : pageBoundaryTaskPerFifo) {
            for (auto [_, nextPageFirstLastBoundaryTask] : nextPageBoundaryTaskPerFifo) {
                auto task1 = pageFirstLastBoundaryTask.second;
                auto task2 = nextPageFirstLastBoundaryTask.first;
                if (isDepFromTaskAToTaskB(task1, task2)) {
                    continue;
                }
                boundaryTaskPairsMissingDepInBetween.push_back(std::make_pair(task1, task2));
            }
        }
    }

    _log.trace("Number of boundary task pairs missing dep in between: {0}",
               boundaryTaskPairsMissingDepInBetween.size());
    return boundaryTaskPairsMissingDepInBetween;
}

// Legalize missing dependencies in between of boundary tasks from neighbor pages
// This method is meant to work on result from getBoundaryTaskPairsMissingDepInBetween
void vpux::VPURT::BarrierPagesSplitHandler::legalizeDepsForBoundaryTasks(
        SmallVector<std::pair<size_t, size_t>>& boundaryTaskPairsMissingDepInBetween) {
    _log.trace("Legalizing deps for boundary tasks");
    for (auto [taskSrc, taskDst] : boundaryTaskPairsMissingDepInBetween) {
        auto waitBars = _barrierInfo.getWaitBarriers(taskDst);
        if (waitBars.empty()) {
            auto prevTaskOpt = _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(taskDst);
            if (prevTaskOpt.has_value()) {
                waitBars = _barrierInfo.getWaitBarriers(prevTaskOpt.value());
            }
        }

        VPUX_THROW_WHEN(waitBars.empty(), "Task {0} has no wait barrier to connect to", taskDst);
        auto waitBar = *waitBars.begin();

        // insert dependency from taskSrc to wait barrier of taskDst to create taskSrc->taskDst dependency
        _log.trace("Create dependency from task {0}(page {1}) to task {2}(page {3})", taskSrc,
                   _taskPageAssignment[taskSrc], taskDst, _taskPageAssignment[taskDst]);
        _log.trace("Add producer {0}(page {1}) to barrier {2}(page {3})", taskSrc, _taskPageAssignment[taskSrc],
                   waitBar, getBarrierPage(waitBar));
        _barrierInfo.addProducer(waitBar, taskSrc);
    }

    // Invalidate task control map after barrier deps have changed
    _blockIdxOfTaskControlMap = std::nullopt;
}

bool vpux::VPURT::BarrierPagesSplitHandler::isSplitToPagesValid() {
    _log.trace("Checking if split to pages is valid");
    if (!areNoDepsGoingBeyondNeighborPage()) {
        return false;
    }

    if (!areBoundaryTasksFromNeighborPagesDependent()) {
        return false;
    }

    return true;
}

// Based on barriers that will be used as wait barriers (pageStartBars) and update barriers (pageEndBars) of barrier DMA
// find a place in IR (index of operation) after which this DMA can be inserted
std::optional<size_t> vpux::VPURT::BarrierPagesSplitHandler::getInsertionPointForDmaProgrammingBarriers(
        const BarrierInfo::TaskSet& pageStartBars, const BarrierInfo::TaskSet& pageEndBars) {
    auto startPoint = _barrierInfo.getBarriersLatestProducer(pageStartBars);
    auto endPoint = _barrierInfo.getBarriersEarliestConsumer(pageEndBars);

    _log.trace("Find insertion point between {0} and {1}", startPoint, endPoint);

    if (startPoint < endPoint) {
        // Simple case where all pageStartBars producers are before pageEndBars consumers
        // and insertion point can be set to latest producer of pageStartBars
        _log.trace("Insert BarProgDMA after task {0}", startPoint);
        return startPoint;
    }

    _log.trace("Check start and end barrier usage before picking insertion point");

    // endPoint < startPoint
    // There can be a case where pageStartBars producers are after pageEndBars consumers based on index
    // in IR, for example this can happen when there are parallel branches and there are multiple ways
    // how tasks from unrelated branches are assigned indexes and placed in IR.
    // There are two solutions
    //  - insert after startPoint if any of pageStartBars are produced by DMA on the same FIFO as BarProgDMA
    //  - insert before endPoint if any of pageEndBars are consumed by DMA on the same FIFO as BarProgDMA

    VPURT::TaskQueueType barProgDmaQueueType;
    barProgDmaQueueType.type = VPU::ExecutorKind::DMA_NN;
    barProgDmaQueueType.id = getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR);

    bool isAnyPageStartBarsProducersOnSameFifo = false;
    for (auto barInd : pageStartBars) {
        auto barProdTasks = _barrierInfo.getBarrierProducers(barInd);
        for (auto barProdTask : barProdTasks) {
            // Interested only in range [endPoint, startPoint]
            if (barProdTask < endPoint) {
                continue;
            }
            if (_barrierInfo.getTaskQueueType(barProdTask) == barProgDmaQueueType) {
                _log.trace("Task {0} which produces barrier {1} is a DMA P:0 CH:DDR", barProdTask, barInd);
                isAnyPageStartBarsProducersOnSameFifo = true;
                break;
            }
        }
        if (isAnyPageStartBarsProducersOnSameFifo) {
            break;
        }
    }

    bool isAnyPageEndBarsConsumersOnSameFifo = false;
    for (auto barInd : pageEndBars) {
        auto barConsTasks = _barrierInfo.getBarrierConsumers(barInd);
        for (auto barConsTask : barConsTasks) {
            // Interested only in range [endPoint, startPoint]
            if (barConsTask > endPoint) {
                continue;
            }
            if (_barrierInfo.getTaskQueueType(barConsTask) == barProgDmaQueueType) {
                _log.trace("Task {0} which consumes barrier {1} is a DMA P:0 CH:DDR", barConsTask, barInd);
                isAnyPageEndBarsConsumersOnSameFifo = true;
                break;
            }
        }
        if (isAnyPageEndBarsConsumersOnSameFifo) {
            break;
        }
    }

    // No valid insertion point was found with provided set of page start and end barriers
    if (isAnyPageEndBarsConsumersOnSameFifo && isAnyPageStartBarsProducersOnSameFifo) {
        _log.trace("Both pageStartBars and pageEndBars have producers/consumers on the same FIFO as BarProgDMA");
        return std::nullopt;
    }

    if (isAnyPageEndBarsConsumersOnSameFifo) {
        _log.trace("Insert BarProgDMA before task {0}", endPoint);
        return endPoint - 1;
    }

    _log.trace("Insert BarProgDMA after task {0}", startPoint);

    return startPoint;
}

// In case page crosses control block boundary then creating a barrier DMA that will wait
// on barriers from one block and update barriers of other block would violate control block split.
// To prevent from that adjust start barriers or end barriers if page crosses control block boundary
// Treat control block sync task as a start or end point for barrier DMA legalization purpose
void vpux::VPURT::BarrierPagesSplitHandler::adjustPageStartAndEndPointsIfOnBlockBoundary(
        BarrierInfo::TaskSet& pageStartBars, BarrierInfo::TaskSet& pageStartTasks, BarrierInfo::TaskSet& pageEndBars,
        BarrierInfo::TaskSet& pageEndAllBars, BarrierInfo::TaskSet& pageEndTasks) {
    // Barrier programming DMA to be inserted after page boundary tasks so that it is the first task on the page
    auto initialInsertionPosition = *std::max_element(pageStartTasks.begin(), pageStartTasks.end());
    // Use initialInsertionPosition task + 1 because bar programming DMA is inserted after this task
    auto barProgDmaBlockInd = _barrierInfo.getControlGraphBlockIndex(initialInsertionPosition + 1);
    size_t pageEndBarBlockInd =
            _barrierInfo.getControlGraphBlockIndex(*std::min_element(pageEndTasks.begin(), pageEndTasks.end()));
    if (barProgDmaBlockInd < pageEndBarBlockInd) {
        // If page end is on another control block then treat control block sync task as page boundary task
        auto syncTaskOpt = _barrierInfo.getControlGraphSyncPoint(initialInsertionPosition);
        VPUX_THROW_UNLESS(syncTaskOpt.has_value(), "No control block sync task for task {0}", initialInsertionPosition);

        auto syncTaskWaitBars = _barrierInfo.getWaitBarriers(syncTaskOpt.value());

        if (syncTaskWaitBars.size() == 1 && pageStartBars == syncTaskWaitBars) {
            // If sync task is the only consumer of page start barriers then use it as start point
            // for legalization - Control block sync point will be page start task
            _log.trace("Control block boundary: {0} and {1}. Use sync task {2} as startpoint for legalization",
                       barProgDmaBlockInd, pageEndBarBlockInd, syncTaskOpt.value());

            auto syncTaskUpdateBars = _barrierInfo.getUpdateBarriers(syncTaskOpt.value());
            VPUX_THROW_UNLESS(!syncTaskUpdateBars.empty(), "Control block sync task {0} has no update barriers",
                              syncTaskOpt.value());
            auto newPageStartBar = *std::min_element(syncTaskUpdateBars.begin(), syncTaskUpdateBars.end());

            pageStartBars.clear();
            pageStartTasks.clear();
            pageStartBars.insert(newPageStartBar);
            pageStartTasks.insert(syncTaskOpt.value());
        } else {
            // Use control block sync point as page end task
            _log.trace("Control block boundary: {0} and {1}. Use sync task {2} as endpoint for legalization",
                       barProgDmaBlockInd, pageEndBarBlockInd, syncTaskOpt.value());

            VPUX_THROW_UNLESS(!syncTaskWaitBars.empty(), "Control block sync task {0} has no wait barriers",
                              syncTaskOpt.value());
            auto newPageEndBar = *std::max_element(syncTaskWaitBars.begin(), syncTaskWaitBars.end());
            pageEndBars.clear();
            pageEndAllBars.clear();
            pageEndTasks.clear();
            pageEndBars.insert(newPageEndBar);
            pageEndAllBars.insert(newPageEndBar);
            pageEndTasks.insert(syncTaskOpt.value());
        }
    }
}

BarrierInfo::TaskSet vpux::VPURT::BarrierPagesSplitHandler::getPageStartBarsDependingOnPageEndBars(
        BarrierInfo::TaskSet& pageStartBars, BarrierInfo::TaskSet& pageEndBars) {
    BarrierInfo::TaskSet pageStartBarsToLegalize;
    auto pageStartBarsToCheck = pageStartBars;
    for (auto pageStartBar : pageStartBars) {
        for (auto pageEndBar : pageEndBars) {
            if (!isDepFromBarAToBarB(pageEndBar, pageStartBar)) {
                continue;
            }
            _log.nest().trace("There is dep from page end bar {0} to page start bar {1}", pageEndBar, pageStartBar);
            pageStartBarsToLegalize.insert(pageStartBar);
            pageStartBarsToCheck.erase(pageStartBar);
            break;
        }
    }

    if (pageStartBarsToCheck.empty()) {
        return pageStartBarsToLegalize;
    }

    // Check if previous DMA on Port 0 Channel DDR does not depend on page end barriers
    // This is to verify if with given set of start and end barriers and determined
    // insertion point, dependency of page end bar on page start bar will not appear after
    // Barrier Programming DMA is inserted due to created FIFO dependency with previous DMA
    // If such dependency is created move largest index start barrier to legalization set
    // and reiterate to check again
    auto pageStartBarMax = *std::max_element(pageStartBarsToCheck.begin(), pageStartBarsToCheck.end());
    auto pageEndBarMin = *std::min_element(pageEndBars.begin(), pageEndBars.end());
    const VPURT::TaskQueueType dmaP0ChDdrQueueType = {VPU::ExecutorKind::DMA_NN,
                                                      getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR)};
    while (pageStartBarMax > pageEndBarMin) {
        auto insertionPointOpt = getInsertionPointForDmaProgrammingBarriers(pageStartBarsToCheck, pageEndBars);
        if (!insertionPointOpt.has_value()) {
            _log.nest().trace("No insertion point found. Legalize page start bar {0}", pageStartBarMax);
            pageStartBarsToCheck.erase(pageStartBarMax);
            VPUX_THROW_UNLESS(!pageStartBarsToCheck.empty(), "No page start bars left to check");
            pageStartBarMax = *std::max_element(pageStartBarsToCheck.begin(), pageStartBarsToCheck.end());
            continue;
        }

        _log.nest().trace("Check if there is dependency from pageEndBars to previous task on same FIFO based on "
                          "insertion point {0}",
                          insertionPointOpt.value());

        size_t taskOnSameFifoAsBarProgDma;
        if (_barrierInfo.getTaskQueueType(insertionPointOpt.value()) == dmaP0ChDdrQueueType) {
            taskOnSameFifoAsBarProgDma = insertionPointOpt.value();
        } else {
            auto prevTaskOnSameFifoAsBarProgDmaOpt =
                    _barrierInfo.getPrevTaskOnQueue(insertionPointOpt.value(), dmaP0ChDdrQueueType);

            if (!prevTaskOnSameFifoAsBarProgDmaOpt.has_value()) {
                break;
            }
            taskOnSameFifoAsBarProgDma = prevTaskOnSameFifoAsBarProgDmaOpt.value();
        }

        bool isDep = false;
        for (auto pageEndBar : pageEndBars) {
            if (isDepFromBarrierToTask(pageEndBar, taskOnSameFifoAsBarProgDma)) {
                isDep = true;
                _log.nest().trace("There is dep from page end bar {0} to task {1}. Legalize page start bar {2}",
                                  pageEndBar, taskOnSameFifoAsBarProgDma, pageStartBarMax);
                pageStartBarsToLegalize.insert(pageStartBarMax);

                pageStartBarsToCheck.erase(pageStartBarMax);
                break;
            }
        }

        if (!isDep || pageStartBarsToCheck.empty()) {
            break;
        }

        pageStartBarMax = *std::max_element(pageStartBarsToCheck.begin(), pageStartBarsToCheck.end());
    }

    return pageStartBarsToLegalize;
}

// Check if any pageStartBar depends on any pageEndBar. Such pageStartBar cannot be used as a starting point
// for a barrier programming DMA which will also produce into pageEndBars. This function will identify such barrier,
// remove it from pageStartBars set and add necessary dependency from boundary task to other, valid pageStartBar
void vpux::VPURT::BarrierPagesSplitHandler::legalizePageStartBarsDependingOnPageEndBars(
        BarrierInfo::TaskSet& pageStartTasks, BarrierInfo::TaskSet& pageStartBars,
        BarrierInfo::TaskSet& pageStartBarsToLegalize) {
    _log.trace("Legalize page start bars depending on page end bars");

    // Pick some other pageStartBar that will be used for legalization
    VPUX_THROW_UNLESS(!pageStartBars.empty(), "No page start bars to use for legalization");
    auto startBarToUseForLegalization = *std::max_element(pageStartBars.begin(), pageStartBars.end());

    for (auto pageStartBarToLegalize : pageStartBarsToLegalize) {
        // Legalize by identifying boundary tasks that were producers of pageStartBarToLegalize
        // and make them update other pageStartBar - startBarToUseForLegalization
        _log.trace("Legalizing page start bar {0} which cannot be used as wait barrier for DMA",
                   pageStartBarToLegalize);

        // Find all boundary tasks that is producers of pageStartBarToLegalize
        auto startBarProducersToBeLegalized = to_small_vector(_barrierInfo.getBarrierProducers(pageStartBarToLegalize));

        // Leave only producers which are boundary tasks
        startBarProducersToBeLegalized.erase(llvm::remove_if(startBarProducersToBeLegalized,
                                                             [&](auto taskInd) {
                                                                 return llvm::find(pageStartTasks, taskInd) ==
                                                                        pageStartTasks.end();
                                                             }),
                                             startBarProducersToBeLegalized.end());

        for (auto startBarProducerToBeLegalized : startBarProducersToBeLegalized) {
            _barrierInfo.addProducer(startBarToUseForLegalization, startBarProducerToBeLegalized);
            _log.trace("Add producer {0} to barrier {1}", startBarProducerToBeLegalized, startBarToUseForLegalization);
        }
    }
}

// Get boundary tasks (last on each HW FIFO) from pageInd-1 to pageInd and get barriers they update
// This information will be used as starting point for a page - mark the start
// of barrier DMA
void vpux::VPURT::BarrierPagesSplitHandler::getPageStartTasksAndBars(size_t pageInd,
                                                                     BarrierInfo::TaskSet& pageStartTasks,
                                                                     BarrierInfo::TaskSet& pageStartBars) {
    VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd - 1].empty(),
                    "No boundary tasks set for page {0}", pageInd - 1);
    for (auto& [taskQueueType, firstLastTaskInd] : _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd - 1]) {
        _log.trace("Get page start tasks and bars for queue {0}:{1}", stringifyEnum(taskQueueType.type).data(),
                   taskQueueType.id);
        auto pageStartTask = firstLastTaskInd.second;
        auto taskUpdateBarsVec = to_small_vector(_barrierInfo.getUpdateBarriers(pageStartTask));

        // Remove barriers that are not on this page - they can be from previous page
        taskUpdateBarsVec.erase(llvm::remove_if(taskUpdateBarsVec,
                                                [&](auto barInd) {
                                                    return getBarrierPage(barInd) != pageInd;
                                                }),
                                taskUpdateBarsVec.end());

        VPUX_THROW_UNLESS(!taskUpdateBarsVec.empty(), "Page start task {0} has no update barriers on page {1}",
                          pageStartTask, pageInd);

        // If task updates multiple barriers, pick only one with smallest index
        // No need to use more barriers as 1 barrier is enough to know that boundary task have finished
        auto startBar = *std::min_element(taskUpdateBarsVec.begin(), taskUpdateBarsVec.end());
        pageStartBars.insert(startBar);
        pageStartTasks.insert(pageStartTask);
        _log.nest().trace("Page start task {0}, start bar {1}", pageStartTask, startBar);
    }
}

// Get boundary tasks from (first on each HW FIFO) pageInd to pageInd+1 and get barriers they wait on
// Get also remaining wait barriers for other boundary tasks on each FIFO for this page
// This information will be used as end point for a page - mark the end
// of barrier DMA
void vpux::VPURT::BarrierPagesSplitHandler::getPageEndTasksAndBars(size_t pageInd, BarrierInfo::TaskSet& pageEndTasks,
                                                                   BarrierInfo::TaskSet& pageEndBars,
                                                                   BarrierInfo::TaskSet& pageEndAllBars) {
    VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].empty(), "No boundary tasks set for page {0}",
                    pageInd);
    for (auto& [taskQueueType, firstLastTaskInd] : _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd]) {
        auto pageEndTask = firstLastTaskInd.first;
        auto taskWaitBars = _barrierInfo.getWaitBarriers(pageEndTask);

        _log.trace("Get page end tasks and bars for queue {0}:{1} start from task {2}",
                   stringifyEnum(taskQueueType.type).data(), taskQueueType.id, pageEndTask);

        if (taskWaitBars.empty()) {
            auto taskWithWaitBarOnSamePageOpt = _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(pageEndTask);
            VPUX_THROW_UNLESS(taskWithWaitBarOnSamePageOpt.has_value(), "No wait barriers for task {0}(page {1})",
                              pageEndTask, _taskPageAssignment[pageEndTask]);

            pageEndTask = taskWithWaitBarOnSamePageOpt.value();
            taskWaitBars = _barrierInfo.getWaitBarriers(pageEndTask);
        }
        // If task wait on multiple barriers, pick only one with latest index
        // No need to use more barriers as 1 barrier is enough to block next boundary task
        auto endBar = *std::max_element(taskWaitBars.begin(), taskWaitBars.end());
        pageEndBars.insert(endBar);
        pageEndTasks.insert(pageEndTask);
        _log.nest().trace("Page end task {0}, end bar {1}", pageEndTask, endBar);

        // Get all wait barriers that are used by boundary tasks
        // Scan whole range of boundary task on each FIFO to get all wait barriers
        std::optional<size_t> currentTaskOpt = pageEndTask;
        auto lastTask = firstLastTaskInd.second;
        while (currentTaskOpt.has_value() && currentTaskOpt.value() <= lastTask) {
            auto taskWaitBars = _barrierInfo.getWaitBarriers(currentTaskOpt.value());
            if (!taskWaitBars.empty()) {
                _log.nest().trace("Page end task {0}, end bars {1}", currentTaskOpt.value(),
                                  to_small_vector(taskWaitBars));
                pageEndAllBars.insert(taskWaitBars.begin(), taskWaitBars.end());
            }
            currentTaskOpt = _barrierInfo.getNextTaskOnSameQueue(currentTaskOpt.value());
        }
    }
}

// Additional legalization is needed if compiler intends to prepare barrier programming DMAs
// Page split needs to be adjusted to create place in schedule with barriers that mark page start and page end
// and allow for insertion of a barrier programming DMA that will run during execution of PageN
// and is guaranteed to start after all barriers from PageN-1 are consumed and will finish before
// any task producing barrier to PageN+1 start.
// Pages which will be legalized for barrier DMAs depend on Barrier FIFO depth.
// Expectation is following in case Barrier FIFO depth = 4:
// -------------------------------------------------------------------------------------
// Barrier DMA executed at: |	Physical barriers programmed | Barrier pages programmed
// -------------------------------------------------------------------------------------
//   Bootstrap	            |  [0, N-1]     (all)	         | Page0-Page7
//   Page7	                |  [0, (N/2)-1] (1st half)	     | Page8/10/12/14
//   Page8	                |  [(N/2), N-1] (2nd half)	     | Page9/11/13/15
//   Page15	                |  [0, (N/2)-1] (1st half)	     | Page16/18/20/22
//   Page16	                |  [(N/2), N-1] (2nd half)	     | Page17/19/21/23
//
void vpux::VPURT::BarrierPagesSplitHandler::legalizeForDmaProgrammingBarriers() {
    size_t barrierFifoDepthInPages = 2 * _barrierFifoDepth;
    if (_pageCount <= barrierFifoDepthInPages) {
        // No need to do any legalization as all barriers will be configured at bootstrap
        _log.trace("No need to legalize for barrier programming DMAs due to small number of pages ({0}) compared to "
                   "barrier FIFO depth ({1})",
                   _pageCount, _barrierFifoDepth);
        return;
    }

    _barProgDmaPosVec.resize(_pageCount - 1);

    auto setBarProgDmaData = [&](size_t pageInd, const BarrierInfo::TaskSet& pageStartBars,
                                 const BarrierInfo::TaskSet& pageEndBars) {
        auto insertionPointOpt = getInsertionPointForDmaProgrammingBarriers(pageStartBars, pageEndBars);
        VPUX_THROW_UNLESS(insertionPointOpt.has_value(),
                          "No insertion point for barrier programming DMA for page {0} found", pageInd);
        _barProgDmaPosVec[pageInd] = {true, SmallVector<size_t>(pageStartBars.begin(), pageStartBars.end()),
                                      SmallVector<size_t>(pageEndBars.begin(), pageEndBars.end()),
                                      insertionPointOpt.value()};

        llvm::sort(_barProgDmaPosVec[pageInd].waitBars);
        llvm::sort(_barProgDmaPosVec[pageInd].updateBars);
    };

    // pageInd refers to page that will contain a barrier programming DMA for pageInd+1
    // Initial pages are expected to be programmed at bootstrap
    for (size_t pageInd = barrierFifoDepthInPages - 1; pageInd < _pageCount - 1; pageInd++) {
        if (pageInd % barrierFifoDepthInPages != 0 && (pageInd + 1) % barrierFifoDepthInPages != 0) {
            // Barrier programming DMA is not needed in this page
            continue;
        }

        _log.trace("Legalizing for barrier programming DMA for page {0}", pageInd);
        _log = _log.nest();
        // Get boundary tasks to identify which barriers mark the start and end of page
        // Those barriers will be used as an indication for where the barrier DMA needs to be placed

        // Get boundary tasks from pageInd-1 to pageInd and get barriers they update
        // Those barriers will mark the start of DMA
        BarrierInfo::TaskSet pageStartTasks;
        BarrierInfo::TaskSet pageStartBars;
        getPageStartTasksAndBars(pageInd, pageStartTasks, pageStartBars);
        VPUX_THROW_WHEN(pageStartBars.empty(), "No start barriers for page {0} identified", pageInd);

        // Get boundary tasks (first on each HW FIFO) from pageInd and get barriers they wait on
        // Those barriers will mark the end of DMA
        // Get also remaining wait barriers for other boundary tasks on each FIFO for this page
        BarrierInfo::TaskSet pageEndTasks;
        BarrierInfo::TaskSet pageEndBars;
        BarrierInfo::TaskSet pageEndAllBars;
        getPageEndTasksAndBars(pageInd, pageEndTasks, pageEndBars, pageEndAllBars);
        VPUX_THROW_WHEN(pageEndBars.empty(), "No end barriers for page {0} identified", pageInd);

        // Adjust start barriers or end barriers if page crosses control block boundary
        // In such case treat control block sync task as a start or end point for barrier DMA
        // legalization purpose
        adjustPageStartAndEndPointsIfOnBlockBoundary(pageStartBars, pageStartTasks, pageEndBars, pageEndAllBars,
                                                     pageEndTasks);

        // Check if there are any common barriers between pageStartBars and pageEndBars
        auto commonStartEndBars = llvm::set_intersection(pageStartBars, pageEndAllBars);

        // Common barriers need to be legalized and related task dependencies updated
        // If there is already other start barrier not part of common barrier, make all common barriers
        // to be end barriers
        auto pageStartOnlyBars = llvm::set_difference(pageStartBars, commonStartEndBars);

        // Check if any pageStartBar depends on any pageEndBar. In that case it cannot be used
        // as wait barrier for barrier DMA
        BarrierInfo::TaskSet pageStartBarsToLegalize =
                getPageStartBarsDependingOnPageEndBars(pageStartOnlyBars, pageEndBars);

        llvm::set_subtract(pageStartOnlyBars, pageStartBarsToLegalize);

        // During legalization dependencies are modified and boundary tasks data may change
        // If tasks and barriers are from different pages, store this information so that
        // boundary data for this page can be updated and barrier programming DMA legalization
        // on next page can have up to date info about boundary tasks
        mlir::DenseSet<size_t> pagesWithPossibleBoundaryTasksChanged;
        auto barrierDepsChangedHandler = [&](size_t barInd, size_t taskInd) {
            auto barPage = getBarrierPage(barInd);
            auto taskPage = _taskPageAssignment[taskInd];
            if (barPage != taskPage) {
                pagesWithPossibleBoundaryTasksChanged.insert(taskPage);
                pagesWithPossibleBoundaryTasksChanged.insert(barPage);
            }
        };

        _log.trace("Page start bars {0}", to_small_vector(pageStartBars));
        _log.trace("Page end bars {0}", to_small_vector(pageEndBars));
        _log.trace("Page end all bars {0}", to_small_vector(pageEndAllBars));
        _log.trace("Page start only bars {0}", to_small_vector(pageStartOnlyBars));
        _log.trace("Page start bars to legalize {0}", to_small_vector(pageStartBarsToLegalize));
        _log.trace("Common start end bars {0}", to_small_vector(commonStartEndBars));

        if (!pageStartOnlyBars.empty()) {
            legalizePageStartBarsDependingOnPageEndBars(pageStartTasks, pageStartOnlyBars, pageStartBarsToLegalize);

            // ----------------------------------------
            // Case 1: No common start and end barriers
            // ----------------------------------------
            if (commonStartEndBars.empty()) {
                _log.trace("No common start and end barriers");

                setBarProgDmaData(pageInd, pageStartOnlyBars, pageEndBars);
                _log = _log.unnest();
                continue;
            }

            _log.trace("Legalizing common start and end barriers - count {0}", commonStartEndBars.size());

            // ------------------------------------------------------------------
            // Case 2: There are barriers which can be treated as start barriers
            // ------------------------------------------------------------------

            // There are barriers which are start only barriers, but some need to be legalized
            // as they are also used as end barriers
            _log.trace("There is at least one start barrier. Page start only bars {0}",
                       to_small_vector(pageStartOnlyBars));

            // All boundary task producers of common barriers need to be updated to program one of start only barriers
            // so that on those barriers all tasks from previous page finished and we are ready to reprogram them

            // Get boundary tasks producing into commonStartEndBars
            BarrierInfo::TaskSet commonStartEndBarBoundaryTaskProducers;
            for (auto commonStartEndBar : commonStartEndBars) {
                auto commonStartEndBarProducers = _barrierInfo.getBarrierProducers(commonStartEndBar);
                for (auto commonStartEndBarProducer : commonStartEndBarProducers) {
                    if (llvm::find(pageStartTasks, commonStartEndBarProducer) != pageStartTasks.end()) {
                        commonStartEndBarBoundaryTaskProducers.insert(commonStartEndBarProducer);
                    }
                }
            }

            // Pick one of start only barriers to be used to legalize those boundary tasks
            // TODO: Maybe some heuristic could be used to pick the best barrier for a task?
            auto startBarForLegalization = *std::max_element(pageStartOnlyBars.begin(), pageStartOnlyBars.end());

            for (auto commonStartEndBarBoundaryTaskProducer : commonStartEndBarBoundaryTaskProducers) {
                _barrierInfo.addProducer(startBarForLegalization, commonStartEndBarBoundaryTaskProducer);
                barrierDepsChangedHandler(startBarForLegalization, commonStartEndBarBoundaryTaskProducer);
                _log.trace("Add producer {0} to barrier {1}", commonStartEndBarBoundaryTaskProducer,
                           startBarForLegalization);
            }

            for (auto pageWithPossibleBoundaryTasksChanged : pagesWithPossibleBoundaryTasksChanged) {
                updateBoundaryTasksDataForPage(pageWithPossibleBoundaryTasksChanged);
            }

            _log.trace("Legalization completed");
            setBarProgDmaData(pageInd, pageStartOnlyBars, pageEndBars);
            _log = _log.unnest();
            continue;
        }

        // ----------------------------------------
        // Case 3: There are no start only barriers
        // ----------------------------------------

        _log.trace("No start only bars");

        const auto minPageEndBar = *std::min_element(pageEndBars.begin(), pageEndBars.end());
        auto newStartBar = minPageEndBar;
        if (commonStartEndBars.empty()) {
            _log.trace("commonStartEndBars is empty for page {0} - will use earliest barrier in page as start",
                       pageInd);
        } else {
            newStartBar = *std::min_element(commonStartEndBars.begin(), commonStartEndBars.end());
            _log.trace("Selected newStartBar {0} from commonStartEndBars for page {1}", newStartBar, pageInd);
        }

        if (!commonStartEndBars.empty() && newStartBar <= minPageEndBar && pageEndAllBars.size() > 1) {
            // ----------------------------------------
            // Case 3a: Use start barrier from common start and end barriers
            // ----------------------------------------
            // If there is no start only bars then pick one from the set of common start and end bars, treat it as
            // start only barrier dependencies for tasks using this barrier
            _log.trace("New start bar {0} picked from common start/end bars", newStartBar);
        } else {
            // ----------------------------------------
            // Case 3b: Create new start bar from earliest barrier in the page
            // ----------------------------------------
            // In case newStartBar has larger index than any pageEndBar then there
            // is a risk newStartBar depends on pageEndBar. To simplify legalization in this case
            // just pick newStartBar as earliest barrier in page as a safe choice to be used as a start barrier
            // If this barrier is also end barrier will need to legalize
            while (getBarrierPage(newStartBar - 1) == pageInd) {
                newStartBar--;
            }

            _log.trace("New start bar {0} which is earliest barrier in page", newStartBar);
        }

        if (pageEndBars.contains(newStartBar)) {
            // Perform legalization of start barrier in case it is also end barrier
            //
            //  boundaryTask0PageN-1                     boundaryTask0PageN-1
            //     |                                              |
            //  startEndBar -.......->  someTask            newStartBar -......-> someTask
            //     |                      |                                         |
            //     |                  otherEndBar   =>                         otherEndBar
            //     |                     |                                      |        |
            //  boundaryTask1PageN  boundaryTask2PageN          boundaryTask1PageN  boundaryTask2PageN
            //

            // Remove barrier from end bars
            pageEndBars.erase(newStartBar);
            pageEndAllBars.erase(newStartBar);

            // Find all boundary tasks that is consumers of newStartBar
            auto boundaryTasksToBeLegalized = to_small_vector(_barrierInfo.getBarrierConsumers(newStartBar));
            // Leave only consumers which are boundary tasks
            boundaryTasksToBeLegalized.erase(llvm::remove_if(boundaryTasksToBeLegalized,
                                                             [&](auto taskInd) {
                                                                 return llvm::find(pageEndTasks, taskInd) ==
                                                                        pageEndTasks.end();
                                                             }),
                                             boundaryTasksToBeLegalized.end());

            // Remove boundary task consumers from start barrier
            // and reattach them to some other end barrier
            VPUX_THROW_WHEN(pageEndAllBars.empty(), "No end barriers to attach boundary tasks to");
            auto endBar = *std::min_element(pageEndAllBars.begin(), pageEndAllBars.end());

            _log.trace("End barrier for legalization {0}", endBar);

            while (!boundaryTasksToBeLegalized.empty()) {
                _log.trace("Boundary tasks dependent on start barrier that need to be legalized {0}",
                           boundaryTasksToBeLegalized);

                BarrierInfo::TaskSet startBarConsumersWhichCannotConsumeEndBar;
                for (auto startBarConsumer : boundaryTasksToBeLegalized) {
                    // Check if boundary task to be legalized also updates endBar or endBar production depend on it
                    // In such case it cannot be consumer of endBar and has to be legalized in a different way
                    if (_barrierInfo.getUpdateBarriers(startBarConsumer).contains(endBar) ||
                        isDepFromTaskToBarrier(startBarConsumer, endBar)) {
                        startBarConsumersWhichCannotConsumeEndBar.insert(startBarConsumer);
                        continue;
                    }

                    _barrierInfo.removeConsumer(newStartBar, startBarConsumer);
                    barrierDepsChangedHandler(newStartBar, startBarConsumer);
                    _log.trace("Remove consumer {0} from barrier {1}", startBarConsumer, newStartBar);
                    _barrierInfo.addConsumer(endBar, startBarConsumer);
                    barrierDepsChangedHandler(endBar, startBarConsumer);
                    _log.trace("Add consumer {0} to barrier {1}", startBarConsumer, endBar);
                }
                pageEndBars.insert(endBar);

                auto endBarConsumer = _barrierInfo.getBarrierEarliestConsumer(endBar);
                _log.trace("End barrier earliest consumer {0}", endBarConsumer);

                // Legalize tasks which are boundary tasks and also update endBar
                // Get barriers from next page produced by such task and have endBarConsumer
                // update those barriers
                //
                //    boundaryTask -> endBar -> endBarConsumer
                //               \-------------------------------> barFromNextPage
                //
                //  Legalize to:
                //
                //    boundaryTask -> endBar -> endBarConsumer
                //                                           \---> barFromNextPage
                //
                _log.trace("New start barrier consumers which cannot consume end barrier - {0}",
                           to_small_vector(startBarConsumersWhichCannotConsumeEndBar));

                for (auto startBarConsumer : startBarConsumersWhichCannotConsumeEndBar) {
                    auto updBarsToLegalizeVec = to_small_vector(_barrierInfo.getUpdateBarriers(startBarConsumer));
                    // Leave only barriers which are from next pages as only those need to be legalize
                    // and updated by other boundary task - endBarConsumer
                    updBarsToLegalizeVec.erase(llvm::remove_if(updBarsToLegalizeVec,
                                                               [&](auto barInd) {
                                                                   return getBarrierPage(barInd) == pageInd;
                                                               }),
                                               updBarsToLegalizeVec.end());

                    for (auto updBarToLegalize : updBarsToLegalizeVec) {
                        _barrierInfo.addProducer(updBarToLegalize, endBarConsumer);
                        barrierDepsChangedHandler(updBarToLegalize, endBarConsumer);
                        _log.trace("Add producer {0} to barrier {1}", endBarConsumer, updBarToLegalize);
                        _barrierInfo.removeProducer(updBarToLegalize, startBarConsumer);
                        barrierDepsChangedHandler(updBarToLegalize, startBarConsumer);
                        _log.trace("Remove producer {0} from barrier {1}", startBarConsumer, updBarToLegalize);
                    }
                    // Such task is now no longer a boundary task
                    pageEndTasks.erase(startBarConsumer);
                }

                // Check next boundary task on the FIFO. They need to be checked if legalization is needed
                // same as previous boundary tasks. All of them need to be guarded by BarProgDma or not have
                // any update barrier in next page
                SmallVector<size_t> nextBatchOfBoundaryTasksToBeLegalized;
                for (auto startBarConsumerWhichCannotConsumeEndBar : startBarConsumersWhichCannotConsumeEndBar) {
                    auto nextTaskOnSameQueueOpt =
                            _barrierInfo.getNextTaskOnSameQueue(startBarConsumerWhichCannotConsumeEndBar);
                    if (nextTaskOnSameQueueOpt.has_value() &&
                        _taskPageAssignment[nextTaskOnSameQueueOpt.value()] == pageInd) {
                        // If next task on same queue with same page is also a boundary task
                        nextBatchOfBoundaryTasksToBeLegalized.push_back(nextTaskOnSameQueueOpt.value());
                    }
                }
                boundaryTasksToBeLegalized = std::move(nextBatchOfBoundaryTasksToBeLegalized);
            }
        }

        _log.trace(" Add dependencies for page start task to new start barrier");
        // All start boundary tasks should update newStartBar so that it is guaranteed that
        // on this barrier all tasks from previous page finished and we are ready to reprogram those barriers
        for (auto pageStartTask : pageStartTasks) {
            _log.trace("Add producer {0} to barrier {1}", pageStartTask, newStartBar);
            _barrierInfo.addProducer(newStartBar, pageStartTask);
            barrierDepsChangedHandler(newStartBar, pageStartTask);
        }

        for (auto pageWithPossibleBoundaryTasksChanged : pagesWithPossibleBoundaryTasksChanged) {
            updateBoundaryTasksDataForPage(pageWithPossibleBoundaryTasksChanged);
        }

        _log.trace("Legalization completed");
        BarrierInfo::TaskSet newPageStartBarSet;
        newPageStartBarSet.insert(newStartBar);
        setBarProgDmaData(pageInd, newPageStartBarSet, pageEndBars);

        _log = _log.unnest();
    }
}

vpux::VPURT::BarrierPagesSplitHandler::DmaProgrammingBarrierPosition
vpux::VPURT::BarrierPagesSplitHandler::getDmaProgrammingBarrierPosition(size_t pageInd) {
    VPUX_THROW_WHEN(_barProgDmaPosVec.empty(), "Barrier programming DMA positions not set");
    VPUX_THROW_UNLESS(pageInd < _pageCount - 1, "Page index {0} not within limit {1}", pageInd, _pageCount - 1);
    VPUX_THROW_WHEN(pageInd < 2 * _barrierFifoDepth - 1, "Page index {0} is expected to be programmed at bootstrap",
                    pageInd);

    return _barProgDmaPosVec[pageInd];
}

SmallVector<vpux::VPURT::BarrierPagesSplitHandler::DmaProgrammingBarrierPosition>
vpux::VPURT::BarrierPagesSplitHandler::getDmaProgrammingBarrierPositions() {
    return _barProgDmaPosVec;
}

// After inserting operations (dummy DMA, BarProgDMA) some subsequent tasks without wait barrier might need to have
// their page assignment updated so that wlmPage index never decrements on same FIFO when looking into next task. In
// case task has wait barrier it needs to be moved in IR earlier as its page assignment cannot be changed.
void vpux::VPURT::BarrierPagesSplitHandler::updateTaskPageAssignmentForQueue(size_t startTaskIndex, size_t newPageIndex,
                                                                             VPURT::TaskQueueType queueType,
                                                                             VPURT::TaskOp moveBeforeOp) {
    for (size_t taskInd = startTaskIndex; taskInd < _taskPageAssignment.size(); taskInd++) {
        if (_barrierInfo.getTaskQueueType(taskInd) != queueType) {
            // Skip tasks that are not on the same queue type
            continue;
        }
        if (_taskPageAssignment[taskInd] >= newPageIndex) {
            // stop further traversal as all following task will have desired page index
            break;
        }

        auto taskOp = _barrierInfo.getTaskOpAtIndex(taskInd);
        // Check if task is already before moveBeforeOp as it might have already been moved
        // and checking just indexes may give false information about task position in IR
        if (taskOp->isBeforeInBlock(moveBeforeOp)) {
            continue;
        }

        if (!_barrierInfo.getWaitBarriers(taskInd).empty()) {
            // If task has wait barriers then it cannot be reassigned to a different page
            _log.trace("Task {0} with wait barriers needs to be relocated. Move it before {1}", taskInd,
                       moveBeforeOp->getLoc());
            taskOp->moveBefore(moveBeforeOp);
            continue;
        }

        _log.trace("Update task {0} page assignment from {1} to {2} for queue {3}:{4}", taskInd,
                   _taskPageAssignment[taskInd], newPageIndex, stringifyEnum(queueType.type).data(), queueType.id);
        _taskPageAssignment[taskInd] = newPageIndex;

        taskOp.setWlmPage(newPageIndex);
    }
}

// Identify pages which have only single barrier in them, except last
// page which can have single barrier (final barrier) only.
// Return those barriers
SmallVector<size_t> VPURT::BarrierPagesSplitHandler::getBarrierOfSingleBarrierPages() {
    SmallVector<size_t> barrierOfSingleBarrierPagesVec;
    SmallVector<size_t> barrierCountPerPage(_pageCount);
    SmallVector<size_t> lastBarrierPerPage(_pageCount);

    for (size_t barInd = 0; barInd < _barrierInfo.getNumOfBarrierOps(); barInd++) {
        auto barPage = getBarrierPage(barInd);
        barrierCountPerPage[barPage]++;
        lastBarrierPerPage[barPage] = barInd;
    }

    for (size_t pageInd = 0; pageInd < _pageCount - 1; pageInd++) {
        if (barrierCountPerPage[pageInd] == 1) {
            barrierOfSingleBarrierPagesVec.push_back(lastBarrierPerPage[pageInd]);
        }
    }

    return barrierOfSingleBarrierPagesVec;
}

// Check if dummy DMA is needed in case there is no DMA of this type in this page
// or if existing DMA does not meet the requirements - DMA depends on all previous page
// boundary tasks to guarantee that all barriers from previous page are fully consumed
bool VPURT::BarrierPagesSplitHandler::isDummyDmaNeeded(size_t pageInd, VPURT::TaskQueueType dmaQueueType,
                                                       std::optional<size_t> lastDmaTaskOnSameQueueInPageOpt) {
    VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].empty(), "No boundary tasks set for page {0}",
                    pageInd);
    if (_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].find(dmaQueueType) !=
        _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd].end()) {
        // There is boundary task of this type on this page. No need to insert dummy DMA
        _log.nest().trace("No need to insert dummy DMA as there is a boundary task of this type");
        return false;
    }

    // Check if in case there is a DMA of this type in a page and is not a boundary task
    // but this DMA depends on all previous page boundary tasks then there is no need to create a dummy DMA
    if (lastDmaTaskOnSameQueueInPageOpt.has_value()) {
        bool isDepFromAllBoundaryTasksToDma = true;
        auto lastDmaTaskOnSameQueueInPage = lastDmaTaskOnSameQueueInPageOpt.value();
        VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[pageInd - 1].empty(),
                        "No boundary tasks set for page {0}", pageInd - 1);
        for (auto& [_, firstLastTaskInd] : _firstAndLastBoundaryTaskForEachPagePerFifo[pageInd - 1]) {
            auto lastTask = firstLastTaskInd.second;
            if (!isDepFromTaskAToTaskB(lastTask, lastDmaTaskOnSameQueueInPage)) {
                isDepFromAllBoundaryTasksToDma = false;
                break;
            }
        }
        if (isDepFromAllBoundaryTasksToDma) {
            // There is a DMA of this type in this page and it depends on all prev page boundary tasks
            // No need to insert dummy DMA
            _log.nest().trace("No need to insert dummy DMA as there is a DMA of this type in this page");
            return false;
        }
    }

    return true;
}

// Method that prepares wait barrier data for dummy DMA
// In case of crossing control graph block boundary it can also legalize prev page boundary tasks update barriers
// so that wait barriers of dummy DMA will depend on those tasks
std::pair<vpux::BarrierInfo::TaskSet, vpux::BarrierInfo::TaskSet> VPURT::BarrierPagesSplitHandler::getDummyDmaBars(
        size_t pageInd) {
    // As wait barrier use wait barrier of earliest boundary task on this page
    // WLM page split guarantees that this wait barrier is updated by all tasks from previous page
    auto boundaryTasks = getFirstBoundaryTasksForPage(pageInd);
    VPUX_THROW_WHEN(boundaryTasks.empty(), "No boundary tasks set for page {0}", pageInd);
    auto boundaryTask = boundaryTasks.front();

    auto dummyDmaProposedWaitBars = _barrierInfo.getWaitBarriers(boundaryTask);
    BarrierInfo::TaskSet dummyDmaProposedUpdateBars;

    if (dummyDmaProposedWaitBars.empty()) {
        auto taskWithWaitBarOnSamePageOpt = _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(boundaryTask);
        VPUX_THROW_UNLESS(taskWithWaitBarOnSamePageOpt.has_value(), "No wait barriers for task {0}(page {1})",
                          boundaryTask, _taskPageAssignment[boundaryTask]);

        boundaryTask = taskWithWaitBarOnSamePageOpt.value();
        dummyDmaProposedWaitBars = _barrierInfo.getWaitBarriers(boundaryTask);
    }

    // Check if wait barrier is used by control graph sync point then it cannot be used as wait barrier
    // of dummy DMA because there will be no update barrier to use that will not break control graph split
    std::optional<size_t> syncPointTaskOpt = std::nullopt;
    for (auto waitBar : dummyDmaProposedWaitBars) {
        for (auto waitBarUser : _barrierInfo.getBarrierConsumers(waitBar)) {
            if (_barrierInfo.isSyncPoint(waitBarUser)) {
                syncPointTaskOpt = waitBarUser;
                _log.nest().trace("Wait barrier {0} is used by control graph sync point {1}", waitBar, waitBarUser);
                break;
            }
        }
        if (syncPointTaskOpt.has_value()) {
            break;
        }
    }

    // In the case of wait barrier used by sync point use different wait barriers for dummy DMA
    if (syncPointTaskOpt.has_value()) {
        auto syncPointWaitBars = _barrierInfo.getWaitBarriers(syncPointTaskOpt.value());
        auto syncPointWaitBarsVec = to_small_vector(syncPointWaitBars);
        // Store information about prev page boundary tasks that will need to be legalized - they
        // no longer can update barrier consumed by sync task but update barrier that will
        // be consumed by dummy DMA
        SmallVector<size_t> prevPageBoundaryTasksToHaveUpdateBarLegalized;

        auto prevPageInd = pageInd - 1;
        _log.nest().trace("Change wait barriers to be update barrier of page {0} boundary tasks", prevPageInd);
        dummyDmaProposedWaitBars.clear();

        // As wait barriers use update barriers of boundary tasks from previous page. After that dummy DMA
        // may have more than 1 wait barrier but since all DMA tasks are expected to be enqueued at bootstrap
        // this is not a problem for enqueue algorithm
        VPUX_THROW_WHEN(_firstAndLastBoundaryTaskForEachPagePerFifo[prevPageInd].empty(),
                        "No boundary tasks set for page {0}", prevPageInd);
        for (auto& [_, firstLastTaskIndPair] : _firstAndLastBoundaryTaskForEachPagePerFifo[prevPageInd]) {
            auto prevPageBoundaryTask = firstLastTaskIndPair.second;
            auto prevPageBoundaryTaskUpdBars = _barrierInfo.getUpdateBarriers(prevPageBoundaryTask);
            _log.nest(2).trace("Page {0} boundary task {1} update barriers {2}", prevPageInd, prevPageBoundaryTask,
                               to_small_vector(prevPageBoundaryTaskUpdBars));

            // Previous page boundary task may have update barrier both from previous page
            // and current page. Leave only barrier from current page as only such can be a candidate
            // for dummy dma wait barrier to not violate page split restrictions - task can wait only
            // on barrier from its page.
            // Check if barrier is consumed by sync point task. In that case it cannot be used
            // as later this barrier will be used as update barrier of dummy DMA
            auto prevPageBoundaryTaskValidUpdBars = prevPageBoundaryTaskUpdBars;
            for (auto prevPageBoundaryTaskUpdBar : prevPageBoundaryTaskUpdBars) {
                if (getBarrierPage(prevPageBoundaryTaskUpdBar) != pageInd) {
                    prevPageBoundaryTaskValidUpdBars.erase(prevPageBoundaryTaskUpdBar);
                    continue;
                }
                if (llvm::any_of(_barrierInfo.getBarrierConsumers(prevPageBoundaryTaskUpdBar), [&](auto userTask) {
                        return _barrierInfo.isSyncPoint(userTask);
                    })) {
                    prevPageBoundaryTaskValidUpdBars.erase(prevPageBoundaryTaskUpdBar);
                }
            }

            if (prevPageBoundaryTaskValidUpdBars.empty()) {
                _log.nest(2).trace("No update barrier not used by sync task for task {0}", prevPageBoundaryTask);
                // If there is no update barrier from prev page boundary task that can be used
                // then need to legalize prev page boundary task to also update some other barrier
                // that later dummy DMA will depend on
                prevPageBoundaryTasksToHaveUpdateBarLegalized.push_back(prevPageBoundaryTask);
                continue;
            }

            // Just 1 update barrier for each boundary task is enough. Pick one with smallest index
            // so that dummy DMA will depend on earliest barriers possible
            dummyDmaProposedWaitBars.insert(*std::min_element(prevPageBoundaryTaskValidUpdBars.begin(),
                                                              prevPageBoundaryTaskValidUpdBars.end()));
        }

        if (dummyDmaProposedWaitBars.empty()) {
            _log.nest().trace("No wait barriers for dummy DMA in page {0} found", pageInd);
            // If no barrier was found pick some earlier barrier before control block sync point
            _log.nest().trace("Use barrier earlier than sync point {0} wait barriers {1}", syncPointTaskOpt.value(),
                              syncPointWaitBarsVec);
            size_t newBarInd = *std::max_element(syncPointWaitBars.begin(), syncPointWaitBars.end()) - 1;
            while (true) {
                _log.nest(2).trace("Check barrier {0}(page {1}) for dummy DMA wait", newBarInd,
                                   getBarrierPage(newBarInd));
                if (getBarrierPage(newBarInd) < pageInd) {
                    break;
                }

                if (llvm::any_of(_barrierInfo.getBarrierConsumers(newBarInd), [&](auto userTask) {
                        return _barrierInfo.isSyncPoint(userTask);
                    })) {
                    newBarInd--;
                    continue;
                }

                VPUX_THROW_UNLESS(isDepFromBarrierToTask(newBarInd, syncPointTaskOpt.value()),
                                  "No dependency from barrier {0} to sync task {1}", newBarInd,
                                  syncPointTaskOpt.value());
                _log.nest(2).trace("Use barrier {0} as wait barrier for dummy DMA", newBarInd);
                dummyDmaProposedWaitBars.insert(newBarInd);
                break;
            }
        }

        // If there are no dummy DMA wait barriers but syncPoint has more than 1 wait barrier then modify the schedule
        // and create insertion point between 2 of those wait barriers and update producers of them
        //
        //  barProd0   barProd1..        barProd0   barProd1
        //       |        |                   |    /    |
        //      bar0      |          =>       bar0      |
        //       |        |                    |        |
        //       |        |                 dummyDma    |
        //       |        |                    |        |
        //       |       bar1                 bar1 <----|
        //       |        |                    |
        //        syncTask                 syncTask

        if (dummyDmaProposedWaitBars.empty() && syncPointWaitBarsVec.size() > 1) {
            _log.nest().trace("Legalize sync task wait barriers to prepare dummy DMA insertion between them");
            _log.nest(2).trace("prevPageBoundaryTasksToHaveUpdateBarLegalized: {0}",
                               to_small_vector(prevPageBoundaryTasksToHaveUpdateBarLegalized));
            llvm::sort(syncPointWaitBarsVec);
            // Use 2nd largest index barrier as wait barrier for dummyDMA
            // Largest barrier will later be used as an update barrier for dummy DMA during ensuring last task
            // per page has an update barrier
            auto waitBarForDummyDma = *(syncPointWaitBarsVec.end() - 2);
            dummyDmaProposedWaitBars.insert(waitBarForDummyDma);
        }

        VPUX_THROW_WHEN(dummyDmaProposedWaitBars.empty(), "No wait barriers for dummy DMA in page {0} found", pageInd);

        // In case of legalizing prevPageBoundaryTasksToHaveUpdateBarLegalized tasks,
        // from all identified wait barriers to be used for dummy DMA pick the latest one which will
        // be used as an update barrier for prev page boundary task to guarantee that dummy DMA execution
        // depends on all prev page boundary tasks.
        // Use barrier with highest index to prevent from picking barrier which is consumed earlier than
        // given boundary task is located what could create cyclic dependency and compilation error
        auto latestTaskWaitBar = *std::max_element(dummyDmaProposedWaitBars.begin(), dummyDmaProposedWaitBars.end());
        for (auto boundaryTaskToHaveUpdateBarLegalized : prevPageBoundaryTasksToHaveUpdateBarLegalized) {
            _log.nest(2).trace("Legalize update barrier for task {0}", boundaryTaskToHaveUpdateBarLegalized);
            _log.nest(3).trace("Add update barrier {0} for task {1}", latestTaskWaitBar,
                               boundaryTaskToHaveUpdateBarLegalized);
            _barrierInfo.addProducer(latestTaskWaitBar, boundaryTaskToHaveUpdateBarLegalized);
        }

        // In case of a sync point dummy DMA needs to update one of its wait barriers to prevent
        // from moving dummy DMA after sync task in IR after reordering IR what will break control graph restrictions
        auto syncPointWaitBarsNotUsedAsDummyDmaWaitBars = std::move(syncPointWaitBars);
        llvm::set_subtract(syncPointWaitBarsNotUsedAsDummyDmaWaitBars, dummyDmaProposedWaitBars);
        dummyDmaProposedUpdateBars.insert(*std::max_element(syncPointWaitBarsNotUsedAsDummyDmaWaitBars.begin(),
                                                            syncPointWaitBarsNotUsedAsDummyDmaWaitBars.end()));
    }

    // Check if there is a dependency from prev page boundary task to any proposed wait barrier
    auto prevPageBoundaryTasks = getLastBoundaryTasksForPage(pageInd - 1);
    for (auto prevPageBoundaryTask : prevPageBoundaryTasks) {
        bool isDep = false;
        for (auto proposedWaitBar : dummyDmaProposedWaitBars) {
            if (isDepFromTaskToBarrier(prevPageBoundaryTask, proposedWaitBar)) {
                isDep = true;
                break;
            }
        }
        if (isDep) {
            continue;
        }
        _log.nest().trace(
                "No dependency from prev page boundary task {0} to any of proposed wait barriers {1}. Create one.",
                prevPageBoundaryTask, to_small_vector(dummyDmaProposedWaitBars));
        // There is a boundary task on which none of proposed wait barriers depend
        // Create such dependency
        _barrierInfo.addProducer(*dummyDmaProposedWaitBars.begin(), prevPageBoundaryTask);
        _log.nest().trace("Add update barrier {0} for task {1}", *dummyDmaProposedWaitBars.begin(),
                          prevPageBoundaryTask);
    }

    return std::make_pair(dummyDmaProposedWaitBars, dummyDmaProposedUpdateBars);
}

// Based on wait barriers of dummy DMA and last DMA task on the same queue in this page (if exists)
// return after which task dummy DMA can be inserted
size_t VPURT::BarrierPagesSplitHandler::getDummyDmaInsertionPoint(
        vpux::BarrierInfo::TaskSet& dummyDmaProposedWaitBars, std::optional<size_t> lastDmaTaskOnSameQueueInPageOpt) {
    // For finding insertion point of dummy DMA take into account:
    // 1. lastWaitBarrierProducer - last task (max index) that updates barriers from dummyDmaProposedWaitBars
    // 2. lastDmaTaskOnSameQueueInPage - last DMA index on the same queue in this page
    // Dummy DMA needs to be inserted after both above
    // insertAfter = max(lastWaitBarrierProducer, lastDmaTaskOnSameQueueInPage)
    auto lastWaitBarrierProducer = _barrierInfo.getBarriersLatestProducer(dummyDmaProposedWaitBars);

    auto insertionPoint = lastWaitBarrierProducer;
    if (lastDmaTaskOnSameQueueInPageOpt.has_value()) {
        insertionPoint = std::max(insertionPoint, lastDmaTaskOnSameQueueInPageOpt.value());
    }

    return insertionPoint;
}

// Get information for each page about last DMA of each type
SmallVector<mlir::DenseMap<VPURT::TaskQueueType, size_t>> VPURT::BarrierPagesSplitHandler::getLastDmaOfTypePerPage() {
    SmallVector<mlir::DenseMap<VPURT::TaskQueueType, size_t>> lastDmaOfTypePerPage(_pageCount);

    for (size_t taskInd = 0; taskInd < _taskPageAssignment.size(); taskInd++) {
        auto pageInd = _taskPageAssignment[taskInd];
        auto taskQueueType = _barrierInfo.getTaskQueueType(taskInd);
        if (taskQueueType.type == VPU::ExecutorKind::DMA_NN) {
            lastDmaOfTypePerPage[pageInd][taskQueueType] = taskInd;
        }
    }
    return lastDmaOfTypePerPage;
}

// To enable enqueue of all DMA tasks at bootstrap as a single link-list each page needs to have a DMA task.
// If such is not present dummy DMA needs to be inserted. This function returns information on where such
// dummy DMAs need to be inserted.
// Requirement is following:
// If there is no DMAx in PageN whose execution guarantees that barriers from PageN-1 have been consumed before
// and that there are DMAs of this type in some later pages, then we need to insert task of DMAx in PageN
// and this DMAx needs to depend on all boundary tasks from PageN-1/PageN border.
SmallVector<VPURT::BarrierPagesSplitHandler::DummyDmaInsertionData>
VPURT::BarrierPagesSplitHandler::getAndLegalizeDummyDmaInsertionData() {
    _log.trace("Getting dummy DMA insertion data");
    SmallVector<DummyDmaInsertionData> dummyDmaInsertionDataVec;

    SmallVector<std::pair<VPURT::TaskQueueType, size_t>> dmaQueueTypesAndLastPageInd;

    // Get all DMA queue types and last page index for DMA on this queue
    // There is no need in inserting dummy DMAs if there are no more DMAs
    // of this type in subsequent pages
    for (auto queueType : _barrierInfo.getNonEmptyTaskQueueTypes()) {
        if (queueType.type == VPU::ExecutorKind::DMA_NN) {
            _barrierInfo.getLastTaskForQueueType(queueType);
            size_t lastPage = _taskPageAssignment[_barrierInfo.getLastTaskForQueueType(queueType)];

            // In case of DMA Port:0 Channel:DDR which is also used for WLM enqueue DMAs
            // it needs to be guaranteed that there is such DMA in each page so that
            // enqueue DMAs can be inserted afterwards and link listed to rest of DMAs
            if (queueType.id == getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR)) {
                lastPage = _taskPageAssignment[_barrierInfo.getNumOfTasks() - 1];
            }

            dmaQueueTypesAndLastPageInd.push_back(std::make_pair(queueType, lastPage));
        }
    }

    // Get information for each page about last DMA of each type
    // This information will be later used to check if dummy DMA is really needed
    // and to find insertion point of dummy DMAs
    auto lastDmaOfTypePerPage = getLastDmaOfTypePerPage();

    // Get information for each page about first and last barrier index
    // This is going to be used to find update barrier for dummy DMA
    // TODO: This should no longer be needed after E#167504 is implemented
    SmallVector<std::pair<size_t, size_t>> firstAndLastBarIndPerPage;
    firstAndLastBarIndPerPage.resize(_pageCount);
    firstAndLastBarIndPerPage[0].first = 0;
    for (size_t barInd = 1; barInd < _barrierInfo.getNumOfBarrierOps(); barInd++) {
        auto pageInd = getBarrierPage(barInd);
        if (firstAndLastBarIndPerPage[pageInd].first == 0) {
            firstAndLastBarIndPerPage[pageInd].first = barInd;
        }
        firstAndLastBarIndPerPage[pageInd].second = barInd;
    }

    // Iterate each page and check if there is a DMA boundary task of each type
    // Skip first page as the earliest place where we might need to insert dummy DMA is Page1 to prevent
    // tasks from Page2 to potentially execute prematurely in Page0
    // Skip last page as there are is no need to insert dummy DMA there as there are no pages afterwards
    for (size_t pageInd = 1; pageInd < _pageCount - 1; pageInd++) {
        // Check if given page has boundary task of each DMA type
        for (auto [dmaQueueType, lastPageInd] : dmaQueueTypesAndLastPageInd) {
            if (pageInd >= lastPageInd) {
                continue;
            }

            std::optional<size_t> lastDmaTaskOnSameQueueInPageOpt =
                    (lastDmaOfTypePerPage[pageInd].find(dmaQueueType) != lastDmaOfTypePerPage[pageInd].end())
                            ? std::make_optional<size_t>(lastDmaOfTypePerPage[pageInd][dmaQueueType])
                            : std::nullopt;

            _log.trace("Check page {0} for DMA of type {1}:{2}", pageInd, stringifyEnum(dmaQueueType.type).data(),
                       dmaQueueType.id);

            if (!isDummyDmaNeeded(pageInd, dmaQueueType, lastDmaTaskOnSameQueueInPageOpt)) {
                continue;
            }

            _log = _log.nest();
            _log.trace("Prepare data for dummy DMA");

            DummyDmaInsertionData dummyDmaInsertionData;
            dummyDmaInsertionData.pageInd = pageInd;
            dummyDmaInsertionData.queueType = dmaQueueType;

            auto [dummyDmaProposedWaitBars, dummyDmaProposedUpdateBars] = getDummyDmaBars(pageInd);
            dummyDmaInsertionData.waitBars = to_small_vector(dummyDmaProposedWaitBars);
            dummyDmaInsertionData.updateBars = to_small_vector(dummyDmaProposedUpdateBars);

            auto insertionPoint = getDummyDmaInsertionPoint(dummyDmaProposedWaitBars, lastDmaTaskOnSameQueueInPageOpt);
            dummyDmaInsertionData.insertAfter = insertionPoint;

            _log.nest().trace("Insert after op {0}", dummyDmaInsertionData.insertAfter);
            _log.nest().trace("Wait barriers: {0}", dummyDmaInsertionData.waitBars);
            if (!dummyDmaInsertionData.updateBars.empty()) {
                _log.nest().trace("Update barriers: {0}", dummyDmaInsertionData.updateBars);
            }

            dummyDmaInsertionDataVec.push_back(dummyDmaInsertionData);
            _log = _log.unnest();
        }
    }

    return dummyDmaInsertionDataVec;
}

// Prepare data for inserting dummy barriers. They are needed to be placed
// in pages which use less than half of available physical barriers. To make barrier
// programming DMAs simple and able to always refill 4 entries for all physical barriers
// as a single transaction, each page except last two need to always use exactly half of
// physical barriers. Dummy barriers will be placed in parallel to existing barriers
// what will not have any impact on performance of schedule.
SmallVector<VPURT::BarrierPagesSplitHandler::DummyBarrierData>
VPURT::BarrierPagesSplitHandler::getDummyBarriersInsertionData() {
    _log.trace("Getting dummy barriers data");
    SmallVector<DummyBarrierData> dummyBarrierDataVec;

    if (_pageCount <= 2) {
        _log.trace("No need to insert dummy barriers if model has {0} <= 2 pages", _pageCount);
        return dummyBarrierDataVec;
    }

    SmallVector<size_t> numberOfBarriersPerPage(_pageCount, 0);
    SmallVector<size_t> lastBarrierIndexPerPage(_pageCount, 0);
    // Iterate each barrier and count number of barriers per page and store information about
    // last barrier index
    for (size_t barInd = 0; barInd < _barrierInfo.getNumOfBarrierOps(); barInd++) {
        auto pageInd = getBarrierPage(barInd);
        numberOfBarriersPerPage[pageInd]++;
        lastBarrierIndexPerPage[pageInd] = barInd;
    }

    // Iterate each page and check the number of barriers
    // Skip last two pages, which are last pages for two halves of physical barrier set.
    // There is no need to insert dummy barriers there as those entries can be filled by dummy values
    // when programming barrier FIFOs using barrier programming DMA
    _log = _log.nest();
    for (size_t pageInd = 0; pageInd < _pageCount - 2; pageInd++) {
        _log.trace("Page {0} barrier count: {1}", pageInd, numberOfBarriersPerPage[pageInd]);
        for (size_t barrierIndexToAdd = numberOfBarriersPerPage[pageInd]; barrierIndexToAdd < _pageSize;
             barrierIndexToAdd++) {
            _log.nest().trace("Missing barrier. Prepare new one {0}", barrierIndexToAdd);

            DummyBarrierData dummyBarrierData;
            dummyBarrierData.pageInd = pageInd;
            dummyBarrierData.insertAfter = lastBarrierIndexPerPage[pageInd];
            dummyBarrierData.consumer = *(_barrierInfo.getBarrierConsumers(lastBarrierIndexPerPage[pageInd]).begin());
            dummyBarrierData.producer = *(_barrierInfo.getBarrierProducers(lastBarrierIndexPerPage[pageInd]).begin());
            _log.nest().trace("New barrier data: insert after bar {0}, producer {1}, consumer {2}",
                              dummyBarrierData.insertAfter, dummyBarrierData.producer, dummyBarrierData.consumer);

            dummyBarrierDataVec.push_back(dummyBarrierData);
        }
    }
    _log = _log.unnest();

    return dummyBarrierDataVec;
}

// Identify all tasks that are the last on FIFO in each page and do not have update barriers but have wait
// barrier as such tasks if started later may delay consumption of a barrier. If they do not have
// update barrier potential next page barrier programming DMA will not be able to correctly identify
// barriers whose production guarantee that previous page barriers have been consumed. Because of that
// each HW FIFO task list present in a given page need to have an update barrier that will be a signal
// for consuming wait barrier on this FIFO.
SmallVector<size_t> vpux::VPURT::BarrierPagesSplitHandler::getLastTasksOnFifoPerPageWithNoUpdBar() {
    SmallVector<mlir::DenseMap<VPURT::TaskQueueType, size_t>> lastTaskPerTypePerPageWithWaitBar(_pageCount);
    SmallVector<mlir::DenseMap<VPURT::TaskQueueType, size_t>> lastTaskPerTypePerPageWithUpdateBar(_pageCount);
    SmallVector<mlir::DenseMap<VPURT::TaskQueueType, size_t>> lastTaskPerTypePerPageWithNoUpdateBar(_pageCount);

    for (size_t taskInd = 0; taskInd < _taskPageAssignment.size(); taskInd++) {
        auto pageInd = _taskPageAssignment[taskInd];
        auto taskQueueType = _barrierInfo.getTaskQueueType(taskInd);

        if (!_barrierInfo.getWaitBarriers(taskInd).empty()) {
            lastTaskPerTypePerPageWithWaitBar[pageInd][taskQueueType] = taskInd;
        }
        if (!_barrierInfo.getUpdateBarriers(taskInd).empty()) {
            lastTaskPerTypePerPageWithUpdateBar[pageInd][taskQueueType] = taskInd;
        } else {
            lastTaskPerTypePerPageWithNoUpdateBar[pageInd][taskQueueType] = taskInd;
        }
    }

    SmallVector<size_t> lastTaskPerTypePerPageThatNeedsUpdateBarrier;
    for (size_t pageInd = 0; pageInd < _pageCount; pageInd++) {
        SmallVector<size_t> lastTaskPerTypeThatNeedUpdateBarrier;
        for (auto& [queueType, taskInd] : lastTaskPerTypePerPageWithWaitBar[pageInd]) {
            bool needUpdateBar = true;
            // Check if there is any next task on this page with update barrier
            auto lastTaskWithUpdateBarIt = lastTaskPerTypePerPageWithUpdateBar[pageInd].find(queueType);
            if (lastTaskWithUpdateBarIt != lastTaskPerTypePerPageWithUpdateBar[pageInd].end()) {
                auto lastTaskWithUpdateBar = lastTaskWithUpdateBarIt->second;
                if (lastTaskWithUpdateBar >= taskInd) {
                    needUpdateBar = false;
                }
            }

            if (needUpdateBar) {
                lastTaskPerTypeThatNeedUpdateBarrier.push_back(taskInd);
            }
        }

        // Check if this is the last task on this FIFO before control block sync task.
        // If yes then update barrier is needed

        // Check if this page is last before control block sync point
        // sync point is always expected to be an only  boundary task

        for (auto& [queueType, taskInd] : lastTaskPerTypePerPageWithNoUpdateBar[pageInd]) {
            if (std::find(lastTaskPerTypeThatNeedUpdateBarrier.begin(), lastTaskPerTypeThatNeedUpdateBarrier.end(),
                          taskInd) != lastTaskPerTypeThatNeedUpdateBarrier.end()) {
                // Task is already considered for new update barrier
                continue;
            }

            auto taskBlockInd = _barrierInfo.getControlGraphBlockIndex(taskInd);
            auto syncPointTaskOpt = _barrierInfo.getControlGraphSyncPoint(taskInd);

            if (!syncPointTaskOpt.has_value()) {
                // If there is no control block sync point then no need to add update barrier
                continue;
            }

            auto syncPointTaskQueueType = _barrierInfo.getTaskQueueType(syncPointTaskOpt.value());
            // If task is on same queue as sync point task then no need to add update barrier
            if (queueType == syncPointTaskQueueType) {
                continue;
            }

            // If there is new block later check if there is next task
            auto nextTaskIndOpt = _barrierInfo.getNextTaskOnSameQueue(taskInd);
            if (nextTaskIndOpt.has_value()) {
                // There is next task on this FIFO. Check its control graph block index
                auto nextTaskBlockInd = _barrierInfo.getControlGraphBlockIndex(nextTaskIndOpt.value());
                if (nextTaskBlockInd == taskBlockInd) {
                    // Next task is in the same control graph block. No need for update barrier
                    // for current task
                    continue;
                }
            }

            // This task needs to have an update barrier to guarantee there is dependency to sync task
            lastTaskPerTypeThatNeedUpdateBarrier.push_back(taskInd);
        }

        lastTaskPerTypePerPageThatNeedsUpdateBarrier.insert(lastTaskPerTypePerPageThatNeedsUpdateBarrier.end(),
                                                            lastTaskPerTypeThatNeedUpdateBarrier.begin(),
                                                            lastTaskPerTypeThatNeedUpdateBarrier.end());
    }

    return lastTaskPerTypePerPageThatNeedsUpdateBarrier;
}

// For identified per page last tasks with no update barriers add new dependency to
// to some wait barrier of boundary task on this page or if this is not possible
// use barrier from next page and make this task a new boundary task
void vpux::VPURT::BarrierPagesSplitHandler::addUpdateBarriersForLastTaskOnFifoInPage(
        SmallVector<size_t>& lastTaskPerTypePerPageThatNeedsUpdateBarrier) {
    _log.trace("Add update barriers for last task on FIFO in each page");

    if (lastTaskPerTypePerPageThatNeedsUpdateBarrier.empty() || _pageCount <= 2) {
        return;
    }

    SmallVector<std::optional<size_t>> updateBarrierToUseOnPage(_pageCount);

    BarrierInfo::TaskSet pagesWithBoundaryTasksChanged;

    for (auto taskInd : lastTaskPerTypePerPageThatNeedsUpdateBarrier) {
        auto pageInd = _taskPageAssignment[taskInd];
        if (!updateBarrierToUseOnPage[pageInd].has_value()) {
            // Find boundary tasks wait barriers
            BarrierInfo::TaskSet boundaryTasksWaitBars;
            auto firstBoundaryTasksOnPage = getFirstBoundaryTasksForPage(pageInd);
            for (auto pageBoundaryTask : firstBoundaryTasksOnPage) {
                auto pageBoundaryTaskWaitBars = _barrierInfo.getWaitBarriers(pageBoundaryTask);
                if (pageBoundaryTaskWaitBars.empty()) {
                    auto prevTaskOpt = _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(pageBoundaryTask);
                    if (prevTaskOpt.has_value()) {
                        pageBoundaryTaskWaitBars = _barrierInfo.getWaitBarriers(prevTaskOpt.value());
                    }
                }
                boundaryTasksWaitBars.insert(pageBoundaryTaskWaitBars.begin(), pageBoundaryTaskWaitBars.end());
            }

            if (!boundaryTasksWaitBars.empty()) {
                // As an update barrier use latest barrier from identified set. It gives highest likelihood
                // of finding a barrier within this page which the task itself does not depend on.
                updateBarrierToUseOnPage[pageInd] =
                        *std::max_element(boundaryTasksWaitBars.begin(), boundaryTasksWaitBars.end());
            } else {
                // In case the pageBoundaryTask has no previous task then use the last barrier from page
                VPUX_THROW_UNLESS(pageInd == 0,
                                  "No wait barrier identified for boundary task that is not on page 0 but page {0}",
                                  pageInd);
                auto pageLastBar = _firstBarrierInPage[pageInd + 1] - 1;
                updateBarrierToUseOnPage[pageInd] = pageLastBar;
            }

            auto barPage = getBarrierPage(updateBarrierToUseOnPage[pageInd].value());
            VPUX_THROW_UNLESS(barPage == pageInd, "Barrier {0} page {1} invalid. Expected {2}",
                              updateBarrierToUseOnPage[pageInd].value(), barPage, pageInd);
        }

        auto updateBar = updateBarrierToUseOnPage[pageInd].value();

        BarrierInfo::TaskSet updateBars;

        // Check if task depends on proposed barrier
        if (isDepFromBarrierToTask(updateBar, taskInd)) {
            // If this task cannot depend on this barrier then need to pick another barrier
            // Solution is to make this task a boundary task for this page
            _log.trace("Task {0} already depends on update barrier {1} on page {2}. Pick next barrier from next page",
                       taskInd, updateBar, pageInd);

            // Use first barrier from next page
            while (getBarrierPage(updateBar) == pageInd) {
                updateBar++;
            }
            pagesWithBoundaryTasksChanged.insert(pageInd);

            // If task is going to be a boundary task then need to pick barriers
            // from other boundary tasks so that page split constraints - next page boundary
            // tasks PageN+1 dependencies on boundary tasks from PageN are satisfied
            // without the need to do legalization
            for (auto pageBoundaryTask : getLastBoundaryTasksForPage(pageInd)) {
                // Get page boundary tasks update bars
                auto pageBoundaryTaskUpdateBars = to_small_vector(_barrierInfo.getUpdateBarriers(pageBoundaryTask));

                pageBoundaryTaskUpdateBars.erase(llvm::remove_if(pageBoundaryTaskUpdateBars,
                                                                 [&](auto barInd) {
                                                                     return getBarrierPage(barInd) == pageInd;
                                                                 }),
                                                 pageBoundaryTaskUpdateBars.end());

                updateBars.insert(pageBoundaryTaskUpdateBars.begin(), pageBoundaryTaskUpdateBars.end());
            }
        }
        updateBars.insert(updateBar);

        auto taskQueueType = _barrierInfo.getTaskQueueType(taskInd);
        _log.trace("Add update barrier for task {0} on queue {1}:{2} in page {3}", taskInd,
                   stringifyEnum(taskQueueType.type).data(), taskQueueType.id, pageInd);

        for (auto bar : updateBars) {
            _barrierInfo.addProducer(bar, taskInd);
            _log.nest().trace("Add update barrier {0} for task {1}", bar, taskInd);
        }
    }

    for (auto pageWithBoundaryTasksChanged : pagesWithBoundaryTasksChanged) {
        updateBoundaryTasksDataForPage(pageWithBoundaryTasksChanged);
    }
}

void vpux::VPURT::BarrierPagesSplitHandler::initPrevPhysBarrierData(SmallVector<size_t>& barrierToPidVec) {
    size_t numOfBarriers = _barrierInfo.getNumOfBarrierOps();

    VPUX_THROW_UNLESS(numOfBarriers == barrierToPidVec.size(), "Not matching number of barriers {0} != {1}",
                      numOfBarriers, barrierToPidVec.size());

    _barrierPidPrevUsageVec.resize(numOfBarriers);
    SmallVector<std::optional<size_t>> lastBarUsingPid(_pageSize * 2);

    for (size_t vid = 0; vid < numOfBarriers; vid++) {
        auto pid = barrierToPidVec[vid];
        _barrierPidPrevUsageVec[vid] = lastBarUsingPid[pid];
        lastBarUsingPid[pid] = vid;
    }
}

// For each barrier find index of previous barrier using same PID
void vpux::VPURT::BarrierPagesSplitHandler::initPrevPhysBarrierData(mlir::func::FuncOp func) {
    // Get information for each barrier about previous barrier that used same PID
    size_t numOfBarriers = _barrierInfo.getNumOfBarrierOps();

    SmallVector<size_t> barrierToPidVec(numOfBarriers);

    func.walk([&](VPURT::ConfigureBarrierOp barOp) {
        size_t pid = barOp.getId();
        auto vid = _barrierInfo.getIndex(barOp);
        barrierToPidVec[vid] = pid;
    });

    initPrevPhysBarrierData(barrierToPidVec);
}

SmallVector<size_t> vpux::VPURT::BarrierPagesSplitHandler::getBarrierPidPrevUsageVec(BarrierInfo::TaskSet& barriers) {
    SmallVector<size_t> barrierPidPrevUsageVec;
    for (auto barInd : barriers) {
        auto prevBar = _barrierPidPrevUsageVec[barInd];
        if (prevBar.has_value()) {
            barrierPidPrevUsageVec.push_back(prevBar.value());
        }
    }
    return barrierPidPrevUsageVec;
}

// Get number of workloads under single processed task
// In case of DPU it is number of DPU variants
// In case of SHV it is number of SHV kernel run ops
// For other return 1
size_t vpux::VPURT::BarrierPagesSplitHandler::getNumberOfWorkloads(size_t taskInd) {
    if (!_getNumberOfWorkloadsFromTaskOpFlag) {
        return 1;
    }

    auto taskOp = _barrierInfo.getTaskOpAtIndex(taskInd);

    if (taskOp.getExecutorKind() == VPU::ExecutorKind::DPU) {
        auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp());
        VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");
        return nceOp.getNumVariants();
    }

    if (taskOp.getExecutorKind() == VPU::ExecutorKind::SHAVE_ACT) {
        if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp())) {
            auto swKernelRun = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
            return std::distance(swKernelRun.begin(), swKernelRun.end());
        }
        return 1;
    }

    if (taskOp.getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
        return 1;
    }

    VPUX_THROW("Unsupported executor: {0}", taskOp.getExecutorKind());
}

VPURT::BarrierPagesSplitHandler::EnqueueDmaData vpux::VPURT::BarrierPagesSplitHandler::getDataForNewEnqueueDmaForTask(
        size_t taskInd, size_t taskWorkloadStartIdx, size_t taskWorkloadEndIdx, VPURT::TaskQueueType queueType,
        SmallVector<mlir::DenseMap<VPURT::TaskQueueType, size_t>>& lastDmaOfTypePerPage) {
    const VPURT::TaskQueueType dmaP0ChDdrQueueType = {VPU::ExecutorKind::DMA_NN,
                                                      getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR)};
    auto taskPage = _taskPageAssignment[taskInd];
    auto lastDmaP0ChDdr = lastDmaOfTypePerPage[taskPage - 1][dmaP0ChDdrQueueType];
    auto waitBars = _barrierInfo.getWaitBarriers(lastDmaP0ChDdr);

    if (waitBars.empty()) {
        auto prevTaskOpt = _barrierInfo.getPrevTaskOnSameQueueWithWaitBar(lastDmaP0ChDdr);
        if (prevTaskOpt.has_value()) {
            lastDmaP0ChDdr = prevTaskOpt.value();
            waitBars = _barrierInfo.getWaitBarriers(lastDmaP0ChDdr);
            _log.nest(2).trace("Use wait barriers of task {0}(page {1}) - {2}", lastDmaP0ChDdr,
                               _taskPageAssignment[lastDmaP0ChDdr], to_small_vector(waitBars));
        }
    }

    VPUX_THROW_UNLESS(!waitBars.empty(), "No wait barriers for last DMA {0}(page {1}) found", lastDmaP0ChDdr,
                      taskPage - 1);

    EnqueueDmaData newEnq;
    newEnq.pageInd = taskPage - 1;
    newEnq.queueType = queueType;
    newEnq.startTaskIdx = taskWorkloadStartIdx;
    newEnq.endTaskIdx = taskWorkloadEndIdx;
    newEnq.waitBars = to_small_vector(waitBars);
    newEnq.insertBefore = lastDmaP0ChDdr;

    return newEnq;
}

// This method prepares enqueue information in case of Full WLM for each task
// Return enqueue barrier for each task
// TODO: In next stage this method will return information about enqueue DMAs (E#170833)
SmallVector<VPURT::BarrierPagesSplitHandler::EnqueueDmaData> vpux::VPURT::BarrierPagesSplitHandler::getEnqueueDmaData(
        const ExecutionGroupAnalysis& execGroupAnalysis,
        const mlir::DenseSet<vpux::VPU::ExecutorKind>& executorEnqAtBootstrap) {
    VPUX_THROW_WHEN(_barrierPidPrevUsageVec.empty(), "Barrier PID previous usage vector is not initialized.");
    SmallVector<EnqueueDmaData> enqDmaDataVec;

    auto lastDmaOfTypePerPage = getLastDmaOfTypePerPage();

    for (auto& [queueType, taskVec] : _taskQueueTypeMap) {
        _log.trace("Enqueue tasks for {0}:{1}", VPU::stringifyExecutorKind(queueType.type), queueType.id);

        // Check if for given queue it is requested to enqueue all tasks at bootstrap
        // In that case skip processing of this queue as no enqueue DMA is needed
        if (executorEnqAtBootstrap.contains(queueType.type)) {
            _log.nest().trace("Enqueue task from that queue at bootstrap");
            continue;
        }

        // DMA tasks can be enqueued at bootstrap as they do not require
        // descriptor fetching but DPU and SHV need to be enqueued after start barrier as earliest point
        // as before it there is space for descriptor fetching DMA
        // TODO: This theoretically is no longer needed for Full WLM if we make sure descriptor fetching DMA is
        // before enqueue DMA. To be removed in future
        bool needsEnqAfterStartBar = (queueType.type != VPU::ExecutorKind::DMA_NN);

        // Earliest enqueue DMA for queue
        EnqueueDmaData earliestEnqDma;
        earliestEnqDma.pageInd = 0;
        earliestEnqDma.queueType = queueType;
        VPUX_THROW_UNLESS(_startBarrierIndex.has_value(), "Start barrier index is not set");
        VPUX_THROW_UNLESS(needsEnqAfterStartBar,
                          "Enqueue DMA logic is intended only for queues that need to be enqueued after start bar. "
                          "Queue {0} is not supported",
                          queueType.type);

        // Get earliest start barrier consumer and set enqueue DMA before it
        auto startBarEarliestConsumer = _barrierInfo.getBarrierEarliestConsumer(_startBarrierIndex.value());
        earliestEnqDma.insertBefore = startBarEarliestConsumer;
        earliestEnqDma.waitBars.push_back(_startBarrierIndex.value());

        SmallVector<EnqueueDmaData> enqDmaDataPerQueueVec;

        // Check if there is a need to check enqueue of each DPU if it would not block
        // submission of DPUs from SHV
        bool dpuEnqCheckForShv = queueType.type == VPU::ExecutorKind::DPU && !_shvTasksWithDpuPerTile.empty() &&
                                 !_shvTasksWithDpuPerTile[queueType.id].empty();
        if (dpuEnqCheckForShv) {
            _log.trace("There are {0} SHV tasks which submit DPU. DPU[{1}] enqueue needs to take that into account",
                       _shvTasksWithDpuPerTile[queueType.id].size(), queueType.id);
        }
        // Initialize start index for SHV tasks with DPU _shvTasksWithDpuPerTile[<queue>] vector processing
        // This is for compile time optimization to not always check whole _shvTasksWithDpuPerTile[<queue>] if
        // there were DPUs that were already delayed because of some already processed SHV with DPU
        size_t shvTasksWithDpuVecStartInd = 0;

        std::optional<size_t> previousTaskIndOpt = std::nullopt;
        std::optional<size_t> prevTaskPage = std::nullopt;

        // At this point we're either going over DPU or SHV queues
        ExecutionGroupListMap executionGrouplistMap;
        if (queueType.type == VPU::ExecutorKind::DPU) {
            executionGrouplistMap = execGroupAnalysis.getDPUExecutionGroups();
        } else {
            executionGrouplistMap = execGroupAnalysis.getActShvExecutionGroups();
        }

        auto executionGroups = executionGrouplistMap[queueType];

        _log = _log.nest();
        // Iterate over all tasks of this queue type and find enqueue barrier for each task
        size_t enqueuedWorkloads = 0;
        for (auto taskInd : taskVec) {
            auto taskPage = _taskPageAssignment[taskInd];

            auto taskWorkloadsCount = getNumberOfWorkloads(taskInd);
            auto taskWorkloadStartIdx = enqueuedWorkloads;
            auto taskWorkloadEndIdx = enqueuedWorkloads + taskWorkloadsCount - 1;

            auto waitBars = _barrierInfo.getWaitBarriers(taskInd);

            _log.trace("Find enqueue for task {0}(page {1}) with wait bars {2}", taskInd, taskPage,
                       to_small_vector(waitBars));

            bool enqueueIdentified = false;
            auto executionGroupIndex = execGroupAnalysis.getGroupIndexForTask(taskInd, queueType);
            VPUX_THROW_WHEN(!executionGroupIndex.has_value(), "Could not find execution group index for task {0}",
                            taskInd);

            // Requirement to have separate enqueue:
            //
            // Case 1: Parent group has multiple tasks
            // ----------------------------------------
            // We legalize the barriers like this:
            //
            //     [LastTaskOfGrandParent]
            //         -> [FirstTaskOfParent]
            //         -> FetchTravelingGroup
            //         -> ... [LastTaskOfParent = TaskX - 1]
            //         -> [FirstTaskOfTravelingGroup = TaskX]
            //
            // In this case, the first task of the traveling group is guaranteed to start
            // only after the parent group has made sufficient progress (at least one task executed).
            // This ensures buffer A is not overwritten prematurely by the management DMA thus we can enqueue all tasks
            // with previous.
            //
            // Case 2: Parent group has only one task
            // --------------------------------------
            // We cannot have the FetchTravelingGroup in parallel to parent tasks , so we legalize like:
            //
            //     [LastTaskOfGrandParent]
            //         -> FetchTravelingGroup
            //         -> [FirstAndLastTaskOfParent = TaskX - 1]
            //         -> [FirstTaskOfTravelingGroup = TaskX]
            //
            // In this edge case, if preemption happens after the last task of grand parent
            // management DMA is unblocked and will replaces descriptors in A
            // Having a separate enqueue avoids this issue
            //
            auto executionGroupIndexValue = executionGroupIndex.value();
            if (executionGroupIndexValue > 1 && executionGroups[executionGroupIndexValue][0] == taskInd &&
                executionGroups[executionGroupIndexValue - 1].size() == 1) {
                auto newEnqueueDmaData = getDataForNewEnqueueDmaForTask(
                        taskInd, taskWorkloadStartIdx, taskWorkloadEndIdx, queueType, lastDmaOfTypePerPage);
                enqDmaDataPerQueueVec.push_back(newEnqueueDmaData);
                _log.nest().trace("Created separate enqueue for task {0} (page {1}) after wait barriers {2}", taskInd,
                                  taskPage, newEnqueueDmaData.waitBars);
            } else {
                // Find enqueue barrier based on conditions
                if (taskPage < 2) {
                    // Case 1:
                    // Tasks from Page 0 and 1 can be enqueued right after startBarrier
                    if (enqDmaDataPerQueueVec.empty()) {
                        // If no task has been enqueue before then insert first enqueue DMA
                        // with task indexes range for current task
                        earliestEnqDma.startTaskIdx = taskWorkloadStartIdx;
                        earliestEnqDma.endTaskIdx = taskWorkloadEndIdx;
                        enqDmaDataPerQueueVec.push_back(earliestEnqDma);
                    } else {
                        // If there is already enqueue DMA for this queue then end task index
                        // range to cover also current task
                        enqDmaDataPerQueueVec.back().endTaskIdx = taskWorkloadEndIdx;
                    }

                    enqueueIdentified = true;
                    _log.nest().trace("Page 0 and 1 tasks can be enqueued at schedule start after barriers {0}",
                                      earliestEnqDma.waitBars);
                } else if (prevTaskPage.has_value() && prevTaskPage.value() == taskPage) {
                    // Case 2:
                    // If task is on same page as previous task then enqueue it together - update end task range
                    VPUX_THROW_WHEN(enqDmaDataPerQueueVec.empty(), "Enqueue DMA data vector is empty");
                    enqDmaDataPerQueueVec.back().endTaskIdx = taskWorkloadEndIdx;

                    enqueueIdentified = true;
                    _log.nest().trace(
                            "Task has same page as previous. Enqueue together with previous after barriers {0}",
                            enqDmaDataPerQueueVec.back().waitBars);
                } else {
                    // Case 3:
                    // Find enqueue barrier based on task wait barriers
                    auto prevBars = getBarrierPidPrevUsageVec(waitBars);

                    if (prevBars.empty()) {
                        // Case 3a:
                        // If task has no previous instances of wait barriers
                        if (prevTaskPage.has_value()) {
                            _log.nest().trace("Task has no previous barrier instances. Enqueue together with previous");
                            VPUX_THROW_WHEN(enqDmaDataPerQueueVec.empty(), "Enqueue DMA data vector is empty");
                            enqDmaDataPerQueueVec.back().endTaskIdx = taskWorkloadEndIdx;

                            enqueueIdentified = true;
                        } else {
                            VPUX_THROW_UNLESS(enqDmaDataPerQueueVec.empty(),
                                              "Enqueue DMA data vector is not empty at point of enqueue task {0}",
                                              taskInd);
                            // If no task has been enqueue before then insert first enqueue DMA
                            // with task indexes range for current task
                            earliestEnqDma.startTaskIdx = taskWorkloadStartIdx;
                            earliestEnqDma.endTaskIdx = taskWorkloadEndIdx;
                            enqDmaDataPerQueueVec.push_back(earliestEnqDma);

                            _log.nest().trace("Task has no previous barrier instances. Enqueue at schedule start after "
                                              "barriers {0}",
                                              earliestEnqDma.waitBars);
                            enqueueIdentified = true;
                        }
                    }

                    if (!enqueueIdentified && previousTaskIndOpt.has_value()) {
                        // Case 3b:
                        // Check if task can be enqueued with previous by checking if there is dependency from all
                        // previous instances of wait barriers users to previous task
                        bool canBeMerged = true;
                        auto prevTask = previousTaskIndOpt.value();
                        _log.nest().trace("Check if task can be merged with previous - {0}(page {1})", prevTask,
                                          _taskPageAssignment[prevTask]);
                        for (auto prevBar : prevBars) {
                            auto prevBarUsers = _barrierInfo.getBarrierConsumers(prevBar);
                            _log.nest(2).trace("Check dependencies from prev bar {0} users: {1} to prevTask {2}",
                                               prevBar, to_small_vector(prevBarUsers), prevTask);
                            for (auto prevBarUser : prevBarUsers) {
                                _log.nest(2).trace("Check dependency from {0} to {1}", prevBarUser, prevTask);
                                if (!isDepFromTaskAToTaskB(prevBarUser, prevTask)) {
                                    _log.nest(2).trace(
                                            "Cannot be enqueued with previous because there is no dependency "
                                            "from {0} to {1}",
                                            prevBarUser, prevTask);
                                    canBeMerged = false;
                                    break;
                                }
                            }
                            if (!canBeMerged) {
                                break;
                            }
                        }

                        if (canBeMerged) {
                            enqDmaDataPerQueueVec.back().endTaskIdx = taskWorkloadEndIdx;

                            enqueueIdentified = true;
                            _log.nest().trace("Can be enqueued with previous task after barriers {0}",
                                              enqDmaDataPerQueueVec.back().waitBars);
                        }
                    }

                    if (!enqueueIdentified) {
                        // Case 3c:
                        // Safe enqueue for task in PageN is to use last DMA of Page N-1 as previous pass which insert
                        // dummy DMAs makes sure such DMA will execute after all barriers from PageN-2 has been consumed
                        // TODO: Experiment with more optimal solutions and find earlier possible enqueue point.
                        // This will be needed in case this method would inject enqueue DMAs directly (E#170833)
                        auto newEnqueueDmaData = getDataForNewEnqueueDmaForTask(
                                taskInd, taskWorkloadStartIdx, taskWorkloadEndIdx, queueType, lastDmaOfTypePerPage);
                        enqDmaDataPerQueueVec.push_back(newEnqueueDmaData);

                        _log.nest().trace("Enqueue task after barriers {0}", enqDmaDataPerQueueVec.back().waitBars);
                    }
                }
            }

            // Check if enqueue barrier needs to be delayed because of SHV task submitting DPU
            if (dpuEnqCheckForShv) {
                bool isDpuDelayedAfterShv = false;
                for (size_t shvTasksWithDpuVecInd = shvTasksWithDpuVecStartInd;
                     shvTasksWithDpuVecInd < _shvTasksWithDpuPerTile[queueType.id].size(); shvTasksWithDpuVecInd++) {
                    auto shvTaskInd = _shvTasksWithDpuPerTile[queueType.id][shvTasksWithDpuVecInd];

                    if (shvTaskInd < taskInd && isDepFromTaskAToTaskB(shvTaskInd, taskInd)) {
                        // DPU task depends on SHV. Check if DPU enqueue DMA barrier is after SHV task
                        _log.nest().trace("DPU task {0} depends on SHV task {1} which submits DPU", taskInd,
                                          shvTaskInd);
                        // Since this DPU depends on this SHV task, there is no need for subsequent DPUs to check
                        // dependency against it, thus move the iteration start index forward
                        shvTasksWithDpuVecStartInd = shvTasksWithDpuVecInd + 1;

                        auto& lastEnqDmaData = enqDmaDataPerQueueVec.back();

                        bool isEnqDmaAfterShv = false;
                        for (auto dpuEnqBar : lastEnqDmaData.waitBars) {
                            if (isDepFromTaskToBarrier(shvTaskInd, dpuEnqBar)) {
                                // If SHV task is before enqueue DMA barrier then it is safe to enqueue
                                _log.nest().trace("SHV task {0} is before enqueue DMA barrier {1} for task {2}",
                                                  shvTaskInd, dpuEnqBar, taskInd);
                                isEnqDmaAfterShv = true;
                                break;
                            }
                        }

                        if (!isEnqDmaAfterShv) {
                            // If enqueue is not after SHV task completion then delay it
                            // Check on which barrier to delay. Currently it is guaranteed by previous
                            // passes that SHV with DPU will have following sequence:
                            //  .. -> SHV(DPU) -> BAR -> SyncDMA -> ...
                            // In such case it is safe to delay on BAR barrier
                            auto shvTaskUpdBars = _barrierInfo.getUpdateBarriers(shvTaskInd);
                            VPUX_THROW_WHEN(shvTaskUpdBars.empty(),
                                            "SHV task {0} has no update barriers. Cannot delay enqueue for task {1}",
                                            shvTaskInd, taskInd);
                            auto newEnqDmaBar = *std::min_element(shvTaskUpdBars.begin(), shvTaskUpdBars.end());
                            auto newInsertBefore = _barrierInfo.getBarrierLatestProducer(newEnqDmaBar) + 1;
                            _log.nest().trace(
                                    "Delay enqueue of task {0} to {1} due to dependency on SHV task {2} which "
                                    "submits DPU",
                                    taskInd, newEnqDmaBar, shvTaskInd);

                            // Check if last enqueue is just for this task and can be updated
                            // or if new one needs to be created
                            if (taskWorkloadStartIdx == lastEnqDmaData.startTaskIdx) {
                                if (!isDpuDelayedAfterShv || lastEnqDmaData.pageInd < getBarrierPage(newEnqDmaBar)) {
                                    // If this is the first time this DPU task is delayed because of SHV task or
                                    // if new delay barrier is on later page
                                    // update last enqueue DMA to use new wait barrier that is produced
                                    // by SHV. Insert this DMA after SHV task
                                    lastEnqDmaData.waitBars = {newEnqDmaBar};
                                    lastEnqDmaData.pageInd = getBarrierPage(newEnqDmaBar);
                                    lastEnqDmaData.insertBefore = newInsertBefore;
                                } else if (lastEnqDmaData.pageInd == getBarrierPage(newEnqDmaBar)) {
                                    // If this task was already delayed by some DPU and new enqueue DMA barrier
                                    // is on the same page extend wait barrier set and update insertion point
                                    lastEnqDmaData.waitBars.push_back(newEnqDmaBar);
                                    lastEnqDmaData.insertBefore =
                                            std::max(lastEnqDmaData.insertBefore, newInsertBefore);
                                }
                                _log.nest().trace("Update last enqueue DMA for task {0} to use barriers {1}", taskInd,
                                                  lastEnqDmaData.waitBars);
                            } else {
                                // Remove this task from previous one and create new one
                                lastEnqDmaData.endTaskIdx = lastEnqDmaData.endTaskIdx - taskWorkloadsCount;

                                EnqueueDmaData newEnqDmaData;
                                newEnqDmaData.pageInd = getBarrierPage(newEnqDmaBar);
                                newEnqDmaData.queueType = queueType;
                                newEnqDmaData.startTaskIdx = taskWorkloadStartIdx;
                                newEnqDmaData.endTaskIdx = taskWorkloadEndIdx;
                                newEnqDmaData.waitBars = {newEnqDmaBar};
                                newEnqDmaData.insertBefore = newInsertBefore;

                                enqDmaDataPerQueueVec.push_back(newEnqDmaData);
                                _log.nest().trace("Create new enqueue DMA for task {0} with barriers {1}", taskInd,
                                                  enqDmaDataPerQueueVec.back().waitBars);
                            }
                            isDpuDelayedAfterShv = true;

                            if (previousTaskIndOpt.has_value()) {
                                auto& lastEnqDmaDataForCheck = enqDmaDataPerQueueVec.back();
                                auto prevTask = previousTaskIndOpt.value();
                                VPURT::TaskQueueType enqueueDmaQueueType{
                                        VPU::ExecutorKind::DMA_NN,
                                        getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR)};

                                auto closestDmaTaskInd = _barrierInfo.getPrevTaskOnQueue(prevTask, enqueueDmaQueueType);
                                while (closestDmaTaskInd.has_value() &&
                                       !isDepFromTaskAToTaskB(closestDmaTaskInd.value(), prevTask)) {
                                    closestDmaTaskInd = _barrierInfo.getPrevTaskOnSameQueue(closestDmaTaskInd.value());
                                }

                                VPUX_THROW_WHEN(!closestDmaTaskInd.has_value(),
                                                "Cannot be enqueued safely with dpuFromShave execution");

                                if (lastEnqDmaDataForCheck.insertBefore <= closestDmaTaskInd.value()) {
                                    _log.nest().trace("DMA {0} is the closest task for DPU {1}",
                                                      closestDmaTaskInd.value(), prevTask);
                                    auto newInsertBefore = closestDmaTaskInd.value() + 1;
                                    _log.nest().trace("Change enqueue insertion position from {0} to {1}",
                                                      lastEnqDmaDataForCheck.insertBefore, newInsertBefore);

                                    lastEnqDmaDataForCheck.insertBefore = newInsertBefore;
                                }
                            }
                        }
                    }
                }
            }

            enqueuedWorkloads += taskWorkloadsCount;
            previousTaskIndOpt = taskInd;
            prevTaskPage = taskPage;
        }

        enqDmaDataVec.insert(enqDmaDataVec.end(), enqDmaDataPerQueueVec.begin(), enqDmaDataPerQueueVec.end());
        _log = _log.unnest();
    }
    return enqDmaDataVec;
}

// Cleanup redundant barriers
// If barrier has no consumers, remove its producers. Ignore final barrier (last barrier)
// Return status if redundant barriers were found
bool vpux::VPURT::BarrierPagesSplitHandler::cleanupRedundantBarriers() {
    _log.trace("Cleaning up redundant barriers");
    bool foundRedundantBarriers = false;
    for (size_t barInd = 0; barInd < _barrierInfo.getNumOfBarrierOps() - 1; barInd++) {
        if (_barrierInfo.getBarrierConsumers(barInd).empty()) {
            _log.trace("No consumers for barrier {0}(page {1})", barInd, getBarrierPage(barInd));
            foundRedundantBarriers = true;
            auto barProdTasks = _barrierInfo.getBarrierProducers(barInd);
            for (auto barProdTask : barProdTasks) {
                _log.trace("Remove producer {0}(page {1}) from barrier {2}(page {3})", barProdTask,
                           _taskPageAssignment[barProdTask], barInd, getBarrierPage(barInd));
                _barrierInfo.removeProducer(barInd, barProdTask);
            }
        }
    }
    return foundRedundantBarriers;
}

void vpux::VPURT::BarrierPagesSplitHandler::ensureBarrierHasProducer() {
    _log.trace("Make sure barrier has producer");

    // If barrier has no producers, add new dummy producer for it
    for (size_t barInd = 0; barInd < _barrierInfo.getNumOfBarrierOps() - 1; barInd++) {
        auto barPage = getBarrierPage(barInd);

        if (_barrierInfo.getBarrierProducers(barInd).empty()) {
            _log.trace("No producer for barrier {0}(page {1}). Attach dummy producer", barInd, barPage);
            // If barrier has no producers use first boundary task from previous page
            size_t firstTask = 0;
            if (barPage > 0) {
                firstTask = getFirstBoundaryTasksForPage(barPage - 1).front();
            }
            _barrierInfo.addProducer(barInd, firstTask);
            _log.nest().trace("Add producer {0} to barrier {1}", firstTask, barInd);
        }
    }
}

// Verify all tasks access barriers only from their page and next page
void vpux::VPURT::BarrierPagesSplitHandler::verifyTaskBarrierPagesAreValid() {
    _log.trace("Verifying task barrier pages are valid");
    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        auto waitBars = _barrierInfo.getWaitBarriers(taskInd);
        auto updateBars = _barrierInfo.getUpdateBarriers(taskInd);

        auto taskPage = _taskPageAssignment[taskInd];

        for (auto waitBar : waitBars) {
            auto waitBarPage = getBarrierPage(waitBar);
            VPUX_THROW_WHEN(waitBarPage != taskPage,
                            "For task {0}(page {1}) its wait barrier {2}(page {3}) is not on the same page", taskInd,
                            taskPage, waitBar, waitBarPage);
        }

        for (auto updateBar : updateBars) {
            auto updateBarPage = getBarrierPage(updateBar);
            VPUX_THROW_WHEN(updateBarPage != taskPage && updateBarPage != taskPage + 1,
                            "For task {0}(page {1}) its update barrier {2}(page {3}) is not on the same or next page",
                            taskInd, taskPage, updateBar, updateBarPage);
        }
    }
}

// Verification function that checks if after legalization and modification of schedule
// no cyclic dependency was created. This is just to catch such issue earlier, otherwise this problem
// would show up later during task and barrier reordering or barrier simulation
void vpux::VPURT::BarrierPagesSplitHandler::verifyNoCyclicDeps() {
    _log.trace("Verifying no cyclic deps");
    for (size_t taskInd = 0; taskInd < _barrierInfo.getNumOfTasks(); taskInd++) {
        auto updateBars = _barrierInfo.getUpdateBarriers(taskInd);

        for (auto bar : updateBars) {
            auto barConsumers = _barrierInfo.getBarrierConsumers(bar);
            for (auto barConsumer : barConsumers) {
                if (barConsumer > taskInd) {
                    continue;
                }

                if (taskInd == barConsumer) {
                    VPUX_THROW("Cyclic dependency detected - task {0}(page {1}) updates barrier {2}(page {3}) which is "
                               "consumed by the same "
                               "task",
                               taskInd, _taskPageAssignment[taskInd], bar, getBarrierPage(bar));
                }

                if (isDepFromTaskAToTaskB(barConsumer, taskInd)) {
                    VPUX_THROW("Cyclic dependency detected - task {0}(page {1}) updates barrier {2}(page {3}) which is "
                               "consumed by task "
                               "{4}(page {5}) which task {0}(page {1}) depends on",
                               taskInd, _taskPageAssignment[taskInd], bar, getBarrierPage(bar), barConsumer,
                               _taskPageAssignment[barConsumer]);
                }
            }
        }
    }
}

bool vpux::VPURT::BarrierPagesSplitHandler::verifyFetchDmaDependencies(mlir::func::FuncOp func,
                                                                       ExecutionGroupListMap& executionGroupListMap) {
    SmallVector<size_t> fetchTasks;
    DenseMap<VPUIP::FetchDMAAttr, size_t> fetchTaskMap;

    // Get all Fetch Tasks
    func.walk([&](VPURT::TaskOp taskOp) {
        if (auto fetchDMAOp = taskOp.getInnerTaskOpOfType<VPUIP::FetchDMAOp>()) {
            fetchTasks.push_back(_barrierInfo.getIndex(taskOp));
        }
    });

    // Create a map for lookup
    for (auto fetchTask : fetchTasks) {
        auto fetchTaskOp = _barrierInfo.getTaskOpAtIndex(fetchTask);
        auto fetchDMAOp = fetchTaskOp.getInnerTaskOpOfType<VPUIP::FetchDMAOp>();
        fetchTaskMap[fetchDMAOp.getFetchDmaAttr()] = fetchTask;
    }

    // For all taskOpQueues go over each execution group and verify the correctness of corresponding FetchTask in the IR
    for (auto& [_, executionGroups] : executionGroupListMap) {
        // 0 and 1 can be fetch simultaneously
        if (executionGroups.size() < 3) {
            continue;
        }
        size_t groupIdx = 2;
        auto grandParentGroup = executionGroups.front();
        auto parentGroup = executionGroups[1];
        auto travelingGroup = executionGroups[groupIdx];
        SmallVector<size_t> nextGroup;
        if (groupIdx != executionGroups.size() - 1) {
            // For the last group there is no next group
            nextGroup = executionGroups[groupIdx + 1];
        }

        while (groupIdx < executionGroups.size()) {
            auto fetchTaskAttr = getFetchDMAAttr(groupIdx, _barrierInfo, executionGroups[groupIdx].front());
            auto fetchTask = fetchTaskMap[fetchTaskAttr];
            // Is FetchTask for Group N dependent on last task of Group N-2
            if (!isDepFromTaskAToTaskB(grandParentGroup[grandParentGroup.size() - 1], fetchTask)) {
                _log.trace("Fetch task {0} is not dependent on last task of Group N-2 {1}", fetchTask,
                           grandParentGroup[grandParentGroup.size() - 1]);
                return false;
            }

            // Is last task of group N+1 dependent on fetchTask for group N
            if (!nextGroup.empty()) {
                if (!isDepFromTaskAToTaskB(fetchTask, nextGroup[nextGroup.size() - 1])) {
                    _log.trace("Last task of Group N+1 {0} is not dependent on fetch task {1}",
                               nextGroup[nextGroup.size() - 1], fetchTask);
                    return false;
                }
            }

            ++groupIdx;
            grandParentGroup = parentGroup;
            parentGroup = travelingGroup;
            if (groupIdx < executionGroups.size()) {
                travelingGroup = executionGroups[groupIdx];
                if (groupIdx != executionGroups.size() - 1) {
                    nextGroup = executionGroups[groupIdx + 1];
                }
            }
        }
    }
    return true;
}

// For all physical barriers check if given virtual barrier using PIDx
// is updated after ALL consumers of previous barrier using same PIDx are guaranteed
// to start
void vpux::VPURT::BarrierPagesSplitHandler::verifyPhysicalBarsDependencies() {
    VPUX_THROW_UNLESS(_barrierPidPrevUsageVec.size() == _barrierInfo.getNumOfBarrierOps(),
                      "Barrier PID previous usage vector size {0} does not match number of barriers {1}",
                      _barrierPidPrevUsageVec.size(), _barrierInfo.getNumOfBarrierOps());

    for (size_t barInd = 0; barInd < _barrierInfo.getNumOfBarrierOps(); barInd++) {
        auto prevBarInd = _barrierPidPrevUsageVec[barInd];

        if (!prevBarInd.has_value()) {
            continue;
        }

        auto prevBarConsumers = _barrierInfo.getBarrierConsumers(prevBarInd.value());
        auto barProducers = _barrierInfo.getBarrierProducers(barInd);

        // Check if each bar producer depends on all consumers of previous barrier
        // This way it will guarantee that barrier is consumed and reloaded for next usage
        // no matter the timings of tasks
        for (auto barProd : barProducers) {
            for (auto prevBarConsumer : prevBarConsumers) {
                if (!isDepFromTaskAToTaskB(prevBarConsumer, barProd)) {
                    VPUX_THROW("VID {0} is not guaranteed to be consumed before VID {1} is produced. No dependency "
                               "from {2} to {3}",
                               prevBarInd.value(), barInd, prevBarConsumer, barProd);
                }
            }
        }
    }
}

// Verify dependencies around Barrier Programming DMA:
// - BarProgDMAOp is guaranteed to be after all barriers from previous page
// - BarProgDMAOp is guaranteed to be before all barriers from next page
// - BarProgDMAOp has correct physical barrier range assigned
// This is needed to guarantee that all barriers are programmed before any task that uses them
// and that no task can use barriers from next page before they are programmed
void vpux::VPURT::BarrierPagesSplitHandler::verifyBarProgDmaDependencies(mlir::func::FuncOp func) {
    if (_pageCount <= 2) {
        _log.trace("No need to verify BarProgDMA dependencies if model has {0} <= 2 pages", _pageCount);
        return;
    }

    // Identify BarProgDmas indexes
    SmallVector<size_t> barProgDmas;
    func->walk([&](VPURT::TaskOp taskOp) {
        auto barProgDmaOp = mlir::dyn_cast<VPUIP::BarProgDMAOp>(taskOp.getInnerTaskOp());
        if (barProgDmaOp == nullptr) {
            return;
        }

        auto pageOpt = taskOp.getWlmPage();
        VPUX_THROW_UNLESS(pageOpt.has_value(), "BarProgDMAOp '{0}' does not have WLM page assigned", taskOp.getLoc());
        auto pageInd = pageOpt.value();
        auto physBarRangeAttr = barProgDmaOp.getPhysicalBarrierRange();
        auto pidStart = physBarRangeAttr.getStart().getValue().getSExtValue();
        auto pidEnd = physBarRangeAttr.getEnd().getValue().getSExtValue();

        _log.trace("Found BarProgDMAOp at page {0} with physical barrier range {1}-{2}", pageInd, pidStart, pidEnd);

        int64_t expectedPageStart, expectedPageEnd;
        if (pageInd == 0) {
            // BootstrapDMA in page 0 programs all barriers
            expectedPageStart = 0;
            expectedPageEnd = (_pageSize * 2) - 1;
        } else {
            // Other pages program only half (lower or upper)
            auto pidOffset = ((pageInd - 1) % 2) * _pageSize;
            expectedPageStart = pidOffset;
            expectedPageEnd = pidOffset + _pageSize - 1;
        }
        VPUX_THROW_WHEN(pidStart != expectedPageStart || pidEnd != expectedPageEnd,
                        "BarProgDMAOp at page {0} has unexpected physical barrier range {1}-{2}, expected "
                        "page range {3}-{4}",
                        pageInd, pidStart, pidEnd, expectedPageStart, expectedPageEnd);

        barProgDmas.push_back(_barrierInfo.getIndex(taskOp));
    });

    if (barProgDmas.empty()) {
        return;
    }

    // Collect information about barriers used by pages
    SmallVector<SmallVector<size_t>> barriersPerPage(_pageCount, SmallVector<size_t>());

    func.walk([&](VPURT::ConfigureBarrierOp barOp) {
        auto vid = _barrierInfo.getIndex(barOp);
        auto pageInd = getBarrierPage(vid);
        barriersPerPage[pageInd].push_back(vid);
    });

    // Check if each page has exactly _pageSize barriers
    for (size_t pageInd = 0; pageInd < _pageCount - 2; ++pageInd) {
        auto barriers = barriersPerPage[pageInd];
        VPUX_THROW_WHEN(barriers.size() != _pageSize, "Page {0} has {1} barriers, expected {2}", pageInd,
                        barriers.size(), _pageSize);
    }

    // Check if each BarProgDma is after all barriers from previous page and before
    // all barriers from next page
    for (auto barProgDma : barProgDmas) {
        auto pageInd = _taskPageAssignment[barProgDma];
        _log.trace("Check BarProgDMA {0} at page {1}", barProgDma, pageInd);

        if (pageInd == 0) {
            // First DMA to program barriers at bootstrap is inserted at pageInd 0
            // This DMA must have index 0 i.e. it must be the very first DMA in schedule to ensure barriers are
            // programmed
            // If there are any other tasks before this DMA then schedule is unsafe
            VPUX_THROW_WHEN(barProgDma != 0,
                            "BarProgDMA {0} at bootstrap at pageInd {1} is not the first task in schedule", barProgDma,
                            pageInd);

            // Check page barrier producers if barrier programming DMA is first DMA in schedule
            for (auto pageBar : barriersPerPage[pageInd]) {
                auto pageBarProducers = to_small_vector(_barrierInfo.getBarrierProducers(pageBar));

                // Filter out relatedTasks to be checked i.e. DMA tasks only
                auto isDmaProducer = [&](auto task) {
                    return _barrierInfo.getTaskQueueType(task).type == VPU::ExecutorKind::DMA_NN;
                };

                auto filteredRange = pageBarProducers | vpux::filtered(std::move(isDmaProducer));
                auto pageDmaBarProducers = to_small_vector(filteredRange);

                for (auto pageDmaBarProducer : pageDmaBarProducers) {
                    VPUX_THROW_UNLESS(
                            isDepFromTaskAToTaskB(barProgDma, pageDmaBarProducer),
                            "BarProgDMA {0} at page {1} is not guaranteed to be before producer {2} of barrier {3} "
                            "from same page",
                            barProgDma, pageInd, pageDmaBarProducer, pageBar);
                }
            }
        } else {
            // Check prev page barrier consumers if barrier programming DMA depends on them
            for (auto prevPageBar : barriersPerPage[pageInd - 1]) {
                auto prevPageBarConsumers = _barrierInfo.getBarrierConsumers(prevPageBar);
                for (auto prevPageBarConsumer : prevPageBarConsumers) {
                    VPUX_THROW_UNLESS(
                            isDepFromTaskAToTaskB(prevPageBarConsumer, barProgDma),
                            "BarProgDMA {0} at page {1} is not guaranteed to be after consumer {2} of barrier "
                            "{3} from previous page",
                            barProgDma, pageInd, prevPageBarConsumer, prevPageBar);
                }
            }

            // Check next page barrier producers if it depends on barrier programming DMA
            for (auto nextPageBar : barriersPerPage[pageInd + 1]) {
                auto nextPageBarProducers = _barrierInfo.getBarrierProducers(nextPageBar);
                for (auto nextPageBarProducer : nextPageBarProducers) {
                    VPUX_THROW_UNLESS(
                            isDepFromTaskAToTaskB(barProgDma, nextPageBarProducer),
                            "BarProgDMA {0} at page {1} is not guaranteed to be before producer {2} of barrier "
                            "{3} from next page",
                            barProgDma, pageInd, nextPageBarProducer, nextPageBar);
                }
            }
        }
    }
}

// Check all enqueue DMAs and verify if tasks enqueued by them do not violate enqueue restrictions:
// - task can be enqueued only at moment when all its wait prev bar instances are consumed at that point
// - OR if execution of previous task on the same queue guarantees the same
// This function also checks if enqueued task update barriers are ready at the moment of task execution
void vpux::VPURT::BarrierPagesSplitHandler::verifyEnqueueDmas(mlir::func::FuncOp func) {
    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = config::getArch(module);
    auto numClusters = config::getTileExecutor(module).getCount();

    // Identify DPU and SHV queues and store corresponding task idexes.
    // This is needed to later be able to map EnqueueDMAOp enqueued range of workloads to task indexes
    mlir::DenseMap<std::tuple<VPU::ExecutorKind, size_t, size_t>, SmallVector<size_t>> taskIndexesPerQueueType;
    func->walk([&](VPURT::TaskOp taskOp) {
        auto taskInd = _barrierInfo.getIndex(taskOp);
        auto queueType = _barrierInfo.getTaskQueueType(taskInd);
        if (queueType.type != VPU::ExecutorKind::DPU && queueType.type != VPU::ExecutorKind::SHAVE_ACT) {
            return;
        }

        auto [tileIdx, listIdx] = VPURT::getTileAndListIndex(queueType, numClusters, arch);
        auto execKindAndTileAndList = std::make_tuple(queueType.type, tileIdx, listIdx);
        // Ops may have multiple workloads. For example DPU tasks have multiple variants
        // Eventually all of them are enqueued together so they all correspond to the same
        // task index (e.g. same NCEClusterTaskOp)
        for (size_t workloadIdx = 0; workloadIdx < getNumberOfWorkloads(taskInd); ++workloadIdx) {
            taskIndexesPerQueueType[execKindAndTileAndList].push_back(taskInd);
        }
    });

    mlir::DenseMap<std::tuple<VPU::ExecutorKind, size_t, size_t>, size_t> lastTaskIndexesEnqueuedPerQueueType;

    // Identify EnqueueDmas indexes and task indexes that are enqueued by it
    SmallVector<std::pair<size_t, SmallVector<size_t>>> enqueueDmasAndTaskIndexes;
    func->walk([&](VPURT::TaskOp taskOp) {
        auto enqueueDmaOp = mlir::dyn_cast<VPUIP::EnqueueDMAOp>(taskOp.getInnerTaskOp());
        if (enqueueDmaOp == nullptr) {
            return;
        }

        auto pageOpt = taskOp.getWlmPage();
        VPUX_THROW_UNLESS(pageOpt.has_value(), "EnqueueDMAOp '{0}' does not have WLM page assigned", taskOp.getLoc());
        auto pageInd = pageOpt.value();
        auto enqueueDmaIdx = _barrierInfo.getIndex(taskOp);
        auto enqueueDmaAttr = enqueueDmaOp.getEnqueueDmaAttr();

        auto executorKind = enqueueDmaAttr.getTargetExecutorKindAttr().getValue();
        auto tileIdx = enqueueDmaAttr.getTileIdx().getValue().getSExtValue();
        auto listIdx = enqueueDmaAttr.getListIdx().getValue().getSExtValue();
        auto startTaskIdx = enqueueDmaAttr.getStartTaskIdx().getValue().getSExtValue();
        auto endTaskIdx = enqueueDmaAttr.getEndTaskIdx().getValue().getSExtValue();

        _log.trace("Found EnqueueDMAOp task {0} at page {1} for tasks {2}:{3}:{4}:{5}-{6}", pageInd,
                   stringifyEnum(executorKind), enqueueDmaIdx, tileIdx, listIdx, startTaskIdx, endTaskIdx);

        // Identify task indexes fr this enqueue op
        SmallVector<size_t> taskIndexes;

        auto execKindAndTileAndList = std::make_tuple(executorKind, tileIdx, listIdx);

        // Make sure enqueued task indexes are in growing order and that no task is missed
        if (lastTaskIndexesEnqueuedPerQueueType.count(execKindAndTileAndList) > 0) {
            VPUX_THROW_WHEN(
                    static_cast<size_t>(startTaskIdx) <= lastTaskIndexesEnqueuedPerQueueType[execKindAndTileAndList],
                    "EnqueueDMAOp {0} has start task index {1} that is less than or equal to last "
                    "enqueued task index {2} for queue type {3}:{4}:{5}",
                    enqueueDmaIdx, startTaskIdx, lastTaskIndexesEnqueuedPerQueueType[execKindAndTileAndList],
                    VPU::stringifyExecutorKind(executorKind), tileIdx, listIdx);
            VPUX_THROW_WHEN(static_cast<size_t>(startTaskIdx) !=
                                    lastTaskIndexesEnqueuedPerQueueType[execKindAndTileAndList] + 1,
                            "EnqueueDMAOp {0} has start task index {1} that is not equal to last enqueued task "
                            "index {2} + 1 for queue type {3}:{4}:{5}",
                            enqueueDmaIdx, startTaskIdx, lastTaskIndexesEnqueuedPerQueueType[execKindAndTileAndList],
                            VPU::stringifyExecutorKind(executorKind), tileIdx, listIdx);
        }
        lastTaskIndexesEnqueuedPerQueueType[execKindAndTileAndList] = endTaskIdx;

        for (size_t perQueueTaskInd = startTaskIdx; perQueueTaskInd <= static_cast<size_t>(endTaskIdx);
             ++perQueueTaskInd) {
            VPUX_THROW_UNLESS(taskIndexesPerQueueType.count(execKindAndTileAndList) > 0,
                              "No task indexes found for queue type {0} at page {1}",
                              VPU::stringifyExecutorKind(executorKind), pageInd);
            VPUX_THROW_UNLESS(perQueueTaskInd < taskIndexesPerQueueType[execKindAndTileAndList].size(),
                              "Task index {0} is out of range ({1}) for queue type {2} at page {3}", perQueueTaskInd,
                              taskIndexesPerQueueType[execKindAndTileAndList].size(),
                              VPU::stringifyExecutorKind(executorKind), pageInd);
            taskIndexes.push_back(taskIndexesPerQueueType[execKindAndTileAndList][perQueueTaskInd]);
        }

        enqueueDmasAndTaskIndexes.push_back(std::make_pair(_barrierInfo.getIndex(taskOp), taskIndexes));
    });

    if (enqueueDmasAndTaskIndexes.empty()) {
        return;
    }

    // For each identified enqueue DMA check if enqueued tasks do not violate enqueue restrictions:
    // - task can be enqueued only at moment when all its wait prev bar instances are consumed at that point
    // - OR if execution of previous task on the same queue guarantees the same
    for (auto& [enqueueDmaInd, taskIndexes] : enqueueDmasAndTaskIndexes) {
        _log.trace("Check EnqueueDMAOp {0} for tasks {1}", enqueueDmaInd, to_small_vector(taskIndexes));

        size_t prevTaskInd = 0;
        size_t taskInd = 0;
        for (size_t i = 0; i < taskIndexes.size(); prevTaskInd = taskInd, i += getNumberOfWorkloads(taskInd)) {
            taskInd = taskIndexes[i];
            auto taskPage = _taskPageAssignment[taskInd];
            _log.trace("Check task {0}(page {1})", taskInd, taskPage);

            // Get wait barriers for the task
            auto waitBars = _barrierInfo.getWaitBarriers(taskInd);
            auto prevWaitBars = getBarrierPidPrevUsageVec(waitBars);

            if (!prevWaitBars.empty()) {
                // Store tasks against which dependency is checked. If any satisfies the conditions then enqueue is
                // valid
                SmallVector<size_t> tasksToCheckDeps;
                if (taskInd != taskIndexes.front()) {
                    _log.nest().trace("Task {0} is enqueued together with previous {1}", taskInd, prevTaskInd);
                    tasksToCheckDeps.push_back(prevTaskInd);
                } else {
                    _log.nest().trace("Task {0} is first enqueued by DMA", taskInd);
                }
                // Check if all previous wait barriers consumers are guaranteed to run before enqueue DMA
                tasksToCheckDeps.push_back(enqueueDmaInd);

                for (auto prevBar : prevWaitBars) {
                    auto prevBarConsumers = _barrierInfo.getBarrierConsumers(prevBar);
                    _log.nest(2).trace("Check consumers of previous wait barrier {0}", prevBar);
                    for (auto prevBarConsumer : prevBarConsumers) {
                        bool isDepSatisfied = false;
                        for (auto taskToCheckDeps : tasksToCheckDeps) {
                            _log.nest(3).trace("Check dependency from {0} to {1}", prevBarConsumer, taskToCheckDeps);
                            isDepSatisfied = isDepFromTaskAToTaskB(prevBarConsumer, taskToCheckDeps);
                            if (isDepSatisfied) {
                                break;
                            }
                        }

                        VPUX_THROW_UNLESS(isDepSatisfied,
                                          "Task {0}(page {1})\n{2}\n cannot be enqueued at {3}(page {4})\n{5}\n "
                                          "because it is not "
                                          "guaranteed that previous wait barriers consumer {6}(page {7})\n{8}\n is "
                                          "executed before "
                                          "enqueue or previous task {9}(page {10})\n{11}\n is executed",
                                          taskInd, _taskPageAssignment[taskInd], _barrierInfo.getTaskOpAtIndex(taskInd),
                                          enqueueDmaInd, _taskPageAssignment[enqueueDmaInd],
                                          _barrierInfo.getTaskOpAtIndex(enqueueDmaInd), prevBarConsumer,
                                          _taskPageAssignment[prevBarConsumer],
                                          _barrierInfo.getTaskOpAtIndex(prevBarConsumer), prevTaskInd,
                                          _taskPageAssignment[prevTaskInd], _barrierInfo.getTaskOpAtIndex(prevTaskInd));
                    }
                }
            }

            // Check if when task runs all previous instances of update barriers were consumed
            auto updateBars = _barrierInfo.getUpdateBarriers(taskInd);
            auto prevUpdateBars = getBarrierPidPrevUsageVec(updateBars);

            if (prevUpdateBars.empty()) {
                continue;
            }

            for (auto prevBar : prevUpdateBars) {
                auto prevBarConsumers = _barrierInfo.getBarrierConsumers(prevBar);
                _log.nest(2).trace("Check consumers of previous update barrier {0}", prevBar);
                for (auto prevBarConsumer : prevBarConsumers) {
                    _log.nest(3).trace("Check dependency from {0} to {1}", prevBarConsumer, taskInd);
                    VPUX_THROW_UNLESS(isDepFromTaskAToTaskB(prevBarConsumer, taskInd),
                                      "Consumer {0} of previous instance of update barrier {1} is not guaranteed to be "
                                      "consumed when task {2} starts",
                                      prevBarConsumer, prevBar, taskInd);
                }
            }
        }
    }
}

// Verify enqueue of DMAs. All DMAs are expected to be enqueued at bootstrap as a single link list.
// Check if first DMA of each queue if exists is present in Page0 or Page1
// For all next DMAs check that execution of previous task on the same queue guarantees that all
// its wait prev barrier instances are consumed before it
// This function also checks if task update barriers are ready at the moment of task execution
void vpux::VPURT::BarrierPagesSplitHandler::verifyEnqueueOfDmas(mlir::func::FuncOp func) {
    auto taskQueues = VPURT::getTaskOpQueues(func, _barrierInfo);
    _log.trace("Verifying enqueue of DMA tasks");
    for (const auto& [queueType, taskOps] : taskQueues) {
        if (queueType.type != VPU::ExecutorKind::DMA_NN) {
            continue;
        }

        _log.trace("Check queue type {0}:{1} with {2} tasks", VPU::stringifyExecutorKind(queueType.type), queueType.id,
                   taskOps.size());

        // First task needs to be present if Page0 or Page1
        auto pageInd = _taskPageAssignment[taskOps.front()];
        VPUX_THROW_UNLESS(pageInd <= 1, "First task in DMA queue {0}:{1} is not on Page0 or Page1, but on Page{2}.",
                          VPU::stringifyExecutorKind(queueType.type), queueType.id, pageInd);

        for (size_t i = 1; i < taskOps.size(); ++i) {
            auto prevTaskInd = taskOps[i - 1];
            auto taskInd = taskOps[i];

            // Check if prev instances of wait barriers of current task are guaranteed to be
            // consumed when previous task runs
            // Get wait barriers for the task
            auto waitBars = _barrierInfo.getWaitBarriers(taskInd);
            auto prevWaitBars = getBarrierPidPrevUsageVec(waitBars);

            if (!prevWaitBars.empty()) {
                for (auto prevBar : prevWaitBars) {
                    auto prevBarConsumers = _barrierInfo.getBarrierConsumers(prevBar);
                    _log.nest(2).trace("Check consumers of previous wait barrier {0}", prevBar);
                    for (auto prevBarConsumer : prevBarConsumers) {
                        _log.nest(3).trace("Check dependency from {0} to {1}", prevBarConsumer, prevTaskInd);
                        VPUX_THROW_UNLESS(isDepFromTaskAToTaskB(prevBarConsumer, prevTaskInd),
                                          "Task {0} cannot be enqueued together with {1} because it is not "
                                          "guaranteed that previous wait barriers consumer {2} is executed before "
                                          "previous task {1} is executed",
                                          taskInd, prevTaskInd, prevBarConsumer);
                    }
                }
            }

            // Check if when task runs all previous instances of update barriers were consumed
            auto updateBars = _barrierInfo.getUpdateBarriers(taskInd);
            auto prevUpdateBars = getBarrierPidPrevUsageVec(updateBars);

            if (prevUpdateBars.empty()) {
                continue;
            }

            for (auto prevBar : prevUpdateBars) {
                auto prevBarConsumers = _barrierInfo.getBarrierConsumers(prevBar);
                _log.nest(2).trace("Check consumers of previous update barrier {0}", prevBar);
                for (auto prevBarConsumer : prevBarConsumers) {
                    _log.nest(3).trace("Check dependency from {0} to {1}", prevBarConsumer, taskInd);
                    VPUX_THROW_UNLESS(isDepFromTaskAToTaskB(prevBarConsumer, taskInd),
                                      "Consumer {0} of previous instance of update barrier {1} is not guaranteed to be "
                                      "consumed when task {2} starts",
                                      prevBarConsumer, prevBar, taskInd);
                }
            }
        }
    }
}

// Perform  legalization of long dependencies of tasks
void vpux::VPURT::BarrierPagesSplitHandler::legalizeLongDependenciesForBarrierPagesSplit() {
    auto tasksToLegalize = getTasksWithNonAdjacentPageDependencyToLegalize();
    legalizeNonAdjacentPageDependencies(tasksToLegalize);
}

// Make sure boundary tasks from neighbor pages are dependent
void vpux::VPURT::BarrierPagesSplitHandler::legalizeBoundaryTasksForBarrierPagesSplit() {
    auto boundaryTaskPairsMissingDepInBetween = getBoundaryTaskPairsMissingDepInBetween();
    legalizeDepsForBoundaryTasks(boundaryTaskPairsMissingDepInBetween);
}

vpux::BarrierInfo vpux::VPURT::BarrierPagesSplitHandler::getUpdatedBarrierInfo() {
    return _barrierInfo;
}

// Helper method for unit testing BarrierPagesSplitHandler
BarrierInfoMaps vpux::VPURT::BarrierPagesSplitHandler::getBarrierMaps() {
    return vpux::getBarrierMaps(_barrierInfo);
}
