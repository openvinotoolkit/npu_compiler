//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"

namespace vpux {
namespace VPUMI40XX {

//
// AddEnqueue Utils
//

bool contains(const llvm::SmallVector<mlir::Value>& vec, const mlir::Value& element) {
    return std::find(vec.begin(), vec.end(), element) != vec.end();
};

VPUMI40XX::ConfigureBarrierOp getBarrierOp(mlir::Operation* op) {
    if (op == nullptr) {
        return nullptr;
    }

    auto maybeBarrier = mlir::dyn_cast_or_null<VPUMI40XX::ConfigureBarrierOp>(op);
    return maybeBarrier;
}

size_t getBarrierIndex(mlir::Operation* op) {
    auto barrierOp = getBarrierOp(op);
    VPUX_THROW_WHEN(barrierOp == nullptr, "Expected barrier: got {0}", op);
    return barrierOp.getType().getValue();
};

bool taskOpComparator(mlir::Operation* lhs, mlir::Operation* rhs) {
    auto lhsTask = mlir::cast<VPURegMapped::TaskOpInterface>(lhs);
    auto rhsTask = mlir::cast<VPURegMapped::TaskOpInterface>(rhs);
    return lhsTask.getIndexType().getValue() < rhsTask.getIndexType().getValue();
}

// Function to get the maximum barrier based on their type values(virtual id)
mlir::Value* getMaxBarrier(SmallVector<mlir::Value>& barriers) {
    return std::max_element(barriers.begin(), barriers.end(), [](mlir::Value lhs, mlir::Value rhs) {
        return mlir::cast<VPUMI40XX::ConfigureBarrierOp>(lhs.getDefiningOp()).getType().getValue() <
               mlir::cast<VPUMI40XX::ConfigureBarrierOp>(rhs.getDefiningOp()).getType().getValue();
    });
}

// Function to get the minimum barrier based on their type values(virtual id)
mlir::Value* getMinBarrier(SmallVector<mlir::Value>& barriers) {
    return std::min_element(barriers.begin(), barriers.end(), [](mlir::Value lhs, mlir::Value rhs) {
        return mlir::cast<VPUMI40XX::ConfigureBarrierOp>(lhs.getDefiningOp()).getType().getValue() <
               mlir::cast<VPUMI40XX::ConfigureBarrierOp>(rhs.getDefiningOp()).getType().getValue();
    });
}

void reindexEnqueueOps(llvm::SmallVector<VPURegMapped::EnqueueOp> enquOps) {
    if (enquOps.size() == 0) {
        return;
    }

    auto ctx = enquOps[0].getContext();
    auto index = [&ctx](auto taskIdx) {
        return VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(taskIdx));
    };

    enquOps[0].getResult().setType(index(0));
    enquOps[0].getPreviousTaskIdxMutable().clear();

    for (size_t i = 1; i < enquOps.size(); i++) {
        auto enqu = enquOps[i];
        enqu.getResult().setType(index(i));
        enqu.getPreviousTaskIdxMutable().assign(enquOps[i - 1]);
    }

    return;
}

mlir::ValueRange getClosestProductionBarriers(VPURegMapped::TaskOpInterface taskOp) {
    do {
        auto executableTaskOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(taskOp.getOperation());
        if (executableTaskOp && (executableTaskOp.updateBarriers().size() != 0)) {
            return executableTaskOp.updateBarriers();
        }

        auto taskOpUsers = taskOp.getOperation()->getResult(0).getUsers();
        auto nextTaskDown = llvm::find_if(taskOpUsers, [&taskOp](mlir::Operation* user) {
            auto next = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(user);
            return next && (next.getPreviousTask() == taskOp);
        });

        taskOp = nextTaskDown != taskOpUsers.end() ? mlir::cast<VPURegMapped::TaskOpInterface>(*nextTaskDown) : nullptr;

    } while (taskOp);

    return mlir::ValueRange{};
}

void dfs(mlir::Value val, llvm::SetVector<mlir::Value>& visited, size_t indexMax) {
    visited.insert(val);
    for (auto user : val.getUsers()) {
        auto barOp = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(user);
        if (!barOp)
            continue;

        auto bar = barOp.getResult();

        auto barIndex = bar.getType().cast<VPURegMapped::IndexType>().getValue();
        if (barIndex > indexMax) {
            continue;
        }

        if (!visited.contains(bar)) {
            dfs(bar, visited, indexMax);
        }
    }
}

llvm::SmallVector<mlir::Value> lca(mlir::Value lhs, mlir::Value rhs, lcaCache& cache, size_t indexMax) {
    if (lhs == rhs)
        return {lhs};

    auto lhsBar = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(lhs.getDefiningOp());
    auto rhsBarr = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(rhs.getDefiningOp());
    auto lhsPos = lhsBar.getType().cast<VPURegMapped::IndexType>().getValue();
    auto rhsPos = rhsBarr.getType().cast<VPURegMapped::IndexType>().getValue();

    if (lhsPos > rhsPos) {
        std::swap(lhs, rhs);
    }

    if (cache.contains({lhsPos, rhsPos})) {
        return cache[{lhsPos, rhsPos}];
    }

    llvm::SmallVector<mlir::Value> lcas;

    llvm::SetVector<mlir::Value> visitedLhs, visitedRhs;
    llvm::SetVector<mlir::Value> intersection;

    dfs(lhs, visitedLhs, indexMax);
    dfs(rhs, visitedRhs, indexMax);

    // get the intersection of the 2
    for (auto lhsIt : visitedLhs) {
        if (visitedRhs.contains(lhsIt)) {
            intersection.insert(lhsIt);
        }
    }

    // each barr who's deps is not in the intersection is an LCA
    for (auto val : intersection) {
        auto barr = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(val.getDefiningOp());
        auto count = llvm::count_if(barr.getDependencies(), [&intersection](mlir::Value val) {
            return intersection.contains(val);
        });
        if (count == 0) {
            lcas.push_back(val);
        }
    }
    cache[{lhsPos, rhsPos}] = lcas;
    return lcas;
}

llvm::SmallVector<mlir::Value> lca(llvm::SmallVector<mlir::Value>& lhs, mlir::Value rhs, lcaCache& cache,
                                   size_t indexMax) {
    llvm::SmallVector<mlir::Value> lcas;

    for (auto val : lhs) {
        lcas.append(lca(val, rhs, cache, indexMax));
    }

    return lcas;
}

// For given barrier get other barrier that depend on it.
// Example bar-to-bar dependencies graph:
// bar0 --> bar1
//    |---> bar2
// Dependent barrier for bar0 are {bar1, bar2}
llvm::SmallVector<mlir::Value> getDependentBarriers(mlir::Value bar) {
    auto barOp = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(bar.getDefiningOp());

    SmallVector<mlir::Value> dependentBarriers;

    for (auto user : barOp.getBarrier().getUsers()) {
        if (auto userBarOp = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(user)) {
            auto depBarIt = llvm::find_if(userBarOp.getDependencies(), [bar](mlir::Value userBarOpDep) {
                return userBarOpDep == bar;
            });
            if (depBarIt != userBarOp.getDependencies().end()) {
                dependentBarriers.push_back(userBarOp.getBarrier());
            }
        }
    }

    return dependentBarriers;
}

// Find enqueue barrier using LCA for given set of barrier
// For those barrier LCA will be run on dependent barriers as enqueue target
// requires barriers consumption guarantee
// Example graph:
//   V0 -> V1 -> V2
//    |-------- >|
// Barriers provided as argument:
//   {V0, V1}
// Dependent barriers:
//   V0: {V1, V2}
//   V1: {V2}
// LCA arguments:
//   LCA({V1, V2})
// Result:
//   V2
// So at V2 consumption is is guaranted that V0 and V1 have been also
// already consumed.
//
// IndexMax is barrier index beyond which traversal of barriers is not done
// to reduce compile time
mlir::Value findEnqTargetUsingLcaForBars(llvm::SmallVector<mlir::Value>& barrierVals, lcaCache& cache,
                                         size_t indexMax) {
    // sanity... only makes sense in debug modes
    assert(std::all_of(barrierVals.begin(), barrierVals.end(),
                       [](mlir::Value val) {
                           return mlir::isa<VPUMI40XX::ConfigureBarrierOp>(val.getDefiningOp());
                       }) &&
           "LCA requires all of the values to be defined by configureBarrierOps {0}");

    if (barrierVals.empty()) {
        return nullptr;
    }

    if (barrierVals.size() == 1) {
        return barrierVals[0];
    }

    // Stage 1:
    // At first check if latest barrier (largest index) from provided barriers is a good LCA candidate
    // This is to cover case where for the given set of barriers, latest barrier is already an LCA
    // barrier for all other barriers. We can check only max index barrier because barriers with smaller
    // index cannot be dependent barrier of barrier with larger index by definition
    //
    // Find barrier with largest index:
    auto maxBar = *getMaxBarrier(barrierVals);

    mlir::DenseSet<mlir::Value> barriersForLcaSet = {maxBar};

    // Prepare a set with users of other barriers
    for (auto& bar : barrierVals) {
        // Skip maxBar as at first check to not delay enqueue of task
        // we want to find out if the latest barrier is a good candidate for LCA
        if (bar == maxBar) {
            continue;
        }
        // Check barriers which depend on provided barrier to identify barriers
        // whose production (all of them), indicate consumption of given barrier
        auto depBars = getDependentBarriers(bar);
        if (!depBars.empty()) {
            barriersForLcaSet.insert(depBars.begin(), depBars.end());
        } else {
            // if barrier has no dependent barriers use given barrier
            barriersForLcaSet.insert(bar);
        }
    }

    // Perform LCA on given set of barriers
    auto getLcaOnSetOfBarriers = [&](mlir::DenseSet<mlir::Value>& barriersSet) {
        auto barrierIt = barriersSet.begin();
        SmallVector<mlir::Value> lcaResults = {*barrierIt};
        while (++barrierIt != barriersSet.end()) {
            lcaResults = lca(lcaResults, *barrierIt, cache, indexMax);
        }
        return lcaResults;
    };

    // Get LCA result for barriers
    auto lcaResVec = getLcaOnSetOfBarriers(barriersForLcaSet);

    // If result vector is empty then return nullptr
    // as enqueue target barrier cannot be found using LCA
    if (lcaResVec.empty()) {
        return nullptr;
    }

    // Check if maxBar is good LCA candidate by comparing it
    // with the earliest barrier (min index) from LCA results
    auto lcaCandid = *getMinBarrier(lcaResVec);
    if (lcaCandid == maxBar) {
        return lcaCandid;
    }

    // If not the proceed to stage 2 search approach

    // Stage 2:
    // Take into account also maxBar users so that LCA can be done over
    // all dependent barriers and identified LCA candidate guarantees
    // all provided barriers consumption event happened

    // Add users of maxBar to set of barriers for LCA
    auto depBars = getDependentBarriers(maxBar);
    if (!depBars.empty()) {
        // Remove maxBar from barriersForLca
        barriersForLcaSet.erase(maxBar);
        // Add its users to set for LCA search
        barriersForLcaSet.insert(depBars.begin(), depBars.end());
    }

    // Get LCA result for barriers
    lcaResVec = getLcaOnSetOfBarriers(barriersForLcaSet);

    // If result vector is empty then return nullptr
    // to indicate no enqueue target barrier was found using LCA
    if (lcaResVec.empty()) {
        return nullptr;
    }

    // Return earliest barrier (min index) from LCA results as
    // enqueue should be done as early as possible
    return *getMinBarrier(lcaResVec);
}

// Get index of barrier which will be a limit for LCA search.
size_t getLcaSearchLimit(SmallVector<mlir::Value>& barriers) {
    // Get a limit of LCA search to largest barrier index of next barriers
    // using same PID as barriers used by task itself. This is compile time
    // optimization step to not do DFS (within LCA) beyond barriers which
    // most likely cannot be used as enqueue targets as they are much further in schedule
    // than task to be enqueued. This limit reduce compile time, but at the same
    // does not guarantee LCA will find any ancestor
    size_t indexMax = 0;
    for (auto& bar : barriers) {
        auto barOp = bar.getDefiningOp<VPUMI40XX::ConfigureBarrierOp>();
        auto nextSameId = barOp.getNextSameId();
        if (nextSameId < 0) {
            // If next barrier with same PID is -1 this means that given barrier
            // is the last instance with this PID. In that case do not limit
            // LCA search as used barriers are close to the end of schedule
            indexMax = std::numeric_limits<size_t>::max();
            break;
        }
        indexMax = std::max(indexMax, static_cast<size_t>(nextSameId));
    }

    return indexMax;
}

VPURegMapped::TaskOpInterface getNextOp(VPURegMapped::TaskOpInterface op) {
    auto users = op.getResult().getUsers();
    auto nexOpIt = llvm::find_if(users, [&op](mlir::Operation* user) {
        auto nextTask = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(user);
        return nextTask && (nextTask.getTaskType() == op.getTaskType()) && (nextTask.getPreviousTask() == op);
    });

    op = nexOpIt != users.end() ? mlir::cast<VPURegMapped::TaskOpInterface>(*nexOpIt) : nullptr;
    return op;
}

llvm::SmallVector<mlir::Value> getPreviousUsages(mlir::ValueRange barrs) {
    llvm::SmallDenseSet<mlir::Value> seenBarriers;
    llvm::SmallVector<mlir::Value> previousUsages;

    for (auto barr : barrs) {
        auto barrOp = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(barr.getDefiningOp());
        if (barrOp == nullptr) {
            continue;
        }

        const auto previousSameId = barrOp.getPreviousSameId();
        if (previousSameId != nullptr && !seenBarriers.contains(previousSameId)) {
            seenBarriers.insert(previousSameId);
            previousUsages.push_back(previousSameId);
        }
    }

    return previousUsages;
}

// TODO E#132327: ned to figure out a clean way to get barriers purely from taskOpInterface
VPUMI40XX::ExecutableTaskOpInterface getBarrieredOp(VPURegMapped::TaskOpInterface primary,
                                                    VPURegMapped::TaskOpInterface secondary) {
    if (primary.getTaskType() == VPURegMapped::TaskType::DPUInvariant) {
        return mlir::cast<VPUMI40XX::ExecutableTaskOpInterface>(primary.getOperation());
    } else if (primary.getTaskType() == VPURegMapped::TaskType::ActKernelRange) {
        return mlir::cast<VPUMI40XX::ExecutableTaskOpInterface>(secondary.getOperation());
    } else {
        VPUX_THROW("Unknown TaskType for pair {0} {1}", primary.getResult(), secondary.getResult());
        return nullptr;
    }

    return nullptr;
}

//
// ConfigureBarrier Utils
//

// Set next_same_id attribute and previousSameId operand for each ConfigureBarrier operation,
// and here we don't need to verify barrier if it has same previousSameId with other same physical id barrier,
// because the previousSameId operand is continuously increasing
void setBarrierIDs(mlir::MLIRContext* ctx, mlir::func::FuncOp funcOp) {
    auto MAX_PID = VPUIP::getNumAvailableBarriers(funcOp);
    std::vector<VPUMI40XX::ConfigureBarrierOp> lastAssignedBarrier(MAX_PID);

    for (auto op : funcOp.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        auto vid = op.getOperation()->getResult(0).getType().cast<VPURegMapped::IndexType>().getValue();
        auto pid = op.getId();

        auto& lastBarrier = lastAssignedBarrier[pid];
        if (lastBarrier != nullptr) {
            op.getPreviousSameIdMutable().assign(lastBarrier.getOperation()->getResult(0));
            lastBarrier.setNextSameIdAttr(
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed), vid));
        }

        lastBarrier = op;
    }
}

//
// Log Fetch Tasks
//

VPUMI40XX::NNDMAOp getPreviousDMAWithBarriers(VPURegMapped::TaskOpInterface taskOpInterface) {
    while (taskOpInterface != nullptr) {
        if (mlir::isa<VPURegMapped::FetchTaskOp>(taskOpInterface.getOperation())) {
            taskOpInterface = taskOpInterface.getPreviousTask();
            continue;
        }

        if (auto previousDma = mlir::dyn_cast<VPUMI40XX::NNDMAOp>(taskOpInterface.getOperation())) {
            return previousDma;
        }
        taskOpInterface = taskOpInterface.getPreviousTask();
    }
    return nullptr;
}

void logFetchOpsDetails(mlir::func::FuncOp netFunc, Logger log) {
    auto fetchTaskOps = netFunc.getOps<VPURegMapped::FetchTaskOp>();
    SmallVector<FetchTaskDetails> groupedTasks;
    std::map<std::string, std::map<size_t, SmallVector<FetchTaskDetails>>> groupedTasksMap;

    for (auto fetchTaskOp : fetchTaskOps) {
        auto primaryTaskOpStart =
                mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getPrimaryStart().getDefiningOp());

        auto primaryTaskOpEnd = mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getPrimaryEnd().getDefiningOp());
        auto secondaryTaskOpStart =
                mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getSecondaryStart().getDefiningOp());
        auto secondaryTaskOpEnd =
                mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getSecondaryEnd().getDefiningOp());

        size_t executionGroupIndex = 0;
        if (fetchTaskOp.getAssociatedExecutionGroupIndex().has_value()) {
            executionGroupIndex = fetchTaskOp.getAssociatedExecutionGroupIndex().value();
        }
        size_t tileIndex = 0;
        if (fetchTaskOp.getAssociatedTileIndex().has_value()) {
            tileIndex = fetchTaskOp.getAssociatedTileIndex().value();
        }

        std::string taskType = "N/A";
        if (fetchTaskOp.getAssociatedTaskType().has_value()) {
            taskType = (fetchTaskOp.getAssociatedTaskType().value() == VPURegMapped::TaskType::ActKernelInvocation ||
                        fetchTaskOp.getAssociatedTaskType().value() == VPURegMapped::TaskType::ActKernelRange)
                               ? "SHV"
                               : "DPU";
        }

        size_t taskIndex = fetchTaskOp.getIndexType().getValue();
        size_t dmaWithBarriers = SIZE_MAX;
        size_t barrierIdx = 0;

        if (fetchTaskOp.getPreviousTask() != nullptr) {
            auto taskOpInterface =
                    mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getPreviousTask().getDefiningOp());
            if (auto prevDmaWithBarriers = getPreviousDMAWithBarriers(taskOpInterface)) {
                dmaWithBarriers = prevDmaWithBarriers.getIndexType().getValue();

                for (auto updateBarr : prevDmaWithBarriers.getWaitBarriers()) {
                    auto barrOp = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(updateBarr.getDefiningOp());
                    barrierIdx = std::max(barrierIdx, static_cast<size_t>(barrOp.getType().getValue()));
                }
            }
        }

        FetchTaskDetails taskDetails = {tileIndex,
                                        taskIndex,
                                        dmaWithBarriers,
                                        barrierIdx,
                                        primaryTaskOpStart.getIndexType().getValue(),
                                        primaryTaskOpEnd.getIndexType().getValue(),
                                        secondaryTaskOpStart.getIndexType().getValue(),
                                        secondaryTaskOpEnd.getIndexType().getValue(),
                                        taskType,
                                        executionGroupIndex};

        groupedTasksMap[taskType][executionGroupIndex].push_back(taskDetails);
    }

    // Iterate and print in the required format
    for (const auto& taskTypePair : groupedTasksMap) {
        const std::string& taskType = taskTypePair.first;
        log.trace("{0}", taskType);

        for (const auto& execGroupPair : taskTypePair.second) {
            size_t execGroup = execGroupPair.first;
            log.nest().trace("Exec Group {0} - {1}", execGroup, (execGroup % 2 == 0 ? "BUF A" : "BUF B"));

            const auto& tasksInGroup = execGroupPair.second;

            for (const auto& task : tasksInGroup) {
                if (taskType == "DPU") {
                    log.nest(2).trace(
                            "Tile {0} - FetchIndex: {1}, Prev DMA: {2}, BarrierBlockingFetch: {3}, Invariant: "
                            "{4}-{5}, Variant: {6}-{7}",
                            task.tileIndex, task.taskIndex,
                            (task.dmaWithBarriers == SIZE_MAX ? "N/A" : std::to_string(task.dmaWithBarriers)),
                            (task.barrierIdx == SIZE_MAX ? "N/A" : std::to_string(task.barrierIdx)), task.primaryStart,
                            task.primaryEnd, task.secondaryStart, task.secondaryEnd);
                } else {
                    log.nest(2).trace(
                            "Tile {0} - FetchIndex: {1}, Prev DMA: {2}, BarrierBlockingFetch: {3}, Invocation: "
                            "{4}-{5}, Range: {6}-{7}",
                            task.tileIndex, task.taskIndex,
                            (task.dmaWithBarriers == SIZE_MAX ? "N/A" : std::to_string(task.dmaWithBarriers)),
                            (task.barrierIdx == SIZE_MAX ? "N/A" : std::to_string(task.barrierIdx)), task.primaryStart,
                            task.primaryEnd, task.secondaryStart, task.secondaryEnd);
                }
            }
        }
    }
}

}  // namespace VPUMI40XX
}  // namespace vpux
