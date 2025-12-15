//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_OPTIMIZESYNCTASKS
#define GEN_PASS_DEF_OPTIMIZESYNCTASKS
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

//
// MergeSyncTasks
//

class MergeSyncTasks final : public mlir::OpRewritePattern<VPURT::TaskOp> {
public:
    MergeSyncTasks(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPURT::TaskOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPURT::TaskOp syncTaskOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MergeSyncTasks::matchAndRewrite(VPURT::TaskOp syncTaskOp, mlir::PatternRewriter& rewriter) const {
    auto syncDmaOp = syncTaskOp.getInnerTaskOpOfType<VPUIP::SyncDMAOp>();
    if (syncDmaOp == nullptr) {
        return mlir::failure();
    }

    // For now support only case when there is only 1 update barrier
    auto updateBarriers = syncTaskOp.getUpdateBarriers();
    if (updateBarriers.size() != 1) {
        return mlir::failure();
    }

    VPURT::TaskOp userSyncTaskOp = nullptr;
    VPUIP::SyncDMAOp userSyncDmaOp = nullptr;
    for (const auto& bar : updateBarriers) {
        for (auto& use : bar.getUses()) {
            auto userOp = use.getOwner();
            if (userOp == syncTaskOp) {
                continue;
            }

            if (auto userTaskOp = mlir::dyn_cast<VPURT::TaskOp>(userOp)) {
                auto userInnerOp = userTaskOp.getInnerTaskOpOfType<VPUIP::SyncDMAOp>();
                if (userInnerOp != nullptr) {
                    userSyncTaskOp = userTaskOp;
                    userSyncDmaOp = userInnerOp;
                    break;
                }
            }
        }
        if (userSyncTaskOp != nullptr) {
            break;
        }
    }

    if (userSyncTaskOp == nullptr) {
        return mlir::failure();
    }

    auto userWaitBarriers = userSyncTaskOp.getWaitBarriers();

    // For now support only cases when there is only 1 barrier in between
    // both sync tasks
    if (userWaitBarriers.size() != 1) {
        return mlir::failure();
    }

    if (*updateBarriers.begin() != *userWaitBarriers.begin()) {
        return mlir::failure();
    }

    auto barrierInBetween = *updateBarriers.begin();
    // Check if barrier has only 2 uses meaning it is not used by other tasks than
    // syncTaskOp and userSyncTaskOp
    if (std::distance(barrierInBetween.use_begin(), barrierInBetween.use_end()) != 2) {
        return mlir::failure();
    }

    // Update barrier so that userSyncTaskOp consumes barrier which is produced by parents
    // of syncTaskOp
    userSyncTaskOp.getWaitBarriersMutable().clear();
    for (auto bar : syncTaskOp.getWaitBarriers()) {
        userSyncTaskOp.getWaitBarriersMutable().append(bar);
    }

    syncTaskOp.getUpdateBarriersMutable().clear();
    syncTaskOp.getWaitBarriersMutable().clear();

    auto barrierOp = barrierInBetween.getDefiningOp();
    VPUX_THROW_UNLESS(barrierInBetween.use_empty(), "Barrier still in use - '{0}'", barrierOp);

    rewriter.eraseOp(barrierOp);
    rewriter.eraseOp(syncTaskOp);

    return mlir::success();
}

// Helper function that identifes task which is a single user of barrier that is also
// used by sync task. If there are more users then nullptr is returned
VPURT::TaskOp getSingleSyncTaskBarUser(VPURT::TaskOp syncTaskOp, mlir::Value bar) {
    VPURT::TaskOp singleUserTask = nullptr;
    for (const auto& use : bar.getUses()) {
        auto userOp = use.getOwner();
        if (userOp == syncTaskOp) {
            continue;
        }

        if (auto userTaskOp = mlir::dyn_cast<VPURT::TaskOp>(userOp)) {
            if (singleUserTask != nullptr) {
                // If user was already encountered it means that there are more users
                // and optimization cannot be performed
                return nullptr;
            } else {
                singleUserTask = userTaskOp;
            }
        }
    }
    return singleUserTask;
}

// Verify if sync task can be replaced with its neighbor op (parent or child) by checking
// if barrier variant count limit will not get exceeded when neighbor op will be connected to sync task barrier
// that already has other users
bool isValidVariantCountForSyncTaskReplacement(VPURT::TaskOp replaceTaskOp,
                                               mlir::Operation::operand_range synTaskBarsRange, size_t maxVariantSum) {
    size_t slotCount = 0;
    for (const auto& bar : synTaskBarsRange) {
        for (const auto* userOp : bar.getUsers()) {
            if (auto userTaskOp = mlir::dyn_cast<VPURT::TaskOp>(userOp)) {
                slotCount += BarrierInfo::getNumOfSlotsUsed(userTaskOp);
            }
        }
    }
    if (slotCount > 0) {
        slotCount--;  // Decrement as syncTask will no longer be producer of this bar
    }
    slotCount += BarrierInfo::getNumOfSlotsUsed(replaceTaskOp);

    return slotCount <= maxVariantSum;
}

// Identify sync tasks that can be removed on a range of tasks: "-> sync -> ... -> sync ->"
// Method will return map where:
//   map key: sync task to remove
//   map value: pair of parent or child task that when sync task is removed can act as a sync task
DenseMap<VPURT::TaskOp, std::pair<VPURT::TaskOp, VPURT::TaskOp>> getSyncTaskOpsToRemoveWithParentChildPairMap(
        mlir::func::FuncOp func) {
    auto taskOpsVec = to_small_vector(func.getOps<VPURT::TaskOp>());
    auto taskOpsVecSize = taskOpsVec.size();

    // Build a map of sync tasks to remove
    // map key: sync task to remove
    // map value: pair of parent or child task that when sync task is removed can act as a sync task
    DenseMap<VPURT::TaskOp, std::pair<VPURT::TaskOp, VPURT::TaskOp>> taskOpsToRemoveWithParentChildPairMap;

    if (taskOpsVecSize < 2) {
        return taskOpsToRemoveWithParentChildPairMap;
    }

    const auto maxVariantSum = VPUIP::getBarrierMaxVariantSum(func);

    auto checkAndAddSyncTaskToReplaceWithParent = [&](VPURT::TaskOp syncTaskOp, VPURT::TaskOp parentTaskOp) {
        if (isValidVariantCountForSyncTaskReplacement(parentTaskOp, syncTaskOp.getWaitBarriers(), maxVariantSum)) {
            taskOpsToRemoveWithParentChildPairMap[syncTaskOp] = std::make_pair(parentTaskOp, nullptr);
        }
    };

    auto checkAndAddSyncTaskToReplaceWithChild = [&](VPURT::TaskOp syncTaskOp, VPURT::TaskOp childTaskOp) {
        if (isValidVariantCountForSyncTaskReplacement(childTaskOp, syncTaskOp.getUpdateBarriers(), maxVariantSum)) {
            taskOpsToRemoveWithParentChildPairMap[syncTaskOp] = std::make_pair(nullptr, childTaskOp);
        }
    };

    // Check if sync task on the beginning of region that is being processed can be removed
    // syncTask -> ....
    // If yes update information on removal and what child task can take its sync responsibility
    auto processSyncTaskForRegionStart = [&](VPURT::TaskOp syncTaskOp, VPURT::TaskQueueType syncTaskQueueType,
                                             auto& taskQueuesFirstAndLastOp) {
        // Check if syncTask can be removed
        // This is possible if only single task depends on it
        // explicitly through barrier or implicitly through FIFO
        auto updateBarriers = syncTaskOp.getUpdateBarriers();
        if (updateBarriers.size() == 1) {
            // If syncTask updates 1 barrier check if there is only
            // one task using it and if this task is on the same queue
            auto singleChildTask = getSingleSyncTaskBarUser(syncTaskOp, updateBarriers.front());
            if (singleChildTask) {
                // If there is a single child task check if this is
                // same as first task on the same queue
                if (taskQueuesFirstAndLastOp.find(syncTaskQueueType) == taskQueuesFirstAndLastOp.end() ||
                    taskQueuesFirstAndLastOp[syncTaskQueueType].first == singleChildTask) {
                    // No other task on this queue or identified first task on this queue is the same as
                    // user syncTaskOp can be removed
                    checkAndAddSyncTaskToReplaceWithChild(syncTaskOp, singleChildTask);
                }
            }
        } else if (updateBarriers.empty()) {
            // If task has no barrier get next task on the FIFO
            if (taskQueuesFirstAndLastOp.find(syncTaskQueueType) != taskQueuesFirstAndLastOp.end()) {
                checkAndAddSyncTaskToReplaceWithChild(syncTaskOp, taskQueuesFirstAndLastOp[syncTaskQueueType].first);
            }
        }
    };

    // Check if sync task on the end of region that is being processed can be removed
    // .... -> syncTask
    // If yes update information on removal and what parent task can take its sync responsibility
    auto processSyncTaskForRegionEnd = [&](VPURT::TaskOp syncTaskOp, VPURT::TaskQueueType syncTaskQueueType,
                                           auto& taskQueuesFirstAndLastOp) {
        // Check if syncTask can be removed
        // This is possible if only single task produces it
        // explicitly through barrier or implicitly through FIFO
        auto waitBarriers = syncTaskOp.getWaitBarriers();
        if (waitBarriers.size() == 1) {
            // If syncTask waits on 1 barrier check if there is only
            // one task producing it and if this task is on the same queue
            auto singleParentTask = getSingleSyncTaskBarUser(syncTaskOp, waitBarriers.front());
            if (singleParentTask) {
                // If there is a single parent task check if this is
                // same as last task on the same queue
                if (taskQueuesFirstAndLastOp.find(syncTaskQueueType) == taskQueuesFirstAndLastOp.end() ||
                    taskQueuesFirstAndLastOp[syncTaskQueueType].second == singleParentTask) {
                    // No other task on this queue or identified first task on this queue is the same as
                    // user. syncTaskOp can be removed
                    checkAndAddSyncTaskToReplaceWithParent(syncTaskOp, singleParentTask);
                }
            }
        } else if (waitBarriers.empty()) {
            // If task has no barrier get prev task on the FIFO
            if (taskQueuesFirstAndLastOp.find(syncTaskQueueType) != taskQueuesFirstAndLastOp.end()) {
                checkAndAddSyncTaskToReplaceWithParent(syncTaskOp, taskQueuesFirstAndLastOp[syncTaskQueueType].second);
            }
        }
    };

    // Store information about first and last ops in a HW queue for a given analysis scope of IR
    // Do not include sync tasks
    DenseMap<VPURT::TaskQueueType, std::pair<VPURT::TaskOp, VPURT::TaskOp>> taskQueuesFirstAndLastOp;

    // Process first task and initialize state before processing rest of tasks
    auto firstTask = taskOpsVec[0];
    auto firstTaskQueueType = VPURT::getTaskQueueType(firstTask, false);
    auto firstSyncDmaOp = firstTask.getInnerTaskOpOfType<VPUIP::SyncDMAOp>();

    VPURT::TaskOp prevSyncTaskOp = nullptr;
    auto prevSyncTaskOpQueueType = VPURT::TaskQueueType{VPU::ExecutorKind::UNKNOWN, 0};

    if (firstSyncDmaOp != nullptr) {
        prevSyncTaskOp = firstTask;
        prevSyncTaskOpQueueType = firstTaskQueueType;
    } else {
        taskQueuesFirstAndLastOp[firstTaskQueueType] = std::make_pair(firstTask, firstTask);
    }

    // Traverse all tasks and identify sync task ops and analyze regions with structure
    //
    //  prevSyncTask (or first task)  -> .... -> syncTask (or last task)
    //
    // For such range check if
    //  - prevSyncTask can be removed and its sync reponsibility can be taken by its child op
    //  - syncTask can be removed and its sync responsibility can be taken by its parent op
    for (size_t index = 1; index < taskOpsVecSize; index++) {
        auto taskOp = taskOpsVec[index];
        auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);

        // If this is last task it doesn't have to be a sync task, just process the region starting
        // from prev sync task and exit loop
        if (index == taskOpsVecSize - 1) {
            if (prevSyncTaskOp != nullptr && !taskQueuesFirstAndLastOp.empty()) {
                processSyncTaskForRegionStart(prevSyncTaskOp, prevSyncTaskOpQueueType, taskQueuesFirstAndLastOp);
            }

            break;
        }

        auto syncDmaOp = taskOp.getInnerTaskOpOfType<VPUIP::SyncDMAOp>();
        if (syncDmaOp != nullptr) {
            if (!taskQueuesFirstAndLastOp.empty()) {
                // Sync task identified. Process beginning and end of region
                //  prevSyncTask -> ....
                //  .... -> syncTask
                if (prevSyncTaskOp != nullptr) {
                    processSyncTaskForRegionStart(prevSyncTaskOp, prevSyncTaskOpQueueType, taskQueuesFirstAndLastOp);
                }

                processSyncTaskForRegionEnd(taskOp, taskQueueType, taskQueuesFirstAndLastOp);
            }

            // Region has been analyzed. Prepare variables for handling new region
            prevSyncTaskOp = taskOp;
            prevSyncTaskOpQueueType = taskQueueType;
            taskQueuesFirstAndLastOp.clear();
            continue;
        }

        if (taskQueuesFirstAndLastOp.find(taskQueueType) == taskQueuesFirstAndLastOp.end()) {
            // First occurrence of task on this queue
            taskQueuesFirstAndLastOp[taskQueueType] = std::make_pair(taskOp, taskOp);
        } else {
            // In case new task spotted, update last task info
            taskQueuesFirstAndLastOp[taskQueueType].second = taskOp;
        }
    }

    return taskOpsToRemoveWithParentChildPairMap;
}

// Remove sync tasks which are redundant based on decision from getSyncTaskOpsToRemoveWithParentChildPairMap() method
// Those sync tasks have neighbor op (parent or child) which can act themselves as sync task
void removeSyncTaskIfSyncCanHappenOnNeighborOp(
        DenseMap<VPURT::TaskOp, std::pair<VPURT::TaskOp, VPURT::TaskOp>>& taskOpsToRemoveWithParentChildPairMap,
        Logger log) {
    // Traverse stored information about sync tasks that can be removed
    // and their sync functionality can be taken by either their parent or child task
    for (auto& t : taskOpsToRemoveWithParentChildPairMap) {
        auto taskOp = t.first;
        auto parentTaskOp = t.second.first;
        auto childTaskOp = t.second.second;

        std::set<VPURT::DeclareVirtualBarrierOp> barsToRemove;

        auto updateBarsToRemove = [&](mlir::Operation::operand_range barsRange) {
            for (const auto& bar : barsRange) {
                auto barOp = bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
                barsToRemove.insert(barOp);
            }
        };
        if (parentTaskOp != nullptr) {
            log.trace("Identified parent task that can act as a sync task - {0}", parentTaskOp->getLoc());
            parentTaskOp.getUpdateBarriersMutable().assign(taskOp.getUpdateBarriers());
            updateBarsToRemove(taskOp.getWaitBarriers());
        } else if (childTaskOp != nullptr) {
            log.trace("Identified child task that can act as a sync task - {0}", childTaskOp->getLoc());
            childTaskOp.getWaitBarriersMutable().assign(taskOp.getWaitBarriers());
            updateBarsToRemove(taskOp.getUpdateBarriers());
        }

        log.trace("Remove sync task - {0}", taskOp->getLoc());
        taskOp.erase();
        for (auto barOp : barsToRemove) {
            if (barOp.getBarrier().use_empty()) {
                barOp.erase();
            }
        }
    }
}

// Remove very first and last sync tasks in the schedule which. Remove related barrier if possible
void removeFirstAndLastSyncTasks(mlir::func::FuncOp func, Logger log) {
    auto taskOpsVec = to_small_vector(func.getOps<VPURT::TaskOp>());
    auto taskOpsVecSize = taskOpsVec.size();

    // Helper function that checks if for a given sync task (first or last in IR) related barrier
    // can be removed as part of removal of sync task. Barrier cannot be removed if
    // in case of first sync task same bar is also produced by other task or in case of last sync task
    // same bar is also consumed by other task
    auto getBarriersRemovalDecisionForSync = [](VPURT::TaskOp syncTaskOp, auto getBarriers) {
        mlir::DenseMap<mlir::Value, bool> barrierCanBeRemovedMap;

        for (const auto& bar : getBarriers(syncTaskOp)) {
            barrierCanBeRemovedMap[bar] = true;
            for (const auto& use : bar.getUses()) {
                auto userOp = use.getOwner();

                if (auto userTaskOp = mlir::dyn_cast<VPURT::TaskOp>(userOp)) {
                    if (userTaskOp == syncTaskOp) {
                        continue;
                    } else {
                        // For any other task check if it also produces this barrier
                        // If yes barrier cannot be removed
                        for (const auto& userBar : getBarriers(userTaskOp)) {
                            if (userBar == bar) {
                                barrierCanBeRemovedMap[bar] = false;
                                break;
                            }
                        }
                    }
                }
            }
        }

        return barrierCanBeRemovedMap;
    };

    auto getBarriersRemovalDecisionForFirstSync = [&](VPURT::TaskOp syncTaskOp) {
        auto getUpdateBarriers = [](VPURT::TaskOp taskOp) {
            return taskOp.getUpdateBarriers();
        };
        return getBarriersRemovalDecisionForSync(syncTaskOp, getUpdateBarriers);
    };

    auto getBarriersRemovalDecisionForLastSync = [&](VPURT::TaskOp syncTaskOp) {
        auto getWaitBarriers = [](VPURT::TaskOp taskOp) {
            return taskOp.getWaitBarriers();
        };
        return getBarriersRemovalDecisionForSync(syncTaskOp, getWaitBarriers);
    };

    // Helper function that removes barrier dependency for sync task and also removes barrier itself
    // if possible
    auto removeBarDepOnFirstOrLastSyncTaskIfPossible = [](VPURT::TaskOp syncTaskOp,
                                                          mlir::DenseMap<mlir::Value, bool>& barrierCanBeRemovedMap,
                                                          auto getSyncTaskBarriers, auto getUserTaskBarriers) {
        mlir::OperandRange syncTaskBarsRange = getSyncTaskBarriers(syncTaskOp);

        for (const auto& bar : llvm::make_early_inc_range(syncTaskBarsRange)) {
            if (!barrierCanBeRemovedMap[bar]) {
                continue;
            }
            for (auto& use : llvm::make_early_inc_range(bar.getUses())) {
                auto userOp = use.getOwner();

                if (auto userTaskOp = mlir::dyn_cast<VPURT::TaskOp>(userOp)) {
                    if (userTaskOp == syncTaskOp) {
                        // If this is sync task just erase dependency on barrier
                        getSyncTaskBarriers(userTaskOp).clear();
                    } else {
                        SmallVector<mlir::Value> newBarriers;
                        mlir::OperandRange userTaskBarsRange = getUserTaskBarriers(userTaskOp);

                        for (const auto& userBar : userTaskBarsRange) {
                            if (userBar == bar) {
                                break;
                            }
                            newBarriers.push_back(userBar);
                        }
                        getUserTaskBarriers(userTaskOp).clear();
                        getUserTaskBarriers(userTaskOp).assign(newBarriers);
                    }
                }
            }
            auto barOp = bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
            VPUX_THROW_UNLESS(barOp.getBarrier().use_empty(), "Bar op still has uses - '{0}'", barOp->getLoc());
            barOp.erase();
        }
    };

    auto removeBarDepOnFirstSyncTaskIfPossible = [&](VPURT::TaskOp syncTaskOp,
                                                     mlir::DenseMap<mlir::Value, bool>& barrierCanBeRemovedMap) {
        auto getSyncTaskBars = [](VPURT::TaskOp taskOp) {
            return taskOp.getUpdateBarriersMutable();
        };
        auto getUserTaskBars = [](VPURT::TaskOp taskOp) {
            return taskOp.getWaitBarriersMutable();
        };

        removeBarDepOnFirstOrLastSyncTaskIfPossible(syncTaskOp, barrierCanBeRemovedMap, getSyncTaskBars,
                                                    getUserTaskBars);
    };

    auto removeBarDepOnLastSyncTaskIfPossible = [&](VPURT::TaskOp syncTaskOp,
                                                    mlir::DenseMap<mlir::Value, bool>& barrierCanBeRemovedMap) {
        auto getSyncTaskBars = [](VPURT::TaskOp taskOp) {
            return taskOp.getWaitBarriersMutable();
        };
        auto getUserTaskBars = [](VPURT::TaskOp taskOp) {
            return taskOp.getUpdateBarriersMutable();
        };

        removeBarDepOnFirstOrLastSyncTaskIfPossible(syncTaskOp, barrierCanBeRemovedMap, getSyncTaskBars,
                                                    getUserTaskBars);
    };

    // If sync task is first it should be removed unconditionally as it is redundant
    if (taskOpsVec[0] != nullptr && taskOpsVec[0].getInnerTaskOpOfType<VPUIP::SyncDMAOp>() != nullptr) {
        auto firstSyncTaskOp = taskOpsVec[0];
        // Remove sync task barrier dependency and check if it can be removed
        // for other tasks
        auto barrierCanBeRemovedMap = getBarriersRemovalDecisionForFirstSync(firstSyncTaskOp);
        removeBarDepOnFirstSyncTaskIfPossible(firstSyncTaskOp, barrierCanBeRemovedMap);

        log.trace("Remove sync task - {0} (start sync task)", firstSyncTaskOp->getLoc());
        firstSyncTaskOp.erase();
    }

    // If sync task is last it should be removed unconditionally as it is redundant
    if (taskOpsVecSize > 1 && taskOpsVec[taskOpsVecSize - 1] != nullptr &&
        taskOpsVec[taskOpsVecSize - 1].getInnerTaskOpOfType<VPUIP::SyncDMAOp>() != nullptr) {
        auto lastSyncTaskOp = taskOpsVec[taskOpsVecSize - 1];

        // Remove sync task barrier dependency and check if it can be removed
        // for other tasks
        auto barrierCanBeRemovedMap = getBarriersRemovalDecisionForLastSync(lastSyncTaskOp);
        removeBarDepOnLastSyncTaskIfPossible(lastSyncTaskOp, barrierCanBeRemovedMap);

        log.trace("Remove sync task - {0} (end sync task)", lastSyncTaskOp->getLoc());
        lastSyncTaskOp.erase();
    }
}

//
// removeRedundantSyncTasks
//
void removeRedundantSyncTasks(mlir::func::FuncOp func, Logger log) {
    auto taskOpsVec = to_small_vector(func.getOps<VPURT::TaskOp>());

    // STEP 1
    // Identify sync tasks that can be removed on a range of tasks: "-> sync -> ... -> sync ->"
    auto taskOpsToRemoveWithParentChildPairMap = getSyncTaskOpsToRemoveWithParentChildPairMap(func);

    // STEP 2
    // Remove sync tasks which are redundant
    removeSyncTaskIfSyncCanHappenOnNeighborOp(taskOpsToRemoveWithParentChildPairMap, log);

    // STEP 3
    // Remove very first and last sync tasks in the schedule. Remove related barrier if possible
    removeFirstAndLastSyncTasks(func, log);
}

// Remove unnecessary sync task that were injected previously by InsertSyncTasks pass.
// Example:
//          |-> op -> .. -> op -|                       |-> op -> .. -> op -|
// SyncTask --> op -> .. -> op --> SyncTask -> SyncTask --> op -> .. -> op --> SyncTask
//          |-> op -> .. -> op -|                       |-> op -> .. -> op -|
//
//                 =>
//
//   op -> .. -> op -|           |-> op -> .. -> op
//   op -> .. -> op --> SyncTask --> op -> .. -> op
//   op -> .. -> op -|           |-> op -> .. -> op
class OptimizeSyncTasksPass final : public VPURT::impl::OptimizeSyncTasksBase<OptimizeSyncTasksPass> {
public:
    explicit OptimizeSyncTasksPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeSyncTasksPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MergeSyncTasks>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    removeRedundantSyncTasks(func, _log);
}
}  // namespace

//
// createOptimizeSyncTasksPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createOptimizeSyncTasksPass(Logger log) {
    return std::make_unique<OptimizeSyncTasksPass>(log);
}
