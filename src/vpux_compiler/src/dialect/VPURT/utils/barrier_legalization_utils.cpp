//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"

using namespace vpux;

size_t VPURT::getMinEntry(const BarrierInfo::TaskSet& entries) {
    if (entries.empty()) {
        return std::numeric_limits<size_t>::min();
    }
    return *std::min_element(entries.begin(), entries.end());
}

size_t VPURT::getMaxEntry(const BarrierInfo::TaskSet& entries) {
    if (entries.empty()) {
        return std::numeric_limits<size_t>::max();
    }
    return *std::max_element(entries.begin(), entries.end());
}

// generate FIFOs of Task Ops using index from BarrierInfo
/*
    FIFO-0 | 0, 2, 4
    FIFO-1 | 1, 3, 5, 6
*/
VPURT::TaskOpQueues VPURT::getTaskOpQueues(mlir::func::FuncOp funcOp, BarrierInfo& barrierInfo,
                                           std::optional<VPU::ExecutorKind> targetExecutorKind) {
    VPURT::TaskOpQueues taskOpQueues;
    for (auto taskOp : funcOp.getOps<VPURT::TaskOp>()) {
        const auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);
        if (targetExecutorKind.has_value() && targetExecutorKind.value() != taskQueueType.type) {
            continue;
        }
        const auto taskInd = barrierInfo.getIndex(taskOp);
        taskOpQueues[taskQueueType].push_back(taskInd);
    }

    return taskOpQueues;
}

// Initialize the iterator for the task op queues
VPURT::TaskOpQueueIterator VPURT::initializeTaskOpQueueIterators(TaskOpQueues& taskOpQueues) {
    std::map<VPURT::TaskQueueType, SmallVector<uint32_t>::iterator> frontTasks;
    for (auto& taskOpQueue : taskOpQueues) {
        frontTasks[taskOpQueue.first] = taskOpQueue.second.begin();
    }
    return frontTasks;
}

// Check all task queues reach end or not
bool VPURT::allQueuesReachedEnd(const TaskOpQueueIterator& frontTasks, const TaskOpQueues& taskOpQueues) {
    return llvm::all_of(frontTasks, [&](const auto& entry) {
        const auto& type = entry.first;
        const auto& queueIter = entry.second;
        VPUX_THROW_UNLESS(taskOpQueues.find(type) != taskOpQueues.end(), "Task op queue doesn't contain queue type {0}",
                          VPU::stringifyExecutorKind(type.type));
        return queueIter == taskOpQueues.at(type).end();
    });
}

// Check that all queues wait for a barrier
bool VPURT::allQueuesWaiting(const TaskOpQueueIterator& frontTasks, const TaskOpQueues& taskOpQueues,
                             BarrierInfo& barrierInfo) {
    return !llvm::any_of(frontTasks, [&](const auto& entry) {
        const auto& type = entry.first;
        const auto& queueIter = entry.second;
        VPUX_THROW_UNLESS(taskOpQueues.find(type) != taskOpQueues.end(), "Task op queue doesn't contain queue type {0}",
                          VPU::stringifyExecutorKind(type.type));
        return queueIter != taskOpQueues.at(type).end() && barrierInfo.getWaitBarriers(*queueIter).empty();
    });
}

/**
 * Finds the ready operations (that don't have wait barriers) from the task operation queues/FIFO.
 *
 * This function iterates over the front tasks and checks if all queues are waiting for a barrier.
 * If not, it checks if the current task operation is ready by checking if there are no wait barriers for it.
 * If the task operation is ready, it adds its index to the `readyOps` vector and increments the iterator.
 *
 * @param frontTasks The iterator pointing to the front tasks that in the task operation queues.
 * @param taskOpQueues The map of task operation queues.
 * @param barrierInfo The barrier information.
 * @return A vector containing the indices of the ready operations.
 */
SmallVector<size_t> VPURT::findReadyOpsFromTaskOpQueues(TaskOpQueueIterator& frontTasks,
                                                        const TaskOpQueues& taskOpQueues, BarrierInfo& barrierInfo) {
    SmallVector<size_t> readyOps;
    while (!allQueuesWaiting(frontTasks, taskOpQueues, barrierInfo)) {
        for (auto& entry : frontTasks) {
            VPUX_THROW_UNLESS(taskOpQueues.find(entry.first) != taskOpQueues.end(),
                              "Task op queue doesn't contain queue type {0}",
                              VPU::stringifyExecutorKind(entry.first.type));
            if (entry.second != taskOpQueues.at(entry.first).end() &&
                barrierInfo.getWaitBarriers(*entry.second).empty()) {
                readyOps.push_back(*entry.second);
                ++entry.second;
            }
        }
    }
    return readyOps;
}

void VPURT::postProcessBarrierOps(mlir::func::FuncOp func) {
    // move barriers to top and erase unused
    auto barrierOps = to_small_vector(func.getOps<VPURT::DeclareVirtualBarrierOp>());
    auto& block = func.getBody().front();

    VPURT::DeclareVirtualBarrierOp prevBarrier = nullptr;
    for (auto& barrierOp : barrierOps) {
        // remove barriers with no use
        if (barrierOp.getBarrier().use_empty()) {
            barrierOp->erase();
            continue;
        }

        // move barriers to top of block
        if (prevBarrier != nullptr) {
            barrierOp->moveAfter(prevBarrier);
        } else {
            barrierOp->moveBefore(&block, block.begin());
        }

        prevBarrier = barrierOp;
    }
}

// It should be called at the end of each pass which may change barriers after SplitExceedingBarrierSlotCountPass
bool VPURT::verifyBarrierSlots(mlir::func::FuncOp func, Logger log) {
    auto barrierSim = VPURT::BarrierSimulator{func};
    if (mlir::failed(barrierSim.checkProducerAndConsumerCount(log))) {
        log.trace("verifyBarrierSlots failed");
        return false;
    }
    return true;
}

bool VPURT::verifyOneWaitBarrierPerTask(mlir::func::FuncOp funcOp, Logger log) {
    bool hasOneWaitBarrierPerTask = true;
    funcOp->walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getWaitBarriers().size() > 1) {
            log.warning("Task '{0}' has more than one wait barrier", taskOp.getLoc());
            hasOneWaitBarrierPerTask = false;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });

    return hasOneWaitBarrierPerTask;
}

// simulate execution of tasks an barriers to generate an order for tasks an barriers which will represent execution
// order tasks and barriers in IR to match that order - required for virtual to physical barrier mapping
// orderByConsumption = false, barriers are ordered based on first producer, orderByConsumption = true: barriers are
// ordered based on consumer
void VPURT::orderExecutionTasksAndBarriers(mlir::func::FuncOp funcOp, BarrierInfo& barrierInfo, Logger log,
                                           bool orderByConsumption) {
    log.trace("Ordering tasks and barriers");
    log = log.nest();
    barrierInfo.updateIR();

    auto taskOpQueues = VPURT::getTaskOpQueues(funcOp, barrierInfo);
    SmallVector<size_t> newTaskOpOrder;
    SmallVector<size_t> newBarrierOrder;
    std::set<size_t> newBarrierOrderSet;

    // initialize front task from each FIFO
    auto frontTasks = VPURT::initializeTaskOpQueueIterators(taskOpQueues);

    // reduce producer count for barrier of ready op
    // reset barrier if producer count reaches 0
    const auto removeBarrierProducer = [&](size_t readyOp) {
        const auto updateBarriers = barrierInfo.getUpdateBarriers(readyOp);
        for (const auto& updateBarrier : updateBarriers) {
            auto barrierOp = barrierInfo.getBarrierOpAtIndex(updateBarrier);
            barrierInfo.removeProducer(barrierOp, readyOp);
            log.trace("Decrement producer count for barrier '{0}' to '{1}'", updateBarrier,
                      barrierInfo.getBarrierProducers(updateBarrier).size());

            // Order the barrier by producer to WA hang issue in runtime
            // TODO: E#109198 find the root cause of runtime hang
            // TODO: E#104375 add the catch such case in compiler simulation
            if (!orderByConsumption && !newBarrierOrderSet.count(updateBarrier)) {
                newBarrierOrder.push_back(updateBarrier);
                newBarrierOrderSet.insert(updateBarrier);
            }

            if (barrierInfo.getBarrierProducers(barrierOp).empty()) {
                if (orderByConsumption) {
                    // barriers will be ordered by order of consumption, each barrier is consumed only once
                    newBarrierOrder.push_back(updateBarrier);
                }
                barrierInfo.resetBarrier(barrierOp);
                log.trace("Reset barrier '{0}'", updateBarrier);
            }
        }
    };

    // simulate per FIFO execution - all FIFOs must reach end
    SmallVector<size_t> lastReadyOps;
    while (!VPURT::allQueuesReachedEnd(frontTasks, taskOpQueues)) {
        auto readyOps = VPURT::findReadyOpsFromTaskOpQueues(frontTasks, taskOpQueues, barrierInfo);

        // If simulation fails (no ready ops), dump previous
        if (readyOps.empty()) {
            if (!lastReadyOps.empty()) {
                log.error("Last readyOps before failure: {0}", lastReadyOps);
            }
            VPUX_THROW("Failed to simulate execution. Possible deadlock or barrier misconfiguration");
        }

        for (auto& readyOp : readyOps) {
            log.trace("Task '{0}' is ready", readyOp);
            // tasks will be ordered by order of becoming ready
            newTaskOpOrder.push_back(readyOp);
            removeBarrierProducer(readyOp);
        }
        lastReadyOps = std::move(readyOps);
    }

    // ensure number of tasks remains the same
    VPUX_THROW_UNLESS(newTaskOpOrder.size() == barrierInfo.getNumOfTasks(),
                      "Failed to order all tasks, there are {0} tasks, got {1}", barrierInfo.getNumOfTasks(),
                      newTaskOpOrder.size());

    size_t barriersWithNoUse = 0;
    funcOp->walk([&](VPURT::BarrierOpInterface barrierOp) {
        if (!barrierOp.getBarrier().use_empty()) {
            return;
        }
        ++barriersWithNoUse;
    });

    // ensure number of barriers remains the same
    VPUX_THROW_UNLESS(newBarrierOrder.size() == barrierInfo.getNumOfBarrierOps() - barriersWithNoUse,
                      "Failed to order all barriers, there are {0} used barriers, got {1}",
                      barrierInfo.getNumOfBarrierOps() - barriersWithNoUse, newBarrierOrder.size());

    // reorder tasks in the IR based on new order
    mlir::Operation* prevTaskOp = nullptr;
    for (auto& opIdx : newTaskOpOrder) {
        mlir::Operation* taskOp = barrierInfo.getTaskOpAtIndex(opIdx);
        if (prevTaskOp != nullptr) {
            taskOp->moveAfter(prevTaskOp);
        } else {
            // Start inserting tasks by placing them after the last task
            // This way reordering of tasks will not depend on placement of
            // declare ops in the IR which can happen to be in between of some taskOps.
            // Without that reordering could cause operand dominance issues
            // e.g. taskOp dependent on some declare op is placed before it
            auto taskOps = funcOp.getOps<VPURT::TaskOp>();
            auto lastTaskOpIt = taskOps.begin();
            std::advance(lastTaskOpIt, std::distance(taskOps.begin(), taskOps.end()) - 1);
            taskOp->moveAfter(*lastTaskOpIt);
        }
        prevTaskOp = taskOp;
    }

    // Align barrier order with WLM page index if present
    if (!newBarrierOrder.empty()) {
        auto firstBarOp = barrierInfo.getBarrierOpAtIndex(0);
        if (firstBarOp.getWlmPage().has_value()) {
            SmallVector<size_t> barPageInd(barrierInfo.getNumOfBarrierOps());
            for (size_t i = 0; i < barrierInfo.getNumOfBarrierOps(); i++) {
                auto barrierOp = barrierInfo.getBarrierOpAtIndex(i);
                VPUX_THROW_WHEN(!barrierOp.getWlmPage().has_value(), "No WLM page assignment for barrier {0} - '{1}'",
                                i, barrierOp);
                barPageInd[i] = barrierOp.getWlmPage().value();
            }

            std::stable_sort(newBarrierOrder.begin(), newBarrierOrder.end(), [&](size_t barOpIdx1, size_t barOpIdx2) {
                return barPageInd[barOpIdx1] < barPageInd[barOpIdx2];
            });
        }
    }

    // reorder barriers in the IR based on new order
    auto& block = funcOp.getBody().front();
    VPURT::BarrierOpInterface prevBarrier = nullptr;
    VPURT::BarrierOpInterface finalBarOp = nullptr;
    for (auto& barrierOpIdx : newBarrierOrder) {
        auto barrierOp = barrierInfo.getBarrierOpAtIndex(barrierOpIdx);
        // move barriers to top of block
        if (prevBarrier != nullptr) {
            barrierOp->moveAfter(prevBarrier);
        } else {
            barrierOp->moveBefore(&block, block.begin());
        }
        prevBarrier = barrierOp;

        if (barrierOp.getIsFinalBarrier()) {
            VPUX_THROW_UNLESS(finalBarOp == nullptr, "More then one final barrier: {0} and {1}", finalBarOp, barrierOp);
            finalBarOp = barrierOp;
        }
    }

    // Other passes expect final barrier to be at the end of IR
    if (finalBarOp != nullptr && prevBarrier != finalBarOp) {
        log.trace("Moving final barrier {0} after barrier {1}", finalBarOp, prevBarrier);
        finalBarOp->moveAfter(prevBarrier);
    }
    log = log.unnest();

    // regenerate barrier info based on new order
    barrierInfo = vpux::BarrierInfo{funcOp};
}

size_t VPURT::getFinalBarriersCount(mlir::func::FuncOp funcOp) {
    size_t finalBarriersCount = 0;
    auto barrierOps = to_small_vector(funcOp.getOps<VPURT::DeclareVirtualBarrierOp>());

    for (auto& barrierOp : barrierOps) {
        if (barrierOp.getIsFinalBarrier()) {
            finalBarriersCount++;
        }
    }

    return finalBarriersCount;
}

bool VPURT::addFinalBarrierIfNotExists(mlir::func::FuncOp funcOp, Logger log) {
    auto finalBarriersCount = VPURT::getFinalBarriersCount(funcOp);
    VPUX_THROW_WHEN(finalBarriersCount > 1, "Found multiple final barriers.");
    if (finalBarriersCount > 0) {
        return false;
    }

    auto findInsertPoint = [&]() {
        auto barrierOpsIt = funcOp.getOps<VPURT::DeclareVirtualBarrierOp>();
        if (barrierOpsIt.empty()) {
            return &funcOp.getBody().front().front();
        }
        auto lastBarrierOpIter = barrierOpsIt.begin();
        std::advance(lastBarrierOpIter, std::distance(barrierOpsIt.begin(), barrierOpsIt.end()) - 1);
        auto lastBarrierOp = *lastBarrierOpIter;
        return lastBarrierOp.getOperation();
    };

    auto hasAnyConsumers = [&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        auto barrier = barrierOp.getBarrier();
        for (const auto& user : barrier.getUsers()) {
            auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(user);
            VPUX_THROW_WHEN(taskOp == nullptr, "VPURT.TaskOp is expected as user for barrier at '{0}'",
                            barrier.getLoc());
            auto waitBarriers = taskOp.getWaitBarriers();
            if (llvm::find(waitBarriers, barrier) != waitBarriers.end()) {
                return true;
            }
        }
        return false;
    };

    auto isNewFinalBarrierOpRequired = [&]() {
        // If any of the final tasks don't update any barrier then new barrier is required.
        // This will happen when final barrier has not yet been created.
        for (auto taskQueueFirstAndLastOp : vpux::VPURT::getTaskQueuesFirstAndLastOp(funcOp)) {
            auto& lastOpInQueue = taskQueueFirstAndLastOp.second.second;
            if (lastOpInQueue.getUpdateBarriers().empty()) {
                return true;
            }
        }

        // If final barrier was created but optimizations in legalization passes removed its attribute then
        // the barrier can be reused as final barrier.
        return false;
    };

    if (isNewFinalBarrierOpRequired()) {
        // create new barrier and update it by FIFO final tasks
        auto ctx = funcOp.getContext();
        auto insertPoint = findInsertPoint();
        mlir::OpBuilder builder(funcOp);
        builder.setInsertionPointAfter(insertPoint);
        auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "finishing_barrier"));
        auto barrierOp = builder.create<VPURT::DeclareVirtualBarrierOp>(loc, mlir::UnitAttr::get(ctx));
        auto finalBarrier = barrierOp.getBarrier();

        for (auto taskQueueFirstAndLastOp : vpux::VPURT::getTaskQueuesFirstAndLastOp(funcOp)) {
            auto& lastOpInQueue = taskQueueFirstAndLastOp.second.second;
            if (lastOpInQueue.getUpdateBarriers().empty()) {
                log.trace("Add finishing barrier for {0}", lastOpInQueue->getLoc());
                lastOpInQueue.getUpdateBarriersMutable().assign(finalBarrier);
            }
        }
    } else {
        // If FIFO final tasks already update a barrier without any consumers, assign to it the final barrier attribute.
        // Final tasks on FIFO that update barriers which are not final (i.e. has other consumers) do not need to update
        // the final barrier. Such connections could have been optimized in the legalization pipeline and do not need to
        // be created.
        std::set<VPURT::DeclareVirtualBarrierOp> newFinalBarriers;
        for (auto taskQueueFirstAndLastOp : vpux::VPURT::getTaskQueuesFirstAndLastOp(funcOp)) {
            auto& lastOpInQueue = taskQueueFirstAndLastOp.second.second;
            for (auto bar : lastOpInQueue.getUpdateBarriers()) {
                auto barrierOp = bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
                if (!hasAnyConsumers(barrierOp)) {
                    newFinalBarriers.insert(barrierOp);
                }
            }
        }
        VPUX_THROW_UNLESS(newFinalBarriers.size() == 1,
                          "Found multiple final barrier candidates. Will not assign final barrier.");
        auto finalBarOp = *newFinalBarriers.begin();
        log.trace("Setting barrier {0} as final barrier", finalBarOp.getLoc());
        finalBarOp.setIsFinalBarrier(true);
    }

    return true;
}

bool VPURT::isShareWaitAndUpdateBarriersNeeded(std::optional<WorkloadManagementMode> workloadManagementMode) {
    // Disable share wait and update barriers for PWLM_V2_PAGE mode only for now
    // For PWLM_V0_LCA mode enable shared barriers in order to avoid performance regressions
    // For PWLM_V0_LCA/PWLM_V1_BARRIER_FIFO mode use shareWaitAndUpdateBarriers to avoid issues in case of
    // rollback to nonWLM and lack of legalization of DMA descriptor fetch and potential issues in enqueue
    if (workloadManagementMode.has_value() && workloadManagementMode.value() >= WorkloadManagementMode::PWLM_V2_PAGES) {
        return false;
    }
    // enable shared wait and update barriers
    return true;
}
