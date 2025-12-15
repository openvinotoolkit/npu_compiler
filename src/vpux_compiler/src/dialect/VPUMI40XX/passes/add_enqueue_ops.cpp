//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

#include <queue>

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_ADDENQUEUEOPS
#define GEN_PASS_DEF_ADDENQUEUEOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

static constexpr int64_t DMA_OUTSTANDING_TRANSACTIONS = 64;

class AddEnqueueOpsPass : public VPUMI40XX::impl::AddEnqueueOpsBase<AddEnqueueOpsPass> {
public:
    explicit AddEnqueueOpsPass(const WorkloadManagementMode workloadManagementMode, Logger log)
            : _enabledPredefinedEnqueue(workloadManagementMode != WorkloadManagementMode::PWLM_V0_LCA) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    bool _enabledPredefinedEnqueue;
    void safeRunOnFunc() final;
};

// Check if barrier that was chosen for enqueuing a task does not depend on a barrier
// that is to be produced by this task itself what will create a deadlock during execution
bool verifyEnqueueBarrierHasNoTopoDepOnBarrs(mlir::Value enqueueBar, mlir::ValueRange taskBars, Logger log) {
    // Identify minimal virtual ID of barrier produced by task. Barriers below
    // this ID will not be analyzed as they cannot be dependant of task barriers
    unsigned int minVid = std::numeric_limits<unsigned int>::max();
    for (auto taskBar : taskBars) {
        auto vid = mlir::cast<VPURegMapped::IndexType>(taskBar.getType()).getValue();
        minVid = std::min(minVid, vid);
    }

    if (mlir::cast<VPURegMapped::IndexType>(enqueueBar.getType()).getValue() < minVid) {
        return true;
    }

    mlir::DenseSet<mlir::Value> explored;
    std::queue<mlir::Value> queue;

    // Perform BFS starting from enqueu barrier up the dependency chain
    // to see if this barrier does not depend on any barriers produced
    // by task itself
    queue.push(enqueueBar);
    explored.insert(enqueueBar);

    while (!queue.empty()) {
        auto bar = queue.front();
        queue.pop();

        auto taskBarIt = std::find(taskBars.begin(), taskBars.end(), bar);
        if (taskBarIt != taskBars.end()) {
            auto enqueueBarOp = enqueueBar.getDefiningOp<VPUMI40XX::ConfigureBarrierOp>();
            auto taskBarOp = (*taskBarIt).getDefiningOp<VPUMI40XX::ConfigureBarrierOp>();
            log.warning("Enqueue barrier '{0}' depends topologically on task to be enqueued itself which updates "
                        "barrier '{1}'",
                        enqueueBarOp, taskBarOp);
            return false;
        }

        auto barOp = bar.getDefiningOp<VPUMI40XX::ConfigureBarrierOp>();
        for (auto barDep : barOp.getDependencies()) {
            // Ignore barriers which are earlier in schedule then minVid
            if (mlir::cast<VPURegMapped::IndexType>(barDep.getType()).getValue() < minVid) {
                continue;
            }

            if (explored.find(barDep) == explored.end()) {
                queue.push(barDep);
                explored.insert(barDep);
            }
        }
    }

    return true;
}

// Go through all enqueue tasks and process whole schedule with respect to barrier consumption events
// and check if no enqueue task chosen barrier is not yet fully consumed at the moment of enqueuement
// what means that it will be consumed by some future tasks not yet enqueued
mlir::LogicalResult verifyEnqueueBarrierIsNotBlockedByFutureTask(
        VPUMI40XX::MappedInferenceOp mpi, SmallVector<VPURegMapped::EnqueueOp>& enquOps,
        SmallVector<VPUMI40XX::ConfigureBarrierOp>& barriers,
        SmallVector<SmallVector<mlir::Operation*>>& lastDmaWithNoEnqueue, int64_t tilesCount, Logger log) {
    log.trace("Verify enqueue barrier is not blocked by future task");
    // Build information on all barrier consumer count
    SmallVector<int64_t> barrierConsumerCounter(barriers.size(), -1);
    auto barrierCount = barrierConsumerCounter.size();
    for (auto& barrier : barriers) {
        auto barrierIdx = mlir::cast<VPURegMapped::IndexType>(barrier.getResult().getType()).getValue();
        VPUX_THROW_WHEN(barrierIdx >= barrierCount,
                        "Invalid barrier index - {0} out of possible amount of barriers {1}", barrierIdx, barrierCount);
        VPUX_THROW_WHEN(barrierConsumerCounter[barrierIdx] > -1, "Barrier {0} has already been updated", barrierIdx);

        if (barrier.getConsumerCount().has_value()) {
            barrierConsumerCounter[barrierIdx] = barrier.getConsumerCount().value();
        }
    }

    // Before processing enqueue tasks first traverse all initial DMAs on each FIFO
    // as they do not have corresponding enqueue op. Update barriers they consume
    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (int64_t listIdx = 0; listIdx < 2; listIdx++) {
            auto dmaTask = mpi.getListHead(VPURegMapped::TaskType::DMA, tileIdx, listIdx);
            if (!dmaTask) {
                continue;
            }

            if (lastDmaWithNoEnqueue[tileIdx][listIdx] == nullptr) {
                continue;
            }

            auto firstDmaTaskIdx = mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue();
            auto lastDmaTaskIdx = mlir::cast<vpux::VPURegMapped::IndexType>(
                                          lastDmaWithNoEnqueue[tileIdx][listIdx]->getResult(0).getType())
                                          .getValue();

            log.trace("Process DMAs with no enqueue op - DMA{0}:{1}:{2}-{3}", tileIdx, listIdx, firstDmaTaskIdx,
                      lastDmaTaskIdx);

            mlir::Operation* dmaOp;
            mlir::Operation* nextDmaOp = dmaTask.getDefiningOp();

            do {
                dmaOp = nextDmaOp;

                if (auto executableTaskOp = mlir::dyn_cast<VPURegMapped::DMATypeOpInterface>(dmaOp)) {
                    for (const auto& waitBar : executableTaskOp.waitBarriers()) {
                        auto barrierIdx = mlir::cast<VPURegMapped::IndexType>(waitBar.getType()).getValue();
                        VPUX_THROW_WHEN(barrierIdx >= barrierCount,
                                        "Invalid barrier index - {0} out of possible amount of barriers {1}",
                                        barrierIdx, barrierCount);
                        barrierConsumerCounter[barrierIdx]--;
                    }
                }

                auto nextDma = VPUMI40XX::getNextOp(mlir::cast<VPURegMapped::TaskOpInterface>(dmaOp));
                nextDmaOp = nextDma ? nextDma.getOperation() : nullptr;

            } while (nextDmaOp != nullptr && dmaOp != lastDmaWithNoEnqueue[tileIdx][listIdx]);
        }
    }

    for (auto& enquOp : enquOps) {
        auto enqBarIdx = mlir::cast<VPURegMapped::IndexType>(enquOp.getBarrier().getType()).getValue();

        log.trace("EnqOp '{0}' at barrier idx '{1}'", enquOp.getIndex().getType(), enqBarIdx);
        if (barrierConsumerCounter[enqBarIdx] != 0) {
            log.warning("Barrier '{0}' for enqueue not yet consumed, remaining counters: '{1}'. Execution blocked at "
                        "enqueue op '{2}'",
                        enqBarIdx, barrierConsumerCounter[enqBarIdx], enquOp);
            return mlir::failure();
        }

        auto nextTaskOpToProcess = mlir::cast<VPURegMapped::TaskOpInterface>(enquOp.getStart().getDefiningOp());
        auto lastTaskOp = mlir::cast<VPURegMapped::TaskOpInterface>(enquOp.getEnd().getDefiningOp());
        VPURegMapped::TaskOpInterface taskOp;
        mlir::DenseSet<VPUMI40XX::ExecutableTaskOpInterface> processedDpuTasks;

        do {
            taskOp = nextTaskOpToProcess;

            VPUMI40XX::ExecutableTaskOpInterface barrieredOp;
            bool isDpuTask = false;  // check if barrieredOp is a DPU task

            auto taskDecrementsConsumerCounter = [&](VPUMI40XX::ExecutableTaskOpInterface& barrieredOp) {
                if (!isDpuTask) {
                    return true;
                }
                if (!processedDpuTasks.contains(barrieredOp)) {
                    // Since only first DPU variant configures wait barriers,
                    // allow to decrement the consumer counter only for the first encountered DPU variant
                    // of any given invariant.
                    processedDpuTasks.insert(barrieredOp);
                    return true;
                }
                return false;
            };

            if (enquOp.getTaskType() == VPURegMapped::TaskType::DPUVariant) {
                auto dpuVariantOp = mlir::cast<VPUMI40XX::DPUVariantOp>(taskOp.getOperation());
                barrieredOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(
                        dpuVariantOp.getInvariant().getDefiningOp());
                isDpuTask = true;
            } else {
                barrieredOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(taskOp.getOperation());
            }

            if (barrieredOp) {
                for (const auto& waitBar : barrieredOp.waitBarriers()) {
                    auto barrierIdx = mlir::cast<VPURegMapped::IndexType>(waitBar.getType()).getValue();
                    VPUX_THROW_WHEN(barrierIdx >= barrierCount,
                                    "Invalid barrier index - {0} out of possible amount of barriers {1}", barrierIdx,
                                    barrierCount);
                    if (taskDecrementsConsumerCounter(barrieredOp)) {
                        VPUX_THROW_UNLESS(barrierConsumerCounter[barrierIdx] > 0,
                                          "Barrier {0} consumer count cannot be decremented - {1}", barrierIdx,
                                          barrierConsumerCounter[barrierIdx]);
                        barrierConsumerCounter[barrierIdx]--;
                    }
                }
            }
            nextTaskOpToProcess = VPUMI40XX::getNextOp(taskOp);
        } while (taskOp != lastTaskOp);
    }

    return mlir::success();
}

mlir::LogicalResult verifyEnqueueOpsOrderIsAlignedWithPerFifoTaskOrder(SmallVector<VPURegMapped::EnqueueOp>& enquOps,
                                                                       Logger log) {
    llvm::DenseMap<VPUMI40XX::HwQueueType, uint32_t> lastTaskPerQueue;

    for (auto& enqu : enquOps) {
        auto tile = mlir::cast<VPURegMapped::IndexType>(enqu.getStart().getType()).getTileIdx();
        auto list = mlir::cast<VPURegMapped::IndexType>(enqu.getStart().getType()).getListIdx();
        auto index = mlir::cast<VPURegMapped::IndexType>(enqu.getStart().getType()).getValue();
        auto taskType = enqu.getTaskType();
        VPUMI40XX::HwQueueType qType({taskType, tile, list});

        if (lastTaskPerQueue.find(qType) != lastTaskPerQueue.end()) {
            if (lastTaskPerQueue[qType] > index) {
                log.warning("Incorrect position for enque {0} in WorkItem list", enqu);
                return mlir::failure();
            }
        }

        lastTaskPerQueue[qType] = index;
    }

    return mlir::success();
}

// For each task that depends also on descriptor fetching (DPU and SHV) search for enqueue barrier
// and create new enqueue op if needed
// This method will use an LCA algorithm on previous instance of tasks barriers and update barrier
// of fetch DMA.
mlir::LogicalResult addEnqusForTasksWithFetch(VPUMI40XX::MappedInferenceOp mpi, const VPURegMapped::TaskType primary,
                                              const VPURegMapped::TaskType secondary,
                                              VPURegMapped::EnqueueOp& globalPreviousEnqu, mlir::OpBuilder& builder,
                                              int64_t& counter, VPUMI40XX::lcaCache& cache, Logger log,
                                              const int64_t tilesCount, const int64_t listsCount = 1) {
    auto ctx = mpi.getContext();

    auto getFetchTask = [](mlir::Value result) {
        auto fetchIt = llvm::find_if(result.getUsers(), [](mlir::Operation* user) {
            return mlir::isa<VPURegMapped::FetchTaskOp>(user);
        });

        auto fetchTask = fetchIt != result.getUsers().end() ? mlir::cast<VPURegMapped::FetchTaskOp>(*fetchIt) : nullptr;
        return fetchTask;
    };

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (int64_t listIdx = 0; listIdx < listsCount; listIdx++) {
            auto startVal = mpi.getListHead(primary, tileIdx, listIdx);
            if (!startVal) {
                continue;
            }

            log.trace("Search for enqueue barriers for {0}:{1}", stringifyTaskType(primary), tileIdx);
            log = log.nest();

            // reset local previousEnqu
            VPURegMapped::EnqueueOp localPreviousEnqu;

            // strongly assume that the FIRST OP always has a fetchTask
            auto previousFetchTask = getFetchTask(startVal);
            VPUX_THROW_UNLESS(previousFetchTask, "Starting OP {0} does not have a fetchTask", startVal);
            VPURegMapped::TaskOpInterface taskOp = mlir::cast<VPURegMapped::TaskOpInterface>(startVal.getDefiningOp());
            do {
                auto filteredRange = to_small_vector(
                        taskOp.getResult().getUsers() | vpux::filtered([&secondary](mlir::Operation* op) {
                            // filter out the usages that are of the secondaryType
                            auto taskOp = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(op);
                            return taskOp && taskOp.getTaskType() == secondary;
                        }));

                auto firstSecondaryIt = vpux::min_element(filteredRange, VPUMI40XX::taskOpComparator);
                auto lastSecondaryIt = vpux::max_element(filteredRange, VPUMI40XX::taskOpComparator);
                auto firstSecondary = mlir::cast<VPURegMapped::TaskOpInterface>(**firstSecondaryIt);
                auto lastSecondary = mlir::cast<VPURegMapped::TaskOpInterface>(**lastSecondaryIt);

                auto fetchTask = getFetchTask(taskOp.getResult());
                previousFetchTask = fetchTask ? fetchTask : previousFetchTask;
                /*
                    FetchTask
                        |
                        |
                        .   Bar1
                        .   /
                       DMA0_1
                        |   \
                        |   Bar2
                        .
                        .

                    Start with the fetchTask, get all the users such that the previous task for user is fetchTask
                    For the next iteration this user becomes the task and the loop continues until we find a task which
                   is of type ExecutableTaskOpInterface e.g. DMA0_1 and use its updateBarriers to ensure the fetch task
                   has been completed
                */
                auto fetchTaskUpdateBarrs = VPUMI40XX::getClosestProductionBarriers(
                        mlir::cast<VPURegMapped::TaskOpInterface>(previousFetchTask.getOperation()));

                // all of them must have the same barrier
                // so take any - in our case last
                auto barrieredOp = VPUMI40XX::getBarrieredOp(taskOp, lastSecondary);
                // Similar approach comparing to DMA L:593, to ensure both wait,update have been reprogrammed before
                // Enqueuing the task have both wait and update barriers in the lca search space
                llvm::SmallVector<mlir::Value> targetBarriers(barrieredOp.updateBarriers().begin(),
                                                              barrieredOp.updateBarriers().end());
                llvm::append_range(targetBarriers, barrieredOp.waitBarriers());

                // you cannot take barriers of the task you enqueue, because enqueue happens at barrier consumption
                // and barrier consumption of wait barrier of a task won't happen before execution start of this task
                // so take previous usage of the same physical id = earliest you can enqueue
                auto previousUsages = VPUMI40XX::getPreviousUsages(targetBarriers);

                // If there are multiple barriers updated by fetch task we only need 1 to
                // take into account for LCA algorithm as it is sufficient to identify fetch
                // task completion. Taking multiple barriers only increases complexity for LCA
                if (!fetchTaskUpdateBarrs.empty()) {
                    previousUsages.push_back(*fetchTaskUpdateBarrs.begin());
                }

                // here searching for a barrier we're going down in a tree-order
                // lca gives us a collection of barrier, at each of them you can enqueue = where all of "previousUsages"
                // = (fetch production + previous barriers to barriers of task) are consumed
                auto enqueueTarget = VPUMI40XX::findEnqTargetUsingLcaForBars(
                        previousUsages, cache, VPUMI40XX::getLcaSearchLimit(targetBarriers));

                if (enqueueTarget == nullptr) {
                    log.warning("Could not find a lowest common ancestor for barriers of task '{0}'", barrieredOp);
                    return mlir::failure();
                }

                // we can get into a corner case where a subsequent DPU can be enqueued before it's preceding enqueue,
                // based on barrier constraints. we could include the previous barriers enqueue barrier into the
                // LCA, but need to prove that will be satisfactory. For now, assuming that barriers are
                // topologically ordered, we will compare with the previous enqueOp's triggering barriers list idx

                // not needed anymore as long as we are sure to topologically sort enqueue tasks themselves
                // since now Runtime guarantees ordered submission of enqueues

                // we order enqueues because RT requires that
                if (localPreviousEnqu) {
                    auto previousEnquBarrier = localPreviousEnqu.getBarrier();

                    enqueueTarget = std::max<mlir::Value>(
                            previousEnquBarrier, enqueueTarget, [](mlir::Value lhs, mlir::Value rhs) {
                                return mlir::cast<VPURegMapped::IndexType>(lhs.getType()).getValue() <
                                       mlir::cast<VPURegMapped::IndexType>(rhs.getType()).getValue();
                            });
                }

                // check we don't enqueue too late = after task should start executing
                if (!verifyEnqueueBarrierHasNoTopoDepOnBarrs(enqueueTarget, targetBarriers, log)) {
                    log.warning("Invalid enqueue barrier found for task '{0}'", taskOp);
                    return mlir::failure();
                }

                // if the previous enqueue's barrier is the same as the target barrier, we can just add this variant
                // range to the previous enqueue. This is made with the assumption that we topologically iterate over
                // the variants list by their listOrder
                if (localPreviousEnqu && (localPreviousEnqu.getBarrier() == enqueueTarget)) {
                    localPreviousEnqu.getEndMutable().assign(lastSecondary->getResult(0));
                    log.trace("Enqueue task {0} with previous task",
                              mlir::cast<VPURegMapped::IndexType>(firstSecondary->getResult(0).getType()).getValue());
                } else {
                    auto index = VPURegMapped::IndexType::get(ctx, counter);
                    mlir::Value previousEnquVal =
                            localPreviousEnqu ? localPreviousEnqu.getResult()
                                              : (globalPreviousEnqu ? globalPreviousEnqu.getResult() : nullptr);
                    localPreviousEnqu = builder.create<VPURegMapped::EnqueueOp>(
                            taskOp->getLoc(), index, previousEnquVal, enqueueTarget,
                            /*previousTaskIdxOnSameBarrier*/ nullptr, secondary, firstSecondary->getResult(0),
                            lastSecondary->getResult(0));
                    counter++;
                    log.trace("New enqueue for task {0} at barrier {1}",
                              mlir::cast<VPURegMapped::IndexType>(firstSecondary->getResult(0).getType()).getValue(),
                              mlir::cast<VPURegMapped::IndexType>(enqueueTarget.getType()).getValue());
                }

                taskOp = VPUMI40XX::getNextOp(taskOp);
            } while (taskOp);

            globalPreviousEnqu = localPreviousEnqu;
            log = log.unnest();
        }
    }

    return mlir::success();
}

// For each DMA task search for enqueue barrier and create new enqueue ops if needed
// This method will use an LCA algorithm on previous instance of task barriers.
// It will also take into account DMA FIFO size to not exceed the limit of allowed outstanding
// independent enqueues for DMA tasks
mlir::LogicalResult addEnqusForDmas(VPUMI40XX::MappedInferenceOp mpi, const int64_t tilesCount,
                                    VPURegMapped::EnqueueOp& globalPreviousEnqu, mlir::OpBuilder& builder,
                                    int64_t& counter, SmallVector<SmallVector<mlir::Operation*>>& lastDmaWithNoEnqueue,
                                    VPUMI40XX::lcaCache& cache, Logger log) {
    auto ctx = mpi.getContext();

    std::array<VPUMI40XX::ExecutableTaskOpInterface, DMA_OUTSTANDING_TRANSACTIONS> outstandingEnqueuedDmas;

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (int64_t listIdx = 0; listIdx < 2; listIdx++) {
            auto dmaTask = mpi.getListHead(VPURegMapped::TaskType::DMA, tileIdx, listIdx);
            if (!dmaTask) {
                continue;
            }

            log.trace("Search for enqueue barrier for {0}:{1}:{2}", stringifyTaskType(VPURegMapped::TaskType::DMA),
                      tileIdx, listIdx);
            log = log.nest();

            // reset local previousEnqu
            VPURegMapped::EnqueueOp localPreviousEnqu;
            // reset previousBuffer
            outstandingEnqueuedDmas.fill(nullptr);
            int64_t outstandingEnquOpsCounter = 0;

            while (dmaTask) {
                auto executableTaskOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(dmaTask.getDefiningOp());
                if (executableTaskOp &&
                    (executableTaskOp.updateBarriers().size() || executableTaskOp.waitBarriers().size())) {
                    // Include both updateBarriers and waitBarriers in the search space for lca
                    // Only having one of them doesn't ensure right place for enqueue leading to enqueue early and
                    // causing barrier underflow
                    /*
                        B2Prev (428:14)
                            |
                            .
                            .
                        B1Prev (429:5) <- Enq DMA0_1 (Enqueue happens after reprogramming of B1Prev by runtime)
                            |
                            |
                            .
                            .

                            B1 (460:5)
                            |
                          DMA_01
                            |
                            B2 (461:14)

                       In the case above if we only choose updateBarrier's (B2) previous usage (B2Prev) for lca and
                       B2Prev is chosen as Enqueue for DMA_01. Then the problem is that DMA_01 is in the FIFO and
                       starts listening to production of Phy5 which is also mapped to B1Prev apart from B1 and hence
                       decrements the counter after execution for Phy5 causing underflow
                    */
                    llvm::SmallVector<mlir::Value> targetBarrs(executableTaskOp.waitBarriers().begin(),
                                                               executableTaskOp.waitBarriers().end());
                    llvm::append_range(targetBarrs, executableTaskOp.updateBarriers());
                    auto previousUsages = VPUMI40XX::getPreviousUsages(targetBarrs);

                    auto oldestOutstandingDma = outstandingEnqueuedDmas[outstandingEnquOpsCounter];
                    if (oldestOutstandingDma) {
                        auto outstandingBarrierCondition = oldestOutstandingDma.updateBarriers();
                        previousUsages.append(outstandingBarrierCondition.begin(), outstandingBarrierCondition.end());
                    }

                    if (!previousUsages.empty()) {
                        auto enqueueTarget = VPUMI40XX::findEnqTargetUsingLcaForBars(
                                previousUsages, cache, VPUMI40XX::getLcaSearchLimit(targetBarrs));

                        if (enqueueTarget == nullptr) {
                            log.warning("Could not find a lowest common ancestor for barriers of task '{0}'",
                                        executableTaskOp);
                            return mlir::failure();
                        }

                        if (localPreviousEnqu) {
                            auto previousEnquBarrier = localPreviousEnqu.getBarrier();
                            enqueueTarget = std::max<mlir::Value>(
                                    previousEnquBarrier, enqueueTarget, [](mlir::Value lhs, mlir::Value rhs) {
                                        return mlir::cast<VPURegMapped::IndexType>(lhs.getType()).getValue() <
                                               mlir::cast<VPURegMapped::IndexType>(rhs.getType()).getValue();
                                    });
                        }

                        if (!verifyEnqueueBarrierHasNoTopoDepOnBarrs(enqueueTarget, targetBarrs, log)) {
                            log.warning("Invalid enqueue barrier found for task '{0}'", executableTaskOp);
                            return mlir::failure();
                        }

                        if (localPreviousEnqu && (localPreviousEnqu.getBarrier() == enqueueTarget)) {
                            log.trace("Enqueue task {0} with previous task",
                                      mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue());
                            localPreviousEnqu.getEndMutable().assign(dmaTask);
                        } else {
                            auto index = VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(counter));
                            mlir::Value previousEnquVal =
                                    localPreviousEnqu ? localPreviousEnqu.getResult()
                                                      : (globalPreviousEnqu ? globalPreviousEnqu.getResult() : nullptr);
                            localPreviousEnqu = builder.create<VPURegMapped::EnqueueOp>(
                                    dmaTask.getLoc(), index, previousEnquVal, enqueueTarget,
                                    /*previousTaskIdxOnSameBarrier*/ nullptr, VPURegMapped::TaskType::DMA, dmaTask,
                                    dmaTask);

                            outstandingEnqueuedDmas[outstandingEnquOpsCounter] = executableTaskOp;
                            outstandingEnquOpsCounter = (outstandingEnquOpsCounter + 1) % DMA_OUTSTANDING_TRANSACTIONS;
                            counter++;
                            log.trace("New enqueue for task {0} at barrier {1}",
                                      mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue(),
                                      mlir::cast<VPURegMapped::IndexType>(enqueueTarget.getType()).getValue());
                        }
                    } else {
                        if (localPreviousEnqu) {
                            log.trace("Enqueue task {0} with previous task",
                                      mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue());
                            localPreviousEnqu.getEndMutable().assign(dmaTask);
                        } else {
                            log.trace("Enqueue task {0} at bootstrap",
                                      mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue());
                            lastDmaWithNoEnqueue[tileIdx][listIdx] = dmaTask.getDefiningOp();
                        }
                    }
                } else if (localPreviousEnqu) {
                    log.trace("Enqueue task {0} with previous task",
                              mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue());
                    localPreviousEnqu.getEndMutable().assign(dmaTask);
                } else {
                    log.trace("Enqueue task {0} at bootstrap",
                              mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue());
                    lastDmaWithNoEnqueue[tileIdx][listIdx] = dmaTask.getDefiningOp();
                }

                auto nextDma = VPUMI40XX::getNextOp(mlir::cast<VPURegMapped::TaskOpInterface>(dmaTask.getDefiningOp()));
                dmaTask = nextDma ? nextDma.getResult() : nullptr;
            }
            log = log.unnest();
        }
    }

    return mlir::success();
}

// For each task that depends also on descriptor fetching (DPU and SHV)
// read preconfigured enqueue barrier and create new enqueue ops
void addPredefinedEnqusForTasksWithFetch(VPUMI40XX::MappedInferenceOp mpi, const VPURegMapped::TaskType primary,
                                         const VPURegMapped::TaskType secondary,
                                         VPURegMapped::EnqueueOp& globalPreviousEnqu, mlir::OpBuilder& builder,
                                         int64_t& counter, Logger log, const int64_t tilesCount,
                                         const int64_t listsCount = 1) {
    auto ctx = mpi.getContext();

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (int64_t listIdx = 0; listIdx < listsCount; listIdx++) {
            auto startVal = mpi.getListHead(primary, tileIdx, listIdx);
            if (!startVal) {
                continue;
            }

            log.trace("Get enqueue barriers for {0}:{0}:{1}:{2}", stringifyTaskType(primary), tileIdx, listIdx);
            log = log.nest();

            // reset local previousEnqu
            VPURegMapped::EnqueueOp localPreviousEnqu;

            VPURegMapped::TaskOpInterface taskOp = mlir::cast<VPURegMapped::TaskOpInterface>(startVal.getDefiningOp());
            do {
                auto filteredRange = to_small_vector(
                        taskOp.getResult().getUsers() | vpux::filtered([&secondary](mlir::Operation* op) {
                            // filter out the usages that are of the secondaryType
                            auto taskOp = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(op);
                            return taskOp && taskOp.getTaskType() == secondary;
                        }));

                auto firstSecondaryIt = vpux::min_element(filteredRange, VPUMI40XX::taskOpComparator);
                auto lastSecondaryIt = vpux::max_element(filteredRange, VPUMI40XX::taskOpComparator);
                auto firstSecondary = mlir::cast<VPURegMapped::TaskOpInterface>(**firstSecondaryIt);
                auto lastSecondary = mlir::cast<VPURegMapped::TaskOpInterface>(**lastSecondaryIt);

                auto barrieredOp = VPUMI40XX::getBarrieredOp(taskOp, lastSecondary);

                auto enqueueTarget = barrieredOp.getEnqueueBarrier();

                VPUX_THROW_UNLESS(enqueueTarget, "No enqueue barrier configured for op {0}", barrieredOp);

                // if the previous enqueue's barrier is the same as the target barrier, we can just add this variant
                // range to the previous enqueue. This is made with the assumption that we topologically iterate over
                // the variants list by their listOrder
                if (localPreviousEnqu && (localPreviousEnqu.getBarrier() == enqueueTarget)) {
                    localPreviousEnqu.getEndMutable().assign(lastSecondary->getResult(0));
                    log.trace("Enqueue task {0} with previous task",
                              mlir::cast<VPURegMapped::IndexType>(firstSecondary->getResult(0).getType()).getValue());
                } else {
                    auto index = VPURegMapped::IndexType::get(ctx, counter);
                    mlir::Value previousEnquVal =
                            localPreviousEnqu ? localPreviousEnqu.getResult()
                                              : (globalPreviousEnqu ? globalPreviousEnqu.getResult() : nullptr);
                    localPreviousEnqu = builder.create<VPURegMapped::EnqueueOp>(
                            taskOp->getLoc(), index, previousEnquVal, enqueueTarget,
                            /*previousTaskIdxOnSameBarrier*/ nullptr, secondary, firstSecondary->getResult(0),
                            lastSecondary->getResult(0));
                    counter++;
                    log.trace("New enqueue for task {0} at barrier {1}",
                              mlir::cast<VPURegMapped::IndexType>(firstSecondary->getResult(0).getType()).getValue(),
                              mlir::cast<VPURegMapped::IndexType>(enqueueTarget.getType()).getValue());
                }

                taskOp = VPUMI40XX::getNextOp(taskOp);
            } while (taskOp);

            globalPreviousEnqu = localPreviousEnqu;
            log = log.unnest();
        }
    }
}

// For each DMA task read preconfigured enqueue barrier and create new enqueue ops
void addPredefinedEnqusForDmas(VPUMI40XX::MappedInferenceOp mpi, const int64_t tilesCount,
                               VPURegMapped::EnqueueOp& globalPreviousEnqu, mlir::OpBuilder& builder, int64_t& counter,
                               SmallVector<SmallVector<mlir::Operation*>>& lastDmaWithNoEnqueue, Logger log) {
    auto ctx = mpi.getContext();

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (int64_t listIdx = 0; listIdx < 2; listIdx++) {
            auto dmaTask = mpi.getListHead(VPURegMapped::TaskType::DMA, tileIdx, listIdx);
            if (!dmaTask) {
                continue;
            }

            log.trace("Get enqueue barriers for {0}:{1}:{2}", stringifyTaskType(VPURegMapped::TaskType::DMA), tileIdx,
                      listIdx);
            log = log.nest();

            // reset local previousEnqu
            VPURegMapped::EnqueueOp localPreviousEnqu;

            while (dmaTask) {
                auto executableTaskOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(dmaTask.getDefiningOp());

                if (executableTaskOp != nullptr && executableTaskOp.getEnqueueBarrier() != nullptr) {
                    auto enqueueTarget = executableTaskOp.getEnqueueBarrier();

                    if (localPreviousEnqu && (localPreviousEnqu.getBarrier() == enqueueTarget)) {
                        log.trace("Enqueue task {0} with previous task",
                                  mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue());
                        localPreviousEnqu.getEndMutable().assign(dmaTask);
                    } else {
                        auto index = VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(counter));
                        mlir::Value previousEnquVal =
                                localPreviousEnqu ? localPreviousEnqu.getResult()
                                                  : (globalPreviousEnqu ? globalPreviousEnqu.getResult() : nullptr);
                        localPreviousEnqu = builder.create<VPURegMapped::EnqueueOp>(
                                dmaTask.getLoc(), index, previousEnquVal, enqueueTarget,
                                /*previousTaskIdxOnSameBarrier*/ nullptr, VPURegMapped::TaskType::DMA, dmaTask,
                                dmaTask);

                        counter++;
                        log.trace("New enqueue for task {0} at barrier {1}",
                                  mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue(),
                                  mlir::cast<VPURegMapped::IndexType>(enqueueTarget.getType()).getValue());
                    }
                } else if (localPreviousEnqu) {
                    log.trace("Enqueue task {0} with previous task",
                              mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue());
                    localPreviousEnqu.getEndMutable().assign(dmaTask);
                } else {
                    log.trace("Enqueue task {0} at bootstrap",
                              mlir::cast<VPURegMapped::IndexType>(dmaTask.getType()).getValue());
                    lastDmaWithNoEnqueue[tileIdx][listIdx] = dmaTask.getDefiningOp();
                }

                auto nextDma = VPUMI40XX::getNextOp(mlir::cast<VPURegMapped::TaskOpInterface>(dmaTask.getDefiningOp()));
                dmaTask = nextDma ? nextDma.getResult() : nullptr;
            }
            log = log.unnest();
        }
    }
}

// After inserting enqueue ops for each FIFO in the IR, enqueue ops need to be ordered
// as there is only 1 enqueue ops (WorkItem task) list that will be processed by VPU-FW
// There are following restrictions:
// 1:
// Prepared enqueues order need to always guarantee that for given HW FIFO tasks are not
// enqueued out of order, meaning enqueue for taskFifoX[N] is not placed after enqueue
// for taskFifoX[N+1]
// This constraint is verified by verifyEnqueueOpsOrderIsAlignedWithPerFifoTaskOrder()
//
// 2:
// For each Enq[j] and Enq[i], such that j > i, tasks enqueued by Enq[j] cannot block barrier consumption
// of any task enqueued by Enq[i]
// Example for incorrect order:
//  Enq[0]: Bar0, taskX[0]
//  Enq[1]: Bar2, taskX[1]
//  Enq[2]: Bar1, taskY[1], taskY[1].wait(Bar2)
//
// Although above order satisfies constraint 1 it will cause a deadlock as Enq[1] happening at Bar2
// will never happen as Bar2 will be consumed by task which is enqueued later at Enq[2]
// In that case Enq[2] should be placed before Enq[1]
//
// This constraint can be satisfied by updating the order of enqueues with order of barriers - smaller
// index first. Barriers are ordered based on consumption order so if Bar[i] depends on Bar[j] it is
// guaranteed to have larger index (i > j)
// Important: Larger index does not always mean barrier dependency, for example Bar[N+1] does not
// need to depend on Bar[N] in schedule if they are on parallel branch
//
// This constraint is verified by verifyEnqueueBarrierIsNotBlockedByFutureTask()
//
// To satisfy both constraints algorithm should process enqueues for tasks following
// their order in given HW FIFO to satisfy constraint 1 and when picking HW FIFO from
// which enqueue to order first should use smaller barrier index to satisfy constraint 2
SmallVector<VPURegMapped::EnqueueOp> getEnqueueOpsOrder(mlir::func::FuncOp netFunc) {
    size_t enquOpsCount = 0;
    // Store enqueue ops information per FIFO
    // For each HW FIFO (identified by VPUMI40XX::HwQueueType) vector of enqueue ops
    // is ordered for given FIFO as this is how they were inserted by previous functions
    // which processed IR top to bottom
    // Example:
    //  HW FIFO X: Enq(Bar1, taskX[0-1]), Enq(Bar3, taskX[2])
    //  HW FIFO Y: Enq(Bar2, taskY[0]), Enq(Bar0, taskY[1-2])
    llvm::DenseMap<VPUMI40XX::HwQueueType, SmallVector<VPURegMapped::EnqueueOp>> enquOpsPerQueue;
    for (auto enquOp : netFunc.getOps<VPURegMapped::EnqueueOp>()) {
        // EnqueueOp can cover range of tasks start(tile:list:startIdx)-end(tile:list:endIdx)
        // Tile and list is guaranteed to be the same for start and end because they both identify
        // HW FIFO and enqueue op can have tasks only from HW FIFO
        auto tile = mlir::cast<VPURegMapped::IndexType>(enquOp.getStart().getType()).getTileIdx();
        auto list = mlir::cast<VPURegMapped::IndexType>(enquOp.getStart().getType()).getListIdx();
        auto taskType = enquOp.getTaskType();
        VPUMI40XX::HwQueueType queueType({taskType, tile, list});
        enquOpsPerQueue[queueType].push_back(enquOp);
        enquOpsCount++;
    }

    // Vector for storing final order of enqueue ops
    SmallVector<VPURegMapped::EnqueueOp> enquOpsOrder;
    enquOpsOrder.reserve(enquOpsCount);

    // For each HW FIFO store index of next enqueue op to be processed
    mlir::DenseMap<VPUMI40XX::HwQueueType, size_t> perQueueCurrentEnqIndex;

    size_t i = 0;
    while (i < enquOpsCount) {
        // From all HW FIFOs pick next enqueue with the smallest barrier
        // index. Take into account only top of the list for each HW FIFO
        // identified by perQueueCurrentEnqIndex
        // Example:
        //  HW FIFO X: Enq(Bar1, TaskX[0]), Enq(Bar2, TaskX[1]), ...
        //  HW FIFO Y: Enq(Bar2, TaskY[0]), Enq(Bar3, TaskX[1]), ...
        //  HW FIFO Z: Enq(Bar1, TaskZ[0]), Enq(Bar2, TaskZ[1]), ...
        // Next enqueue barriers: Bar1, Bar2, Bar1
        // Min barrier: Bar1
        auto minEnqOpBar = std::numeric_limits<uint32_t>::max();
        for (const auto& [queueType, enqVecOnQueue] : enquOpsPerQueue) {
            auto indexInQueue = perQueueCurrentEnqIndex[queueType];
            // Check if all enqueues from this FIFO were accounted for already and skip
            if (indexInQueue == enqVecOnQueue.size()) {
                continue;
            }
            auto enqOp = enqVecOnQueue[indexInQueue];
            auto enqOpBar = mlir::cast<VPURegMapped::IndexType>(enqOp.getBarrier().getType()).getValue();
            minEnqOpBar = std::min(minEnqOpBar, enqOpBar);
        }

        // After identifying next enqueue barrier pick all enqueues from top of each FIFOs that match this barrier
        // For previous example that would be 2 enqueues: for HW FIFO X and Z:
        // Enq(Bar1, TaskX[0]) and Enq(Bar1, TaskZ[0])
        // Push those enqueues to final order vector and increment counters
        // for those FIFOs
        for (auto& [queueType, enqVecOnQueue] : enquOpsPerQueue) {
            auto indexInQueue = perQueueCurrentEnqIndex[queueType];
            // Check if all enqueues from this FIFO were accounted for already and skip
            if (indexInQueue >= enqVecOnQueue.size()) {
                continue;
            }
            auto enqOp = enqVecOnQueue[indexInQueue];
            auto enqOpBar = mlir::cast<VPURegMapped::IndexType>(enqOp.getBarrier().getType()).getValue();
            // Barrier for this HW FIFO does not match indetified barrier, do not pick this enqueue
            if (enqOpBar != minEnqOpBar) {
                continue;
            }
            enquOpsOrder.push_back(enqOp);
            perQueueCurrentEnqIndex[queueType]++;
            i++;
        }
    }

    return enquOpsOrder;
}

void AddEnqueueOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto module = netFunc->getParentOfType<mlir::ModuleOp>();

    if (enablePredefinedEnqueueOpt.hasValue()) {
        _enabledPredefinedEnqueue = enablePredefinedEnqueueOpt.getValue();
    }

    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = config::getTileExecutor(parentModule).getCount();
    const auto shavesCountPerTile = config::getAvailableExecutor(parentModule, VPU::ExecutorKind::SHAVE_ACT).getCount();

    auto barriers = to_small_vector(netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>());

    VPURegMapped::EnqueueOp globalPreviousEnqu;
    int64_t globalEnquCounter = 0;

    // Store information on last DMAs on each FIFO which does not have a corresponding
    // enqueue op added in this pass.
    SmallVector<SmallVector<mlir::Operation*>> lastDmaWithNoEnqueue(tilesCount, SmallVector<mlir::Operation*>(2));

    if (_enabledPredefinedEnqueue) {
        _log.trace("Use already configured enqueue barriers by algorithm from VPURT");

        addPredefinedEnqusForTasksWithFetch(mpi, VPURegMapped::TaskType::DPUInvariant,
                                            VPURegMapped::TaskType::DPUVariant, globalPreviousEnqu, builder,
                                            globalEnquCounter, _log, tilesCount);
        addPredefinedEnqusForTasksWithFetch(mpi, VPURegMapped::TaskType::ActKernelRange,
                                            VPURegMapped::TaskType::ActKernelInvocation, globalPreviousEnqu, builder,
                                            globalEnquCounter, _log, tilesCount, shavesCountPerTile);

        addPredefinedEnqusForDmas(mpi, tilesCount, globalPreviousEnqu, builder, globalEnquCounter, lastDmaWithNoEnqueue,
                                  _log);

        // After inserting enqueue ops for each FIFO in the IR, enqueue ops need to be ordered
        // as there is only 1 enqueue ops (WorkItem task) list that will be processed by VPU-FW
        auto enquOpsOrder = getEnqueueOpsOrder(netFunc);

        // Update enqueue ops order in IR
        for (auto& enqu : enquOpsOrder) {
            enqu.getOperation()->moveBefore(mpi.getOperation());
        }

    } else {
        _log.trace("Perform enqueue search");

        // We often call LCA for same pair of barriers in that case having cache is beneficial
        VPUMI40XX::lcaCache cache;

        if (mlir::failed(addEnqusForTasksWithFetch(mpi, VPURegMapped::TaskType::DPUInvariant,
                                                   VPURegMapped::TaskType::DPUVariant, globalPreviousEnqu, builder,
                                                   globalEnquCounter, cache, _log, tilesCount))) {
            config::setWorkloadManagementStatus(module, WorkloadManagementStatus::FAILED);
            signalPassFailure();
            return;
        }
        if (mlir::failed(addEnqusForTasksWithFetch(
                    mpi, VPURegMapped::TaskType::ActKernelRange, VPURegMapped::TaskType::ActKernelInvocation,
                    globalPreviousEnqu, builder, globalEnquCounter, cache, _log, tilesCount, shavesCountPerTile))) {
            config::setWorkloadManagementStatus(module, WorkloadManagementStatus::FAILED);
            signalPassFailure();
            return;
        }

        if (mlir::failed(addEnqusForDmas(mpi, tilesCount, globalPreviousEnqu, builder, globalEnquCounter,
                                         lastDmaWithNoEnqueue, cache, _log))) {
            config::setWorkloadManagementStatus(module, WorkloadManagementStatus::FAILED);
            signalPassFailure();
            return;
        }

        // for multi tile need to sort enqueuOps to be contiguous for the same barrier
        for (auto barrier : barriers) {
            llvm::DenseMap<VPURegMapped::TaskType, llvm::SmallVector<VPURegMapped::EnqueueOp>> buckets;
            for (auto user : barrier.getResult().getUsers()) {
                auto enqu = mlir::dyn_cast<VPURegMapped::EnqueueOp>(user);
                if (!enqu) {
                    continue;
                }

                buckets[enqu.getTaskType()].push_back(enqu);
            }

            for (auto& mapIt : buckets) {
                llvm::sort(mapIt.getSecond(), [](VPURegMapped::EnqueueOp lhs, VPURegMapped::EnqueueOp rhs) {
                    return mlir::cast<VPURegMapped::IndexType>(lhs.getResult().getType()).getValue() <
                           mlir::cast<VPURegMapped::IndexType>(rhs.getResult().getType()).getValue();
                });
                for (auto enqu : mapIt.getSecond()) {
                    enqu.getOperation()->moveBefore(mpi.getOperation());
                }
            }
        }
    }

    auto enquOps = to_small_vector(netFunc.getOps<VPURegMapped::EnqueueOp>());
    if (!enquOps.empty()) {
        mpi.getWorkItemTasksMutable().assign(enquOps[0].getResult());
    }
    mpi.setWorkItemCount(enquOps.size());
    VPUMI40XX::reindexEnqueueOps(enquOps);

    // Verify enqueue ops can be enqueued at given barriers
    if (mlir::failed(verifyEnqueueBarrierIsNotBlockedByFutureTask(mpi, enquOps, barriers, lastDmaWithNoEnqueue,
                                                                  tilesCount, _log))) {
        config::setWorkloadManagementStatus(module, WorkloadManagementStatus::FAILED);
        signalPassFailure();
        return;
    }

    // Check if enqueues order for given HW FIFO is not enqueueing tasks
    // for this FIFO out of order - task N needs to be enqueued before task N+1
    if (mlir::failed(verifyEnqueueOpsOrderIsAlignedWithPerFifoTaskOrder(enquOps, _log))) {
        config::setWorkloadManagementStatus(module, WorkloadManagementStatus::FAILED);
        signalPassFailure();
        return;
    }
}

}  // namespace

//
// createAddEnqueueOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createAddEnqueueOpsPass(WorkloadManagementMode workloadManagementMode,
                                                                     Logger log) {
    return std::make_unique<AddEnqueueOpsPass>(workloadManagementMode, log);
}
