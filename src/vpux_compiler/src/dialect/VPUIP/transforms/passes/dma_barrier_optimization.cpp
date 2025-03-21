//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

#include <llvm/ADT/SetOperations.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_DMABARRIEROPTIMIZATION
#define GEN_PASS_DEF_DMABARRIEROPTIMIZATION
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
namespace {

// remove barrier consumers and/or producers which are controlled
// by FIFO dependency. DMA[{FIFO}]
/*
    DMA[0] DMA[0] DMA[1]       DMA[0] DMA[1]
        \    |    /               \   /
            Bar           =>       Bar
        /    |    \               /   \
    DMA[0] DMA[0] DMA[1]       DMA[0] DMA[1]
*/
void removeRedundantDependencies(BarrierInfo& barrierInfo, bool considerTaskFifoDependency, vpux::Logger log) {
    const auto findRedundantDependencies = [&](const SmallVector<llvm::BitVector>& taskControlMap,
                                               size_t taskControlMapOffset, const BarrierInfo::TaskSet& dependencies,
                                               bool producer = true) {
        // find dependencies to remove
        BarrierInfo::TaskSet dependenciesToRemove;
        for (auto taskIndexIter = dependencies.begin(); taskIndexIter != dependencies.end(); ++taskIndexIter) {
            if (barrierInfo.isSyncPoint(*taskIndexIter)) {
                continue;
            }
            for (auto nextIndex = std::next(taskIndexIter); nextIndex != dependencies.end(); ++nextIndex) {
                if (barrierInfo.isSyncPoint(*nextIndex) ||
                    !barrierInfo.controlPathExistsBetweenTasksInSameBlock(
                            taskControlMap, *taskIndexIter - taskControlMapOffset, *nextIndex - taskControlMapOffset,
                            /* biDirection */ false)) {
                    continue;
                }

                if (producer) {
                    dependenciesToRemove.insert(*taskIndexIter);
                } else {
                    dependenciesToRemove.insert(*nextIndex);
                }
            }
        }
        return dependenciesToRemove;
    };

    // Perform optimization in tasks blocks matching the distribution of synchronization points.
    for (size_t taskBlockIndex = 0; taskBlockIndex < barrierInfo.getControlGraphBlockCount(); ++taskBlockIndex) {
        // Build or update the control relationship between any two tasks. Note that the relationship includes the
        // dependency by the barriers as well as the implicit dependence by FIFO
        auto [taskControlMap, controlMapOffset] =
                barrierInfo.buildTaskControlMap(taskBlockIndex, considerTaskFifoDependency);

        // get update barriers range for current block
        auto blockUpdateBarriers =
                barrierInfo.getBarriersForTaskBlock(taskBlockIndex, /* blockStartSyncPoint */ true,
                                                    /* blockEndSyncPoint */ false, /* updateBarriers */ true);

        if (blockUpdateBarriers.empty()) {
            log.trace("No update barriers found in tasks block {0}", taskBlockIndex);
        } else {
            log.trace("Optimize producers for barrier range [{0}, {1}] in tasks block {2}", blockUpdateBarriers.front(),
                      blockUpdateBarriers.back(), taskBlockIndex);
        }

        for (auto barrierIdx : blockUpdateBarriers) {
            // find producers to remove
            const auto& barrierProducers = barrierInfo.getBarrierProducers(barrierIdx);
            const auto producersToRemove =
                    findRedundantDependencies(taskControlMap, controlMapOffset, barrierProducers);
            barrierInfo.removeProducers(barrierIdx, producersToRemove);
        }

        // get wait barriers range for current block
        auto blockWaitBarriers =
                barrierInfo.getBarriersForTaskBlock(taskBlockIndex, /* blockStartSyncPoint */ false,
                                                    /* blockEndSyncPoint */ true, /* updateBarriers */ false);
        if (blockWaitBarriers.empty()) {
            log.trace("No wait barriers found in tasks block {0}", taskBlockIndex);
        } else {
            log.trace("Optimize consumers for barrier range [{0}, {1}] in tasks block {2}", blockWaitBarriers.front(),
                      blockWaitBarriers.back(), taskBlockIndex);
        }

        for (auto barrierIdx : blockWaitBarriers) {
            // find consumers to remove
            const auto& barrierConsumers = barrierInfo.getBarrierConsumers(barrierIdx);
            const auto consumersToRemove =
                    findRedundantDependencies(taskControlMap, controlMapOffset, barrierConsumers, false);
            barrierInfo.removeConsumers(barrierIdx, consumersToRemove);
        }
    }
}

// Remove explicit barrier dependency between DMAs
// 1) if a barrier only has DMAs using single port as its producer,
//    remove all DMAs using the same port from its consumers. DMA[{FIFO}]
/*
    DMA[0] DMA[0]       DMA[0] DMA[0]
       \   /               \   /
        Bar         =>      Bar
       /   \                 |
    DMA[0] DMA[1]          DMA[1]
*/
// 2) if a barrier only has DMAs using single port as its consumer,
//    remove all DMAs using the same port from its producers. DMA[{FIFO}]
/*
    DMA[0] DMA[1]          DMA[1]
       \   /                 |
        Bar         =>      Bar
       /   \               /   \
    DMA[0] DMA[0]       DMA[0] DMA[0]
*/

void removeExplicitDependencies(BarrierInfo& barrierInfo) {
    const auto findExplicitDependencies = [&](const BarrierInfo::TaskSet& dependencies,
                                              const VPURT::TaskQueueType& type) {
        BarrierInfo::TaskSet dependenciesToRemove;
        for (auto& taskInd : dependencies) {
            if (barrierInfo.isSyncPoint(taskInd)) {
                continue;
            }
            if (type == VPURT::getTaskQueueType(barrierInfo.getTaskOpAtIndex(taskInd), false)) {
                dependenciesToRemove.insert(taskInd);
            }
        }
        return dependenciesToRemove;
    };

    // Perform optimization in tasks blocks matching the distribution of synchronization points.
    for (size_t taskBlockIndex = 0; taskBlockIndex < barrierInfo.getControlGraphBlockCount(); ++taskBlockIndex) {
        // get update barriers range for current block
        auto blockUpdateBarriers =
                barrierInfo.getBarriersForTaskBlock(taskBlockIndex, /* blockStartSyncPoint */ true,
                                                    /* blockEndSyncPoint */ false, /* updateBarriers */ true);

        for (auto barrierIdx : blockUpdateBarriers) {
            // try to optimize consumers (1)
            const auto& barrierProducers = barrierInfo.getBarrierProducers(barrierIdx);
            auto producerTaskQueueType = barrierInfo.haveSameImplicitDependencyTaskQueueType(barrierProducers);
            if (producerTaskQueueType.has_value()) {
                // barrier produced by tasks with same type
                auto consumersToRemove = findExplicitDependencies(barrierInfo.getBarrierConsumers(barrierIdx),
                                                                  producerTaskQueueType.value());
                // remove consumers
                barrierInfo.removeConsumers(barrierIdx, consumersToRemove);
            }

            // try to optimize producers (2)
            const auto& barrierConsumers = barrierInfo.getBarrierConsumers(barrierIdx);
            auto consumerTaskQueueType = barrierInfo.haveSameImplicitDependencyTaskQueueType(barrierConsumers);
            if (consumerTaskQueueType.has_value() || barrierConsumers.empty()) {
                // barrier consumed by tasks with same type
                BarrierInfo::TaskSet producersToRemove;
                // find producers to remove
                if (barrierConsumers.empty()) {
                    for (const auto& barProd : barrierProducers) {
                        if (barrierInfo.isSyncPoint(barProd)) {
                            continue;
                        }
                        producersToRemove.insert(barProd);
                    }
                } else {
                    producersToRemove = findExplicitDependencies(barrierProducers, consumerTaskQueueType.value());
                }

                // remove producers
                barrierInfo.removeProducers(barrierIdx, producersToRemove);
            }

            bool invalidOptimization = barrierInfo.getBarrierConsumers(barrierIdx).empty() ^
                                       barrierInfo.getBarrierProducers(barrierIdx).empty();
            VPUX_THROW_WHEN(
                    invalidOptimization, "Invalid optimization : Only barrier {0} became empty for barrier '{1}'",
                    barrierProducers.empty() ? "producers" : "consumers", barrierInfo.getBarrierOpAtIndex(barrierIdx));
        }
    }
}

// Merge barriers using FIFO order. DMA-{IR-order}
// DMA-0 and DMA-1 are before DMA-2 and DMA-3 in FIFO
/*
    DMA-0 DMA-1      DMA-0 DMA-1
      |    |            \  /
    Bar0  Bar1   =>      Bar
      |    |            /   \
    DMA-2 DMA-3      DMA-2 DMA-3
*/

void mergeBarriers(BarrierInfo& barrierInfo, ArrayRef<BarrierInfo::TaskSet> origWaitBarriersMap) {
    // Perform optimization in tasks blocks matching the distribution of synchronization points.
    for (size_t taskBlockIndex = 0; taskBlockIndex < barrierInfo.getControlGraphBlockCount(); ++taskBlockIndex) {
        // get update barriers range for current block
        auto blockUpdateBarriers =
                barrierInfo.getBarriersForTaskBlock(taskBlockIndex, /* blockStartSyncPoint */ true,
                                                    /* blockEndSyncPoint */ false, /* updateBarriers */ true);

        auto numBarriersInBlock = blockUpdateBarriers.size();

        // Order barriers based on largest producer
        //
        // After already applied optimizations in this pass barrier state could have changed
        // and barriers might not have been ordered based on largest producer value (which corresponds to
        // largest barrier release time).
        // For compile time improvement - early termination of merge barrier logic, we need
        // barriers to be reordered so new vector is prepared that will be used as a base for iterating
        // over all barriers
        SmallVector<std::pair<size_t, std::optional<size_t>>> barIndAndMaxProdVec;
        barIndAndMaxProdVec.reserve(numBarriersInBlock);

        // Store number of barriers which do not have producers which nevertheless are not a candidate
        // for merge barriers logic. Later this value will be used to skip all the barriers
        // with no producers. After sorting barIndAndMaxProdVec they will be placed at the beginning
        size_t numOfBarriersWithNoProducers = 0;

        // For each barrier get the largest producer index
        for (auto barrierInd : blockUpdateBarriers) {
            const auto producers = barrierInfo.getBarrierProducers(barrierInd);
            std::optional<size_t> maxProducer;
            if (producers.empty()) {
                numOfBarriersWithNoProducers++;
            } else {
                maxProducer = *std::max_element(producers.begin(), producers.end());
            }

            barIndAndMaxProdVec.push_back(std::make_pair(barrierInd, maxProducer));
        }

        // Sort the barrier indexes based on largest producer value. If barrier has no producers they will
        // be placed at the beginning
        llvm::sort(barIndAndMaxProdVec.begin(), barIndAndMaxProdVec.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.second == rhs.second) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });

        const auto allProducersAfterConsumers = [](const BarrierInfo::TaskSet& producers,
                                                   const BarrierInfo::TaskSet& consumers) {
            const auto maxConsumer = *std::max_element(consumers.begin(), consumers.end());
            const auto minProducer = *std::min_element(producers.begin(), producers.end());

            return minProducer > maxConsumer;
        };

        // Merge barriers if possible.
        // Skip initial barriers with no producers as they are not candidates for merge
        for (size_t ind = numOfBarriersWithNoProducers; ind < numBarriersInBlock; ++ind) {
            const auto barrierInd = barIndAndMaxProdVec[ind].first;
            auto barrierProducersA = barrierInfo.getBarrierProducers(barrierInd);
            if (barrierProducersA.empty()) {
                continue;
            }
            auto barrierConsumersA = barrierInfo.getBarrierConsumers(barrierInd);
            if (barrierConsumersA.empty()) {
                continue;
            }

            for (auto nextInd = ind + 1; nextInd < numBarriersInBlock; ++nextInd) {
                const auto nextBarrierInd = barIndAndMaxProdVec[nextInd].first;
                const auto barrierProducersB = barrierInfo.getBarrierProducers(nextBarrierInd);
                if (barrierProducersB.empty()) {
                    continue;
                }
                const auto barrierConsumersB = barrierInfo.getBarrierConsumers(nextBarrierInd);
                if (barrierConsumersB.empty()) {
                    continue;
                }

                // If for a given barrier B (nextBarrierInd) all producers are after all consumers of
                // barrier A (barrierInd) then neither this nor any later barrier will be a candidate to merge
                // with barrier A as they do not overlap their lifetime in schedule. Such early return is possible
                // because barriers are processed in order following barrier release time (latest producer)
                if (allProducersAfterConsumers(barrierProducersB, barrierConsumersA)) {
                    break;
                }

                if (!barrierInfo.canBarriersBeMerged(barrierProducersA, barrierConsumersA, barrierProducersB,
                                                     barrierConsumersB, origWaitBarriersMap)) {
                    continue;
                }

                // need to update barriers
                barrierInfo.addProducers(barrierInd, barrierProducersB);
                barrierInfo.addConsumers(barrierInd, barrierConsumersB);
                barrierInfo.resetBarrier(nextBarrierInd);
                llvm::set_union(barrierProducersA, barrierProducersB);
                llvm::set_union(barrierConsumersA, barrierConsumersB);
            }
        }
    }
}

//
//  DMABarrierOptimizationPass
//

class DMABarrierOptimizationPass final : public VPUIP::impl::DMABarrierOptimizationBase<DMABarrierOptimizationPass> {
public:
    explicit DMABarrierOptimizationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    const bool _considerTaskFifoDependency = true;
};

void DMABarrierOptimizationPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);

    barrierInfo.optimizeBarriers(/* checkValidSlotCount */ false, _considerTaskFifoDependency);
    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log);

    if (_considerTaskFifoDependency) {
        barrierInfo.buildTaskQueueTypeMap();
    }

    // get original wait barrier map
    const auto origWaitBarriersMap = barrierInfo.getWaitBarriersMap();

    // DMA operation in the same FIFO do not require a barrier between them
    // optimize dependencies between DMA tasks in the same FIFO
    // (Some of these dependencies may been removed during optimizeBarriers step, E137500)
    removeRedundantDependencies(barrierInfo, _considerTaskFifoDependency, _log);
    removeExplicitDependencies(barrierInfo);
    mergeBarriers(barrierInfo, origWaitBarriersMap);
    removeRedundantDependencies(barrierInfo, _considerTaskFifoDependency, _log);

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(func);
}

}  // namespace

//
// createDMABarrierOptimizationPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDMABarrierOptimizationPass(Logger log) {
    return std::make_unique<DMABarrierOptimizationPass>(log);
}
