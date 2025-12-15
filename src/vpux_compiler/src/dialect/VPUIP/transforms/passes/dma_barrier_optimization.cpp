//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/shave.hpp"

#include <llvm/ADT/SetOperations.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_DMABARRIEROPTIMIZATION
#define GEN_PASS_DEF_DMABARRIEROPTIMIZATION
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
namespace {

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
    explicit DMABarrierOptimizationPass(std::optional<WorkloadManagementMode> workloadManagementMode, Logger log)
            : _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    const bool _considerTaskFifoDependency = true;
    std::optional<WorkloadManagementMode> _workloadManagementMode = std::nullopt;
};

void DMABarrierOptimizationPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& barrierInfo = getAnalysis<BarrierInfo>();
    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log, true);

    if (workloadManagementModeOpt.hasValue()) {
        _workloadManagementMode = workloadManagementModeOpt.getValue();
    }

    // Constrain executor types being optimized depending on WLM mode so as to avoid deadlocks and rollback regressions
    // in early WLM modes.
    bool allowOptimizationOfAllQueueTypes =
            _workloadManagementMode.has_value() &&
            _workloadManagementMode.value() > WorkloadManagementMode::PWLM_V1_BARRIER_FIFO;

    mlir::DenseSet<vpux::VPU::ExecutorKind> executors = {VPU::ExecutorKind::DMA_NN};

    if (allowOptimizationOfAllQueueTypes) {
        executors.insert(VPU::ExecutorKind::SHAVE_ACT);
        executors.insert(VPU::ExecutorKind::DPU);
    }

    barrierInfo.optimizeBarriers(/* checkValidSlotCount */ false, _considerTaskFifoDependency, executors);
    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log);

    if (_considerTaskFifoDependency) {
        // First, initialize DMA queues to pre-optimize producers and consumers for the subsequent barrier merging.
        // Experiments show that optimizing on all queues before barrier merge can have negative impact on the amount of
        // merging, schedule parallelism, inference performance and PWLM_V0_LCA mode stability.
        barrierInfo.initializeTaskQueueTypeMap({VPU::ExecutorKind::DMA_NN});
        barrierInfo.buildTaskQueueTypeMap();
    }

    // get original wait barrier map
    const auto origWaitBarriersMap = barrierInfo.getWaitBarriersMap();

    // DMA operation in the same FIFO do not require a barrier between them
    // optimize dependencies between DMA tasks in the same FIFO
    // (Some of these dependencies may been removed during optimizeBarriers step, E137500)
    barrierInfo.removeRedundantBarrierProducersAndConsumers(_considerTaskFifoDependency);
    barrierInfo.removeExplicitDependencies();
    mergeBarriers(barrierInfo, origWaitBarriersMap);
    if (allowOptimizationOfAllQueueTypes) {
        // For platforms that have enabled support for independent shave queues without risking rollback regressions,
        // initialize and optimize dependencies on all queues.
        barrierInfo.clearTaskQueueTypeMap();
        barrierInfo.buildTaskQueueTypeMap();
        barrierInfo.removeRedundantBarrierProducersAndConsumers(_considerTaskFifoDependency);

        barrierInfo.clearTaskQueueTypeMap();
        barrierInfo.initializeTaskQueueTypeMap({VPU::ExecutorKind::DMA_NN, VPU::ExecutorKind::SHAVE_ACT});
        // TODO: include DPU executor (E#168496, E#190467)
        allowOptimizationOfAllQueueTypes = false;
        barrierInfo.buildTaskQueueTypeMap();
        barrierInfo.removeExplicitDependencies();
    } else {
        barrierInfo.removeRedundantBarrierProducersAndConsumers(_considerTaskFifoDependency);
    }
    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log);

    if (VPUIP::supportsPerVariantBarrierConfiguration(func) && config::isFifoPerShaveEngineEnabled(func) &&
        allowOptimizationOfAllQueueTypes) {
        // If each producer/consumer utilizes only a single slot from available pool of barrier slots, and if SHV tasks
        // use their dedicated FIFOs and redundant connections to barriers from unrolled DPU tasks have been optimized
        // out, then the number of barrier producers/consumers cannot be larger than the number of independent task
        // executors.
        VPUX_THROW_WHEN(!barrierInfo.verifyBarriersUsersCount(VPURT::countIndependentTaskExecutors(func)),
                        "Encountered unexpected number of barrier users.");
    }
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(func);
}

}  // namespace

//
// createDMABarrierOptimizationPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDMABarrierOptimizationPass(
        std::optional<WorkloadManagementMode> workloadManagementMode, Logger log) {
    return std::make_unique<DMABarrierOptimizationPass>(workloadManagementMode, log);
}
