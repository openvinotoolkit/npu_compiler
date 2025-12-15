//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <memory>
#include <optional>

namespace vpux {
namespace VPURT {

//
// Barrier Legalization Pipeline
//

void buildBarrierLegalizationPipeline(
        mlir::OpPassManager& pm, std::optional<int> virtualBarrierThresholdForWlm = std::nullopt,
        std::optional<WorkloadManagementMode> workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA,
        const bool unevenVariantSplitFlag = false, Logger log = Logger::global());

//
// Passes
//

std::unique_ptr<mlir::Pass> createCheckWlmPageSplitConstraintsPass(
        WorkloadManagementMode workloadManagementMode = WorkloadManagementMode::FWLM_V1_PAGES,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFindWlmEnqueueDmasBarrierPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createOptimizeBarriersSlotsUsagePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWlmInsertDummyBarriersInPagesPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createWlmInsertDummyDmasInPagesPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createWlmLegalizePagesForBarrierDmasPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWlmLegalizeSplitGraphToPagesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWlmSplitGraphToPagesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOrderBarriersForWlmPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFindWlmEnqueueBarrierPass(
        WorkloadManagementMode workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA,
        bool disableDmaSwFifo = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSyncTasksPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInsertSyncTasksPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddFinalBarrierPass(
        std::optional<WorkloadManagementMode> workloadManagementMode = std::nullopt, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitControlGraphPass(
        const int controlGraphSplitBlockSize = CONTROL_GRAPH_SPLIT_BLOCK_SIZE, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSimplifySchedulePass(
        const bool reduceParallelControlFlowsFlag = true,
        std::optional<WorkloadManagementMode> workloadManagementMode = std::nullopt, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitExceedingBarrierSlotCountPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSatisfyOneWaitBarrierPerTaskPass(
        std::optional<int> virtualBarrierThresholdForWlm = std::nullopt, const bool unevenVariantSplitFlag = false,
        std::optional<WorkloadManagementMode> workloadManagementMode = std::nullopt, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createReduceExceedingActiveCountBarriersPass(
        std::optional<int> virtualBarrierThresholdForWlm = std::nullopt,
        std::optional<WorkloadManagementMode> workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA,
        const bool unevenVariantSplitFlag = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAssignPhysicalBarriersPass(
        const bool barrierColorBinFlag = false,
        std::optional<WorkloadManagementMode> workloadManagementMode = std::nullopt,
        std::optional<int> virtualBarrierThresholdForWlm = std::nullopt, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBarrierSimulationPass(const bool wlmRollbackFlag = false,
                                                        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createIntermediateBufferOutputPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInferenceExecutionAnalysisPass(
        std::string compileSchedTraceFileName = "compileTimeScheduleTrace.json", bool dumpToJson = false,
        bool enableActivityFactor = true, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInsertBarrierToMarkTheEndOfDescriptorGroupPass(
        std::optional<size_t> virtualBarrierThresholdForWlm = VIRTUAL_BARRIER_THRESHOLD_WLM,
        std::optional<WorkloadManagementMode> workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA,
        Logger log = Logger::global());

//
// Registration
//

void registerVPURTPipelines();
void registerPasses();

}  // namespace VPURT
}  // namespace vpux
