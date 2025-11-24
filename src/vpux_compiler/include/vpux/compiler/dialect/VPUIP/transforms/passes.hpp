//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/VPU/utils/dry_run_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/pipelines_options.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

#include <functional>
#include <memory>
#include <optional>

namespace vpux::VPU {
enum class MemoryKind : uint64_t;
}

namespace vpux {
namespace VPUIP {

//
// Passes
//

using MemKindCreateFunc = std::function<std::optional<VPU::MemoryKind>(StringRef)>;
using ConditionFunc = std::function<bool(mlir::Operation*)>;

template <typename T>
bool isOp(mlir::Operation* op) {
    return mlir::isa<T>(op);
}

ConditionFunc makeStubCondition();

std::unique_ptr<mlir::Pass> createLegalizeScheduleForPartialWlmFetchDmasPass(
        const int virtualBarrierThreshold = VIRTUAL_BARRIER_THRESHOLD_WLM, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseSegmentedDmaPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitDMAToBalanceLoadPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDetectDMASplitCandidatePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddStartBarrierPass(
        std::optional<WorkloadManagementMode> workloadManagementMode = std::nullopt, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createComputeTaskStrippingPass(
        Logger log = Logger::global(), VPU::DPUDryRunMode dryRunStripTarget = VPU::DPUDryRunMode::NONE,
        bool shaveDryRun = false);
std::unique_ptr<mlir::Pass> createDMAOutOfOrderOptimizationPass(
        std::optional<WorkloadManagementMode> workloadManagementMode = std::nullopt, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeConvertDMAOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCompressSpillDmaPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConstantDpuProfHwpBasePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMATaskProfilingHwDdrPass(const std::string& enableDMAProfiling = "true",
                                                            Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createComputeHaloRegionForDPUTaskOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddSwKernelCacheHandlingOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertWeightsTableOp2ConstPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUpdateSwKernelParamsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSyncShvDpuPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDumpStatisticsOfTaskOpsPass(Logger log = Logger::global(), bool forceLogging = true);
std::unique_ptr<mlir::Pass> createUnrollDistributedOpsPass(Logger log = Logger::global(),
                                                           std::optional<bool> enableSegmentedDmaFusion = std::nullopt);
std::unique_ptr<mlir::Pass> createUnrollSwKernelPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMABarrierOptimizationPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createFuseConstantsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveDMAWithSwizzlingPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createMovePureViewOpBeforeCopyPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveSubViewBeforeSparseBufferPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCopyOpTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSetMemorySpacePass(MemKindCreateFunc memKindCb,
                                                     bool setMemorySpaceForFunctionBoundaries = true,
                                                     Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertEltwiseToInPlacePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSprLUTToConstPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPalletLUTToConstPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDynamicReshapeToInPlacePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInsertCopyForEltwiseInPlaceInputPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLinearizationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBreakDataFlowPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPatchWeightsTablePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPatchPopulateWeightTableWithShavePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMATaskProfilingAfterBarrierSchedPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCaptureWorkpointPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDPUProfilingPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createM2IProfilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createGroupProfilingBuffersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createActShaveProfilingPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapWithPermuteAsNNDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeTileOpAsNNDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeExpandSubviewPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertExpandPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwizzlingPass(const bool enableWeightSwizzling = true,
                                                const bool enableActivationSwizzling = true,
                                                Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOperationStubbingPass(ConditionFunc condition = makeStubCondition(),
                                                        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvWeightsCompressionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUngroupSparseBuffersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateSparsityCompressionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFlattenSparseWeightsTypesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createComputeSESizesPass(std::optional<bool> onlyInputsConcatOverC = std::nullopt,
                                                     Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createComputeSEBasePtrsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSETablesToConstantsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustInputDataForExplicitSETablePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createTileActShaveKernelTaskPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSetZeroOffsetWeightsTablePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSegmentHalosPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustSpillSizePass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createLegalizeRepeatingFuncCallsPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertVPUIPCopyToSWCopyPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddCopyBetweenSWKernelsAndNetworkIOPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDispatchedInlinerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddSwKernelInstructionPrefetchPass(Logger log = Logger::global());

//
// Asynchronous Scheduling pipeline
//

void buildAsyncSchedulingPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createAsyncRegionsOutliningPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAsyncRegionsOutliningPass(size_t asyncRegionOutliningMinOpsInBlock,
                                                            Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapIntoAsyncRegionsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveWaitResultToAsyncBlockArgsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCalculateAsyncRegionCycleCostPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createGroupAsyncExecuteOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveViewOpsIntoAsyncRegionsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeAsyncDepsPass(Logger log = Logger::global());

//
// Hardware Adaptation pipeline
//

void buildHardwareAdaptationPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertTransferOpsToDMAsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAllocationsToDeclarationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertFuncArgsToDeclarationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertViewOpsToDeclarationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLinearizeCallOpsPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAsyncOpsToTasksPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCompressWeightsBTCPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createNNDMATilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUngroupBoundedBuffersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUngroupBoundedBuffersAsFuncArgsPass(Logger log = Logger::global());

//
// Memory allocation pipeline
//

std::unique_ptr<mlir::Pass> createFeasibleAllocationPass(
        MemKindCreateFunc memKindCb, MemKindCreateFunc secondLvlMemKindCb = nullptr,
        const bool linearizeSchedule = false, const bool enablePipelining = true, const bool enablePrefetching = true,
        const bool optimizeFragmentation = true, const bool optimizeDynamicSpilling = true,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createQueryArgsAllocationAnalysisPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createStaticAllocationPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBatchMatMulToMatMulPass(Logger log = Logger::global());

//
// DMA Unrolling Pipeline
//

void buildDMAUnrollingPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollDMAAnalysisPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollDepthToSpaceDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollSpaceToDepthDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollUpsamplingDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollPermuteDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollExpandDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollPerAxisTileDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInvalidateUnrollDMAAnalysisPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollGatherDMAPass(Logger log = Logger::global());

//
// ShaveCodeGen Pipeline
//

void buildShaveCodeGenPipeline(mlir::OpPassManager& pm);

//
// Optimized Copying pipeline
//

void buildOptimizeCopiesPipeline(mlir::OpPassManager& pm, const OptimizeCopiesOptionsBase& options,
                                 Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeCopiesPass(
        const WorkloadManagementMode workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUniquifyWeightsTableCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeConcatViewCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseDDRCopiesIntoConcats(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeParallelCopiesPass(bool enableOptimizeConstCopy = true,
                                                             Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSubviewCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseLastCopyPass(Logger log = Logger::global());

//
// DefaultHWOptions(for all devices)
//

struct DefaultHWOptionsDialectBase : public virtual vpux::DefaultHWOptionsBase {
    BoolOption enableM2IProfiling{*this, "m2i-profiling", llvm::cl::desc("Enable M2I task profiling"),
                                  llvm::cl::init(true)};

    BoolOption enableOptimizeCopies{*this, "optimize-copies", llvm::cl::desc("Enable optimize-copies pass"),
                                    llvm::cl::init(true)};

    BoolOption enableOptimizeConstCopies{*this, "optimize-const-copies", llvm::cl::desc("Enable optimize-const-copies"),
                                         llvm::cl::init(true)};

    BoolOption enableGroupAsyncExecuteOps{*this, "group-async-execute-ops",
                                          llvm::cl::desc("Enable group-async-execute-ops pass"), llvm::cl::init(false)};

    BoolOption enableConstantFusion{*this, "constant-fusion", llvm::cl::desc("Enable constant fusion"),
                                    llvm::cl::init(true)};

    BoolOption enableOpsAsDMA{*this, "enable-ops-as-dma",
                              llvm::cl::desc("Force using DMA transformations instead of SW ops"),
                              llvm::cl::init(true)};

    BoolOption optimizeFragmentation{*this, "optimize-fragmentation",
                                     ::llvm::cl::desc("Enables compiler to optimize CMX fragmentation"),
                                     ::llvm::cl::init(true)};

    BoolOption optimizeDynamicSpilling{*this, "optimize-dynamic-spilling",
                                       ::llvm::cl::desc("Enables compiler to optimize dynamic spilling DMAs"),
                                       ::llvm::cl::init(true)};

    BoolOption linearizeSchedule{*this, "linearize-schedule", llvm::cl::desc("Linearize tasks on all engines"),
                                 llvm::cl::init(false)};

    BoolOption enableShaveKernelTiling{*this, "enable-shave-kernel-tiling",
                                       ::llvm::cl::desc("Enable shave kernel tiling"), ::llvm::cl::init(true)};

    BoolOption setMemorySpaceForFunctionBoundaries{
            *this, "set-memory-space-for-function-boundaries",
            ::llvm::cl::desc("Tells --set-memory-space to update function boundaries with the specified memory space"),
            ::llvm::cl::init(true)};
};

//
// Registration
//

void registerVPUIPPipelines();
void registerPasses();

}  // namespace VPUIP
}  // namespace vpux
