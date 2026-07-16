//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

namespace {

VPU::ActivationSparsityProfile getActSparsityProfile(const StrOption& actProfile) {
    const auto actProfileStr = actProfile.getValue();
    const auto parsed = VPU::symbolizeActivationSparsityProfile(actProfileStr);
    VPUX_THROW_UNLESS(parsed.has_value(), "Unsupported activation sparsity profile '{0}'", actProfileStr);
    return parsed.value();
}

template <VPU::ActivationSparsityProfile PROFILE>
std::optional<VPU::ActivationSparsityProfile> getSparsityProfile(StringRef) {
    return PROFILE;
}

auto getSparsityProfileCallback(VPU::ActivationSparsityProfile actSparsityProfile) {
    switch (actSparsityProfile) {
    case VPU::ActivationSparsityProfile::S0:
        return getSparsityProfile<VPU::ActivationSparsityProfile::S0>;
    case VPU::ActivationSparsityProfile::S1:
        return getSparsityProfile<VPU::ActivationSparsityProfile::S1>;
    default:
        VPUX_THROW("Unknown ActSparsityProfile");
    }
}

VPU::WeightsSparsityHeuristic getWeightsSparsityHeuristic(const StrOption& weightsSparsityHeuristic) {
    const auto weightsSparsityHeuristicStr = weightsSparsityHeuristic.getValue();
    const auto parsed = VPU::symbolizeWeightsSparsityHeuristic(weightsSparsityHeuristicStr);
    VPUX_THROW_UNLESS(parsed.has_value(), "Unsupported weights sparsity heuristic '{0}'", weightsSparsityHeuristicStr);
    return parsed.value();
}

std::optional<double> getWeightsSparsityThreshold(const DoubleOption& weightsSparsityThreshold) {
    const auto threshold = weightsSparsityThreshold.getValue();
    if (threshold >= 0.0) {
        return threshold;
    }
    return std::nullopt;
}

}  // namespace

//
// buildInitCompilerPipeline
//

void vpux::VPU::buildInitCompilerPipeline(mlir::OpPassManager& pm, const VPU::InitCompilerOptions& options,
                                          Logger log) {
    log.info("InitCompilerOptions:\n platform = {0}\n DPU groups = {1}\n DMA ports = {2}\n"
             " compilation mode = {3}\n adaptive stripping = {4}\n aggressive "
             "QDQ = {5}\n weights dynamic dequantization {6}\n",
             options.platform, options.numberOfDPUGroups, options.numberOfDMAPorts, options.compilationMode,
             options.enableAdaptiveStripping, options.enableQDQOptimizationAggressive,
             options.enableWeightsDynamicDequantization);

    pm.addPass(VPU::createInitResourcesPass(options, log));
#ifdef VPUX_DEVELOPER_BUILD
    pm.addPass(VPU::createSetupPassToolsPass(options, log));
#endif  // VPUX_DEVELOPER_BUILD
    pm.addPass(VPU::createSetupPipelineOptionsPass(options, log));
    pm.addPass(VPU::createSetTargetIndependentPassOptionsPass(options, log));

#ifdef VPUX_DEVELOPER_BUILD
    pm.addPass(VPU::createSetupCustomConstraintsPass(options, log));
#endif  // VPUX_DEVELOPER_BUILD
    pm.addPass(VPU::createSetupNpuConstraintPass(options, log));
    pm.addPass(VPU::createSetupTilingConstraintPass(options, log));
}

//
// buildActivationSparsityPipeline
//

void vpux::VPU::buildActivationSparsityPipeline(mlir::OpPassManager& pm, const VPU::ActivationSparsityOptions& options,
                                                Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    const auto actSparsityProfile = getActSparsityProfile(options.actSparsityProfile);
    const auto profileCallback = getSparsityProfileCallback(actSparsityProfile);

    pm.addPass(VPU::createWrapOpsInSparsifyDesparsifyPairsPass(
            VPU::getActSparsityMode(options.enableActivationSparsity), actSparsityProfile, log));

    if (actSparsityProfile == VPU::ActivationSparsityProfile::S1) {
        pm.addPass(VPU::createFuseSparsityOpsPass(/*fuseSparsify=*/false, log));
    }

    pm.addPass(VPU::createOptimizeSparsifyDesparsifyPairsPass(options, log));
    pm.addPass(VPU::createFuseSparsityOpsPass(/*fuseSparsify=*/true, log));
    pm.addPass(VPU::createOptimizeSparsityOpsPass(profileCallback, log));
    pm.addPass(VPU::createAddSparsityMapToSparseActivationsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// buildWeightsSparsityPipeline
//

void vpux::VPU::buildWeightsSparsityPipeline(mlir::OpPassManager& pm, const VPU::WeightsSparsityOptions& options,
                                             Logger log) {
    const auto weightsSparsityHeuristic = getWeightsSparsityHeuristic(options.weightsSparsityHeuristic);
    const auto weightsSparsityThreshold = getWeightsSparsityThreshold(options.weightsSparsityThreshold);
    pm.addPass(VPU::createSparsifyWeightsPass(
            weightsSparsityHeuristic, weightsSparsityThreshold, options.weightsSparsityLargeConstThreshold,
            options.weightsSparsityComputeOpThreshold, options.enableWeightSwizzling, log));
    pm.addPass(VPU::createRecomputeSparsityPtrsPass(log));
}

void VPU::registerVPUPipelines() {
    mlir::PassPipelineRegistration<VPU::InitCompilerOptions>(
            "init-compiler", "Init compiler resorces and options",
            [](mlir::OpPassManager& pm, const VPU::InitCompilerOptions& options) {
                VPU::buildInitCompilerPipeline(pm, options);
            });
    mlir::PassPipelineRegistration<VPU::ActivationSparsityOptions>(
            "enable-act-sparsity", "Enable activation sparsity",
            [](mlir::OpPassManager& pm, const VPU::ActivationSparsityOptions& options) {
                VPU::buildActivationSparsityPipeline(pm, options);
            });
    mlir::PassPipelineRegistration<VPU::WeightsSparsityOptions>(
            "enable-weights-sparsity", "Enable weights sparsity",
            [](mlir::OpPassManager& pm, const VPU::WeightsSparsityOptions& options) {
                VPU::buildWeightsSparsityPipeline(pm, options);
            });
    mlir::PassPipelineRegistration<VPU::TilingOptions>("tiling", "Apply tiling",
                                                       [](mlir::OpPassManager& pm, const VPU::TilingOptions& options) {
                                                           VPU::buildTilingPipeline(pm, options);
                                                       });

    mlir::PassPipelineRegistration<VPU::TilingOptions>("vertical-fusion", "Apply VF Pipeline",
                                                       [](mlir::OpPassManager& pm, const VPU::TilingOptions& options) {
                                                           VPU::buildVFPipeline(pm, options);
                                                       });

    mlir::PassPipelineRegistration<VPU::ScfComputeOpsOutliningOptions>(
            "scf-ops-outlining", "SCF compute ops outlining transformations",
            [](mlir::OpPassManager& pm, const VPU::ScfComputeOpsOutliningOptions& options) {
                VPU::buildScfComputeOpsOutliningPipeline(pm, options.loopUnrollFactor, options.enableProfiling,
                                                         options.enableCascadedUnrolling, options.autoUnrollingMode,
                                                         options.enableWeightsExtraction, Logger::global());
            });
}

void vpux::VPU::buildTilingPipeline(mlir::OpPassManager& pm, const VPU::TilingOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(VPU::createFlashSDPATilingPass(/*enablePipelining=*/true, log));

    pm.addPass(VPU::createTilingStrategyAssignmentPass(
            options.enablePrefetchTiling, options.enableVPUNNCostForTiling, options.enableShaveDDRAccessOptimization,
            options.enableDynamicDimAlignment, options.enableTilingFullSearchSpace, log));
    if (options.enablePrintStatistics) {
        pm.addPass(VPU::createPrintNNCacheStatisticsPass(log, "tiling-strategy-assignment"));
    }

    if (options.enableLegacyConvSplitOverIC) {
        pm.addPass(VPU::createConvolutionSplitOverInputChannelPass(log));
    }

    // We call this as part of VF Pipeline, no need to call it here in such case
    if (!options.enableVerticalFusion) {
        pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyDefaultFileLocation,
                                                      options.readStrategyFromJson, readStrategyDefaultFileLocation,
                                                      options.dumpStrategyToLog, false, log));
    }
    pm.addPass(VPU::createEfficientIROrderPass(options.enableReorderConcatBranches, log));
    if (options.enableVerticalFusion) {
        if (!options.enableSCFTiling) {
            VPU::buildVFPipeline(pm, options, log);
        } else {
            pm.addPass(VPU::createSCFVerticalFusionPass(options.vfMergeConfiguration, options.enableDynamicDimAlignment,
                                                        log));
            // cleaning up after SCFVerticalFusionPass
            pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
            pm.addPass(mlir::createCSEPass());
            pm.addPass(mlir::createCanonicalizerPass(grc));
        }
    }

    if (!options.enableSCFTiling && options.enableOutputPipelining && !options.enableTilingFullSearchSpace) {
        pm.addPass(VPU::createOutputPipelineTilingPass(options.enablePrefetchTiling, log));
        if (options.enablePrintStatistics) {
            pm.addPass(VPU::createPrintNNCacheStatisticsPass(log, "output-pipeline-tiling"));
        }
        // manual strategy debug configuration
        pm.addPass(VPU::createManualStrategyUtilsPass(
                options.writeStrategyToJson, writeStrategyDefaultFileLocation, options.readStrategyFromJson,
                readStrategyDefaultFileLocation, options.dumpStrategyToLog,
                /*updateStrategyForOutputPipelining*/ true, /*contextId*/ "outputPipelining", log));
    }

    pm.addPass(VPU::createApplyTilingPass(options.enableSCFTiling, options.enableDynamicDimAlignment, log));
    if (options.enableSCFTiling) {
        // cleaning up after ApplyTilingPass
        pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
        pm.addPass(mlir::createCSEPass());
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(VPU::createCorrectStorageElementTableSeSizeForSEPDWConvPass(log));
}

//
// Scf Compute Ops outlining Pipeline
//

void vpux::VPU::buildScfComputeOpsOutliningPipeline(
        mlir::OpPassManager& pm, const vpux::StrOption& loopUnrollFactor, bool enableProfiling,
        const vpux::BoolOption& enableCascadedUnrolling,
        const mlir::detail::PassOptions::Option<AutoUnrollingMode>& autoUnrollingMode,
        const vpux::BoolOption& enableWeightsExtraction, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(VPU::createSCFFuseLastViewLikeOpPass(log));
    pm.addPass(VPU::createRestorePadAttrAfterSCFTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createScfComputeOpsOutliningPass(log));
    pm.addPass(VPU::createAdjustBlockSizeForScfTilingPass(log));
    if (enableWeightsExtraction) {
        pm.addPass(VPU::createExtractWeightsPass(log));
        pm.addPass(VPU::createConcatHostConstsPass(log));
    }
    pm.addPass(VPU::createConvertDynamicToStaticKernelsPass(log));

    const bool hasManualFactor = loopUnrollFactor.hasValue() && !loopUnrollFactor.getValue().empty();
    const auto unrollingModeValue =
            autoUnrollingMode.hasValue() ? autoUnrollingMode.getValue() : AutoUnrollingMode::DISABLED;
    if (hasManualFactor || unrollingModeValue != AutoUnrollingMode::DISABLED) {
        pm.addPass(VPU::createUnrollSCFLoopPass(hasManualFactor ? loopUnrollFactor.getValue() : "",
                                                enableCascadedUnrolling.getValue(), unrollingModeValue, log));
    }
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    pm.addPass(VPU::createFinalizeComputeFunctionBoundariesPass(log));
    pm.addPass(Core::createPackNestedModulesPass(log, Core::NestingMode::Default, enableProfiling));

    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(mlir::createCSEPass());
}

//
// Strategy Pipeline
//

void vpux::VPU::buildVFPipeline(mlir::OpPassManager& pm, const VPU::TilingOptions& options, Logger log) {
    pm.addPass(VPU::createWrapVerticalFusionRegionPass(options.workloadManagementMode, log));
    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyDefaultFileLocation,
                                                  options.readStrategyFromJson, readStrategyDefaultFileLocation,
                                                  options.dumpStrategyToLog, false, log));

    pm.addPass(VPU::createMoveViewOpsToVerticalFusionPass(options.workloadManagementMode, log));
    if (options.workloadManagementMode > WorkloadManagementMode::PWLM_V0_1_PAGES) {
        pm.addPass(VPU::createPatternBasedVFPass(options.workloadManagementMode, log));
    }
    pm.addPass(VPU::createMergeVfSubgraphsPass(options.enableVerticalFusionPipelining, options.enablePrefetchTiling,
                                               options.workloadManagementMode, log));
    if (options.enablePrintStatistics) {
        pm.addPass(VPU::createPrintNNCacheStatisticsPass(log, "merge-vertical-fusion-subgraphs"));
    }
    pm.addPass(VPU::createEfficientIROrderPass(options.enableReorderConcatBranches, log));
    pm.addPass(VPU::createUnrollUnusedVerticalFusionRegionPass(log));
    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyDefaultFileLocation,
                                                  options.readStrategyFromJson, readStrategyDefaultFileLocation,
                                                  options.dumpStrategyToLog, false, log));
    // TODO: E#140041 enable profiling with function outlining
    const auto grc = getDefaultGreedyRewriteConfig();
    if (options.enableVerticalFusionOutlining && canOutlineFromProfilingPerspective(options)) {
        pm.addPass(VPU::createVerticalFusionOutliningPass(options, log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }
    pm.addPass(VPU::createVfTilingPass(options.enableVerticalFusionPipelining, options.enableVFScheduleTrace,
                                       options.workloadManagementMode, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::VPU::buildSMPipeline(mlir::OpPassManager& pm, const vpux::MCAndTilingOptionsBase& options, Logger log) {
    // TO DO - SM Assignment Optimization Pass
    // Keep enableSMpipleline Option - false till SM pipeline is built
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(VPU::createStrategyManagerImplPass(options.enablePrefetching, log));
    pm.addPass(VPU::createEfficientIROrderPass(options.enableReorderConcatBranches, log));
    if (options.enableVerticalFusion) {
        VPU::buildVFPipeline(pm, VPU::TilingOptions(options), log);
    }

    // We have already dumped the strategies in above pipeline
    if (!options.enableVerticalFusion) {
        pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyDefaultFileLocation,
                                                      options.readStrategyFromJson, readStrategyDefaultFileLocation,
                                                      options.dumpStrategyToLog, false, log));
    }
    pm.addPass(VPU::createApplyTilingPass(options.enableSCFTiling, options.enableDynamicDimAlignment, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(VPU::createCorrectStorageElementTableSeSizeForSEPDWConvPass(log));
    pm.addPass(VPU::createMakeOpsWithDistributedTensorPass(options.enableExplicitDistributionInfoAttr, log));
    pm.addPass(VPU::createMakeDistributedCopiesPass(log));
    pm.addPass(VPU::createAdjustDistributedTensorAroundOpsPass(log));

    // Ensure the nce op size requirements are met
    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(/*enableOutputEnsurance=*/true,
                                                           /*enableDequantWeightEnsuranceBeforeStrategy=*/false,
                                                           /*skipConvOC=*/SkipOCMode::SKIP_NONE,
                                                           /*skipEltwiseOC=*/SkipOCMode::SKIP_NONE,
                                                           /*enableSplitChannelForDynamicDequantize=*/false, log));
}
