//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

namespace {

VPU::ActivationSparsityProfile getActSparsityProfile(const StrOption& actProfile) {
    VPUX_THROW_UNLESS(actProfile.hasValue(),
                      "Activation sparsity profile is not provided. Please try 'act-sparsity-profile=S1'");
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
    VPUX_THROW_UNLESS(weightsSparsityHeuristic.hasValue(),
                      "Weights sparsity heuristic is not provided. Please try 'weights-sparsity-heuristic=RATIO'");
    const auto weightsSparsityHeuristicStr = weightsSparsityHeuristic.getValue();
    const auto parsed = VPU::symbolizeWeightsSparsityHeuristic(weightsSparsityHeuristicStr);
    VPUX_THROW_UNLESS(parsed.has_value(), "Unsupported weights sparsity heuristic '{0}'", weightsSparsityHeuristicStr);
    return parsed.value();
}

std::optional<double> getWeightsSparsityThreshold(const DoubleOption& weightsSparsityThreshold) {
    if (weightsSparsityThreshold.hasValue()) {
        const auto threshold = weightsSparsityThreshold.getValue();
        if (threshold >= 0.0) {
            return threshold;
        }
    }
    return std::nullopt;
}

}  // namespace

//
// buildInitCompilerPipeline
//

void vpux::VPU::buildInitCompilerPipeline(mlir::OpPassManager& pm, const VPU::InitCompilerOptions& options,
                                          Logger log) {
    log.info("InitCompilerOptions:\n arch = {0}\n DPU groups = {1}\n DMA ports = {2}\n"
             " compilation mode = {3}\n WLM rollback = {4}\n PPE version = {5}",
             options.arch, options.numberOfDPUGroups, options.numberOfDMAPorts, options.compilationMode,
             options.wlmRollback, options.ppeVersion);

    pm.addPass(VPU::createInitResourcesPass(options, log));
    pm.addPass(VPU::createSetupPipelineOptionsPass(options, log));
    pm.addPass(VPU::createSetupMaxKernelSizePass(options, log));
    pm.addPass(VPU::createSetupNpuConstraintPass(options, log));
    pm.addPass(VPU::createSetupChannelsAutoPaddingPass(options, log));
    pm.addPass(VPU::createSetupIsReduceSupportedPass(options, log));
    pm.addPass(VPU::createSetupEnableFP16CompressedConvPass(options, log));
    pm.addPass(VPU::createSetupEnableSEPtrsOperationsPass(options, log));
    pm.addPass(VPU::createSetupEnableAdaptiveStrippingPass(options, log));
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
}

void vpux::VPU::buildTilingPipeline(mlir::OpPassManager& pm, const VPU::TilingOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(VPU::createTilingStrategyAssignmentPass(options.enablePrefetchTiling, options.enableVPUNNCostForTiling,
                                                       options.enableShaveDDRAccessOptimization, log));

    // We call this as part of VF Pipeline, no need to call it here in such case
    if (!options.enableVerticalFusion) {
        pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                      options.readStrategyFromJson, readStrategyFileLocation,
                                                      /*enableMCSideLoadDump*/ false, options.modelHash, log));
    }
    pm.addPass(VPU::createEfficientIROrderPass(log));
    if (options.enableVerticalFusion) {
        VPU::buildVFPipeline(pm, options, log);
    }

    if (options.enableOutputPipelining) {
        pm.addPass(VPU::createOutputPipelineTilingPass(options.enablePrefetchTiling, log));
        pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                      options.readStrategyFromJson, readStrategyFileLocation,
                                                      /*updateStrategyForOutputPipelining*/ true,
                                                      /*enableMCSideLoadDump*/ false, options.modelHash, log));
    }
    // manual strategy debug configuration
    pm.addPass(VPU::createApplyTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// Strategy Pipeline
//

void vpux::VPU::buildVFPipeline(mlir::OpPassManager& pm, const VPU::TilingOptions& options, Logger log) {
    pm.addPass(VPU::createWrapVerticalFusionRegionPass(log));
    pm.addPass(VPU::createMoveViewOpsToVerticalFusionPass(log));
    pm.addPass(
            VPU::createMergeVfSubgraphsPass(options.enableVerticalFusionPipelining, options.enablePrefetchTiling, log));
    pm.addPass(VPU::createEfficientIROrderPass(log));
    pm.addPass(VPU::createUnrollUnusedVerticalFusionRegionPass(log));
    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                  options.readStrategyFromJson, readStrategyFileLocation,
                                                  /*enableMCSideLoadDump*/ false, options.modelHash, log));
    // TODO: E#140041 enable profiling with function outlining
    if (options.enableVerticalFusionOutlining && !options.enableProfiling) {
        pm.addPass(VPU::createVerticalFusionOutliningPass(options, log));
        const auto grc = getDefaultGreedyRewriteConfig();
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }
    pm.addPass(VPU::createVfTilingPass(options.enableVerticalFusionPipelining, log));
}

void vpux::VPU::buildSMPipeline(mlir::OpPassManager& pm, const vpux::MCAndTilingOptionsBase& options, Logger log) {
    // TO DO - SM Assignment Optimization Pass
    // Keep enableSMpipleline Option - false till SM pipeline is built

    pm.addPass(VPU::createStrategyManagerImplPass(options.enablePrefetching, log));
    pm.addPass(VPU::createEfficientIROrderPass(log));
    if (options.enableVerticalFusion) {
        VPU::buildVFPipeline(pm, VPU::TilingOptions(options), log);
    }

    // We have already dumped the strategies in above pipeline
    if (!options.enableVerticalFusion) {
        pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                      options.readStrategyFromJson, readStrategyFileLocation,
                                                      /*enableMCSideLoadDump*/ false, options.modelHash, log));
    }
    pm.addPass(VPU::createApplyTilingPass(log));
    pm.addPass(VPU::createMakeOpsWithDistributedTensorPass(options.enableExplicitDistributionInfoAttr, log));
    pm.addPass(VPU::createMakeDistributedCopiesPass(log));
    pm.addPass(VPU::createAdjustDistributedTensorAroundOpsPass(log));

    // Ensure the nce op size requirements are met
    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(true, log));
}
