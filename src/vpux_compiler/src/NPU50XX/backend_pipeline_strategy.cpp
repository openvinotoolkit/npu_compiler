//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/backend_pipeline_strategy.hpp"

#include "vpux/compiler/NPU50XX/conversion.hpp"

#include "vpux/compiler/compilation_options.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/pipelines/options_mapper.hpp"
#include "vpux/compiler/pipelines/options_setup.hpp"
#include "vpux/utils/IE/config.hpp"

#include "intel_npu/config/options.hpp"

using namespace vpux;

//
// BackendPipelineStrategy50XX::buildELFPipeline
//

void BackendPipelineStrategy50XX::buildELFPipeline(mlir::OpPassManager& pm, const intel_npu::Config& config,
                                                   mlir::TimingScope& rootTiming, Logger log, bool useWlm) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");
    const auto backendCompilationOptions =
            BackendCompilationOptions50XX::createFromString(config.get<intel_npu::BACKEND_COMPILATION_PARAMS>());

    const auto options = parseCompilationModeParams<DefaultHWOptions50XX>(
            config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    VPUX_THROW_UNLESS(options != nullptr, "build ELF pipeline failed to parse COMPILATION_MODE_PARAMS: {0}",
                      config.get<intel_npu::COMPILATION_MODE_PARAMS>());

    if (config.get<intel_npu::TURBO>()) {
        overwriteIfUnset(options->workloadManagementMode, WorkloadManagementMode::FWLM_V1_PAGES);
    }

    setupPWLMParams50XX(*options, getLogLevel(config));
    backendCompilationOptions->npu5PPEBackwardsCompatibilityMode = options->npu5PPEBackwardsCompatibilityMode;
    backendCompilationOptions->enableDumpStatisticsOfWlmOps = options->enableDumpTaskStats;
    backendCompilationOptions->workloadManagementMode = options->workloadManagementMode;
    backendCompilationOptions->workloadManagementEnable = useWlm;
    backendCompilationOptions->workloadManagementBarrierCountThreshold =
            options->workloadManagementBarrierCountThreshold;
    backendCompilationOptions->workloadManagementBarrierProgrammingMode =
            options->workloadManagementBarrierProgrammingMode;
    backendCompilationOptions->workloadManagementDmaFifoType = options->workloadManagementDmaFifoType;
    backendCompilationOptions->modelIdentifier = options->modelIdentifier;

    if (getCompilationMode(config) != config::CompilationMode::ReferenceSW) {
        auto enableProfiling = config.get<intel_npu::PERF_COUNT>();
        backendCompilationOptions->enableDMAProfiling =
                enableProfiling ? options->enableDMAProfiling.getValue() : "false";
        backendCompilationOptions->enableShaveDDRAccessOptimization = options->enableShaveDDRAccessOptimization;
    }

    arch50xx::buildLowerVPUIP2ELFPipeline(pm, *backendCompilationOptions, log.nest());
}
