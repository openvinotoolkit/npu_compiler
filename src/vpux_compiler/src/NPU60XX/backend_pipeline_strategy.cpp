//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/backend_pipeline_strategy.hpp"

#include "vpux/compiler/NPU60XX/conversion.hpp"

#include "vpux/compiler/compilation_options.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/pipelines/options_mapper.hpp"
#include "vpux/utils/ov/options.hpp"

using namespace vpux;

//
// BackendPipelineStrategy60XX::buildELFPipeline
//

void BackendPipelineStrategy60XX::buildELFPipeline(mlir::OpPassManager& pm, const vpux::OV::Config& config,
                                                   mlir::TimingScope& rootTiming, Logger log) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");
    const auto backendCompilationOptions =
            BackendCompilationOptions60XX::createFromString(config.get<vpux::OV::BACKEND_COMPILATION_PARAMS>());
    const auto options = parseCompilationModeParams<DefaultHWOptions60XX>(
            config.get<vpux::OV::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    VPUX_THROW_WHEN(options == nullptr, "build ELF pipeline failed to parse COMPILATION_MODE_PARAMS: {0}",
                    config.get<vpux::OV::COMPILATION_MODE_PARAMS>());
    setupPWLMParams60XX(*options);
    backendCompilationOptions->enableDumpStatisticsOfWlmOps = options->enableDumpTaskStats;
    backendCompilationOptions->workloadManagementMode = options->workloadManagementMode;
    backendCompilationOptions->workloadManagementBarrierProgrammingMode =
            options->workloadManagementBarrierProgrammingMode;

    backendCompilationOptions->modelIdentifier = options->modelIdentifier;
    backendCompilationOptions->allocateDDRStackFrames = options->allocateDDRStackFrames;

    if (getCompilationMode(config) != config::CompilationMode::ReferenceSW) {
        auto enableProfiling = config.get<vpux::OV::PERF_COUNT>();
        backendCompilationOptions->enableDMAProfiling =
                enableProfiling ? options->enableDMAProfiling.getValue() : "false";
        backendCompilationOptions->enableShaveDDRAccessOptimization = options->enableShaveDDRAccessOptimization;
    }

    arch60xx::buildLowerVPUIP2ELFPipeline(pm, *backendCompilationOptions, log.nest());
}
