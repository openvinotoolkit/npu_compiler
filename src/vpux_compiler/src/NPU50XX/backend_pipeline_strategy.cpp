//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/backend_pipeline_strategy.hpp"

#include "vpux/compiler/NPU50XX/conversion.hpp"

#include "vpux/compiler/compilation_options.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/pipelines/options_mapper.hpp"
#include "vpux/compiler/pipelines/options_setup.hpp"
#include "vpux/utils/ov/config.hpp"

using namespace vpux;

//
// BackendPipelineStrategy50XX::buildELFPipeline
//

void BackendPipelineStrategy50XX::buildELFPipeline(mlir::OpPassManager& pm, const vpux::OV::Config& config,
                                                   mlir::TimingScope& rootTiming, Logger log) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");
    const auto backendCompilationOptions =
            BackendCompilationOptions50XX::createFromString(config.get<vpux::OV::BACKEND_COMPILATION_PARAMS>());

    const auto options = parseCompilationModeParams<DefaultHWOptions50XX>(
            config.get<vpux::OV::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    VPUX_THROW_UNLESS(options != nullptr, "build ELF pipeline failed to parse COMPILATION_MODE_PARAMS: {0}",
                      config.get<vpux::OV::COMPILATION_MODE_PARAMS>());

    if (config.get<vpux::OV::TURBO>()) {
        overwriteIfUnset(options->workloadManagementMode, WorkloadManagementMode::FWLM_V1_PAGES);
    }

    setupPWLMParams50XX(*options, config.get<vpux::OV::LOG_LEVEL>());
    backendCompilationOptions->npu5PPEBackwardsCompatibilityMode = options->npu5PPEBackwardsCompatibilityMode;
    backendCompilationOptions->enableDumpStatisticsOfWlmOps = options->enableDumpTaskStats;
    backendCompilationOptions->workloadManagementMode = options->workloadManagementMode;
    backendCompilationOptions->workloadManagementEnable = true;
    backendCompilationOptions->workloadManagementBarrierProgrammingMode =
            options->workloadManagementBarrierProgrammingMode;
    backendCompilationOptions->workloadManagementDmaFifoType = options->workloadManagementDmaFifoType;
    backendCompilationOptions->modelIdentifier = options->modelIdentifier;
    backendCompilationOptions->allocateDDRStackFrames = options->allocateDDRStackFrames;

    if (getCompilationMode(config) != config::CompilationMode::ReferenceSW) {
        auto enableProfiling = config.get<vpux::OV::PERF_COUNT>();
        backendCompilationOptions->enableDMAProfiling =
                enableProfiling ? options->enableDMAProfiling.getValue() : "false";
        backendCompilationOptions->enableShaveDDRAccessOptimization = options->enableShaveDDRAccessOptimization;
    }

    arch50xx::buildLowerVPUIP2ELFPipeline(pm, *backendCompilationOptions, log.nest());
}
