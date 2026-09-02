//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/backend_pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/pipeline_options.hpp"

#include "vpux/compiler/NPU40XX/conversion.hpp"

#include "vpux/compiler/compilation_options.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/pipelines/options_mapper.hpp"
#include "vpux/compiler/pipelines/options_setup.hpp"
#include "vpux/utils/ov/config.hpp"

using namespace vpux;

//
// BackendPipelineStrategy40XX::buildELFPipeline
//

void BackendPipelineStrategy40XX::buildELFPipeline(mlir::OpPassManager& pm, const vpux::OV::Config& config,
                                                   mlir::TimingScope& rootTiming, Logger log) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");

    auto dpuDryRunMode = VPU::DPUDryRunMode::NONE;
    const auto compilationMode = getCompilationMode(config);
    auto backendCompilationOptions =
            BackendCompilationOptions40XX::createFromString(config.get<vpux::OV::BACKEND_COMPILATION_PARAMS>());

    VPUX_THROW_UNLESS(backendCompilationOptions != nullptr,
                      "build ELF pipeline failed to parse BACKEND_COMPILATION_PARAMS: {0}",
                      config.get<vpux::OV::BACKEND_COMPILATION_PARAMS>());

    if (compilationMode == config::CompilationMode::DefaultHW || config::isHostCompileMode(compilationMode)) {
        auto options = parseCompilationModeParams<DefaultHWOptions40XX>(config.get<vpux::OV::COMPILATION_MODE_PARAMS>(),
                                                                        getArchKind(config));
        VPUX_THROW_UNLESS(options != nullptr, "build ELF pipeline failed to parse COMPILATION_MODE_PARAMS: {0}",
                          config.get<vpux::OV::COMPILATION_MODE_PARAMS>());
        options->enableProfiling = config.get<vpux::OV::PERF_COUNT>();
        if (config.get<vpux::OV::TURBO>()) {
            overwriteIfUnset(options->optimizationLevel, 3);
        }
        setupParamsAccordingToOptimizationLevel(options->optimizationLevel, *options);
        setupPWLMParams(*options, config.get<vpux::OV::LOG_LEVEL>());
        dpuDryRunMode = VPU::getDPUDryRunMode(options->dpuDryRun);
        backendCompilationOptions->enableDMAProfiling =
                options->enableProfiling ? options->enableDMAProfiling.getValue() : "false";
        backendCompilationOptions->enableShaveDDRAccessOptimization = options->enableShaveDDRAccessOptimization;
        backendCompilationOptions->enableDumpStatisticsOfWlmOps = options->enableDumpTaskStats;
        backendCompilationOptions->workloadManagementMode = options->workloadManagementMode;
        backendCompilationOptions->workloadManagementEnable = options->workloadManagementEnable;
        backendCompilationOptions->workloadManagementBarrierProgrammingMode =
                options->workloadManagementBarrierProgrammingMode;
        backendCompilationOptions->workloadManagementDmaFifoType = options->workloadManagementDmaFifoType;
        backendCompilationOptions->modelIdentifier = options->modelIdentifier;
        backendCompilationOptions->allocateDDRStackFrames = options->allocateDDRStackFrames;
    }
    arch40xx::buildLowerVPUIP2ELFPipeline(pm, *backendCompilationOptions, log.nest(), dpuDryRunMode);
}
