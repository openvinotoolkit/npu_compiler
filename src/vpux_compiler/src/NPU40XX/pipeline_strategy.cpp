//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/compiler.hpp"

#include "intel_npu/config/common.hpp"
#include "intel_npu/config/compiler.hpp"

#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU40XX/pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/pipelines.hpp"

#include "vpux/compiler/options_mapper.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/dry_run_utils.hpp"

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/profiling/common.hpp"

using namespace vpux;

//
// PipelineStrategy40XX::buildPipeline
//

namespace {
void setupPWLMCompilationParams(int optimizationLevel, DefaultHWOptions40XX& compilationOptions, bool useWlm) {
    if (!useWlm) {
        compilationOptions.workloadManagementEnable = false;
        return;
    }
    bool isworkloadManagementEnableSet = compilationOptions.workloadManagementEnable.hasValue() ? true : false;

    if (!isworkloadManagementEnableSet) {
        std::optional<int> originalValueBarrierCountThreshold = std::nullopt;
        std::optional<WorkloadManagementMode> originalValueWorkloadManagementMode = std::nullopt;

        if (compilationOptions.workloadManagementMode.hasValue()) {
            originalValueWorkloadManagementMode = compilationOptions.workloadManagementMode;
        }

        if (compilationOptions.workloadManagementBarrierCountThreshold.hasValue()) {
            originalValueBarrierCountThreshold = compilationOptions.workloadManagementBarrierCountThreshold;
        }

        switch (optimizationLevel) {
        case 0:
            compilationOptions.workloadManagementEnable = false;
            break;
        case 1:
            compilationOptions.workloadManagementEnable = true;
            break;
        case 2: {
            compilationOptions.workloadManagementEnable = true;
            compilationOptions.workloadManagementBarrierCountThreshold = std::numeric_limits<int>::max();
            break;
        }
        case 3: {
            compilationOptions.workloadManagementEnable = true;
            compilationOptions.workloadManagementBarrierCountThreshold = std::numeric_limits<int>::max();
            compilationOptions.workloadManagementMode = WorkloadManagementMode::PWLM_V1_BARRIER_FIFO;
            break;
        }
        default:
            VPUX_THROW("Unexpected optimization-level. Actual value = {0}\n"
                       "Possible values: 0 - optimization for compilation time, "
                       "1 - optimization for execution time (default), 2 - high optimization for execution time, 3 - "
                       "optimization for maximaze HW utilization, may affect compilation time and memory footprint",
                       optimizationLevel);
            break;
        }

        if (originalValueWorkloadManagementMode.has_value()) {
            compilationOptions.workloadManagementMode = originalValueWorkloadManagementMode.value();
        }

        if (originalValueBarrierCountThreshold.has_value()) {
            compilationOptions.workloadManagementBarrierCountThreshold = originalValueBarrierCountThreshold.value();
        }
    }
}

}  // namespace

void PipelineStrategy40XX::buildPipeline(mlir::PassManager& pm, const intel_npu::Config& config,
                                         mlir::TimingScope& rootTiming, Logger log) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");

    const auto initCompilerOptions = getInitCompilerOptions(config);
    const auto& numOfDPUGroups = initCompilerOptions.numberOfDPUGroups;
    const auto& numOfDMAPorts = initCompilerOptions.numberOfDMAPorts;

    VPUX_THROW_WHEN(
            numOfDPUGroups.hasValue() && numOfDMAPorts.hasValue() &&
                    numOfDMAPorts.getValue() > numOfDPUGroups.getValue(),
            "Requested configuration not supported by runtime. Number of DMA ports ({0}) larger than NCE clusters "
            "({1})",
            numOfDMAPorts.getValue(), numOfDPUGroups.getValue());

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log.nest());

    const auto enableProfiling = config.get<intel_npu::PERF_COUNT>();
    const auto compilationMode = getCompilationMode(config);

    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        const auto options = ReferenceSWOptions40XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->matchAndCopyOptionValuesFrom(initCompilerOptions);
        options->enableProfiling = enableProfiling;
        buildReferenceSWModePipeline(pm, *options, log.nest());
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        auto options = DefaultHWOptions40XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->matchAndCopyOptionValuesFrom(initCompilerOptions);
        setupPWLMCompilationParams(options->optimizationLevel, *options, options->workloadManagementEnable);
        options->enableProfiling = enableProfiling;
        options->enableConvertAvgPoolToDWConv = false;
        options->enableHandleAsymmetricStrides = false;
        // TODO: E#-108844 Support Compressed activation with Partial workload management
        if (options->workloadManagementEnable) {
            options->enableCompressActivationSpill = false;
        }
        buildDefaultHWModePipeline(pm, *options, log.nest());
    } else if (compilationMode == VPU::CompilationMode::ShaveCodeGen) {
        auto options = ShaveCodeGenOptions40XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->enableProfiling = enableProfiling;
        buildShaveCodeGenPipeline(pm, *options, log.nest());
    } else {
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

void PipelineStrategy40XX::buildELFPipeline(mlir::PassManager& pm, const intel_npu::Config& config,
                                            mlir::TimingScope& rootTiming, Logger log, bool useWlm) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");

    auto dpuDryRunMode = VPU::DPUDryRunMode::NONE;
    const auto compilationMode = getCompilationMode(config);
    auto backendCompilationOptions =
            BackendCompilationOptions40XX::createFromString(config.get<intel_npu::BACKEND_COMPILATION_PARAMS>());

    VPUX_THROW_UNLESS(backendCompilationOptions != nullptr,
                      "build ELF pipeline failed to parse BACKEND_COMPILATION_PARAMS: {0}",
                      config.get<intel_npu::BACKEND_COMPILATION_PARAMS>());

    if (compilationMode == VPU::CompilationMode::DefaultHW) {
        auto options = DefaultHWOptions40XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "build ELF pipeline failed to parse COMPILATION_MODE_PARAMS: {0}",
                          config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        setupPWLMCompilationParams(options->optimizationLevel, *options, useWlm);
        dpuDryRunMode = VPU::getDPUDryRunMode(options->dpuDryRun);
        backendCompilationOptions->enableDMAProfiling = options->enableDMAProfiling.getValue();
        backendCompilationOptions->enableShaveDDRAccessOptimization = options->enableShaveDDRAccessOptimization;
        backendCompilationOptions->enableDumpStatisticsOfWlmOps = options->enableDumpTaskStats;
        backendCompilationOptions->workloadManagementBarrierCountThreshold =
                options->workloadManagementBarrierCountThreshold;
        backendCompilationOptions->workloadManagementMode = options->workloadManagementMode;
        backendCompilationOptions->workloadManagementEnable = options->workloadManagementEnable;
    }
    arch40xx::buildLowerVPUIP2ELFPipeline(pm, *backendCompilationOptions, log.nest(), dpuDryRunMode);
}
