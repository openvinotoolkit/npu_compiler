//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/compiler.hpp"

#include "intel_npu/config/common.hpp"
#include "intel_npu/config/compiler.hpp"

#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/NPU37XX/pipeline_strategy.hpp"
#include "vpux/compiler/NPU37XX/pipelines.hpp"

#include "vpux/compiler/options_mapper.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

using namespace vpux;

//
// PipelineStrategy37XX::buildPipeline
//

void PipelineStrategy37XX::buildPipeline(mlir::PassManager& pm, const intel_npu::Config& config,
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
        const auto options = ReferenceSWOptions37XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->matchAndCopyOptionValuesFrom(initCompilerOptions);
        options->enableProfiling = enableProfiling;
        buildReferenceSWModePipeline(pm, *options, log.nest());
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        const auto options = DefaultHWOptions37XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->matchAndCopyOptionValuesFrom(initCompilerOptions);
        options->enableProfiling = enableProfiling;
        options->enableConvertAvgPoolToDWConv = false;
        options->enableHandleAsymmetricStrides = false;

        buildDefaultHWModePipeline(pm, *options, log.nest());
    } else if (compilationMode == VPU::CompilationMode::ShaveCodeGen) {
        ShaveCodeGenOptions37XX emptyOptions;
        buildShaveCodeGenPipeline(pm, emptyOptions, log.nest());
    } else {
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

void PipelineStrategy37XX::buildELFPipeline(mlir::PassManager& pm, const intel_npu::Config&,
                                            mlir::TimingScope& rootTiming, Logger log, /*useWlm*/ bool) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");
    arch37xx::buildLowerVPUIP2ELFPipeline(pm, log.nest());
}
