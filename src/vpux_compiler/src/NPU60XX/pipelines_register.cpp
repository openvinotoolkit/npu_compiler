//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/pipelines_register.hpp"
#include "vpux/compiler/NPU60XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU60XX/pipeline_options.hpp"

#include "vpux/compiler/NPU60XX/conversion.hpp"
#include "vpux/compiler/NPU60XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/pipeline_strategies.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// PipelineRegistry60XX::PipelineRegistry60XX
//

PipelineRegistry60XX::PipelineRegistry60XX(config::Platform platform): _platform(platform) {
    VPUX_THROW_UNLESS(getArch(platform) == config::ArchKind::NPU60XX,
                      "Failed to initialize PipelineRegistry60XX with platform: {0}",
                      config::stringifyPlatform(platform));
}

//
// PipelineRegistry60XX::registerPipelines
//

void PipelineRegistry60XX::registerPipelines() {
    mlir::PassPipelineRegistration<DefaultHWOptions60XX>(
            "reference-sw-mode", "Compile IE Network in Reference Software mode (SW only execution) for NPU60XX",
            [platform = _platform](mlir::OpPassManager& pm, const DefaultHWOptions60XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{platform, config::CompilationMode::ReferenceSW, options};
                auto createPipelineStartegy = [&](config::CompilationMode) {
                    return createDialectPipelineStrategy60XXReferenceSW<DefaultHWOptions60XX>(&initCompilerOptions,
                                                                                              &options);
                };
                ReferenceSWStrategy factory(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions60XX>(
            "default-hw-mode", "Compile IE Network in Default Hardware mode (HW and SW execution) for NPU60XX",
            [platform = _platform](mlir::OpPassManager& pm, const DefaultHWOptions60XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{platform, config::CompilationMode::DefaultHW, options};
                auto createPipelineStartegy = [&](config::CompilationMode) {
                    return createDialectPipelineStrategy60XX<DefaultHWOptions60XX>(&initCompilerOptions, &options);
                };
                DefaultHwStrategy factory(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions60XX>(
            "host-compile", "Compile IE Network in Host mode (host and HW execution) for NPU60XX",
            [platform = _platform](mlir::OpPassManager& pm, const DefaultHWOptions60XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{platform, config::CompilationMode::HostCompile_Interpreter,
                                                             options};
                auto createPipelineStrategy = [&](config::CompilationMode compilationMode) {
                    return createDialectPipelineStrategy60XXHostCompile<DefaultHWOptions60XX>(
                            compilationMode, &initCompilerOptions, &options);
                };
                HostPipelineStrategy factory(createPipelineStrategy, Logger::global());
                factory.buildPipeline(pm);
            });

    vpux::IE::arch60xx::registerIEPipelines();
    vpux::VPU::arch60xx::registerVPUPipelines();
    vpux::VPUIP::arch60xx::registerVPUIPPipelines();
    vpux::arch60xx::registerConversionPipeline();
}
