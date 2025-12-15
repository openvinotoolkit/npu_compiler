//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/pipelines_register.hpp"
#include "vpux/compiler/NPU50XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU50XX/pipeline_options.hpp"

#include "vpux/compiler/NPU50XX/conversion.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/pipeline_strategies.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// PipelineRegistry50XX::registerPipelines
//

void PipelineRegistry50XX::registerPipelines() {
    mlir::PassPipelineRegistration<DefaultHWOptions50XX>(
            "reference-sw-mode", "Compile IE Network in Reference Software mode (SW only execution) for NPU50XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions50XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{config::ArchKind::NPU50XX,
                                                             config::CompilationMode::ReferenceSW, options};
                auto createPipelineStartegy = [&](config::CompilationMode) {
                    return createDialectPipelineStrategy50XXReferenceSW<DefaultHWOptions50XX>(&initCompilerOptions,
                                                                                              &options);
                };
                ReferenceSWStrategy factory(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions50XX>(
            "default-hw-mode", "Compile IE Network in Default Hardware mode (HW and SW execution) for NPU50XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions50XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{config::ArchKind::NPU50XX,
                                                             config::CompilationMode::DefaultHW, options};
                auto createPipelineStartegy = [&](config::CompilationMode) {
                    return createDialectPipelineStrategy50XX<DefaultHWOptions50XX>(&initCompilerOptions, &options);
                };
                DefaultHwStrategy factory(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions50XX>(
            "host-compile", "Compile IE Network in Host mode (host and HW execution) for NPU50XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions50XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{config::ArchKind::NPU50XX,
                                                             config::CompilationMode::HostCompile, options};
                auto createPipelineStrategy = [&](config::CompilationMode compilationMode) {
                    return createDialectPipelineStrategy50XXHostCompile<DefaultHWOptions50XX>(
                            compilationMode, &initCompilerOptions, &options);
                };
                HostPipelineStrategy factory(createPipelineStrategy, Logger::global());
                factory.buildPipeline(pm);
            });

    vpux::IE::arch50xx::registerIEPipelines();
    vpux::VPU::arch50xx::registerVPUPipelines();
    vpux::VPUIP::arch50xx::registerVPUIPPipelines();
    vpux::arch50xx::registerConversionPipeline();
}
