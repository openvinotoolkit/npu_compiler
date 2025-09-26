//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/pipelines_register.hpp"
#include "vpux/compiler/NPU40XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/pipeline_options.hpp"

#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/pipeline_strategies.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// PipelineRegistry40XX::registerPipelines
//

void PipelineRegistry40XX::registerPipelines() {
    mlir::PassPipelineRegistration<DefaultHWOptions40XX>(
            "reference-sw-mode", "Compile IE Network in Reference Software mode (SW only execution) for VPU40XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions40XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{config::ArchKind::NPU40XX,
                                                             config::CompilationMode::ReferenceSW, options};
                auto createPipelineStartegy = [&](config::CompilationMode) {
                    return createDialectPipelineStrategy40XXReferenceSW<DefaultHWOptions40XX>(&initCompilerOptions,
                                                                                              &options);
                };
                ReferenceSWStrategy factory(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions40XX>(
            "default-hw-mode", "Compile IE Network in Default Hardware mode (HW and SW execution) for VPU40XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions40XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{config::ArchKind::NPU40XX,
                                                             config::CompilationMode::DefaultHW, options};
                auto createPipelineStartegy = [&](config::CompilationMode) {
                    return createDialectPipelineStrategy40XX<DefaultHWOptions40XX>(&initCompilerOptions, &options);
                };
                DefaultHwStrategy factory(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions40XX>(
            "ws-monolithic", "Compile IE Network in Weights separation Monolithic mode for NPU40XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions40XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{config::ArchKind::NPU40XX,
                                                             config::CompilationMode::WSMonolithic, options};
                auto createPipelineStartegy = [&](config::CompilationMode compilationMode) {
                    return createDialectPipelineStrategy40XXWS<DefaultHWOptions40XX>(compilationMode,
                                                                                     &initCompilerOptions, &options);
                };
                WSMonolithicStrategy factory(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions40XX>(
            "ws-monolithic-partial", "Compile IE Network in Weights separation Monolithic mode for NPU40XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions40XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{config::ArchKind::NPU40XX,
                                                             config::CompilationMode::WSMonolithic, options};
                auto createPipelineStartegy = [&](config::CompilationMode compilationMode) {
                    return createDialectPipelineStrategy40XXWS<DefaultHWOptions40XX>(compilationMode,
                                                                                     &initCompilerOptions, &options);
                };
                auto factory = WSMonolithicStrategy::createForLITTests(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions40XX>(
            "host-compile", "Compile IE Network in Host mode (host and HW execution) for NPU40XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions40XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{config::ArchKind::NPU40XX,
                                                             config::CompilationMode::HostCompile, options};
                auto createPipelineStrategy = [&](config::CompilationMode compilationMode) {
                    return createDialectPipelineStrategy40XXHostCompile<DefaultHWOptions40XX>(
                            compilationMode, &initCompilerOptions, &options);
                };
                HostPipelineStrategy factory(createPipelineStrategy, Logger::global());
                factory.buildPipeline(pm);
            });

    vpux::IE::arch40xx::registerIEPipelines();
    vpux::VPU::arch40xx::registerVPUPipelines();
    vpux::VPUIP::arch40xx::registerVPUIPPipelines();
    vpux::arch40xx::registerConversionPipeline();
}
