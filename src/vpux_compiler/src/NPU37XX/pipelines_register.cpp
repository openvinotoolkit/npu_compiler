//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/pipelines_register.hpp"
#include "vpux/compiler/NPU37XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU37XX/pipeline_options.hpp"

#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/pipeline_strategies.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// PipelineRegistry37XX::registerPipelines
//

void PipelineRegistry37XX::registerPipelines() {
    mlir::PassPipelineRegistration<DefaultHWOptions37XX>(
            "reference-sw-mode", "Compile IE Network in Reference Software mode (SW only execution) for NPU37XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions37XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{config::ArchKind::NPU37XX,
                                                             config::CompilationMode::ReferenceSW, options};
                auto createPipelineStartegy = [&](config::CompilationMode) {
                    return createDialectPipelineStrategy37XXReferenceSW<DefaultHWOptions37XX>(&initCompilerOptions,
                                                                                              &options);
                };
                ReferenceSWStrategy factory(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions37XX>(
            "default-hw-mode", "Compile IE Network in Default Hardware mode (HW and SW execution) for NPU37XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions37XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{config::ArchKind::NPU37XX,
                                                             config::CompilationMode::DefaultHW, options};
                auto createPipelineStartegy = [&](config::CompilationMode) {
                    return createDialectPipelineStrategy37XX<DefaultHWOptions37XX>(&initCompilerOptions, &options);
                };
                DefaultHwStrategy factory(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions37XX>(
            "ws-monolithic", "Compile IE Network in Weights separation Monolithic mode for NPU37XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions37XX& options) {
                VPU::InitCompilerOptions initCompilerOptions{config::ArchKind::NPU37XX,
                                                             config::CompilationMode::WSMonolithic, options};
                auto createPipelineStartegy = [&](config::CompilationMode) {
                    return createDialectPipelineStrategy37XX<DefaultHWOptions37XX>(&initCompilerOptions, &options);
                };
                WSMonolithicStrategy factory(createPipelineStartegy, Logger::global());
                factory.buildPipeline(pm);
            });
    vpux::IE::arch37xx::registerIEPipelines();
    vpux::VPU::arch37xx::registerVPUPipelines();
    vpux::VPUIP::arch37xx::registerVPUIPPipelines();
    vpux::arch37xx::registerConversionPipeline();
}
