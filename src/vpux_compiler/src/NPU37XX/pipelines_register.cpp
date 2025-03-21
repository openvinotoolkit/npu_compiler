//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/pipelines_register.hpp"
#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/pipelines.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// PipelineRegistry37XX::registerPipelines
//

void PipelineRegistry37XX::registerPipelines() {
    mlir::PassPipelineRegistration<ShaveCodeGenOptions37XX>(
            "ShaveCodeGen", "Compile both from IE to VPUIP and from IERT to LLVM for NPU37XX",
            [](mlir::OpPassManager& pm, const ShaveCodeGenOptions37XX& options) {
                buildShaveCodeGenPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ReferenceSWOptions37XX>(
            "reference-sw-mode", "Compile IE Network in Reference Software mode (SW only execution) for NPU37XX",
            [](mlir::OpPassManager& pm, const ReferenceSWOptions37XX& options) {
                VPU::buildInitCompilerPipeline(pm,
                                               {VPU::ArchKind::NPU37XX, VPU::CompilationMode::ReferenceSW, options});

                buildReferenceSWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions37XX>(
            "default-hw-mode", "Compile IE Network in Default Hardware mode (HW and SW execution) for NPU37XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions37XX& options) {
                VPU::buildInitCompilerPipeline(pm, {VPU::ArchKind::NPU37XX, VPU::CompilationMode::DefaultHW, options});

                buildDefaultHWModePipeline(pm, options);
            });
    vpux::IE::arch37xx::registerIEPipelines();
    vpux::VPU::arch37xx::registerVPUPipelines();
    vpux::VPUIP::arch37xx::registerVPUIPPipelines();
    vpux::arch37xx::registerConversionPipeline();
}
