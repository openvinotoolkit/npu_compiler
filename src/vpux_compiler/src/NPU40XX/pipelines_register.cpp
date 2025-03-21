//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/pipelines_register.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/pipelines.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include "vpux/utils/core/optional.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// PipelineRegistry40XX::registerPipelines
//

void PipelineRegistry40XX::registerPipelines() {
    mlir::PassPipelineRegistration<ShaveCodeGenOptions40XX>(
            "ShaveCodeGen", "Compile both from IE to VPUIP and from IERT to LLVM for NPU40XX",
            [](mlir::OpPassManager& pm, const ShaveCodeGenOptions40XX& options) {
                buildShaveCodeGenPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ReferenceSWOptions40XX>(
            "reference-sw-mode", "Compile IE Network in Reference Software mode (SW only execution) for NPU40XX",
            [](mlir::OpPassManager& pm, const ReferenceSWOptions40XX& options) {
                VPU::buildInitCompilerPipeline(pm,
                                               {VPU::ArchKind::NPU40XX, VPU::CompilationMode::ReferenceSW, options});

                buildReferenceSWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions40XX>(
            "default-hw-mode", "Compile IE Network in Default Hardware mode (HW and SW execution) for NPU40XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions40XX& options) {
                VPU::buildInitCompilerPipeline(pm, {VPU::ArchKind::NPU40XX, VPU::CompilationMode::DefaultHW, options});

                buildDefaultHWModePipeline(pm, options);
            });

    vpux::IE::arch40xx::registerIEPipelines();
    vpux::VPU::arch40xx::registerVPUPipelines();
    vpux::VPUIP::arch40xx::registerVPUIPPipelines();
    vpux::arch40xx::registerConversionPipeline();
}
