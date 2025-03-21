//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/pipelines.hpp"
#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/ELFNPU37XX/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// LowerIE2VPU
//

void vpux::arch37xx::buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(createConvertDynamicQuantToVPUNCEPass(log));

    pm.addPass(vpux::arch37xx::createConvertIEToVPUNCEPass(log));
    pm.addPass(createConvertLayers2VPUPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// LowerVPU2VPUIPSWKernel
//

void vpux::arch37xx::buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, bool enableInPlaceBufferization, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(createAdjustDynamicOpsBeforeBufferizationPass());
    pm.addPass(VPU::createLegalizeDynamicShapeConcatForSWLayersPass(log));
    if (enableInPlaceBufferization) {
        pm.addPass(createInPlaceBufferizationAnalyzePass());
    }
    pm.addPass(createOneShotBufferizeVPU2VPUIPPass());
    pm.addPass(VPUIP::createUngroupBoundedBuffersAsFuncArgsPass(log));
    pm.addPass(createAddBuffersForNetResults(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// LowerVPUIP2VPUMI37XXAndELF
//

void vpux::arch37xx::buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(createConvertVPUIP2VPUMI37XXPass(log));
    pm.addPass(VPUMI37XX::createAssignFullKernelPathPass(log));
    pm.addPass(VPUMI37XX::createBarrierComputationPass(log));

    pm.addPass(createConvertVPUMI37XX2ELFPass(log));
    pm.addPass(ELFNPU37XX::createRemoveEmptyELFSectionsPass(log));
    pm.addPass(ELFNPU37XX::createUpdateELFSectionFlagsPass(log));
}

//
// registerConversionPipelines
//

void vpux::arch37xx::registerConversionPipeline() {
    mlir::PassPipelineRegistration<>("lower-IE-to-VPU", "Performs full lowering from the IE Dialect to VPU Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         vpux::arch37xx::buildLowerIE2VPUPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<vpux::DefaultHWOptions37XX>(
            "lower-VPU-to-VPUIP",
            "Performs full lowering from the VPU Dialect to VPUIP Dialect, SW operations are converted to SWKernelOp",
            [](mlir::OpPassManager& pm, const vpux::DefaultHWOptions37XX& options) {
                vpux::arch37xx::buildLowerVPU2VPUIPPipeline(pm, options.enableInPlaceBufferization);
            });

    mlir::PassPipelineRegistration<>("lower-VPUIP-to-ELF",
                                     "Performs full lowering from the VPUIP Dialect to the VPUMI37XX and ELF Dialects",
                                     [](mlir::OpPassManager& pm) {
                                         vpux::arch37xx::buildLowerVPUIP2ELFPipeline(pm);
                                     });
}
