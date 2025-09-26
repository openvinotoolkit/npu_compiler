//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/ELFNPU37XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/Passes.h>

using namespace vpux;

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

void vpux::arch37xx::buildLowerIE2VPUPipelineReferenceSW(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(createConvertLayers2VPUPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// registerConversionPipelines
//

void vpux::arch37xx::registerConversionPipeline() {
    mlir::PassPipelineRegistration<>("lower-IE-to-VPU-referense-sw",
                                     "Performs full lowering from the IE Dialect to VPU Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         vpux::arch37xx::buildLowerIE2VPUPipelineReferenceSW(pm);
                                     });
    mlir::PassPipelineRegistration<>("lower-VPUIP-to-ELF",
                                     "Performs full lowering from the VPUIP Dialect to the VPUMI37XX and ELF Dialects",
                                     [](mlir::OpPassManager& pm) {
                                         vpux::arch37xx::buildLowerVPUIP2ELFPipeline(pm);
                                     });
}
