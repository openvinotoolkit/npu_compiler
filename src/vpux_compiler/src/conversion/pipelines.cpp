//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// LowerIE2IERT
//

void vpux::buildLowerIE2IERTPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(createBufferizeIEPass(log));
    pm.addPass(createOneShotBufferizeVPU2VPUIPPass());
    pm.addPass(createAddBuffersForNetResults(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// ShaveCodeGen
//

void vpux::ShaveCodeGen::buildLowerSwLayers2LinalgPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(createConvertEltwiseLayers2MathPass(log));
    pm.addPass(mlir::createConvertElementwiseToLinalgPass());
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// registerConversionPipelines
//

void vpux::registerConversionPipelines() {
    mlir::PassPipelineRegistration<>("lower-IE-to-IERT", "Performs full lowering from the IE Dialect to IERT Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerIE2IERTPipeline(pm);
                                     });
    mlir::PassPipelineRegistration<>("lower-sw-layers-to-linalg",
                                     "Performs full lowering of compatible SW Layers from IE to Linalg",
                                     [](mlir::OpPassManager& pm) {
                                         ShaveCodeGen::buildLowerSwLayers2LinalgPipeline(pm);
                                     });
}
