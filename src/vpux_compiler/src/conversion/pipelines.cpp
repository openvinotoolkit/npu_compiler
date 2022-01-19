//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// LowerIE2IERT
//

void vpux::buildLowerIE2IERTPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(createBufferizeIEPass(log));
    pm.addPass(createBufferizeFuncAndReturnPass(log));
    pm.addPass(createAddBuffersForNetResults(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// LowerIERT2VPUIP
//

void vpux::buildLowerIERT2VPUIPPipeline(mlir::OpPassManager& pm, const LowerIERT2VPUIPOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(createConvertLayers2VPUIPPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(createConvertDeclarations2VPUIPPass(log));
    pm.addPass(createConvertViewOps2VPUIPPass(log));
    if (options.enableCompressWeights) {
        pm.addPass(vpux::VPUIP::createCompressWeightsPass(log));
    }
    pm.addPass(createConvertAsyncOps2VPUIPPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(createMoveDeclarationsToTopPass(log));
}

//
// LowerLinalg2VPUIP
//

void vpux::buildLowerLinalg2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) {
    //pm.addPass(createConvertLinalg2VPUIPPass(log));
}

//
// registerConversionPipelines
//

void vpux::registerConversionPipelines() {
    mlir::PassPipelineRegistration<>("lower-IE-to-IERT", "Performs full lowering from the IE Dialect to IERT Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerIE2IERTPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<LowerIERT2VPUIPOptions>(
            "lower-IERT-to-VPUIP", "Performs full lowering from the IERT Dialect to VPUIP Dialect",
            [](mlir::OpPassManager& pm, const LowerIERT2VPUIPOptions& options) {
                buildLowerIERT2VPUIPPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<>("lower-Linalg-to-IERT", "Performs full lowering from the Linalg Dialect to VPUIP Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerLinalg2VPUIPPipeline(pm);
                                     });
}
