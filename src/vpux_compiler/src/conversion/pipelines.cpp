//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/Quant/Transforms/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// LowerIE2VPU
//

void vpux::buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(createConvertDynamicQuantToVPUNCEPass(log));

    pm.addPass(createConvertIEToVPUNCEPass(log));
    pm.addPass(createConvertLayers2VPUPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// LowerVPU2VPUIP
//

void vpux::buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, bool enableInPlaceBufferization,
                                       bool useMemrefForHostFunctionBufferization, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (enableInPlaceBufferization) {
        pm.addPass(createInPlaceBufferizationAnalyzePass());
    }
    pm.addPass(createOneShotBufferizeVPU2VPUIPPass());
    pm.addPass(VPUIP::createUngroupBoundedBuffersAsFuncArgsPass(log));
    pm.addPass(VPUIP::createUngroupHostBuffersAsFuncArgsPass(log));
    pm.addPass(createAddBuffersForNetResults(useMemrefForHostFunctionBufferization, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// ShaveCodeGen
//

void vpux::ShaveCodeGen::buildLowerSwLayers2LinalgPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(ShaveCodeGen::createConvertEltwiseLayers2MathPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// ShaveCodeGen specific passes included in DefaultHW and ReferenceSW
//

void vpux::ShaveCodeGen::buildShaveCodeGenPipelineIE(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(ShaveCodeGen::createEncapsulateCodeGenOpsPass());
    pm.addPass(ShaveCodeGen::createEarlyCodeGenCapsuleFusionPass());

    ShaveCodeGen::buildLowerSwLayers2LinalgPipeline(pm, log);
    pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(ShaveCodeGen::createFoldUnitDimReshapesPass(log));

    pm.addPass(ShaveCodeGen::createOutlineCodeGenCapsulesPass());
}

void vpux::ShaveCodeGen::buildShaveCodeGenPipelineVPU(mlir::OpPassManager& pm, Logger) {
    pm.addPass(ShaveCodeGen::createShaveKernelSimplifyPass());
    const auto grc = getDefaultGreedyRewriteConfig();
    // Move kernel results to arguments before doing any other
    // optimizations. This allows us to see a simpler form of the IR
    // and helps with the empty tensor elimination performed by this pass.
    //
    // In particular, empty tensor elimination needs to walk use-def
    // chains to find tensor.empty() ops but will refuse to traverse
    // any reshape-like ops. However these ops are introduced by our
    // optimization passes (specifically FlattenEltwiseKernel).
    pm.addPass(ShaveCodeGen::createMoveKernelResultsToArgumentsPass());

    pm.addPass(ShaveCodeGen::createDecomposeAggregateOpsPass());
    pm.addPass(ShaveCodeGen::createFlattenEltwiseKernelPass());
    pm.addPass(ShaveCodeGen::createLinalgTileAndFuseSwLayersPass());
    pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(ShaveCodeGen::createOneShotBufferizeSWKernelsPass());
    pm.addPass(ShaveCodeGen::createShaveStackAllocationPass());
}

void vpux::ShaveCodeGen::buildShaveCodeGenPipelineVPUIP(mlir::OpPassManager& pm, Logger) {
    pm.addPass(
            mlir::createConvertLinalgToAffineLoopsPass());  // E#154403 Analyze the pros/cons & replace Affine with SCF
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(ShaveCodeGen::createExpandLayersPass());
    pm.addPass(ShaveCodeGen::createLowerMathToShaveIntrinsicsPass());
    pm.addPass(ShaveCodeGen::createConvertAffine2LLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(ShaveCodeGen::createAdaptLLVMFuncsForShavePass());
}

//
// registerConversionPipelines
//

void vpux::registerConversionPipelines() {
    mlir::PassPipelineRegistration<>("lower-IE-to-VPU", "Performs full lowering from the IE Dialect to VPU Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         vpux::buildLowerIE2VPUPipeline(pm);
                                     });
    mlir::PassPipelineRegistration<vpux::DefaultHWOptionsBase>(
            "lower-VPU-to-VPUIP",
            "Performs full lowering from the VPU Dialect to VPUIP Dialect, SW operations are converted to SWKernelOp",
            [](mlir::OpPassManager& pm, const vpux::DefaultHWOptionsBase& options) {
                vpux::buildLowerVPU2VPUIPPipeline(pm, options.enableInPlaceBufferization,
                                                  options.useMemrefForHostFunctionBufferization);
            });
    mlir::PassPipelineRegistration<>("lower-sw-layers-to-linalg",
                                     "Performs full lowering of compatible SW Layers from IE to Linalg",
                                     [](mlir::OpPassManager& pm) {
                                         ShaveCodeGen::buildLowerSwLayers2LinalgPipeline(pm);
                                     });
    mlir::PassPipelineRegistration<>("shavecodegen-ie", "ShaveCodeGen specific passes", [](mlir::OpPassManager& pm) {
        ShaveCodeGen::buildShaveCodeGenPipelineIE(pm);
    });
    mlir::PassPipelineRegistration<>("shavecodegen-vpu", "ShaveCodeGen specific passes", [](mlir::OpPassManager& pm) {
        ShaveCodeGen::buildShaveCodeGenPipelineVPU(pm);
    });
    mlir::PassPipelineRegistration<>("shavecodegen-vpuip", "ShaveCodeGen specific passes", [](mlir::OpPassManager& pm) {
        ShaveCodeGen::buildShaveCodeGenPipelineVPUIP(pm);
    });
}
