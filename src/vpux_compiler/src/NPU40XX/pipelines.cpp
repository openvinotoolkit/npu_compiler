//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/pipelines.hpp"

#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"
#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPUASM/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/profiling/common.hpp"

#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// ReferenceSWMode
//

void vpux::buildReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions40XX& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // No passes should be run before this pipeline, with very few exceptions.
    IE::buildPostImportPipeline(pm, log);

    // Level 3 : Topology

    IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::arch37xx::buildInitialTransformationsPipeline(pm, IE::TransformOptions(options), log);
    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    // Resolve group quant MatMul pattern
    pm.addPass(IE::createUniquifyOpsPass(log));
    pm.addPass(IE::createMergeParallelFullyConnectedPass(log));
    pm.addPass(IE::createUnrollGroupQuantizePass(log));
    pm.addPass(IE::createUnrollFullyConnectedPass(log));
    pm.addPass(IE::createMergeFullyConnectedPass(log));
    if (options.fuseScalesToAccumulate) {
        pm.addPass(IE::createFuseScalesToAccumulatePass(log));
    }
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }

    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertStridedSlice2ConvPass(log));
    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(
            IE::createConvertToSpatialOpPass(false, isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(options.runtimeDequantizationLimit,
                                             isOptionEnabled(options.enableRuntimeDequant), log));
    if (options.enableMergeFakeQuant) {
        pm.addPass(IE::createMergeFakeQuantPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::arch37xx::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Lowering to VPU
    pm.addPass(createConvertLayers2VPUPass(log));
    pm.addPass(VPU::createDetectionOutputDecompositionPass(log));
    pm.addPass(VPU::arch37xx::createSplitRealDFTOpsPass(log));
    pm.addPass(VPU::createAddSwOpAuxiliaryBufferPass(log));
    pm.addPass(VPU::createSplitGRUSequencePass(log));
    pm.addPass(VPU::arch37xx::createDecomposeMVNPass(log));

    pm.addPass(VPU::createTilingStrategyAssignmentPass(/*enablePrefetchTiling=*/false, false, "true", log));
    pm.addPass(VPU::arch37xx::createApplyTilingMVN1SumPass(/*enablePrefetchTiling=*/false, log));
    pm.addPass(VPU::createApplyTilingPass(log));

    pm.addPass(VPU::createComputeInterpolateCoordinatesPass(/*enableExplicitDistributionInfoAttr=*/true, log));

    // Lowering to VPUIP
    vpux::arch37xx::buildLowerVPU2VPUIPPipeline(pm, options.enableInPlaceBufferization, log);

    // Level 2 : Abstract RunTime

    pm.addPass(VPUIP::createSetMemorySpacePass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));

    pm.addPass(VPUIP::createAddCopyBetweenSWKernelsAndNetworkIOPass(log));

    pm.addPass(VPUIP::createCopyOpTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    pm.addPass(VPUIP::createUngroupBoundedBuffersPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPUIP::createConvertTransferOpsToDMAsPass(log));

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    pm.addPass(VPUIP::createDMATaskProfilingReserveMemPass(DMAProfilingMode::SCRATCH, log));

    if (options.enableSWKernelPrefetchingReserveMem) {
        pm.addPass(VPUIP::createSWKernelPrefetchingReserveMemPass(log));
    }

    pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(VPUIP::createLinearizationPass(log));
    pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));

    pm.addPass(VPUIP::arch37xx::createAddSwKernelCacheHandlingOpsPass(log));

    VPUIP::buildHardwareAdaptationPipeline(pm, log);

    pm.addPass(VPUIP::arch40xx::createAddStartBarrierPass(log));
    pm.addPass(VPURT::arch37xx::createAddFinalBarrierPass(log));

    // Level 1 : VPU RunTime

    if (options.enableProfiling) {
        pm.addPass(VPUIP::createCaptureWorkpointPass(log));
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(Core::createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(options.enableColorBinPhysicalBarrierAssignment, std::nullopt,
                                                       log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createUpdateSwKernelParamsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// ShaveCodeGen
//

void vpux::buildShaveCodeGenPipeline(mlir::OpPassManager& pm, const ShaveCodeGenOptions40XX& options, Logger log) {
    log.trace("Entered buildShaveCodeGenPipeline()");

    DefaultHWOptions40XX defaultHWOptions;
    defaultHWOptions.locationsVerificationMode =
            "off";  // E#154882 Ensure standard VPUX passes compatibility with ShaveCodeGen path
    defaultHWOptions.enableProfiling = options.enableProfiling;
    defaultHWOptions.enableSWProfiling = options.enableSWProfiling;

    IE::arch40xx::buildDefaultHWPipeline(pm, defaultHWOptions, log);

    ShaveCodeGen::buildLowerSwLayers2LinalgPipeline(pm);

    /*
        E#108579
        High-level Linalg on tensors IR at this point.
        Due to inline IR & advantageous level of abstraction, most optimizations will probably reside here.
    */

    pm.addPass(ShaveCodeGen::createOutlineLinalgSwLayersPass());

    vpux::arch37xx::buildLowerIE2VPUPipeline(pm, log);
    VPU::arch40xx::buildDefaultHWPipeline(pm, defaultHWOptions, log);

    // Lowering to VPUIP
    vpux::arch37xx::buildLowerVPU2VPUIPPipeline(pm, defaultHWOptions.enableInPlaceBufferization, log);

    pm.addPass(
            mlir::createConvertLinalgToAffineLoopsPass());  // E#154403 Analyze the pros/cons & replace Affine with SCF
    pm.addPass(ShaveCodeGen::createConvertAffine2LLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(ShaveCodeGen::createAdaptLLVMFuncsForShavePass());

    defaultHWOptions.enableShaveKernelTiling = false;
    defaultHWOptions.enableOptimizeCopies =
            false;  // E#154882 Ensure standard VPUX passes compatibility with ShaveCodeGen path

    VPUIP::arch40xx::buildDefaultHWPipeline(pm, defaultHWOptions, log);

    log.trace("Exiting buildShaveCodeGenPipeline()");
}

//
// DefaultHWMode
//

void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions40XX& options, Logger log) {
    IE::arch40xx::buildDefaultHWPipeline(pm, options, log);

    // Lowering to VPU
    if (options.enableM2I) {
        pm.addPass(createConvertIEToVPUM2IPass(log));
    }

    vpux::arch37xx::buildLowerIE2VPUPipeline(pm, log);
    VPU::arch40xx::buildDefaultHWPipeline(pm, options, log);

    // Lowering to VPUIP
    vpux::arch37xx::buildLowerVPU2VPUIPPipeline(pm, options.enableInPlaceBufferization, log);
    VPUIP::arch40xx::buildDefaultHWPipeline(pm, options, log);
}
