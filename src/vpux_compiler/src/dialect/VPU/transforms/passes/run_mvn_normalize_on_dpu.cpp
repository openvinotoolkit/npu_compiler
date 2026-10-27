//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/core/IR/ops.hpp>
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_RUNMVNNORMALIZEONDPU
#define GEN_PASS_DEF_RUNMVNNORMALIZEONDPU
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
namespace {

// Forward declarations: defined below, used by RunMVNNormalizeOnDPU's methods further up the file.
mlir::Value createWeightsTable(mlir::Location origOpLoc, mlir::Value scale, mlir::Value bias, int64_t weightsTableC,
                               mlir::Type fp32ElemType, const DimsOrder& inOrder, mlir::PatternRewriter& rewriter);
mlir::Value convertToFp32ViaIdentityMaxPool(mlir::Location loc, mlir::Value input, mlir::Operation* ppeRefOp,
                                            mlir::PatternRewriter& rewriter);

//
// RunMVNNormalizeOnDPUPass
//

class RunMVNNormalizeOnDPU final : public mlir::OpRewritePattern<VPU::MVN1NormalizeOp> {
public:
    RunMVNNormalizeOnDPU(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::MVN1NormalizeOp>(ctx), _log(std::move(log)) {
        setDebugName("RunMVNNormalizeOnDPU");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::MVN1NormalizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    // Extract mean from meanVar, returns shape [1, C, 1, 1] for NCE operations
    mlir::Value extractMean(mlir::Location origOpLoc, mlir::Value meanVar, int64_t weightsTableC,
                            mlir::PatternRewriter& rewriter) const;
    // Extract scale from meanVar, returns shape [C, 1, 1, 1] for weights table
    mlir::Value extractScale(mlir::Location origOpLoc, mlir::Value meanVar, int64_t weightsTableC,
                             VPU::MVN1NormalizeOp origOp, mlir::PatternRewriter& rewriter) const;
    // Compute bias = -mean, input shape [1, C, 1, 1], output shape [C, 1, 1, 1] (after reshape)
    mlir::Value computeBias(mlir::Location origOpLoc, mlir::Value mean, int64_t weightsTableC, const DimsOrder& inOrder,
                            VPU::MVN1NormalizeOp origOp, mlir::PatternRewriter& rewriter) const;
    Logger _log;
};

// Extract mean from meanVar at W=0 (slice on the last dimension)
// Input:  meanVar with shape [1, C, 1, 2]
// Output: mean with shape [1, C, 1, 1] (keep NCE-friendly format)
mlir::Value RunMVNNormalizeOnDPU::extractMean(mlir::Location origOpLoc, mlir::Value meanVar, int64_t weightsTableC,
                                              mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();

    // meanVar shape: [1, C, 1, 2] with NHWC layout
    // Extract mean at W=0: slice [0, 0, 0, 0] with size [1, C, 1, 1]
    const SmallVector<int64_t> meanOffsets = {0, 0, 0, 0};
    const SmallVector<int64_t> meanSizes = {1, weightsTableC, 1, 1};
    auto mean = VPU::SliceOp::create(rewriter, appendLoc(origOpLoc, "_extract_mean"), meanVar,
                                     getIntArrayAttr(ctx, meanOffsets), getIntArrayAttr(ctx, meanSizes))
                        .getOutput();

    // Keep [1, C, 1, 1] shape for NCE.Eltwise operation
    return mean;
}

// Extract scale from meanVar at W=1 (slice on the last dimension)
// Input:  meanVar with shape [1, C, 1, 2]
// Output: scale with shape [C, 1, 1, 1] in fp32 (for weights table)
mlir::Value RunMVNNormalizeOnDPU::extractScale(mlir::Location origOpLoc, mlir::Value meanVar, int64_t weightsTableC,
                                               VPU::MVN1NormalizeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();

    // meanVar shape: [1, C, 1, 2] with NHWC layout
    // Extract scale at W=1: slice [0, 0, 0, 1] with size [1, C, 1, 1]
    const SmallVector<int64_t> scaleOffsets = {0, 0, 0, 1};
    const SmallVector<int64_t> scaleSizes = {1, weightsTableC, 1, 1};
    auto scale = VPU::SliceOp::create(rewriter, appendLoc(origOpLoc, "_extract_scale"), meanVar,
                                      getIntArrayAttr(ctx, scaleOffsets), getIntArrayAttr(ctx, scaleSizes))
                         .getOutput();

    // Convert to fp32 via a DPU identity MaxPool rather than a SW Convert (also reshapes to
    // [C, 1, 1, 1] for the weights table concat).
    return convertToFp32ViaIdentityMaxPool(origOpLoc, scale, origOp.getOperation(), rewriter);
}

// Compute bias = -mean
// Input:  mean with shape [1, C, 1, 1]
// Output: bias with shape [C, 1, 1, 1] in fp32 (for weights table)
mlir::Value RunMVNNormalizeOnDPU::computeBias(mlir::Location origOpLoc, mlir::Value mean, int64_t weightsTableC,
                                              const DimsOrder& inOrder, VPU::MVN1NormalizeOp origOp,
                                              mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();

    // mean has shape [1, C, 1, 1] with NHWC layout
    // Create constant -1 with same shape [1, C, 1, 1] for NCE.Eltwise broadcast
    const auto negOneShape = Shape({1, weightsTableC, 1, 1});
    const auto negOneType = mlir::RankedTensorType::get(negOneShape.raw(), mlir::Float16Type::get(ctx));
    const auto negOneAttr = Const::createConstContent(negOneType, ArrayRef({-1.0f}));
    const auto negOneContentAttr = Const::ContentAttr::get(negOneAttr).transform().reorder(inOrder).get();
    auto negOne = Const::DeclareOp::create(rewriter, appendLoc(origOpLoc, "_neg_one"), negOneContentAttr.getType(),
                                           std::move(negOneContentAttr))
                          .getOutput();

    // Multiply mean by -1 to get -mean
    // Both inputs have shape [1, C, 1, 1]
    const auto opType = VPU::EltwiseType::MULTIPLY;
    auto bias_ppeAttr = VPU::getPpeConfig(ctx).retrievePPEAttribute(origOp);
    VPU::MPEEngineAttr mpeEngineAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        mpeEngineAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngine());
    }

    auto bias = VPU::NCEEltwiseOp::create(rewriter, appendLoc(origOpLoc, "_compute_bias"), mean.getType(),
                                          /*reduce_xy_max=*/nullptr, /*reduce_xy_min=*/nullptr,
                                          /*reduce_tensor_min_max=*/nullptr, mean, negOne,
                                          /*weight_table_scale=*/nullptr, /*weight_table_bias=*/nullptr,
                                          VPU::EltwiseTypeAttr::get(ctx, opType), bias_ppeAttr, mpeEngineAttr,
                                          /*multi_cluster_strategy*/ nullptr,
                                          /*is_inplace*/ nullptr, nullptr, nullptr,
                                          /*axes_value=*/nullptr)
                        .getOutput();

    // Convert to fp32 via a DPU identity MaxPool rather than a SW Convert (also reshapes to
    // [C, 1, 1, 1] for the weights table concat).
    return convertToFp32ViaIdentityMaxPool(origOpLoc, bias, origOp.getOperation(), rewriter);
}

// Create weights table with shape [C, 1, 1, 4]
// Layout: [sparsityPointers(2), scale(1), bias(1)]
// Free function used by RunMVNNormalizeOnDPU.
mlir::Value createWeightsTable(mlir::Location origOpLoc, mlir::Value scale, mlir::Value bias, int64_t weightsTableC,
                               mlir::Type fp32ElemType, const DimsOrder& inOrder, mlir::PatternRewriter& rewriter) {
    auto ctx = rewriter.getContext();

    // Create weightsSparsityPointers with shape [C, 1, 1, 2] and initialized to 0
    const auto sparsityPointerShape = Shape({weightsTableC, 1, 1, 2});
    const auto sparsityPointerType = mlir::RankedTensorType::get(sparsityPointerShape.raw(), fp32ElemType);
    const auto baseAttr = Const::createConstContent(sparsityPointerType, ArrayRef({0.0f}));
    const auto contentAttr = Const::ContentAttr::get(baseAttr).transform().reorder(inOrder).get();
    auto weightsSparsityPointers =
            Const::DeclareOp::create(rewriter, appendLoc(origOpLoc, "_weights_sparsity_pointers"),
                                     contentAttr.getType(), std::move(contentAttr))
                    .getOutput();

    // Concat {weightsSparsityPointers [C,1,1,2], scale [C,1,1,1], bias [C,1,1,1]} to create weights table [C,1,1,4]
    const auto weightsTableOutType =
            mlir::cast<NDTypeInterface>(mlir::RankedTensorType::get({weightsTableC, 1, 1, 4}, fp32ElemType))
                    .changeDimsOrder(inOrder);
    const SmallVector<SmallVector<int64_t>> staticOffsets = {{0, 0, 0, 0}, {0, 0, 0, 2}, {0, 0, 0, 3}};
    auto weightsTable = VPU::ConcatOp::create(rewriter, appendLoc(origOpLoc, "_concat"), weightsTableOutType,
                                              mlir::ValueRange{weightsSparsityPointers, scale, bias},
                                              getIntArrayOfArray(ctx, staticOffsets))
                                .getOutput();

    // Reinterpret weights table to si32
    const auto desiredType = mlir::cast<NDTypeInterface>(weightsTable.getType())
                                     .changeElemType(getSInt32Type(ctx))
                                     .changeDimsOrder(inOrder);
    weightsTable =
            Core::ReinterpretCastOp::create(rewriter, appendLoc(origOpLoc, "_weights_table"), desiredType, weightsTable)
                    .getOutput();

    return weightsTable;
}

// Converts a small per-channel tensor (e.g. a bias/scale vector) to fp32 via an identity
// NCE.MaxPool (1x1 kernel/stride, weight table scale=1.0/bias=0.0) rather than a SW Convert,
// mirroring IE::createConvertPoolingForScaleTable (pooling_utils.cpp) at the VPU dialect level.
// Reshapes to [1,C,H,W] to give the pool genuine channel and spatial dimensions rather than the
// degenerate [totalElems,1,1,1] shape, then reshapes back to [totalElems,1,1,1].
// Returns the input unchanged if it is already fp32.
// ppeRefOp is only used to derive the (identity, since this op has no post-op/clamp) required
// ppe attribute -- any op in the pattern's match is suitable, matching how this file's other
// synthesized NCE ops (e.g. computeBias's NCEEltwiseOp) reuse the pattern's origOp for this.
mlir::Value convertToFp32ViaIdentityMaxPool(mlir::Location loc, mlir::Value input, mlir::Operation* ppeRefOp,
                                            mlir::PatternRewriter& rewriter) {
    auto* ctx = rewriter.getContext();
    const auto inputType = mlir::cast<NDTypeInterface>(input.getType());
    if (inputType.getElementType().isF32()) {
        return input;
    }

    const auto totalElems = getBoundedShape(input).totalSize();

    // Find the largest supported DW channel count that divides totalElems.
    // getSupportedChannelsDW() returns channels in descending order
    // so the first match is the largest valid channel count.
    // This assumption is load-bearing: a smaller poolC would produce a less-efficient workload.
    const auto& strategyFactory = VPU::getVPUStrategyFactory(ctx);
    int64_t poolC = 1;
    for (const int64_t candidate : strategyFactory->getSupportedChannelsDW()) {
        if (candidate <= totalElems && totalElems % candidate == 0) {
            poolC = candidate;
            break;
        }
    }

    // Split the remaining elements into H and W using the factor pair closest to a square split.
    // getFactorsList returns all (larger, smaller) factor pairs of rem for smaller in [1, sqrt(rem)],
    // in increasing order of smaller, so factors.back() is the closest-to-square pair.
    // When rem is small (e.g. rem=2 with poolC=totalElems/2), H may still be 1, which is harmless
    // for a 1x1-kernel MaxPool but means the shape is effectively 1D in that dimension.
    const auto rem = totalElems / poolC;
    const auto factors = getFactorsList(rem);
    const auto& factor = factors.back();
    const auto poolW = factor.first;
    const auto poolH = factor.second;

    // NCEMaxPool requires NHWC input. In practice all callers pass NHWC tensors (NCEEltwise /
    // NCEMaxPool outputs), but we guard defensively to avoid silently producing invalid NCE IR.
    auto nhwcInput = input;
    if (DimsOrder::fromValue(input) != DimsOrder::NHWC) {
        const auto nhwcType = mlir::cast<NDTypeInterface>(input.getType()).changeDimsOrder(DimsOrder::NHWC);
        nhwcInput = rewriter.createOrFold<VPU::PermuteCastOp>(loc, nhwcType, input, DimsOrder::NHWC.toAffineMap(ctx),
                                                              DimsOrder::fromValue(input).toAffineMap(ctx));
    }
    const auto inOrder = DimsOrder::NHWC;
    auto reshapedIn = rewriter.createOrFold<VPU::ShapeCastOp>(loc, nhwcInput,
                                                              getIntArrayAttr(ctx, Shape{1, poolC, poolH, poolW}));

    const auto fp32ElemType = mlir::Float32Type::get(ctx);

    // Identity weight table: scale=1.0, bias=0.0 (both compile-time fp32 constants), so the
    // pool computes value * 1.0 + 0.0 == value, promoted to fp32 by the ODU on write-out.
    const auto oneShape = Shape({poolC, 1, 1, 1});
    const auto oneType = mlir::RankedTensorType::get(oneShape.raw(), fp32ElemType);
    const auto oneContentAttr = Const::ContentAttr::get(Const::createConstContent(oneType, ArrayRef({1.0f})))
                                        .transform()
                                        .reorder(inOrder)
                                        .get();
    auto scaleConst = Const::DeclareOp::create(rewriter, appendLoc(loc, "_convert_scale"), oneContentAttr.getType(),
                                               std::move(oneContentAttr))
                              .getOutput();
    const auto zeroContentAttr = Const::ContentAttr::get(Const::createConstContent(oneType, ArrayRef({0.0f})))
                                         .transform()
                                         .reorder(inOrder)
                                         .get();
    auto biasConst = Const::DeclareOp::create(rewriter, appendLoc(loc, "_convert_bias"), zeroContentAttr.getType(),
                                              std::move(zeroContentAttr))
                             .getOutput();

    const auto poolOutType = mlir::cast<NDTypeInterface>(reshapedIn.getType()).changeElemType(fp32ElemType);
    const auto zeroPad = getIntArrayAttr(ctx, ArrayRef<int64_t>{0, 0});
    auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(zeroPad, zeroPad));
    const auto ppeAttr = VPU::getPpeConfig(ctx).retrievePPEAttribute(ppeRefOp);

    auto weightsTable = createWeightsTable(loc, scaleConst, biasConst, poolC, fp32ElemType, inOrder, rewriter);

    auto maxPoolOp = VPU::NCEMaxPoolOp::create(
            rewriter, appendLoc(loc, "_convert_maxpool"), poolOutType,
            /*reduceXyMax=*/nullptr, /*reduceXyMin=*/nullptr, /*reduceTensorMinMax=*/nullptr, reshapedIn, weightsTable,
            /*wtScale=*/nullptr, /*wtBias=*/nullptr, getIntArrayAttr(ctx, ArrayRef<int64_t>{1, 1}),
            getIntArrayAttr(ctx, ArrayRef<int64_t>{1, 1}), padAttr, ppeAttr, /*mpeEngineMode=*/nullptr,
            /*multiClusterStrategy=*/nullptr, /*output_padding=*/nullptr,
            /*input_padding=*/nullptr, /*s2dd2s_config=*/nullptr, /*axesValue=*/nullptr);

    return rewriter.createOrFold<VPU::ShapeCastOp>(loc, maxPoolOp.getOutput(),
                                                   getIntArrayAttr(ctx, Shape{totalElems, 1, 1, 1}));
}

mlir::LogicalResult RunMVNNormalizeOnDPU::matchAndRewrite(VPU::MVN1NormalizeOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto origOpLoc = origOp->getLoc();
    _log.trace("Found MVN1NormalizeOp operation '{0}' at '{1}'.", origOp->getName(), origOpLoc);

    auto ctx = origOp.getContext();

    // NCEMaxPool requires NHWC input layout. For other layouts (e.g. NCHW), skip and
    // let MVN1Normalize run as a SHAVE kernel instead of crashing the layout verifier.
    const auto inputOrder = DimsOrder::fromValue(origOp.getInput());
    if (inputOrder != DimsOrder::NHWC) {
        _log.trace("Skipping MVN1NormalizeOp: input layout {0} is not NHWC, falling back to SHAVE", inputOrder);
        return mlir::failure();
    }

    // NCE.Eltwise and NCE.MaxPool only accept f16/bf16 inputs. Bail out for f32 to avoid
    // creating invalid IR that fails the op verifier.
    const auto inputElemType = mlir::cast<NDTypeInterface>(origOp.getInput().getType()).getElementType();
    if (!mlir::isa<mlir::Float16Type, mlir::BFloat16Type>(inputElemType)) {
        _log.trace("Skipping MVN1NormalizeOp: element type {0} not supported by NCE", inputElemType);
        return mlir::failure();
    }

    // Only convert MVN1NormalizeOps that were created by the forceDecompose path in
    // DecomposeMVNPass. Those came from CMX-fitting MVN ops (previously monolithic Shave)
    // that were deliberately split so the normalize step can run on DPU -- freeing Shave
    // capacity is the benefit.
    // MVN1NormalizeOps from the normal decompose path (internalReshape / CMX-overflow MVN)
    // were already Shave-efficient; converting those to DPU creates serial DPU tiles that
    // cause resource contention with no Shave savings.
    if (!origOp.getFromForceDecompose()) {
        _log.trace("Skipping MVN1NormalizeOp: not from forceDecompose path, leaving as SHAVE");
        return mlir::failure();
    }

    // Validate input
    auto meanVar = origOp.getMeanVar();
    auto meanVarShape = getShape(meanVar);
    auto weightsTableC = meanVarShape[Dims4D::Act::C];
    if (weightsTableC % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT != 0) {
        return mlir::failure();
    }

    // Block conversion when the ORIGINAL MVN input was 1D (orig_shape W == 1). DecomposeMVN
    // always flattens NHWC tensors to [1,C,H*W,1], so the MVN1Normalize input has W=1
    // regardless of the original spatial layout -- checking the post-flatten shape cannot
    // distinguish originally-2D from originally-1D. The original shape is preserved in
    // MVN1MeanVar's orig_shape attribute. Converting originally-1D tensors serializes the
    // DPU pipeline (the NCE.MaxPool blocks the next DPU convolution) with no spatial
    // throughput benefit.
    if (auto* defOp = origOp.getMeanVar().getDefiningOp()) {
        if (auto meanVarOp = mlir::dyn_cast<VPU::MVN1MeanVarOp>(defOp)) {
            const auto origShape = parseIntArrayAttr<int64_t>(meanVarOp.getOrigShape());
            if (origShape.size() == 4 && origShape[Dims4D::Act::W.ind()] <= 1) {
                _log.trace("Skipping MVN1NormalizeOp: original MVN input was 1D (orig_W={0})",
                           origShape[Dims4D::Act::W.ind()]);
                return mlir::failure();
            }
        }
    }

    // meanVar shape: [1, C, 1, 2] with NHWC layout
    // W=0 contains mean (μ), W=1 contains scale (1/sqrt(σ² + ε))
    const auto inOrder = DimsOrder::fromValue(meanVar);
    auto fp32ElemType = mlir::Float32Type::get(ctx);

    // Extract mean (μ) from meanVar at W=0 (slice on the last dimension)
    // Output shape: [1, C, 1, 1]
    auto mean = extractMean(origOpLoc, meanVar, weightsTableC, rewriter);

    // Extract scale (1/sqrt(σ² + ε)) from meanVar at W=1 (slice on the last dimension)
    // Output shape: [C, 1, 1, 1] in fp32
    auto scale = extractScale(origOpLoc, meanVar, weightsTableC, origOp, rewriter);

    // Compute bias = -μ
    // Input shape: [1, C, 1, 1], Output shape: [C, 1, 1, 1] in fp32
    auto bias = computeBias(origOpLoc, mean, weightsTableC, inOrder, origOp, rewriter);

    auto weightsTable = createWeightsTable(origOpLoc, scale, bias, weightsTableC, fp32ElemType, inOrder, rewriter);

    // Create NCEMaxPoolOp to replace the original operation
    const SmallVector<int64_t> maxPoolStrides = {1, 1};
    const SmallVector<int64_t> maxPoolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(getIntArrayAttr(ctx, pads), getIntArrayAttr(ctx, pads)));
    auto ppeAttr = VPU::getPpeConfig(ctx).retrievePPEAttribute(origOp);
    VPU::MPEEngineAttr mpeEngineAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        const auto activationZp = getPerTensorZeroPointAttr(origOp.getInput());

        mpeEngineAttr = mlir::cast<VPU::MPEEngineAttr>(
                mpeEngineInterface.getMPEEngineWithZP(/*weightZp=*/nullptr, activationZp));
    }
    auto newMaxPoolOp = VPU::NCEMaxPoolOp::create(
            rewriter, origOp->getLoc(), origOp.getOutput().getType(),
            /*reduceXyMax */ nullptr,
            /*reduceXyMin */ nullptr,
            /*reduceTensorMinMax */ nullptr, origOp.getInput(), weightsTable, /*wtScale=*/nullptr, /*wtBias=*/nullptr,
            getIntArrayAttr(ctx, maxPoolKernels), getIntArrayAttr(ctx, maxPoolStrides), padAttr, ppeAttr, mpeEngineAttr,
            nullptr,
            /*MultiClusterStrategyAttr=*/nullptr, /*maxPoolOp.getOutputPaddingAttr()=*/nullptr,
            /*maxPoolOp.getInputPaddingAttr()=*/nullptr, /*axesValue=*/nullptr);
    rewriter.replaceOp(origOp, newMaxPoolOp.getOutput());

    return mlir::success();
}

class RunMVNNormalizeOnDPUPass final : public VPU::impl::RunMVNNormalizeOnDPUBase<RunMVNNormalizeOnDPUPass> {
public:
    explicit RunMVNNormalizeOnDPUPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void RunMVNNormalizeOnDPUPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RunMVNNormalizeOnDPU>(&ctx, _log);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createRunMVNNormalizeOnDPUPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createRunMVNNormalizeOnDPUPass(Logger log) {
    return std::make_unique<RunMVNNormalizeOnDPUPass>(log);
}
