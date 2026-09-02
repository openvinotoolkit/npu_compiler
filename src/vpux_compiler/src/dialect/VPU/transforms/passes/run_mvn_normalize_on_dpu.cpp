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
    auto mean = rewriter.create<VPU::SliceOp>(appendLoc(origOpLoc, "_extract_mean"), meanVar,
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
    auto scale = rewriter.create<VPU::SliceOp>(appendLoc(origOpLoc, "_extract_scale"), meanVar,
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
    auto negOne = rewriter.create<Const::DeclareOp>(appendLoc(origOpLoc, "_neg_one"), negOneContentAttr.getType(),
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

    auto bias = rewriter.create<VPU::NCEEltwiseOp>(appendLoc(origOpLoc, "_compute_bias"), mean.getType(),
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
// Free function so both RunMVNNormalizeOnDPU and RunAddBroadcastOnDPU can use it.
mlir::Value createWeightsTable(mlir::Location origOpLoc, mlir::Value scale, mlir::Value bias, int64_t weightsTableC,
                               mlir::Type fp32ElemType, const DimsOrder& inOrder, mlir::PatternRewriter& rewriter) {
    auto ctx = rewriter.getContext();

    // Create weightsSparsityPointers with shape [C, 1, 1, 2] and initialized to 0
    const auto sparsityPointerShape = Shape({weightsTableC, 1, 1, 2});
    const auto sparsityPointerType = mlir::RankedTensorType::get(sparsityPointerShape.raw(), fp32ElemType);
    const auto baseAttr = Const::createConstContent(sparsityPointerType, ArrayRef({0.0f}));
    const auto contentAttr = Const::ContentAttr::get(baseAttr).transform().reorder(inOrder).get();
    auto weightsSparsityPointers = rewriter.create<Const::DeclareOp>(appendLoc(origOpLoc, "_weights_sparsity_pointers"),
                                                                     contentAttr.getType(), std::move(contentAttr))
                                           .getOutput();

    // Concat {weightsSparsityPointers [C,1,1,2], scale [C,1,1,1], bias [C,1,1,1]} to create weights table [C,1,1,4]
    const auto weightsTableOutType =
            mlir::cast<NDTypeInterface>(mlir::RankedTensorType::get({weightsTableC, 1, 1, 4}, fp32ElemType))
                    .changeDimsOrder(inOrder);
    const SmallVector<SmallVector<int64_t>> staticOffsets = {{0, 0, 0, 0}, {0, 0, 0, 2}, {0, 0, 0, 3}};
    auto weightsTable = rewriter.create<VPU::ConcatOp>(appendLoc(origOpLoc, "_concat"), weightsTableOutType,
                                                       mlir::ValueRange{weightsSparsityPointers, scale, bias},
                                                       getIntArrayOfArray(ctx, staticOffsets))
                                .getOutput();

    // Reinterpret weights table to si32
    const auto desiredType = mlir::cast<NDTypeInterface>(weightsTable.getType())
                                     .changeElemType(getSInt32Type(ctx))
                                     .changeDimsOrder(inOrder);
    weightsTable =
            rewriter.create<Core::ReinterpretCastOp>(appendLoc(origOpLoc, "_weights_table"), desiredType, weightsTable)
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

    const auto totalElems = inputType.getShape().totalSize();

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
    const auto& [poolW, poolH] = factors.back();

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
    auto scaleConst = rewriter.create<Const::DeclareOp>(appendLoc(loc, "_convert_scale"), oneContentAttr.getType(),
                                                        std::move(oneContentAttr))
                              .getOutput();
    const auto zeroContentAttr = Const::ContentAttr::get(Const::createConstContent(oneType, ArrayRef({0.0f})))
                                         .transform()
                                         .reorder(inOrder)
                                         .get();
    auto biasConst = rewriter.create<Const::DeclareOp>(appendLoc(loc, "_convert_bias"), zeroContentAttr.getType(),
                                                       std::move(zeroContentAttr))
                             .getOutput();

    const auto poolOutType = mlir::cast<NDTypeInterface>(reshapedIn.getType()).changeElemType(fp32ElemType);
    const auto zeroPad = getIntArrayAttr(ctx, ArrayRef<int64_t>{0, 0});
    auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(zeroPad, zeroPad));
    const auto ppeAttr = VPU::getPpeConfig(ctx).retrievePPEAttribute(ppeRefOp);

    mlir::Value weightsTable = nullptr;
    mlir::Value wtScale = nullptr;
    mlir::Value wtBias = nullptr;
    if (VPU::MPEEngineConfig::useNewWeightTableFormat(ppeRefOp, /*isCompressConv=*/false)) {
        wtScale = scaleConst;
        wtBias = biasConst;
    } else {
        weightsTable = createWeightsTable(loc, scaleConst, biasConst, poolC, fp32ElemType, inOrder, rewriter);
    }

    auto maxPoolOp = rewriter.create<VPU::NCEMaxPoolOp>(
            appendLoc(loc, "_convert_maxpool"), poolOutType,
            /*reduceXyMax=*/nullptr, /*reduceXyMin=*/nullptr, /*reduceTensorMinMax=*/nullptr, reshapedIn, weightsTable,
            wtScale, wtBias, getIntArrayAttr(ctx, ArrayRef<int64_t>{1, 1}),
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

    mlir::Value weightsTable = nullptr;
    mlir::Value wtScale = nullptr;
    mlir::Value wtBias = nullptr;
    if (VPU::MPEEngineConfig::useNewWeightTableFormat(origOp.getOperation(), /*isCompressConv=*/false)) {
        wtScale = scale;
        wtBias = bias;
    } else {
        weightsTable = createWeightsTable(origOpLoc, scale, bias, weightsTableC, fp32ElemType, inOrder, rewriter);
    }

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
    auto newMaxPoolOp = rewriter.create<VPU::NCEMaxPoolOp>(
            origOp->getLoc(), origOp.getOutput().getType(),
            /*reduceXyMax */ nullptr,
            /*reduceXyMin */ nullptr,
            /*reduceTensorMinMax */ nullptr, origOp.getInput(), weightsTable, wtScale, wtBias,
            getIntArrayAttr(ctx, maxPoolKernels), getIntArrayAttr(ctx, maxPoolStrides), padAttr, ppeAttr, mpeEngineAttr,
            nullptr,
            /*MultiClusterStrategyAttr=*/nullptr, /*maxPoolOp.getOutputPaddingAttr()=*/nullptr,
            /*maxPoolOp.getInputPaddingAttr()=*/nullptr, /*axesValue=*/nullptr);
    rewriter.replaceOp(origOp, newMaxPoolOp.getOutput());

    return mlir::success();
}

// Returns true when val ultimately comes from a compile-time constant, walking through
// any unary passthrough op that preserves constant-ness (type casts, layout permutations,
// shape changes). Handles chains like:
//   Const(FP16) -> ConvertOp(FP32) -> ShapeCastOp([1,C,1,1]) -> Add
//   Const(INT8)  -> QuantizeCastOp  -> Add
//   Const(NCHW)  -> MemPermuteOp(NHWC) -> Add
//   Const        -> AffineReshapeOp -> ConvertOp -> Add
bool isConstantBias(mlir::Value val) {
    while (auto* op = val.getDefiningOp()) {
        // Leaf: any op with the ConstantLike trait (includes Const::DeclareOp and future constant ops).
        if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
            return true;
        }
        // Passthrough: single-operand ops that preserve constant-ness.
        //   VPU::isPureViewOp: covers ViewLikeOpInterface ops (ShapeCastOp, AffineReshapeOp,
        //     QuantizeCastOp, ...) -- automatically includes future ops that register the interface.
        //   VPU::ConvertOp: element-type cast, not ViewLike but constant-preserving.
        //   VPU::MemPermuteOp: layout permutation, not ViewLike but constant-preserving.
        if (op->getNumOperands() == 1 && (VPU::isPureViewOp(op) || mlir::isa<VPU::ConvertOp, VPU::MemPermuteOp>(op))) {
            val = op->getOperand(0);
        } else {
            break;
        }
    }
    return false;
}

//
// RunAddBroadcastOnDPU: folds VPU.Add(feature_NxCxHxW_NHWC, bias_1xCx1x1_NHWC) {post_op}
// into VPU.NCEMaxPool with per-channel weights table (scale=1, bias=conditioning_bias).
// This eliminates the SW broadcast dispatch and intermediate DMA for the conditional
// instance-norm scale+bias additions in AdaIN-style workloads.
//
class RunAddBroadcastOnDPU final : public mlir::OpRewritePattern<VPU::AddOp> {
public:
    RunAddBroadcastOnDPU(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::AddOp>(ctx), _log(std::move(log)) {
        setDebugName("RunAddBroadcastOnDPU");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// This pattern is applied twice in the NPU50XX pipeline (pre- and post-ScfComputeOpsOutlining).
// Re-application is safe because converted ops become NCE.MaxPool, which does not match
// VPU::AddOp, so previously converted broadcast adds are never visited a second time.
mlir::LogicalResult RunAddBroadcastOnDPU::matchAndRewrite(VPU::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    auto origOpLoc = origOp->getLoc();
    _log.trace("Found AddOp '{0}' at '{1}'.", origOp->getName(), origOpLoc);

    auto ctx = origOp.getContext();

    // Identify feature map (H>1 or W>1) and broadcast bias (H==1 && W==1).
    // Use getBoundedShape so that a bias with BoundedTensorType [1,C,1,1]
    // (where H/W are formally dynamic but bounded to 1) is identified correctly.
    auto input1 = origOp.getInput1();
    auto input2 = origOp.getInput2();
    const auto shape1 = getBoundedShape(input1);
    const auto shape2 = getBoundedShape(input2);
    // VPU.AddOp is not rank-restricted; guard against non-4D inputs before indexing
    // with Dims4D::Act::{H,W,...} to avoid out-of-bounds access.
    if (shape1.size() != 4 || shape2.size() != 4) {
        return mlir::failure();
    }

    mlir::Value featureMap, biasVal;
    if (shape1[Dims4D::Act::H] == 1 && shape1[Dims4D::Act::W] == 1) {
        featureMap = input2;
        biasVal = input1;
    } else if (shape2[Dims4D::Act::H] == 1 && shape2[Dims4D::Act::W] == 1) {
        featureMap = input1;
        biasVal = input2;
    } else {
        return mlir::failure();
    }

    // NCEMaxPool requires NHWC, and weights-table construction expects matching NHWC descriptors on bias as well.
    if (DimsOrder::fromValue(featureMap) != DimsOrder::NHWC || DimsOrder::fromValue(biasVal) != DimsOrder::NHWC) {
        return mlir::failure();
    }

    const auto featureShape = getBoundedShape(featureMap);

    // NCEMaxPool requires true 2D spatial data: both H>1 AND W>1.
    // When either spatial dimension is 1 (e.g. 3D MVN output [1,32,N] converted to [1,32,1,N]:
    // H=1, W=N), the tensor is effectively 1D. NCEMaxPool DPU dispatch overhead for 1D sequential
    // data outweighs the broadcast benefit regardless of total area.
    const int64_t H = featureShape[Dims4D::Act::H];
    const int64_t W = featureShape[Dims4D::Act::W];
    if (H == 1 || W == 1) {
        return mlir::failure();
    }

    // Require minimum spatial area for DPU broadcast to beat SHAVE dispatch overhead.
    // Below this threshold, SHAVE dispatch overhead dominates and DPU adds latency.
    constexpr int64_t MIN_BROADCAST_SPATIAL = 256;
    if (H * W < MIN_BROADCAST_SPATIAL) {
        return mlir::failure();
    }

    // Only apply DPU conversion when the bias is a dynamic value (not a compile-time constant).
    // Static Conv/BN/GroupNorm bias adds (bias = Const, possibly via Convert/ShapeCast) should
    // stay as SHAVE: their featureMap comes from a SHAVE op, so the SHAVE->DPU transition
    // incurs a DDR roundtrip for every Add.
    // Dynamic conditioning biases (AdaIN pattern) come from DPU MVN1Normalize output,
    // so the all-DPU chain keeps data CMX-resident throughout.
    if (isConstantBias(biasVal)) {
        return mlir::failure();
    }

    const auto biasShape = getBoundedShape(biasVal);
    const int64_t C = featureShape[Dims4D::Act::C];

    // Channel alignment and bias shape sanity checks
    if (C % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT != 0) {
        return mlir::failure();
    }
    if (biasShape[Dims4D::Act::N] != 1 || biasShape[Dims4D::Act::C] != C) {
        return mlir::failure();
    }

    // NCE.MaxPool only accepts f16/bf16 inputs; reject f32 to avoid creating invalid IR.
    const auto featureElemType = mlir::cast<NDTypeInterface>(featureMap.getType()).getElementType();
    if (!mlir::isa<mlir::Float16Type, mlir::BFloat16Type>(featureElemType)) {
        return mlir::failure();
    }

    // VPU::AddOp does not implement IE::LayerWithPostOpInterface, so retrievePPEAttribute
    // cannot preserve any post_op activation. Bail out rather than silently drop it.
    // Checked before any IR is created below, since a failed match must not leave dead ops behind.
    if (origOp.getPostOpAttr() != nullptr) {
        _log.trace("RunAddBroadcastOnDPU: AddOp has post_op that cannot be preserved on NCEMaxPool, skipping");
        return mlir::failure();
    }

    // Look ahead: if the Add's sole consumer is VPU::LeakyReluOp we can fuse its activation
    // into the NCEMaxPool PPE (mode=LPRELU), avoiding a separate SHAVE dispatch.
    // Must be checked here (before IR creation) so a failed look-ahead never leaves dead ops.
    VPU::LeakyReluOp downstreamLRelu = nullptr;
    if (origOp.getOutput().hasOneUse()) {
        downstreamLRelu = mlir::dyn_cast<VPU::LeakyReluOp>(*origOp.getOutput().user_begin());
    }

    const auto inOrder = DimsOrder::NHWC;
    auto fp32ElemType = mlir::Float32Type::get(ctx);

    // scale = 1.0, shape [C, 1, 1, 1] fp32
    const auto scaleShape = Shape({C, 1, 1, 1});
    const auto scaleType = mlir::RankedTensorType::get(scaleShape.raw(), fp32ElemType);
    const auto scaleBaseAttr = Const::createConstContent(scaleType, ArrayRef({1.0f}));
    const auto scaleContentAttr = Const::ContentAttr::get(scaleBaseAttr).transform().reorder(inOrder).get();
    auto scale = rewriter.create<Const::DeclareOp>(appendLoc(origOpLoc, "_add_scale"), scaleContentAttr.getType(),
                                                   std::move(scaleContentAttr))
                         .getOutput();

    // bias: convert to fp32 via an identity NCE.MaxPool (convertToFp32ViaIdentityMaxPool reshapes
    // to [1,C',H,W] to avoid the degenerate [C,1,1,1] shape) rather than a SW Convert, since a
    // DPU passthrough op is faster than dispatching a SHAVE kernel for this conversion.
    mlir::Value bias = convertToFp32ViaIdentityMaxPool(origOpLoc, biasVal, origOp.getOperation(), rewriter);

    mlir::Value weightsTable = nullptr;
    mlir::Value wtScale = nullptr;
    mlir::Value wtBias = nullptr;
    if (VPU::MPEEngineConfig::useNewWeightTableFormat(origOp.getOperation(), /*isCompressConv=*/false)) {
        wtScale = scale;
        wtBias = bias;
    } else {
        weightsTable = createWeightsTable(origOpLoc, scale, bias, C, fp32ElemType, inOrder, rewriter);
    }

    // Build the PPE attribute.  When we found a downstream LeakyRelu, override the mode to
    // LPRELU so the activation runs inside the DPU rather than as a separate SHAVE dispatch.
    VPU::PPEAttr ppeAttr;
    if (downstreamLRelu != nullptr) {
        const auto basePpeAttr = mlir::cast<VPU::PPEFpAttr>(VPU::getPpeConfig(ctx).retrievePPEAttribute(origOp));
        const auto lpreluMode = VPU::PPEModeAttr::get(ctx, VPU::PPEMode::LPRELU);
        const double negativeSlope = downstreamLRelu.getNegativeSlopeAttr().getValueAsDouble();
        ppeAttr = VPU::PPEFpAttr::get(ctx, lpreluMode, basePpeAttr.getClampLow(), basePpeAttr.getClampHigh(),
                                      basePpeAttr.getScale(), getFPArrayAttr(ctx, SmallVector<double>{negativeSlope}),
                                      basePpeAttr.getBias(), basePpeAttr.getAdder(), basePpeAttr.getIn1Mult(),
                                      basePpeAttr.getIn2Mult(), basePpeAttr.getSprlut());
    } else {
        ppeAttr = VPU::getPpeConfig(ctx).retrievePPEAttribute(origOp);
    }

    VPU::MPEEngineAttr mpeEngineModeAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        mpeEngineModeAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngine());
    }

    const auto maxPoolKernels = getIntArrayAttr(ctx, ArrayRef<int64_t>{1, 1});
    const auto maxPoolStrides = getIntArrayAttr(ctx, ArrayRef<int64_t>{1, 1});
    const auto zeroPad = getIntArrayAttr(ctx, ArrayRef<int64_t>{0, 0});
    auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(zeroPad, zeroPad));

    auto newMaxPool = rewriter.create<VPU::NCEMaxPoolOp>(
            origOp->getLoc(), origOp.getOutput().getType(),
            /*reduceXyMax=*/nullptr, /*reduceXyMin=*/nullptr, /*reduceTensorMinMax=*/nullptr, featureMap, weightsTable,
            wtScale, wtBias, maxPoolKernels, maxPoolStrides, padAttr, ppeAttr, mpeEngineModeAttr,
            /*multiClusterStrategy=*/nullptr, /*output_padding=*/nullptr,
            /*input_padding=*/nullptr, /*s2dd2s_config=*/nullptr, /*axesValue=*/nullptr);

    if (downstreamLRelu != nullptr) {
        rewriter.replaceAllUsesWith(downstreamLRelu.getOutput(), newMaxPool.getOutput());
        rewriter.eraseOp(downstreamLRelu);
    }
    rewriter.replaceOp(origOp, newMaxPool.getOutput());
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
    patterns.add<RunAddBroadcastOnDPU>(&ctx, _log);
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
