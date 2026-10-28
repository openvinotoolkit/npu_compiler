//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

namespace vpux::IE {
#define GEN_PASS_DECL_BROADCASTINPUTFORMULTIPLY
#define GEN_PASS_DEF_BROADCASTINPUTFORMULTIPLY
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

const uint32_t levelCount = 2;
const SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

//
// ConvertPerChannelMultiplyToScaleShift
//
// Runs after FuseBroadCastRewriter restores the small scale tensor.
// At that point the Multiply has:
//   activation: large tensor matching the output shape  (either input)
//   scale:      per-channel [1,C,1,1] tensor            (either input)
//
// ConvertToScaleShiftPass only handles compile-time const scale; this rewriter
// also handles runtime (network-parameter) scale, and explicitly checks both
// operand orderings (scale may be at input1 or input2).
// The resulting ScaleShift is converted to NCE GroupConvolution by
// ConvertScaleShiftToDWPass, which is added to buildFinalTransformationPipeline
// after BroadcastInputForMultiply.
//

class ConvertPerChannelMultiplyToScaleShift final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    ConvertPerChannelMultiplyToScaleShift(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefit), _log(log) {
        setDebugName("ConvertPerChannelMultiplyToScaleShift");
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp mulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertPerChannelMultiplyToScaleShift::matchAndRewrite(IE::MultiplyOp mulOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), mulOp->getName(), mulOp->getLoc());

    // GroupConvolution (DW Conv) supports F16 only.
    const auto elemType = mlir::cast<mlir::ShapedType>(mulOp.getInput1().getType()).getElementType();
    if (!elemType.isF16()) {
        return mlir::failure();
    }

    if (mulOp.getOutputPaddingAttr() != nullptr || mulOp.getInputPaddingAttr() != nullptr) {
        return mlir::failure();
    }

    const auto mulOutShape = getShape(mulOp.getOutput());
    const auto lhsShape = getShape(mulOp.getInput1());
    const auto rhsShape = getShape(mulOp.getInput2());

    if (mulOutShape.size() != 4) {
        return mlir::failure();
    }

    // Exactly one input must match the output shape (activation); the other is the scale.
    // Check both orderings.
    mlir::Value activationInput, scaleInput;
    if (lhsShape == mulOutShape && rhsShape != mulOutShape) {
        activationInput = mulOp.getInput1();
        scaleInput = mulOp.getInput2();
    } else if (rhsShape == mulOutShape && lhsShape != mulOutShape) {
        activationInput = mulOp.getInput2();
        scaleInput = mulOp.getInput1();
    } else {
        return mlir::failure();
    }

    const auto scaleShape = getShape(scaleInput);
    if (scaleShape.size() != 4) {
        return mlir::failure();
    }
    if (!mlir::cast<vpux::NDTypeInterface>(scaleInput.getType()).getShape().isStatic()) {
        return mlir::failure();
    }

    auto* ctx = mulOp.getContext();
    const auto outC = mulOutShape[Dims4D::Act::C];
    const auto scaleC = scaleShape[Dims4D::Act::C];
    const auto scaleW = scaleShape[Dims4D::Act::W];
    const auto outW = mulOutShape[Dims4D::Act::W];

    // Reject a fully-splat scale [1,1,1,1] — it has no per-channel information.
    if (scaleShape.totalSize() == 1) {
        return mlir::failure();
    }

    // The future DW Conv channel count must satisfy the HW alignment requirement.
    const auto hwAlignment = VPU::NCEInvariant::getAlignment(elemType);

    // Case 1: C-expanded scale [1,C,1,1] — direct DW Conv without layout change.
    if (scaleC == outC && scaleShape[Dims4D::Act::N] == 1 && scaleShape[Dims4D::Act::H] == 1 &&
        scaleShape[Dims4D::Act::W] == 1) {
        if (scaleC % hwAlignment != 0) {
            return mlir::failure();
        }
        // NCE accepts NHWC operands only
        if (DimsOrder::fromValue(activationInput) != DimsOrder::NHWC ||
            DimsOrder::fromValue(scaleInput) != DimsOrder::NHWC) {
            return mlir::failure();
        }
        _log.nest().trace("Case 1: C-expanded scale → direct DW GroupConvolution");
        const SmallVector<int64_t> filterShape = {(scaleC == 1 ? outC : scaleC), 1, 1, 1};
        auto filter =
                rewriter.createOrFold<IE::ShapeCastOp>(mulOp->getLoc(), scaleInput, getIntArrayAttr(ctx, filterShape));
        const SmallVector<int32_t> s = {1, 1}, p = {0, 0}, d = {1, 1};
        rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
                mulOp, mulOp.getType(), activationInput, filter, /*bias=*/nullptr, getIntArrayAttr(ctx, s),
                getIntArrayAttr(ctx, p), getIntArrayAttr(ctx, p), getIntArrayAttr(ctx, d),
                getIntAttr(ctx, filterShape[0]), mulOp.getPostOpAttr(), mulOp.getClampAttr(), nullptr, nullptr);
        return mlir::success();
    }

    // Case 2: W-expanded scale [1,1,1,W] where W = activation's W.
    // Apply PermuteCast (NCHW physical → NHWC logical) to both tensors, making W→C,
    // then create DW Conv in NHWC, then PermuteCast the output back.
    // Require activation H == 1 so the PermuteCast-based layout change is safe
    // (activation [1,C,1,W] NCHW → [1,W,C,1] NHWC with no stride/padding issues).
    if (scaleShape[Dims4D::Act::N] == 1 && scaleShape[Dims4D::Act::C] == 1 && scaleShape[Dims4D::Act::H] == 1 &&
        scaleW == outW && mulOutShape[Dims4D::Act::H] == 1) {
        if (outW % hwAlignment != 0) {
            return mlir::failure();
        }
        // The PermuteCasts below reinterpret the operands' memory assuming they are laid out
        // in canonical NCHW order. If they are not, the reinterpretation silently produces a
        // wrongly-shaped tensor, which later yields an invalid GroupConvolution.
        if (DimsOrder::fromValue(activationInput) != DimsOrder::NCHW ||
            DimsOrder::fromValue(scaleInput) != DimsOrder::NCHW) {
            return mlir::failure();
        }
        _log.nest().trace("Case 2: W-expanded scale → PermuteCast + DW GroupConvolution");

        // PermuteCast reinterprets NCHW physical memory as NHWC logical for the activation:
        //   activation [1,N_seq,1,C_hidden] NCHW → [1,C_hidden,N_seq,1] NHWC
        const auto nhwcOrderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
        const auto identMemPerm = mlir::AffineMapAttr::get(mlir::AffineMap::getMultiDimIdentityMap(4, ctx));

        auto actNhwc = rewriter.createOrFold<IE::PermuteCastOp>(appendLoc(mulOp->getLoc(), "act_to_nhwc"),
                                                                activationInput, nhwcOrderAttr, identMemPerm);

        // The scale [1,1,1,C_hidden] is essentially 1-D. A single PermuteCastOp with
        // mem_perm=(d3,d0,d1,d2) moves W=C_hidden to physical position 0, giving
        // [C_hidden,1,1,1] NHWC in one zero-copy step (all other dims are 1).
        const int64_t groups = outW;
        const auto filterMemPerm =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{3, 0, 1, 2}, ctx));
        auto filter = rewriter.createOrFold<IE::PermuteCastOp>(appendLoc(mulOp->getLoc(), "filter_to_nhwc"), scaleInput,
                                                               nhwcOrderAttr, filterMemPerm);

        // Output type: same shape and layout as the NHWC-cast activation.
        const auto outNhwcType = mlir::cast<vpux::NDTypeInterface>(actNhwc.getType());
        const int64_t seqLen = outNhwcType.getShape()[Dims4D::Act::H];

        // Balance the H×W spatial dims for efficient NCE tiling.
        // Find the largest power of 2 ≤ 8 that evenly divides seqLen.
        int64_t spatialW = 1;
        for (int64_t w = 8; w >= 2; w /= 2) {
            if (seqLen % w == 0) {
                spatialW = w;
                break;
            }
        }
        const int64_t spatialH = seqLen / spatialW;
        const bool needReshape = (spatialW > 1);

        // Before DW Conv: AffineReshape [1,C,seqLen,1] → [1,C,spatialH,spatialW]
        // dim_mapping (indexed by input): [[N],[C],[H,W],[W]] — splits H into HxW.
        const SmallVector<SmallVector<int64_t>> splitDimMapping{{Dims4D::Act::N.ind()},
                                                                {Dims4D::Act::C.ind()},
                                                                {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()},
                                                                {Dims4D::Act::W.ind()}};
        mlir::Value actConv = actNhwc;
        mlir::Type convOutType = outNhwcType;
        if (needReshape) {
            const Shape splitShape{1, groups, spatialH, spatialW};
            convOutType = outNhwcType.changeShape(splitShape);
            actConv = IE::AffineReshapeOp::create(rewriter, appendLoc(mulOp->getLoc(), "pre_conv_reshape"), actNhwc,
                                                  getIntArrayOfArray(ctx, splitDimMapping),
                                                  getIntArrayAttr(ctx, splitShape))
                              .getOutput();
        }

        const SmallVector<int32_t> s = {1, 1}, p = {0, 0}, d = {1, 1};
        auto dwConv = IE::GroupConvolutionOp::create(
                rewriter, mulOp->getLoc(), mlir::cast<mlir::Type>(convOutType), actConv, filter, /*bias=*/nullptr,
                getIntArrayAttr(ctx, s), getIntArrayAttr(ctx, p), getIntArrayAttr(ctx, p), getIntArrayAttr(ctx, d),
                getIntAttr(ctx, groups), mulOp.getPostOpAttr(), mulOp.getClampAttr(), nullptr, nullptr);

        // After DW Conv: AffineReshape [1,C,spatialH,spatialW] → [1,C,seqLen,1] (same dim_mapping, inverse shape).
        mlir::Value dwConvOut = dwConv.getOutput();
        if (needReshape) {
            const Shape interShape{1, groups, seqLen, 1};
            dwConvOut = IE::AffineReshapeOp::create(rewriter, appendLoc(mulOp->getLoc(), "post_conv_reshape"),
                                                    dwConvOut, getIntArrayOfArray(ctx, splitDimMapping),
                                                    getIntArrayAttr(ctx, interShape))
                                .getOutput();
        }

        // PermuteCast output back to original NCHW layout.
        const auto origOrderAttr = mlir::AffineMapAttr::get(
                mlir::cast<vpux::NDTypeInterface>(mulOp.getType()).getDimsOrder().toAffineMap(ctx));
        auto outNchw = rewriter.createOrFold<IE::PermuteCastOp>(appendLoc(mulOp->getLoc(), "out_to_nchw"), dwConvOut,
                                                                origOrderAttr, identMemPerm);

        rewriter.replaceOp(mulOp, outNchw);
        return mlir::success();
    }

    return mlir::failure();
}

class TileBroadcastInputRewriter final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    TileBroadcastInputRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefit), _log(log) {
        setDebugName("TileBroadcastInputRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value tileInput(mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx, mlir::Location loc,
                          mlir::Value broadcastInput, ShapeRef targetShape) const;
    mlir::Value castToDimsOrder(mlir::Value value, mlir::AffineMap dstOrderMap, mlir::AffineMap memOrderMap,
                                StringRef suffix, mlir::PatternRewriter& rewriter) const;
    bool isBroadcastEfficient(NDTypeInterface inputType, NDTypeInterface outputType) const;

    Logger _log;
};

mlir::Value TileBroadcastInputRewriter::tileInput(mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx,
                                                  mlir::Location loc, mlir::Value broadcastInput,
                                                  ShapeRef targetShape) const {
    // Build the Tile directly on the input's own shape/order, the same way ConvertBroadcastToTile
    // computes repeats for an IE::BroadcastOp(broadcastInput, targetShape). This avoids inserting
    // a Broadcast here that would need to be converted to a Tile later by a separate pass, and
    // avoids unnecessary PermuteCasts around the Tile.
    const auto inputShape = getShape(broadcastInput);
    SmallVector<int64_t> repeats(targetShape.size());
    for (size_t i = 0; i < targetShape.size(); ++i) {
        VPUX_THROW_UNLESS(inputShape[Dim(i)] != 0 && targetShape[Dim(i)] % inputShape[Dim(i)] == 0,
                          "Target shape {0} is not a multiple of input shape {1} at dim {2}", targetShape, inputShape,
                          i);
        repeats[i] = targetShape[Dim(i)] / inputShape[Dim(i)];
    }
    const auto repeatsAttr = getIntArrayAttr(ctx, repeats);
    const auto tileOutputType = mlir::cast<NDTypeInterface>(broadcastInput.getType()).changeShape(targetShape);
    auto tile = IE::TileOp::create(rewriter, appendLoc(loc, "tile"), tileOutputType, broadcastInput, repeatsAttr);
    return tile.getOutput();
}

mlir::Value TileBroadcastInputRewriter::castToDimsOrder(mlir::Value value, mlir::AffineMap dstOrderMap,
                                                        mlir::AffineMap memOrderMap, StringRef suffix,
                                                        mlir::PatternRewriter& rewriter) const {
    return rewriter.createOrFold<IE::PermuteCastOp>(appendLoc(value.getLoc(), suffix), value, dstOrderMap, memOrderMap);
}

bool TileBroadcastInputRewriter::isBroadcastEfficient(NDTypeInterface inputType, NDTypeInterface outputType) const {
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();
    auto diffDims = IE::getDiffInOutSizeDims(inputShape, outputShape);

    // If more than one dimension differs, broadcasting is not efficient
    if (diffDims.size() > 1) {
        _log.trace("  isBroadcastEfficient: More than one dim differs - not efficient");
        return false;
    }

    // No need to broadcast if shapes are the same
    if (diffDims.empty()) {
        _log.trace("  isBroadcastEfficient: Shapes are the same - efficient");
        return true;
    }

    auto diffDim = diffDims.front();
    auto inputOrder = inputType.getDimsOrder();
    auto innermostNonOneDim = getInnermostNonTrivialDim(inputShape, inputOrder);
    // If the input is all ones and the shapes differ in at most one dimension,
    // broadcasting is treated as efficient.
    if (!innermostNonOneDim.has_value()) {
        _log.trace("  isBroadcastEfficient: All ones shape - efficient");
        return true;
    }

    // Legality gate: only expand 1 -> N on the differing dimension.
    if (!(inputShape[diffDim] == 1 && outputShape[diffDim] > 1)) {
        _log.trace("  isBroadcastEfficient: Not 1->N expansion at diffDim");
        return false;
    }

    // broadcasting over large spatial extents is not efficient, as it can lead to performance regressions in some
    // models.
    int64_t spatialSize = 1;
    for (const auto& dim : {Dims4D::Act::N, Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W}) {
        if (dim == diffDim) {
            continue;
        }
        spatialSize *= outputShape[dim];
    }
    constexpr int64_t kMaxSpatialSizeForBroadcast =
            4192;  // experimental value based on regressions observed in E#227477 & E#215801
    if (spatialSize > kMaxSpatialSizeForBroadcast) {
        _log.trace("  isBroadcastEfficient: Broadcast over large spatial extent ({0}) - not efficient", spatialSize);
        return false;
    }

    auto innermostNonOneDimPos = inputOrder.dimPos(innermostNonOneDim.value());
    auto broadcastDimPos = inputOrder.dimPos(diffDim);

    // Ensure broadcast does not target a lower-priority dim than the innermost non-trivial one.
    return broadcastDimPos <= innermostNonOneDimPos;
}

mlir::LogicalResult TileBroadcastInputRewriter::matchAndRewrite(IE::MultiplyOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    const auto ctx = origOp->getContext();
    const auto loc = origOp->getLoc();

    mlir::Value lhsInput = origOp.getInput1();
    const auto lhsShape = getShape(lhsInput);

    mlir::Value rhsInput = origOp.getInput2();
    const auto rhsShape = getShape(rhsInput);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto outputShape = outputType.getShape();

    // Dynamic shape multiply is handled by dedicated dynamic-shape pipelines.
    // This rewrite relies on static shapes (alignment checks and totalSize()) and can throw on '?'.
    if (lhsShape.isDynamic() || rhsShape.isDynamic() || outputShape.isDynamic()) {
        _log.trace("Skip broadcast rewrite for dynamic shapes");
        return mlir::failure();
    }

    if (lhsShape.size() != 4 || rhsShape.size() != 4 || outputShape.size() != 4) {
        _log.trace("Only support 4D tensor, but got {0}D and {1}D", lhsShape.size(), rhsShape.size());
        return mlir::failure();
    }

    if (lhsShape == outputShape && rhsShape == outputShape) {
        _log.trace("Inputs have same shape, no need for broadcast");
        return mlir::failure();
    }

    // Cast lhs & rhs input to NHWC layout
    const auto dstOrder = DimsOrder::NHWC;
    const auto dstOrderMap = dstOrder.toAffineMap(ctx);
    const auto dimsOrder = outputType.getDimsOrder();

    auto alignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());
    const auto doesInputNeedBroadCast = [&](mlir::Value input) {
        return getShape(input) != outputShape;
    };

    // Workaround: skip rewrite when the output C dimension is very large (regressions observed on some models).
    // On models with regression, broadcast never happens on the C dimension, and C dim is considerably larger
    // than the rest. Threshold chosen based on output C dim of the multiplies that would show regressions (e.g. 1024,
    // 35840, 201088). E#215801
    constexpr int64_t kRegressionChannelThreshold = 1024;  // last updated E#228711
    if (outputShape[Dims4D::Act::C] >= kRegressionChannelThreshold) {
        _log.trace("Skip broadcast rewrite: output C dimension is too large: {0}", outputShape[Dims4D::Act::C]);
        return mlir::failure();
    }

    // Support multiply when its inner most dim is aligned
    // We do not limit the broadcast check to HighestNonTrivialDim, since improvements have been observed in many models
    // that broadcasted on dimensions other than HighestNonTrivialDim.
    auto getMemMapOfInnermostMultiply = [&]() -> std::optional<SmallVector<mlir::AffineMap>> {
        auto getInnermostDim = [](const vpux::DimsOrder& order) {
            return order.toDim(MemDim(order.numDims() - 1));
        };

        auto innerMostDim = getInnermostDim(dimsOrder);

        auto hasLhsNotAligned = lhsShape[innerMostDim] % alignment != 0;
        auto hasRhsNotAligned = rhsShape[innerMostDim] % alignment != 0;

        if (hasLhsNotAligned || hasRhsNotAligned) {
            _log.trace("Innermost dim size is not aligned to {0}, Innermost: {1}", alignment, innerMostDim);
            return std::nullopt;
        }

        if (!isBroadcastEfficient(lhsInput.getType(), outputType) ||
            !isBroadcastEfficient(rhsInput.getType(), outputType)) {
            _log.trace("It is not efficient to broadcast the non-innermost dim");
            return std::nullopt;
        }

        SmallVector<mlir::AffineMap> memOrderMap;
        const auto inMemOrderMap = mlir::AffineMap::getMultiDimIdentityMap(dstOrder.numDims(), ctx);
        memOrderMap.push_back(inMemOrderMap);
        const auto outMemOrderMap = mlir::AffineMap::getMultiDimIdentityMap(dimsOrder.numDims(), ctx);
        memOrderMap.push_back(outMemOrderMap);
        return memOrderMap;
    };

    // Support multiply when it has splat input
    // Like input1([1, 1, 1, 1]) x input2([1, 512, 1, 1])
    auto getMemMapOfSplatMultiply = [&]() -> std::optional<SmallVector<mlir::AffineMap>> {
        auto lhsTotalSize = lhsShape.totalSize();
        auto rhsTotalSize = rhsShape.totalSize();

        auto alignDim = Dims4D::Act::C;
        auto hasSplatValue = (lhsTotalSize == 1 && rhsTotalSize == rhsShape[alignDim]) ||
                                             (rhsTotalSize == 1 && lhsTotalSize == lhsShape[alignDim])
                                     ? true
                                     : false;
        if (!hasSplatValue) {
            _log.trace("Multiply has no splat or aligned value");
            return std::nullopt;
        }

        SmallVector<mlir::AffineMap> memOrderMap;
        const auto inMemOrderMap = getPermutationFromOrders(DimsOrder::fromValue(lhsInput), dstOrder, ctx);
        memOrderMap.push_back(inMemOrderMap);
        const auto outMemOrderMap = getPermutationFromOrders(dstOrder, DimsOrder::fromValue(origOp.getOutput()), ctx);
        memOrderMap.push_back(outMemOrderMap);
        return memOrderMap;
    };

    SmallVector<mlir::AffineMap> memOrderMap;
    if (getMemMapOfInnermostMultiply().has_value()) {
        memOrderMap = getMemMapOfInnermostMultiply().value();
    } else if (getMemMapOfSplatMultiply().has_value()) {
        memOrderMap = getMemMapOfSplatMultiply().value();
    } else {
        return mlir::failure();
    }

    // Tile input (in order to avoid converting the Broadcast to a Tile op later)
    if (doesInputNeedBroadCast(lhsInput)) {
        lhsInput = tileInput(rewriter, ctx, appendLoc(loc, "broadcast_lhs"), origOp.getInput1(), outputShape);
    }
    auto newLhsInput = castToDimsOrder(lhsInput, dstOrderMap, memOrderMap[0], "_lhs_permute_cast", rewriter);

    if (doesInputNeedBroadCast(rhsInput)) {
        rhsInput = tileInput(rewriter, ctx, appendLoc(loc, "broadcast_rhs"), origOp.getInput2(), outputShape);
    }
    auto newRhsInput = castToDimsOrder(rhsInput, dstOrderMap, memOrderMap[0], "_rhs_permute_cast", rewriter);

    // Create new Multiply
    auto newOutputType = outputType.changeDimsOrder(dstOrder).changeShape(getShape(newRhsInput));
    auto newMultiplyOp = IE::MultiplyOp::create(
            rewriter, origOp.getLoc(), newOutputType, newLhsInput, newRhsInput, origOp.getAutoBroadcastAttr(),
            origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
    // Cast to the original dims order
    auto result = castToDimsOrder(newMultiplyOp.getOutput(), dimsOrder.toAffineMap(ctx), memOrderMap[1],
                                  "_output_permute_cast", rewriter);
    rewriter.replaceOp(origOp, result);
    return mlir::success();
}

//
// FuseBroadCastRewriter
//
// Matches the pattern: Tile (broadcast) -> optional(PermuteCast) -> Multiply
// and fuses the Tile into the Multiply by replacing the Tile output with the
// Tile input, so the broadcast is performed implicitly by NUMPY Multiply rather
// than by an explicit Tile op. If a PermuteCast sits between the Tile and the
// Multiply, a new PermuteCast on the un-tiled input is inserted instead.

class FuseBroadCastRewriter final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    FuseBroadCastRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefit), _log(log) {
        setDebugName("FuseBroadCastRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp multiplyOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isBroadcastShape(ShapeRef inShape, ShapeRef outShape) const;
    Shape inferShapeThroughPermuteCast(MemShapeRef inMemShape, const mlir::AffineMap memPerm,
                                       const DimsOrder& dstOrder) const;

    Logger _log;
};

bool FuseBroadCastRewriter::isBroadcastShape(ShapeRef inShape, ShapeRef outShape) const {
    if (inShape.size() != outShape.size()) {
        return false;
    }

    for (size_t i = 0; i < inShape.size(); i++) {
        const auto inDim = inShape[Dim(i)];
        const auto outDim = outShape[Dim(i)];
        // The dimension is either the same or broadcasted from 1
        if (inDim != 1 && inDim != outDim) {
            return false;
        }
    }
    return true;
}

Shape FuseBroadCastRewriter::inferShapeThroughPermuteCast(MemShapeRef inMemShape, const mlir::AffineMap memPerm,
                                                          const DimsOrder& dstOrder) const {
    const auto newInMemShape = applyPerm(inMemShape.toValues(), memPerm);
    return dstOrder.toLogicalOrder(newInMemShape);
}

mlir::LogicalResult FuseBroadCastRewriter::matchAndRewrite(IE::MultiplyOp multiplyOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), multiplyOp->getName(), multiplyOp->getLoc());

    if (multiplyOp.getAutoBroadcast() != IE::AutoBroadcastType::NUMPY) {
        _log.trace("The auto_broadcast is not NUMPY — skipping");
        return mlir::failure();
    }

    if (multiplyOp.getOutputPaddingAttr() || multiplyOp.getInputPaddingAttr()) {
        return mlir::failure();
    }

    // Find the following pattern:
    //   Tile (broadcast) -> optional(PermuteCast) -> Multiply
    IE::PermuteCastOp permuteCastOp;
    IE::TileOp tileOp;
    for (auto* defOp : {multiplyOp.getInput1().getDefiningOp(), multiplyOp.getInput2().getDefiningOp()}) {
        auto permuteCandidate = mlir::dyn_cast_if_present<IE::PermuteCastOp>(defOp);
        auto tileCandidate = mlir::dyn_cast_if_present<IE::TileOp>(defOp);
        if (!permuteCandidate && !tileCandidate) {
            continue;
        }
        if (permuteCandidate) {
            tileCandidate = mlir::dyn_cast_if_present<IE::TileOp>(permuteCandidate.getInput().getDefiningOp());
        }
        if (!tileCandidate || !tileCandidate->hasOneUse()) {
            continue;
        }
        if (!isBroadcastShape(getShape(tileCandidate.getInput()), getShape(tileCandidate.getOutput()))) {
            continue;
        }
        permuteCastOp = permuteCandidate;
        tileOp = tileCandidate;
        break;
    }

    if (!tileOp) {
        _log.trace("No broadcast op found");
        return mlir::failure();
    }

    auto tileInput = tileOp.getInput();
    const auto tileInType = mlir::cast<NDTypeInterface>(tileInput.getType());
    Shape newInShape(tileInType.getShape().raw());
    if (permuteCastOp != nullptr) {
        // Pattern: Tile -> PermuteCast -> Multiply
        // The PermuteCast reorders dimensions in memory, so we can't use the Tile's
        // logical shapes directly. Instead, project both the Tile input and output
        // memory shapes through the same permutation to obtain their logical shapes
        // as seen by the Multiply operand (i.e. after the PermuteCast).
        const auto tileInMemShape = tileInType.getMemShape();
        const auto tileOutMemShape = mlir::cast<NDTypeInterface>(tileOp.getOutput().getType()).getMemShape();
        const auto permuteDstOrder = DimsOrder::fromAffineMap(permuteCastOp.getDstOrder());
        const auto permuteMemPerm = permuteCastOp.getMemPerm();
        newInShape = inferShapeThroughPermuteCast(tileInMemShape, permuteMemPerm, permuteDstOrder);
        auto newOutShape = inferShapeThroughPermuteCast(tileOutMemShape, permuteMemPerm, permuteDstOrder);
        // Verify the projected shapes still form a valid broadcast pair (each dim is
        // either equal or the input dim is 1), otherwise the fusion is unsafe.
        if (!isBroadcastShape(newInShape, newOutShape)) {
            return mlir::failure();
        }

        // Sanity-check: the projected output shape must match the actual PermuteCast
        // output shape; a mismatch indicates an unexpected layout transformation.
        if (newOutShape != getShape(permuteCastOp.getOutput())) {
            return mlir::failure();
        }
    }

    // For dims where the fused (small) PermuteCast output has size 1, the other Multiply input
    // must cover that dim fully (i.e. equal the output shape), so NUMPY broadcast produces the
    // correct output shape.
    const auto isTileOnInput1 = permuteCastOp != nullptr ? (multiplyOp.getInput1() == permuteCastOp.getOutput())
                                                         : (multiplyOp.getInput1() == tileOp.getOutput());
    const auto otherInput = isTileOnInput1 ? multiplyOp.getInput2() : multiplyOp.getInput1();
    const auto otherInputShape = getShape(otherInput);
    const auto outputShape = getShape(multiplyOp.getOutput());
    const auto rank = outputShape.size();

    if (newInShape.size() != rank || otherInputShape.size() != rank) {
        return mlir::failure();
    }
    for (size_t i = 0; i < rank; ++i) {
        if (newInShape[Dim(i)] == 1 && otherInputShape[Dim(i)] != outputShape[Dim(i)]) {
            return mlir::failure();
        }
    }

    const auto multiplyLoc = multiplyOp->getLoc();
    auto newInput = tileInput;
    if (permuteCastOp != nullptr) {
        auto newPermuteCast =
                IE::PermuteCastOp::create(rewriter, takeOpLoc(permuteCastOp, "permute_in"), newInput,
                                          permuteCastOp.getDstOrderAttr(), permuteCastOp.getMemPermAttr());
        newInput = newPermuteCast.getOutput();
    }

    const auto multiplyOutputType = mlir::cast<vpux::NDTypeInterface>(multiplyOp.getOutput().getType());
    const auto newInput1 = isTileOnInput1 ? newInput : multiplyOp.getInput1();
    const auto newInput2 = isTileOnInput1 ? multiplyOp.getInput2() : newInput;
    auto newMultiplyOp = IE::MultiplyOp::create(rewriter, takeOpLoc(multiplyOp, "fuse_broadcast"), multiplyOutputType,
                                                newInput1, newInput2, multiplyOp.getAutoBroadcastAttr(),
                                                multiplyOp.getPostOpAttr(), multiplyOp.getClampAttr(),
                                                multiplyOp.getOutputPaddingAttr(), multiplyOp.getInputPaddingAttr());
    rewriter.replaceOp(multiplyOp, newMultiplyOp.getOutput());

    _log.trace("Fuse broadcast into multiply for better performance at '{0}'", multiplyLoc);
    return mlir::success();
}

//
// BroadcastInputForMultiplyPass
//
class BroadcastInputForMultiplyPass final :
        public IE::impl::BroadcastInputForMultiplyBase<BroadcastInputForMultiplyPass> {
public:
    explicit BroadcastInputForMultiplyPass(bool broadcastInputForMultiply, Logger log)
            : _enableBroadcastInputForMultiply(broadcastInputForMultiply) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableBroadcastInputForMultiply;
};

mlir::LogicalResult BroadcastInputForMultiplyPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (broadcastInputForMultiply.hasValue()) {
        _enableBroadcastInputForMultiply = broadcastInputForMultiply.getValue();
    }

    return mlir::success();
}

void BroadcastInputForMultiplyPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    // Phase 1: Always apply FuseBroadCastRewriter
    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<FuseBroadCastRewriter>(&ctx, benefitLevels[0], _log);

        if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
            signalPassFailure();
            return;
        }
    }

    // Phase 1.5: After FuseBroadCastRewriter the Tile is gone and the Multiply has
    // scale=[1,C,1,1] (small) vs activation (large).  ConvertPerChannelMultiplyToScaleShift
    // converts it to ScaleShift; ConvertScaleShiftToDWPass (added to the pipeline after
    // this pass) then converts ScaleShift to NCE GroupConvolution.
    // Runs unconditionally (like Phase 1) — independent of broadcastInputForMultiply flag.
    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<ConvertPerChannelMultiplyToScaleShift>(&ctx, benefitLevels[0], _log);
        if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
            signalPassFailure();
            return;
        }
    }

    // Phase 2 & 3: Apply TileBroadcastInputRewriter and canonicalizer only when enabled
    if (_enableBroadcastInputForMultiply) {
        // Phase 2: Apply TileBroadcastInputRewriter
        bool broadcastRewriteOccurred = false;
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<TileBroadcastInputRewriter>(&ctx, benefitLevels[1], _log);

        if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig(),
                                                     &broadcastRewriteOccurred))) {
            signalPassFailure();
            return;
        }
        // Phase 3: Apply canonicalizer if any broadcast rewrite occurred
        if (broadcastRewriteOccurred) {
            mlir::RewritePatternSet canonPatterns(&ctx);
            IE::PermuteCastOp::getCanonicalizationPatterns(canonPatterns, &ctx);
            if (mlir::failed(
                        mlir::applyPatternsGreedily(func, std::move(canonPatterns), getDefaultGreedyRewriteConfig()))) {
                signalPassFailure();
                return;
            }

            mlir::OpPassManager dynamicPM(func.getOperationName());
            dynamicPM.addPass(mlir::createCSEPass());
            if (mlir::failed(runPipeline(dynamicPM, func))) {
                signalPassFailure();
                return;
            }
        }
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createBroadcastInputForMultiplyPass(bool broadcastInputForMultiply, Logger log) {
    return std::make_unique<BroadcastInputForMultiplyPass>(broadcastInputForMultiply, log);
}
