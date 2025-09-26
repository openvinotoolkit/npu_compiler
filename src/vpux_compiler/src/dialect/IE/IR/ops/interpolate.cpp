//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::InterpolateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::InterpolateOpAdaptor interpolate(operands, attrs, prop);
    if (mlir::failed(interpolate.verify(loc))) {
        return mlir::failure();
    }

    auto outShape = IE::calcOutputShapes(interpolate, loc, Logger::global(), ctx);

    const auto inType = mlir::cast<mlir::ShapedType>(interpolate.getInput().getType());
    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

namespace {

//
// ConvertInputsToAttr
//

class ConvertInputsToAttr final : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    using mlir::OpRewritePattern<IE::InterpolateOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp interpolateOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertInputsToAttr::matchAndRewrite(IE::InterpolateOp interpolateOp,
                                                         mlir::PatternRewriter& rewriter) const {
    if (interpolateOp.getSizesAttr().has_value() || interpolateOp.getScalesAttr().has_value() ||
        interpolateOp.getAxesAttr().has_value()) {
        return mlir::failure();
    }
    const auto loc = interpolateOp.getLoc();

    // Infer sizes, scales and axes from input, output and pads.
    const auto inShape = getShape(interpolateOp.getInput()).raw();
    const auto outShape = getShape(interpolateOp.getOutput()).raw();

    const auto extractPads = [&](const mlir::ArrayAttr padsAttr) {
        const auto pads = IE::extractIntVector(loc, nullptr, padsAttr);
        if (mlir::failed(pads) || pads.value().size() != outShape.size()) {
            return SmallVector<int64_t>(outShape.size(), 0);
        }
        return pads.value();
    };
    const SmallVector<int64_t> padsBeginVal = extractPads(interpolateOp.getAttr().getPadsBegin());
    const SmallVector<int64_t> padsEndVal = extractPads(interpolateOp.getAttr().getPadsEnd());

    SmallVector<int64_t> sizesVal;
    SmallVector<double> scalesVal;
    SmallVector<int64_t> axesVal;

    for (size_t i = 0; i < inShape.size(); i++) {
        const auto paddedInDim = inShape[i] + padsBeginVal[i] + padsEndVal[i];
        if (paddedInDim != outShape[i]) {
            sizesVal.push_back(outShape[i]);
            scalesVal.push_back(static_cast<double>(outShape[i]) / static_cast<double>(paddedInDim));
            axesVal.push_back(static_cast<int64_t>(i));
        }
    }

    const auto sizesAttr = getIntArrayAttr(interpolateOp.getContext(), sizesVal);
    const auto scalesAttr = getFPArrayAttr(interpolateOp.getContext(), scalesVal);
    const auto axesAttr = getIntArrayAttr(interpolateOp.getContext(), axesVal);

    // Convert `shape_calculation_mode` from `Scales` to `Sizes`
    // After Scales input converted to Scales FPArrayAttr, the original Scale precision will become FP64.
    // It is possible to calculate the wrong output size, if the original Scale precision is not FP64.
    auto interpolateAttr = interpolateOp.getAttr();
    const auto calcModeAttr = interpolateAttr.getShapeCalcMode();
    if (calcModeAttr != nullptr && calcModeAttr.getValue() == IE::InterpolateCalcMode::SCALES) {
        const auto newCalcModeAttr =
                IE::InterpolateCalcModeAttr::get(interpolateOp.getContext(), IE::InterpolateCalcMode::SIZES);
        interpolateAttr = IE::InterpolateAttr::get(
                interpolateOp.getContext(), interpolateAttr.getMode(), newCalcModeAttr, interpolateAttr.getCoordMode(),
                interpolateAttr.getNearestMode(), interpolateAttr.getAntialias(), interpolateAttr.getPadsBegin(),
                interpolateAttr.getPadsEnd(), interpolateAttr.getCubeCoeff());
    }

    // rewrite layer
    rewriter.replaceOpWithNewOp<IE::InterpolateOp>(
            interpolateOp, interpolateOp.getInput(), nullptr, nullptr, nullptr, sizesAttr, scalesAttr, axesAttr,
            interpolateOp.getTileOffsetAttrAttr(), interpolateOp.getInitialInputDimsAttrAttr(),
            interpolateOp.getInitialOutputDimsAttrAttr(), interpolateAttr, interpolateOp.getOutputPaddingAttr(),
            interpolateOp.getInputPaddingAttr());

    return mlir::success();
}

class ConvertInputToFP16 final : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    using mlir::OpRewritePattern<IE::InterpolateOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp Op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertInputToFP16::matchAndRewrite(IE::InterpolateOp op, mlir::PatternRewriter& rewriter) const {
    const auto inputType = mlir::cast<mlir::ShapedType>(op.getInput().getType()).getElementType();
    const auto arch = config::getArch(op);

    // VPU4000-M2I does not support C-minor FP16
    if (arch >= config::ArchKind::NPU40XX && (config::getCompilationMode(op) != config::CompilationMode::ReferenceSW)) {
        return mlir::failure();
    }

    if (inputType.isUnsignedInteger(8)) {
        auto convertOpBefore =
                rewriter.create<IE::ConvertOp>(op.getLoc(), op.getInput(), mlir::Float16Type::get(getContext()));
        auto interpolateOp = rewriter.create<IE::InterpolateOp>(
                op.getLoc(), convertOpBefore.getOutput(), op.getSizes(), op.getScales(), op.getAxes(),
                op.getSizesAttrAttr(), op.getScalesAttrAttr(), op.getAxesAttrAttr(), op.getTileOffsetAttrAttr(),
                op.getInitialInputDimsAttrAttr(), op.getInitialOutputDimsAttrAttr(), op.getAttr(),
                op.getOutputPaddingAttr(), op.getInputPaddingAttr());

        rewriter.replaceOpWithNewOp<IE::ConvertOp>(op, interpolateOp.getOutput(), inputType);
        return mlir::success();
    }

    return mlir::failure();
}

class ConvertToNearest final : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    using mlir::OpRewritePattern<IE::InterpolateOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp Op, mlir::PatternRewriter& rewriter) const final;
};

// Convert any type of Interpolate that is like BroadCast axes to NEAREST Interpolate and with ASYMMETRIC CoordMode
// For example: inShape: 1x16x1x1, outShape: 1x16x32x32, broadcast at H and W dimensions
// This kind of NEAREST Interpolate will further optimization at `ConvertNearestToBroadcastOrStridedConcatPass`
//  - Convert to NCEInterpolateOp that can be executed using the Storage Element hardware feature
//  - Convert to BroadCastOp and further convert to TileOp, finally to PerAxisTileDMAOp
mlir::LogicalResult ConvertToNearest::matchAndRewrite(IE::InterpolateOp op, mlir::PatternRewriter& rewriter) const {
    auto* ctx = op->getContext();

    if (!IE::isBroadCastInterpolate(op) && !IE::isEquivalentToNearestAsymmetricInterpolate(op)) {
        return mlir::failure();
    }

    const auto originalAttr = op.getAttr();
    if (originalAttr.getMode().getValue() == IE::InterpolateMode::NEAREST &&
        originalAttr.getCoordMode().getValue() == IE::InterpolateCoordMode::ASYMMETRIC) {
        return mlir::failure();
    }

    auto nearestMode = originalAttr.getNearestMode();

    if (IE::isEquivalentToNearestAsymmetricInterpolate(op)) {
        nearestMode = IE::InterpolateNearestModeAttr::get(ctx, IE::InterpolateNearestMode::FLOOR);
    }

    const auto newInterpolateAttr = IE::InterpolateAttr::get(
            ctx, IE::InterpolateModeAttr::get(ctx, IE::InterpolateMode::NEAREST), originalAttr.getShapeCalcMode(),
            IE::InterpolateCoordModeAttr::get(ctx, IE::InterpolateCoordMode::ASYMMETRIC), nearestMode,
            originalAttr.getAntialias(), originalAttr.getPadsBegin(), originalAttr.getPadsEnd(),
            originalAttr.getCubeCoeff());

    rewriter.replaceOpWithNewOp<IE::InterpolateOp>(
            op, op.getInput(), op.getSizes(), op.getScales(), op.getAxes(), op.getSizesAttrAttr(),
            op.getScalesAttrAttr(), op.getAxesAttrAttr(), op.getTileOffsetAttrAttr(), op.getInitialInputDimsAttrAttr(),
            op.getInitialOutputDimsAttrAttr(), newInterpolateAttr, op.getOutputPaddingAttr(), op.getInputPaddingAttr());

    return mlir::success();
}

}  // namespace

//
// fold
//

mlir::OpFoldResult vpux::IE::InterpolateOp::fold(FoldAdaptor adaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }
    const auto ctx = getContext();
    auto operands = adaptor.getOperands();
    // If the input is all const, fold interp into const
    if (const auto cst = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        const auto inType = mlir::cast<vpux::NDTypeInterface>(cst.getType());
        const auto inOrder = inType.getDimsOrder();
        // only support NCHW cst input
        if (inOrder != DimsOrder::NCHW) {
            return nullptr;
        }
        // only support CUBIC mode + HALF_PIXEL coord mode
        const auto interpAttr = getAttr();
        if (interpAttr.getMode().getValue() != IE::InterpolateMode::CUBIC ||
            interpAttr.getCoordMode().getValue() != IE::InterpolateCoordMode::HALF_PIXEL) {
            return nullptr;
        }
        // Infer sizes and axes from input, output and pads.
        const auto inShape = getShape(getInput()).raw();
        const auto outShape = getShape(getOutput()).raw();

        const auto extractPads = [&](const mlir::ArrayAttr padsAttr) {
            const auto pads = IE::extractIntVector(getLoc(), nullptr, padsAttr);
            if (mlir::failed(pads) || pads.value().size() != outShape.size()) {
                return SmallVector<int64_t>(outShape.size(), 0);
            }
            return pads.value();
        };
        const SmallVector<int64_t> padsBeginVal = extractPads(interpAttr.getPadsBegin());
        const SmallVector<int64_t> padsEndVal = extractPads(interpAttr.getPadsEnd());

        SmallVector<int64_t> sizesVal;
        SmallVector<int64_t> axesVal;
        for (size_t i = 0; i < inShape.size(); i++) {
            const auto paddedInDim = inShape[i] + padsBeginVal[i] + padsEndVal[i];
            if (paddedInDim != outShape[i]) {
                sizesVal.push_back(outShape[i]);
                axesVal.push_back(static_cast<int64_t>(i));
            }
        }
        const auto sizesAttr = getIntArrayAttr(ctx, sizesVal);
        const auto axesAttr = getIntArrayAttr(ctx, axesVal);

        return static_cast<Const::ContentAttr>(cst)
                .transform()
                .interpolate(axesAttr, sizesAttr,
                             /*mode=*/mlir::StringAttr::get(ctx, "CUBIC"),
                             /*coordMode=*/mlir::StringAttr::get(ctx, "HALF_PIXEL"),
                             /*nearestMode=*/mlir::StringAttr::get(ctx, "FLOOR"),
                             /*antialias=*/interpAttr.getAntialias(),
                             /*padsBegin=*/interpAttr.getPadsBegin(),
                             /*padsEnd=*/interpAttr.getPadsEnd(),
                             /*cubeCoeff=*/interpAttr.getCubeCoeff())
                .get();
    }

    return nullptr;
}

//
// getCanonicalizationPatterns
//

void vpux::IE::InterpolateOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.add<ConvertInputsToAttr>(context);
    patterns.add<ConvertInputToFP16>(context);
    patterns.add<ConvertToNearest>(context);
}
