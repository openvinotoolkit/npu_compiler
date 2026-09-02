//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::InterpolateOp::verifyShapeInfo() {
    return vpux::IE::verifyInputIs4D(getInput());
}

mlir::LogicalResult vpux::IE::InterpolateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto logger = Logger::global().nest("ie-interpolate-dynamism", 0);
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::InterpolateOpAdaptor interpolate(operands, attrs, prop);
    if (mlir::failed(interpolate.verify(loc))) {
        return mlir::failure();
    }

    if (IE::isSizesAsParameter(interpolate.getSizes(), interpolate.getSizesAttr())) {
        logger.trace("Shape with parameter sizes is unsupported now.");
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(interpolate.getInput().getType());

    // Scale-as-parameter path: output shape is inferred from bounded scales independently.
    if (IE::isScalesAsParameter(interpolate.getScales(), interpolate.getScalesAttr())) {
        const auto inShape = getBoundedShape(interpolate.getInput());
        const auto axesVal = IE::getInterpAxesVal(loc, interpolate.getAxes(), interpolate.getAxesAttr(), inputType);
        const auto beginPads = IE::extractIntVector(loc, nullptr, interpolate.getAttr().getPadsBegin());
        const auto endPads = IE::extractIntVector(loc, nullptr, interpolate.getAttr().getPadsEnd());

        // Scales are runtime — use bounded scales for compile-time shape inference
        const auto scaleType = mlir::cast<vpux::NDTypeInterface>(interpolate.getScales().getType());
        VPUX_THROW_WHEN(scaleType.getNumElements() != int64_t(axesVal.size()),
                        "Scales input number of elements must match the number of interpolated axes, got {0} and {1}",
                        scaleType.getNumElements(), axesVal.size());
        // Two trailing spatial axes (H, W) are resized; reject IR that provides fewer axes.
        if (axesVal.size() < 2) {
            return errorAt(loc, "Interpolate with scales as parameter requires at least 2 axes, got {0}",
                           axesVal.size());
        }
        auto scalesBound = SmallVector<double>(axesVal.size(), 1.0);
        const auto spatialScalesBound = getInterpolateScalesBound(inputType);
        scalesBound[scalesBound.size() - 1] = spatialScalesBound;
        scalesBound[scalesBound.size() - 2] = spatialScalesBound;

        const auto scalesElemType =
                mlir::cast<vpux::NDTypeInterface>(interpolate.getScales().getType()).getElementType();

        const auto outShapeVec =
                IE::inferInterpOutShape(loc, axesVal, inShape, beginPads, endPads, IE::InterpolateCalcMode::SCALES,
                                        mlir::FailureOr<ArrayRef<int64_t>>(mlir::failure()),
                                        ArrayRef<double>(scalesBound), scalesElemType, Logger::global());

        const auto isVariableAxis = [&](size_t dimIdx) {
            auto axisIt = llvm::find(axesVal, static_cast<int64_t>(dimIdx));
            if (axisIt == axesVal.end()) {
                return false;
            }
            const auto scaleIdx = axisIt - axesVal.begin();
            return scalesBound[scaleIdx] != 1.0;
        };

        // Interpolated axes with bounded scale != 1.0 are treated as dynamic.
        // Axes with bounded scale == 1.0 preserve the input static/dynamic characteristics
        auto [outStaticShape, outBounds, outDimMask] = callOnShapeOf(inputType, [&](const auto& inShapeRepr) {
            using ShapeT = std::decay_t<decltype(inShapeRepr)>;
            if constexpr (std::is_same_v<ShapeT, BoundedShape>) {
                BoundedShape bounded;
                bounded.reserve(outShapeVec.size());
                for (size_t i = 0; i < outShapeVec.size(); ++i) {
                    if (isVariableAxis(i) || inShapeRepr[Dim(i)].isDynamic()) {
                        bounded.push_back(BoundedDim(mlir::ShapedType::kDynamic, outShapeVec[i]));
                    } else {
                        bounded.push_back(BoundedDim(outShapeVec[i]));
                    }
                }
                return splitShapeAndRepresentation(bounded);
            } else if constexpr (std::is_same_v<ShapeT, DimsMaskedShape>) {
                DimsMaskedShape masked;
                masked.reserve(outShapeVec.size());
                for (size_t i = 0; i < outShapeVec.size(); ++i) {
                    masked.push_back(MaskedDim(outShapeVec[i], isVariableAxis(i) || inShapeRepr[Dim(i)].isDynamic()));
                }
                return splitShapeAndRepresentation(masked);
            } else {
                BoundedShape bounded;
                bounded.reserve(outShapeVec.size());
                for (size_t i = 0; i < outShapeVec.size(); ++i) {
                    if (isVariableAxis(i)) {
                        bounded.push_back(BoundedDim(mlir::ShapedType::kDynamic, outShapeVec[i]));
                    } else {
                        bounded.push_back(BoundedDim(outShapeVec[i]));
                    }
                }
                return splitShapeAndRepresentation(bounded);
            }
        });

        SmallVector<int64_t> outShape(outStaticShape.begin(), outStaticShape.end());
        const auto outDesc =
                vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace(), outBounds, outDimMask);
        inferredReturnShapes.emplace_back(outShape, inputType.getElementType(), outDesc);
        return mlir::success();
    }

    // Normal path: output shape from sizes/scales attributes.
    auto outShapeVec = IE::calcOutputShapes(interpolate, loc, Logger::global(), ctx);

    auto [outStaticShape, outBounds, outDimMask] = callOnShapeOf(inputType, [&](const auto& inShape) {
        using ShapeT = std::decay_t<decltype(inShape)>;
        if constexpr (std::is_same_v<ShapeT, BoundedShape>) {
            // For bounded tensors, propagate dynamic dims with computed bounds
            BoundedShape outBounded;
            outBounded.reserve(outShapeVec.size());
            for (size_t i = 0; i < outShapeVec.size(); ++i) {
                if (inShape[Dim(i)].isDynamic()) {
                    outBounded.push_back(BoundedDim(mlir::ShapedType::kDynamic, outShapeVec[i]));
                } else {
                    outBounded.push_back(BoundedDim(outShapeVec[i]));
                }
            }
            return splitShapeAndRepresentation(outBounded);
        } else if constexpr (std::is_same_v<ShapeT, DimsMaskedShape>) {
            DimsMaskedShape outMasked;
            outMasked.reserve(outShapeVec.size());
            for (size_t i = 0; i < outShapeVec.size(); ++i) {
                outMasked.push_back(MaskedDim(outShapeVec[i], inShape[Dim(i)].isDynamic()));
            }
            return splitShapeAndRepresentation(outMasked);
        } else {
            Shape outShape(outShapeVec);
            return splitShapeAndRepresentation(outShape);
        }
    });

    SmallVector<int64_t> outShape(outStaticShape.begin(), outStaticShape.end());
    const auto outDesc =
            vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace(), outBounds, outDimMask);
    inferredReturnShapes.emplace_back(outShape, inputType.getElementType(), outDesc);

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
    auto logger = Logger::global().nest("ie-interpolate-dynamism-ConvertInputsToAttr", 1);
    if (interpolateOp.getSizesAttr().has_value() || interpolateOp.getScalesAttr().has_value() ||
        interpolateOp.getAxesAttr().has_value()) {
        return mlir::failure();
    }

    const auto loc = interpolateOp.getLoc();
    if (IE::isScalesAsParameter(interpolateOp.getScales(), interpolateOp.getScalesAttr())) {
        logger.trace("Scales is parameter, skipping conversion.");
        auto inType = mlir::cast<vpux::NDTypeInterface>(interpolateOp.getInput().getType());
        auto axes = IE::getInterpAxesVal(loc, interpolateOp.getAxes(), interpolateOp.getAxesAttr(), inType);

        logger.trace("Extracted axes: {0}", axes);
        auto axesAttr = getIntArrayAttr(interpolateOp.getContext(), axes);
        rewriter.replaceOpWithNewOp<IE::InterpolateOp>(
                interpolateOp, interpolateOp.getInput(), nullptr, interpolateOp.getScales(), nullptr,
                getIntArrayAttr(interpolateOp.getContext(), SmallVector<int64_t>()), nullptr, axesAttr,
                interpolateOp.getTileOffsetAttrAttr(), interpolateOp.getInitialInputDimsAttrAttr(),
                interpolateOp.getInitialOutputDimsAttrAttr(), interpolateOp.getAttr(),
                interpolateOp.getOutputPaddingAttr(), interpolateOp.getInputPaddingAttr());

        return mlir::success();
    }

    // Infer sizes, scales and axes from input, output and pads.
    const auto inShape = getBoundedShape(interpolateOp.getInput()).raw();
    const auto outShape = getBoundedShape(interpolateOp.getOutput()).raw();

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

    // For SCALES mode, preserve original scales (don't use derived output/input ratio).
    SmallVector<double> finalScalesVal = std::move(scalesVal);  // Default: derived scales for SIZES mode

    const auto calcModeAttr = interpolateOp.getAttr().getShapeCalcMode();
    if (calcModeAttr != nullptr && calcModeAttr.getValue() == IE::InterpolateCalcMode::SCALES) {
        // Extract original scales from input operand
        const auto originalScalesResult =
                IE::extractFPVector(loc, interpolateOp.getScales(), interpolateOp.getScalesAttr());
        if (mlir::succeeded(originalScalesResult)) {
            const auto& originalScales = originalScalesResult.value();

            if (originalScales.size() == axesVal.size()) {
                // Spatially-indexed scales (per-axis order)
                finalScalesVal = SmallVector<double>(originalScales.begin(), originalScales.end());
            } else if (originalScales.size() == static_cast<size_t>(inShape.size())) {
                // Full-rank scales tensor: extract only the axes that changed
                finalScalesVal.clear();
                for (size_t k = 0; k < axesVal.size(); k++) {
                    finalScalesVal.push_back(originalScales[axesVal[k]]);
                }
            }
        }
    }

    const auto sizesAttr = getIntArrayAttr(interpolateOp.getContext(), sizesVal);
    const auto scalesAttr = getFPArrayAttr(interpolateOp.getContext(), finalScalesVal);
    const auto axesAttr = getIntArrayAttr(interpolateOp.getContext(), axesVal);

    // Per OpenVino IR spec, preserve original calc_mode.
    // If calc_mode=SCALES: downstream passes will use scalesAttr directly (per spec semantics)
    // If calc_mode=SIZES: downstream passes will derive scales from sizes (per spec semantics)
    auto interpolateAttr = interpolateOp.getAttr();
    // Keep calcModeAttr unchanged—do NOT convert SCALES to SIZES

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
// ReifyRankedShapedTypeOpInterface
//

mlir::LogicalResult vpux::IE::InterpolateOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                               mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();
    const auto outputShapedType = mlir::cast<mlir::ShapedType>(getOutput().getType());
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto axesVal = IE::getInterpAxesVal(loc, getAxes(), getAxesAttrAttr(), inputType);

    auto scales = getScales();
    if (scales != nullptr) {
        if (auto convertOp = scales.getDefiningOp<IE::ConvertOp>()) {
            scales = convertOp.getInput();
        }
    }

    return reifyInterpolateResultShape(builder, loc, getInput(), scales, getScalesAttr(), axesVal, outputShapedType,
                                       reifiedReturnShapes);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::InterpolateOp::fold(FoldAdaptor adaptor) {
    if (getInput().getType() == getOutput().getType() && !getShape(getInput()).isDynamic()) {
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
// verify
//

mlir::LogicalResult vpux::IE::InterpolateOp::verify() {
    if (!IE::isScalesAsParameter(getScales(), getScalesAttr())) {
        return mlir::success();
    }

    // Scale-as-parameter path: only LINEAR and LINEAR_ONNX modes are supported.
    const auto mode = getAttr().getMode().getValue();
    if (mode != IE::InterpolateMode::LINEAR && mode != IE::InterpolateMode::LINEAR_ONNX) {
        return errorAt(*this,
                       "Interpolate with scales as parameter only supports LINEAR and LINEAR_ONNX modes, got {0}",
                       getAttr().getMode());
    }

    // Only 4D input is supported.
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    if (inputType.getRank() != 4) {
        return errorAt(*this, "Interpolate with scales as parameter only supports 4D input tensors, got rank {0}",
                       inputType.getRank());
    }

    // Scales must be rank 1 to match the scales-as-parameter shape inference path.
    const auto scaleType = mlir::cast<vpux::NDTypeInterface>(getScales().getType());
    if (scaleType.getRank() != 1) {
        return errorAt(*this, "Scales input must be rank 1, got shape {0}", scaleType.getShape());
    }

    // At least 2 interpolation axes are required.
    if (getAxesAttr().has_value()) {
        const auto axesSize = getAxesAttr()->size();
        if (axesSize < 2) {
            return errorAt(*this,
                           "Interpolate with scales as parameter requires at least 2 interpolation axes, got {0}",
                           axesSize);
        }
    }

    // Only constant axes input is supported before canonicalization folds axes into axes_attr.
    if (getAxes() != nullptr && getAxes().getDefiningOp<Const::DeclareOp>() == nullptr) {
        auto* defOp = getAxes().getDefiningOp();
        return errorAt(*this, "Only constant axes input is supported, got {0}",
                       defOp ? mlir::StringAttr::get(getContext(), defOp->getName().getStringRef())
                             : mlir::StringAttr::get(getContext(), "block argument"));
    }

    return mlir::success();
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
