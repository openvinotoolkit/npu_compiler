//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<SmallVector<int64_t>> extractIntVector(mlir::Location loc, const mlir::Value value,
                                                       const std::optional<mlir::ArrayAttr>& attr) {
    if (attr.has_value() && attr.value() != nullptr) {
        return parseIntArrayAttr<int64_t>(attr.value());
    } else if (value != nullptr) {
        auto valueConst = value.getDefiningOp<Const::DeclareOp>();
        if (valueConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for interpolate attribute");
        }

        const auto valueContent = valueConst.getContent();
        return to_small_vector(valueContent.getValues<int64_t>());
    }
    return errorAt(loc, "Parameter were not provided");
}

mlir::Value createPadding(mlir::PatternRewriter& rewriter, IE::InterpolateOp origOp, mlir::Value input, Dim axis,
                          int64_t forwardPad, int64_t backpad, StringRef locPrefix) {
    auto inputShape = getShape(input);
    auto forwardOffsets = SmallVector<int64_t>(inputShape.size(), 0);
    auto backOffsets = SmallVector<int64_t>(inputShape.size(), 0);

    auto forwardSizes = SmallVector<int64_t>(inputShape.begin(), inputShape.end());
    auto backSizes = SmallVector<int64_t>(inputShape.begin(), inputShape.end());

    if (forwardPad < 0) {
        forwardSizes[axis.ind()] = forwardSizes[axis.ind()] + forwardPad;
        forwardOffsets[axis.ind()] = -forwardPad;
    } else {
        forwardSizes[axis.ind()] = 1;
        forwardOffsets[axis.ind()] = 0;
    }

    backOffsets[axis.ind()] = inputShape[axis] - 1;
    backSizes[axis.ind()] = 1;
    auto forwardSubSlice =
            rewriter.create<IE::SliceOp>(takeOpLoc(origOp, StringLiteral("{0}_forward_slice"), locPrefix), input,
                                         getIntArrayAttr(origOp.getContext(), forwardOffsets),
                                         getIntArrayAttr(origOp.getContext(), forwardSizes))
                    .getResult();
    auto backSubSlice = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, StringLiteral("{0}_backward_slice"), locPrefix),
                                                     input, getIntArrayAttr(origOp.getContext(), backOffsets),
                                                     getIntArrayAttr(origOp.getContext(), backSizes))
                                .getResult();

    SmallVector<mlir::Value> subSlices;
    if (forwardPad == 0) {
        subSlices.push_back(input);
    } else if (forwardPad < 0) {
        subSlices.push_back(forwardSubSlice);
    } else {
        subSlices.push_back(input);
        subSlices.insert(subSlices.begin(), forwardPad, forwardSubSlice);
    }
    if (backpad != 0) {
        subSlices.insert(subSlices.end(), backpad, backSubSlice);
    }
    return rewriter
            .create<IE::ConcatOp>(takeOpLoc(origOp, StringLiteral("{0}_pad_slice_concat"), locPrefix), subSlices, axis)
            .getOutput();
}

mlir::FailureOr<SmallVector<double>> extractFPVector(mlir::Location loc, const mlir::Value value,
                                                     const std::optional<mlir::ArrayAttr>& attr) {
    if (attr.has_value() && attr.value() != nullptr) {
        return parseFPArrayAttr<double>(attr.value());
    } else if (value != nullptr) {
        auto valueConst = value.getDefiningOp<Const::DeclareOp>();

        if (valueConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for interpolate attribute");
        }

        const auto valueContent = valueConst.getContent();
        return to_small_vector(valueContent.getValues<double>());
    }
    return mlir::failure();
}

SmallVector<int64_t> getInterpAxesVal(mlir::Location loc, const mlir::Value axes,
                                      const std::optional<mlir::ArrayAttr>& attr, NDTypeInterface inType) {
    const bool isExplicitAxes = (axes != nullptr) || (attr.has_value() && attr.value() != nullptr);
    if (isExplicitAxes) {
        auto explicitAxes = extractIntVector(loc, axes, attr);
        VPUX_THROW_UNLESS(mlir::succeeded(explicitAxes), "Cannot get Interpolate Axes value");
        return explicitAxes.value();
    }

    // If Interpolate is not explicit Axes, it should be default value [0,...,rank(input) - 1]
    SmallVector<int64_t> axesVal(inType.getRank());
    std::iota(axesVal.begin(), axesVal.end(), 0);
    return axesVal;
}

mlir::FailureOr<int64_t> getInnermostAxis(mlir::Location loc, DimsOrder dimsOrder, ArrayRef<int64_t> axes) {
    if (axes.empty()) {
        return errorAt(loc, "Got empty axes");
    }
    const auto numDims = static_cast<int64_t>(dimsOrder.numDims());
    auto innermostAxis = axes[0];
    for (auto axis : axes) {
        if (axis < 0 || axis >= numDims) {
            return errorAt(loc, "Axis {0} is out of expected range [0, {1}]", axis, numDims - 1);
        }
        if (dimsOrder.dimPos(Dim(innermostAxis)) < dimsOrder.dimPos(Dim(axis))) {
            innermostAxis = axis;
        }
    }
    return innermostAxis;
}

int64_t getInterpCoordinatesSize(mlir::Value output, int64_t innermostAxis) {
    const auto outShape = getShape(output).raw();
    const auto lenghtResized = outShape[innermostAxis];
    return lenghtResized;
}

int64_t getInterpLambdasSize(mlir::Value output, int64_t innermostAxis) {
    const auto outShape = getShape(output).raw();
    const auto lenghtResized = outShape[innermostAxis];
    return lenghtResized * 2;
}

void applyInterpPads(MutableArrayRef<int64_t> outShape, ArrayRef<int64_t> padsBegin, ArrayRef<int64_t> padsEnd) {
    // pads might be zero initialized
    if (padsBegin.size() != padsEnd.size() || padsBegin.size() != outShape.size()) {
        return;
    }
    // naive implementation only apply pads to calculated output shape
    for (auto d : outShape | indexed) {
        d.value() += padsBegin[d.index()] + padsEnd[d.index()];
    }
}

template <typename StorageType>
SmallVector<int64_t> inferOutputShapeWithScalesMode(ShapeRef inputShape, ArrayRef<double> scalesVal,
                                                    ArrayRef<int64_t> axesVal, vpux::Logger log) {
    VPUX_THROW_UNLESS(scalesVal.size() == axesVal.size(),
                      "Num of elements in `Scales` tensor: {0} should be equal to number of indices in `Axes`: {1}",
                      scalesVal.size(), axesVal.size());

    auto outputShape = to_small_vector(inputShape);
    auto scalesIter = scalesVal.begin();
    for (const auto& axis : axesVal) {
        outputShape[axis] = static_cast<int64_t>(
                floor(static_cast<double>(static_cast<StorageType>(*scalesIter++) * inputShape[Dim(axis)])));
        log.trace("Infer Scales mode at axis {0}: {1} -> {2}", axis, inputShape[Dim(axis)], outputShape[axis]);
    }
    return outputShape;
}

SmallVector<int64_t> inferOutputShapeWithSizesMode(ShapeRef inputShape, ArrayRef<int64_t> sizesVal,
                                                   ArrayRef<int64_t> axesVal, vpux::Logger log) {
    VPUX_THROW_UNLESS(sizesVal.size() == axesVal.size(),
                      "Num of elements in `Sizes` tensor: {0} should be equal to number of indices in `Axes`: {1}",
                      sizesVal.size(), axesVal.size());

    auto outputShape = to_small_vector(inputShape);
    auto sizesIter = sizesVal.begin();
    for (const auto& axis : axesVal) {
        outputShape[axis] = *sizesIter++;
        log.trace("Infer Sizes mode at axis {0}: {1} -> {2}", axis, inputShape[Dim(axis)], outputShape[axis]);
    }
    return outputShape;
}

SmallVector<int64_t> inferInterpOutShape(mlir::Location loc, ArrayRef<int64_t> axesVal, ShapeRef origShape,
                                         mlir::FailureOr<ArrayRef<int64_t>> padsBegin,
                                         mlir::FailureOr<ArrayRef<int64_t>> padsEnd,
                                         vpux::IE::InterpolateCalcMode calcMode,
                                         mlir::FailureOr<ArrayRef<int64_t>> sizes,
                                         mlir::FailureOr<ArrayRef<double>> scales, mlir::Type scalesElemType,
                                         vpux::Logger log) {
    log.trace("Interp propagate shape: input = {0}", origShape);
    auto inferedShape = to_small_vector(origShape);

    for (auto axis : axesVal) {
        VPUX_THROW_UNLESS(checked_cast<size_t>(axis) < origShape.size(), "Invalid axis {0} for shape of rank {1}", axis,
                          origShape.size());
    }

    if (calcMode == IE::InterpolateCalcMode::SIZES) {
        VPUX_THROW_UNLESS(mlir::succeeded(sizes), "Cannot get Interpolate Sizes value at {0}", loc);
        const auto sizesVal = sizes.value();

        inferedShape = inferOutputShapeWithSizesMode(origShape, sizesVal, axesVal, log);
    } else if (calcMode == IE::InterpolateCalcMode::SCALES) {
        VPUX_THROW_UNLESS(mlir::succeeded(scales), "Cannot get Interpolate Scales value at {0}", loc);
        const auto scalesVal = scales.value();

        if (scalesElemType.isF16()) {
            inferedShape = inferOutputShapeWithScalesMode<vpux::type::float16>(origShape, scalesVal, axesVal, log);
        } else if (scalesElemType.isF32()) {
            inferedShape = inferOutputShapeWithScalesMode<float>(origShape, scalesVal, axesVal, log);
        } else if (scalesElemType.isF64()) {
            inferedShape = inferOutputShapeWithScalesMode<double>(origShape, scalesVal, axesVal, log);
        } else {
            VPUX_THROW("Unexpected Interpolate `Scale` float data type: {0}", scalesElemType);
        }
    } else {
        VPUX_THROW("Got unsupported shape_calculation_mode: {0} at {1}", calcMode, loc);
    }

    // meaning pads provided in attributes
    if (mlir::succeeded(padsBegin) && mlir::succeeded(padsEnd)) {
        applyInterpPads(inferedShape, padsBegin.value(), padsEnd.value());
    }

    log.trace("Interp propagate shape: output = {0}", inferedShape);

    return inferedShape;
}

bool isEquivalentToNearestAsymmetricInterpolate(IE::InterpolateOp op) {
    const auto originalAttr = op.getAttr();
    if (originalAttr.getMode().getValue() != IE::InterpolateMode::NEAREST) {
        return false;
    }

    const auto inputShape = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getShape();
    const auto outputShape = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType()).getShape();
    if (inputShape.size() != 4 || outputShape.size() != 4) {
        return false;
    }

    if (inputShape[Dims4D::Act::W] * 2 != outputShape[Dims4D::Act::W] ||
        inputShape[Dims4D::Act::H] * 2 != outputShape[Dims4D::Act::H]) {
        return false;
    }

    const auto coordMode = originalAttr.getCoordMode().getValue();

    if (coordMode != IE::InterpolateCoordMode::HALF_PIXEL &&
        coordMode != IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL) {
        return false;
    }

    const auto nearestMode = originalAttr.getNearestMode().getValue();

    return nearestMode == IE::InterpolateNearestMode::ROUND_PREFER_CEIL ||
           nearestMode == IE::InterpolateNearestMode::ROUND_PREFER_FLOOR;
}

bool isBroadCastInterpolate(IE::InterpolateOp op) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto inputShape = inputType.getShape();

    auto padValueIsNull = [](mlir::ArrayAttr pad) {
        if (pad == nullptr) {
            return true;
        }
        auto padValue = parseIntArrayAttr<int64_t>(pad);
        return llvm::all_of(padValue, [](int64_t value) {
            return value == 0;
        });
    };

    if (!padValueIsNull(op.getAttr().getPadsBegin()) || !padValueIsNull(op.getAttr().getPadsEnd())) {
        return false;
    }

    const auto axes = extractIntVector(op.getLoc(), op.getAxes(), op.getAxesAttr());
    VPUX_THROW_UNLESS(mlir::succeeded(axes), "Cannot get axes Attr");
    const auto axesValues = axes.value();
    const auto calcMode = op.getAttr().getShapeCalcMode().getValue();
    if (calcMode == IE::InterpolateCalcMode::SIZES) {
        const auto size = extractIntVector(op.getLoc(), op.getSizes(), op.getSizesAttr());
        VPUX_THROW_UNLESS(mlir::succeeded(axes), "Cannot get size Attr");
        const auto sizeValues = size.value();
        VPUX_THROW_UNLESS(sizeValues.size() == axesValues.size(),
                          "Num of elements sizes tensor tensor: {0} should be equal to number of indices in axes: {1}",
                          sizeValues.size(), axesValues.size());

        const auto shapes = zip(axesValues, sizeValues);
        return llvm::all_of(shapes, [&](const std::tuple<int64_t, int64_t>& dimSize) {
            const auto inDimSize = inputShape[Dim(std::get<0>(dimSize))];
            const auto outDimSize = std::get<1>(dimSize);
            return (inDimSize == outDimSize) || (inDimSize != outDimSize && inDimSize == 1);
        });
    } else if (calcMode == IE::InterpolateCalcMode::SCALES) {
        const auto scales = extractFPVector(op.getLoc(), op.getScales(), op.getScalesAttr());
        VPUX_THROW_UNLESS(mlir::succeeded(scales), "Cannot get scales Attr");
        const auto scalesVal = scales.value();
        VPUX_THROW_UNLESS(scalesVal.size() == axesValues.size(),
                          "Num of elements scales tensor tensor: {0} should be equal to number of indices in axes: {1}",
                          scalesVal.size(), axesValues.size());

        const auto shapes = zip(axesValues, scalesVal);
        return llvm::all_of(shapes, [&](const std::tuple<int64_t, double>& dimSize) {
            const auto inDimSize = inputShape[Dim(std::get<0>(dimSize))];
            const auto scaleSize = std::get<1>(dimSize);
            return isDoubleEqual(scaleSize, 1.0) || (!isDoubleEqual(scaleSize, 1.0) && inDimSize == 1);
        });
    }

    VPUX_THROW("Doesn't support shape_calculation_mode: {0}", calcMode);
}

std::pair<double, double> computeFractionCoefficients(double fraction) {
    return std::pair<double, double>{1.0 - fraction, fraction};
}

//
// Coordinate transformation mode functions
//

// HalfPixel coordintate transformation mode
std::pair<int32_t, double> mapCoordHalfPixel(int32_t x, double scale, int32_t, int32_t) {
    auto fractionX = scale * (x + 0.5) - 0.5;
    auto integerX = static_cast<int32_t>(std::floor(fractionX));
    fractionX -= integerX;
    return std::pair<int32_t, double>{integerX, fractionX};
}

// PytorchHalfPixel coordintate transformation mode
std::pair<int32_t, double> mapCoordPytorchHalfPixel(int32_t x, double scale, int32_t outputSize, int32_t) {
    auto fractionX = scale * (x + 0.5) - 0.5;
    if (outputSize <= 1) {
        fractionX = 0;
    }
    auto integerX = static_cast<int32_t>(std::floor(fractionX));
    fractionX -= integerX;
    return std::pair<int32_t, double>{integerX, fractionX};
}

// Asymmetric coordintate transformation mode
std::pair<int32_t, double> mapCoordAsymmetric(int32_t x, double scale, int32_t, int32_t) {
    auto fractionX = scale * x;
    auto integerX = static_cast<int32_t>(std::floor(fractionX));
    fractionX -= integerX;
    return std::pair<int32_t, double>{integerX, fractionX};
}

// TfHalfPixelForNN coordintate transformation mode
std::pair<int32_t, double> mapCoordTfHalfPixelForNN(int32_t x, double scale, int32_t, int32_t) {
    auto fractionX = scale * (x + 0.5);
    auto integerX = static_cast<int32_t>(std::floor(fractionX));
    fractionX -= integerX;
    return std::pair<int32_t, double>{integerX, fractionX};
}

// AlignCorners coordintate transformation mode
std::pair<int32_t, double> mapCoordAlignCorners(int32_t x, double, int32_t outputSize, int32_t inputSize) {
    double fractionX;
    if (outputSize == 1) {
        fractionX = 0;
    } else {
        fractionX = static_cast<double>(x) * (inputSize - 1) / (outputSize - 1);
    }
    auto integerX = static_cast<int32_t>(std::floor(fractionX));
    fractionX -= integerX;
    return std::pair<int32_t, double>{integerX, fractionX};
}

MapCoordFuncT getMapCoordMethod(InterpolateCoordMode coordMode) {
    if (coordMode == IE::InterpolateCoordMode::HALF_PIXEL) {
        return mapCoordHalfPixel;
    } else if (coordMode == IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL) {
        return mapCoordPytorchHalfPixel;
    } else if (coordMode == IE::InterpolateCoordMode::ASYMMETRIC) {
        return mapCoordAsymmetric;
    } else if (coordMode == IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN) {
        return mapCoordTfHalfPixelForNN;
    } else if (coordMode == IE::InterpolateCoordMode::ALIGN_CORNERS) {
        return mapCoordAlignCorners;
    } else {
        VPUX_THROW("Unsupported coodintate transformation mode: {0}.", coordMode);
    }
}

bool isFusingConvertIntoBilinearInterpolateOnDpuBeneficial([[maybe_unused]] IE::InterpolateOp op,
                                                           [[maybe_unused]] mlir::Type convertOutType) {
    return false;
}

}  // namespace IE
}  // namespace vpux
