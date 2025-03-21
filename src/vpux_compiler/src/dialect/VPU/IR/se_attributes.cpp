//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/se_attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_interpolate_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// SEInterpolateAttr
//

mlir::LogicalResult VPU::SEInterpolateAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                   vpux::VPU::NCEInterpolateModeAttr modeAttr,
                                                   vpux::IE::InterpolateCoordModeAttr coordTransformModeAttr,
                                                   mlir::ArrayAttr scalesAttr,
                                                   vpux::IE::InterpolateNearestModeAttr nearestModeAttr,
                                                   mlir::ArrayAttr /*offsetsAttr*/, mlir::ArrayAttr /*sizesAttr*/,
                                                   mlir::ArrayAttr /*initialInputShapeAttr*/,
                                                   mlir::ArrayAttr /*initialOutputShapeAttr*/) {
    if (modeAttr == nullptr) {
        return printTo(emitError(), "Got NULL 'NCEInterpolateMode' in 'SEInterpolateAttr'");
    }
    if (coordTransformModeAttr == nullptr) {
        return printTo(emitError(), "Got NULL 'NCEInterpolateCoordMode' in 'SEInterpolateAttr'");
    }
    if (scalesAttr == nullptr) {
        return printTo(emitError(), "Got NULL scales in 'SEInterpolateAttr'");
    }
    const auto scales = parseFPArrayAttr<double>(scalesAttr);
    if (scales.size() != 4) {
        return printTo(emitError(), "Got scales with {0} dimensions in 'SEInterpolateAttr'. Expected 4 dimensions",
                       scales.size());
    }
    for (const auto scale : scales) {
        if (!isDoubleEqual(std::floor(scale), scale)) {
            return printTo(emitError(), "'SEInterpolateAttr' supports only integer scale values but got {0}", scale);
        }
    }
    if (nearestModeAttr == nullptr && modeAttr.getValue() == VPU::NCEInterpolateMode::NEAREST) {
        return printTo(emitError(),
                       "Got NULL 'NCEInterpolateNearestMode' in 'SEInterpolateAttr' with interpolate mode NEAREST");
    }

    return mlir::success();
}

//
// SEInterpolateAttr (SEAttrInterface)
//

namespace {

SmallVector<double> extractInterpScales(mlir::ArrayAttr scalesAttr) {
    VPUX_THROW_UNLESS(scalesAttr != nullptr, "SEInterpolateAttr: Missing scales attribute");
    const auto scales = parseFPArrayAttr<double>(scalesAttr);
    VPUX_THROW_UNLESS(scales.size() == 4, "SEInterpolateAttr: scales should have rank 4, but got rank {0}",
                      scales.size());
    const auto scaleN = scales[Dims4D::Act::N.ind()];
    const auto scaleC = scales[Dims4D::Act::C.ind()];
    const auto scaleH = scales[Dims4D::Act::H.ind()];
    const auto scaleW = scales[Dims4D::Act::W.ind()];
    return SmallVector<double>{scaleN, scaleC, scaleH, scaleW};
}

std::tuple<int64_t, int64_t> extractInterpFactors(ArrayRef<int64_t> factors) {
    VPUX_THROW_UNLESS(factors.size() == 2, "SEInterpolateAttr: factors should have rank 2, but got rank {0}",
                      factors.size());
    const auto factorH = factors[VPU::SE_INTERPOLATE_FACTOR_H];
    const auto factorW = factors[VPU::SE_INTERPOLATE_FACTOR_W];
    return {factorH, factorW};
}

std::tuple<int64_t, int64_t, int64_t, int64_t> extractInterpPads(ArrayRef<int64_t> padsBegin,
                                                                 ArrayRef<int64_t> padsEnd) {
    VPUX_THROW_UNLESS(padsBegin.size() == 2, "SEInterpolateAttr: padsBegin should have rank 2, but got rank {0}",
                      padsBegin.size());
    VPUX_THROW_UNLESS(padsEnd.size() == 2, "SEInterpolateAttr: padsEnd should have rank 2, but got rank {0}",
                      padsEnd.size());
    const auto padLeft = padsBegin[Dims4D::PadsBegin::Left.ind()];
    const auto padTop = padsBegin[Dims4D::PadsBegin::Top.ind()];
    const auto padRight = padsEnd[Dims4D::PadsEnd::Right.ind()];
    const auto padBottom = padsEnd[Dims4D::PadsEnd::Bottom.ind()];
    return {padLeft, padTop, padRight, padBottom};
}

std::tuple<int64_t, int64_t> extractInterpKernelSize(ArrayRef<int64_t> kernelSize) {
    VPUX_THROW_UNLESS(kernelSize.size() == 2, "SEInterpolateAttr: kernelSize should have rank 2, but got rank {0}",
                      kernelSize.size());
    const auto kernelY = kernelSize[VPU::SE_INTERPOLATE_KERNEL_Y];
    const auto kernelX = kernelSize[VPU::SE_INTERPOLATE_KERNEL_X];
    return {kernelY, kernelX};
}

std::tuple<int64_t, int64_t> extractInterpStrides(ArrayRef<int64_t> strides) {
    VPUX_THROW_UNLESS(strides.size() == 2, "SEInterpolateAttr: strides should have rank 2, but got rank {0}",
                      strides.size());
    const auto strideY = strides[VPU::SE_INTERPOLATE_STRIDE_Y];
    const auto strideX = strides[VPU::SE_INTERPOLATE_STRIDE_X];
    return {strideY, strideX};
}

// Calculate output shape without tiling info
Shape inferFullOutputShape(ShapeRef inShape, ArrayRef<int64_t> factors, ArrayRef<int64_t> padsBegin,
                           ArrayRef<int64_t> padsEnd) {
    const auto [factorH, factorW] = extractInterpFactors(factors);
    const auto [padLeft, padTop, padRight, padBottom] = extractInterpPads(padsBegin, padsEnd);

    auto inferShapeImpl = [](const int64_t inputSize, const int64_t factor, const int64_t padBefore,
                             const int64_t padAfter) -> int64_t {
        return inputSize * factor + padBefore + padAfter;
    };

    Shape fullOutputShape(inShape.toValues());
    fullOutputShape[Dims4D::Act::H] = inferShapeImpl(inShape[Dims4D::Act::H], factorH, padTop, padBottom);
    fullOutputShape[Dims4D::Act::W] = inferShapeImpl(inShape[Dims4D::Act::W], factorW, padLeft, padRight);

    return fullOutputShape;
}

// Calculate input shape without tiling info
Shape inferFullInputShape(ShapeRef outShape, ArrayRef<int64_t> factors, ArrayRef<int64_t> padsBegin,
                          ArrayRef<int64_t> padsEnd) {
    const auto [factorH, factorW] = extractInterpFactors(factors);
    const auto [padLeft, padTop, padRight, padBottom] = extractInterpPads(padsBegin, padsEnd);

    auto inferShapeImpl = [](const int64_t outputSize, const int64_t factor, const int64_t padBefore,
                             const int64_t padAfter) -> int64_t {
        const auto noPaddingSize = outputSize - padBefore - padAfter;
        VPUX_THROW_UNLESS(noPaddingSize > 0 && noPaddingSize % factor == 0,
                          "No padding size `{0}` should be greater than `0` and divisible by factor `{1}`",
                          noPaddingSize, factor);
        return noPaddingSize / factor;
    };

    Shape fullInputShape(outShape.toValues());
    fullInputShape[Dims4D::Act::H] = inferShapeImpl(outShape[Dims4D::Act::H], factorH, padTop, padBottom);
    fullInputShape[Dims4D::Act::W] = inferShapeImpl(outShape[Dims4D::Act::W], factorW, padLeft, padRight);

    return fullInputShape;
}

// Infer the Interpolate output coordinates using the coordinates of the effective data
// Using the definition of Convolution coordinate transformation function.
// It represent the maximum output pixel that can be obtained up to the current input offsets.
//
// For Example: Bilinear HALF_PIXEL InterPolate
// The data is duplicated using the Storage Element pointers into:
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6
//     4 4 4 4 4 5 5 5 5 6 6 6 6 [6] <- effectiveDataCoord_0 [6, 13]
//     4 4 4 4 4 5 5 5 5 6 6 6 [6] 6 <- effectiveDataCoord_1 [7, 12]
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9
// The kernel size is [4, 4], stride is [2, 2]
// The output data shape is 6 x 6:
//     1.00 1.25 1.75 2.25 2.75 3.00
//     1.75 2.00 2.25 3.00 3.50 [3.75] <- interpOutCoord_0 [1, 5]
//     3.25 3.50 4.00 4.50 [5.00] 5.25 <- interpOutCoord_1 [2, 4]
//     4.75 5.00 5.55 6.00 6.50 6.75
//     6.25 6.50 7.00 7.50 8.00 8.25
//     7.00 7.25 7.75 8.25 8.75 9.00
// If the input effectiveDataCoord is [6, 13]
//    the output coordinate is [floor((6 - 4 + 1) / 2), floor((13 - 4 + 1) / 2)] = [1, 5]
// If the input effectiveDataCoord is [7, 12]
//    the output coordinate is [floor((7 - 4 + 1) / 2), floor((12 - 4 + 1) / 2)] = [2, 4]
//
Shape inferInterpOutCoordWithEffectiveDataCoord(ShapeRef effectiveDataCoord, ArrayRef<int64_t> kernelSize,
                                                ArrayRef<int64_t> strides) {
    const auto [kernelY, kernelX] = extractInterpKernelSize(kernelSize);
    const auto [strideY, strideX] = extractInterpStrides(strides);

    auto inferShapeCoordImpl = [](auto coord, auto kernelSize, auto stride) {
        auto inferedCoord = static_cast<int64_t>(floor((coord - kernelSize + 1) / stride));
        return std::max(int64_t(0), inferedCoord);
    };

    Shape interpOutCoord(effectiveDataCoord.toValues());
    interpOutCoord[Dims4D::Act::H] = inferShapeCoordImpl(effectiveDataCoord[Dims4D::Act::H], kernelY, strideY);
    interpOutCoord[Dims4D::Act::W] = inferShapeCoordImpl(effectiveDataCoord[Dims4D::Act::W], kernelX, strideX);

    return interpOutCoord;
}

// Infer the Interpolate input coordinates using the coordinates of the Interpolate output
// Using the definition of Interpolate coordinate transformation function
std::function<double(double, double, int64_t, int64_t)> getCoordinateTransformationFn(
        IE::InterpolateCoordModeAttr coordModeAttr) {
    VPUX_THROW_WHEN(coordModeAttr == nullptr, "Missing coordinate transformation mode");
    const auto coordMode = coordModeAttr.getValue();

    std::function<double(double, double, int64_t, int64_t)> coordTransform;
    switch (coordMode) {
    case IE::InterpolateCoordMode::HALF_PIXEL:
        coordTransform = [&](double coord, double scale, int64_t /*inputSize*/, int64_t /*outputSize*/) {
            return (coord + 0.5) / scale - 0.5;
        };
        break;
    case IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL:
        coordTransform = [&](double coord, double scale, int64_t /*inputSize*/, int64_t outputSize) {
            return (outputSize > 1) ? (coord + 0.5) / scale - 0.5 : 0.0;
        };
        break;
    case IE::InterpolateCoordMode::ASYMMETRIC:
        coordTransform = [&](double coord, double scale, int64_t /*inputSize*/, int64_t /*outputSize*/) {
            return coord / scale;
        };
        break;
    case IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN:
        coordTransform = [&](double coord, double scale, int64_t /*inputSize*/, int64_t /*outputSize*/) {
            return (coord + 0.5) / scale;
        };
        break;
    case IE::InterpolateCoordMode::ALIGN_CORNERS:
        coordTransform = [&](double coord, double /*scale*/, int64_t inputSize, int64_t outputSize) {
            return (outputSize == 1) ? 0.0 : coord * (inputSize - 1) / (outputSize - 1);
        };
        break;
    default:
        VPUX_THROW("SEInterpolateAttr: Cannot get CoordinateTransformation function of {0}", coordMode);
    }

    return coordTransform;
}

// Calculate the input Dim that required for the current Interpolate output Dim
// - Nearest interpolate: get the closest pixel depending on the nearest mode
// - Bilinear interpolate: get the ceil pixel
std::function<int64_t(double, double)> getNearestDimFn(VPU::NCEInterpolateModeAttr modeAttr,
                                                       IE::InterpolateNearestModeAttr nearestModeAttr) {
    std::function<int64_t(double, double)> nearestDim;
    VPUX_THROW_WHEN(modeAttr == nullptr, "SEInterpolateAttr: Missing mode attribute");
    const auto mode = modeAttr.getValue();
    if (mode == VPU::NCEInterpolateMode::BILINEAR) {
        nearestDim = [](double coord, double /*scale*/) -> int64_t {
            return std::ceil(coord);
        };
    } else if (mode == VPU::NCEInterpolateMode::NEAREST) {
        VPUX_THROW_WHEN(nearestModeAttr == nullptr, "SEInterpolateAttr: Nearest interpolate miss nearest mode");
        switch (nearestModeAttr.getValue()) {
        case IE::InterpolateNearestMode::ROUND_PREFER_FLOOR:
            nearestDim = [](double dim, double /*scale*/) -> int64_t {
                if (isDoubleEqual(dim, std::floor(dim) + 0.5)) {
                    return std::floor(dim);
                }
                return std::round(dim);
            };
            break;
        case IE::InterpolateNearestMode::ROUND_PREFER_CEIL:
            nearestDim = [](double dim, double /*scale*/) -> int64_t {
                return std::round(dim);
            };
            break;
        case IE::InterpolateNearestMode::FLOOR:
            nearestDim = [](double dim, double /*scale*/) -> int64_t {
                return std::floor(dim);
            };
            break;
        case IE::InterpolateNearestMode::CEIL:
            nearestDim = [](double dim, double /*scale*/) -> int64_t {
                return std::ceil(dim);
            };
            break;
        case IE::InterpolateNearestMode::SIMPLE:
            nearestDim = [](double dim, double scale) -> int64_t {
                if (scale < 1.) {
                    return std::ceil(dim);
                }
                return dim;
            };
            break;
        default:
            VPUX_THROW("SEInterpolateAttr: Unsupported InterpolateNearestMode. {0}", nearestModeAttr.getValue());
        }
    } else {
        VPUX_THROW("SEInterpolateAttr: Unsupported NCEInterpolateMode {0}", mode);
    }

    return nearestDim;
}

std::vector<int32_t> computeSpatialSEPtrs(mlir::MLIRContext* ctx, ShapeRef dataShape, ShapeRef outputShape,
                                          Byte elemSize, int64_t seSize, mlir::ArrayAttr offsetsAttr,
                                          llvm::function_ref<Shape(ShapeRef, ShapeRef)> backInferCoordFn) {
    VPUX_THROW_UNLESS(dataShape.size() == 4, "Expected 4D data shape, got {0} dimensions", dataShape.size());
    VPUX_THROW_UNLESS(elemSize.count() > 0, "Invalid element byte size {0}", elemSize.count());
    VPUX_THROW_UNLESS(seSize > 0 && (seSize % 16 == 0), "Invalid Storage Element size {0}", seSize);

    const auto inputC = dataShape[Dims4D::Act::C];
    const auto inputW = dataShape[Dims4D::Act::W];
    const auto getInputAddress = [&](ShapeRef inputCoord) {
        const auto offsetC = inputCoord[Dims4D::Act::C];
        const auto offsetH = inputCoord[Dims4D::Act::H];
        const auto offsetW = inputCoord[Dims4D::Act::W];

        const auto pixelOffset = (offsetH * inputW + offsetW) * inputC;
        const auto channelOffset = offsetC;
        return (pixelOffset + channelOffset) * elemSize.count();
    };

    const auto outputOffsets = (offsetsAttr != nullptr && !offsetsAttr.empty())
                                       ? parseIntArrayAttr<int64_t>(offsetsAttr)
                                       : SmallVector<int64_t>({0, 0, 0, 0});

    const auto startH = outputOffsets[Dims4D::Act::H.ind()];
    const auto startW = outputOffsets[Dims4D::Act::W.ind()];

    const auto sizeH = outputShape[Dims4D::Act::H];
    const auto sizeW = outputShape[Dims4D::Act::W];

    const auto outputC = outputShape[Dims4D::Act::C];
    const auto seDepth = outputC / seSize;
    const auto seTableNumElements = sizeH * sizeW * seDepth;
    std::vector<int32_t> sePtrs(seTableNumElements, 0);

    loop_3d(LoopExecPolicy::Parallel, ctx, sizeH, sizeW, seDepth, [&](int64_t h, int64_t w, int64_t se) {
        const auto outputCoord = Shape({0, se * seSize, startH + h, startW + w});
        const auto interpInputCoord = backInferCoordFn(outputCoord, dataShape);
        const auto offset = getInputAddress(interpInputCoord);
        auto seOffset = (h * sizeW + w) * seDepth + se;  // NHWC
        sePtrs[seOffset] = offset;
    });

    return sePtrs;
}

}  // namespace

// Infer final output shape with respect to tiling:
// - If sizes attr presents then its values are returned
// - If not then shape is calculated with factors and mode
Shape VPU::SEInterpolateAttr::inferOutputShape(ShapeRef inputShape) const {
    if (auto sizes = getSizes()) {
        return Shape(parseIntArrayAttr<int64_t>(sizes));
    }

    const auto scales = extractInterpScales(getScale());
    const auto factors = VPU::getNCEInterpolateFactors(scales, getMode(), getCoordinateTransformationMode());
    const auto padsBegin = VPU::getNCEInterpolatePadsBegin(scales, getMode(), getCoordinateTransformationMode());
    const auto padsEnd = VPU::getNCEInterpolatePadsEnd(scales, getMode(), getCoordinateTransformationMode());
    return inferFullOutputShape(inputShape, factors, padsBegin, padsEnd);
}

// Infer the full input shape given an output shape
Shape VPU::SEInterpolateAttr::backInferInputShape(ShapeRef outputShape) const {
    const auto scales = extractInterpScales(getScale());
    const auto factors = VPU::getNCEInterpolateFactors(scales, getMode(), getCoordinateTransformationMode());
    const auto padsBegin = VPU::getNCEInterpolatePadsBegin(scales, getMode(), getCoordinateTransformationMode());
    const auto padsEnd = VPU::getNCEInterpolatePadsEnd(scales, getMode(), getCoordinateTransformationMode());
    return inferFullInputShape(outputShape, factors, padsBegin, padsEnd);
}

// It is difficult to infer the Interpolate input coordinates directly using output coordinates
// The most general approach is to take two steps:
// Step 1: Infer the Interpolate output coordinates using the coordinates of the effective data
//         Using the definition of Convolution coordinate transformation function
// Step 2: Infer the Interpolate input coordinates using the coordinates of the Interpolate output
//         Using the definition of Interpolate coordinate transformation function
//
// For example: Bilinear Interpolate with PYTORCH_HALF_PIXEL coordMode
// The Interpolate input data:
//     1 2 3
//     4 5 [6] <- Step2: interpInputCoord [1, 2]
//     7 8 9
// With the following configuration:
//     - scale: [1, 1, 2, 2]
//     - factors: [4, 4]
//     - pads: [1, 1, 1, 1]
//     - kernel_size: [4, 4]
//     - strides: [2, 2]
// The data is duplicated using the Storage Element pointers into:
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6
//     4 4 4 4 4 5 5 5 5 6 6 6 [6] 6 <- Input: outputCoord [7, 12]
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9
// The Interpolate output data:
//     1.00 1.25 1.75 2.25 2.75 3.00
//     1.75 2.00 2.25 3.00 3.50 3.75
//     3.25 3.50 4.00 4.50 [5.00] 5.25 <- Step1: intepOutputCoord [2, 4]
//     4.75 5.00 5.55 6.00 6.50 6.75
//     6.25 6.50 7.00 7.50 8.00 8.25
//     7.00 7.25 7.75 8.25 8.75 9.00
//
// If the outputCoord = [7, 12]
// Step 1: Infer the Interpolate output coordinates using the coordinates of the effective data
//     Convolution transformation function: floor((coord - kernelSize + 1) / stride)
//     - intepOutputCoord = [floor((7 - 4 + 1) / 2), floor((12 - 4 + 1) / 2)] = [2, floor(4.5)] = [2, 4]
// Step 2: Infer the Interpolate input coordinates using the coordinates of the Interpolate output
//     Bilinear PYTORCH_HALF_PIXEL coordTransformFn function is: (coord + 0.5) / scale - 0.5
//     - inCoord = [(2 + 0.5) / 2 - 0.5, (4 + 0.5) / 2 - 0.5] = [0.75, 1.75]
//     Bilinear Interpolate nearestDimFn function is: ceil(coord)
//     - interpInputCoord = [1, 2]
//
Shape VPU::SEInterpolateAttr::backInferInputCoord(ShapeRef outputCoord, ShapeRef inputShape) const {
    const auto coordTransformFn = getCoordinateTransformationFn(getCoordinateTransformationMode());
    const auto nearestDimFn = getNearestDimFn(getMode(), getNearestMode());

    const auto scales = extractInterpScales(getScale());
    const auto factors = VPU::getNCEInterpolateFactors(scales, getMode(), getCoordinateTransformationMode());
    const auto padsBegin = VPU::getNCEInterpolatePadsBegin(scales, getMode(), getCoordinateTransformationMode());
    const auto padsEnd = VPU::getNCEInterpolatePadsEnd(scales, getMode(), getCoordinateTransformationMode());
    const auto outputShape = inferFullOutputShape(inputShape, factors, padsBegin, padsEnd);

    const auto initialInputShape = (getInitialInputShape() != nullptr)
                                           ? Shape(parseIntArrayAttr<int64_t>(getInitialInputShape()))
                                           : Shape(inputShape.raw());
    const auto initialOutputShape = (getInitialOutputShape() != nullptr)
                                            ? Shape(parseIntArrayAttr<int64_t>(getInitialOutputShape()))
                                            : Shape(outputShape.raw());

    const auto kernelSize = VPU::getNCEInterpolateKernelSize(scales, getMode(), getCoordinateTransformationMode());
    const auto strides = VPU::getNCEInterpolateStrides(scales, getMode(), getCoordinateTransformationMode());
    const auto intepOutputCoord = inferInterpOutCoordWithEffectiveDataCoord(outputCoord, kernelSize, strides);

    auto interpInputCoord = intepOutputCoord;
    for (size_t dim = 0; dim < intepOutputCoord.size(); dim++) {
        auto outCoord = static_cast<double>(intepOutputCoord[Dim(dim)]);
        auto inCoord = isDoubleEqual(scales[dim], 1.0)
                               ? outCoord
                               : coordTransformFn(outCoord, scales[dim], initialInputShape[Dim(dim)],
                                                  initialOutputShape[Dim(dim)]);
        auto nearestPixel = nearestDimFn(inCoord, scales[dim]);
        inCoord = std::clamp(nearestPixel, int64_t(0), inputShape[Dim(dim)] - 1);
        interpInputCoord[Dim(dim)] = inCoord;
    }

    return interpInputCoord;
}

//
// For example: Bilinear Interpolate with PYTORCH_HALF_PIXEL coordMode
// The Interpolate input data:
//     1 2 3
//     4 5 6 <-
//     7 8 9 <-
// With the following configuration:
//     - scale: [1, 1, 2, 2]
//     - factors: [4, 4]
//     - pads: [1, 1, 1, 1]
//     - kernel_size: [4, 4]
//     - strides: [2, 2]
// The data is duplicated using the Storage Element pointers into:
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6 <-
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6 <-
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
// The Interpolate output data:
//     1.00 1.25 1.75 2.25 2.75 3.00
//     1.75 2.00 2.25 3.00 3.50 3.75
//     3.25 3.50 4.00 4.50 5.00 5.25
//     4.75 5.00 5.55 6.00 6.50 6.75 <-
//     6.25 6.50 7.00 7.50 8.00 8.25 <-
//     7.00 7.25 7.75 8.25 8.75 9.00 <-
//
// If the Interpolate output tile contains the last three lines.
// The input parameters for this function is:
//
//     - outputTileOffset: [0, 0, 6, 0]
//     - outputTileShape:  [1, 16, 8, 14]
//     - inputShape: [1, 16, 3, 3]
//
// The method will return:
//
//     - inputTileOffset: [0, 0, 1, 0]
//     - inputTileShape:  [1, 16, 2, 3]
//
// Since the same scale configuration is used but only on two lines
// The data is duplicated using the Storage Element pointers into:
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6 <-
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6 <-
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
//
// If we further tile the duplicated data, will get the final effective data:
//
//     - offsets: [0, 0, 2, 0]
//     - sizes: [1, 16, 8, 14]
//
// }],
//
VPU::SEAttr VPU::SEInterpolateAttr::extractTile(ShapeRef outputTileOffset, ShapeRef outputTileShape,
                                                ShapeRef inputShape, Shape& inputTileOffset,
                                                Shape& inputTileShape) const {
    std::tie(inputTileShape, inputTileOffset) =
            inferInputTileShapeAndOffset(outputTileOffset, outputTileShape, inputShape);
    const auto scales = extractInterpScales(getScale());
    const auto factors = VPU::getNCEInterpolateFactors(scales, getMode(), getCoordinateTransformationMode());
    SmallVector<int64_t> offsets(inputShape.size(), 0);
    const auto getDimOffset = [&](Dim dim, const int64_t factor) {
        return outputTileOffset[dim] - inputTileOffset[dim] * factor;
    };
    offsets[Dims4D::Act::H.ind()] = getDimOffset(Dims4D::Act::H, factors[VPU::SE_INTERPOLATE_FACTOR_H]);
    offsets[Dims4D::Act::W.ind()] = getDimOffset(Dims4D::Act::W, factors[VPU::SE_INTERPOLATE_FACTOR_W]);

    return VPU::SEInterpolateAttr::get(getContext(), getMode(), getCoordinateTransformationMode(), getScale(),
                                       getNearestMode(), getIntArrayAttr(getContext(), offsets),
                                       getIntArrayAttr(getContext(), Shape(outputTileShape.raw())),
                                       getInitialInputShape(), getInitialOutputShape())
            .cast<VPU::SEAttr>();
}

std::pair<Shape, Shape> VPU::SEInterpolateAttr::inferInputTileShapeAndOffset(ShapeRef outputTileOffset,
                                                                             ShapeRef outputTileShape,
                                                                             ShapeRef inputShape) const {
    auto inputTileOffset = backInferInputCoord(outputTileOffset, inputShape);
    Shape outputTileOffsetEnd(outputTileOffset.raw());
    std::transform(outputTileOffset.begin(), outputTileOffset.end(), outputTileShape.raw().begin(),
                   outputTileOffsetEnd.begin(), [](auto offset, auto size) {
                       return offset + size - 1;
                   });
    Shape inputTileShape(outputTileShape);
    auto inputTileOffsetEnd = backInferInputCoord(outputTileOffsetEnd, inputShape);
    std::transform(inputTileOffset.begin(), inputTileOffset.end(), inputTileOffsetEnd.begin(), inputTileShape.begin(),
                   [](auto start, auto end) {
                       return end - start + 1;
                   });
    return std::make_pair(inputTileShape, inputTileOffset);
}

//
// For example: Bilinear Interpolate with PYTORCH_HALF_PIXEL coordMode
// The Interpolate input data:
//     1 2 3
//     4 5 6 <-
//     7 8 9 <-
// With the following configuration:
//     - scale: [1, 1, 2, 2]
//     - factors: [4, 4]
//     - pads: [1, 1, 1, 1]
//     - kernel_size: [4, 4]
//     - strides: [2, 2]
// The data is duplicated using the Storage Element pointers into:
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     1 1 1 1 1 2 2 2 2 3 3 3 3 3
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6 <-
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6 <-
//     4 4 4 4 4 5 5 5 5 6 6 6 6 6 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
//     7 7 7 7 7 8 8 8 8 9 9 9 9 9 <-
// The Interpolate output data:
//     1.00 1.25 1.75 2.25 2.75 3.00
//     1.75 2.00 2.25 3.00 3.50 3.75
//     3.25 3.50 4.00 4.50 5.00 5.25
//     4.75 5.00 5.55 6.00 6.50 6.75 <-
//     6.25 6.50 7.00 7.50 8.00 8.25 <-
//     7.00 7.25 7.75 8.25 8.75 9.00 <-
//
// If the Interpolate output tile contains the last three lines.
// The input parameters for this function is:
//
//     - dataShape: [1, 16, 2, 3]
//     - outputShape: [1, 16, 8, 14]
//     - elemSize: 1
//     - seSize: 16
//     - offsetsAttr: [0, 0, 2, 0]
//
// The addresses of each Storage Element for the data are:
//     0x0   0x10  0x20
//     0x30  0x40  0x50
//
// The method will return:
//     0x0  0x0  0x0  0x0  0x0  0x10 0x10 0x10 0x10 0x20 0x20 0x20 0x20 0x20
//     0x0  0x0  0x0  0x0  0x0  0x10 0x10 0x10 0x10 0x20 0x20 0x20 0x20 0x20
//     0x0  0x0  0x0  0x0  0x0  0x10 0x10 0x10 0x10 0x20 0x20 0x20 0x20 0x20
//     0x30 0x30 0x30 0x30 0x30 0x40 0x40 0x40 0x40 0x50 0x50 0x50 0x50 0x50
//     0x30 0x30 0x30 0x30 0x30 0x40 0x40 0x40 0x40 0x50 0x50 0x50 0x50 0x50
//     0x30 0x30 0x30 0x30 0x30 0x40 0x40 0x40 0x40 0x50 0x50 0x50 0x50 0x50
//     0x30 0x30 0x30 0x30 0x30 0x40 0x40 0x40 0x40 0x50 0x50 0x50 0x50 0x50
//     0x30 0x30 0x30 0x30 0x30 0x40 0x40 0x40 0x40 0x50 0x50 0x50 0x50 0x50
// }],
//
std::vector<int32_t> VPU::SEInterpolateAttr::computeSEOffsets(ShapeRef dataShape, StridesRef /*dataStrides*/,
                                                              Byte elemSize, int64_t seSize) const {
    const auto outputShape = inferOutputShape(dataShape);
    return computeSpatialSEPtrs(getContext(), dataShape, outputShape, elemSize, seSize, getOffsets(),
                                [&](ShapeRef outputCoord, ShapeRef inputShape) -> Shape {
                                    return backInferInputCoord(outputCoord, inputShape);
                                });
}

std::optional<VPU::SETileInfo> VPU::SEInterpolateAttr::getTileInfo() const {
    return VPU::SETileInfo{getOffsets(), getSizes()};
}

//
// SEUpsamplingAttr
//

mlir::LogicalResult VPU::SEUpsamplingAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                  mlir::ArrayAttr factorsAttr, mlir::ArrayAttr paddingAttr,
                                                  mlir::ArrayAttr offsetsAttr, mlir::ArrayAttr sizesAttr) {
    const auto hasNegativeValues = [](ArrayRef<int64_t> values) {
        return llvm::any_of(values, [](const int64_t value) {
            return value < 0;
        });
    };

    if (factorsAttr == nullptr) {
        return printTo(emitError(), "Got NULL factors in 'SEUpsamplingAttr'");
    }
    const auto factors = parseIntArrayAttr<int64_t>(factorsAttr);
    if (factors.size() != 2) {
        return printTo(emitError(), "Got factors with {0} dimensions in 'SEUpsamplingAttr'. Expected 2 dimensions",
                       factors.size());
    }
    if (hasNegativeValues(factors)) {
        return printTo(emitError(), "Got negative factors in 'SEUpsamplingAttr'");
    }

    if (paddingAttr != nullptr) {
        const auto padding = parseIntArrayAttr<int64_t>(paddingAttr);
        if (padding.size() != 4) {
            return printTo(emitError(), "Got padding with {0} dimensions in 'SEUpsamplingAttr'. Expected 4 dimensions",
                           padding.size());
        }
        if (hasNegativeValues(padding)) {
            return printTo(emitError(), "Got negative padding in 'SEUpsamplingAttr'");
        }
    }

    if (offsetsAttr != nullptr) {
        const auto offsets = parseIntArrayAttr<int64_t>(offsetsAttr);
        if (offsets.size() != 4) {
            return printTo(emitError(), "Got offsets with {0} dimensions in 'SEUpsamplingAttr'. Expected 4 dimensions",
                           offsets.size());
        }
        if (hasNegativeValues(offsets)) {
            return printTo(emitError(), "Got negative offsets in 'SEUpsamplingAttr'");
        }
    }

    if (sizesAttr != nullptr) {
        const auto sizes = parseIntArrayAttr<int64_t>(sizesAttr);
        if (sizes.size() != 4) {
            return printTo(emitError(), "Got sizes with {0} dimensions in 'SEUpsamplingAttr'. Expected 4 dimensions",
                           sizes.size());
        }
        if (hasNegativeValues(sizes)) {
            return printTo(emitError(), "Got negative sizes in 'SEUpsamplingAttr'");
        }
    }

    return mlir::success();
}

//
// SEUpsamplingAttr (SEAttrInterface)
//

namespace {

std::tuple<int64_t, int64_t> extractFactors(mlir::ArrayAttr factorsAttr) {
    VPUX_THROW_UNLESS(factorsAttr != nullptr, "Missing factors attribute");
    const auto factors = parseIntArrayAttr<int64_t>(factorsAttr);
    const auto factorH = factors[VPU::SE_UPSAMPLING_FACTOR_H];
    const auto factorW = factors[VPU::SE_UPSAMPLING_FACTOR_W];
    return {factorH, factorW};
}

std::tuple<int64_t, int64_t, int64_t, int64_t> extractPadding(mlir::ArrayAttr paddingAttr) {
    VPUX_THROW_UNLESS(paddingAttr != nullptr, "Missing padding attribute");
    const auto padding = parseIntArrayAttr<int64_t>(paddingAttr);
    const auto padLeft = padding[VPU::SE_PAD_LEFT];
    const auto padTop = padding[VPU::SE_PAD_TOP];
    const auto padRight = padding[VPU::SE_PAD_RIGHT];
    const auto padBottom = padding[VPU::SE_PAD_BOTTOM];
    return {padLeft, padTop, padRight, padBottom};
}

Shape inferShapeImpl(
        ShapeRef shape, mlir::ArrayAttr factorsAttr, mlir::ArrayAttr paddingAttr,
        llvm::function_ref<int64_t(const int64_t, const int64_t, const int64_t, const int64_t)> computeShapeFn) {
    const auto [factorH, factorW] = extractFactors(factorsAttr);
    const auto [padLeft, padTop, padRight, padBottom] = extractPadding(paddingAttr);

    const auto newHeight = computeShapeFn(shape[Dims4D::Act::H], factorH, padTop, padBottom);
    const auto newWidth = computeShapeFn(shape[Dims4D::Act::W], factorW, padLeft, padRight);

    return Shape({shape[Dims4D::Act::N], shape[Dims4D::Act::C], newHeight, newWidth});
}

// Infers the starting output offset for the given input offset
// For example, let's take an upsampling attribute with a factor and pad before of 3 with the following input:
// -------------
// | A | B | C |
// -------------
//   0   1   2
//
// It produces the following output:
// -----------------------------------------
// | p | p | p | A | z | z | B | z | z | C |
// -----------------------------------------
//   0   1   2   3   4   5   6   7   8   9
//
// The following output tile offsets are returned:
//   - offset 0 if the input tile offset is between [0-5]
//   - offset 6 if the input tile offset is between [6-8]
//   - offset 9 if the input tile offset is 9+
Shape inferOutputTileStartOffset(ShapeRef inputTileOffset, VPU::SEUpsamplingAttr seAttr) {
    const auto [factorH, factorW] = extractFactors(seAttr.getFactors());
    const auto [padLeft, padTop, padRight, padBottom] = extractPadding(seAttr.getPadding());

    auto outputTileOffset = Shape(inputTileOffset.toValues());
    outputTileOffset[Dims4D::Act::H] = inputTileOffset[Dims4D::Act::H] * (factorH + 1);
    outputTileOffset[Dims4D::Act::W] = inputTileOffset[Dims4D::Act::W] * (factorW + 1);

    if (inputTileOffset[Dims4D::Act::H] != 0) {
        outputTileOffset[Dims4D::Act::H] += padTop;
    }
    if (inputTileOffset[Dims4D::Act::W] != 0) {
        outputTileOffset[Dims4D::Act::W] += padLeft;
    }

    return outputTileOffset;
}

}  // namespace

Shape VPU::SEUpsamplingAttr::inferOutputShape(ShapeRef inputShape) const {
    auto outputShape = inferShapeImpl(inputShape, getFactors(), getPadding(),
                                      [](const int64_t inputSize, const int64_t factor, const int64_t padBefore,
                                         const int64_t padAfter) -> int64_t {
                                          return inputSize + factor * (inputSize - 1) + padBefore + padAfter;
                                      });

    if (getOffsets() == nullptr && getSizes() == nullptr) {
        return outputShape;
    }
    const auto offsets = getOffsets() != nullptr ? parseIntArrayAttr<int64_t>(getOffsets())
                                                 : SmallVector<int64_t>(outputShape.size(), 0);
    const auto sizes =
            getSizes() != nullptr ? parseIntArrayAttr<int64_t>(getSizes()) : SmallVector<int64_t>(outputShape.raw());

    for (auto idx : irange(outputShape.size())) {
        outputShape[Dim(idx)] = std::min(outputShape[Dim(idx)] - offsets[idx], sizes[idx]);
    }
    return outputShape;
}

Shape VPU::SEUpsamplingAttr::backInferInputShape(ShapeRef outputShape) const {
    return inferShapeImpl(
            outputShape, getFactors(), getPadding(),
            [](const int64_t outputSize, const int64_t factor, const int64_t padBefore, const int64_t padAfter) {
                return (outputSize + factor - padBefore - padAfter) / (1 + factor);
            });
}

// Infers the input coordinates that are used to generate the given output coordinate.
// SEUpsamplingAttr will duplicate elements a number of times depending on the factor and padding configuration,
// where some of duplicated elements will be filtered-out by the sparsity map. For this to work, each spatial
// element in the output will correspond to one input spatial element.
//
// For example, let's take the following input data that exemplifies one spatial dimension:
// -------------
// |   |   |   |
// -------------
//   0   1   2     <- input index
//
// If the factor is set to 1, padding before to 2 and padding after to 3, the following output data is obtained:
//   0   1   2   3   4   5   6   7   8   9     <- output index
// -----------------------------------------
// | p | p |   | z |   | z |   | p | p | p |   <- p - padding, z - value zero from the factor
// -----------------------------------------
//   0   0   0   0   1   1   2   2   2   2     <- input index
//
// In other words, for this example the function will infer:
// - input coordinate 0 for output coordinates [0-3]
// - input coordinate 1 for output coordinates [4-5]
// - input coordinate 2 for output coordinates [6-9]
Shape VPU::SEUpsamplingAttr::backInferInputCoord(ShapeRef outputCoord, ShapeRef inputShape) const {
    VPUX_THROW_UNLESS(outputCoord.size() == 4, "Expected 4D output coordinates, got {0}D", outputCoord.size());
    VPUX_THROW_UNLESS(inputShape.size() == 4, "Expected 4D input shape, got {0}D", inputShape.size());

    const auto [padLeft, padTop, padRight, padBottom] = extractPadding(getPadding());
    auto inputH = outputCoord[Dims4D::Act::H] - padTop;
    auto inputW = outputCoord[Dims4D::Act::W] - padLeft;

    const auto [factorH, factorW] = extractFactors(getFactors());
    inputH = (inputH >= 0) ? inputH / (factorH + 1) : inputH;
    inputW = (inputW >= 0) ? inputW / (factorW + 1) : inputW;

    inputH = std::clamp(inputH, int64_t(0), inputShape[Dims4D::Act::H] - 1);
    inputW = std::clamp(inputW, int64_t(0), inputShape[Dims4D::Act::W] - 1);

    return Shape({outputCoord[Dims4D::Act::N], outputCoord[Dims4D::Act::C], inputH, inputW});
}

// Infers the input tile for the given output tile.
// Receives the offset and shape of the output tile and the input shape of the data. The offset and shape of the
// inferred input tile is returned via the reference parameters. The return value contains the SEUpsamplingAttr with
// updated parameters which, coupled with the inferred input tile, can generate the output tile.
//
// For example, let's take an upsampling attribute with a factor and pad before of 2 with the following input:
// -------------
// | A | B | C |
// -------------
//   0   1   2
//
// It produces the following output:
// -------------------------------------
// | p | p | A | z | z | B | z | z | C |
// -------------------------------------
//   0   1   2   3   4   5   6   7   8
//
// If the output tile has offset 6 and size 3, the output tile covers range [6-8].
// The inferred input tile has offset 1 and size 2, covering the range [1-2]. The returned SEUpsamplingAttr will contain
// the same factor and padding values as the original attribute, so the full output contains:
// -------------------------
// | p | p | B | z | z | C |
// -------------------------
//   0   1   2   3   4   5
//
// In order to generate the same output as the output tile received as a parameter, the returned attribute will contain
// offset 3 and size 3, thus producing the following final output:
// -------------
// | z | z | C |
// -------------
//   0   1   2
VPU::SEAttr VPU::SEUpsamplingAttr::extractTile(ShapeRef outputTileOffset, ShapeRef outputTileShape, ShapeRef inputShape,
                                               Shape& inputTileOffset, Shape& inputTileShape) const {
    std::tie(inputTileShape, inputTileOffset) =
            inferInputTileShapeAndOffset(outputTileOffset, outputTileShape, inputShape);
    auto outputTileStartOffset = inferOutputTileStartOffset(inputTileOffset, *this);
    Shape relativeOffsets(inputTileOffset);
    std::transform(outputTileOffset.raw().begin(), outputTileOffset.raw().end(), outputTileStartOffset.begin(),
                   relativeOffsets.begin(), [](auto offset, auto outerTileOffset) {
                       return offset - outerTileOffset;
                   });
    const auto [padLeft, padTop, padRight, padBottom] = extractPadding(getPadding());
    if (inputTileOffset[Dims4D::Act::H] != 0) {
        relativeOffsets[Dims4D::Act::H] += padTop;
    }
    if (inputTileOffset[Dims4D::Act::W] != 0) {
        relativeOffsets[Dims4D::Act::W] += padLeft;
    }

    const auto [factorH, factorW] = extractFactors(getFactors());
    auto newPadding = parseIntArrayAttr<int64_t>(getPadding());
    newPadding[VPU::SE_PAD_RIGHT] += factorW;
    newPadding[VPU::SE_PAD_BOTTOM] += factorH;

    return VPU::SEUpsamplingAttr::get(getContext(), getFactors(), getIntArrayAttr(getContext(), newPadding),
                                      getIntArrayAttr(getContext(), relativeOffsets),
                                      getIntArrayAttr(getContext(), Shape(outputTileShape.raw())))
            .cast<VPU::SEAttr>();
}

std::pair<Shape, Shape> VPU::SEUpsamplingAttr::inferInputTileShapeAndOffset(ShapeRef outputTileOffset,
                                                                            ShapeRef outputTileShape,
                                                                            ShapeRef inputShape) const {
    auto inputTileOffset = backInferInputCoord(outputTileOffset, inputShape);
    Shape outputTileEnd(outputTileOffset.raw());
    std::transform(outputTileEnd.begin(), outputTileEnd.end(), outputTileShape.raw().begin(), outputTileEnd.begin(),
                   [](auto offset, auto size) {
                       return offset + size - 1;
                   });

    auto inputTileEnd = backInferInputCoord(outputTileEnd, inputShape);
    Shape inputTileShape(outputTileShape);
    std::transform(inputTileOffset.begin(), inputTileOffset.end(), inputTileEnd.begin(), inputTileShape.begin(),
                   [](auto start, auto end) {
                       return end - start + 1;
                   });
    return std::make_pair(inputTileShape, inputTileOffset);
}

std::vector<int32_t> VPU::SEUpsamplingAttr::computeSEOffsets(ShapeRef dataShape, StridesRef /*dataStrides*/,
                                                             Byte elemSize, int64_t seSize) const {
    const auto outputShape = inferOutputShape(dataShape);
    return computeSpatialSEPtrs(getContext(), dataShape, outputShape, elemSize, seSize, getOffsets(),
                                [&](ShapeRef outputCoord, ShapeRef inputShape) -> Shape {
                                    return backInferInputCoord(outputCoord, inputShape);
                                });
}

std::optional<VPU::SETileInfo> VPU::SEUpsamplingAttr::getTileInfo() const {
    return VPU::SETileInfo{getOffsets(), getSizes()};
}

//
// SEPaddingAttr
//

mlir::LogicalResult VPU::SEPaddingAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                               vpux::IE::PadModeAttr padModeAttr, mlir::ArrayAttr paddingAttr,
                                               mlir::ArrayAttr offsetsAttr, mlir::ArrayAttr sizesAttr) {
    const auto hasNegativeValues = [](ArrayRef<int64_t> values) {
        return llvm::any_of(values, [](const int64_t value) {
            return value < 0;
        });
    };

    if (padModeAttr == nullptr) {
        return printTo(emitError(), "Got NULL 'padModeAttr' in 'SEPaddingAttr'");
    }
    if (paddingAttr == nullptr) {
        return printTo(emitError(), "Got NULL 'paddingAttr' in 'SEPaddingAttr'");
    }

    const auto padding = parseIntArrayAttr<int64_t>(paddingAttr);
    if (padding.size() != 4) {
        return printTo(emitError(), "Got padding with {0} dimensions in 'SEPaddingAttr'. Expected 4 dimensions",
                       padding.size());
    }
    if (hasNegativeValues(padding)) {
        return printTo(emitError(), "Got negative padding in 'SEPaddingAttr'");
    }

    if (offsetsAttr != nullptr) {
        const auto offsets = parseIntArrayAttr<int64_t>(offsetsAttr);
        if (offsets.size() != 4) {
            return printTo(emitError(), "Got offsets with {0} dimensions in 'SEPaddingAttr'. Expected 4 dimensions",
                           offsets.size());
        }
        if (hasNegativeValues(offsets)) {
            return printTo(emitError(), "Got negative offsets in 'SEPaddingAttr'");
        }
    }

    if (sizesAttr != nullptr) {
        const auto sizes = parseIntArrayAttr<int64_t>(sizesAttr);
        if (sizes.size() != 4) {
            return printTo(emitError(), "Got sizes with {0} dimensions in 'SEPaddingAttr'. Expected 4 dimensions",
                           sizes.size());
        }
        if (hasNegativeValues(sizes)) {
            return printTo(emitError(), "Got negative sizes in 'SEPaddingAttr'");
        }
    }

    return mlir::success();
}

//
// SEPaddingAttr (SEAttrInterface)
//

namespace {

enum CoordLocation { padBegin, inData, padEnd };

std::function<int64_t(int64_t, int64_t, int64_t)> getPadCoordinateTransformationFn(IE::PadModeAttr padModeAttr) {
    VPUX_THROW_WHEN(padModeAttr == nullptr, "Missing padding mode attribution");
    const auto padMode = padModeAttr.getValue();

    std::function<int64_t(int64_t, int64_t, int64_t)> coordTransform;
    switch (padMode) {
    case IE::PadMode::CONSTANT:
    case IE::PadMode::EDGE:
        coordTransform = [&](int64_t coord, int64_t padBegin, int64_t inputSize) {
            if (coord < padBegin) {
                return static_cast<int64_t>(0);
            } else if (coord >= padBegin + inputSize) {
                return inputSize - 1;
            }
            return coord - padBegin;
        };
        break;
    case IE::PadMode::REFLECT:
        coordTransform = [&](int64_t coord, int64_t padBegin, int64_t inputSize) {
            if (coord < padBegin) {
                return padBegin - coord;
            } else if (coord >= padBegin + inputSize) {
                return (inputSize - 1) - (coord - inputSize - padBegin + 1);
            }
            return coord - padBegin;
        };
        break;
    case IE::PadMode::SYMMETRIC:
        coordTransform = [&](int64_t coord, int64_t padBegin, int64_t inputSize) {
            if (coord < padBegin) {
                return padBegin - coord - 1;
            } else if (coord >= padBegin + inputSize) {
                return inputSize - (coord - inputSize - padBegin + 1);
            }
            return coord - padBegin;
        };
        break;
    default:
        VPUX_THROW("SEPaddingAttr: Cannot get CoordinateTransformation function of {0}", padMode);
    }

    return coordTransform;
}

Shape inferPadShapeImpl(ShapeRef shape, mlir::ArrayAttr paddingAttr,
                        llvm::function_ref<int64_t(const int64_t, const int64_t, const int64_t)> computeShapeFn) {
    const auto [padLeft, padTop, padRight, padBottom] = extractPadding(paddingAttr);

    const auto newHeight = computeShapeFn(shape[Dims4D::Act::H], padTop, padBottom);
    const auto newWidth = computeShapeFn(shape[Dims4D::Act::W], padLeft, padRight);

    return Shape({shape[Dims4D::Act::N], shape[Dims4D::Act::C], newHeight, newWidth});
}
}  // namespace

Shape VPU::SEPaddingAttr::inferOutputShape(ShapeRef inputShape) const {
    if (auto sizes = getSizes()) {
        return Shape(parseIntArrayAttr<int64_t>(sizes));
    }

    return inferPadShapeImpl(inputShape, getPadding(),
                             [](const int64_t inputSize, const int64_t padBefore, const int64_t padAfter) -> int64_t {
                                 return inputSize + padBefore + padAfter;
                             });
}

Shape VPU::SEPaddingAttr::backInferInputShape(ShapeRef outputShape) const {
    VPUX_THROW_UNLESS(getOffsets() == nullptr && getSizes() == nullptr,
                      "SEPaddingAttr: Cannot support back infer input shape with offsets and sizes attibution");
    return inferPadShapeImpl(outputShape, getPadding(),
                             [](const int64_t outputSize, const int64_t padBefore, const int64_t padAfter) {
                                 return (outputSize - padBefore - padAfter);
                             });
}

// Infers the input coordinates that are used to generate the given output coordinate.
//
// For example: PadOp with Reflect Mode
// The PadOp input data:
//     1 2 3
//     4 [5] 6 <- Output: inputCoord [1, 1]
//     7 8 9
// With the following configuration:
//     - padding: [1, 2, 2, 1]
// The data is duplicated using the Storage Element pointers into:
//     8 7 8 9 8 7
//     5 4 5 6 5 4
//     2 1 2 3 2 1
//     5 4 5 6 [5] 4 <- Input: outputCoord [3, 4]
//     8 7 8 9 8 7
//     5 4 5 6 5 4
//
Shape VPU::SEPaddingAttr::backInferInputCoord(ShapeRef outputCoord, ShapeRef inputShape) const {
    VPUX_THROW_UNLESS(outputCoord.size() == 4, "Expected 4D output coordinates, got {0}D", outputCoord.size());
    VPUX_THROW_UNLESS(inputShape.size() == 4, "Expected 4D input shape, got {0}D", inputShape.size());

    const auto [padLeft, padTop, padRight, padBottom] = extractPadding(getPadding());
    VPUX_THROW_UNLESS(outputCoord[Dims4D::Act::H] < inputShape[Dims4D::Act::H] + padTop + padBottom &&
                              outputCoord[Dims4D::Act::W] < inputShape[Dims4D::Act::W] + padLeft + padRight,
                      "Get unexpected output coordinate {0}", outputCoord);
    const auto coordTransformFn = getPadCoordinateTransformationFn(getMode());

    auto inputCoord = Shape(outputCoord.raw());
    for (size_t dim = 0; dim < outputCoord.size(); dim++) {
        auto outCoord = static_cast<int64_t>(outputCoord[Dim(dim)]);
        auto inCoord = outCoord;
        if (Dim(dim) == Dims4D::Act::H) {
            inCoord = coordTransformFn(outCoord, padTop, inputShape[Dim(dim)]);
        } else if (Dim(dim) == Dims4D::Act::W) {
            inCoord = coordTransformFn(outCoord, padLeft, inputShape[Dim(dim)]);
        }
        inCoord = std::clamp(inCoord, int64_t(0), inputShape[Dim(dim)] - 1);
        inputCoord[Dim(dim)] = inCoord;
    }

    return inputCoord;
}

std::tuple<Shape, Shape, Shape> VPU::SEPaddingAttr::inferPadInputTileParams(ShapeRef outputTileOffset,
                                                                            ShapeRef outputTileShape,
                                                                            ShapeRef inputShape) const {
    Shape outputTileEnd(outputTileOffset.raw());
    std::transform(outputTileEnd.begin(), outputTileEnd.end(), outputTileShape.raw().begin(), outputTileEnd.begin(),
                   [](auto offset, auto size) {
                       return offset + size - 1;
                   });

    const auto getCoordLocation = [&](int64_t coord, Dim dim, int64_t padBeginValue) {
        if (dim == Dims4D::Act::H || dim == Dims4D::Act::W) {
            if (coord < padBeginValue) {
                return CoordLocation::padBegin;
            }
            return coord >= (inputShape[dim] + padBeginValue) ? CoordLocation::padEnd : CoordLocation::inData;
        }

        return CoordLocation::inData;
    };

    const auto padModeAttr = getMode();
    const auto [padLeft, padTop, padRight, padBottom] = extractPadding(getPadding());
    const auto inputTileStart = backInferInputCoord(outputTileOffset, inputShape);
    const auto inputTileEnd = backInferInputCoord(outputTileEnd, inputShape);

    Shape inputTileOffset(outputTileOffset);
    Shape inputTileShape(outputTileShape);
    Shape relativeOffsets(inputTileOffset);
    for (size_t axis = 0; axis < inputTileOffset.size(); axis++) {
        const auto dim = Dim(axis);
        const auto inputTileDimStart = inputTileStart[dim];
        const auto inputTileDimEnd = inputTileEnd[dim];

        int64_t padBeginValue = 0;
        if (dim == Dims4D::Act::H) {
            padBeginValue = padTop;
        } else if (dim == Dims4D::Act::W) {
            padBeginValue = padLeft;
        }

        // All possible tiled output shapes are divided into six scenarios based on
        // whether their starting and ending coordinates are in padbegin, inData, or padEnd.
        // In each case, it is necessary to calculate the offsets of input shape, input offsets, and SEAttr offsets
        // separately. But all with the following calculation logic:
        //
        // For example: Reflect Mode (padBegin: 5) with both output start and end at padBegin:
        //          Index:     0     1     2     3     4     5     6     7     8     9    10
        // Effective Data:  | P_5 | P_4 | P_3 | P_2 | P_1 | D_0 | D_1 | D_2 | D_3 | D_4 | D_5 | …
        //
        // Scenario 1: outputTileStart at padBegin, outputTileEnd at padBegin
        // outputTileStart Index: 1 (P_4) -> inputTileStart Coord: 4 (D_4)
        // outputTileEnd Index: 3 (P_2) -> inputTileEnd Coord: 2 (D_2)
        // - inputTileShape: Calculate the covered data area based on the input start and end coordinates obtained from
        // the backInferInputCoord function
        //   For Reflect Mode, an additional value needs to be offset.
        //   The tiled input data is: | D_1 | D_2 | D_3 | D_4 |
        //   inputTileShape = (inputTileDimStart - inputTileDimEnd + 1) + 1 = 4
        //
        // Scenario 2: outputTileStart at padBegin, outputTileEnd at inData
        // outputTileStart Index: 2 (P_3) -> inputTileStart Coord: 3 (D_3)
        // outputTileEnd Index: 6 (D_1) -> inputTileEnd Coord: 1 (D_1)
        // - inputTileOffset: Calculate the minimum coordinate offset of the covered data area
        //   The tiled input data is: | D_0 | D_1 | D_2 | D_3 |
        //   inputTileOffset = 0
        //
        // Scenario 3: outputTileStart at inData, outputTileEnd at inData
        // outputTileStart Index: 6 (D_1) -> inputTileStart Coord: 1 (D_1)
        // outputTileEnd Index: 9 (D_4) -> inputTileEnd Coord: 4 (D_4)
        // - seAttrOffsets: Calculate offset to obtain the effective data from the shape obtained from tiled input and
        // original padding
        //   The tiled input data: | D_1 | D_2 | D_3 | D_4 |
        //   The data from tiled input and original padding: | P_4 | P_4 | P_4 | P_3 | P_2 | D_1 | D_2 | D_3 | D_4 |
        //   The effective data: | D_1 | D_2 | D_3 | D_4 |
        //   seAttrOffsets = original_padding = 5
        //
        // For the other cases, the same calculation logic applies. When an accuracy issue is discovered
        // a schematic diagram can help analyze it
        auto outputStartOffsetLocation = getCoordLocation(outputTileOffset[dim], dim, padBeginValue);
        auto outputEndOffsetLocation = getCoordLocation(outputTileEnd[dim], dim, padBeginValue);
        if (outputStartOffsetLocation == CoordLocation::padBegin &&
            outputEndOffsetLocation == CoordLocation::padBegin) {
            inputTileShape[dim] = inputTileDimStart - inputTileDimEnd + 1;
            inputTileOffset[dim] = inputTileDimEnd;
            relativeOffsets[dim] = padBeginValue;
            if (padModeAttr.getValue() == IE::PadMode::REFLECT) {
                inputTileShape[dim] += 1;
                inputTileOffset[dim] -= 1;
                relativeOffsets[dim] += 1;
            }
        } else if (outputStartOffsetLocation == CoordLocation::padBegin &&
                   outputEndOffsetLocation == CoordLocation::inData) {
            inputTileShape[dim] = std::max(inputTileDimStart, inputTileDimEnd) + 1;
            inputTileOffset[dim] = static_cast<int64_t>(0);
            relativeOffsets[dim] = outputTileOffset[dim];
        } else if (outputStartOffsetLocation == CoordLocation::padBegin &&
                   outputEndOffsetLocation == CoordLocation::padEnd) {
            inputTileShape[dim] = inputShape[dim];
            inputTileOffset[dim] = 0;
            relativeOffsets[dim] = outputTileOffset[dim];
        } else if (outputStartOffsetLocation == CoordLocation::inData &&
                   outputEndOffsetLocation == CoordLocation::inData) {
            inputTileShape[dim] = inputTileDimEnd - inputTileDimStart + 1;
            inputTileOffset[dim] = inputTileDimStart;
            relativeOffsets[dim] = padBeginValue;
        } else if (outputStartOffsetLocation == CoordLocation::inData &&
                   outputEndOffsetLocation == CoordLocation::padEnd) {
            inputTileShape[dim] = inputShape[dim] - std::min(inputTileDimStart, inputTileDimEnd);
            inputTileOffset[dim] = std::min(inputTileDimStart, inputTileDimEnd);
            relativeOffsets[dim] = padBeginValue;
        } else if (outputStartOffsetLocation == CoordLocation::padEnd &&
                   outputEndOffsetLocation == CoordLocation::padEnd) {
            inputTileShape[dim] = inputTileDimStart - inputTileDimEnd + 1;
            inputTileOffset[dim] = inputTileDimEnd;
            relativeOffsets[dim] = padBeginValue + inputTileDimStart - inputTileDimEnd + 1;
            if (padModeAttr.getValue() == IE::PadMode::REFLECT) {
                inputTileShape[dim] += 1;
                relativeOffsets[dim] += 1;
            }
        } else {
            VPUX_THROW("SEPaddingAttr: Got unexpected tiled offsets and tiled shape of output");
        }
    }
    return std::make_tuple(inputTileShape, inputTileOffset, relativeOffsets);
}

// Infers the input tile for the given output tile.
// Receives the offset and shape of the output tile and the input shape of the data. The offset and shape of the
// inferred input tile is returned via the reference parameters. The return value contains the SEPaddingAttr with
// updated parameters which, coupled with the inferred input tile, can generate the output tile.
//
// For example: PadOp with Reflect Mode
// The PadOp input data:
//     1 2 3
//     4 5 6 <-
//     7 8 9 <-
// With the following configuration:
//     - padding: [1, 2, 2, 1]
// The data is duplicated using the Storage Element pointers into:
//     8 7 8 9 8 7
//     5 4 5 6 5 4
//     2 1 2 3 2 1
//     5 4 5 6 5 4 <-
//     8 7 8 9 8 7 <-
//     5 4 5 6 5 4 <-
//
// If the Pad output tile contains the last three lines.
// The input parameters for this function is:
//
//     - outputTileOffset: [0, 0, 3, 0]
//     - outputTileShape:  [1, 16, 3, 6]
//     - inputShape: [1, 16, 3, 3]
//
// The method will return:
//
//     - inputTileOffset: [0, 0, 1, 0]
//     - inputTileShape:  [1, 16, 2, 3]
//
// Since the same scale configuration is used but only on two lines
// The data is duplicated using the Storage Element pointers into:
//     8 7 8 9 8 7
//     8 7 8 9 8 7
//     5 4 5 6 5 4 <-
//     8 7 8 9 8 7 <-
//     5 4 5 6 5 4 <-
//
// If we further tile the duplicated data, will get the final effective data:
//
//     - offsets: [0, 0, 2, 0]
//     - sizes: [1, 16, 3, 6]
//
// }],
//
VPU::SEAttr VPU::SEPaddingAttr::extractTile(ShapeRef outputTileOffset, ShapeRef outputTileShape, ShapeRef inputShape,
                                            Shape& inputTileOffset, Shape& inputTileShape) const {
    Shape relativeOffsets;
    std::tie(inputTileShape, inputTileOffset, relativeOffsets) =
            inferPadInputTileParams(outputTileOffset, outputTileShape, inputShape);
    return VPU::SEPaddingAttr::get(getContext(), getMode(), getPadding(),
                                   getIntArrayAttr(getContext(), relativeOffsets),
                                   getIntArrayAttr(getContext(), Shape(outputTileShape.raw())))
            .cast<VPU::SEAttr>();
}

std::pair<Shape, Shape> VPU::SEPaddingAttr::inferInputTileShapeAndOffset(ShapeRef outputTileOffset,
                                                                         ShapeRef outputTileShape,
                                                                         ShapeRef inputShape) const {
    auto [inputTileShape, inputTileOffset, relativeOffsets] =
            inferPadInputTileParams(outputTileOffset, outputTileShape, inputShape);
    return std::make_pair(inputTileShape, inputTileOffset);
}

std::vector<int32_t> VPU::SEPaddingAttr::computeSEOffsets(ShapeRef dataShape, StridesRef /*dataStrides*/, Byte elemSize,
                                                          int64_t seSize) const {
    const auto outputShape = inferOutputShape(dataShape);
    return computeSpatialSEPtrs(getContext(), dataShape, outputShape, elemSize, seSize, getOffsets(),
                                [&](ShapeRef outputCoord, ShapeRef inputShape) -> Shape {
                                    return backInferInputCoord(outputCoord, inputShape);
                                });
}

std::optional<VPU::SETileInfo> VPU::SEPaddingAttr::getTileInfo() const {
    return VPU::SETileInfo{getOffsets(), getSizes()};
}

//
// SERollAttr
//

mlir::LogicalResult VPU::SERollAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                            mlir::ArrayAttr shiftAttr, mlir::ArrayAttr axesAttr,
                                            mlir::ArrayAttr offsetsAttr, mlir::ArrayAttr sizesAttr) {
    const auto hasNegativeValues = [](ArrayRef<int64_t> values) {
        return llvm::any_of(values, [](const int64_t value) {
            return value < 0;
        });
    };

    if (shiftAttr == nullptr) {
        return printTo(emitError(), "Got NULL shift in 'SERollAttr'");
    }
    const auto shifts = parseIntArrayAttr<int64_t>(shiftAttr);
    if (shifts.size() != 2) {
        return printTo(emitError(), "Got shifts with {0} dimensions in 'SERollAttr'. Expected 2 dimensions",
                       shifts.size());
    }

    if (axesAttr == nullptr) {
        return printTo(emitError(), "Got NULL axes in 'SERollAttr'");
    }
    const auto axes = parseIntArrayAttr<int64_t>(axesAttr);
    if (axes.size() != 2) {
        return printTo(emitError(), "Got shifts with {0} dimensions in 'SERollAttr'. Expected 2 dimensions",
                       axes.size());
    }

    if (!(axes[SE_ROLL_SPATIAL_H] == Dims4D::Act::H.ind() && axes[SE_ROLL_SPATIAL_W] == Dims4D::Act::W.ind())) {
        return printTo(emitError(), "Got invalid axes {0}", axes);
    }

    if (offsetsAttr != nullptr) {
        const auto offsets = parseIntArrayAttr<int64_t>(offsetsAttr);
        if (offsets.size() != 4) {
            return printTo(emitError(), "Got offsets with {0} dimensions in 'SERollAttr'. Expected 4 dimensions",
                           offsets.size());
        }
        if (hasNegativeValues(offsets)) {
            return printTo(emitError(), "Got negative offsets in 'SERollAttr'");
        }
    }

    if (sizesAttr != nullptr) {
        const auto sizes = parseIntArrayAttr<int64_t>(sizesAttr);
        if (sizes.size() != 4) {
            return printTo(emitError(), "Got sizes with {0} dimensions in 'SERollAttr'. Expected 4 dimensions",
                           sizes.size());
        }
        if (hasNegativeValues(sizes)) {
            return printTo(emitError(), "Got negative sizes in 'SERollAttr'");
        }
    }

    return mlir::success();
}

//
// SERollAttr (SEAttrInterface)
//

Shape VPU::SERollAttr::inferOutputShape(ShapeRef inputShape) const {
    if (getSizes() != nullptr) {
        return Shape(parseIntArrayAttr<int64_t>(getSizes()));
    }

    return inputShape.raw();
}

Shape VPU::SERollAttr::backInferInputShape(ShapeRef outputShape) const {
    VPUX_THROW_UNLESS(getOffsets() == nullptr && getSizes() == nullptr,
                      "SERollAttr: Cannot support back infer input shape with offsets and sizes attibution");

    return outputShape.raw();
}

// Infers the input coordinates that are used to generate the given output coordinate.
// Reversely roll back to get the input coordinates.
//
// For example, let's take the following input data that exemplifies one spatial dimension:
// -------------------------------------
// | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
// -------------------------------------
//
// If the shift is set to 3 and axis is 0, the following output data is obtained:
//    0   1   2   3   4   5   6   7   8     <- output index
//  -------------------------------------
//  | 6 | 7 | 8 | 0 | 1 | 2 | 3 | 4 | 5 |
//  -------------------------------------
//
// In other words, for this example the function will infer:
// - output coordinate 0 for input coordinates [6]
// - output coordinate 1 for input coordinates [7]
// - output coordinate 8 for input coordinates [5]
//
Shape VPU::SERollAttr::backInferInputCoord(ShapeRef outputCoord, ShapeRef inputShape) const {
    const auto shifts = parseIntArrayAttr<int64_t>(getShift());
    const auto axes = parseIntArrayAttr<int64_t>(getAxes());

    const auto getInputCoord = [&](int64_t outCoord, int64_t shift, int64_t dimShape) {
        if (shift == 0) {
            return outCoord;
        }
        auto inputCoord = outCoord + shift;
        if (inputCoord < 0) {
            inputCoord = dimShape + inputCoord;
        } else if (inputCoord >= dimShape) {
            inputCoord = inputCoord % dimShape;
        }
        return inputCoord;
    };

    const auto inputH = getInputCoord(outputCoord[Dim(axes[SE_ROLL_SPATIAL_H])], -1 * shifts[SE_ROLL_SPATIAL_H],
                                      inputShape[Dim(axes[SE_ROLL_SPATIAL_H])]);
    const auto inputW = getInputCoord(outputCoord[Dim(axes[SE_ROLL_SPATIAL_W])], -1 * shifts[SE_ROLL_SPATIAL_W],
                                      inputShape[Dim(axes[SE_ROLL_SPATIAL_W])]);
    return Shape({outputCoord[Dims4D::Act::N], outputCoord[Dims4D::Act::C], inputH, inputW});
}

// Infers the input tile for the given output tile.
// Receives the offset and shape of the output tile and the input shape of the data. The offset and shape of the
// inferred input tile is returned via the reference parameters. The return value contains the SERollAttr with
// updated parameters which, coupled with the inferred input tile, can generate the output tile.
//
// For example, let's take the following input data that exemplifies one spatial dimension with shift 3:
// -------------------------------------
// | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
// -------------------------------------
//
// It produces the following output:
//    0   1   2   3   4   5   6   7   8     <- output index
//  -------------------------------------
//  | 6 | 7 | 8 | 0 | 1 | 2 | 3 | 4 | 5 |
//  -------------------------------------
//
// If the output tile has offset 2 and size 3, the output tile covers range [2-4].
// The inferred input tile has offset 0 and size 9, covering the range [0-8]. The returned SERollAttr will contain
// the new shifts 1, so the full output contains:
// -------------------------------------
// | 8 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
// -------------------------------------
//
// In order to generate the same output as the output tile received as a parameter, the returned attribute will contain
// offset 0 and size 3, thus producing the following final output:
// -------------
// | 8 | 0 | 1 |
// -------------
//
VPU::SEAttr VPU::SERollAttr::extractTile(ShapeRef outputTileOffset, ShapeRef outputTileShape, ShapeRef inputShape,
                                         Shape& inputTileOffset, Shape& inputTileShape) const {
    std::tie(inputTileShape, inputTileOffset) =
            inferInputTileShapeAndOffset(outputTileOffset, outputTileShape, inputShape);
    const auto axes = parseIntArrayAttr<int64_t>(getAxes());
    const auto shifts = parseIntArrayAttr<int64_t>(getShift());
    const auto getNewShift = [&](int64_t dim) -> int64_t {
        const auto offset = outputTileOffset[Dim(axes[dim])];
        if (offset >= shifts[dim]) {
            return 0;
        } else {
            return shifts[dim] - offset;
        }
    };

    const auto newShifts = Shape{getNewShift(SE_ROLL_SPATIAL_H), getNewShift(SE_ROLL_SPATIAL_W)};

    auto ctx = getContext();
    return VPU::SERollAttr::get(ctx, getIntArrayAttr(ctx, newShifts), getAxes(),
                                getIntArrayAttr(ctx, SmallVector<int64_t>(outputTileShape.size(), 0)),
                                getIntArrayAttr(ctx, outputTileShape.raw()))
            .cast<VPU::SEAttr>();
}

std::pair<Shape, Shape> VPU::SERollAttr::inferInputTileShapeAndOffset(ShapeRef outputTileOffset,
                                                                      ShapeRef outputTileShape,
                                                                      ShapeRef inputShape) const {
    const auto outputTileShapeH = outputTileShape[Dims4D::Act::H];
    const auto outputTileShapeW = outputTileShape[Dims4D::Act::W];

    const auto outputTileOffsetH = outputTileOffset[Dims4D::Act::H];
    const auto outputTileOffsetW = outputTileOffset[Dims4D::Act::W];

    const auto outputTileOffsetEndH = outputTileOffsetH + outputTileShapeH - 1;
    const auto outputTileOffsetEndW = outputTileOffsetW + outputTileShapeW - 1;

    const auto getMinPoint = [](int64_t shift, int64_t offsetBegin, int64_t offsetEnd) {
        if (shift == 0) {
            return offsetBegin;
        }
        if (offsetEnd >= shift) {
            return offsetBegin >= shift ? offsetBegin : shift;
        } else {
            return offsetBegin;
        }
    };

    const auto getMaxPoint = [](int64_t shift, int64_t offsetBegin, int64_t offsetEnd) {
        if (shift == 0) {
            return offsetEnd;
        }
        if (offsetEnd >= shift) {
            return offsetBegin >= shift ? offsetEnd : shift - 1;
        } else {
            return offsetEnd;
        }
    };

    const auto getInputCoord = [&](int64_t offsetH, int64_t offsetW) {
        const auto outCoord =
                Shape({outputTileShape[Dims4D::Act::N], outputTileShape[Dims4D::Act::C], offsetH, offsetW});
        return backInferInputCoord(outCoord, inputShape);
    };
    const auto shifts = parseIntArrayAttr<int64_t>(getShift());
    const auto offsetInOutputMinH = getMinPoint(shifts[SE_ROLL_SPATIAL_H], outputTileOffsetH, outputTileOffsetEndH);
    const auto offsetInOutputMinW = getMinPoint(shifts[SE_ROLL_SPATIAL_W], outputTileOffsetW, outputTileOffsetEndW);
    const auto minInputCoord = getInputCoord(offsetInOutputMinH, offsetInOutputMinW);

    const auto offsetInOutputMaxH = getMaxPoint(shifts[SE_ROLL_SPATIAL_H], outputTileOffsetH, outputTileOffsetEndH);
    const auto offsetInOutputMaxW = getMaxPoint(shifts[SE_ROLL_SPATIAL_W], outputTileOffsetW, outputTileOffsetEndW);
    const auto maxInputCoord = getInputCoord(offsetInOutputMaxH, offsetInOutputMaxW);

    VPUX_THROW_WHEN(maxInputCoord[Dims4D::Act::H] < minInputCoord[Dims4D::Act::H] ||
                            maxInputCoord[Dims4D::Act::W] < minInputCoord[Dims4D::Act::W],
                    "illegal min/max values");

    Shape inputTileOffset({outputTileOffset[Dims4D::Act::N], outputTileOffset[Dims4D::Act::C],
                           minInputCoord[Dims4D::Act::H], minInputCoord[Dims4D::Act::W]});

    Shape inputTileShape({outputTileShape[Dims4D::Act::N], outputTileShape[Dims4D::Act::C],
                          maxInputCoord[Dims4D::Act::H] - minInputCoord[Dims4D::Act::H] + 1,
                          maxInputCoord[Dims4D::Act::W] - minInputCoord[Dims4D::Act::W] + 1});
    return std::make_pair(inputTileShape, inputTileOffset);
}

std::vector<int32_t> VPU::SERollAttr::computeSEOffsets(ShapeRef dataShape, StridesRef /*dataStrides*/, Byte elemSize,
                                                       int64_t seSize) const {
    const auto outputShape = inferOutputShape(dataShape);
    return computeSpatialSEPtrs(getContext(), dataShape, outputShape, elemSize, seSize, getOffsets(),
                                [&](ShapeRef outputCoord, ShapeRef inputShape) -> Shape {
                                    return backInferInputCoord(outputCoord, inputShape);
                                });
}

std::optional<VPU::SETileInfo> VPU::SERollAttr::getTileInfo() const {
    return VPU::SETileInfo{getOffsets(), getSizes()};
}

//
// SEDilatedConvAttr (SEAttrInterface)
//

mlir::LogicalResult VPU::SEDilatedConvAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                   mlir::ArrayAttr dilation, mlir::ArrayAttr kernelStride,
                                                   [[maybe_unused]] mlir::ArrayAttr kernelSize,
                                                   [[maybe_unused]] mlir::ArrayAttr dataOffset,
                                                   [[maybe_unused]] mlir::ArrayAttr dataSizes,
                                                   [[maybe_unused]] mlir::ArrayAttr offsetsAttr,
                                                   [[maybe_unused]] mlir::ArrayAttr sizesAttr) {
    auto [dilateY, dilateX] = DilationUtils::extractDilationFactors(dilation);
    auto [strideY, strideX] = DilationUtils::extractDilationStrides(kernelStride);

    if (dilateX < 2 && dilateY < 2) {
        return printTo(emitError(), "Dilation has no effect, factors are < 2");
    }

    if (strideX > 1 || strideY > 1) {
        return printTo(emitError(), "Non-trivial strides are not supported");
    }

    return mlir::success();
}

//                                                 EXAMPLE SUB-GRAPH

//
//                          NOTE: We use HxW and YxX format to align with other SEAttrs.
//

//
//      Input                     Subviews                     Sub-convolution        Outputs     Re-constructed Output
//
//                              N = 4  (Dy * Dx)               N = 4  (Dy * Dx)
//
//     Size = 8x8         Size = 8x8        Size = 8x7                              Size = 2x2       Size = 4x4
//   Dilation = 2,2      Offset = 0,0      Offset = 0,1
//                                                              4x4        4x4
//  1 2 1 2 1 2 1 2     1 2 1 2 1 2 1 2    2 1 2 1 2 1 2      1 1 1 1    2 2 2 2     1 1  2 2         1 2 1 2
//  3 4 3 4 3 4 3 4     3 4 3 4 3 4 3 4    4 3 4 3 4 3 4      1 1 1 1    2 2 2 2     1 1  2 2         3 4 3 4
//  1 2 1 2 1 2 1 2     1 2 1 2 1 2 1 2    2 1 2 1 2 1 2      1 1 1 1    2 2 2 2                      1 2 1 2
//  3 4 3 4 3 4 3 4     3 4 3 4 3 4 3 4    4 3 4 3 4 3 4      1 1 1 1    2 2 2 2     3 3  4 4         3 4 3 4
//  1 2 1 2 1 2 1 2     1 2 1 2 1 2 1 2    2 1 2 1 2 1 2                             3 3  4 4
//  3 4 3 4 3 4 3 4     3 4 3 4 3 4 3 4    4 3 4 3 4 3 4        4x4        4x4
//  1 2 1 2 1 2 1 2     1 2 1 2 1 2 1 2    2 1 2 1 2 1 2      3 3 3 3    4 4 4 4
//  3 4 3 4 3 4 3 4     3 4 3 4 3 4 3 4    4 3 4 3 4 3 4      3 3 3 3    4 4 4 4
//                                                            3 3 3 3    4 4 4 4
//      Kernel            Size = 7x8        Size = 7x7        3 3 3 3    4 4 4 4
//                       Offset = 1,0      Offset = 1,1
//    Size = 3x3
//                      3 4 3 4 3 4 3 4    4 3 4 3 4 3 4
//      1 2 3           1 2 1 2 1 2 1 2    2 1 2 1 2 1 2
//      4 5 6           3 4 3 4 3 4 3 4    4 3 4 3 4 3 4
//      7 8 9           1 2 1 2 1 2 1 2    2 1 2 1 2 1 2
//                      3 4 3 4 3 4 3 4    4 3 4 3 4 3 4
//                      1 2 1 2 1 2 1 2    2 1 2 1 2 1 2
//                      3 4 3 4 3 4 3 4    4 3 4 3 4 3 4

// Calculate output shape given an input shape.
// In this case, we calculate the output shape assuming `dataSizes` always extends
// the shape to the extents of the input shape. This means we may include
// an extra column or row of elements that we don't need to generate the output.
// This is done to make things simpler to reason about but potentially could be
// more efficient.
Shape VPU::SEDilatedConvAttr::inferOutputShape(ShapeRef inputShape) const {
    auto [dilateY, dilateX] = DilationUtils::extractDilationFactors(getDilation());
    auto [offsetN, offsetC, offsetY, offsetX] = DilationUtils::extractDilationDataOffsets(getDataOffset());

    const auto rowCount = inputShape[Dims4D::Act::H];
    const auto colCount = inputShape[Dims4D::Act::W];
    const auto dataRowCount = (rowCount - offsetY + dilateY - 1) / dilateY;
    const auto dataColCount = (colCount - offsetX + dilateX - 1) / dilateX;

    Shape outputShape(inputShape.toValues());
    outputShape[Dims4D::Act::H] = dataRowCount;
    outputShape[Dims4D::Act::W] = dataColCount;
    return outputShape;
}

// Calculate input shape given an output shape.
Shape VPU::SEDilatedConvAttr::backInferInputShape(ShapeRef outputShape) const {
    auto [dilateY, dilateX] = DilationUtils::extractDilationFactors(getDilation());
    Shape inputShape(outputShape.toValues());

    inputShape[Dims4D::Act::H] = outputShape[Dims4D::Act::H] * dilateY;
    inputShape[Dims4D::Act::W] = outputShape[Dims4D::Act::W] * dilateX;

    return inputShape;
}

Shape VPU::SEDilatedConvAttr::backInferInputCoord(ShapeRef outputCoord, [[maybe_unused]] ShapeRef inputShape) const {
    auto [dilateY, dilateX] = DilationUtils::extractDilationFactors(getDilation());
    auto [offsetN, offsetC, offsetH, offsetW] = DilationUtils::extractDilationDataOffsets(getDataOffset());
    Shape inputCoord(outputCoord.toValues());

    // TODO (E#134656): Handle padding and stride
    inputCoord[Dims4D::Act::H] = outputCoord[Dims4D::Act::H] * dilateY + offsetH;
    inputCoord[Dims4D::Act::W] = outputCoord[Dims4D::Act::W] * dilateX + offsetW;

    return inputCoord;
}

VPU::SEAttr VPU::SEDilatedConvAttr::extractTile(ShapeRef outputTileOffset, ShapeRef outputTileShape,
                                                ShapeRef inputShape, Shape& inputTileOffset,
                                                Shape& inputTileShape) const {
    auto [offsetN, offsetC, offsetH, offsetW] = DilationUtils::extractDilationDataOffsets(getDataOffset());
    // auto [dilateY, dilateX] = DilationUtils::extractDilationFactors(getDilation());
    Shape relativeOffsets;  // Unused, just zero it out. Always start from zero.

    // Calc input tile start offset
    inputTileOffset = backInferInputCoord(outputTileOffset, inputShape);

    // Calc output tile end offset
    // it is output tile start + output tile shape - 1
    Shape outputTileEnd(outputTileOffset.raw());
    std::transform(outputTileEnd.begin(), outputTileEnd.end(), outputTileShape.raw().begin(), outputTileEnd.begin(),
                   [](auto offset, auto size) {
                       return offset + size - 1;
                   });

    // Calc input tile end offset
    // backinfer output tile coordinate
    const auto inputTileEnd = backInferInputCoord(outputTileEnd, inputShape);

    // Calc input tile shape
    // input tile end - input tile starts
    inputTileShape.resize(inputTileOffset.size());
    std::transform(inputTileOffset.begin(), inputTileOffset.end(), inputTileEnd.begin(), inputTileShape.begin(),
                   [](auto start, auto end) {
                       return end - start + 1;
                   });

    auto dataSizes = getIntArrayAttr(getContext(), inputTileShape);

    // change input tile shape to full shape
    inputTileShape = backInferInputShape(outputTileShape);

    auto attr = VPU::SEDilatedConvAttr::get(
                        getContext(), getDilation(), getKernelStride(), getKernelSize(),  // keep the same
                        getDataOffset(),  // Always use same offsets, they encode subconv position
                        dataSizes,        // data size is a precise size of input data used
                        /* offsets = */ getIntArrayAttr(getContext(), relativeOffsets),  // always zero
                                                                                         /* sizes = */
                        getIntArrayAttr(getContext(),
                                        Shape(outputTileShape.raw())))  // exact output tile shape, provided by tiler
                        .cast<VPU::SEAttr>();

    // count from 0 offset
    inputTileOffset[Dims4D::Act::H] -= offsetH;
    inputTileOffset[Dims4D::Act::W] -= offsetW;
    // but use more data
    inputTileShape[Dims4D::Act::H] += offsetH;
    inputTileShape[Dims4D::Act::W] += offsetW;

    // clamp input tile shape to input shape
    if (inputTileShape[Dims4D::Act::H] + inputTileOffset[Dims4D::Act::H] > inputShape[Dims4D::Act::H]) {
        inputTileShape[Dims4D::Act::H] =
                inputTileShape[Dims4D::Act::H] -
                ((inputTileShape[Dims4D::Act::H] + inputTileOffset[Dims4D::Act::H]) - inputShape[Dims4D::Act::H]);
    }
    if (inputTileShape[Dims4D::Act::W] + inputTileOffset[Dims4D::Act::W] > inputShape[Dims4D::Act::W]) {
        inputTileShape[Dims4D::Act::W] =
                inputTileShape[Dims4D::Act::W] -
                ((inputTileShape[Dims4D::Act::W] + inputTileOffset[Dims4D::Act::W]) - inputShape[Dims4D::Act::W]);
    }
    return attr;
}

std::pair<vpux::Shape, vpux::Shape> VPU::SEDilatedConvAttr::inferInputTileShapeAndOffset(ShapeRef outputTileOffset,
                                                                                         ShapeRef outputTileShape,
                                                                                         ShapeRef inputShape) const {
    return std::make_pair(backInferInputShape(outputTileShape), backInferInputCoord(outputTileOffset, inputShape));
}

std::vector<int32_t> VPU::SEDilatedConvAttr::computeSEOffsets(ShapeRef dataShape,
                                                              [[maybe_unused]] StridesRef dataStrides, Byte elemSize,
                                                              int64_t seSize) const {
    const auto outputShape = inferOutputShape(dataShape);
    return computeSpatialSEPtrs(getContext(), dataShape, outputShape, elemSize, seSize, getOffsets(),
                                [&](ShapeRef outputCoord, ShapeRef inputShape) -> Shape {
                                    return backInferInputCoord(outputCoord, inputShape);
                                });
}

std::optional<VPU::SETileInfo> VPU::SEDilatedConvAttr::getTileInfo() const {
    return VPU::SETileInfo{getOffsets(), getSizes()};
}
