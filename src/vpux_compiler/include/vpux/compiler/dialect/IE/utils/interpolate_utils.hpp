//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/PatternMatch.h>
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"

namespace vpux {
namespace IE {

//
// Constants
//

// This constant represents the supported scale of convert_bilinear_to_strided_concat_and_conv v2 convert (Referred to
// as v2 in subsequent comments), and will also be used in pass split_bilinear_into_H_and_W, which is a further
// optimization of v2, split_bilinear_into_H_and_W use a specially constructed kernel to place a portion of W's data on
// C, and then achieves interleaved concatenation (performance bottleneck of v2) on W through reshaping.
constexpr int64_t CONVERT_BILINEAR_TO_STRIDED_CONCAT_CONVOLUTION_V2_SUPPORTED_SCALE = 2;

mlir::FailureOr<SmallVector<int64_t>> extractIntVector(mlir::Location loc, const mlir::Value value,
                                                       const std::optional<mlir::ArrayAttr>& attr);
mlir::FailureOr<SmallVector<double>> extractFPVector(mlir::Location loc, const mlir::Value value,
                                                     const std::optional<mlir::ArrayAttr>& attr);

SmallVector<int64_t> getInterpAxesVal(mlir::Location loc, const mlir::Value value,
                                      const std::optional<mlir::ArrayAttr>& attr, NDTypeInterface inType);

// Get the innermost axis from `axes` according to the `dimsOrder`.
// For example, if the `dimsOrder` is NHWC and `axes` are `[1, 2]`, it will return 1.
mlir::FailureOr<int64_t> getInnermostAxis(mlir::Location loc, DimsOrder dimsOrder, ArrayRef<int64_t> axes);

int64_t getInterpCoordinatesSize(mlir::Value output, int64_t innermostAxis);
int64_t getInterpLambdasSize(mlir::Value output, int64_t innermostAxis);

void applyInterpPads(MutableArrayRef<int64_t> outShape, ArrayRef<int64_t> padsBegin, ArrayRef<int64_t> padsEnd);
SmallVector<int64_t> inferInterpOutShape(mlir::Location loc, ArrayRef<int64_t> axes, ShapeRef origShape,
                                         mlir::FailureOr<ArrayRef<int64_t>> padsBegin,
                                         mlir::FailureOr<ArrayRef<int64_t>> padsEnd,
                                         vpux::IE::InterpolateCalcMode calcMode,
                                         mlir::FailureOr<ArrayRef<int64_t>> sizes,
                                         mlir::FailureOr<ArrayRef<double>> scales, mlir::Type scalesElemType,
                                         vpux::Logger log);
mlir::Value createPadding(mlir::PatternRewriter& rewriter, IE::InterpolateOp origOp, mlir::Value input, Dim axis,
                          int64_t forwardPad, int64_t backpad, StringRef locPrefix);

template <typename InterpolateAdaptor>
SmallVector<int64_t> calcOutputShapes(InterpolateAdaptor interpolate, mlir::Location loc, vpux::Logger log,
                                      mlir::MLIRContext* ctx) {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(interpolate.getInput().getType());
    const auto inShape = inType.getShape();

    const auto axesVal = getInterpAxesVal(loc, interpolate.getAxes(), interpolate.getAxesAttr(), inType);
    const auto beginPads = extractIntVector(loc, nullptr, interpolate.getAttr().getPadsBegin());
    const auto endPads = extractIntVector(loc, nullptr, interpolate.getAttr().getPadsEnd());
    const auto calcMode = interpolate.getAttr().getShapeCalcMode().getValue();
    const auto sizes = extractIntVector(loc, interpolate.getSizes(), interpolate.getSizesAttr());

    const auto scalesIn = interpolate.getScales();
    const auto scales = extractFPVector(loc, scalesIn, interpolate.getScalesAttr());
    const auto scalesElemType = scalesIn != nullptr
                                        ? mlir::cast<vpux::NDTypeInterface>(scalesIn.getType()).getElementType()
                                        : mlir::Float64Type::get(ctx);

    return inferInterpOutShape(loc, axesVal, inShape, beginPads, endPads, calcMode, sizes, scales, scalesElemType, log);
}

// For some specific Interpolate that shape size of scaling axis is always equal 1.
// Those Interpolate can be considered as Broadcast. For example:
// - InterpolateMode: Any; InterpolateCoordMode: Any; Pad_Begin and Pad_End is null or zero
// - Input shape: 1x16x1x1; Output shape: 1x16x32x32; Scales: [1, 1, 32, 32]
// It can be convert to NEAREST with ASYMMETRIC mode that will benefit from SEP feature.
bool isBroadCastInterpolate(IE::InterpolateOp op);

/*
Half_Pixel nearest interpolate (NearestMode ROUND_PREFER_CEIL or ROUND_PREFER_FLOOR) and Asymmetric nearest
interpolate (NearestMode FLOOR) are equivalent when scale is 2X.
For example :
A nearest interpolate 1x1x1x3 -> 1x1x1x6
Input date is 1, 2, 3.

For Asymmetric model the original tensor axis x is calculated according to the formula x_resized/2.

So the the output mapping coordinates are [0, 0.5, 1, 1.5, 2, 2.5].

InputDate       1       2       3
In_coordinates  0       1       2
                +---+---+---+---+---+--->
Out_coordinates 0  0.5  1  1.5  2  2.5

If the nearest model is FLOOR, the coordinates is [0, 0, 1, 1, 2, 2], output date is [1, 1, 2, 2, 3, 3].

For Half_Pixel model the original tensor axis x is calculated according to the formula ((x_resized + 0.5) / 2) - 0.5.

So the the output mapping coordinates are [-0.25, 0.25, 0.75, 1.25, 1.75, 2.25].

InputDate              1               2               3
In_coordinates         0               1               2
                   +---+---+---+---+---+---+---+---+---+---+--->
Out_coordinates -0.25    0.25    0.75    1.25    1.75    2.25

If the nearest model is ROUND_PREFER_CEIL or ROUND_PREFER_FLOOR, the coordinates is [0, 0, 1, 1, 2, 2], output date is
[1, 1, 2, 2, 3, 3].
*/
bool isEquivalentToNearestAsymmetricInterpolate(IE::InterpolateOp op);

// Generation of interpolation fraction coefficients
std::pair<double, double> computeFractionCoefficients(double fraction);

// Function prototype for coodinate transformation mode
using MapCoordFuncT = llvm::function_ref<std::pair<int32_t, double>(int32_t, double, int32_t, int32_t)>;

// Determine the coordintate transformation function
MapCoordFuncT getMapCoordMethod(InterpolateCoordMode coordMode);

bool isFusingConvertIntoBilinearInterpolateOnDpuBeneficial(IE::InterpolateOp op, mlir::Type outType);

}  // namespace IE
}  // namespace vpux
