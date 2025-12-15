//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/const/attributes/content.hpp"

namespace vpux::Const {

std::optional<mlir::Type> inferElemTypeAffineReshape(ShapeRef inputShape, mlir::Type inputElementType,
                                                     const SmallVector<SmallVector<int64_t>>& dimMapping,
                                                     ArrayRef<int64_t> shapeValue);

/**
    @brief Back-infers the input element type of an AffineReshape from its output element type.

    @param inShape - input shape
    @param outputElemType - desired element type for AffineReshape's output
    @param dimMapping - attribute of the AffineReshape op which indicates how the input dims are combined to result in
   the output dims
    @param shapeValue - output shape

    @return the back-inferred input type, or std::nullopt if it cannot be obtained

    When the outputElemType is either a non-quantized type (e.g. f32, f16, u8 etc.) or a quantized per tensor type (one
    scale & zeropoint for the whole tensor), input element type is equal to outputElemType.

    When outputElemType is quantized per axis, the input type can be back-inferred in the following scenarios:
    1. the output quantization axis is merged with other output dims and all of them are equal to 1:
        e.g. 4x32x2x3 -> AffineReshape {dimMapping = [[0], [1,2,3], [4], [5]]} -> 4x1x32x1x2x3 {quant-axis = 2}
              => input quantization axis is 1
    2. the output quantization axis is split into multiple input dims, but all the new input dims are 1:
        e.g. 2x32x1x1x10 -> AffineReshape {dimMapping = [[0], [1], [1], [1], [2, 3]]} -> 2x32x2x5 {quant-axis = 1}
              => input quantization axis is 1
    3. output quantization axis is untouched by reshape
        e.g. 2x32x10 -> AffineReshape {dimMapping = [[0], [1], [2, 3]]} -> 2x32x2x5 {quant-axis = 1}
              => input quantization axis is 1

    Element type back-inference cannot be done when output quantization axis is merged or split. E.g.:
        2x2x16x2x5 -> AffineReshape {dimMapping = [[0], [1], [1], [2], [3]]} -> 2x32x2x5 {quant-axis = 1}
            => out axis = 32 is split among input dims 1 and 2: 2x16 => cannot back-infer the axis
        2x64x5x1 -> AffineReshape {dimMapping = [[0], [1, 2], [3], [3]]} -> 2x32x2x5 {quant-axis = 1}
            => out axis = 32 is merged with the following output dim 32x2 -> 64 => cannot back-infer the axis

    Caveat: When quantization axis is of size 1, function return nullopt. In that scenario there might be multiple
            valid input quantization axes, therefore this case is skipped.
*/
std::optional<mlir::Type> backInferElemTypeAffineReshape(ShapeRef inShape, mlir::Type outputElemType,
                                                         const SmallVector<SmallVector<int64_t>>& dimMapping,
                                                         ArrayRef<int64_t> shapeValue);

std::optional<DimsOrder> inferAffineReshapeOutputLayout(const DimArr& inPerm, mlir::ArrayAttr dimMapAttr);

/// For a [..., #const.AffineReshape, #const.SubView, ...] transformation pair this function
/// returns the new offset and shape of the SubView if that SubView can be ordered before AffineReshape or failure if
/// this is not possible.
mlir::FailureOr<std::tuple<SmallVector<int64_t>, SmallVector<int64_t>>> swapAffineReshapeAndSubView(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape,
        const SmallVector<SmallVector<int64_t>>& dimMapping, ArrayRef<int64_t> subViewOffset,
        ArrayRef<int64_t> subViewShape);

}  // namespace vpux::Const
