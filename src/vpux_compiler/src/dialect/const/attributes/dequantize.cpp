//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/subspaces.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

using namespace vpux;

//
// DequantizeAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::DequantizeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto qElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(input.getElementType());
    VPUX_THROW_UNLESS(qElemType != nullptr, "Got non quantized type '{0}' in 'DequantizeAttr'");

    return input.changeElemType(qElemType.getExpressedType());
}

bool vpux::Const::DequantizeAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface input) const {
    // Splat value cannot be used to store weights for per-axis quantization.
    // Applying different scales to the same splat input value yields non-splat results.
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(input.getElementType())) {
        return false;
    }
    return inputIsSplat;
}

//
// DequantizeAttr::transform
//

Const::Content vpux::Const::DequantizeAttr::transform(vpux::Const::Content& input) const {
    const auto qElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(input.getType().getElementType());
    VPUX_THROW_UNLESS(qElemType != nullptr, "Got non quantized type '{0}' in 'DequantizeAttr'");

    auto output =
            Const::Content::allocTempBuffer(inferOutputType(input.getType()), mlir::Float32Type::get(getContext()),
                                            inferOutputSplat(input.isSplat(), input.getType()));
    auto realVals = output.getTempBuf<float>();

    if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(qElemType)) {
        const auto scale = uniformType.getScale();
        const auto zeroPoint = uniformType.getZeroPoint();

        input.read([&](auto qVals) {
            if (const auto quantileUniformType = mlir::dyn_cast<mlir::quant::QuantileQuantizedType>(qElemType)) {
                const auto quantilesLUT = quantileUniformType.getQuantiles();
                for (size_t i = 0; i < realVals.size(); ++i) {
                    const auto qVal = checked_cast<int64_t>(qVals[i]);
                    realVals[i] = dequantizeDouble(quantilesLUT[qVal], scale, zeroPoint);
                }
            } else {
                for (size_t i = 0; i < realVals.size(); ++i) {
                    const auto qVal = checked_cast<int64_t>(qVals[i]);
                    realVals[i] = dequantize(qVal, scale, zeroPoint);
                }
            }
        });

    } else if (const auto uniformPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(qElemType)) {
        const auto qVals = input.getValues<int64_t>();

        const auto scales = uniformPerAxisType.getScales();
        const auto zeroPoints = uniformPerAxisType.getZeroPoints();
        const auto axis = Dim(uniformPerAxisType.getQuantizedDimension());

        const auto dimsOrder = input.getType().getDimsOrder();
        const auto memAxis = dimsOrder.toMemDim(axis);
        const auto memShape = dimsOrder.toMemoryOrder(input.getType().getShape());

        // Get the volume of the dimensions less significant than the quantized axis,
        // which use the same dequantization parameters.
        // (QuantAxisDim, ..., InnerMostDim]
        int64_t innerSize = 1;
        for (size_t i = memAxis.ind() + 1; i < memShape.size(); ++i) {
            innerSize *= memShape[MemDim(i)];
        }
        VPUX_THROW_WHEN(innerSize == 0, "Inner size is zero");

        // Get the size of the quantized dimension.
        const int64_t quantAxisSize = memShape[memAxis];
        VPUX_THROW_WHEN(quantAxisSize == 0, "Quantized axis size is zero");

        const int64_t quantAxisTotalSize = quantAxisSize * innerSize;  // = [QuantAxisDim, ..., InnerMostDim]
        // Get the volume of the dimensions more significant than the quantized axis.
        // [OuterMostDim, ..., QuantAxisDim)
        const int64_t outerSize = memShape.totalSize() / quantAxisTotalSize;
        VPUX_THROW_WHEN(outerSize == 0, "Outer size is zero");

        VPUX_THROW_UNLESS(scales.size() == checked_cast<size_t>(quantAxisSize),
                          "Wrong scales size '{0}', expected '{1}'", scales.size(), quantAxisSize);
        VPUX_THROW_UNLESS(zeroPoints.size() == checked_cast<size_t>(quantAxisSize),
                          "Wrong zeroPoints size '{0}', expected '{1}'", zeroPoints.size(), quantAxisSize);

        // Outermost loop goes through the volume of the outer dimensions.
        // Middle loop goes through the quantized axis. Scale/ZP are updated based on this index.
        // Innermost loop goes through the volume of the innermost dimensions, which share the same quantization
        // parameters.
        if (const auto quantilePerAxisType = mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(qElemType)) {
            const auto quantilesLUT = quantilePerAxisType.getQuantiles();
            loop_3d(LoopExecPolicy::Parallel, getContext(), outerSize, quantAxisSize, innerSize,
                    [&](int64_t outerInd, int64_t quantAxisInd, int64_t innerInd) {
                        const auto scale = scales[quantAxisInd];
                        const auto zp = zeroPoints[quantAxisInd];
                        const auto idx = outerInd * quantAxisTotalSize + quantAxisInd * innerSize + innerInd;
                        realVals[idx] = dequantizeDouble(quantilesLUT[qVals[idx]], scale, zp);
                    });

        } else {
            loop_3d(LoopExecPolicy::Parallel, getContext(), outerSize, quantAxisSize, innerSize,
                    [&](int64_t outerInd, int64_t quantAxisInd, int64_t innerInd) {
                        const auto scale = scales[quantAxisInd];
                        const auto zp = zeroPoints[quantAxisInd];
                        const auto idx = outerInd * quantAxisTotalSize + quantAxisInd * innerSize + innerInd;
                        realVals[idx] = dequantize(qVals[idx], scale, zp);
                    });
        }

    } else {
        VPUX_THROW("Unsupported Quantized Type '{0}'", qElemType);
    }

    return output;
}

//
// DequantizeAttr::getStableHashValue
//

llvm::hash_code vpux::Const::DequantizeAttr::getStableHashValue() const {
    return llvm::hash_value(getMnemonic());
}
