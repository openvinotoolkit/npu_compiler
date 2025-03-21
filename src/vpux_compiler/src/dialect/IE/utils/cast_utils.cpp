//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/cast_utils.hpp"

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux {

mlir::LogicalResult isQuantizeCastValid(mlir::Location loc, mlir::Type srcType, mlir::Type dstType) {
    const auto srcBitSize = getElemTypeSize(srcType).count();
    const auto dstBitSize = getElemTypeSize(dstType).count();

    if (srcBitSize != dstBitSize) {
        return errorAt(loc, "Src type bit-width: {0} is different from dst type bit-width: {1}", srcBitSize,
                       dstBitSize);
    }

    if (srcBitSize >= 16) {
        return errorAt(loc, "Src and dst types must be low precision types or quantized types");
    }

    // Admitting all cases except I1, as currently we're treating it as pure I1 data type
    // not requiring any quant cast
    if (srcBitSize < 2 || !isPowerOfTwo(srcBitSize)) {
        return errorAt(loc, "Src type bit size: {0} is not a power of two", srcBitSize);
    }

    if (!mlir::isa<mlir::quant::QuantizedType>(srcType) && !mlir::isa<mlir::quant::QuantizedType>(dstType)) {
        return errorAt(loc, "Either src or dst type must be a quantized type");
    }

    if (mlir::isa<mlir::quant::QuantileQuantizedType, mlir::quant::QuantileQuantizedPerAxisType,
                  type::QuantileFloatType>(srcType) !=
        mlir::isa<mlir::quant::QuantileQuantizedType, mlir::quant::QuantileQuantizedPerAxisType,
                  type::QuantileFloatType>(dstType)) {
        return errorAt(loc, "Quantile types can only be casted to other quantile types");
    }

    if ((isFloat8(srcType) && srcType != mlir::cast<mlir::quant::QuantizedType>(dstType).getStorageType()) ||
        (isFloat8(dstType) && dstType != mlir::cast<mlir::quant::QuantizedType>(srcType).getStorageType())) {
        return errorAt(loc, "Low precision float types can only be casted to types of the same bit format");
    }

    return mlir::success();
}

}  // namespace vpux
