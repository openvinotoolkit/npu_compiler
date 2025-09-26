//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/convert_to_palletization_lut.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

bool vpux::IE::isLegalTensorElemForPalletization(mlir::Type elementType, const bool convertOnlyAsymmetricZp,
                                                 const bool allowPerChannelZp) {
    const auto isQuantizedTypeLegal = [convertOnlyAsymmetricZp,
                                       allowPerChannelZp](mlir::quant::QuantizedType quantizedType) -> bool {
        auto storageTypeInt = mlir::dyn_cast<mlir::IntegerType>(quantizedType.getStorageType());
        // conversion rule for u2/i2 and u4/i4; when convertOnlyAsymmetricZp == true don't execute the
        // conversion if the zp is symmetric
        const bool symmetryConvertCondition =
                (storageTypeInt != nullptr) && (convertOnlyAsymmetricZp ? !isSymmetricZeroPoint(quantizedType) : true);
        // If the quantisation is per channel instead of per tensor, the conversion can be supported if all the zp are
        // the same in value (checked by getSingleZeroPointOrFail(quantizedType))
        const bool isQuantizationSchemeAllowed =
                allowPerChannelZp || mlir::succeeded(getSingleZeroPointOrFail(quantizedType));
        const bool isConversionRequired =
                symmetryConvertCondition && storageTypeInt.getWidth() <= 4 && isQuantizationSchemeAllowed;
        return !isConversionRequired;
    };

    if (mlir::isa<mlir::quant::QuantileQuantizedType, mlir::quant::QuantileQuantizedPerAxisType>(elementType)) {
        // Evaluate whether introducing a zp subtraction for quant.quantile<u4:i8:f16, ....> cases (which are currently
        // not really used)
        return true;
    } else if (mlir::isa<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(elementType)) {
        return isQuantizedTypeLegal(mlir::cast<mlir::quant::QuantizedType>(elementType));
    }

    return true;
}
