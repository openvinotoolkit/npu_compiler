//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/impl/convert_weights_to_unsigned_strategy.hpp"
#include "vpux/compiler/dialect/IE/interfaces/convert_weights_to_unsigned_strategy.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

using namespace vpux;

mlir::quant::QuantizedType vpux::IE::arch50xx::ConvertWeightsToUnsignedStrategy::tryChangeStorageTypeToUnsigned(
        mlir::quant::QuantizedType quantType) const {
    if (llvm::dyn_cast<mlir::IntegerType>(quantType.getStorageType()) && quantType.isSigned() &&
        quantType.getStorageTypeIntegralWidth() == 8) {
        return IE::changeStorageTypeToUnsigned(quantType, getUInt8Type(quantType.getContext()));
    }

    return quantType;
}
