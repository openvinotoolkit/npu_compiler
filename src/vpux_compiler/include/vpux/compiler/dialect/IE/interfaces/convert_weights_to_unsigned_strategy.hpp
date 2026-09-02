//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {

// Shift a signed quantized type to its unsigned counterpart (e.g. I8 -> U8, I4 -> U4).
// Adjusts zero points and storage range by subtracting storageTypeMin.
mlir::quant::QuantizedType changeStorageTypeToUnsigned(mlir::quant::QuantizedType originQType,
                                                       mlir::Type unsignedStorageType);

// Strategy interface for the ConvertWeightsToUnsigned pass.
// Allows platform-specific configuration of which weight type conversions are applied.
// Default: I8 -> U8 only.
// On subbyte-capable platforms: I8 -> U8 and I4 -> U4 (same-signedness).
class IConvertWeightsToUnsignedStrategy {
public:
    virtual ~IConvertWeightsToUnsignedStrategy() = default;

    // Return the unsigned counterpart of a signed quantized weights type when this platform converts
    // it (e.g. I8 -> U8, symmetric I4 -> U4). Return the type unchanged when no conversion applies.
    // Single source of truth for keeping weights and activation signedness aligned.
    virtual mlir::quant::QuantizedType tryChangeStorageTypeToUnsigned(mlir::quant::QuantizedType quantType) const = 0;

    // Add platform-specific weight type conversions to the TypeConverter.
    // Shared implementation that delegates to tryChangeStorageTypeToUnsigned.
    void addTypeConversions(mlir::TypeConverter& typeConverter) const;
};

}  // namespace vpux::IE
