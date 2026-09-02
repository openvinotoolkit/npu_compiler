//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/convert_weights_to_unsigned_strategy.hpp"

namespace vpux::IE::arch50xx {

// Default strategy: converts I8 weights to U8 only.
class ConvertWeightsToUnsignedStrategy : public IConvertWeightsToUnsignedStrategy {
public:
    mlir::quant::QuantizedType tryChangeStorageTypeToUnsigned(mlir::quant::QuantizedType quantType) const override;
};

}  // namespace vpux::IE::arch50xx
