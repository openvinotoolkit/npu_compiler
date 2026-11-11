//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/Types.h>

namespace vpux::IE::arch60xx {

// Returns false when type conversion is required;
bool isLegalTensorElem60XX(mlir::Type elementType);

// Converts a quantized weight type to a QuantileType backed by a pre-computed LUT.
mlir::quant::QuantizedType changeWeightTypeToLUT(mlir::quant::QuantizedType originWeightType, mlir::Type actType);

class ConvertToPalletizationLUTStrategy : public IConversionPassStrategy {
public:
    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override final;
    void markOpLegality(mlir::ConversionTarget& target, Logger& log) const override final;
};

}  // namespace vpux::IE::arch60xx
