//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <array>
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/Types.h"
#include "vpux/utils/core/array_ref.hpp"

namespace vpux {
namespace type {

/// NF4 quantile type helper.
/// NF4 is represented as mlir::quant::QuantileType with NF4-specific quantile values.
/// This struct provides static utilities for creating and recognising NF4 instances.
struct NF4Type {
public:
    /// Create a QuantileType with the NF4 quantile table.
    /// If quantiles is non-empty it must contain exactly 16 values.
    static mlir::quant::QuantileType get(mlir::MLIRContext* ctx, mlir::Type storageType, mlir::Type quantileType,
                                         ArrayRef<double> quantiles = {});

    static ArrayRef<double> getSpecQuantiles();

    /// Returns true when type is a QuantileType whose quantile table matches the NF4 spec.
    static bool classof(mlir::Type type);
};

}  // namespace type
}  // namespace vpux
