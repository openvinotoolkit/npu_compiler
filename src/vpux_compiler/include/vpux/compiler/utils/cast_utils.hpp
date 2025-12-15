//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>

namespace vpux {

mlir::LogicalResult isQuantizeCastValid(mlir::Location loc, mlir::Type srcType, mlir::Type dstType);

}  // namespace vpux
