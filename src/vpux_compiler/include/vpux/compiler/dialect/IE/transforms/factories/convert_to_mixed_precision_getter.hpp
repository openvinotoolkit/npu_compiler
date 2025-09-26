//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/convert_to_mixed_precision_strategy.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::IE {

std::unique_ptr<IConvertToMixedPrecisionStrategy> createConvertToMixedPrecisionStrategy(
        mlir::func::FuncOp funcOp, bool enableFloatInQuantWeightsMixedMode);

}  // namespace vpux::IE
