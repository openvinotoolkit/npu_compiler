//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/convert_divide_to_multiply_benefit_strategy.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::IE {

std::unique_ptr<IConvertDivideToMultiplyBenefitStrategy> createConvertDivideToMultiplyBenefitStrategy(
        mlir::func::FuncOp funcOp);

}  // namespace vpux::IE
