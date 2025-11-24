//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"

namespace vpux {
namespace IE {

class IConvertDivideToMultiplyBenefitStrategy {
public:
    virtual ~IConvertDivideToMultiplyBenefitStrategy() = default;

    virtual mlir::LogicalResult isNonConstBeneficialConversion(IE::DivideOp divideOp) = 0;
};

}  // namespace IE
}  // namespace vpux
