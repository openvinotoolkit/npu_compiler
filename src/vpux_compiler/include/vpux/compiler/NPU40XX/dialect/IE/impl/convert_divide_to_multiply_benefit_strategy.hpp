//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/convert_divide_to_multiply_benefit_strategy.hpp"

namespace vpux::IE::arch40xx {

class ConvertDivideToMultiplyBenefitStrategy final : public vpux::IE::IConvertDivideToMultiplyBenefitStrategy {
public:
    mlir::LogicalResult isNonConstBeneficialConversion(IE::DivideOp divideOp) override;

};  // class ConvertDivideToMultiplyBenefitStrategy

}  // namespace vpux::IE::arch40xx
