//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/convert_divide_to_multiply_benefit_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/convert_divide_to_multiply_benefit_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/impl/convert_divide_to_multiply_benefit_strategy.hpp"
#include "vpux/compiler/dialect/IE/interfaces/convert_divide_to_multiply_benefit_strategy.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::IE {

std::unique_ptr<IConvertDivideToMultiplyBenefitStrategy> createConvertDivideToMultiplyBenefitStrategy(
        mlir::func::FuncOp funcOp) {
    const auto arch = config::getArch(funcOp);

    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<arch37xx::ConvertDivideToMultiplyBenefitStrategy>();
    default:
        return std::make_unique<arch40xx::ConvertDivideToMultiplyBenefitStrategy>();
    }
}

}  // namespace vpux::IE
