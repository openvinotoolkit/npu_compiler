//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/IE/impl/convert_divide_to_multiply_benefit_strategy.hpp"
#include <mlir/Support/LLVM.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::arch40xx::ConvertDivideToMultiplyBenefitStrategy::isNonConstBeneficialConversion(
        IE::DivideOp divideOp) {
    auto outputShape = getShape(divideOp.getOutput());
    constexpr int64_t THRESHOLD_FOR_BENEFICIAL_CONVERSION = 4096;
    if (outputShape.totalSize() < THRESHOLD_FOR_BENEFICIAL_CONVERSION) {
        return mlir::failure();
    }

    return mlir::success();
}
