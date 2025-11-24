//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/convert_divide_to_multiply_benefit_strategy.hpp"
#include <mlir/Support/LLVM.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::arch37xx::ConvertDivideToMultiplyBenefitStrategy::isNonConstBeneficialConversion(
        IE::DivideOp divideOp) {
    // Experimental threshold data
    constexpr int64_t SIZE_RATIO_THRESHOLD = 1024;
    auto divisorShape = getShape(divideOp.getInput2());
    auto outputShape = getShape(divideOp.getOutput());
    // The transformation will create a new Divide(1, divisor)
    // It's beneficial when the new Divide will be much smaller than the original Divide
    if (outputShape.totalSize() / divisorShape.totalSize() < SIZE_RATIO_THRESHOLD) {
        return mlir::failure();
    }

    constexpr int64_t THRESHOLD_FOR_BENEFICIAL_CONVERSION = 4096;
    if (outputShape.totalSize() < THRESHOLD_FOR_BENEFICIAL_CONVERSION) {
        return mlir::failure();
    }

    return mlir::success();
}
