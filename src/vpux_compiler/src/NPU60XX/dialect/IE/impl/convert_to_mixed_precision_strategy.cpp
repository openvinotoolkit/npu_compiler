//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/IE/impl/convert_to_mixed_precision_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/interfaces/convert_to_mixed_precision_strategy.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/convert_to_mixed_precision.hpp"

namespace vpux::IE::arch60xx {

void ConvertToMixedPrecisionStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    arch50xx::ConvertToMixedPrecisionStrategy::addPatterns(patterns, log);

    // Patterns for mixed precision of float input and quant weights
    if (_enableFloatInQuantWeightsMixedMode) {
        patterns.add<vpux::IE::MixedFloatInQuantWeightsWithDynamicDequantizeRewriter<IE::ConvolutionOp>>(ctx, log);
    }
}

}  // namespace vpux::IE::arch60xx
