//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/impl/convert_to_mixed_precision_strategy.hpp"
#include "vpux/compiler/dialect/IE/interfaces/convert_to_mixed_precision_strategy.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/convert_to_mixed_precision.hpp"

namespace vpux::IE::arch50xx {

void ConvertToMixedPrecisionStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    // E#67754 - MaxPool is omitted intentionally because it generates accuracy issues.
    patterns.add<vpux::IE::FloatOutConvRewriter>(ctx, log);
    patterns.add<vpux::IE::FloatOutGroupConvRewriter>(ctx, log);
    patterns.add<vpux::IE::FloatOutAddRewriter>(ctx, true, log);
    patterns.add<vpux::IE::FloatOutTransposedConvRewriter>(ctx, log);
    patterns.add<vpux::IE::FloatOutMatMulRewriter>(ctx, log);

    patterns.add<vpux::IE::FloatOutAvgPoolRewriter>(ctx, log);
    patterns.add<vpux::IE::QuantizeWithNCERewriter>(ctx, log);

    // Patterns for mixed precision of float input and quant weights
    if (_enableFloatInQuantWeightsMixedMode) {
        patterns.add<vpux::IE::MixedFloatInQuantWeightsRewriter<IE::ConvolutionOp>>(ctx, log);
        patterns.add<vpux::IE::MixedFloatInQuantWeightsRewriter<IE::GroupConvolutionOp>>(ctx, log);
        patterns.add<vpux::IE::MixedFloatInQuantWeightsRewriter<IE::MatMulOp>>(ctx, log);
    }
}

}  // namespace vpux::IE::arch50xx
