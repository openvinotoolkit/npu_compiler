//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/convert_to_mixed_precision_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/convert_to_mixed_precision.hpp"

namespace vpux::IE::arch37xx {

void ConvertToMixedPrecisionStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    // E#67754 - MaxPool is omitted intentionally because it generates accuracy issues.
    patterns.add<vpux::IE::FloatOutConvRewriter>(ctx, IE::arch37xx::isMixPrecisionSupported, log);
    patterns.add<vpux::IE::FloatOutGroupConvRewriter>(ctx, IE::arch37xx::isMixPrecisionSupported, log);
    patterns.add<vpux::IE::FloatOutAddRewriter>(ctx, IE::arch37xx::isMixPrecisionSupported, true, log);
    patterns.add<vpux::IE::FloatOutTransposedConvRewriter>(ctx, IE::arch37xx::isMixPrecisionSupported, log);
    patterns.add<vpux::IE::FloatOutMatMulRewriter>(ctx, IE::arch37xx::isMixPrecisionSupported, log);

    patterns.add<vpux::IE::FloatOutAvgPoolRewriter>(ctx, log);
    patterns.add<vpux::IE::QuantizeWithNCERewriter>(ctx, IE::arch37xx::isMixPrecisionSupported,
                                                    IE::arch37xx::checkPostOp, false, log);

    // Patterns for mixed precision of float input and quant weights
    if (_enableFloatInQuantWeightsMixedMode) {
        patterns.add<vpux::IE::MixedFloatInQuantWeightsRewriter<IE::ConvolutionOp>>(
                ctx, IE::arch37xx::isMixPrecisionSupported, log);
        patterns.add<vpux::IE::MixedFloatInQuantWeightsRewriter<IE::GroupConvolutionOp>>(
                ctx, IE::arch37xx::isMixPrecisionSupported, log);
    }
}

}  // namespace vpux::IE::arch37xx
