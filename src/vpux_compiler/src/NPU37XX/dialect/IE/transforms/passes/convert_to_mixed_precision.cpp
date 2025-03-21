//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/convert_to_mixed_precision.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE::arch37xx {
#define GEN_PASS_DECL_CONVERTTOMIXEDPRECISION
#define GEN_PASS_DEF_CONVERTTOMIXEDPRECISION
#include "vpux/compiler/NPU37XX/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE::arch37xx

using namespace vpux;

namespace {

//
// ConvertToMixedPrecisionPass
//

class ConvertToMixedPrecisionPass final :
        public IE::arch37xx::impl::ConvertToMixedPrecisionBase<ConvertToMixedPrecisionPass> {
public:
    explicit ConvertToMixedPrecisionPass(const bool enableFloatInQuantWeightsMixedMode, Logger log)
            : _enableFloatInQuantWeightsMixedMode(enableFloatInQuantWeightsMixedMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _enableFloatInQuantWeightsMixedMode;
};

mlir::LogicalResult ConvertToMixedPrecisionPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (enableFloatInQuantWeightsMixedMode.hasValue()) {
        _enableFloatInQuantWeightsMixedMode = enableFloatInQuantWeightsMixedMode.getValue();
    }

    return mlir::success();
}

void ConvertToMixedPrecisionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    // E#67754 - MaxPool is omitted intentionally because it generates accuracy issues.
    patterns.add<vpux::IE::FloatOutConvRewriter>(&ctx, IE::arch37xx::isMixPrecisionSupported, _log);
    patterns.add<vpux::IE::FloatOutGroupConvRewriter>(&ctx, IE::arch37xx::isMixPrecisionSupported, _log);
    patterns.add<vpux::IE::FloatOutAddRewriter>(&ctx, IE::arch37xx::isMixPrecisionSupported, true, _log);
    patterns.add<vpux::IE::FloatOutTransposedConvRewriter>(&ctx, IE::arch37xx::isMixPrecisionSupported, _log);
    patterns.add<vpux::IE::FloatOutMatMulRewriter>(&ctx, IE::arch37xx::isMixPrecisionSupported, _log);

    patterns.add<vpux::IE::FloatOutAvgPoolRewriter>(&ctx, _log);
    patterns.add<vpux::IE::QuantizeWithNCERewriter>(&ctx, IE::arch37xx::isMixPrecisionSupported,
                                                    IE::arch37xx::checkPostOp, false, _log);

    // Patterns for mixed precision of float input and quant weights
    if (_enableFloatInQuantWeightsMixedMode) {
        patterns.add<vpux::IE::MixedFloatInQuantWeightsRewriter<IE::ConvolutionOp>>(
                &ctx, IE::arch37xx::isMixPrecisionSupported, _log);
        patterns.add<vpux::IE::MixedFloatInQuantWeightsRewriter<IE::GroupConvolutionOp>>(
                &ctx, IE::arch37xx::isMixPrecisionSupported, _log);
    }

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToMixedPrecision
//

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createConvertToMixedPrecision(
        const bool enableFloatInQuantWeightsMixedMode, Logger log) {
    return std::make_unique<ConvertToMixedPrecisionPass>(enableFloatInQuantWeightsMixedMode, log);
}
