//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/expand_activation_channels_strategy.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/expand_activation_channels.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

namespace vpux::IE::arch37xx {

void ExpandActivationChannelsStrategy::addTargets(mlir::ConversionTarget& target) {
    const auto isLegal = [&](mlir::Operation* op) {
        if (!_seOpsEnabled && mlir::isa<IE::SEOpInterface>(op)) {
            return true;
        }

        if (auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
            return iface.verifyChannels().succeeded();
        }

        return true;
    };

    target.markUnknownOpDynamicallyLegal(isLegal);
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ExpandOp, IE::SliceOp>();
    target.addLegalOp<IE::MultiplyOp, IE::SubtractOp>();
}

void ExpandActivationChannelsStrategy::addPatterns(mlir::RewritePatternSet& patterns) {
    auto ctx = patterns.getContext();
    patterns.add<IE::MaxPoolRewriter>(ctx, _log);
    patterns.add<IE::AvgPoolRewriter>(ctx, _log);
    patterns.add<IE::EltwiseRewriter<IE::AddOp>>(ctx, _log);
    patterns.add<IE::ConvolutionRewriter>(ctx, _log);
    patterns.add<IE::GroupConvolutionRewriter>(ctx, _log);
    patterns.add<IE::MatMulRewriter>(ctx, _log);
    patterns.add<IE::SoftMaxRewriter>(ctx, _log);

    if (_seOpsEnabled) {
        patterns.add<IE::InterpolateRewriter>(ctx, _log);
        patterns.add<IE::TransposedConvolutionRewriter>(ctx, _log);
        patterns.add<IE::PadRewriter>(ctx, _log);
    }
}

}  // namespace vpux::IE::arch37xx
