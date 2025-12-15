//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/impl/expand_activation_channels_strategy.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/expand_activation_channels.hpp"

#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"

// For ReduceRewriter
#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"

namespace vpux::arch50xx {

//
// ReduceRewriter
//

template <typename ConcreteOp>
class ReduceRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ReduceRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("ReduceRewriter");
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult ReduceRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Reduce layer at '{1}'", this->getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t inChanPadEnd,
                               int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);
        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        auto [inputPaddingAttr, outputPaddingAttr] =
                getPaddingAttributes(origOp, expandedInput, inChanPadEnd, outPadAfter);

        auto axes = getIntArrayAttr(this->getContext(), IE::extractAxes(origOp->getLoc(), origOp));
        return rewriter.create<ConcreteOp>(origOp.getLoc(), newOutputType, expandedInput,
                                           /*axes*/ nullptr,
                                           /*axes_value*/ axes, /*keep_dims*/ true, outputPaddingAttr,
                                           inputPaddingAttr);
    };

    return IE::generalRewrite(origOp, rewriter, opCreator, IE::extractMeaningfulOutput, _log.nest());
}

}  // namespace vpux::arch50xx

namespace vpux::IE::arch50xx {

void ExpandActivationChannelsStrategy::addTargets(mlir::ConversionTarget& target) {
    const auto isLegal = [&](mlir::Operation* op) {
        // It's sometimes beneficial to align Interpolate even when it's running on Shave
        if (!_seOpsEnabled && mlir::isa<IE::SEOpInterface>(op) && !mlir::isa<IE::InterpolateOp>(op)) {
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
    patterns.add<IE::EltwiseRewriter<IE::MultiplyOp>>(ctx, _log);
    patterns.add<IE::EltwiseRewriter<IE::SubtractOp>>(ctx, _log);
    patterns.add<vpux::arch50xx::ReduceRewriter<IE::ReduceMeanOp>>(ctx, _log);
    patterns.add<vpux::arch50xx::ReduceRewriter<IE::ReduceSumOp>>(ctx, _log);
    patterns.add<IE::InterpolateRewriter>(ctx, _log);
    patterns.add<IE::SDPAExtendedRewriter>(ctx, _log);
    patterns.add<IE::FlashSDPARewriter>(ctx, _log);

    if (_seOpsEnabled) {
        patterns.add<IE::TransposedConvolutionRewriter>(ctx, _log);
        patterns.add<IE::PadRewriter>(ctx, _log);
    }
}

}  // namespace vpux::IE::arch50xx
