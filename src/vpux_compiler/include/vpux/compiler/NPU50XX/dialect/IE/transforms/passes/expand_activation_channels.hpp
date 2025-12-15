//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/transforms/passes/expand_activation_channels.hpp"
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
