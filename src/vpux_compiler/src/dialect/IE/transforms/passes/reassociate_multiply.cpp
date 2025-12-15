//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_REASSOCIATEMULTIPLY
#define GEN_PASS_DEF_REASSOCIATEMULTIPLY
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//    1x32x1024x1024      1x32x1x1                            1x32x1x1        1x1x1x1
//              \          /                                       \        /
//                Multiply       1x1x1x1   =>   1x32x1024x1024       Multiply
//                      \         /                          \       /
//                        Multiply                            Multiply

class MultiplyRewriter final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    MultiplyRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        this->setDebugName("MultiplyRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp op, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value getInputWithSmallerSize(mlir::Value input1, mlir::Value input2) const;
    Logger _log;
};

mlir::Value MultiplyRewriter::getInputWithSmallerSize(mlir::Value input1, mlir::Value input2) const {
    return getBoundedShape(input1).totalSize() > getBoundedShape(input2).totalSize() ? input2 : input1;
}

mlir::LogicalResult MultiplyRewriter::matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const {
    auto origOpLhs = origOp.getInput1();
    auto origOpRhs = origOp.getInput2();
    if (getBoundedShape(origOpLhs).totalSize() == getBoundedShape(origOpRhs).totalSize()) {
        return matchFailed(_log, rewriter, origOp, "two inputs of multiply have the same size");
    }

    auto origOpSmallSizeInput = getInputWithSmallerSize(origOpLhs, origOpRhs);
    auto origOpLargeSizeInput = origOpLhs == origOpSmallSizeInput ? origOpRhs : origOpLhs;

    auto producerMultiplyOp = origOpLargeSizeInput.getDefiningOp<IE::MultiplyOp>();
    if (producerMultiplyOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "no producer multiply");
    }

    if (!producerMultiplyOp.getOutput().hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "producer multiply has multi-uses");
    }

    if (producerMultiplyOp.getPostOpAttr() != nullptr || producerMultiplyOp.getClampAttr() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "producer multiply has post op attr or clamp attr");
    }

    auto producerLhs = producerMultiplyOp.getInput1();
    auto producerRhs = producerMultiplyOp.getInput2();
    if (getBoundedShape(producerLhs).totalSize() == getBoundedShape(producerRhs).totalSize()) {
        return matchFailed(_log, rewriter, origOp, "two inputs of producer multiply have the same size");
    }

    auto producerSmallSizeInput = getInputWithSmallerSize(producerLhs, producerRhs);
    auto producerLargeSizeInput = producerLhs == producerSmallSizeInput ? producerRhs : producerLhs;

    auto smallestSizeInput = getInputWithSmallerSize(origOpSmallSizeInput, producerSmallSizeInput);
    auto middleSizeInput = smallestSizeInput == origOpSmallSizeInput ? producerSmallSizeInput : origOpSmallSizeInput;
    const auto validBroadcastShape =
            IE::broadcastEltwiseShape(getBoundedShape(middleSizeInput), getBoundedShape(smallestSizeInput),
                                      IE::AutoBroadcastType::NUMPY, origOp.getLoc());
    if (mlir::failed(validBroadcastShape)) {
        return matchFailed(_log, rewriter, origOp, "broadcast input failed");
    }

    const auto newMultiplyOutShape = Shape(validBroadcastShape.value());
    auto producerOutput = producerMultiplyOp.getOutput();
    if (newMultiplyOutShape.totalSize() >= getBoundedShape(producerOutput).totalSize()) {
        return matchFailed(_log, rewriter, origOp, "new multiply output is too big");
    }

    auto multiply_1 = rewriter.create<IE::MultiplyOp>(appendLoc(origOp.getLoc(), "_new_multiply_1"), middleSizeInput,
                                                      smallestSizeInput, IE::AutoBroadcastType::NUMPY, nullptr, nullptr,
                                                      nullptr, nullptr);

    auto multiply_2 = rewriter.create<IE::MultiplyOp>(
            appendLoc(origOp.getLoc(), "_new_multiply_2"), producerLargeSizeInput, multiply_1.getOutput(),
            origOp.getAutoBroadcastAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputPaddingAttr(),
            origOp.getInputPaddingAttr());

    rewriter.replaceAllUsesWith(origOp.getOutput(), multiply_2.getOutput());

    return mlir::success();
}

//
// ReassociateMultiplyPass
//

class ReassociateMultiplyPass final : public IE::impl::ReassociateMultiplyBase<ReassociateMultiplyPass> {
public:
    explicit ReassociateMultiplyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ReassociateMultiplyPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MultiplyRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createReassociateMultiplyPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createReassociateMultiplyPass(Logger log) {
    return std::make_unique<ReassociateMultiplyPass>(log);
}
