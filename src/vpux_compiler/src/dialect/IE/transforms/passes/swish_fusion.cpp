//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_SWISHFUSION
#define GEN_PASS_DEF_SWISHFUSION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// SwishFusion
//

class SwishFusion final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    SwishFusion(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        setDebugName("SwishFusion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwishFusion::matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.debug("Got Multiply '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    // quantized case
    //
    //    FakeQuantize                                FakeQuantize
    //        /  \                                          |
    //       /   Sigmoid                                    |
    //      |      \                 ->                     |
    //      |      FakeQuantize                           Swish
    //       \      /                                       |
    //        Multiply                                      |
    //           |                                          |
    //      FakeQuantize                                FakeQuantize

    // dequantized case
    //
    //    ParentTensor                                  ParentTensor
    //         |     \                                        |
    //      Sigmoid   |          ->                           |
    //         |     /                                      Swish
    //      Multiply

    // dequantized sigmoid case
    IE::SigmoidOp sigmoidOp = origOp.getInput1().getDefiningOp<IE::SigmoidOp>();
    if (sigmoidOp == nullptr) {
        sigmoidOp = origOp.getInput2().getDefiningOp<IE::SigmoidOp>();
    }

    // quantized sigmoid case
    if (sigmoidOp == nullptr) {
        for (auto input : {origOp.getInput1(), origOp.getInput2()}) {
            if (auto fqOp = input.getDefiningOp<IE::FakeQuantizeOp>()) {
                if ((sigmoidOp = fqOp.getInput().getDefiningOp<IE::SigmoidOp>())) {
                    break;
                }
            }
        }
    }

    if (sigmoidOp == nullptr) {
        return matchFailed(rewriter, origOp, "No Sigmoid operation connected to Multiply");
    }

    if (sigmoidOp.getInput() != origOp.getInput1() && sigmoidOp.getInput() != origOp.getInput2()) {
        return matchFailed(rewriter, origOp, "No Sigmoid operation connected to Multiply");
    }

    auto swishOp = rewriter.create<IE::SwishOp>(origOp.getLoc(), sigmoidOp.getInput(), nullptr, nullptr);

    _log.debug("Replace '{0}' and '{1}' with new op '{2}'", origOp.getLoc(), sigmoidOp.getLoc(), swishOp);
    rewriter.replaceOp(origOp, swishOp.getResult());

    return mlir::success();
}

//
// SwishFusionPass
//

class SwishFusionPass final : public IE::impl::SwishFusionBase<SwishFusionPass> {
public:
    explicit SwishFusionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void SwishFusionPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwishFusion>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSwishFusionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSwishFusionPass(Logger log) {
    return std::make_unique<SwishFusionPass>(log);
}
