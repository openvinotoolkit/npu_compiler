//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <vpux/compiler/utils/rewriter.hpp>

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_BROADCASTINPUTFORADD
#define GEN_PASS_DEF_BROADCASTINPUTFORADD
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// BroadcastInputRewriter
//

class BroadcastInputRewriter final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    BroadcastInputRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        setDebugName("BroadcastInputRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::TypedValue<mlir::RankedTensorType> createBroadcastInput(mlir::PatternRewriter& rewriter,
                                                                  mlir::MLIRContext* ctx, mlir::Location loc,
                                                                  mlir::Value broadcastInput,
                                                                  ShapeRef targetShape) const;
    Logger _log;
};

mlir::TypedValue<mlir::RankedTensorType> BroadcastInputRewriter::createBroadcastInput(mlir::PatternRewriter& rewriter,
                                                                                      mlir::MLIRContext* ctx,
                                                                                      mlir::Location loc,
                                                                                      mlir::Value broadcastInput,
                                                                                      ShapeRef targetShape) const {
    auto targetShapeConst = vpux::IE::createShapeConstForBroadCast(rewriter, ctx, appendLoc(loc, "shape"), targetShape);
    return rewriter
            .create<IE::BroadcastOp>(loc, broadcastInput, targetShapeConst, /*axes_mapping*/ nullptr,
                                     IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY))
            .getResult();
}

mlir::LogicalResult BroadcastInputRewriter::matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    const auto ctx = origOp->getContext();
    const auto loc = origOp->getLoc();

    const auto lhsShape = origOp.getInput1().getType().cast<vpux::NDTypeInterface>().getShape();
    const auto rhsShape = origOp.getInput2().getType().cast<vpux::NDTypeInterface>().getShape();
    const auto outputShape = origOp.getOutput().getType().cast<vpux::NDTypeInterface>().getShape();

    if (lhsShape.size() != 4) {
        _log.trace("Only support 4D tensor, but got {0}D", lhsShape.size());
        return mlir::failure();
    }

    if (lhsShape == rhsShape) {
        _log.trace("Inputs have same shape, no need for broadcast");
        return mlir::failure();
    }

    const auto findTrivialBiasInput = [&](IE::AddOp origOp) {
        const auto biasInput = (origOp.getInput1().getType().cast<vpux::NDTypeInterface>() ==
                                origOp.getOutput().getType().cast<vpux::NDTypeInterface>())
                                       ? origOp.getInput2()
                                       : origOp.getInput1();
        const auto biasShape = biasInput.getType().cast<vpux::NDTypeInterface>().getShape();

        const auto trivialDimExceptDimC = [](ShapeRef inputShape) -> bool {
            return inputShape[Dims4D::Act::N] == 1 && inputShape[Dims4D::Act::H] == 1 &&
                   inputShape[Dims4D::Act::W] == 1;
        };

        return mlir::succeeded(IE::getConstParentOp(biasInput)) && trivialDimExceptDimC(biasShape);
    };

    // For constant bias input like 1xCx1x1xf16, convert to ScaleShift can get better performance
    // Otherwise we need to broadcast input to let it meet eltwise Add requirement.
    if (findTrivialBiasInput(origOp)) {
        _log.trace("Can convert to ScaleShift, no need to broadcast");
        return mlir::failure();
    }

    const auto doesInputNeedBroadCast = [&](mlir::Value input) {
        return getShape(input) != outputShape;
    };

    auto lhsInput = origOp.getInput1();
    if (doesInputNeedBroadCast(origOp.getInput1())) {
        lhsInput =
                createBroadcastInput(rewriter, ctx, appendLoc(loc, "broadcast_lhs"), origOp.getInput1(), outputShape);
    }

    auto rhsInput = origOp.getInput2();
    if (doesInputNeedBroadCast(origOp.getInput2())) {
        rhsInput =
                createBroadcastInput(rewriter, ctx, appendLoc(loc, "broadcast_rhs"), origOp.getInput2(), outputShape);
    }

    auto addOp = rewriter.replaceOpWithNewOp<IE::AddOp>(origOp, lhsInput, rhsInput, origOp.getAutoBroadcast(),
                                                        origOp.getPostOpAttr(), origOp.getClampAttr(),
                                                        origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());
    extendOpLoc(addOp, "as_add");

    return mlir::success();
}

//
// BroadcastInputForAddPass
//
class BroadcastInputForAddPass final : public IE::impl::BroadcastInputForAddBase<BroadcastInputForAddPass> {
public:
    explicit BroadcastInputForAddPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void BroadcastInputForAddPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<BroadcastInputRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createBroadcastInputForAddPass(Logger log) {
    return std::make_unique<BroadcastInputForAddPass>(log);
}
