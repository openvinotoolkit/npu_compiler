//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

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
    Logger _log;
};

mlir::LogicalResult BroadcastInputRewriter::matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    const auto loc = origOp->getLoc();

    const auto lhsShape = mlir::cast<vpux::NDTypeInterface>(origOp.getInput1().getType()).getShape();
    const auto rhsShape = mlir::cast<vpux::NDTypeInterface>(origOp.getInput2().getType()).getShape();
    const auto outputShape = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getShape();

    if (lhsShape.size() != 4) {
        _log.trace("Only support 4D tensor, but got {0}D", lhsShape.size());
        return mlir::failure();
    }

    if (lhsShape == rhsShape) {
        _log.trace("Inputs have same shape, no need for broadcast");
        return mlir::failure();
    }

    const auto findTrivialBiasInput = [&](IE::AddOp origOp) {
        const auto biasInput = (mlir::cast<vpux::NDTypeInterface>(origOp.getInput1().getType()) ==
                                mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()))
                                       ? origOp.getInput2()
                                       : origOp.getInput1();
        const auto biasShape = mlir::cast<vpux::NDTypeInterface>(biasInput.getType()).getShape();

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

    const auto doesInputNeedBroadCast = [&](ShapeRef inputShape) {
        return inputShape != outputShape;
    };

    const auto lhsBroadcast = doesInputNeedBroadCast(lhsShape);
    const auto lhsDynBroadcast = lhsBroadcast ? rhsShape.isDynamic() : false;
    const auto rhsBroadcast = doesInputNeedBroadCast(rhsShape);
    const auto rhsDynBroadcast = rhsBroadcast ? lhsShape.isDynamic() : false;

    // DynamicBroadcast requires an inference-time determined target shape. If only one of the inputs needs to be
    // broadcasted, the target shape can be obtained from the other input (using ShapeOf). If both inputs need broadcast
    // we need some other mechanism to determine the targed shape during inference.
    VPUX_THROW_WHEN((lhsDynBroadcast && rhsBroadcast) || (lhsBroadcast && rhsDynBroadcast),
                    "Cross-broadcast is not currently supported for dynamic shapes: {0} x {1} -> {2}", lhsShape,
                    rhsShape, outputShape);

    mlir::Value lhsInput = origOp.getInput1();
    mlir::Value rhsInput = origOp.getInput2();

    if (lhsBroadcast) {
        if (lhsDynBroadcast) {
            lhsInput = IE::createDynamicBroadcast(rewriter, appendLoc(loc, "broadcast_rhs"), lhsInput, rhsInput);
        } else {
            lhsInput = IE::createBroadcast(rewriter, appendLoc(loc, "broadcast_lhs"), lhsInput, outputShape);
        }
    }

    if (rhsBroadcast) {
        if (rhsDynBroadcast) {
            rhsInput = IE::createDynamicBroadcast(rewriter, appendLoc(loc, "broadcast_rhs"), rhsInput, lhsInput);
        } else {
            rhsInput = IE::createBroadcast(rewriter, appendLoc(loc, "broadcast_rhs"), rhsInput, outputShape);
        }
    }

    auto addOp = rewriter.replaceOpWithNewOp<IE::AddOp>(origOp, lhsInput, rhsInput, origOp.getAutoBroadcast(),
                                                        origOp.getPostOpAttr(), origOp.getClampAttr(),
                                                        origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
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

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createBroadcastInputForAddPass(Logger log) {
    return std::make_unique<BroadcastInputForAddPass>(log);
}
