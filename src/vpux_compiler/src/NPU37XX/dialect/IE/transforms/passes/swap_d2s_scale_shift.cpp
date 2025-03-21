//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPD2SANDSCALESHIFT
#define GEN_PASS_DEF_SWAPD2SANDSCALESHIFT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// SwapDepth2SpaceAndScaleShift
//

class SwapDepth2SpaceAndScaleShift final : public mlir::OpRewritePattern<IE::DepthToSpaceOp> {
public:
    SwapDepth2SpaceAndScaleShift(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DepthToSpaceOp>(ctx), _log(log) {
        setDebugName("SwapDepth2SpaceAndScaleShift");
    }

    mlir::LogicalResult matchAndRewrite(IE::DepthToSpaceOp, mlir::PatternRewriter&) const final;

private:
    Logger _log;
};

//
// matchAndRewrite
//

mlir::LogicalResult SwapDepth2SpaceAndScaleShift::matchAndRewrite(IE::DepthToSpaceOp d2sOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("found '{0}' at '{1}'", d2sOp->getName(), d2sOp->getLoc());

    auto ctx = rewriter.getContext();
    auto loc = d2sOp.getLoc();

    if (!d2sOp->hasOneUse()) {
        return mlir::failure();
    }
    auto scaleShift = mlir::dyn_cast<IE::ScaleShiftOp>(*d2sOp.getOutput().getUsers().begin());
    if (scaleShift == nullptr) {
        return mlir::failure();
    }

    if (!VPU::isNullOrConstWithSingleValue(scaleShift.getWeights()) ||
        !VPU::isNullOrConstWithSingleValue(scaleShift.getBiases())) {
        _log.trace("Weights/Biases is not splat constant");
        return mlir::failure();
    }

    auto d2sInShape = d2sOp.getInput().getType().cast<vpux::NDTypeInterface>().getShape();

    auto getNewValue = [&](mlir::Value origValue) {
        if (origValue == nullptr) {
            return origValue;
        }

        auto broadcastShape = Shape({1, d2sInShape[Dims4D::Act::C], 1, 1});
        auto reshapeShape = Shape({1, d2sInShape[Dims4D::Act::C], 1, 1});

        auto broadcastOp = rewriter.createOrFold<IE::BroadcastOp>(
                loc, origValue, vpux::IE::createShapeConstForBroadCast(rewriter, ctx, loc, broadcastShape), nullptr,
                IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY));

        return rewriter.createOrFold<IE::ReshapeOp>(loc, broadcastOp, nullptr, false,
                                                    getIntArrayAttr(ctx, ShapeRef(reshapeShape)));
    };
    auto newScaleShift = rewriter.create<IE::ScaleShiftOp>(loc, d2sOp.getInput(), getNewValue(scaleShift.getWeights()),
                                                           getNewValue(scaleShift.getBiases()));
    rewriter.replaceOpWithNewOp<IE::DepthToSpaceOp>(scaleShift, d2sOp.getType(), newScaleShift.getOutput(),
                                                    d2sOp.getBlockSizeAttr(), d2sOp.getModeAttr());

    return mlir::success();
}

//
// SwapD2SAndScaleShiftPass
//

class SwapD2SAndScaleShiftPass final : public IE::impl::SwapD2SAndScaleShiftBase<SwapD2SAndScaleShiftPass> {
public:
    explicit SwapD2SAndScaleShiftPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void SwapD2SAndScaleShiftPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    // auto module = func->getParentOfType<mlir::ModuleOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SwapDepth2SpaceAndScaleShift>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSwapD2SAndScaleShiftPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSwapD2SAndScaleShiftPass(Logger log) {
    return std::make_unique<SwapD2SAndScaleShiftPass>(log);
}
