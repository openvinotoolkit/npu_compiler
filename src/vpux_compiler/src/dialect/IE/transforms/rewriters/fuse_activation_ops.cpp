//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEACTIVATIONOPS
#define GEN_PASS_DEF_FUSEACTIVATIONOPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// GenericConverter
//

class FusePostOpsRewriter final : public mlir::OpTraitRewritePattern<IE::EltwiseOp> {
public:
    FusePostOpsRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpTraitRewritePattern<IE::EltwiseOp>(ctx, benefit), _log(log) {
        this->setDebugName("FusePostOps::FusePostOpsRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* postOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FusePostOpsRewriter::matchAndRewrite(mlir::Operation* postOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Eltwise operation '{1}' at '{2}'", getDebugName(), postOp->getName(), postOp->getLoc());
    auto inElemType = postOp->getOperand(0).getType();

    if (inElemType.isF32()) {
        return matchFailed(_log, rewriter, postOp, "PostOp is not supported for FP32");
    }
    if (!postOp->getOperand(0).hasOneUse()) {
        return matchFailed(_log, rewriter, postOp, "PostOp is not the only user of its input Value");
    }

    auto producerOp = postOp->getOperand(0).getDefiningOp<IE::LayerWithPostOpInterface>();
    if (producerOp == nullptr) {
        return matchFailed(
                _log, rewriter, postOp,
                "PostOp input was not produced by another Operation or the producer does not support post-processing");
    }
    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };
    if (!producerOp.isSupportedPostOp(postOp, logCb)) {
        return matchFailed(_log, rewriter, postOp, "PostOp producer does not support post-processing for current case");
    }
    if (const auto postOpAttr = producerOp.getPostOp()) {
        return matchFailed(_log, rewriter, postOp, "PostOp producer already has post-processing '{0}'",
                           postOpAttr.getName());
    }
    if (postOp->getNumOperands() != 1) {
        return matchFailed(_log, rewriter, postOp,
                           "Only single input operation can be attached as PostOp via attributes. Got '{0}' inputs",
                           postOp->getNumOperands());
    }

    producerOp.setPostOp(postOp);
    auto origElemType = mlir::cast<vpux::NDTypeInterface>(postOp->getResult(0).getType()).getElementType();
    auto newType = mlir::cast<vpux::NDTypeInterface>(producerOp->getOpResult(0).getType());
    producerOp->getOpResult(0).setType(newType.changeElemType(origElemType));
    rewriter.replaceOp(postOp, producerOp->getResult(0));

    return mlir::success();
}

//
// FuseClampRewriter
//

class FuseClampRewriter final : public mlir::OpRewritePattern<IE::ClampOp> {
public:
    FuseClampRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::ClampOp>(ctx, benefit), _log(log) {
        this->setDebugName("FuseClamp::FuseClampRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ClampOp clampOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseClampRewriter::matchAndRewrite(IE::ClampOp clampOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Clamp operation '{1}' at '{2}'", getDebugName(), clampOp->getName(), clampOp->getLoc());

    if (!clampOp.getInput().hasOneUse()) {
        return matchFailed(_log, rewriter, clampOp, "Clamp is not the only user of its input Value");
    }

    auto producerOp = clampOp.getInput().getDefiningOp<IE::LayerWithPostOpInterface>();
    if (producerOp == nullptr) {
        return matchFailed(
                _log, rewriter, clampOp,
                "Clamp input was not produced by another Operation or the producer does not support post-processing");
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };
    if (!producerOp.isSupportedClampOp(clampOp, logCb)) {
        return matchFailed(_log, rewriter, clampOp,
                           "ClampOp producer does not support post-processing for current case");
    }

    producerOp.setClampOp(clampOp);
    rewriter.replaceOp(clampOp, producerOp->getResult(0));

    return mlir::success();
}

//
// SwapSliceWithActivation
//
// Moves a unary eltwise activation from after a Slice to before it, enabling
// subsequent fusion as a post-op on the preceding DPU operation.
//   DPUOp -> Slice -> Activation  =>  DPUOp -> Activation -> Slice
//

class SwapSliceWithActivation final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    SwapSliceWithActivation(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::SliceOp>(ctx, benefit), _log(log) {
        this->setDebugName("FusePostOps::SwapSliceWithActivation");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapSliceWithActivation::matchAndRewrite(IE::SliceOp sliceOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got SliceOp at '{1}'", getDebugName(), sliceOp->getLoc());

    if (!sliceOp.getResult().hasOneUse()) {
        return mlir::failure();
    }

    auto* userOp = *sliceOp.getResult().getUsers().begin();

    if (!userOp->hasTrait<IE::EltwiseOp>() || userOp->getNumOperands() != 1) {
        return mlir::failure();
    }

    if (!sliceOp.getSource().hasOneUse()) {
        return mlir::failure();
    }

    auto producerOp = sliceOp.getSource().getDefiningOp<IE::LayerWithPostOpInterface>();
    if (producerOp == nullptr) {
        return mlir::failure();
    }

    if (producerOp.getPostOp()) {
        return mlir::failure();
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };
    if (!producerOp.isSupportedPostOp(userOp, logCb)) {
        return mlir::failure();
    }

    // Swap: create activation on the unsliced tensor, then slice the result
    rewriter.setInsertionPoint(sliceOp);
    auto* newActivation = rewriter.clone(*userOp);
    auto sourceType = mlir::cast<vpux::NDTypeInterface>(sliceOp.getSource().getType());
    auto activationElemType = mlir::cast<vpux::NDTypeInterface>(userOp->getResult(0).getType()).getElementType();
    rewriter.modifyOpInPlace(newActivation, [&] {
        newActivation->setOperand(0, sliceOp.getSource());
        newActivation->getResult(0).setType(sourceType.changeElemType(activationElemType));
    });

    auto newSlice = rewriter.create<IE::SliceOp>(sliceOp.getLoc(), newActivation->getResult(0),
                                                 sliceOp.getStaticOffsetsAttr(), sliceOp.getStaticSizesAttr());
    extendOpLoc(newSlice, "swap_act");

    rewriter.replaceOp(userOp, newSlice.getResult());
    rewriter.eraseOp(sliceOp);

    return mlir::success();
}

//
// FuseActivationOpsPass
//

class FuseActivationOpsPass final : public IE::impl::FuseActivationOpsBase<FuseActivationOpsPass> {
public:
    explicit FuseActivationOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
};

mlir::LogicalResult FuseActivationOpsPass::initialize(mlir::MLIRContext* ctx) {
    return Base::initialize(ctx);
}

void FuseActivationOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    // Note the below patterns exec order is defined by "benefitLevels" at the head
    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SwapSliceWithActivation>(&ctx, vpux::benefitHigh, _log);
    patterns.insert<FusePostOpsRewriter>(&ctx, vpux::benefitLow, _log);
    patterns.insert<FuseClampRewriter>(&ctx, vpux::benefitMid, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

void vpux::IE::registerFuseActivationOpsRewriters(RewriterRegistry& registry, Logger log) {
    registry.registerRewriterSet("fuse-activation-ops-set", [&registry, log]() {
        registry.registerRewriter<SwapSliceWithActivation>("swap-slice-with-activation", vpux::benefitHigh, log);
        registry.registerRewriter<FusePostOpsRewriter>("fuse-post-ops", vpux::benefitLow, log);
        registry.registerRewriter<FuseClampRewriter>("fuse-clamp", vpux::benefitMid, log);
    });
}

// E-187110: Remove unnecessary pass declaration and definition
std::unique_ptr<mlir::Pass> vpux::IE::createFuseActivationOpsPass(Logger log) {
    return std::make_unique<FuseActivationOpsPass>(log);
}
