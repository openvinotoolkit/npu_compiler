//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEMEMPERMUTEPASS
#define GEN_PASS_DEF_FUSEMEMPERMUTEPASS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// MemPermuteRewriter
//

class MemPermuteRewriter final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    MemPermuteRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
        this->setDebugName("MemPermuteRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MemPermuteRewriter::matchAndRewrite(IE::MemPermuteOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto inShape = getShape(origOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    if (isTrivialPermute(inMemShape, origOp.getMemPerm())) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp is actually a permute cast");
    }

    auto layerWithPermute = getFusableLayerWithPermuteInterface(origOp.getOperation());
    if (layerWithPermute == nullptr) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteRewriter applies for NCE tasks");
    }

    if (!layerWithPermute.isSupportedPermutation(origOp)) {
        return matchFailed(_log.nest(), rewriter, origOp, "ODU permutation does not support {0} at {1}",
                           origOp->getName(), origOp->getLoc());
    }

    if (!layerWithPermute->getResult(0).hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origOp,
                           "ReorderRewriter applies only for NCE tasks with one consumer");
    }

    auto output = layerWithPermute->getResult(0);
    const auto origType = output.getType().cast<vpux::NDTypeInterface>();
    if (origType == nullptr) {
        return matchFailed(_log.nest(), rewriter, origOp, "NCE task does not implement vpux::NDTypeInterface");
    }

    _log.trace("Fuse {0} to {1}", origOp->getLoc(), layerWithPermute->getLoc());

    auto maybeQuantizeCastOp = mlir::dyn_cast_or_null<IE::QuantizeCastOp>(*(layerWithPermute->getUsers().begin()));

    const auto targetOrder = applyPermutation(inOrder, DimsOrder::fromAffineMap(origOp.getMemPerm()));
    const auto adjustedOrder = moveD0ToTheFront(targetOrder);
    const auto newType = origType.changeDimsOrder(adjustedOrder);
    layerWithPermute->getResult(0).setType(newType);

    auto ctx = rewriter.getContext();
    const auto dstOrderMap = origOp.getDstOrder();
    const auto trivialMemPerm = getPermutationFromOrders(adjustedOrder, targetOrder, ctx);
    auto newOutput = rewriter.createOrFold<IE::PermuteCastOp>(origOp.getLoc(), layerWithPermute->getResult(0),
                                                              dstOrderMap, trivialMemPerm);

    if (maybeQuantizeCastOp != nullptr) {
        newOutput = rewriter.createOrFold<IE::QuantizeCastOp>(
                maybeQuantizeCastOp->getLoc(), origOp.getType(), newOutput,
                maybeQuantizeCastOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType());
    }

    rewriter.replaceOp(origOp, newOutput);

    return mlir::success();
}

//
// FuseMemPermutePass
//

class FuseMemPermutePass final : public IE::impl::FuseMemPermutePassBase<FuseMemPermutePass> {
public:
    explicit FuseMemPermutePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void FuseMemPermutePass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MemPermuteRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFuseMemPermutePass(Logger log) {
    return std::make_unique<FuseMemPermutePass>(log);
}
