//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE::arch37xx {
#define GEN_PASS_DECL_FUSEREORDERSPASS
#define GEN_PASS_DEF_FUSEREORDERSPASS
#include "vpux/compiler/NPU37XX/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE::arch37xx

using namespace vpux;

namespace {

//
// PermuteRewriter
//

class ReorderRewriter final : public mlir::OpRewritePattern<IE::ReorderOp> {
public:
    ReorderRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReorderOp>(ctx), _log(log) {
        this->setDebugName("ReorderRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Match following patterns:
// Pattern 1: NCE task -> Reorder -> ReturnOp
// Pattern 2: NCE task -> Reorder -> ConvertOp -> ReturnOp
// Pattern 3: NCE task -> Reorder -> SW layer (e.g., Tanh) -> ReturnOp
// Pattern 4: NCE task -> Reorder -> SW layer (e.g., Tanh) -> ConvertOp -> ReturnOp
bool isReturnOrConvertWithReturn(mlir::Operation* op) {
    auto isReturnOp = [](mlir::Operation* op) {
        return mlir::isa<mlir::func::ReturnOp>(op);
    };

    if (isReturnOp(op)) {
        return true;
    }

    auto convertOp = mlir::dyn_cast<IE::ConvertOp>(op);
    if (convertOp && llvm::all_of(convertOp->getUsers(), isReturnOp)) {
        return true;
    }

    return false;
}

bool isEligiblePatternForFuse(IE::ReorderOp reorderOp) {
    for (const auto& reorderUser : reorderOp->getUsers()) {
        if (isReturnOrConvertWithReturn(reorderUser)) {
            continue;
        }
        if (IE::isActShaveKernel(reorderUser) && llvm::all_of(reorderUser->getUsers(), isReturnOrConvertWithReturn)) {
            continue;
        }
        return false;
    }
    return true;
}

mlir::LogicalResult ReorderRewriter::matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!isEligiblePatternForFuse(origOp)) {
        return matchFailed(_log.nest(), rewriter, origOp, "ODU permutation applies only to the last reorder");
    }

    if (isTrivialReorder(origOp)) {
        return matchFailed(_log.nest(), rewriter, origOp, "ReorderOp is actually a permute cast");
    }

    auto layerWithPermute = origOp.getInput().getDefiningOp<IE::LayerWithPermuteInterface>();
    if (layerWithPermute == nullptr) {
        return matchFailed(_log.nest(), rewriter, origOp, "ReorderRewriter applies for NCE tasks");
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
    const auto newType = origType.changeDimsOrder(DimsOrder::fromAffineMap(origOp.getDstOrder()));
    layerWithPermute->getResult(0).setType(newType);

    _log.trace("Fuse {0} to {1}", origOp->getLoc(), layerWithPermute->getLoc());
    rewriter.replaceOp(origOp, layerWithPermute->getResult(0));

    return mlir::success();
}

//
// FuseReordersPass
//

class FuseReordersPass final : public IE::arch37xx::impl::FuseReordersPassBase<FuseReordersPass> {
public:
    explicit FuseReordersPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void FuseReordersPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReorderRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createFuseReordersPass(Logger log) {
    return std::make_unique<FuseReordersPass>(log);
}
