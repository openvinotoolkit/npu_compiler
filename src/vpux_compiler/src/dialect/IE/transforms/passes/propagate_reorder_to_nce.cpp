//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/act_shave_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/softmax_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEREORDERTONCE
#define GEN_PASS_DEF_PROPAGATEREORDERTONCE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class ActShaveRewriter final : public mlir::OpRewritePattern<IE::ReorderOp> {
public:
    ActShaveRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReorderOp>(ctx), _log(log) {
        this->setDebugName("ActShaveRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Match following patterns:
// Pattern 1: NCE task -> SW layer (e.g., Tanh) -> Reorder -> ReturnOp
// Pattern 2: NCE task -> SW layer (e.g., Tanh) -> Reorder -> ConvertOp -> ReturnOp
bool isLastReorderInGraph(IE::ReorderOp reorderOp) {
    auto isReturnOp = [](mlir::Operation* op) {
        return mlir::isa<mlir::func::ReturnOp>(op);
    };

    for (const auto& user : reorderOp->getUsers()) {
        if (isReturnOp(user)) {
            continue;
        }

        auto convertOp = mlir::dyn_cast<IE::ConvertOp>(user);
        if (convertOp && llvm::all_of(convertOp->getUsers(), isReturnOp)) {
            continue;
        }

        return false;
    }
    return true;
}

mlir::LogicalResult ActShaveRewriter::matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!isLastReorderInGraph(origOp)) {
        return matchFailed(_log.nest(), rewriter, origOp, "Reorder is not the last operation in the graph.");
    }

    // Check that the producer of this IE.Reorder is a software layer.
    auto producer = origOp.getInput().getDefiningOp();
    if (producer == nullptr) {
        return matchFailed(_log.nest(), rewriter, origOp, "Reorder producer is a block argument.");
    }
    if (!IE::isActShaveKernel(producer)) {
        return matchFailed(_log.nest(), rewriter, origOp, "Reorder producer is not a software layer.");
    }

    if (producer->getNumOperands() > 1) {
        return matchFailed(_log.nest(), rewriter, origOp, "Reorder producer has multiple inputs.");
    }

    if (!producer->hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origOp, "Reorder producer has multiple users.");
    }

    // If the software layer is SoftMaxOp, do not propagate while the axis memory position has been in the last
    // dimension which is most efficient
    if (auto softMaxOp = mlir::dyn_cast<IE::SoftMaxOp>(producer)) {
        if (vpux::IE::isSoftMaxAxisInLastMemDim(softMaxOp)) {
            return matchFailed(_log.nest(), rewriter, origOp,
                               "The axis of SoftMaxOp has been in the last dimension which is most efficient.");
        }
    }

    if (auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(producer)) {
        const auto propagatingOrder = DimsOrder::fromValue(origOp.getOutput());

        auto orderInfo = iface.getLayoutInfo();
        const auto producerInputRank = orderInfo.getInput(0).numDims();
        const auto propagatingRank = propagatingOrder.numDims();
        if (producerInputRank != propagatingRank) {
            return matchFailed(_log.nest(), rewriter, producer,
                               "Producer input rank {0} does not match propagating rank {1}", producerInputRank,
                               propagatingOrder);
        }

        orderInfo.setInput(0, propagatingOrder);
        iface.inferLayoutInfo(orderInfo, /*seOpsEnabled=*/false, /*seExperimentalOpsEnabled=*/false);
        if (orderInfo.getInput(0) != propagatingOrder || orderInfo.getOutput(0) != propagatingOrder) {
            return matchFailed(_log.nest(), rewriter, producer,
                               "Act shave kernel doesn't support propagating order {0}", propagatingOrder);
        }
    }

    // Check that there is NCE task above
    auto maybeNCE = producer->getOperand(0).getDefiningOp();

    if (maybeNCE == nullptr || VPU::NCEInvariant::verifyKernel(maybeNCE, _log).failed()) {
        return matchFailed(_log.nest(), rewriter, origOp, "Act shave producer is not a NCE layer.");
    }

    if (!maybeNCE->hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origOp, "NCE operation has more than one user");
    }

    const auto dstOrder = DimsOrder::fromValue(origOp->getResult(0));
    const auto dstOrderMap = dstOrder.toAffineMap(rewriter.getContext());
    auto reorder = rewriter.create<IE::ReorderOp>(origOp->getLoc(), producer->getOperand(0), dstOrderMap);
    mlir::IRMapping mapper;
    mapper.map(producer->getOperand(0), reorder->getResult(0));
    auto newProducer = rewriter.clone(*producer, mapper);
    vpux::inferReturnTypes(newProducer, vpux::InferShapedTypeMode::ALL);

    rewriter.replaceOp(producer, newProducer->getResult(0));

    return mlir::success();
}

//
// PropagateReorderToNCE
//

class PropagateReorderToNCE final : public IE::impl::PropagateReorderToNCEBase<PropagateReorderToNCE> {
public:
    explicit PropagateReorderToNCE(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PropagateReorderToNCE::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ActShaveRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateReorderToNCEPass(Logger log) {
    return std::make_unique<PropagateReorderToNCE>(log);
}
