//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/softmax_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEMEMPERMUTETHROUGHSOFTMAX
#define GEN_PASS_DEF_PROPAGATEMEMPERMUTETHROUGHSOFTMAX
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// SwapSoftmaxAndMemPermute
//

class SwapSoftmaxAndMemPermute final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    SwapSoftmaxAndMemPermute(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SoftMaxOp>(ctx, benefitHigh), _log(log) {
        this->setDebugName("SwapSoftmaxAndMemPermute");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp softMaxOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// a sub graph in below:
//
//       SoftmaxOp
//           |
//      MemPermuteOp
//           |
//          Ops
//
// Swap the softmaxOp and MemPermuteOp
//
//      MemPermuteOp
//           |
//       SoftmaxOp
//           |
//          Ops
// If the Axis of SoftmaxOp is in the last memory dimension, the SW kernel is more efficient.

mlir::LogicalResult SwapSoftmaxAndMemPermute::matchAndRewrite(IE::SoftMaxOp softMaxOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), softMaxOp->getName(), softMaxOp->getLoc());

    auto memPermuteOp = mlir::dyn_cast_or_null<IE::MemPermuteOp>(*softMaxOp.getOutput().getUsers().begin());

    if (memPermuteOp == nullptr || !softMaxOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    const auto axis = softMaxOp.getAxisInd();
    const auto inOrder = DimsOrder::fromValue(memPermuteOp.getInput());
    const auto outOrder = DimsOrder::fromValue(memPermuteOp.getOutput());

    // Softmax kernel is optimal in case when Softmax axis is the inner most axis.
    // Check whether the new Softmax Op after swap with MemPermute will be such optimal case.
    const auto outMemOrder = applyPermutation(inOrder, DimsOrder::fromAffineMap(memPermuteOp.getMemPerm()));
    const auto outMemOrderVec = to_small_vector(outMemOrder.toPermutation() | transformed([](Dim dim) {
                                                    return checked_cast<int64_t>(dim.ind());
                                                }));
    auto permuteOutputInnerMostDim = outMemOrderVec.back();
    if (axis != permuteOutputInnerMostDim) {
        return mlir::failure();
    }

    auto newMemPermuteOp = rewriter.create<IE::MemPermuteOp>(memPermuteOp->getLoc(), softMaxOp.getInput(),
                                                             memPermuteOp.getDstOrder(), memPermuteOp.getMemPerm());

    const auto optimalAxis = outOrder.toDim(MemDim(outOrder.numDims() - 1)).ind();
    const auto optimalAxisAttr = getIntAttr(rewriter.getContext(), optimalAxis);
    auto newSoftMaxOp = rewriter.replaceOpWithNewOp<IE::SoftMaxOp>(memPermuteOp, newMemPermuteOp.getOutput(),
                                                                   optimalAxisAttr, nullptr);
    changeDimsOrder(newSoftMaxOp, outOrder, _log.nest());
    rewriter.eraseOp(softMaxOp);

    return mlir::success();
}

//
// InsertMemPermuteBeforeAndAfterSoftmax
//

class InsertMemPermuteBeforeAndAfterSoftmax final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    InsertMemPermuteBeforeAndAfterSoftmax(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SoftMaxOp>(ctx, benefitLow), _log(log) {
        this->setDebugName("InsertMemPermuteBeforeAndAfterSoftmax");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp softMaxOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// If the Axis of SoftmaxOp is in the last memory dimension, the SW kernel is more efficient.
// insert MemPermuteOp before and after SoftMaxOp, if the Axis is not in last memory dim.
// origin softmax:
//
//       SoftmaxOp       -> Axis is not in last memory dim
//
// insert MemPermuteOp:
//
//      MemPermuteOp
//           |
//       SoftmaxOp       -> Axis in last memory dim
//           |
//      MemPermuteOp

mlir::LogicalResult InsertMemPermuteBeforeAndAfterSoftmax::matchAndRewrite(IE::SoftMaxOp softMaxOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), softMaxOp->getName(), softMaxOp->getLoc());

    const auto ctx = rewriter.getContext();
    const auto axis = softMaxOp.getAxisInd();
    const auto inOrder = DimsOrder::fromValue(softMaxOp.getInput());

    // Check the Axis memory position, if it's on the last dimension, the SW kernel is more efficient.
    if (vpux::IE::isSoftMaxAxisInLastMemDim(softMaxOp)) {
        return mlir::failure();
    }

    mlir::DenseSet<mlir::Operation*> nonMemPermuteUser;
    for (auto user : softMaxOp->getUsers()) {
        if (!mlir::isa<IE::MemPermuteOp>(user)) {
            nonMemPermuteUser.insert(user);
        }
    }
    // Has more than one non mempermute user, skip this to avoid the potential performance regression.
    if (nonMemPermuteUser.size() > 1) {
        return mlir::failure();
    }

    auto calcuteOptimalOrderMapForSoftmax = [](DimsOrder origOrder, int64_t softmaxAxis, auto ctx) -> DimsOrder {
        auto size = origOrder.numDims();
        SmallVector<unsigned int> permVec;

        auto memDimInd = origOrder.toMemDim(Dim(softmaxAxis)).ind();
        for (unsigned int i = 0; i < checked_cast<unsigned int>(size); i++) {
            if (checked_cast<unsigned int>(memDimInd) != i) {
                permVec.push_back(i);
            }
        }
        permVec.push_back(checked_cast<unsigned int>(memDimInd));
        const auto permMap = mlir::AffineMap::getPermutationMap(permVec, ctx);

        const auto optimalOrder = applyPermutation(origOrder, DimsOrder::fromAffineMap(permMap));

        return optimalOrder;
    };

    // Create input MemPermute to transpose softmax axis to the inner most dimension
    const auto optimalDstOrder = calcuteOptimalOrderMapForSoftmax(inOrder, axis, ctx);
    const auto permMapOfInputMemPermute = getPermutationFromOrders(inOrder, optimalDstOrder, ctx);
    const auto optimalDstOrderMap = optimalDstOrder.toAffineMap(ctx);
    auto inputMemPermuteOp = rewriter.create<IE::MemPermuteOp>(softMaxOp->getLoc(), softMaxOp.getInput(),
                                                               optimalDstOrderMap, permMapOfInputMemPermute);

    // Create new Softmax with optimal axis
    const auto optimalAxis = optimalDstOrder.toDim(MemDim(optimalDstOrder.numDims() - 1)).ind();
    const auto optimalAxisAttr = getIntAttr(ctx, optimalAxis);
    auto newSoftMaxOp = rewriter.create<IE::SoftMaxOp>(softMaxOp->getLoc(), inputMemPermuteOp.getOutput(),
                                                       optimalAxisAttr, nullptr);
    changeDimsOrder(newSoftMaxOp, optimalDstOrder, _log.nest());

    // Create output MemPermute for inverse data transposition
    auto permMapOfOutputMemPermute = mlir::inversePermutation(permMapOfInputMemPermute);
    rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(softMaxOp, newSoftMaxOp.getOutput(), inOrder.toAffineMap(ctx),
                                                  permMapOfOutputMemPermute);

    return mlir::success();
}

//
// PropagateMemPermuteThroughSoftMaxPass
//

class PropagateMemPermuteThroughSoftMaxPass final :
        public IE::impl::PropagateMemPermuteThroughSoftMaxBase<PropagateMemPermuteThroughSoftMaxPass> {
public:
    explicit PropagateMemPermuteThroughSoftMaxPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PropagateMemPermuteThroughSoftMaxPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapSoftmaxAndMemPermute>(&ctx, _log);
    patterns.add<InsertMemPermuteBeforeAndAfterSoftmax>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateMemPermuteThroughSoftMaxPass(Logger log) {
    return std::make_unique<PropagateMemPermuteThroughSoftMaxPass>(log);
}
