//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZEREDUCEOPSWITHMEMPERMUTE
#define GEN_PASS_DEF_OPTIMIZEREDUCEOPSWITHMEMPERMUTE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// InsertMemPermuteBeforeAndAfterReduceOp
//
// If the ReduceOp axis is not on the inner most memory dimension, the SW.Kernel is inefficient.
// Insert MemPermute before and after ReduceOp in case the axis is not on inner most memory dimension.
// For example, ReduceSum:
//
//       ReduceSumOp       -> Axis not on inner most memory dim
//
// Insert MemPermuteOp:
//
//      MemPermuteOp
//           |
//       ReduceSumOp       -> Axis on inner most memory dim
//           |
//      MemPermuteOp
//

template <class ConcreteOp>
class InsertMemPermuteBeforeAndAfterReduceOp final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    InsertMemPermuteBeforeAndAfterReduceOp(mlir::MLIRContext* ctx, Logger log,
                                           IE::ChannelAxisReductionWithDPUParentCheckerBase* checker)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefitLow), _log(log), _channelReductionChecker(checker) {
        this->setDebugName("InsertMemPermuteBeforeAndAfterReduceOp");
    }

private:
    mlir::LogicalResult matchAndRewrite(ConcreteOp reduceOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isReduceAxisOnInnerMostMemDim(ConcreteOp reduceOp, int64_t reduceAxis) const {
        const auto inputRank = mlir::cast<vpux::NDTypeInterface>(reduceOp.getInput().getType()).getRank();
        const auto inputOrder = DimsOrder::fromValue(reduceOp.getInput());
        const auto axisMemPos = inputOrder.toMemDim(Dim(reduceAxis));
        return axisMemPos.ind() == inputRank - 1;
    }

    DimsOrder calculateOptimalOrderMapForReduce(const DimsOrder& origOrder, int64_t reduceAxis,
                                                mlir::MLIRContext* ctx) const {
        auto size = origOrder.numDims();
        SmallVector<unsigned int> permVec;
        auto memDimInd = origOrder.toMemDim(Dim(reduceAxis)).ind();
        for (unsigned int i = 0; i < checked_cast<unsigned int>(size); i++) {
            if (checked_cast<unsigned int>(memDimInd) != i) {
                permVec.push_back(i);
            }
        }
        permVec.push_back(checked_cast<unsigned int>(memDimInd));
        const auto permMap = mlir::AffineMap::getPermutationMap(permVec, ctx);
        return applyPermutation(origOrder, DimsOrder::fromAffineMap(permMap));
    }

private:
    Logger _log;
    IE::ChannelAxisReductionWithDPUParentCheckerBase* _channelReductionChecker;
};

template <class ConcreteOp>
mlir::LogicalResult InsertMemPermuteBeforeAndAfterReduceOp<ConcreteOp>::matchAndRewrite(
        ConcreteOp reduceOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), reduceOp->getName(), reduceOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };
    if (config::isReduceOpSupportedOnNCE(reduceOp) && VPU::isNCEReduceSupported(reduceOp, logCb)) {
        return matchFailed(rewriter, reduceOp, "Reduce op has been supported on NCE");
    }

    const auto ctx = rewriter.getContext();
    const auto inOrder = DimsOrder::fromValue(reduceOp.getInput());
    const auto origLoc = reduceOp->getLoc();
    const auto axes = parseIntArrayAttr<int64_t>(reduceOp.getAxesValue().value());
    if (_channelReductionChecker->isChannelAxisReductionWithDPUParent(reduceOp, axes, _log)) {
        return matchFailed(rewriter, reduceOp,
                           "Reduce op has a DPU parent on channel axis; preserving fusion opportunity");
    }
    if (axes.size() != 1) {
        return matchFailed(rewriter, reduceOp, "Only support Reduce op with one dimension");
    }
    const auto reduceAxis = axes[0];

    // Check the Reduce axis is already on the inner most memory dimension
    if (isReduceAxisOnInnerMostMemDim(reduceOp, reduceAxis)) {
        return matchFailed(rewriter, reduceOp, "The Reduce axis is already on the inner most memory dim");
    }

    mlir::DenseSet<mlir::Operation*> nonMemPermuteUser;
    for (auto user : reduceOp->getUsers()) {
        if (!mlir::isa<IE::MemPermuteOp>(user)) {
            nonMemPermuteUser.insert(user);
        }
    }
    if (nonMemPermuteUser.size() > 1) {
        return matchFailed(rewriter, reduceOp, "The Reduce has more than one non mempermute user");
    }

    // Create input MemPermute
    const auto optimalDstOrder = calculateOptimalOrderMapForReduce(inOrder, reduceAxis, ctx);
    const auto permMapOfInputMemPermute = getPermutationFromOrders(inOrder, optimalDstOrder, ctx);
    const auto optimalDstOrderMap = optimalDstOrder.toAffineMap(ctx);
    auto inputMemPermuteOp = rewriter.createOrFold<IE::MemPermuteOp>(
            appendLoc(origLoc, "input_reorder"), reduceOp.getInput(), optimalDstOrderMap, permMapOfInputMemPermute);

    // Create new reduce operation
    const auto optimalAxis = optimalDstOrder.toDim(MemDim(optimalDstOrder.numDims() - 1)).ind();
    const auto newAxesAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{optimalAxis});
    auto newReduceOp = rewriter.createOrFold<ConcreteOp>(origLoc, inputMemPermuteOp, nullptr, newAxesAttr,
                                                         reduceOp.getKeepDimsAttr());
    rewriter.modifyOpInPlace(newReduceOp.getDefiningOp(), [&] {
        changeDimsOrder(newReduceOp, optimalDstOrder, _log.nest());
    });

    // Create output MemPermute
    auto permMapOfOutputMemPermute = mlir::inversePermutation(permMapOfInputMemPermute);
    auto outputMemPermuteOp = rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(
            reduceOp, newReduceOp, inOrder.toAffineMap(ctx), permMapOfOutputMemPermute);
    rewriter.modifyOpInPlace(outputMemPermuteOp, [&] {
        outputMemPermuteOp->setLoc(appendLoc(origLoc, "output_reorder"));
    });

    return mlir::success();
}

//
// OptimizeReduceOpsWithMemPermutePass
//

class OptimizeReduceOpsWithMemPermutePass final :
        public IE::impl::OptimizeReduceOpsWithMemPermuteBase<OptimizeReduceOpsWithMemPermutePass> {
public:
    OptimizeReduceOpsWithMemPermutePass(bool enableFuseReduceMinMaxToDpu, Logger log) {
        this->enableFuseReduceMinMaxToDpu = enableFuseReduceMinMaxToDpu;
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeReduceOpsWithMemPermutePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto& strategyFactory = IE::getIEStrategyFactory(&ctx);
    const auto channelReductionChecker =
            strategyFactory->getChannelAxisReductionWithDPUParentChecker(enableFuseReduceMinMaxToDpu);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<InsertMemPermuteBeforeAndAfterReduceOp<IE::ReduceSumOp>>(&ctx, _log, channelReductionChecker.get());
    patterns.add<InsertMemPermuteBeforeAndAfterReduceOp<IE::ReduceMeanOp>>(&ctx, _log, channelReductionChecker.get());
    patterns.add<InsertMemPermuteBeforeAndAfterReduceOp<IE::ReduceMinOp>>(&ctx, _log, channelReductionChecker.get());
    patterns.add<InsertMemPermuteBeforeAndAfterReduceOp<IE::ReduceMaxOp>>(&ctx, _log, channelReductionChecker.get());

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeReduceOpsWithMemPermutePass(bool enableFuseReduceMinMaxToDpu,
                                                                                Logger log) {
    return std::make_unique<OptimizeReduceOpsWithMemPermutePass>(enableFuseReduceMinMaxToDpu, log);
}
