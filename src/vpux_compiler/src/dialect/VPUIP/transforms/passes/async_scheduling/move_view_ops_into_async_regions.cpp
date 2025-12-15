//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SmallPtrSet.h>

#include <algorithm>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_MOVEVIEWOPSINTOASYNCREGIONS
#define GEN_PASS_DEF_MOVEVIEWOPSINTOASYNCREGIONS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// AsyncRegionRewriter
//

SmallVector<mlir::Operation*> getOuterViewLikeDeps(mlir::Block* block) {
    llvm::SmallPtrSet<mlir::Operation*, 16> outSet;

    for (auto& op : block->getOperations()) {
        for (auto arg : op.getOperands()) {
            auto* producer = arg.getDefiningOp();

            if (producer != nullptr && producer->getBlock() != block && VPUIP::isPureViewOp(producer)) {
                outSet.insert(producer);
            }
        }
    }

    auto outVec = to_small_vector(outSet);

    std::sort(outVec.begin(), outVec.end(), [](mlir::Operation* op1, mlir::Operation* op2) {
        return op1->isBeforeInBlock(op2);
    });

    return outVec;
}

class AsyncRegionRewriter final : public mlir::OpRewritePattern<mlir::async::ExecuteOp> {
public:
    AsyncRegionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::async::ExecuteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::ExecuteOp execOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AsyncRegionRewriter::matchAndRewrite(mlir::async::ExecuteOp execOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got 'async.execute' Operation at '{0}'", execOp->getLoc());

    auto* bodyBlock = execOp.getBody();

    const auto outerDeps = getOuterViewLikeDeps(bodyBlock);
    if (outerDeps.empty()) {
        return matchFailed(rewriter, execOp, "The 'async.execute' inner region has no outer view-like dependencies");
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(bodyBlock);

    for (auto* origOp : outerDeps) {
        auto* newOp = rewriter.clone(*origOp);
        rewriter.replaceOpUsesWithinBlock(origOp, newOp->getResults(), bodyBlock);
    }

    return mlir::success();
}

//
// MoveViewOpsIntoAsyncRegionsPass
//

class MoveViewOpsIntoAsyncRegionsPass final :
        public VPUIP::impl::MoveViewOpsIntoAsyncRegionsBase<MoveViewOpsIntoAsyncRegionsPass> {
public:
    explicit MoveViewOpsIntoAsyncRegionsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void MoveViewOpsIntoAsyncRegionsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AsyncRegionRewriter>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsGreedily(getOperation(), std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveViewOpsIntoAsyncRegionsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createMoveViewOpsIntoAsyncRegionsPass(Logger log) {
    return std::make_unique<MoveViewOpsIntoAsyncRegionsPass>(log);
}
