//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// RemoveDuplicatingReorders
//

class RemoveDuplicatingReorders final : public mlir::OpRewritePattern<IE::ReorderOp> {
public:
    RemoveDuplicatingReorders(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ReorderOp>(ctx), _log(log) {
        setDebugName("RemoveDuplicatingReorders");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RemoveDuplicatingReorders::matchAndRewrite(IE::ReorderOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    for (auto user : origOp.input().getUsers()) {
        if (user == origOp) {
            continue;
        }
        if (auto reorder = mlir::dyn_cast<IE::ReorderOp>(user)) {
            if (origOp.getType() == reorder.getType()) {
                _log.trace("Current node has a duplicate. Eliminate usage of current node:\n{0}\n{1} {2}", origOp,
                           reorder, origOp.getLoc());

                rewriter.replaceOp(reorder, origOp->getResults());

                return mlir::success();
            }
        }
    }

    return mlir::failure();
}

//
// UniquifyOpsPass
//

class UniquifyOpsPass final : public IE::UniquifyOpsBase<UniquifyOpsPass> {
public:
    explicit UniquifyOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UniquifyOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RemoveDuplicatingReorders>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUniquifyOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUniquifyOpsPass(Logger log) {
    return std::make_unique<UniquifyOpsPass>(log);
}
