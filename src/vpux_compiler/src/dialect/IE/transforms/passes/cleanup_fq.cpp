//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CLEANUPFQ
#define GEN_PASS_DEF_CLEANUPFQ
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool isIdentityFQ(IE::FakeQuantizeOp fqOp) {
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == outLowConst && inHighConst == outHighConst) {
        return true;
    }

    const auto areValuesEqual = [](Const::DeclareOp inCstOp, Const::DeclareOp outCstOp) {
        auto inData = IE::getConst(inCstOp);
        auto outData = IE::getConst(outCstOp);

        if (inData.size() != outData.size()) {
            return false;
        }

        auto allOfValuesAreEqual = llvm::all_of(llvm::zip(inData, outData), [](auto pair) {
            return isFloatEqual(std::get<0>(pair), std::get<1>(pair));
        });

        return allOfValuesAreEqual;
    };

    auto isLowEqual = areValuesEqual(inLowConst, outLowConst);
    auto isHighEqual = areValuesEqual(inHighConst, outHighConst);

    return isLowEqual && isHighEqual;
}

bool isViewLikeOrFQ(mlir::Operation* op) {
    // Keep FQs at I/O to
    // - simplify tests
    // - preserve the original behavior of the nGraph pass

    if (op == nullptr) {
        // BlockArgument
        return false;
    }

    if (mlir::isa<mlir::func::ReturnOp>(op)) {
        return false;
    }

    // Using IE::isPureViewOp(op) results in some performance regressions
    return mlir::isa<IE::VariadicSplitOp, IE::StridedSliceOp, IE::SplitOp, IE::ReorgYoloOp, IE::TransposeOp,
                     IE::SqueezeOp, IE::ReshapeOp, IE::ConcatOp, IE::TileOp, IE::UnsqueezeOp, IE::ScatterNDUpdateOp>(
                   op) ||
           mlir::isa<IE::FakeQuantizeOp>(op);
}

//
// CleanupFQRewriter
//

class CleanupFQRewriter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    CleanupFQRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        this->setDebugName("CleanupFQ");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CleanupFQRewriter::matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const {
    if (!isViewLikeOrFQ(fqOp.getInput().getDefiningOp())) {
        return mlir::failure();
    }

    for (auto user : fqOp.getOutput().getUsers()) {
        if (!isViewLikeOrFQ(user)) {
            return mlir::failure();
        }
    }

    if (!isIdentityFQ(fqOp)) {
        return mlir::failure();
    }

    rewriter.replaceOp(fqOp, fqOp.getInput());

    return mlir::success();
}

//
// CleanupFQ
//

class CleanupFQ final : public IE::impl::CleanupFQBase<CleanupFQ> {
public:
    explicit CleanupFQ(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void CleanupFQ::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<CleanupFQRewriter>(&ctx, _log);

    auto config = getDefaultGreedyRewriteConfig();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createCleanupFQPass(Logger log) {
    return std::make_unique<CleanupFQ>(log);
}
