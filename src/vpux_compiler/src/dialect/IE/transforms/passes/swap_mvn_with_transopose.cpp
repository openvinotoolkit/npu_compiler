//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/transpose_op_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPMVNWITHTRANSPOSE
#define GEN_PASS_DEF_SWAPMVNWITHTRANSPOSE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// SwapMVNWithTranspose
//

class SwapMVNWithTranspose final : public IE::impl::SwapMVNWithTransposeBase<SwapMVNWithTranspose> {
public:
    explicit SwapMVNWithTranspose(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class OpSwapConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// OpSwapConverter
//

class SwapMVNWithTranspose::OpSwapConverter final : public mlir::OpRewritePattern<IE::MVNOp> {
public:
    OpSwapConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MVNOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/* This rewriter swaps MVNOp with parent TransposeOp in case:
    1.MVNOp is not cross_channel.
    2.Parent TransposeOp is swapping W and H axis.
The benifits are:
    1.Prevent inserting Reorder between TransposeOp and MVNOp.
    2.Original input and output TransposeOps have opportinuty to be elimented.
*/
mlir::LogicalResult SwapMVNWithTranspose::OpSwapConverter::matchAndRewrite(IE::MVNOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    const auto mvnIn = origOp.getInput();
    auto origTransposeOp = mvnIn.getDefiningOp<IE::TransposeOp>();
    auto mvnOp =
            rewriter.create<IE::MVNOp>(origOp->getLoc(), origTransposeOp.getInput(), origOp.getAcrossChannelsAttr(),
                                       origOp.getNormalizeVarianceAttr(), origOp.getEpsAttr());

    rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, mvnOp.getOutput(), nullptr,
                                                 origTransposeOp.getOrderValueAttr());
    rewriter.eraseOp(origTransposeOp);

    return mlir::success();
}

void SwapMVNWithTranspose::safeRunOnFunc() {
    auto func = getOperation();

    auto& ctx = getContext();

    const auto isLegalOp = [](IE::MVNOp op) -> bool {
        if (!op->hasOneUse()) {
            return true;
        }

        if (op.getAcrossChannels()) {
            return true;
        }

        auto prevTranspoe = op.getInput().getDefiningOp<IE::TransposeOp>();
        auto nextTranspoe = mlir::dyn_cast<IE::TransposeOp>(*op.getOutput().getUsers().begin());
        if (!prevTranspoe || !nextTranspoe || !prevTranspoe.getOutput().hasOneUse() ||
            !nextTranspoe.getOutput().hasOneUse()) {
            return true;
        }

        if (vpux::IE::isWHSwappingTranspose(prevTranspoe) && vpux::IE::isWHSwappingTranspose(nextTranspoe)) {
            return prevTranspoe.getInput().isa<mlir::BlockArgument>();
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::MVNOp>(isLegalOp);
    target.addLegalOp<IE::TransposeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapMVNWithTranspose::OpSwapConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapMVNWithTransposePass(Logger log) {
    return std::make_unique<SwapMVNWithTranspose>(log);
}
