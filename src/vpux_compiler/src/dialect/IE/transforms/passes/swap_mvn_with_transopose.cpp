//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/transpose_op_utils.hpp"

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPMVNWITHTRANSPOSE
#define GEN_PASS_DEF_SWAPMVNWITHTRANSPOSE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool shouldConvertMVNOp(IE::MVNOp op) {
    if (!op->hasOneUse()) {
        return false;
    }

    if (op.getAcrossChannels()) {
        return false;
    }

    auto prevTranspoe = op.getInput().getDefiningOp<IE::TransposeOp>();
    auto nextTranspoe = mlir::dyn_cast<IE::TransposeOp>(*op.getOutput().getUsers().begin());
    if (!prevTranspoe || !nextTranspoe || !prevTranspoe.getOutput().hasOneUse() ||
        !nextTranspoe.getOutput().hasOneUse()) {
        return false;
    }

    if (!vpux::IE::isWHSwappingTranspose(prevTranspoe) || !vpux::IE::isWHSwappingTranspose(nextTranspoe)) {
        return false;
    }

    return !mlir::isa<mlir::BlockArgument>(prevTranspoe.getInput());
}

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
    if (!shouldConvertMVNOp(origOp)) {
        return mlir::failure();
    }

    const auto mvnIn = origOp.getInput();
    auto origTransposeOp = mvnIn.getDefiningOp<IE::TransposeOp>();
    auto mvnOp =
            IE::MVNOp::create(rewriter, origOp->getLoc(), origTransposeOp.getInput(), origOp.getAcrossChannelsAttr(),
                              origOp.getNormalizeVarianceAttr(), origOp.getEpsAttr());

    rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, mvnOp.getOutput(), nullptr,
                                                 origTransposeOp.getOrderValueAttr());
    rewriter.eraseOp(origTransposeOp);

    return mlir::success();
}

void SwapMVNWithTranspose::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapMVNWithTranspose::OpSwapConverter>(&ctx, _log);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapMVNWithTransposePass(Logger log) {
    return std::make_unique<SwapMVNWithTranspose>(log);
}
