//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTTOMEMPERMUTE
#define GEN_PASS_DEF_CONVERTTOMEMPERMUTE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertToMemPermutePass
//

class ConvertToMemPermutePass final : public IE::impl::ConvertToMemPermuteBase<ConvertToMemPermutePass> {
public:
    explicit ConvertToMemPermutePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ReorderOpConverter;
    class TransposeOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ReorderOpConverter
//

class ConvertToMemPermutePass::ReorderOpConverter final : public mlir::OpRewritePattern<IE::ReorderOp> {
public:
    ReorderOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReorderOp>(ctx), _log(log) {
        setDebugName("ConvertToMemPermutePass::ReorderOpConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToMemPermutePass::ReorderOpConverter::matchAndRewrite(
        IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::Reorder Operation '{0}'", origOp->getLoc());

    auto inOrder = DimsOrder::fromValue(origOp.getInput());
    auto outOrder = DimsOrder::fromValue(origOp.getOutput());

    auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(inOrder, outOrder, origOp->getContext()));

    rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(origOp, origOp.getInput(), origOp.getDstOrderAttr(), memPermAttr);

    _log.trace("Replaced with 'IE::MemPermute'");

    return mlir::success();
}

//
// TransposeOpConverter
//

class ConvertToMemPermutePass::TransposeOpConverter final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    TransposeOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TransposeOp>(ctx), _log(log) {
        setDebugName("ConvertToMemPermutePass::TransposeOpConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToMemPermutePass::TransposeOpConverter::matchAndRewrite(
        IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::Transpose Operation '{0}'", origOp->getLoc());
    VPUX_THROW_UNLESS(origOp.getOrderValue().has_value(), "IE::Transpose Operation doesn't have order_value attribute");

    auto outputOrder = DimsOrder::fromValue(origOp.getOutput());
    auto dstOrder = mlir::AffineMapAttr::get(outputOrder.toAffineMap(origOp.getContext()));
    auto inputOrder = DimsOrder::fromValue(origOp.getInput());
    auto inPerm = inputOrder.toAffineMap(origOp.getContext());
    auto memPerm = inPerm.compose(origOp.getOrderValue().value());
    auto memPermAttr = mlir::AffineMapAttr::get(memPerm);

    rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(origOp, origOp.getInput(), dstOrder, memPermAttr);

    _log.trace("Replaced with 'IE::MemPermute'");

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertToMemPermutePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::TransposeOp>();
    target.addIllegalOp<IE::ReorderOp>();
    target.addLegalOp<IE::MemPermuteOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<TransposeOpConverter>(&ctx, _log);
    patterns.add<ReorderOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToMemPermutePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToMemPermutePass(Logger log) {
    return std::make_unique<ConvertToMemPermutePass>(log);
}
