//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/transpose_op_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_TRANSPOSETOPERMUTECAST
#define GEN_PASS_DEF_TRANSPOSETOPERMUTECAST
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// TransposeToPermuteCast
//

class TransposeToPermuteCast final : public IE::impl::TransposeToPermuteCastBase<TransposeToPermuteCast> {
public:
    explicit TransposeToPermuteCast(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class TransposeOpConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// TransposeOpConverter
//

class TransposeToPermuteCast::TransposeOpConverter final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    TransposeOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TransposeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TransposeToPermuteCast::TransposeOpConverter::matchAndRewrite(
        IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto transposeIn = origOp.getInput();
    const auto targetDimsOrder = IE::deduceInverseOrder(origOp);
    const auto dstOrder = mlir::AffineMapAttr::get(targetDimsOrder.toAffineMap(rewriter.getContext()));
    const auto origOutOrder = DimsOrder::fromValue(origOp.getOutput());
    const auto numDims = checked_cast<unsigned>(origOutOrder.numDims());
    const auto memPerm =
            mlir::AffineMapAttr::get(mlir::AffineMap::getMinorIdentityMap(numDims, numDims, rewriter.getContext()));
    auto permuteCast = rewriter.create<IE::PermuteCastOp>(origOp->getLoc(), transposeIn, dstOrder, memPerm);
    rewriter.replaceOpWithNewOp<IE::ReorderOp>(
            origOp, permuteCast.getOutput(), mlir::AffineMapAttr::get(origOutOrder.toAffineMap(rewriter.getContext())));

    return mlir::success();
}

void TransposeToPermuteCast::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::ReorderOp>();
    target.addLegalOp<IE::PermuteCastOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<TransposeToPermuteCast::TransposeOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createTransposeToPermuteCastPass(Logger log) {
    return std::make_unique<TransposeToPermuteCast>(log);
}
