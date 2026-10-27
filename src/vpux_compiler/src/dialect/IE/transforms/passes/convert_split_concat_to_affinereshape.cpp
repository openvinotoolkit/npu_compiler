//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/split_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSPLITCONCATTOAFFINERESHAPE
#define GEN_PASS_DEF_CONVERTSPLITCONCATTOAFFINERESHAPE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// SplitConcatRewriter
//

//
//               |
//            SplitOp
//              | |                                    |
//            ConcatOp          ->              AffineReshapeOp
//               |                                     |

class SplitConcatRewriter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    SplitConcatRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("SplitConcatRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitConcatRewriter::matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite ConcatOp operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto splitOp = origOp.getOperand(0).getDefiningOp<IE::SplitOp>();
    if (splitOp == nullptr) {
        return mlir::failure();
    }

    auto getConsumerResult = IE::getConcatOpConsumer<IE::ConvolutionOp>(splitOp, false, false);
    if (mlir::failed(getConsumerResult)) {
        return mlir::failure();
    }

    VPUX_THROW_WHEN(mlir::dyn_cast_or_null<IE::ConcatOp>(getConsumerResult.value()) == nullptr,
                    "Not a Concat operation");

    if (splitOp.getOutputs().size() != origOp.getInputs().size()) {
        return mlir::failure();
    }

    // Supported case for splitOp: split the dim to shape 1
    auto getSplitDim = IE::getSplitDimToShape1(splitOp);
    if (mlir::failed(getSplitDim)) {
        return mlir::failure();
    }

    // Supported case for concatOp: axis dim or adjust dims of concat with shape 1
    auto getconcatDims = getConcatDimWithShape1(origOp, true);
    if (mlir::failed(getconcatDims)) {
        return mlir::failure();
    }
    const auto concatDims = getconcatDims.value();

    const auto origOutputShape = getShape(origOp.getOutput());
    const auto reassociationMap =
            vpux::IE::getReassociationMap(getShape(splitOp.getInput()).raw(), origOutputShape.raw());
    if (mlir::failed(reassociationMap)) {
        return mlir::failure();
    }

    auto affineReshape = IE::AffineReshapeOp::create(rewriter, takeOpLoc(origOp, "reshape_in"), splitOp.getInput(),
                                                     getIntArrayOfArray(getContext(), reassociationMap.value()),
                                                     getIntArrayAttr(rewriter.getContext(), origOutputShape));
    rewriter.replaceOp(origOp, affineReshape.getOutput());

    _log.trace("[{0}] Replaced with 'IE::AffineReshapeOp'", getDebugName());

    return mlir::success();
}

//
// ConvertSplitConcatToAffineReshapePass
//

class ConvertSplitConcatToAffineReshapePass final :
        public IE::impl::ConvertSplitConcatToAffineReshapeBase<ConvertSplitConcatToAffineReshapePass> {
public:
    explicit ConvertSplitConcatToAffineReshapePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertSplitConcatToAffineReshapePass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SplitConcatRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertSplitConcatToAffineReshapePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertSplitConcatToAffineReshapePass(Logger log) {
    return std::make_unique<ConvertSplitConcatToAffineReshapePass>(log);
}
