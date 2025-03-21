//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_MOVEDYNAMICDEQUANTIZETOUSER
#define GEN_PASS_DEF_MOVEDYNAMICDEQUANTIZETOUSER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// This function verifies whether the output shape (outShape) is created by padding the lower dimensions of the input
// shape (inShape) with ones.
// For example, it returns true when reshaping 2x3 to 2x3x1x1, as the lower dimensions are padded with ones.
// It returns false when reshaping 2x3 to 1x1x2x3, as the padding is not applied to the lower dimensions.
bool isPaddingOneOnLowerDimension(ShapeRef inShape, ShapeRef outShape) {
    if (inShape.size() > outShape.size()) {
        return false;
    }

    if (!std::equal(inShape.begin(), inShape.end(), outShape.begin())) {
        return false;
    }

    for (size_t i = inShape.size(); i < outShape.size(); ++i) {
        if (outShape[Dim(i)] != 1) {
            return false;
        }
    }

    return true;
}

//
// AffineReshapePermuteCastRewriter
//

class AffineReshapePermuteCastRewriter final : public mlir::OpRewritePattern<IE::DynamicDequantizeOp> {
public:
    AffineReshapePermuteCastRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DynamicDequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DynamicDequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// Move DynamicDequantize after AffineReshape and PermuteCast
// to ease conversion of Dynamic Quantization subgraph to VPU dialect.
//  DynamicDequantize               AffineReshape
//        |                               |
//  AffineReshape            ->      PermuteCast
//        |                               |
//   PermuteCast                  DynamicDequantize

mlir::LogicalResult AffineReshapePermuteCastRewriter::matchAndRewrite(IE::DynamicDequantizeOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!origOp->hasOneUse()) {
        return mlir::failure();
    }

    auto reshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(*origOp.getOutput().getUsers().begin());
    if (reshapeOp == nullptr || !reshapeOp->hasOneUse()) {
        return matchFailed(rewriter, origOp, "Pattern mismatch");
    }

    // Currently, scale is simply reshaped by padding the lower dimension with ones.
    // Need to ensure AffineShape is reshaping tensor in the same way.
    // Otherwise, it would lead to mismatched shapes of input and scale after conversion.
    if (!isPaddingOneOnLowerDimension(getShape(reshapeOp.getInput()), getShape(reshapeOp.getOutput()))) {
        return matchFailed(rewriter, origOp, "Unsupported reshape");
    }

    auto permuteOp = mlir::dyn_cast<IE::PermuteCastOp>(*reshapeOp.getOutput().getUsers().begin());
    if (permuteOp == nullptr) {
        return matchFailed(rewriter, origOp, "Pattern mismatch");
    }

    auto newReshapeOp = rewriter.create<IE::AffineReshapeOp>(
            origOp->getLoc(), origOp.getInput(), reshapeOp.getDimMappingAttr(), reshapeOp.getShapeValueAttr());

    auto newPermuteOp = rewriter.create<IE::PermuteCastOp>(origOp->getLoc(), newReshapeOp.getOutput(),
                                                           permuteOp.getDstOrderAttr(), permuteOp.getMemPermAttr());

    const auto origReshapeRank = reshapeOp.getOutput().getType().getRank();
    auto scaleShape = getShape(origOp.getScale()).raw();
    SmallVector<int64_t> newScaleShape = SmallVector<int64_t>(origReshapeRank, 1);
    for (auto i : irange(scaleShape.size())) {
        newScaleShape[i] = scaleShape[i];
    }

    const auto scaleShapeAttr = getIntArrayAttr(getContext(), newScaleShape);
    auto scaleReshapeOp = rewriter.create<IE::AffineReshapeOp>(origOp->getLoc(), origOp.getScale(),
                                                               reshapeOp.getDimMappingAttr(), scaleShapeAttr);

    auto scalePermuteOp = rewriter.create<IE::PermuteCastOp>(origOp->getLoc(), scaleReshapeOp.getOutput(),
                                                             permuteOp.getDstOrderAttr(), permuteOp.getMemPermAttr());

    rewriter.replaceOpWithNewOp<IE::DynamicDequantizeOp>(permuteOp, newPermuteOp.getOutput(),
                                                         scalePermuteOp.getOutput(), origOp.getZp(),
                                                         origOp.getDstElemTypeAttr());
    rewriter.eraseOp(reshapeOp);
    rewriter.eraseOp(origOp);
    return mlir::success();
}

//
// MoveDynamicDequantizeToUserPass
//

class MoveDynamicDequantizeToUserPass final :
        public vpux::IE::impl::MoveDynamicDequantizeToUserBase<MoveDynamicDequantizeToUserPass> {
public:
    explicit MoveDynamicDequantizeToUserPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void MoveDynamicDequantizeToUserPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<AffineReshapePermuteCastRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveDynamicDequantizeToUserPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMoveDynamicDequantizeToUserPass(Logger log) {
    return std::make_unique<MoveDynamicDequantizeToUserPass>(log);
}
