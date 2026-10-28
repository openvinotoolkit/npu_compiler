//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_UPSTREAMSLICE
#define GEN_PASS_DEF_UPSTREAMSLICE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// UpstreamSlicePass
//

class UpstreamSlicePass final : public IE::impl::UpstreamSliceBase<UpstreamSlicePass> {
public:
    explicit UpstreamSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GenericSliceUpstreaming;
    class SliceUpstreamWithAffineReshape;

private:
    void safeRunOnFunc() final;
};

//
// GenericSliceUpstreaming
//

class UpstreamSlicePass::GenericSliceUpstreaming final : public mlir::OpInterfaceRewritePattern<IE::LayerOpInterface> {
public:
    GenericSliceUpstreaming(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayerOpInterface>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayerOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isUpstreamPossible(IE::LayerOpInterface sliceOp, mlir::Value tensor, Logger log) {
    if (mlir::isa<mlir::BlockArgument>(tensor)) {
        return false;
    }
    mlir::Operation* parentOp = tensor.getDefiningOp();
    // Unary and eltwise ops are primary candidates for upstreaming slice ops.
    // Later on, implementation could handle also Conv, Pool upstreaming
    if (parentOp == nullptr || !parentOp->hasTrait<IE::EltwiseOp>()) {
        return false;
    }

    if (parentOp->getNumResults() > 1) {
        return false;
    }

    // Consider a strided slice with begins, ends or strides unknown at compile time:
    // %GELU = IE.Gelu(%arg0) -> 1x2x3x4xf16
    // %SLICE = IE.StridedSlice(%GELU, %arg1) -> ?x?x?x?xf16
    // The upstreaming may be unsafe because the producer does not guarantee dynamic shapes support:
    // %SLICE = IE.StridedSlice(%arg0, %arg1) -> ?x?x?x?xf16    // OK
    // %GELU = IE.Gelu(%arg0) -> ?x?x?x?xf16                    // Not OK, GeLU kernel expected static input
    if (getShape(sliceOp->getResult(0)).isDynamic()) {
        return false;
    }

    // Strided slice does not support datatypes generically
    // so we can't afford changing datatype of this operation.
    if (mlir::isa<IE::StridedSliceOp>(sliceOp)) {
        auto sliceOpElementType = mlir::cast<vpux::NDTypeInterface>(tensor.getType()).getElementType();
        auto parentOpElementType =
                mlir::cast<vpux::NDTypeInterface>(parentOp->getOperand(0).getType()).getElementType();
        if (sliceOpElementType != parentOpElementType) {
            return false;
        }
    }

    // Another restriction due to limited implementation.
    // Upstreaming through the graph using op interfaces it's hard
    // enough to reason about which operands are path of the activation
    // path and which are parameters.
    // An interface that makes this dinstinction easy would be of help.
    const auto operands = parentOp->getOperands();
    if (!mlir::isa<IE::FakeQuantizeOp>(parentOp) && operands.size() > 1 &&
        std::adjacent_find(operands.begin(), operands.end(), [](mlir::Value val1, mlir::Value val2) {
            return getShape(val1) != getShape(val2);
        }) != operands.end()) {
        return false;
    }

    const auto inputShape = getShape(sliceOp.getInputs()[0]);
    const auto outputShape = getShape(sliceOp.getOutputs()[0]);

    // Can't reason yet with generic shape dimensions
    if (inputShape.size() != 4 || outputShape.size() != 4) {
        return false;
    }

    // Can't handle yet upstreaming and adapting channelwise parameters
    if (auto fqParentOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(parentOp)) {
        const auto fqInputShape = getShape(fqParentOp.getInputLow());
        const auto fqOutputShape = getShape(fqParentOp.getOutputLow());

        // check if FQ is per tensor
        if (IE::isPerTensorFQ({fqParentOp})) {
            return true;
        }

        // If FQ is not per tensor make sure that the slice doesn't happen in quantization axis
        for (size_t i = 0; i < fqInputShape.size(); ++i) {
            if (fqInputShape[Dim(i)] == 1) {
                continue;
            }

            if (inputShape[Dim(i)] != outputShape[Dim(i)]) {
                return false;
            }
        }

        for (size_t i = 0; i < fqOutputShape.size(); ++i) {
            if (fqOutputShape[Dim(i)] == 1) {
                continue;
            }

            if (inputShape[Dim(i)] != outputShape[Dim(i)]) {
                return false;
            }
        }
    } else {
        // Can't handle yet upstreaming and adapting channelwise parameters
        if (const auto quantAxis = IE::getQuantAxisIndex(parentOp, log)) {
            if (inputShape[Dim(quantAxis.value())] != outputShape[Dim(quantAxis.value())]) {
                return false;
            }
        }
    }

    return true;
}

mlir::LogicalResult UpstreamSlicePass::GenericSliceUpstreaming::matchAndRewrite(IE::LayerOpInterface origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    if (!mlir::isa<IE::SliceOp, IE::StridedSliceOp>(origOp)) {
        return mlir::failure();
    }

    auto origInput = origOp.getInputs()[0];
    if (!origInput.hasOneUse()) {
        return mlir::failure();
    }

    if (!isUpstreamPossible(origOp, origInput, _log)) {
        return mlir::failure();
    }

    auto parentOp = mlir::cast<mlir::InferTypeOpInterface>(origInput.getDefiningOp());
    rewriter.setInsertionPoint(parentOp);
    auto opOperands = parentOp->getOpOperands();
    if (std::adjacent_find(opOperands.begin(), opOperands.end(), [&](mlir::OpOperand& val1, mlir::OpOperand& val2) {
            return val1.get() != val2.get();
        }) == opOperands.end()) {
        auto newSlice = mlir::cast<mlir::InferTypeOpInterface>(rewriter.clone(*origOp));
        newSlice->setOperand(0, opOperands[0].get());
        inferReturnTypes(newSlice, InferShapedTypeMode::ALL);
        for (auto& operand : opOperands) {
            operand.set(newSlice->getResult(0));
        }
        extendOpLoc(newSlice, "{0}", opOperands[0].getOperandNumber());
    } else {
        for (auto& operand : opOperands) {
            auto newSlice = mlir::cast<mlir::InferTypeOpInterface>(rewriter.clone(*origOp));
            newSlice->setOperand(0, operand.get());
            extendOpLoc(newSlice, "{0}", operand.getOperandNumber());
            inferReturnTypes(newSlice, InferShapedTypeMode::ALL);
            operand.set(newSlice->getResult(0));
            // For FakeQuantize the activation input is represented by first operand
            if (mlir::isa<IE::FakeQuantizeOp>(parentOp)) {
                break;
            }
        }
    }

    VPUX_THROW_UNLESS(parentOp->getResults().size() == 1, "Don't support backprop for multiple outputs yet '{0}'",
                      parentOp);
    inferReturnTypes(parentOp, InferShapedTypeMode::SHAPE);

    rewriter.replaceOp(origOp, parentOp->getResults());
    return mlir::success();
}

// the GenericSliceUpstreaming pattern only matches SliceOp adjacent to Eltwise-like ops.
// propagating SliceOp forward could be generalized to cast-like ops and reshape-like ops, to enable further
// optimization, E#197021.
//              parentOp                           parentOp
//                | |                                 |
//             EltwiseOp                       AffineReshapeOp (optional)
//                 |                                  |
//          AffineReshapeOp (optional)    ==>       SliceOp
//                 |                                  |
//          QuantizeCastOp  (optional)         AffineReshapeOp (optional)
//                 |                                 | |
//              SliceOp                           EltwiseOp
//                                                    |
//                                              QuantizeCastOp (optional)
//                                                    |
//                                              AffineReshapeOp (optional)
//

class UpstreamSlicePass::SliceUpstreamWithAffineReshape final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    SliceUpstreamWithAffineReshape(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        this->setDebugName("SliceUpstreamWithAffineReshape");
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp op, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UpstreamSlicePass::SliceUpstreamWithAffineReshape::matchAndRewrite(
        IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    auto input = origOp.getInput();
    auto nextParentOp = input.getDefiningOp();
    if (nextParentOp == nullptr || !nextParentOp->hasOneUse()) {
        return mlir::failure();
    }
    auto output = origOp.getOutput();

    auto quantCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(nextParentOp);
    if (quantCastOp) {
        if (!quantCastOp->hasOneUse()) {
            return mlir::failure();
        }
        nextParentOp = quantCastOp.getInput().getDefiningOp();
    }

    auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(nextParentOp);
    if (affineReshapeOp) {
        if (!affineReshapeOp->hasOneUse()) {
            return mlir::failure();
        }
        nextParentOp = affineReshapeOp.getInput().getDefiningOp();
    }

    auto eltwiseOp = nextParentOp;
    if (eltwiseOp == nullptr || !eltwiseOp->hasTrait<IE::EltwiseOp>() || !eltwiseOp->hasOneUse()) {
        return mlir::failure();
    }

    auto lhs = eltwiseOp->getOperand(0);
    if (llvm::any_of(eltwiseOp->getOperands(), [&lhs](mlir::Value operand) {
            return operand != lhs;
        })) {
        return mlir::failure();
    }

    auto origEltwiseShape = getShape(eltwiseOp->getResult(0));
    auto sliceInShape = getShape(input);
    auto sliceOutShape = getShape(output);
    // infer the dim order for the second AffineReshapeOp in the post-transformation op chain
    // this reshape aims to recover the shape for the EltwiseOp
    auto reshapeBackDim = IE::getReassociationMap(sliceInShape, origEltwiseShape);
    if (mlir::failed(reshapeBackDim)) {
        return mlir::failure();
    }

    // infer the output shape for the second AffineReshapeOp in the post-transformation op chain
    auto sliceAxes = IE::getDiffInOutSizeDims(sliceInShape, sliceOutShape);
    if (sliceAxes.empty() || sliceAxes.size() != 1) {
        return mlir::failure();
    }
    auto sliceAxis = sliceAxes[0];
    auto dstAxis = reshapeBackDim.value()[sliceAxis.ind()];
    if (dstAxis.size() != 1) {
        return mlir::failure();
    }
    auto sliceInShapeOnAxis = sliceInShape[sliceAxis];
    auto sliceOutShapeOnAxis = sliceOutShape[sliceAxis];
    auto origEltwiseShapeOnDstAxis = origEltwiseShape[Dim(dstAxis[0])];
    auto newAddShapeOnDstAxis = origEltwiseShapeOnDstAxis / (sliceInShapeOnAxis / sliceOutShapeOnAxis);
    SmallVector<int64_t> newOutShape = to_small_vector(origEltwiseShape);
    newOutShape[dstAxis[0]] = newAddShapeOnDstAxis;
    auto newOutShapeAttr = getIntArrayAttr(getContext(), newOutShape);

    // infer the dim order for the last AffineReshapeOp in the post-transformation op chain
    auto newLastAffineReshapeDim = IE::getReassociationMap(ShapeRef(newOutShape), sliceOutShape);
    if (mlir::failed(newLastAffineReshapeDim)) {
        return mlir::failure();
    }
    mlir::Value currentValue = lhs;
    if (affineReshapeOp) {
        auto lhsReshape =
                IE::AffineReshapeOp::create(rewriter, affineReshapeOp->getLoc(), lhs,
                                            affineReshapeOp.getDimMappingAttr(), affineReshapeOp.getShapeValueAttr());
        currentValue = lhsReshape.getResult();
    }
    auto lhsSlice = IE::SliceOp::create(rewriter, origOp.getLoc(), currentValue, origOp.getStaticOffsetsAttr(),
                                        origOp.getStaticSizesAttr());
    currentValue = lhsSlice.getResult();
    if (affineReshapeOp) {
        auto lhsReshapeBack =
                IE::AffineReshapeOp::create(rewriter, origOp.getLoc(), currentValue,
                                            getIntArrayOfArray(getContext(), reshapeBackDim.value()), newOutShapeAttr);
        currentValue = lhsReshapeBack.getResult();
    }

    auto newEltwiseInput = currentValue;
    auto newEltwise = rewriter.clone(*eltwiseOp);
    auto newEltwiseOperands = newEltwise->getOpOperands();
    for (auto& operand : newEltwiseOperands) {
        operand.set(newEltwiseInput);
    }
    inferReturnTypes(newEltwise, InferShapedTypeMode::SHAPE);

    currentValue = newEltwise->getResult(0);
    if (quantCastOp) {
        auto newQuantCast =
                IE::QuantizeCastOp::create(rewriter, quantCastOp->getLoc(),
                                           mlir::cast<vpux::NDTypeInterface>(quantCastOp.getOutput().getType())
                                                   .changeShape(getShape(currentValue)),
                                           currentValue, quantCastOp.getDstElemTypeAttr());
        currentValue = newQuantCast->getResult(0);
    }
    if (affineReshapeOp) {
        auto newLastAffineReshapeOp =
                IE::AffineReshapeOp::create(rewriter, origOp.getLoc(), currentValue,
                                            getIntArrayOfArray(getContext(), newLastAffineReshapeDim.value()),
                                            getIntArrayAttr(getContext(), sliceOutShape));
        currentValue = newLastAffineReshapeOp.getResult();
    }

    rewriter.replaceOp(origOp, currentValue);
    if (quantCastOp) {
        rewriter.eraseOp(quantCastOp);
    }
    if (affineReshapeOp) {
        rewriter.eraseOp(affineReshapeOp);
    }
    rewriter.eraseOp(eltwiseOp);
    return mlir::success();
}

//
// safeRunOnFunc
//

void UpstreamSlicePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericSliceUpstreaming>(&ctx, _log);
    patterns.add<SliceUpstreamWithAffineReshape>(&ctx, _log);

    IE::SliceOp::getCanonicalizationPatterns(patterns, &ctx);
    IE::StridedSliceOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUpstreamSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUpstreamSlicePass(Logger log) {
    return std::make_unique<UpstreamSlicePass>(log);
}
