//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_BROADCASTINPUTFORMULTIPLY
#define GEN_PASS_DEF_BROADCASTINPUTFORMULTIPLY
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// BroadcastInputRewriter
//

class BroadcastInputRewriter final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    BroadcastInputRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        setDebugName("BroadcastInputRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value broadcastInput(mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx, mlir::Location loc,
                               mlir::Value broadcastInput, ShapeRef targetShape) const;
    mlir::Value castToDimsOrder(mlir::Value value, mlir::AffineMap dstOrderMap, mlir::AffineMap memOrderMap,
                                StringRef suffix, mlir::PatternRewriter& rewriter) const;
    bool isBroadcastEfficient(NDTypeInterface inputType, NDTypeInterface outputType) const;

    Logger _log;
};

mlir::Value BroadcastInputRewriter::broadcastInput(mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx,
                                                   mlir::Location loc, mlir::Value broadcastInput,
                                                   ShapeRef targetShape) const {
    // Cast to canonical order
    const auto canonicalOrder = DimsOrder::NCHW;
    const auto canonicalOrderMap = canonicalOrder.toAffineMap(ctx);
    auto canonicalPermuteCast = rewriter.createOrFold<IE::PermuteCastOp>(
            appendLoc(broadcastInput.getLoc(), "_canonical_permute_cast"), broadcastInput, canonicalOrderMap,
            mlir::AffineMap::getMultiDimIdentityMap(getShape(broadcastInput).size(), ctx));

    // Broadcast to target shape
    const auto origOrder = mlir::cast<NDTypeInterface>(broadcastInput.getType()).getDimsOrder();
    const auto memShape = origOrder.toMemoryOrder(targetShape);
    auto newTargetShape = canonicalOrder.toLogicalOrder(memShape);
    auto broadCast = IE::createBroadcast(rewriter, appendLoc(loc, "shape"), canonicalPermuteCast, newTargetShape);

    // Cast to the original dims order
    return rewriter.createOrFold<IE::PermuteCastOp>(appendLoc(broadcastInput.getLoc(), "_broadcast_permute_cast"),
                                                    broadCast, origOrder.toAffineMap(ctx),
                                                    mlir::AffineMap::getMultiDimIdentityMap(origOrder.numDims(), ctx));
}

mlir::Value BroadcastInputRewriter::castToDimsOrder(mlir::Value value, mlir::AffineMap dstOrderMap,
                                                    mlir::AffineMap memOrderMap, StringRef suffix,
                                                    mlir::PatternRewriter& rewriter) const {
    return rewriter.createOrFold<IE::PermuteCastOp>(appendLoc(value.getLoc(), suffix), value, dstOrderMap, memOrderMap);
}

bool BroadcastInputRewriter::isBroadcastEfficient(NDTypeInterface inputType, NDTypeInterface outputType) const {
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();
    auto diffDims = IE::getDiffInOutSizeDims(inputShape, outputShape);

    // If more than one dimension differs, broadcasting is not efficient
    if (diffDims.size() > 1) {
        return false;
    }

    // No need to broadcast if shapes are the same
    if (diffDims.empty()) {
        return true;
    }

    auto diffDim = diffDims.front();
    auto inputOrder = inputType.getDimsOrder();
    auto highestNonOneDim = getHighestNonTrivialDim(inputShape, inputOrder);
    // If input shape is 1x1x1x1, broadcasting is efficient
    if (!highestNonOneDim.has_value()) {
        return true;
    }

    auto highestNonOneDimPos = inputOrder.dimPos(highestNonOneDim.value());
    auto broadcastDimPos = inputOrder.dimPos(diffDim);

    // Ensure broadcasting is on the highest dimension
    return broadcastDimPos <= highestNonOneDimPos;
}

mlir::LogicalResult BroadcastInputRewriter::matchAndRewrite(IE::MultiplyOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    const auto ctx = origOp->getContext();
    const auto loc = origOp->getLoc();

    mlir::Value lhsInput = origOp.getInput1();
    const auto lhsShape = getShape(lhsInput);

    mlir::Value rhsInput = origOp.getInput2();
    const auto rhsShape = getShape(rhsInput);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto outputShape = outputType.getShape();

    if (lhsShape.size() != 4) {
        _log.trace("Only support 4D tensor, but got {0}D", lhsShape.size());
        return mlir::failure();
    }

    if (lhsShape == outputShape && rhsShape == outputShape) {
        _log.trace("Inputs have same shape, no need for broadcast");
        return mlir::failure();
    }

    // Cast lhs & rhs input to NHWC layout
    const auto dstOrder = DimsOrder::NHWC;
    const auto dstOrderMap = dstOrder.toAffineMap(ctx);
    const auto dimsOrder = outputType.getDimsOrder();

    auto alignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());
    const auto doesInputNeedBroadCast = [&](mlir::Value input) {
        return getShape(input) != outputShape;
    };

    // Support multiply when its inner most dim is aligned
    auto getMemMapOfInnermostMultiply = [&]() -> std::optional<SmallVector<mlir::AffineMap>> {
        auto getInnermostDim = [](const vpux::DimsOrder& order) {
            return order.toDim(MemDim(order.numDims() - 1));
        };

        auto innerMostDim = getInnermostDim(dimsOrder);

        auto hasLhsNotAligned = lhsShape[innerMostDim] % alignment != 0;
        auto hasRightNotAligned = rhsShape[innerMostDim] % alignment != 0;

        if (hasLhsNotAligned || hasRightNotAligned) {
            _log.trace("Innermost dim size is not aligned to {0}, Innermost: {1}", alignment, innerMostDim);
            return std::nullopt;
        }

        if (!isBroadcastEfficient(lhsInput.getType(), outputType) ||
            !isBroadcastEfficient(rhsInput.getType(), outputType)) {
            _log.trace("It is not efficient to broadcast the non-highest dimension");
            return std::nullopt;
        }

        SmallVector<mlir::AffineMap> memOrderMap;
        const auto inMemOrderMap = mlir::AffineMap::getMultiDimIdentityMap(dstOrder.numDims(), ctx);
        memOrderMap.push_back(inMemOrderMap);
        const auto outMemOrderMap = mlir::AffineMap::getMultiDimIdentityMap(dimsOrder.numDims(), ctx);
        memOrderMap.push_back(outMemOrderMap);
        return memOrderMap;
    };

    // Support multiply when it has splat input
    // Like input1([1, 1, 1, 1]) x input2([1, 512, 1, 1])
    auto getMemMapOfSplatMultiply = [&]() -> std::optional<SmallVector<mlir::AffineMap>> {
        auto lhsTotalSize = lhsShape.totalSize();
        auto rhsTotalSize = rhsShape.totalSize();

        auto alignDim = Dims4D::Act::C;
        auto hasSplatValue = (lhsTotalSize == 1 && rhsTotalSize == rhsShape[alignDim]) ||
                                             (rhsTotalSize == 1 && lhsTotalSize == lhsShape[alignDim])
                                     ? true
                                     : false;
        if (!hasSplatValue) {
            _log.trace("Multiply has no splat or aligned value");
            return std::nullopt;
        }

        SmallVector<mlir::AffineMap> memOrderMap;
        const auto inMemOrderMap = getPermutationFromOrders(DimsOrder::fromValue(lhsInput), dstOrder, ctx);
        memOrderMap.push_back(inMemOrderMap);
        const auto outMemOrderMap = getPermutationFromOrders(dstOrder, DimsOrder::fromValue(origOp.getOutput()), ctx);
        memOrderMap.push_back(outMemOrderMap);
        return memOrderMap;
    };

    SmallVector<mlir::AffineMap> memOrderMap;
    if (getMemMapOfInnermostMultiply().has_value()) {
        memOrderMap = getMemMapOfInnermostMultiply().value();
    } else if (getMemMapOfSplatMultiply().has_value()) {
        memOrderMap = getMemMapOfSplatMultiply().value();
    } else {
        return mlir::failure();
    }

    // Broadcast input
    if (doesInputNeedBroadCast(lhsInput)) {
        lhsInput = broadcastInput(rewriter, ctx, appendLoc(loc, "broadcast_lhs"), origOp.getInput1(), outputShape);
    }
    auto newLhsInput = castToDimsOrder(lhsInput, dstOrderMap, memOrderMap[0], "_lhs_permute_cast", rewriter);

    if (doesInputNeedBroadCast(rhsInput)) {
        rhsInput = broadcastInput(rewriter, ctx, appendLoc(loc, "broadcast_rhs"), origOp.getInput2(), outputShape);
    }
    auto newRhsInput = castToDimsOrder(rhsInput, dstOrderMap, memOrderMap[0], "_rhs_permute_cast", rewriter);

    // Create new Multiply
    auto newOutputType = outputType.changeDimsOrder(dstOrder).changeShape(getShape(newRhsInput));
    auto newMultiplyOp = rewriter.create<IE::MultiplyOp>(
            origOp.getLoc(), newOutputType, newLhsInput, newRhsInput, origOp.getAutoBroadcastAttr(),
            origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
    // Cast to the original dims order
    auto result = castToDimsOrder(newMultiplyOp.getOutput(), dimsOrder.toAffineMap(ctx), memOrderMap[1],
                                  "_output_permute_cast", rewriter);
    rewriter.replaceOp(origOp, result);
    return mlir::success();
}

//
// BroadcastInputForMultiplyPass
//
class BroadcastInputForMultiplyPass final :
        public IE::impl::BroadcastInputForMultiplyBase<BroadcastInputForMultiplyPass> {
public:
    explicit BroadcastInputForMultiplyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void BroadcastInputForMultiplyPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<BroadcastInputRewriter>(&ctx, _log);
    IE::PermuteCastOp::getCanonicalizationPatterns(patterns, &ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createBroadcastInputForMultiplyPass(Logger log) {
    return std::make_unique<BroadcastInputForMultiplyPass>(log);
}
