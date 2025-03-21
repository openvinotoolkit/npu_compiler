//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_SHRINKMATMULGROUPS
#define GEN_PASS_DEF_SHRINKMATMULGROUPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ShrinkMatmulGroups
//

/*

Case 1:
    Convert below 24 groups Matmul:
                        RHS
                    1x8x1x1024x64
                        |
                    Broadcast
                        |
                    1x8x3x1024x64
                        |
                    AffineReshape
        LHS             |
    1x24x1x64       1x24x1024x64
        \               /
             MatMul

    to a new 8 groups Matmul:
        LHS             RHS
    1x24x1x64       1x8x1x1024x64
        |               |
    Reshape         Reshape
        |               |
    1x8x3x64        1x8x1024x64
        \               /
            MatMul

Case 2:
    Convert below 24 groups Matmul:

                        RHS
                    1x8x1x1024x64
                        |
                    Broadcast
                        |
                    1x8x3x1024x64
                        |
                    AffineReshape
                        |
                    1x24x1024x64
                        |
                    Transpose
        LHS             |
    1x24x1x1024     1x24x64x1024
        \               /
             MatMul

    to a new 8 groups Matmul:

                        RHS
                    1x8x1x1024x64
                        |
                    Reshape
        LHS             |
    1x24x1x64       1x8x1024x64
        |               |
    Reshape         Transpose
        |               |
    1x8x3x64        1x24x64x1024
        \                /
            MatMul
*/

class ShrinkMatmulGroups final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    ShrinkMatmulGroups(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MatMulOp>(ctx), _log(log) {
        setDebugName("ShrinkMatmulGroups");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool checkMatMul(IE::MatMulOp origOp) {
    const auto is4DShape = [](ShapeRef shape) {
        return shape.size() == 4;
    };
    auto lhs = origOp.getInput1();
    auto rhs = origOp.getInput2();

    auto lhsShape = getShape(lhs);
    auto rhsShape = getShape(rhs);

    if (!is4DShape(lhsShape) || !is4DShape(rhsShape)) {
        return false;
    }

    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;

    // Right now it's expected to be the case when transposeA = false and transposeB = true
    if (!IE::isMatmulWithRHSTransposition(origOp)) {
        return false;
    }
    if (lhsShape[N] != rhsShape[N] || lhsShape[C] != rhsShape[C] || lhsShape[W] != rhsShape[W]) {
        return false;
    }

    // Restrict to Dim H = 1, otherwise it would break the VF pattern for MatMul-Add-Softmax-MatMul in LLM
    // TODO: Remove the restriction, see E#138709
    if (lhsShape[H] != 1) {
        return false;
    }

    return true;
}

bool checkTranspose(IE::TransposeOp transposeOp) {
    // TransposeOp should only transpose 4D spatial dims
    const auto transposePerm = DimsOrder::fromAffineMap(transposeOp.getOrderValue().value());
    return transposePerm == DimsOrder::NCWH;
}

bool checkAffineReshape(IE::AffineReshapeOp affineReshapeOp) {
    if (affineReshapeOp == nullptr) {
        return false;
    }

    auto inputShape = getShape(affineReshapeOp.getInput());
    auto outputShape = getShape(affineReshapeOp.getOutput());

    // ensure the input/output shape of AffineReshapeOp to be 5/4D
    if (inputShape.size() != 5 || outputShape.size() != 4) {
        return false;
    }

    // Since Matmul actually happens on 2D tensors, we should only check the last 2 dims
    return (inputShape[Dims5D::Act::H] == outputShape[Dims4D::Act::H] &&
            inputShape[Dims5D::Act::W] == outputShape[Dims4D::Act::W]);
}

bool checkBroadCast(IE::BroadcastOp broadcastOp) {
    if (broadcastOp == nullptr) {
        return false;
    }

    const auto is5DShape = [](ShapeRef shape) {
        return shape.size() == 5;
    };
    auto inputShape = getShape(broadcastOp.getInput());
    auto outputShape = getShape(broadcastOp.getOutput());
    if (!is5DShape(inputShape) || !is5DShape(outputShape)) {
        return false;
    }

    auto broadCastDim = IE::getDiffInOutSizeDims(inputShape, outputShape);
    if (broadCastDim.size() != 1) {
        return false;
    }

    // BroadcastOp should broadcast 5D tensor on the first spatial dim (d2)
    return broadCastDim.front() == Dims5D::Act::getSpatialDim(0) && inputShape[broadCastDim.front()] == 1;
}

mlir::LogicalResult ShrinkMatmulGroups::matchAndRewrite(IE::MatMulOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MatMulOp at '{1}'", origOp->getName(), origOp->getLoc());

    auto lhs = origOp.getInput1();
    auto rhs = origOp.getInput2();

    if (!checkMatMul(origOp)) {
        return mlir::failure();
    }
    IE::AffineReshapeOp reshapeOp = nullptr;
    auto transposeOp = rhs.getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr) {
        reshapeOp = rhs.getDefiningOp<IE::AffineReshapeOp>();
    } else {
        if (!checkTranspose(transposeOp)) {
            return mlir::failure();
        }
        reshapeOp = transposeOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    }

    if (!checkAffineReshape(reshapeOp)) {
        return mlir::failure();
    }

    auto broadCastOp = reshapeOp.getInput().getDefiningOp<IE::BroadcastOp>();
    if (!checkBroadCast(broadCastOp)) {
        return mlir::failure();
    }

    auto ctx = rewriter.getContext();
    auto broadcastOutputShape = getShape(broadCastOp.getOutput());
    int64_t newGroupNum = broadcastOutputShape[Dims5D::Act::C];

    // Create new LHS by reshaping the original LHS
    auto origLhsShape = getShape(lhs);
    SmallVector<int64_t> lhsTargetShape = to_small_vector(origLhsShape);
    lhsTargetShape[Dims4D::Act::C.ind()] = newGroupNum;
    lhsTargetShape[Dims4D::Act::H.ind()] = origLhsShape[Dims4D::Act::H] * origLhsShape[Dims4D::Act::C] / newGroupNum;
    VPUX_THROW_WHEN(origLhsShape[Dims4D::Act::C] % newGroupNum != 0, "Unexpected origLhsShape {0} and newGroupNum {1}",
                    origLhsShape, newGroupNum);
    const auto lhsTargetShapeAttr = getIntArrayAttr(ctx, lhsTargetShape);
    auto newLhs = rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "lhs_reshape"), lhs, nullptr, false,
                                                 lhsTargetShapeAttr)
                          .getOutput();

    // Create new RHS chain: reshape and transpose the original input of BroadCastOp
    SmallVector<int64_t> targetShape = to_small_vector(getShape(reshapeOp.getOutput()));
    targetShape[Dims4D::Act::C.ind()] = newGroupNum;
    const auto targetShapeAttr = getIntArrayAttr(ctx, targetShape);
    auto newRhs = rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "rhs_reshape"), broadCastOp.getInput(),
                                                 nullptr, false, targetShapeAttr)
                          .getOutput();

    if (transposeOp != nullptr) {
        newRhs = rewriter.create<IE::TransposeOp>(appendLoc(origOp->getLoc(), "rhs_transpose"), newRhs, nullptr,
                                                  transposeOp.getOrderValueAttr())
                         .getOutput();
    }

    // Create new group Matmul
    auto newMatMul =
            rewriter.create<IE::MatMulOp>(appendLoc(origOp->getLoc(), "new_group_mul"), newLhs, newRhs,
                                          origOp.getTransposeA(), origOp.getTransposeB(), origOp.getPostOpAttr());

    auto outputShape = getShape(origOp.getOutput());
    const auto outputShapeAttr = getIntArrayAttr(ctx, outputShape);
    auto outReshape = rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "output_reshape"),
                                                     newMatMul.getOutput(), nullptr, false, outputShapeAttr);

    _log.trace("Successfully shrunk number of groups at {0}", origOp.getLoc());
    rewriter.replaceOp(origOp, outReshape.getOutput());

    return mlir::success();
}

//
// ShrinkMatmulGroupsPass
//

class ShrinkMatmulGroupsPass final : public IE::impl::ShrinkMatmulGroupsBase<ShrinkMatmulGroupsPass> {
public:
    explicit ShrinkMatmulGroupsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ShrinkMatmulGroupsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ShrinkMatmulGroups>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createShrinkMatmulGroupsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createShrinkMatmulGroupsPass(Logger log) {
    return std::make_unique<ShrinkMatmulGroupsPass>(log);
}
