//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/check_shrink_matmul_groups.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Transforms/DialectConversion.h>

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

Case 3:
    Convert below 14 groups Matmul where the 5D Transpose precedes AffineReshape:

                        RHS
                    1x2x1x1152x64
                        |
                    Broadcast
                        |
                    1x2x7x1152x64
                        |
                    Transpose (5D, swaps last 2 dims)
                        |
                    1x2x7x64x1152
                        |
                    AffineReshape
        LHS             |
    1x14x1x1152     1x14x64x1152
        \               /
             MatMul

    to a new 2 groups Matmul:

                        RHS
                    1x2x1x1152x64
                        |
                    Reshape
                        |
                    1x2x1152x64
                        |
                    Transpose (4D NCWH, swaps last 2 dims)
        LHS             |
    1x14x1x1152     1x2x64x1152
        |               |
    Reshape             |
        |               |
    1x2x7x1152          |
        \               /
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

mlir::LogicalResult ShrinkMatmulGroups::matchAndRewrite(IE::MatMulOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}'", getDebugName(), origOp->getLoc());

    auto lhs = origOp.getInput1();

    const auto matched = IE::matchShrinkMatmulGroupsPattern(origOp);
    if (!matched.has_value()) {
        return mlir::failure();
    }
    auto broadCastOp = matched->broadCastOp;
    auto innerTransposeOp = matched->innerTransposeOp;
    auto outerTransposeOp = matched->outerTransposeOp;

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
    auto newLhs = rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "lhs_reshape"), lhs, lhsTargetShapeAttr)
                          .getOutput();

    // Build new RHS: drop the unit d2 from the 5D broadcast input to get a 4D base shape,
    // then re-apply any transposes present in the original chain in order.
    const auto& broadcastInputShape = getShape(broadCastOp.getInput());
    const SmallVector<int64_t> rhsBaseShape = {broadcastInputShape[Dims5D::Act::N], newGroupNum,
                                               broadcastInputShape[Dims5D::Act::H],
                                               broadcastInputShape[Dims5D::Act::W]};
    mlir::Value newRhs = rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "rhs_reshape"),
                                                        broadCastOp.getInput(), getIntArrayAttr(ctx, rhsBaseShape))
                                 .getOutput();

    if (innerTransposeOp != nullptr) {
        // Re-apply the inner 5D last-2-dims swap as a 4D NCWH transpose.
        const auto ncwhOrder =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{0, 1, 3, 2}, ctx));
        newRhs = rewriter.create<IE::TransposeOp>(appendLoc(origOp->getLoc(), "rhs_inner_transpose"), newRhs, nullptr,
                                                  ncwhOrder)
                         .getOutput();
    }

    if (outerTransposeOp != nullptr) {
        newRhs = rewriter.create<IE::TransposeOp>(appendLoc(origOp->getLoc(), "rhs_transpose"), newRhs, nullptr,
                                                  outerTransposeOp.getOrderValueAttr())
                         .getOutput();
    }

    // Create new group Matmul
    auto newMatMul = cloneMatMulOp(rewriter, origOp, newLhs, newRhs);
    newMatMul->setLoc(appendLoc(origOp->getLoc(), "new_group_mul"));

    auto outputShape = getShape(origOp.getOutput());
    const auto outputShapeAttr = getIntArrayAttr(ctx, outputShape);
    auto outReshape = rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "output_reshape"),
                                                     newMatMul->getResult(0), outputShapeAttr);

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
    collectOpsAndApplyPatterns(func, std::move(patterns));
}
}  // namespace

//
// createShrinkMatmulGroupsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createShrinkMatmulGroupsPass(Logger log) {
    return std::make_unique<ShrinkMatmulGroupsPass>(log);
}
