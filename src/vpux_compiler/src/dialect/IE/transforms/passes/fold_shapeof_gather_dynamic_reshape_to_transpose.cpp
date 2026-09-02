//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FOLDSHAPEOFGATHERDYNAMICRESHAPETOTRANSPOSE
#define GEN_PASS_DEF_FOLDSHAPEOFGATHERDYNAMICRESHAPETOTRANSPOSE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class FoldShapeOfGatherDynamicReshapeToTransposeRewriter final : public mlir::OpRewritePattern<IE::DynamicReshapeOp> {
public:
    FoldShapeOfGatherDynamicReshapeToTransposeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DynamicReshapeOp>(ctx), _log(log) {
        setDebugName("FoldShapeOfGatherDynamicReshapeToTransposeRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::DynamicReshapeOp reshapeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

std::optional<SmallVector<unsigned>> extractPermutationFromConstant(mlir::Value indices, int64_t expectedRank) {
    auto constOp = indices.getDefiningOp<Const::DeclareOp>();
    if (constOp == nullptr) {
        return std::nullopt;
    }

    const auto indicesType = mlir::dyn_cast<mlir::RankedTensorType>(indices.getType());
    if (indicesType == nullptr || indicesType.getRank() != 1) {
        return std::nullopt;
    }
    if (indicesType.getShape()[0] != expectedRank) {
        return std::nullopt;
    }

    const auto content = constOp.getContent();
    const auto rawValues = to_small_vector(content.getValues<int64_t>());

    SmallVector<unsigned> perm;
    perm.reserve(rawValues.size());
    SmallVector<bool> seen(expectedRank, false);
    for (const auto value : rawValues) {
        if (value < 0 || value >= expectedRank) {
            return std::nullopt;
        }
        if (seen[value]) {
            return std::nullopt;
        }
        seen[value] = true;
        perm.push_back(checked_cast<unsigned>(value));
    }
    return perm;
}

mlir::LogicalResult FoldShapeOfGatherDynamicReshapeToTransposeRewriter::matchAndRewrite(
        IE::DynamicReshapeOp reshapeOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", reshapeOp->getName(), reshapeOp->getLoc());
    const auto nested = _log.nest();

    if (reshapeOp.getOnlySetShape()) {
        return matchFailed(nested, rewriter, reshapeOp, "DynamicReshape has only_set_shape attribute");
    }

    auto gatherOp = reshapeOp.getShape().getDefiningOp<IE::GatherOp>();
    if (gatherOp == nullptr) {
        return matchFailed(nested, rewriter, reshapeOp, "Shape operand is not produced by IE.Gather");
    }
    if (gatherOp.getAxisValue() != 0 || gatherOp.getBatchDims() != 0) {
        return matchFailed(nested, rewriter, reshapeOp, "IE.Gather is not axis=0, batch_dims=0");
    }

    auto shapeOfOp = gatherOp.getInput().getDefiningOp<IE::ShapeOfOp>();
    if (shapeOfOp == nullptr) {
        return matchFailed(nested, rewriter, reshapeOp, "IE.Gather input is not produced by IE.ShapeOf");
    }

    if (shapeOfOp.getInput() != reshapeOp.getInput()) {
        return matchFailed(nested, rewriter, reshapeOp,
                           "IE.ShapeOf and IE.DynamicReshape do not share the same data tensor");
    }

    const auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(reshapeOp.getInput().getType());
    if (inputType == nullptr) {
        return matchFailed(nested, rewriter, reshapeOp, "IE.DynamicReshape input is not a RankedTensorType");
    }
    const auto inputRank = inputType.getRank();

    auto permOpt = extractPermutationFromConstant(gatherOp.getIndices(), inputRank);
    if (!permOpt.has_value()) {
        return matchFailed(nested, rewriter, reshapeOp,
                           "IE.Gather indices are not a constant permutation of the input rank");
    }
    const auto& perm = *permOpt;

    const auto inShape = inputType.getShape();
    const auto outShape = parseIntArrayAttr<int64_t>(reshapeOp.getOutputShapeAttr());
    if (static_cast<int64_t>(outShape.size()) != inputRank) {
        return matchFailed(nested, rewriter, reshapeOp, "Rank mismatch between input and DynamicReshape output_shape");
    }
    for (int64_t i = 0; i < inputRank; ++i) {
        if (inShape[perm[i]] != outShape[i]) {
            return matchFailed(nested, rewriter, reshapeOp,
                               "DynamicReshape output_shape does not match permutation of input shape");
        }
    }

    if (auto inBounded = mlir::dyn_cast<Core::BoundedTensorType>(inputType)) {
        const auto inBounds = inBounded.getBounds().raw();
        const auto outBounds = parseIntArrayAttr<int64_t>(reshapeOp.getOutputBoundsAttr());
        if (static_cast<int64_t>(outBounds.size()) != inputRank || static_cast<int64_t>(inBounds.size()) != inputRank) {
            return matchFailed(nested, rewriter, reshapeOp,
                               "Rank mismatch between input bounds and DynamicReshape output_bounds");
        }
        for (int64_t i = 0; i < inputRank; ++i) {
            if (inBounds[perm[i]] != outBounds[i]) {
                return matchFailed(nested, rewriter, reshapeOp,
                                   "DynamicReshape output_bounds do not match permutation of input bounds");
            }
        }
    }

    const auto orderMap = mlir::AffineMap::getPermutationMap(ArrayRef<unsigned>(perm), rewriter.getContext());
    const auto orderAttr = mlir::AffineMapAttr::get(orderMap);

    nested.trace("Rewriting DynamicReshape(ShapeOf/Gather) to IE.Transpose with perm '{0}'", ArrayRef<unsigned>(perm));

    auto newTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(reshapeOp, reshapeOp.getOutput().getType(),
                                                                     reshapeOp.getInput(),
                                                                     /*order=*/nullptr, orderAttr);
    extendOpLoc(newTranspose, "as_transpose");

    return mlir::success();
}

//
// FoldShapeOfGatherDynamicReshapeToTransposePass
//

class FoldShapeOfGatherDynamicReshapeToTransposePass final :
        public IE::impl::FoldShapeOfGatherDynamicReshapeToTransposeBase<
                FoldShapeOfGatherDynamicReshapeToTransposePass> {
public:
    explicit FoldShapeOfGatherDynamicReshapeToTransposePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FoldShapeOfGatherDynamicReshapeToTransposePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FoldShapeOfGatherDynamicReshapeToTransposeRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFoldShapeOfGatherDynamicReshapeToTransposePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFoldShapeOfGatherDynamicReshapeToTransposePass(Logger log) {
    return std::make_unique<FoldShapeOfGatherDynamicReshapeToTransposePass>(log);
}
