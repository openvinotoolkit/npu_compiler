//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZEINNERMOSTCONCATPASS
#define GEN_PASS_DEF_OPTIMIZEINNERMOSTCONCATPASS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// InnermostConcatOptimizer
//

class InnermostConcatOptimizer final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    InnermostConcatOptimizer(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(std::move(log)) {
        this->setDebugName("InnermostConcatOptimizer");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;
    bool canOptimizeConcat(IE::ConcatOp concatOp) const;

private:
    Logger _log;
};

bool InnermostConcatOptimizer::canOptimizeConcat(IE::ConcatOp concatOp) const {
    const auto inputs = concatOp.getInputs();
    if (inputs.size() < 2) {
        return false;
    }

    if (llvm::any_of(inputs, [](auto input) {
            return getShape(input).isDynamic();
        })) {
        return false;
    }

    // Get concat axis using the utility function
    auto concatAxis = IE::getConcatAxis(concatOp);
    if (!concatAxis.has_value()) {
        return false;
    }

    const auto firstInputType = mlir::cast<vpux::NDTypeInterface>(inputs[0].getType());
    const auto firstInputShape = firstInputType.getShape();
    const auto dimsOrder = firstInputType.getDimsOrder();

    // Only handle 4D tensors for now
    if (firstInputShape.size() != 4) {
        return false;
    }

    // Get the innermost dimension according to the layout order
    auto innermostDim = dimsOrder.dimAt(dimsOrder.numDims() - 1);

    // Check if concat axis is the innermost dimension
    if (concatAxis.value() != innermostDim) {
        return false;
    }

    // Check if the pattern matches: innermost dimension is 1
    if (firstInputShape[innermostDim] != 1) {
        return false;
    }

    // Check if all inputs have the same shape
    for (const auto& input : inputs) {
        if (getShape(input) != firstInputShape) {
            return false;
        }
    }

    // Check output shape: if all non-concat dimensions have size 1, skip optimization
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(concatOp.getType());
    const auto outputShape = outputType.getShape();

    size_t nonUnitNonConcatDimCount = 0;
    for (size_t i = 0; i < dimsOrder.numDims(); ++i) {
        auto dim = dimsOrder.dimAt(i);
        if (dim != concatAxis.value() && outputShape[dim] != 1) {
            nonUnitNonConcatDimCount++;
        }
    }

    // If all non-concat dimensions are unit size, skip optimization
    return nonUnitNonConcatDimCount > 0;
}

mlir::LogicalResult InnermostConcatOptimizer::matchAndRewrite(IE::ConcatOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Check if this is an innermost dimension concat that can be optimized
    if (!canOptimizeConcat(origOp)) {
        return matchFailed(_log.nest(), rewriter, origOp, "Not an optimizable innermost dimension concat");
    }

    const auto inputs = origOp.getInputs();
    const auto inputShape = getShape(inputs[0]);
    const auto inputMemShape = getMemShape(inputs[0]);
    const auto numInputs = inputs.size();

    // Get original layout information
    const auto origInputType = mlir::cast<vpux::NDTypeInterface>(inputs[0].getType());
    const auto origOutputType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
    const auto origInputOrder = origInputType.getDimsOrder();
    const auto origOutputOrder = origOutputType.getDimsOrder();

    _log.trace("Optimizing concat from shape {0} with layout {1}", inputShape, origInputOrder);

    auto ctx = rewriter.getContext();

    // Step 1: Convert all inputs to NCHW layout using PermuteCast
    SmallVector<mlir::Value> permuteCastInputs;
    permuteCastInputs.reserve(numInputs);

    for (size_t i = 0; i < numInputs; ++i) {
        // Swap MemDim(3) to MemDim(1)
        auto perm = DimsOrder::NWCH.toAffineMap(ctx);
        auto inputPermuteCast = rewriter.create<IE::PermuteCastOp>(
                appendLoc(origOp.getLoc(), llvm::formatv("_input{0}_swap_inner_dim", i)), inputs[i],
                mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx)), mlir::AffineMapAttr::get(perm));

        permuteCastInputs.push_back(inputPermuteCast.getOutput());
    }

    // Step 2: Create new efficient Concat on axis=1 (Channel dimension)
    // Calculate output shape based on the first reshaped input
    const auto firstReshapedShape = getShape(permuteCastInputs[0]);
    SmallVector<SmallVector<int64_t>> newOffsets;
    newOffsets.reserve(numInputs);
    for (size_t i = 0; i < numInputs; ++i) {
        newOffsets.push_back({0, static_cast<int64_t>(i), 0, 0});
    }

    // Output shape: [dim0, numInputs, dim2, dim3] where dim0, dim2, dim3 come from reshaped inputs
    SmallVector<int64_t> concatOutputShape = {firstReshapedShape[Dim(0)], static_cast<int64_t>(numInputs),
                                              firstReshapedShape[Dim(2)], firstReshapedShape[Dim(3)]};
    auto concatOutputType =
            mlir::cast<vpux::NDTypeInterface>(permuteCastInputs[0].getType()).changeShape(ShapeRef(concatOutputShape));
    auto newConcatOp = rewriter.create<IE::ConcatOp>(appendLoc(origOp.getLoc(), "_optimized_concat"), concatOutputType,
                                                     permuteCastInputs,
                                                     nullptr,  // per_axis
                                                     getIntArrayOfArray(ctx, newOffsets));

    // Step 3: Apply MemPermute to move channel dimension to last position
    // Permutation: [0, 2, 3, 1] to get [dim0, dim2, dim3, numInputs]
    SmallVector<uint32_t> permutationIndices = {0, 2, 3, 1};
    auto permutationMap = mlir::AffineMap::getPermutationMap(permutationIndices, ctx);

    SmallVector<int64_t> memPermuteOutputShape = {firstReshapedShape[Dim(0)], firstReshapedShape[Dim(2)],
                                                  firstReshapedShape[Dim(3)], static_cast<int64_t>(numInputs)};
    auto memPermuteOutputShapeMemOrder = DimsOrder::NCHW.toMemoryOrder(ShapeRef(memPermuteOutputShape));
    auto memPermuteOutputShapeLogicalOrder = origOutputOrder.toLogicalOrder(memPermuteOutputShapeMemOrder);
    auto memPermuteOutputType =
            concatOutputType.changeShape(memPermuteOutputShapeLogicalOrder).changeDimsOrder(origOutputOrder);

    auto memPermuteOp = rewriter.create<IE::MemPermuteOp>(
            appendLoc(origOp.getLoc(), "_mem_permute"), memPermuteOutputType, newConcatOp.getOutput(),
            mlir::AffineMapAttr::get(origOutputOrder.toAffineMap(ctx)), mlir::AffineMapAttr::get(permutationMap));

    _log.trace("Successfully optimized innermost concat at {0}", origOp.getLoc());

    rewriter.replaceOp(origOp, memPermuteOp.getOutput());

    return mlir::success();
}

//
// OptimizeInnermostConcatPass
//

class OptimizeInnermostConcatPass : public IE::impl::OptimizeInnermostConcatPassBase<OptimizeInnermostConcatPass> {
public:
    explicit OptimizeInnermostConcatPass(Logger log): _log(std::move(log)) {
        _log.setName("optimize-innermost-concat");
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void OptimizeInnermostConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<InnermostConcatOptimizer>(&ctx, _log);

    IE::MemPermuteOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeInnermostConcatPass(Logger log) {
    return std::make_unique<OptimizeInnermostConcatPass>(std::move(log));
}
