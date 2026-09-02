//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSEMATMULTHROUGHSLICE
#define GEN_PASS_DEF_DECOMPOSEMATMULTHROUGHSLICE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Slice \p value along \p axis [offset, offset+size).
static mlir::Value sliceValue(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value value, int64_t axis,
                              int64_t offset, int64_t size) {
    const auto type = mlir::cast<NDTypeInterface>(value.getType());
    const auto shape = type.getShape();
    SmallVector<int64_t> offsets(type.getRank(), 0);
    SmallVector<int64_t> sizes(shape.begin(), shape.end());
    offsets[axis] = offset;
    sizes[axis] = size;
    return rewriter.createOrFold<IE::SliceOp>(loc, value, getIntArrayAttr(rewriter, offsets),
                                              getIntArrayAttr(rewriter, sizes));
}

// ---------------------------------------------------------------------------
// DecomposeMatMulThroughSliceVS — anchors on VariadicSplitOp, walks back the mandatory chain:
//
//   MatMul(act, W) → Add(bias) → Reshape/AffineReshape → VariadicSplit
//
// Decomposes the fused projection into N independent MatMuls, one per output.
// ---------------------------------------------------------------------------

class DecomposeMatMulThroughSliceVS final : public mlir::OpRewritePattern<IE::VariadicSplitOp> {
public:
    DecomposeMatMulThroughSliceVS(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::VariadicSplitOp>(ctx), _log(log) {
        setDebugName("DecomposeMatMulThroughSliceVS");
    }

    mlir::LogicalResult matchAndRewrite(IE::VariadicSplitOp variadicSplitOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DecomposeMatMulThroughSliceVS::matchAndRewrite(IE::VariadicSplitOp variadicSplitOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] DecomposeMatMulThroughSliceVS checking '{1}'", getDebugName(), variadicSplitOp->getLoc());

    // Only process splits on an interior axis.
    const auto inputType = mlir::cast<NDTypeInterface>(variadicSplitOp.getInput().getType());
    const auto rank = inputType.getRank();
    const auto axis = variadicSplitOp.getInferredAxis();
    if (axis <= 0 || axis >= rank - 1) {
        _log.nest().trace("VariadicSplit not on a middle axis — skip");
        return mlir::failure();
    }

    // Check for equal split lengths
    const auto splitLengths = variadicSplitOp.getInferredSplitLengths();
    if (splitLengths.empty()) {
        return mlir::failure();
    }
    const auto expectedLen = splitLengths.front();
    for (auto len : splitLengths) {
        if (len != expectedLen) {
            _log.nest().trace("VariadicSplit split_lengths are not all equal — skip");
            return mlir::failure();
        }
    }
    const int64_t nSplits = static_cast<int64_t>(splitLengths.size());

    // Walk back: VariadicSplit ← Reshape/AffineReshape ← Add(bias) ← MatMul
    IE::MatMulOp matmulOp;
    IE::AddOp addOp;
    IE::ReshapeOp reshapeOp;
    IE::AffineReshapeOp affineReshapeOp;
    mlir::Value biasValue;

    auto* inputDefOp = variadicSplitOp.getInput().getDefiningOp();

    // Reshape/AffineReshape bridges the flat 2D MatMul output to the multi-dim shape the VariadicSplit operates on.
    if (auto reshape = mlir::dyn_cast_or_null<IE::ReshapeOp>(inputDefOp)) {
        if (!reshape.getOutput().hasOneUse()) {
            return mlir::failure();
        }
        reshapeOp = reshape;
        inputDefOp = reshape.getInput().getDefiningOp();
    } else if (auto reshape = mlir::dyn_cast_or_null<IE::AffineReshapeOp>(inputDefOp)) {
        if (!reshape.getOutput().hasOneUse()) {
            return mlir::failure();
        }
        affineReshapeOp = reshape;
        inputDefOp = reshape.getInput().getDefiningOp();
    } else {
        _log.nest().trace("no Reshape/AffineReshape before VariadicSplit — skip");
        return mlir::failure();
    }

    // Bias is applied to the full fused projection in 2D space, before the per-split slicing.
    auto add = mlir::dyn_cast_or_null<IE::AddOp>(inputDefOp);
    if (!add) {
        _log.nest().trace("no Add before Reshape — skip");
        return mlir::failure();
    }
    if (!add.getOutput().hasOneUse()) {
        return mlir::failure();
    }
    if (auto matmul = mlir::dyn_cast_or_null<IE::MatMulOp>(add.getInput1().getDefiningOp())) {
        matmulOp = matmul;
        biasValue = add.getInput2();
    } else if (auto matmul = mlir::dyn_cast_or_null<IE::MatMulOp>(add.getInput2().getDefiningOp())) {
        matmulOp = matmul;
        biasValue = add.getInput1();
    } else {
        _log.nest().trace("no MatMul feeding Add — skip");
        return mlir::failure();
    }
    if (!matmulOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }
    addOp = add;

    // Determine which axis of W_qkv corresponds to the output-channel dimension.
    //   transposeB=false: W shape [dim_in, 3E] → out-ch axis = 1
    //   transposeB=true:  W shape [3E, dim_in] → out-ch axis = 0
    const bool transposeB = matmulOp.getTransposeB();
    const int64_t weightAxis = transposeB ? 0 : 1;

    const auto weightType = mlir::cast<NDTypeInterface>(matmulOp.getInput2().getType());
    if (weightType.getRank() != 2) {
        _log.nest().trace("weight rank is not 2 - skip");
        return mlir::failure();
    }
    const auto weightShape = weightType.getShape();
    const int64_t weightInCh = weightShape.raw()[1 - weightAxis];
    if (weightInCh == mlir::ShapedType::kDynamic || weightInCh <= 0) {
        _log.nest().trace("weight in-ch is dynamic/invalid - skip");
        return mlir::failure();
    }

    const int64_t totalOutCh = weightShape.raw()[weightAxis];
    if (totalOutCh == mlir::ShapedType::kDynamic || totalOutCh <= 0) {
        _log.nest().trace("weight out-ch is dynamic/invalid - skip");
        return mlir::failure();
    }
    const auto vsInputShape = inputType.getShape();

    // Dims from the split axis to the end must together span exactly totalOutCh,
    // ensuring the split is over the output-channel dimension and not a batch/sequence dim.
    int64_t axisSuffixProd = 1;
    for (int64_t d = axis; d < rank; ++d) {
        const auto dim = vsInputShape.raw()[d];
        if (dim == mlir::ShapedType::kDynamic || dim <= 0) {
            _log.nest().trace("VariadicSplit input shape has dynamic/invalid dim — skip");
            return mlir::failure();
        }
        axisSuffixProd *= dim;
    }
    if (axisSuffixProd != totalOutCh) {
        _log.nest().trace("VariadicSplit axis suffix product {0} != weight out-ch {1} — skip", axisSuffixProd,
                          totalOutCh);
        return mlir::failure();
    }

    if (totalOutCh % nSplits != 0) {
        _log.nest().trace("weight out-ch {0} not divisible by nSplits {1} — skip", totalOutCh, nSplits);
        return mlir::failure();
    }
    // sliceFeatures derived from the weight's innermost output dimension.
    const int64_t sliceFeatures = totalOutCh / nSplits;

    // Validate bias shape.
    const auto biasType = mlir::cast<NDTypeInterface>(biasValue.getType());
    if (biasType.getRank() == 0) {
        _log.nest().trace("bias rank is 0 - skip");
        return mlir::failure();
    }
    const auto biasShape = biasType.getShape();
    const int64_t biasAxis = biasType.getRank() - 1;
    for (int64_t d = 0; d < biasAxis; ++d) {
        if (biasShape.raw()[d] == mlir::ShapedType::kDynamic || biasShape.raw()[d] <= 0) {
            _log.nest().trace("bias non-sliced dim is dynamic/invalid - skip");
            return mlir::failure();
        }
    }
    const int64_t biasLastDim = biasShape.raw()[biasAxis];
    if (biasLastDim == mlir::ShapedType::kDynamic || biasLastDim != totalOutCh) {
        _log.nest().trace("bias last-dim != weight out-ch - skip");
        return mlir::failure();
    }

    _log.nest().trace("Splitting '{0}' into {1} MatMuls (weightAxis={2}, hasReshape={3})", matmulOp->getLoc(), nSplits,
                      weightAxis, static_cast<bool>(reshapeOp) || static_cast<bool>(affineReshapeOp));

    rewriter.setInsertionPoint(addOp);
    SmallVector<mlir::Value> newOutputs;
    newOutputs.reserve(nSplits);

    for (int64_t i = 0; i < nSplits; ++i) {
        auto wSlice = sliceValue(rewriter, appendLoc(matmulOp->getLoc(), "W_{0}", i), matmulOp.getInput2(), weightAxis,
                                 i * sliceFeatures, sliceFeatures);
        auto newMatMul =
                rewriter.create<IE::MatMulOp>(appendLoc(matmulOp->getLoc(), "qkv_{0}", i), matmulOp.getInput1(), wSlice,
                                              matmulOp.getTransposeA(), matmulOp.getTransposeB());
        mlir::Value out = newMatMul.getOutput();

        auto bSlice = sliceValue(rewriter, appendLoc(addOp->getLoc(), "bias_{0}", i), biasValue, biasAxis,
                                 i * sliceFeatures, sliceFeatures);
        auto newAdd = rewriter.create<IE::AddOp>(appendLoc(addOp->getLoc(), "qkv_add_{0}", i), out, bSlice,
                                                 addOp.getScale(), addOp.getAutoBroadcastAttr(), addOp.getPostOpAttr(),
                                                 addOp.getClampAttr(), addOp.getStaticScaleAttr(),
                                                 addOp.getOutputPaddingAttr(), addOp.getInputPaddingAttr());
        out = newAdd.getOutput();

        // Re-apply the reshape so each output matches the i-th result type of the original VariadicSplit.
        if (reshapeOp || affineReshapeOp) {
            const auto targetType = mlir::cast<NDTypeInterface>(variadicSplitOp.getResult(i).getType());
            const auto currentType = mlir::cast<NDTypeInterface>(out.getType());
            if (currentType.getShape() != targetType.getShape()) {
                mlir::Location reshapeLoc = reshapeOp ? reshapeOp->getLoc() : affineReshapeOp->getLoc();
                auto shapeAttr = getIntArrayAttr(rewriter, targetType.getShape().raw());
                if (affineReshapeOp) {
                    out = rewriter.create<IE::AffineReshapeOp>(appendLoc(reshapeLoc, "qkv_rs_{0}", i), out,
                                                               affineReshapeOp.getDimMapping(), shapeAttr)
                                  .getOutput();
                } else {
                    out = rewriter.create<IE::ReshapeOp>(appendLoc(reshapeLoc, "qkv_rs_{0}", i), out, shapeAttr)
                                  .getOutput();
                }
            }
        }

        newOutputs.push_back(out);
    }

    // Replace the VariadicSplit with the N separate outputs, then eagerly
    // erase the now-dead reshape/add/matmul so the greedy driver skips them.
    rewriter.replaceOp(variadicSplitOp, newOutputs);
    if (reshapeOp) {
        rewriter.eraseOp(reshapeOp);
    }
    if (affineReshapeOp) {
        rewriter.eraseOp(affineReshapeOp);
    }
    rewriter.eraseOp(addOp);
    rewriter.eraseOp(matmulOp);
    return mlir::success();
}

// ---------------------------------------------------------------------------
// DecomposeMatMulThroughSlicePass
// ---------------------------------------------------------------------------

class DecomposeMatMulThroughSlicePass final :
        public IE::impl::DecomposeMatMulThroughSliceBase<DecomposeMatMulThroughSlicePass> {
public:
    explicit DecomposeMatMulThroughSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DecomposeMatMulThroughSlicePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DecomposeMatMulThroughSliceVS>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeMatMulThroughSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeMatMulThroughSlicePass(Logger log) {
    return std::make_unique<DecomposeMatMulThroughSlicePass>(log);
}
