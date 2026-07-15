//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/check_shrink_matmul_groups.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_ELIMINATESLICEINSOFTMAXMATMUL
#define GEN_PASS_DEF_ELIMINATESLICEINSOFTMAXMATMUL
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// EliminateSliceInSoftmaxMatMulRewriter
//
// Pattern: Softmax -> Slice -> MatMul
// Optimization:
//   1. Pad Softmax input to alignment using -inf padding
//   2. Apply Softmax on aligned input (exp(-inf) = 0 for padded positions)
//   3. Remove Slice operation
//   4. Pad MatMul RHS with zeros to match aligned dimension
//   5. Compute MatMul with aligned dimensions
//
// Example (user's case):
//   Before:
//     Input(1x64x1024x1025) -> Softmax -> (1x64x1024x1025) -> Slice[0:1024] -> (1x64x1024x1024)
//                                                                                  |
//     RHS(1x64x64x1024) --------------------------------------------------------> MatMul -> (1x64x1024x64)
//
//   After:
//     Input(1x64x1024x1025) -> Concat[-inf x31] -> (1x64x1024x1056) -> Softmax -> (1x64x1024x1056)
//                                                                                       |
//     RHS(1x64x64x1024) -> Concat[0 x32] -> (1x64x64x1056) -----------------------> MatMul -> (1x64x1024x64)
//
// Mathematical equivalence:
//   - Softmax([x, -inf]): exp(-inf) = 0, so softmax([x, -inf]) = [softmax(x), 0]
//   - MatMul([a, 0] @ [b, 0]^T) = a @ b^T (zero padding contributes 0)
//   - Therefore: (Softmax(x)[0:n] @ W) == (Softmax([x, -inf]) @ [W, 0])
//
// Benefits:
//   - Removes Slice operation overhead
//   - Enables better memory alignment for SW layers
//   - Maintains mathematical correctness
//

class EliminateSliceInSoftmaxMatMulRewriter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    EliminateSliceInSoftmaxMatMulRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx), _log(log) {
        this->setDebugName("EliminateSliceInSoftmaxMatMulRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EliminateSliceInSoftmaxMatMulRewriter::matchAndRewrite(IE::MatMulOp matmulOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MatMul at '{1}'", this->getDebugName(), matmulOp->getLoc());

    // Pattern matching: MatMul LHS = Slice -> Softmax
    if (IE::shouldShrinkMatmulGroups(matmulOp)) {
        _log.nest().trace("Skip: MatMul has groups that would be shrunk, which may interfere with this optimization");
        return mlir::failure();
    }

    auto lhsSliceOp = matmulOp.getInput1().getDefiningOp<IE::SliceOp>();
    if (lhsSliceOp == nullptr) {
        return mlir::failure();
    }

    auto softmaxOp = lhsSliceOp.getSource().getDefiningOp<IE::SoftMaxOp>();
    if (softmaxOp == nullptr) {
        return mlir::failure();
    }

    _log.nest().trace("Found pattern: Softmax -> Slice -> MatMul");

    // Validate: Only process float types (need -infinity for padding)
    const auto softmaxInputType = mlir::cast<vpux::NDTypeInterface>(softmaxOp.getInput().getType());
    const auto elemType = softmaxInputType.getElementType();
    if (!mlir::isa<mlir::FloatType>(elemType)) {
        _log.nest().trace("Skip: Non-float element type");
        return mlir::failure();
    }

    // Get softmax parameters
    const auto softmaxInputShapeVec = to_small_vector(softmaxInputType.getShape());
    const auto rank = softmaxInputShapeVec.size();

    // Normalize negative axis
    auto axis = softmaxOp.getAxisInd();
    if (axis < 0) {
        axis += rank;
    }

    // Only process Softmax on innermost dimension in memory layout
    const auto innermostDim = softmaxInputType.getDimsOrder().dimAt(rank - 1);
    if (Dim(axis) != innermostDim) {
        _log.nest().trace("Skip: Softmax axis {0} is not innermost dimension {1}", axis, innermostDim);
        return mlir::failure();
    }

    // Validate Slice operation
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(lhsSliceOp.getStaticOffsets());
    const auto sliceSizes = parseIntArrayAttr<int64_t>(lhsSliceOp.getStaticSizesAttr());

    // Slice must start from offset 0 on the softmax axis
    if (sliceOffsets[axis] != 0) {
        _log.nest().trace("Skip: Slice offset {0} != 0 on axis {1}", sliceOffsets[axis], axis);
        return mlir::failure();
    }

    // Calculate dimensions
    const int64_t softmaxOutputDim = softmaxInputShapeVec[innermostDim.ind()];
    const int64_t slicedDim = sliceSizes[axis];

    // Only optimize if Slice actually removes elements
    if (softmaxOutputDim <= slicedDim) {
        _log.nest().trace("Skip: No slicing detected ({0} <= {1})", softmaxOutputDim, slicedDim);
        return mlir::failure();
    }

    // Calculate aligned dimension based on element type
    const auto alignment = VPU::NCEInvariant::getAlignment(elemType);

    // Skip optimization if Softmax dimension is already aligned
    if (softmaxOutputDim % alignment == 0) {
        _log.nest().trace("Skip: Softmax dimension {0} is already aligned to {1}", softmaxOutputDim, alignment);
        return mlir::failure();
    }

    const int64_t alignedDim = alignValUp(softmaxOutputDim, alignment);
    const int64_t softmaxPadSize = alignedDim - softmaxOutputDim;
    const int64_t rhsPadSize = alignedDim - slicedDim;

    _log.nest().trace("Applying optimization: softmax_dim={0}, sliced_dim={1}, aligned_dim={2}, "
                      "softmax_pad={3}, rhs_pad={4}, alignment={5}",
                      softmaxOutputDim, slicedDim, alignedDim, softmaxPadSize, rhsPadSize, alignment);

    const auto ctx = rewriter.getContext();
    const auto loc = matmulOp->getLoc();

    // Step 1: Pad Softmax input with -inf if needed
    mlir::Value alignedSoftmaxInput = softmaxOp.getInput();

    if (softmaxPadSize > 0) {
        auto padShape = std::move(softmaxInputShapeVec);
        padShape[innermostDim.ind()] = softmaxPadSize;

        const Shape padShapeObj(padShape);
        auto padType = softmaxInputType.changeShape(padShapeObj);
        auto padTensorType = mlir::cast<mlir::RankedTensorType>(padType);

        // Create -infinity constant (exp(-inf) = 0 in Softmax)
        auto negInfPadding = Const::createDenseConst(rewriter, appendLoc(softmaxOp.getLoc(), "neg_inf_padding"),
                                                     padTensorType, -std::numeric_limits<float>::infinity());

        // Concatenate original input with -inf padding
        auto concatOp = rewriter.create<IE::ConcatOp>(appendLoc(softmaxOp.getLoc(), "pad_softmax_input"),
                                                      mlir::ValueRange{softmaxOp.getInput(), negInfPadding},
                                                      getIntAttr(ctx, axis));

        alignedSoftmaxInput = concatOp.getOutput();
    }

    // Step 2: Create new Softmax with aligned input
    auto newSoftmax = rewriter.create<IE::SoftMaxOp>(appendLoc(softmaxOp.getLoc(), "aligned"), alignedSoftmaxInput,
                                                     softmaxOp.getAxisIndAttr(), softmaxOp.getPadSizeAttr(),
                                                     softmaxOp.getDstElemTypeAttr(), softmaxOp.getMaskAwareAttr());

    // Step 3: Pad MatMul RHS with zeros
    const auto rhsInput = matmulOp.getInput2();
    const auto rhsType = mlir::cast<vpux::NDTypeInterface>(rhsInput.getType());
    const auto rhsShape = to_small_vector(rhsType.getShape());
    const auto rhsRank = rhsShape.size();
    const auto rhsInnermostDim = rhsType.getDimsOrder().dimAt(rhsRank - 1);

    auto rhsPadShape = std::move(rhsShape);
    rhsPadShape[rhsInnermostDim.ind()] = rhsPadSize;

    const Shape rhsPadShapeObj(rhsPadShape);
    auto rhsPadType = rhsType.changeShape(rhsPadShapeObj);
    auto rhsPadTensorType = mlir::cast<mlir::RankedTensorType>(rhsPadType);

    // Create zero constant for RHS padding
    auto zeroPadding = Const::createDenseConst(rewriter, appendLoc(loc, "zero_padding"), rhsPadTensorType, 0.0f);

    // Concatenate RHS with zero padding
    auto rhsPaddedOp = rewriter.create<IE::ConcatOp>(appendLoc(loc, "pad_rhs"), mlir::ValueRange{rhsInput, zeroPadding},
                                                     getIntAttr(ctx, rhsInnermostDim.ind()));

    // Step 4: Create new MatMul with aligned inputs (Slice is eliminated)
    auto newMatMul =
            rewriter.create<IE::MatMulOp>(takeOpLoc(matmulOp, "optimized"), newSoftmax.getOutput(),
                                          rhsPaddedOp.getOutput(), matmulOp.getTransposeA(), matmulOp.getTransposeB());

    _log.nest().trace("Successfully optimized Softmax-Slice-MatMul pattern");

    // Replace original MatMul
    rewriter.replaceOp(matmulOp, newMatMul.getOutput());

    return mlir::success();
}

//
// EliminateSliceInSoftmaxMatMulPass
//

class EliminateSliceInSoftmaxMatMulPass final :
        public IE::impl::EliminateSliceInSoftmaxMatMulBase<EliminateSliceInSoftmaxMatMulPass> {
public:
    explicit EliminateSliceInSoftmaxMatMulPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void EliminateSliceInSoftmaxMatMulPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<EliminateSliceInSoftmaxMatMulRewriter>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createEliminateSliceInSoftmaxMatMulPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createEliminateSliceInSoftmaxMatMulPass(Logger log) {
    return std::make_unique<EliminateSliceInSoftmaxMatMulPass>(log);
}
