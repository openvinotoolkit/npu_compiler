//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::MemPermuteOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MemPermuteOpAdaptor mem_permute(operands, attrs, prop);
    if (mlir::failed(mem_permute.verify(loc))) {
        return mlir::failure();
    }

    inferPermuteReturnTypeComponents(mem_permute.getInput(), mem_permute.getMemPerm(), mem_permute.getDstOrder(),
                                     inferredReturnShapes, false);

    return mlir::success();
}

namespace {

//
// FuseMemPermutes
//

class FuseMemPermutes final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    FuseMemPermutes(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit) {
        this->setDebugName("FuseMemPermutes");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseMemPermutes::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                     mlir::PatternRewriter& rewriter) const {
    return fusePermutations<IE::MemPermuteOp, IE::MemPermuteOp>(memPermuteOp, rewriter);
}

//
// FusePermCastAndMemPerm
//

// PermuteCast -> MemPermute ===> MemPermute

class FusePermCastAndMemPerm final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    FusePermCastAndMemPerm(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit) {
        this->setDebugName("FusePermCastAndMemPerm");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FusePermCastAndMemPerm::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                            mlir::PatternRewriter& rewriter) const {
    return fusePermutations<IE::PermuteCastOp, IE::MemPermuteOp>(memPermuteOp, rewriter);
}

//
// FuseMemPermuteThroughConcat
//

class FuseMemPermuteThroughConcat final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    FuseMemPermuteThroughConcat(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit) {
        this->setDebugName("FuseMemPermuteThroughConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseMemPermuteThroughConcat::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    auto concatOp = memPermuteOp.getInput().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr || !concatOp->hasOneUse()) {
        return mlir::failure();
    }

    auto collectMemPermuteInputs = [&](IE::ConcatOp concat) -> std::optional<SmallVector<IE::MemPermuteOp>> {
        SmallVector<IE::MemPermuteOp> memPermutes;
        memPermutes.reserve(concat.getInputs().size());

        for (const auto input : concat.getInputs()) {
            auto memPermute = input.getDefiningOp<IE::MemPermuteOp>();
            if (memPermute == nullptr) {
                return std::nullopt;
            }
            memPermutes.push_back(memPermute);
        }
        return memPermutes;
    };

    auto maybeMemPermutes = collectMemPermuteInputs(concatOp);
    if (!maybeMemPermutes.has_value()) {
        return mlir::failure();
    }

    auto inputMemPermutes = maybeMemPermutes.value();
    auto referenceMemPermute = inputMemPermutes.front();

    auto validateMemPermuteConsistency = [&](const SmallVector<IE::MemPermuteOp>& memPermutes) -> bool {
        const auto refInputOrder = DimsOrder::fromValue(referenceMemPermute.getInput());
        const auto refDstOrder = referenceMemPermute.getDstOrder();
        const auto refMemPerm = referenceMemPermute.getMemPerm();

        return llvm::all_of(memPermutes, [&](IE::MemPermuteOp memPermute) {
            return DimsOrder::fromValue(memPermute.getInput()) == refInputOrder &&
                   memPermute.getDstOrder() == refDstOrder && memPermute.getMemPerm() == refMemPerm;
        });
    };

    if (!validateMemPermuteConsistency(inputMemPermutes)) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> newConcatInputs;
    newConcatInputs.reserve(inputMemPermutes.size());
    llvm::transform(inputMemPermutes, std::back_inserter(newConcatInputs), [](IE::MemPermuteOp memPermute) {
        return memPermute.getInput();
    });

    const auto inMemShape = mlir::cast<vpux::NDTypeInterface>(referenceMemPermute.getOutput().getType()).getMemShape();
    const auto outMemShape = mlir::cast<vpux::NDTypeInterface>(concatOp.getOutput().getType()).getMemShape();

    const auto refPerm = referenceMemPermute.getMemPerm();
    const auto permuteInputOrder = DimsOrder::fromValue(referenceMemPermute.getInput());

    // Find the axis for concatenation after permutation
    int32_t newConcatAxis = -1;
    for (size_t idx = 0; idx < inMemShape.size(); ++idx) {
        if (inMemShape.raw()[idx] == outMemShape.raw()[idx]) {
            continue;
        }

        if (newConcatAxis != -1) {
            return mlir::failure();
        }

        newConcatAxis = refPerm.getDimPosition(static_cast<uint32_t>(idx));
        newConcatAxis = permuteInputOrder.dimAt(static_cast<size_t>(newConcatAxis)).ind();
    }

    auto newConcat = rewriter.replaceOpWithNewOp<IE::ConcatOp>(concatOp, newConcatInputs, newConcatAxis);

    const auto composedMemPerm = memPermuteOp.getMemPerm().compose(referenceMemPermute.getMemPerm());
    rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(memPermuteOp, memPermuteOp.getType(), newConcat,
                                                  memPermuteOp.getDstOrderAttr(),
                                                  mlir::AffineMapAttr::get(composedMemPerm));

    return mlir::success();
}

//
// FuseMemPermuteThroughExpand
//

/*  If we meet this pattern

    MemPermute()
        |
    Expand()
        |
    MemPermute()

    We can fuse the two MemPermute if it can convert to trivial permute
*/

mlir::ArrayAttr getNewPaddingAttr(mlir::MLIRContext* ctx, ArrayRef<int64_t> pads, const vpux::DimsOrder& targetOrder,
                                  const vpux::DimsOrder& outOrder) {
    SmallVector<int64_t> newPads(pads.size(), 0);

    for (auto ind : irange(pads.size())) {
        if (pads[ind] != 0) {
            auto dimPos = outOrder.dimPos(Dim(ind));
            auto dim = targetOrder.dimAt(dimPos);
            newPads[dim.ind()] = pads[ind];
        }
    }

    return getIntArrayAttr(ctx, std::move(newPads));
}

class FuseMemPermuteThroughExpand final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    FuseMemPermuteThroughExpand(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit) {
        this->setDebugName("FuseMemPermuteThroughExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseMemPermuteThroughExpand::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    auto expandOp = memPermuteOp.getInput().getDefiningOp<IE::ExpandOp>();
    if (expandOp == nullptr || !expandOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto topMemPermuteOp = expandOp.getInput().getDefiningOp<IE::MemPermuteOp>();
    if (topMemPermuteOp == nullptr || !topMemPermuteOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto topInOrder = DimsOrder::fromValue(topMemPermuteOp.getInput());
    const auto padsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd());

    // Get the real expanding axis
    const auto memPerm = DimsOrder::fromAffineMap(topMemPermuteOp.getMemPerm());
    const auto targetOrder = vpux::applyPermutation(topInOrder, memPerm);
    const auto topMemPermuteOutOrder = DimsOrder::fromValue(topMemPermuteOp.getOutput());

    const auto newPadsBeginAttr = getNewPaddingAttr(getContext(), padsBegin, targetOrder, topMemPermuteOutOrder);
    const auto newPadsEndAttr = getNewPaddingAttr(getContext(), padsEnd, targetOrder, topMemPermuteOutOrder);

    auto outputType = mlir::cast<vpux::NDTypeInterface>(memPermuteOp.getOutput().getType());
    auto outputOrder = outputType.getDimsOrder();

    auto newExpandOp = rewriter.create<IE::ExpandOp>(expandOp.getLoc(), topMemPermuteOp.getInput(), newPadsBeginAttr,
                                                     newPadsEndAttr);
    const auto permuteCastInOrder = DimsOrder::fromValue(newExpandOp.getOutput());
    const auto permuteCastInShape = getShape(newExpandOp.getOutput());
    const auto permuteCastInMemShape = permuteCastInOrder.toMemoryOrder(permuteCastInShape);
    auto newMemPerm = memPermuteOp.getMemPerm().compose(topMemPermuteOp.getMemPerm());
    if (!isTrivialPermute(permuteCastInMemShape, newMemPerm)) {
        rewriter.eraseOp(newExpandOp);
        return mlir::failure();
    }

    auto newPermuteCastOp = rewriter.create<IE::PermuteCastOp>(
            memPermuteOp.getLoc(), memPermuteOp.getOutput().getType(), newExpandOp.getOutput(),
            mlir::AffineMapAttr::get(outputOrder.toAffineMap(getContext())), mlir::AffineMapAttr::get(newMemPerm));

    memPermuteOp.replaceAllUsesWith(newPermuteCastOp.getOutput());

    return mlir::success();
}

//
// FuseMemPermuteAndPermuteQuantize
//

class FuseMemPermuteAndPermuteQuantize final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    FuseMemPermuteAndPermuteQuantize(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit) {
        this->setDebugName("FuseMemPermuteAndPermuteQuantize");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseMemPermuteAndPermuteQuantize::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto permuteQuantizeOp = memPermuteOp.getInput().getDefiningOp<IE::PermuteQuantizeOp>();
    if (permuteQuantizeOp == nullptr) {
        return mlir::failure();
    }

    // Could not fuse permuteQuantize with pad to memPermute because memPermute do not support pad,
    // Missing the parameter will cause shape difference between infer and expect
    // TODO: if cannot convert to memPermute, consider convert to permuteQuantize.
    auto padsBegin = parseIntArrayAttr<int64_t>(permuteQuantizeOp.getPadsBegin());
    auto padsEnd = parseIntArrayAttr<int64_t>(permuteQuantizeOp.getPadsEnd());

    const auto notZero = [](auto pad) {
        return pad != 0;
    };
    if (llvm::any_of(padsBegin, notZero) || llvm::any_of(padsEnd, notZero)) {
        return mlir::failure();
    }

    // Can fuse MemPermute with PermuteQuantization in case only permutation (no quantization) is performed by this
    // PermuteQuantization Op.
    const auto permuteQuantizeOutElemType =
            mlir::cast<vpux::NDTypeInterface>(permuteQuantizeOp.getOutput().getType()).getElementType();
    const auto permuteQuantizeInElemType =
            mlir::cast<vpux::NDTypeInterface>(permuteQuantizeOp.getInput().getType()).getElementType();
    if (!(IE::isPurePermuteCompatiblePrecision(permuteQuantizeInElemType, permuteQuantizeOutElemType))) {
        return mlir::failure();
    }

    return fusePermutations<IE::PermuteQuantizeOp, IE::MemPermuteOp>(memPermuteOp, rewriter);
}

//
// ConvertToPermuteCast
//

class ConvertToPermuteCast final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    ConvertToPermuteCast(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit) {
        this->setDebugName("ConvertToPermuteCast");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToPermuteCast::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                          mlir::PatternRewriter& rewriter) const {
    if (!isTrivialMemPermute(memPermuteOp)) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::PermuteCastOp>(memPermuteOp, memPermuteOp.getInput(),
                                                   memPermuteOp.getDstOrderAttr(), memPermuteOp.getMemPermAttr());
    return mlir::success();
}

//
// ConvertShapeCastToPermuteCast
//

// MemPermute -> ShapeCast ===> MemPermute

class ConvertShapeCastToPermuteCast final : public mlir::OpRewritePattern<IE::ShapeCastOp> {
public:
    ConvertShapeCastToPermuteCast(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::ShapeCastOp>(ctx, benefit) {
        this->setDebugName("ConvertShapeCastToPermuteCast");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertShapeCastToPermuteCast::matchAndRewrite(IE::ShapeCastOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    // Check if input is MemPermute
    auto memPermuteOp = origOp.getOperand().getDefiningOp<IE::MemPermuteOp>();

    // If input is not MemPermute, only proceed if this is part of ShapeCast -> PermuteCast -> Reorder pattern
    if (memPermuteOp == nullptr) {
        if (!origOp->hasOneUse()) {
            return mlir::failure();
        }

        auto permuteCastUser = mlir::dyn_cast<IE::PermuteCastOp>(*origOp->getUsers().begin());
        if (permuteCastUser == nullptr || !permuteCastUser->hasOneUse()) {
            return mlir::failure();
        }

        auto reorderUser = mlir::dyn_cast<IE::ReorderOp>(*permuteCastUser->getUsers().begin());
        if (reorderUser == nullptr) {
            return mlir::failure();
        }
    }

    const auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getResult().getType());
    const auto outOrder = outType.getDimsOrder();
    auto hasValidPermuteCast = IE::tryToFindPermuteCastOp(origOp.getLoc(), origOp.getOperand(), outOrder,
                                                          getShape(origOp.getResult()), rewriter);

    if (!hasValidPermuteCast.has_value()) {
        return mlir::failure();
    }

    rewriter.replaceOp(origOp, hasValidPermuteCast.value().getResult());
    return mlir::success();
}

//
// EliminateMemPermuteThroughReshapeSlice
//
// Eliminates IE.MemPermute followed by IE.AffineReshape when all AffineReshape users
// are IE.Slice ops that cut a single dimension that passes through the reshape unchanged.
//
// Pattern:
//   MemPermute(src_order=NCHW, dst_order=NCHW, mem_perm=P) → AffineReshape → Slice(dim=D, size=1) × N
//
// Note: "src_order=NCHW, dst_order=NCHW" describes the logical layout of the input and output
// tensors, not the mem_perm attribute. mem_perm P is deliberately non-trivial (e.g. the swap
// (d0,d1,d2,d3)→(d0,d2,d1,d3) in the canonical test case).
//
// Replaced by:
//   Slice(origSliceDim, size=1) → AffineReshape  (per original Slice)
//
// Matching conditions:
//   1. MemPermute has NCHW input logical order and dst_order=#NCHW (mem_perm P may be non-trivial).
//   2. MemPermute has a single user: AffineReshape.
//   3. All AffineReshape users are Slice ops.
//   4. All Slices cut the same single dim D with size=1.
//   5. Exactly one non-trivial (size>1) AffineReshape input dim maps to output dim D,
//      meaning D is not split or merged with other non-trivial dims by the reshape.

// Validate that newDimMapping is monotonically non-decreasing, covers all output dims, and
// that the total element count is consistent.
bool isValidReshapeMapping(ArrayRef<SmallVector<int64_t>> newDimMapping, ArrayRef<int64_t> newInputShape,
                           ShapeRef sliceOutShape) {
    int64_t prevMax = -1;
    for (size_t i = 0; i < newDimMapping.size(); ++i) {
        if (newDimMapping[i].empty()) {
            return false;
        }
        int64_t localPrev = prevMax;
        for (auto outIdx : newDimMapping[i]) {
            if (outIdx < localPrev) {
                return false;
            }
            localPrev = outIdx;
        }
        prevMax = localPrev;
    }

    int64_t totalIn = 1;
    for (auto s : newInputShape) {
        totalIn *= s;
    }
    int64_t totalOut = 1;
    for (size_t d = 0; d < sliceOutShape.size(); ++d) {
        totalOut *= sliceOutShape[Dim(d)];
    }
    if (totalIn != totalOut) {
        return false;
    }

    SmallVector<bool> outputCovered(sliceOutShape.size(), false);
    for (size_t i = 0; i < newDimMapping.size(); ++i) {
        for (auto outIdx : newDimMapping[i]) {
            if (outIdx < 0 || outIdx >= static_cast<int64_t>(sliceOutShape.size())) {
                return false;
            }
            outputCovered[outIdx] = true;
        }
    }
    for (size_t d = 0; d < outputCovered.size(); ++d) {
        if (!outputCovered[d]) {
            return false;
        }
    }
    return true;
}

class EliminateMemPermuteThroughReshapeSlice final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    using mlir::OpRewritePattern<IE::MemPermuteOp>::OpRewritePattern;

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult EliminateMemPermuteThroughReshapeSlice::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    // Step 1: Require NCHW→NCHW so that logical dims and memory dims coincide,
    // making the permutation vector directly usable as logical dim indices.
    const auto inType = mlir::cast<vpux::NDTypeInterface>(memPermuteOp.getInput().getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(memPermuteOp.getOutput().getType());
    if (inType.getRank() != 4 || outType.getRank() != 4) {
        return matchFailed(rewriter, memPermuteOp, "MemPermute input/output rank is not 4");
    }
    if (inType.getDimsOrder() != DimsOrder::NCHW || outType.getDimsOrder() != DimsOrder::NCHW) {
        return matchFailed(rewriter, memPermuteOp, "MemPermute input/output order is not NCHW");
    }

    // permVec[outPos] = inputDim: maps each MemPermute output dim to its source input dim
    // (this is the inverse/pull permutation: output[outPos] = input[permVec[outPos]])
    const auto memPerm = DimsOrder::fromAffineMap(memPermuteOp.getMemPerm());
    const auto permVec = to_small_vector(memPerm.toPermutation() | transformed([](Dim dim) {
                                             return checked_cast<int64_t>(dim.ind());
                                         }));

    // Step 2: MemPermute must have exactly one user: AffineReshape
    if (!memPermuteOp->hasOneUse()) {
        return matchFailed(rewriter, memPermuteOp, "MemPermute has multiple users");
    }
    auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(*memPermuteOp->getUsers().begin());
    if (affineReshapeOp == nullptr) {
        return matchFailed(rewriter, memPermuteOp, "MemPermute user is not AffineReshape");
    }

    // Steps 3 & 4: Collect Slice users and validate each cuts the same single dim D with size=1.
    const auto reshapeOutType = mlir::cast<vpux::NDTypeInterface>(affineReshapeOp.getOutput().getType());
    const auto reshapeOutShape = reshapeOutType.getShape();

    SmallVector<IE::SliceOp> sliceOps;
    int64_t sliceDim = -1;
    for (auto* user : affineReshapeOp->getUsers()) {
        auto sliceOp = mlir::dyn_cast<IE::SliceOp>(user);
        if (sliceOp == nullptr) {
            return matchFailed(rewriter, memPermuteOp, "AffineReshape has non-Slice user");
        }
        const auto staticOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr());
        const auto staticSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizesAttr());

        int64_t cutDim = -1;
        for (int64_t d = 0; d < static_cast<int64_t>(staticSizes.size()); ++d) {
            if (staticSizes[d] != reshapeOutShape[Dim(d)]) {
                if (cutDim != -1) {
                    return matchFailed(rewriter, memPermuteOp, "Slice cuts more than one dim");
                }
                cutDim = d;
            }
        }
        if (cutDim == -1) {
            return matchFailed(rewriter, memPermuteOp, "Slice does not cut any dim");
        }
        if (staticSizes[cutDim] != 1) {
            return matchFailed(rewriter, memPermuteOp, "Slice size on cutting dim is not 1");
        }
        for (int64_t d = 0; d < static_cast<int64_t>(staticOffsets.size()); ++d) {
            if (d != cutDim && staticOffsets[d] != 0) {
                return matchFailed(rewriter, memPermuteOp, "Slice has non-zero offset on non-cutting dim {0}", d);
            }
        }
        if (sliceDim == -1) {
            sliceDim = cutDim;
        } else if (sliceDim != cutDim) {
            return matchFailed(rewriter, memPermuteOp, "Slices cut different dims");
        }
        sliceOps.push_back(sliceOp);
    }
    if (sliceOps.empty()) {
        return matchFailed(rewriter, memPermuteOp, "AffineReshape has no users");
    }
    if (sliceDim < 0) {
        return matchFailed(rewriter, memPermuteOp, "Could not determine slice dim");
    }

    // Step 5: Slice dim D passes through the AffineReshape unchanged:
    // exactly one non-trivial (size>1) AffineReshape input dim maps to output dim D,
    // and that input dim maps exclusively to D (not split across other non-trivial output dims).
    // If the input dim were split, slicing size=1 on D would correspond to a slice of size>1
    // on the original input dim, making the rewrite incorrect.
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMapping());
    const auto permuteOutShape = outType.getShape();  // = AffineReshape input shape

    int64_t reshapeInDim = -1;
    for (int64_t i = 0; i < static_cast<int64_t>(dimMapping.size()); ++i) {
        for (auto outDim : dimMapping[i]) {
            if (outDim == sliceDim && permuteOutShape[Dim(i)] > 1) {
                if (reshapeInDim != -1) {
                    return matchFailed(rewriter, memPermuteOp,
                                       "More than one non-trivial input dim maps to slice dim {0}", sliceDim);
                }
                reshapeInDim = i;
                break;
            }
        }
    }
    if (reshapeInDim == -1) {
        return matchFailed(rewriter, memPermuteOp, "No non-trivial AffineReshape input dim maps to slice dim {0}",
                           sliceDim);
    }
    // Require that reshapeInDim maps exclusively to sliceDim (not split across multiple output dims).
    if (dimMapping[reshapeInDim].size() != 1) {
        return matchFailed(rewriter, memPermuteOp, "AffineReshape input dim {0} is split across multiple output dims",
                           reshapeInDim);
    }

    // Step 6: Back-infer origSliceDim via the inverse permutation.
    // permVec[outPos] = inputDim, so the input dim for MemPermute output dim reshapeInDim is
    // directly permVec[reshapeInDim].
    const auto permuteInShape = inType.getShape();
    const int64_t origSliceDim = permVec[reshapeInDim];

    // Step 7: Build the new AffineReshape dim_mapping.
    // For each MemPermute input dim i, find the MemPermute output dim it produces (forward perm),
    // and use the AffineReshape mapping for that output dim.
    // forwardPermVec[inputDim] = outputDim = inverse of permVec.
    const auto forwardPermMap = mlir::inversePermutation(memPermuteOp.getMemPerm());
    const auto forwardPermVec =
            to_small_vector(DimsOrder::fromAffineMap(forwardPermMap).toPermutation() | transformed([](Dim dim) {
                                return checked_cast<int64_t>(dim.ind());
                            }));
    SmallVector<SmallVector<int64_t>> newDimMapping;
    for (size_t i = 0; i < dimMapping.size(); ++i) {
        newDimMapping.push_back(SmallVector<int64_t>(dimMapping[forwardPermVec[i]]));
    }

    // Monotonicity fixup: origSliceDim becomes size=1 after slicing, which may break the
    // monotonically non-decreasing requirement on dim_mapping. Since a size-1 dim is a no-op
    // factor, we can remap it to an adjacent dim's output index.
    // The fixup is only valid for single-entry mappings (merging, not splitting).
    if (origSliceDim > 0) {
        auto prevMax = newDimMapping[origSliceDim - 1].back();
        auto curMin = newDimMapping[origSliceDim].front();
        if (curMin < prevMax) {
            if (newDimMapping[origSliceDim].size() != 1 || newDimMapping[origSliceDim - 1].size() != 1) {
                return matchFailed(rewriter, memPermuteOp,
                                   "Cannot apply monotonicity fixup to multi-entry dim mapping");
            }
            newDimMapping[origSliceDim] = SmallVector<int64_t>{prevMax};
        }
    } else if (origSliceDim == 0 && newDimMapping.size() > 1) {
        auto nextMin = newDimMapping[1].front();
        auto curMax = newDimMapping[0].back();
        if (curMax > nextMin) {
            if (newDimMapping[0].size() != 1 || newDimMapping[1].size() != 1) {
                return matchFailed(rewriter, memPermuteOp,
                                   "Cannot apply monotonicity fixup to multi-entry dim mapping");
            }
            newDimMapping[0] = SmallVector<int64_t>{nextMin};
        }
    }

    // Validate the new mapping before emitting any ops
    const auto sliceOutShape = getShape(sliceOps[0].getResult());
    SmallVector<int64_t> newInputShape;
    for (int64_t d = 0; d < static_cast<int64_t>(permuteInShape.size()); ++d) {
        newInputShape.push_back(d == origSliceDim ? 1 : permuteInShape[Dim(d)]);
    }
    if (!isValidReshapeMapping(newDimMapping, newInputShape, sliceOutShape)) {
        return matchFailed(rewriter, memPermuteOp, "New dim mapping is not valid after adjustment");
    }

    // Sort slices by their offset on sliceDim for deterministic IR output order
    llvm::sort(sliceOps, [sliceDim](IE::SliceOp a, IE::SliceOp b) {
        return parseIntArrayAttr<int64_t>(a.getStaticOffsetsAttr())[sliceDim] <
               parseIntArrayAttr<int64_t>(b.getStaticOffsetsAttr())[sliceDim];
    });

    auto* ctx = rewriter.getContext();

    // Step 8: Replace each Slice with Slice(origSliceDim) → AffineReshape
    for (auto sliceOp : sliceOps) {
        const auto origOffset = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr());
        const auto sliceIndex = origOffset[sliceDim];

        SmallVector<int64_t> newOffsets(permuteInShape.size(), 0);
        newOffsets[origSliceDim] = sliceIndex;

        SmallVector<int64_t> newSizes;
        for (int64_t d = 0; d < static_cast<int64_t>(permuteInShape.size()); ++d) {
            newSizes.push_back(d == origSliceDim ? 1 : permuteInShape[Dim(d)]);
        }

        auto newSlice = rewriter.create<IE::SliceOp>(sliceOp.getLoc(), memPermuteOp.getInput(),
                                                     getIntArrayAttr(ctx, newOffsets), getIntArrayAttr(ctx, newSizes));

        auto newDimMappingAttr = getIntArrayOfArray(ctx, newDimMapping);
        auto newShapeAttr = getIntArrayAttr(ctx, sliceOutShape.raw());
        auto newReshape = rewriter.create<IE::AffineReshapeOp>(sliceOp.getLoc(), newSlice.getResult(),
                                                               newDimMappingAttr, newShapeAttr);

        rewriter.replaceOp(sliceOp, newReshape.getResult());
    }

    rewriter.eraseOp(affineReshapeOp);
    rewriter.eraseOp(memPermuteOp);

    return mlir::success();
}

}  // namespace

void vpux::IE::MemPermuteOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                         mlir::MLIRContext* context) {
    patterns.add<FuseMemPermutes>(context);
    patterns.add<ConvertToPermuteCast>(context);
    patterns.add<FusePermCastAndMemPerm>(context);
    patterns.add<FuseMemPermuteThroughConcat>(context);
    patterns.add<FuseMemPermuteThroughExpand>(context);
    patterns.add<FuseMemPermuteAndPermuteQuantize>(context);
    patterns.add<ConvertShapeCastToPermuteCast>(context);
    patterns.add<EliminateMemPermuteThroughReshapeSlice>(context);
}

void vpux::IE::registerMemPermuteOpRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels,
                                             size_t index) {
    registry.registerRewriter<FuseMemPermutes>("fuse-mem-permutes", benefitLevels[index]);
    registry.registerRewriter<ConvertToPermuteCast>("convert-to-permute-cast", benefitLevels[index]);
    registry.registerRewriter<FusePermCastAndMemPerm>("fuse-perm-cast-and-mem-perm", benefitLevels[index]);
    registry.registerRewriter<FuseMemPermuteThroughConcat>("fuse-mem-permute-through-concat", benefitLevels[index]);
    registry.registerRewriter<FuseMemPermuteThroughExpand>("fuse-mem-permute-through-expand", benefitLevels[index]);
    registry.registerRewriter<FuseMemPermuteAndPermuteQuantize>("fuse-mem-permute-and-permute-quantize",
                                                                benefitLevels[index]);
    registry.registerRewriter<ConvertShapeCastToPermuteCast>("convert-shape-cast-to-permute-cast",
                                                             benefitLevels[index]);
    registry.registerRewriter<EliminateMemPermuteThroughReshapeSlice>("eliminate-mem-permute-through-reshape-slice",
                                                                      benefitLevels[index]);
}

mlir::OpFoldResult vpux::IE::MemPermuteOp::fold(FoldAdaptor adaptor) {
    if (getInput().getType() == getOutput().getType() && getMemPerm().isIdentity()) {
        return getInput();
    }

    auto operands = adaptor.getOperands();
    if (const auto cst = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        auto dstOrder = DimsOrder::fromAffineMap(getDstOrder());
        auto memPerm = DimsOrder::fromAffineMap(getMemPerm());
        return static_cast<Const::ContentAttr>(cst).transform().memPermute(dstOrder, memPerm).get();
    }

    return nullptr;
}
