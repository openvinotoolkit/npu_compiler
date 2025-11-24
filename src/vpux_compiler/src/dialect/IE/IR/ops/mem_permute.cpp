//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

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
    using mlir::OpRewritePattern<IE::MemPermuteOp>::OpRewritePattern;

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
    using mlir::OpRewritePattern<IE::MemPermuteOp>::OpRewritePattern;

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
    using mlir::OpRewritePattern<IE::MemPermuteOp>::OpRewritePattern;

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

mlir::ArrayAttr getNewPaddingAttr(mlir::MLIRContext* ctx, ArrayRef<int64_t> pads, vpux::DimsOrder targetOrder,
                                  vpux::DimsOrder outOrder) {
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
    using mlir::OpRewritePattern<IE::MemPermuteOp>::OpRewritePattern;

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
    using mlir::OpRewritePattern<IE::MemPermuteOp>::OpRewritePattern;

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
    if (mlir::isa<mlir::quant::QuantizedType>(permuteQuantizeOutElemType)) {
        return mlir::failure();
    }

    return fusePermutations<IE::PermuteQuantizeOp, IE::MemPermuteOp>(memPermuteOp, rewriter);
}

//
// ConvertToPermuteCast
//

class ConvertToPermuteCast final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    using mlir::OpRewritePattern<IE::MemPermuteOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToPermuteCast::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto inOrder = DimsOrder::fromValue(memPermuteOp.getInput());
    const auto inShape = getShape(memPermuteOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);

    if (!isTrivialPermute(inMemShape, memPermuteOp.getMemPerm())) {
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
    using mlir::OpRewritePattern<IE::ShapeCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertShapeCastToPermuteCast::matchAndRewrite(IE::ShapeCastOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    auto memPermuteOp = origOp.getOperand().getDefiningOp<IE::MemPermuteOp>();
    if (memPermuteOp == nullptr) {
        return mlir::failure();
    }

    const auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getResult().getType());
    const auto outOrder = outType.getDimsOrder();
    auto hasValidPermuteCast = vpux::tryToFindPermuteCastOp(origOp.getLoc(), origOp.getOperand(), outOrder,
                                                            getShape(origOp.getResult()), rewriter);

    if (!hasValidPermuteCast.has_value()) {
        return mlir::failure();
    }

    rewriter.replaceOp(origOp, hasValidPermuteCast.value().getResult());
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
