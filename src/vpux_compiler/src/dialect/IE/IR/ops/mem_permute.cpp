//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

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
    if (concatOp == nullptr) {
        return mlir::failure();
    }

    SmallVector<IE::MemPermuteOp> topMemPermutes;
    for (const auto& input : concatOp.getInputs()) {
        auto topMemPermute = input.getDefiningOp<IE::MemPermuteOp>();
        if (topMemPermute == nullptr) {
            return mlir::failure();
        }
        topMemPermutes.push_back(topMemPermute);
    }

    IE::MemPermuteOp refMemPermute = topMemPermutes.front();
    SmallVector<mlir::Value> newConcatInputs;
    for (auto& topMemPermute : topMemPermutes) {
        if (refMemPermute.getDstOrder() != topMemPermute.getDstOrder()) {
            return mlir::failure();
        }
        if (refMemPermute.getMemPerm() != topMemPermute.getMemPerm()) {
            return mlir::failure();
        }
        newConcatInputs.push_back(topMemPermute.getInput());
    }

    const auto inputShape = getShape(refMemPermute.getOutput());
    const auto inOrder = DimsOrder::fromValue(refMemPermute.getOutput());
    const auto inMemShape = inOrder.toMemoryOrder(inputShape);
    const auto outputShape = getShape(concatOp.getOutput());
    const auto outOrder = DimsOrder::fromValue(concatOp.getOutput());
    const auto outMemShape = outOrder.toMemoryOrder(outputShape);
    const auto perm = refMemPermute.getMemPerm();

    const auto permuteInOrder = DimsOrder::fromValue(refMemPermute.getInput());

    int32_t newAxis = -1;

    for (size_t idx = 0; idx < inMemShape.size(); ++idx) {
        if (inMemShape.raw()[idx] == outMemShape.raw()[idx]) {
            continue;
        }
        if (newAxis != -1) {
            // 2 axis concat
            return mlir::failure();
        }
        newAxis = perm.getDimPosition(static_cast<uint32_t>(idx));
        newAxis = permuteInOrder.dimAt(static_cast<size_t>(newAxis)).ind();
    }

    auto newConcat = rewriter.replaceOpWithNewOp<IE::ConcatOp>(concatOp, newConcatInputs, newAxis);

    auto prevMemPerm = refMemPermute.getMemPerm();
    auto memPerm = memPermuteOp.getMemPerm();
    auto newMemPerm = memPerm.compose(prevMemPerm);

    rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(memPermuteOp, memPermuteOp.getType(), newConcat,
                                                  memPermuteOp.getDstOrderAttr(), mlir::AffineMapAttr::get(newMemPerm));

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

    auto outputType = memPermuteOp.getOutput().getType().cast<NDTypeInterface>();
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
            permuteQuantizeOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();
    if (permuteQuantizeOutElemType.isa<mlir::quant::QuantizedType>()) {
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

}  // namespace

void vpux::IE::MemPermuteOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                         mlir::MLIRContext* context) {
    patterns.add<FuseMemPermutes>(context);
    patterns.add<ConvertToPermuteCast>(context);
    patterns.add<FusePermCastAndMemPerm>(context);
    patterns.add<FuseMemPermuteThroughConcat>(context);
    patterns.add<FuseMemPermuteThroughExpand>(context);
    patterns.add<FuseMemPermuteAndPermuteQuantize>(context);
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
