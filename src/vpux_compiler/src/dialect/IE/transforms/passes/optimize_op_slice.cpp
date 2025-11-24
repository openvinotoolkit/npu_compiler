//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZEOPSLICE
#define GEN_PASS_DEF_OPTIMIZEOPSLICE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// TileSliceRewriter
//

class TileSliceRewriter final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    TileSliceRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("TileSliceRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TileSliceRewriter::matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite Slice operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    const auto ctx = rewriter.getContext();
    auto reshapeOp = origOp.getSource().getDefiningOp<IE::AffineReshapeOp>();

    bool hasReshape = reshapeOp != nullptr;
    auto tileOp = hasReshape ? reshapeOp.getInput().getDefiningOp<IE::TileOp>()
                             : origOp.getSource().getDefiningOp<IE::TileOp>();

    if (tileOp == nullptr) {
        return mlir::failure();
    }
    if (!tileOp->hasOneUse()) {
        return mlir::failure();
    }
    auto repeatsValue = parseIntArrayAttr<int64_t>(tileOp.getRepeatsValuesAttr());
    const auto nonOneRepeatsValueNum = llvm::count_if(repeatsValue, [](auto repeat) {
        return repeat != 1;
    });
    if (nonOneRepeatsValueNum != 1) {
        return mlir::failure();
    }
    auto tileAxis = IE::getSingleDiffAxis(getShape(tileOp.getInput()), getShape(tileOp.getOutput()));
    if (!tileAxis.has_value()) {
        return mlir::failure();
    }
    auto sliceAxis = IE::getSingleDiffAxis(getShape(origOp.getSource()), getShape(origOp.getResult()));
    if (!sliceAxis.has_value()) {
        return mlir::failure();
    }

    const auto sliceOffset = parseIntArrayAttr<int64_t>(origOp.getStaticOffsetsAttr());
    const auto sliceSize = parseIntArrayAttr<int64_t>(origOp.getStaticSizesAttr());
    const auto nonZeroOffset = llvm::any_of(sliceOffset, [](auto offset) {
        return offset > 0;
    });

    if (nonZeroOffset) {
        return mlir::failure();
    }

    int64_t newRepeatValue = 0;
    if (!hasReshape) {
        // If there is not reshape operation present, tile and slice axis need to be the same
        if (tileAxis.value() != sliceAxis.value()) {
            return mlir::failure();
        }
        newRepeatValue = sliceSize[sliceAxis.value().ind()];
    } else {
        const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(reshapeOp.getDimMapping());
        auto mappedDim = dimMapping[tileAxis.value().ind()];

        if (mappedDim.size() > 1 || mappedDim[0] != sliceAxis.value().ind()) {
            return mlir::failure();
        }
        auto reshapeInputShape = getShape(reshapeOp.getInput());
        SmallVector<int64_t> mergedDim;
        for (int64_t index = 0; index < checked_cast<int64_t>(reshapeInputShape.size()); index++) {
            if (index == tileAxis.value().ind()) {
                continue;
            }
            auto localMappedDim = dimMapping[index];
            if (localMappedDim.size() == 1 && localMappedDim[0] == mappedDim[0]) {
                mergedDim.push_back(index);
            }
        }

        if (mergedDim.empty()) {
            newRepeatValue = sliceSize[sliceAxis.value().ind()];
        } else if (mergedDim.size() == 1) {
            if (sliceSize[sliceAxis.value().ind()] % reshapeInputShape[Dim(mergedDim[0])] == 0) {
                newRepeatValue = sliceSize[sliceAxis.value().ind()] / reshapeInputShape[Dim(mergedDim[0])];
            } else {
                return mlir::failure();
            }
        } else {
            return mlir::failure();
        }
    }

    auto repeatsOnNewShape = std::move(repeatsValue);
    repeatsOnNewShape[tileAxis.value().ind()] = newRepeatValue;
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(tileOp.getOutput().getType());
    auto tileOpInputShape = getShape(tileOp.getInput());
    Shape newShape(tileOpInputShape.raw());
    newShape[tileAxis.value()] = newRepeatValue;
    const auto newOutputType = outputType.changeShape(newShape);
    auto newTileOp = rewriter.create<IE::TileOp>(takeOpLoc(origOp, "tile_in"), newOutputType, tileOp.getInput(),
                                                 nullptr, getIntArrayAttr(ctx, repeatsOnNewShape));
    if (hasReshape) {
        const auto sliceOutShape = getShape(origOp.getResult()).raw();
        rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(origOp, newTileOp.getOutput(), reshapeOp.getDimMappingAttr(),
                                                         getIntArrayAttr(ctx, sliceOutShape));
    } else {
        rewriter.replaceOp(origOp, newTileOp);
    }
    return mlir::success();
}

//
// ConcatSliceRewriter
//

class ConcatSliceRewriter final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    ConcatSliceRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("ConcatSliceRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConcatSliceRewriter::matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite Slice operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto reshapeOp = origOp.getSource().getDefiningOp<IE::AffineReshapeOp>();
    bool hasReshape = reshapeOp != nullptr;
    auto concatOp = hasReshape ? reshapeOp.getInput().getDefiningOp<IE::ConcatOp>()
                               : origOp.getSource().getDefiningOp<IE::ConcatOp>();

    if (concatOp == nullptr || !concatOp.getStaticOffsetsAttr()) {
        return mlir::failure();
    }

    // Previously the rewriter is before adjust layout, which only handle NCHW compatible layout.
    // After add the rewriter to run after adjust layout, we see some performance regression for
    // some very small model, acutally caused by runtime idle. #E135787
    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getResult().getType());
    if (outType.getDimsOrder() != DimsOrder::fromNumDims(outType.getShape().size())) {
        return mlir::failure();
    }

    auto sliceOffset = parseIntArrayAttr<int64_t>(origOp.getStaticOffsets());
    const auto sliceOffsetShape = Shape(sliceOffset);
    const auto sliceOutShape = getShape(origOp.getResult());
    auto concatOffsets = parseIntArrayOfArrayAttr<int64_t>(concatOp.getStaticOffsetsAttr());
    const auto inputs = concatOp.getInputs();
    SmallVector<ShapeRef> newInputShapes;
    SmallVector<SmallVector<int64_t>> newInputShapesVec;
    std::unordered_set<Dim> newConcatAxes;

    if (hasReshape) {
        for (const auto& input : inputs) {
            auto newShapeInfo = IE::inferOutputShapeAfterAffineReshapeBeforeConcat(input, concatOp, reshapeOp);
            if (!newShapeInfo.has_value()) {
                return mlir::failure();
            }
            const auto newConcatDim = newShapeInfo.value().first;
            const auto newShape = newShapeInfo.value().second;
            newInputShapesVec.emplace_back(to_small_vector(newShape));
            newConcatAxes.insert(newConcatDim);
        }

        if (newConcatAxes.size() != 1) {
            return mlir::failure();
        }

        const auto sliceAxis = IE::getSingleDiffAxis(getShape(origOp.getSource()), sliceOutShape);
        if (!sliceAxis.has_value()) {
            return matchFailed(rewriter, origOp, "Not a single axis slice operation");
        }

        const auto concatAxis = *getConcatAxes(concatOp).begin();
        const auto concatOutShape = getShape(concatOp.getResult());
        for (auto i = sliceAxis.value().ind(); i < concatAxis; i++) {
            if (concatOutShape[Dim(i)] != 1) {
                return matchFailed(rewriter, origOp, "Slice shape is not one on the front of Concat axis");
            }
        }

        for (const auto& vector : newInputShapesVec) {
            newInputShapes.emplace_back(ShapeRef(vector));
        }

        auto newOffsetsAttr = IE::inferConcatOffsets(newInputShapes, *newConcatAxes.begin(), origOp.getContext());
        concatOffsets = parseIntArrayOfArrayAttr<int64_t>(newOffsetsAttr);
    } else {
        for (const auto& input : inputs) {
            newInputShapes.push_back(getShape(input));
        }
    }

    for (const auto& p : zip(inputs, concatOffsets, newInputShapes)) {
        auto curVal = std::get<0>(p);
        const auto curOffset = std::get<1>(p);
        const auto curShape = std::get<2>(p);
        const auto curOffsetShape = ShapeRef(curOffset);
        auto isSubTensor = [&]() -> bool {
            for (const auto ind : irange(sliceOutShape.size())) {
                const auto d = Dim(ind);
                if ((sliceOffsetShape[d] < curOffsetShape[d]) ||
                    (curOffsetShape[d] + curShape[d] < sliceOffsetShape[d] + sliceOutShape[d])) {
                    return false;
                }
            }
            return true;
        };

        if (!isSubTensor()) {
            continue;
        }

        _log.trace("ConcatSliceRewriter '{0}' at '{1}', {2}->{3}, {4}->{5}", origOp->getName(), origOp->getLoc(),
                   sliceOffsetShape, curOffsetShape, sliceOutShape, curShape);

        if (hasReshape) {
            curVal = rewriter.create<IE::AffineReshapeOp>(takeOpLoc(reshapeOp, "affine_in"), curVal,
                                                          reshapeOp.getDimMapping(),
                                                          getIntArrayAttr(rewriter.getContext(), curShape));
        }

        for (auto i : irange(sliceOffset.size())) {
            sliceOffset[i] -= curOffset[i];
        }

        rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, curVal, getIntArrayAttr(getContext(), sliceOffset),
                                                 origOp.getStaticSizes());

        return mlir::success();
    }

    return mlir::failure();
}

//
// SliceConcatRewriter
//

class SliceConcatRewriter final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    SliceConcatRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("SliceConcatRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isContinuousSlices(SmallVector<IE::SliceOp>& sliceOps, vpux::Dim sliceDim) {
    for (auto idx : irange(sliceOps.size() - 1)) {
        const auto preOffsets = parseIntArrayAttr<int64_t>(sliceOps[idx].getStaticOffsets());
        const auto preSizes = parseIntArrayAttr<int64_t>(sliceOps[idx].getStaticSizes());
        const auto nextOffsets = parseIntArrayAttr<int64_t>(sliceOps[idx + 1].getStaticOffsets());
        if (preOffsets[sliceDim.ind()] + preSizes[sliceDim.ind()] != nextOffsets[sliceDim.ind()]) {
            return false;
        }
    }
    return true;
}

mlir::LogicalResult SliceConcatRewriter::matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite Slice Concat operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto ctx = origOp->getContext();
    auto sliceParent = origOp.getSource();
    auto outShape = getShape(origOp.getResult());
    auto refSliceAxis = IE::getSingleDiffAxis(getShape(origOp.getSource()), outShape);
    if (!refSliceAxis.has_value()) {
        return matchFailed(rewriter, origOp, "Not a single axis slice operation");
    }

    if (!origOp->hasOneUse()) {
        return matchFailed(rewriter, origOp, "Slice has more than one user");
    }

    auto concatOp = mlir::dyn_cast<IE::ConcatOp>(*origOp.getResult().getUsers().begin());
    if (concatOp == nullptr) {
        return matchFailed(rewriter, origOp, "Slice user is not a concat");
    }

    auto concatAxis = IE::getSingleDiffAxis(outShape, getShape(concatOp.getOutput()));
    if (!concatAxis.has_value()) {
        return matchFailed(rewriter, origOp, "Not a single axis concat operation");
    }

    SmallVector<IE::SliceOp> sliceOps;
    for (auto user : sliceParent.getUsers()) {
        if (auto sliceOp = mlir::dyn_cast<IE::SliceOp>(user)) {
            auto sliceAxis = IE::getSingleDiffAxis(getShape(sliceOp.getSource()), getShape(sliceOp.getResult()));
            if (!sliceAxis.has_value() || sliceAxis.value() != refSliceAxis.value()) {
                return matchFailed(rewriter, origOp, "Don't have same slice axis");
            }
            if (!sliceOp->hasOneUse()) {
                return matchFailed(rewriter, origOp, "Slice has more than one user");
            }
            // slice have the same concat user
            if (concatOp == (*sliceOp.getResult().getUsers().begin())) {
                sliceOps.push_back(sliceOp);
            }
        }
    }
    if (sliceOps.size() < 2) {
        return matchFailed(rewriter, origOp, "Need to have at least 2 slice operation");
    }

    // sort slice ops according to the offset
    std::sort(sliceOps.begin(), sliceOps.end(), [&](IE::SliceOp firstOp, IE::SliceOp secondOp) {
        const auto firstOffsets = parseIntArrayAttr<int64_t>(firstOp.getStaticOffsets());
        const auto secondOffsets = parseIntArrayAttr<int64_t>(secondOp.getStaticOffsets());
        return firstOffsets[refSliceAxis.value().ind()] < secondOffsets[refSliceAxis.value().ind()];
    });

    // only continuous slices can be optimized.
    if (!isContinuousSlices(sliceOps, refSliceAxis.value())) {
        return matchFailed(rewriter, origOp, "Not continious slice from parent");
    }

    SmallVector<IE::SliceOp> concatSliceParent;
    bool firstSliceMatched = false;
    for (auto operand : concatOp->getOperands()) {
        if (auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(operand.getDefiningOp())) {
            if (sliceOp == sliceOps.front()) {
                firstSliceMatched = true;
            }
            if (firstSliceMatched) {
                concatSliceParent.push_back(sliceOp);
                if (sliceOp == sliceOps.back()) {
                    break;
                } else {
                    continue;
                }
            }
        }
        if (firstSliceMatched) {
            break;
        }
    }

    if (concatSliceParent != sliceOps) {
        return matchFailed(rewriter, origOp, "Slice is not the same sequence as concat");
    }

    // if concat axis is different with slice axis, try to add a permuteCastOp.
    if (concatAxis.value() != refSliceAxis.value()) {
        const auto inType = mlir::cast<vpux::NDTypeInterface>(sliceParent.getType());
        const auto dimsOrder = inType.getDimsOrder();
        auto permVec = to_small_vector(dimsOrder.toPermutation() | transformed([](Dim dim) {
                                           return checked_cast<int64_t>(dim.ind());
                                       }));
        std::swap(permVec[dimsOrder.dimPos(concatAxis.value())], permVec[dimsOrder.dimPos(refSliceAxis.value())]);
        const auto newOrder = DimsOrder::fromAffineMap(mlir::AffineMap::getPermutationMap(ArrayRef(permVec), ctx));
        const auto memPerm = vpux::getPermutationFromOrders(dimsOrder, newOrder, ctx);
        const auto memShape = inType.getMemShape();
        if (!isTrivialPermute(memShape, memPerm)) {
            return matchFailed(rewriter, origOp,
                               "Could not add permuteCast when slice axis and concat axis is different");
        }
        sliceParent =
                rewriter.create<IE::PermuteCastOp>(origOp->getLoc(), sliceParent, dimsOrder.toAffineMap(ctx), memPerm);
    }

    const auto firstOffsets = parseIntArrayAttr<int64_t>(sliceOps.front().getStaticOffsets());
    Shape sliceOffset(firstOffsets.size());
    sliceOffset[concatAxis.value()] = firstOffsets[refSliceAxis.value().ind()];
    Shape sliceShape(getShape(sliceParent));
    sliceShape[concatAxis.value()] = std::accumulate(sliceOps.begin(), sliceOps.end(), 0, [&](auto& a, auto& b) {
        const auto sliceSize = parseIntArrayAttr<int64_t>(b.getStaticSizes());
        return a + sliceSize[refSliceAxis.value().ind()];
    });

    auto newSlice = rewriter.create<IE::SliceOp>(origOp->getLoc(), sliceParent,
                                                 getIntArrayAttr(getContext(), sliceOffset.raw()),
                                                 getIntArrayAttr(getContext(), sliceShape.raw()));

    // replace several slice input with one big slice input
    SmallVector<mlir::Value> concatInput;
    for (auto parent : concatOp->getOperands()) {
        auto slice = mlir::dyn_cast_or_null<IE::SliceOp>(parent.getDefiningOp());
        if (slice == nullptr) {
            concatInput.push_back(parent);
            continue;
        }
        if (std::find(sliceOps.begin(), sliceOps.end(), slice) == sliceOps.end()) {
            concatInput.push_back(parent);
        } else {
            if (std::find(concatInput.begin(), concatInput.end(), newSlice.getResult()) == concatInput.end()) {
                concatInput.push_back(newSlice.getResult());
            }
        }
    }

    auto newConcat = rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, "_concat"), mlir::ValueRange(concatInput),
                                                   concatAxis.value());
    for (auto operand : newConcat->getOperands()) {
        auto parentOp = operand.getDefiningOp();
        if (parentOp && newConcat->isBeforeInBlock(parentOp)) {
            parentOp->moveAfter(parentOp);
        }
    }

    rewriter.replaceOp(concatOp, newConcat.getOutput());
    _log.trace("Optimize slice and concat operations successfully");
    return mlir::success();
}

//
// OptimizeOpSlicePass
//

class OptimizeOpSlicePass final : public IE::impl::OptimizeOpSliceBase<OptimizeOpSlicePass> {
public:
    explicit OptimizeOpSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeOpSlicePass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConcatSliceRewriter>(&ctx, _log);
    patterns.insert<TileSliceRewriter>(&ctx, _log);
    patterns.insert<SliceConcatRewriter>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeOpSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeOpSlicePass(Logger log) {
    return std::make_unique<OptimizeOpSlicePass>(log);
}
