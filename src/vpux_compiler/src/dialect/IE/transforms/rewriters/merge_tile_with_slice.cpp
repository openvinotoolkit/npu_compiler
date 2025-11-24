//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// MergeTileWithReshapeSlice
//

//                                Input(2x1x10x80)
//                                        |
//                                IE.Tile(2x2x10x80)
//                                        |
//                                IE.Reshape(1x4x10x80)
//                                        |
//             -----------------------------------------------------------
//             |                   |                   |                 |
//        IE.Slice(1x1x10x80) IE.Slice(1x1x10x80) IE.Slice(1x1x10x80) IE.Slice(1x1x10x80)

// To:

//                                Input(2x1x10x80)
//                                        |
//             -----------------------------------------------------------
//             |                   |                   |                 |
//        IE.Slice(1x1x10x80) IE.Slice(1x1x10x80) IE.Slice(1x1x10x80) IE.Slice(1x1x10x80)

class MergeTileWithReshapeSlice final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    MergeTileWithReshapeSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TileOp>(ctx), _log(log) {
        setDebugName("MergeTileWithReshapeSlice");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp tileOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

SmallVector<int64_t> getNewSliceOffset(IE::SliceOp sliceOp, IE::TileOp tileOp, ArrayRef<int64_t> mergeDim,
                                       int64_t repeatDim) {
    auto sliceOffset = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr());
    const auto sliceInShape = getShape(sliceOp.getSource());
    const auto sliceOutShape = getShape(sliceOp.getResult());
    const auto tileOutShape = getShape(tileOp.getOutput());
    const auto tileInShape = getShape(tileOp.getInput());

    auto sliceDim = IE::getSingleDiffAxis(sliceInShape, sliceOutShape);
    VPUX_THROW_WHEN(!sliceDim.has_value(), "Slice is not a single Dim slice");
    VPUX_THROW_WHEN(mergeDim.size() != 2, "mergeDim size is not correct");
    SmallVector<int64_t> newOffset = sliceOffset;
    newOffset[mergeDim.front()] = sliceOffset[sliceDim.value().ind()] / tileOutShape[Dim(mergeDim.back())];
    newOffset[mergeDim.back()] = sliceOffset[sliceDim.value().ind()] % tileOutShape[Dim(mergeDim.back())];

    // Adjust the offset before tile op
    if (repeatDim == mergeDim.front()) {
        newOffset[mergeDim.front()] = newOffset[mergeDim.front()] % tileInShape[Dim(repeatDim)];
    }
    if (repeatDim == mergeDim.back()) {
        newOffset[mergeDim.back()] = newOffset[mergeDim.back()] % tileInShape[Dim(repeatDim)];
    }

    return newOffset;
}

SmallVector<int64_t> getNewSliceSize(IE::SliceOp sliceOp, ArrayRef<int64_t> mergeDim) {
    auto sliceSize = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizesAttr());
    const auto sliceInShape = getShape(sliceOp.getSource());
    const auto sliceOutShape = getShape(sliceOp.getResult());
    auto sliceDim = IE::getSingleDiffAxis(sliceInShape, sliceOutShape);
    VPUX_THROW_WHEN(!sliceDim.has_value(), "Slice is not a single Dim slice");

    if (sliceDim.value().ind() == mergeDim.front()) {
        std::swap(sliceSize[mergeDim.front()], sliceSize[mergeDim.back()]);
    }
    return sliceSize;
}

bool doesSliceMeetRequirement(IE::SliceOp sliceOp, IE::TileOp tileOp, ArrayRef<int64_t> mergeDim, int64_t repeatDim) {
    const auto inShape = getShape(sliceOp.getSource());
    const auto outShape = getShape(sliceOp.getResult());
    auto sliceDim = IE::getSingleDiffAxis(inShape, outShape);
    if (!sliceDim.has_value()) {
        return false;
    }

    if (std::find(mergeDim.begin(), mergeDim.end(), sliceDim.value().ind()) == mergeDim.end()) {
        return false;
    }

    // check if slice in the whole original tensor
    auto newOffset = getNewSliceOffset(sliceOp, tileOp, mergeDim, repeatDim);
    if (newOffset[mergeDim.front()] != 0 && newOffset[mergeDim.back()] != 0) {
        return false;
    }
    auto origShape = getShape(tileOp.getInput());
    if (newOffset[mergeDim.back()] + outShape[sliceDim.value()] > origShape[Dim(mergeDim.back())]) {
        return false;
    }

    return true;
}

bool doesTransposeMeetRequirement(IE::TransposeOp transposeOp, ArrayRef<int64_t> mergeDim) {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(transposeOp.getInput().getType());
    const auto inOrder = inType.getDimsOrder();
    const auto orderAttr = transposeOp.getOrderValueAttr();
    const auto transposeOrder = DimsOrder::fromAffineMap(orderAttr.getValue());
    const auto inPerm = inOrder.toPermutation();
    const auto transposePerm = transposeOrder.toPermutation();
    for (auto p : inPerm | indexed) {
        auto index = p.index();
        if (inPerm[index] != transposePerm[index] &&
            std::find(mergeDim.begin(), mergeDim.end(), index) != mergeDim.end()) {
            return false;
        }
    }

    return true;
}

mlir::LogicalResult MergeTileWithReshapeSlice::matchAndRewrite(IE::TileOp tileOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Tile layer at '{1}'", tileOp->getName(), tileOp->getLoc());

    const int64_t rank4D = 4;
    auto tileInShape = getShape(tileOp.getInput());
    auto tileOutShape = getShape(tileOp.getOutput());

    if (tileInShape.size() != rank4D || tileOutShape.size() != rank4D) {
        return matchFailed(rewriter, tileOp, "Input is not 4D");
    }

    auto repeatDim = IE::getSingleDiffAxis(tileInShape, tileOutShape);
    if (!repeatDim.has_value()) {
        return mlir::failure();
    }

    if (!tileOp->hasOneUse()) {
        return matchFailed(rewriter, tileOp, "Tile has more than one user");
    }

    auto reshapeOp = mlir::dyn_cast<IE::ReshapeOp>(*tileOp.getOutput().getUsers().begin());
    if (reshapeOp == nullptr) {
        return matchFailed(rewriter, tileOp, "Tile user is not reshape");
    }
    auto reshapeOutShape = getShape(reshapeOp.getOutput());
    if (reshapeOutShape.size() != rank4D) {
        return matchFailed(rewriter, tileOp, "Reshape is not 4D");
    }

    SmallVector<int64_t> mergeDim;
    for (auto ind : irange(reshapeOutShape.size())) {
        if (reshapeOutShape[Dim(ind)] != tileOutShape[Dim(ind)]) {
            mergeDim.push_back(ind);
        }
    }

    // two adjecent dims merged and reshape to 1xN or Nx1 like: 2x3x4x5 -> 1x6x4x5 or 6x1x4x5
    if (mergeDim.size() != 2) {
        return matchFailed(rewriter, tileOp, "Can only support merge two dims");
    }

    if ((mergeDim.back() - mergeDim.front()) != 1) {
        return matchFailed(rewriter, tileOp, "Merge dims should be adjecent");
    }

    const auto mergedValue = tileOutShape[Dim(mergeDim.front())] * tileOutShape[Dim(mergeDim.back())];
    if ((reshapeOutShape[Dim(mergeDim.front())] != 1 || reshapeOutShape[Dim(mergeDim.back())] != mergedValue) &&
        (reshapeOutShape[Dim(mergeDim.back())] != 1 || reshapeOutShape[Dim(mergeDim.front())] != mergedValue)) {
        return matchFailed(rewriter, tileOp, "Merge dims do not meet requirement");
    }

    mlir::Value sliceParent = reshapeOp.getOutput();
    IE::TransposeOp transposeOp = nullptr;
    if (reshapeOp->hasOneUse()) {
        auto user = *reshapeOp.getOutput().getUsers().begin();
        auto transpose = mlir::dyn_cast<IE::TransposeOp>(user);
        if (transpose != nullptr && doesTransposeMeetRequirement(transpose, mergeDim)) {
            sliceParent = transpose.getOutput();
            transposeOp = transpose;
        }
    }

    for (auto user : sliceParent.getUsers()) {
        auto sliceOp = mlir::dyn_cast<IE::SliceOp>(user);
        if (sliceOp == nullptr) {
            return matchFailed(rewriter, tileOp, "Reshape user is not a slice");
        }
        if (!doesSliceMeetRequirement(sliceOp, tileOp, mergeDim, repeatDim.value().ind())) {
            return matchFailed(rewriter, tileOp, "Slice does not meet requirement");
        }
    }

    mlir::Value tileInput = tileOp.getInput();
    if (transposeOp) {
        auto newTranspose = rewriter.create<IE::TransposeOp>(transposeOp->getLoc(), tileInput, nullptr,
                                                             transposeOp.getOrderValueAttr());
        tileInput = newTranspose.getOutput();
    }

    std::map<SmallVector<SmallVector<int64_t>>, mlir::Value> sliceMap;
    for (auto user : llvm::make_early_inc_range(sliceParent.getUsers())) {
        rewriter.setInsertionPointAfter(tileOp);
        auto sliceOp = mlir::cast<IE::SliceOp>(user);
        auto newSliceOffset = getNewSliceOffset(sliceOp, tileOp, mergeDim, repeatDim.value().ind());
        auto newSliceSize = getNewSliceSize(sliceOp, mergeDim);
        SmallVector<SmallVector<int64_t>> mapKey = {newSliceOffset, newSliceSize};
        auto iterator = sliceMap.find(mapKey);

        if (iterator != sliceMap.end()) {
            rewriter.replaceOp(sliceOp, iterator->second);
        } else {
            auto newSlice = rewriter.create<IE::SliceOp>(sliceOp->getLoc(), tileInput,
                                                         getIntArrayAttr(getContext(), newSliceOffset),
                                                         getIntArrayAttr(getContext(), newSliceSize));
            rewriter.replaceOp(sliceOp, newSlice.getResult());
            sliceMap.insert(std::make_pair(mapKey, newSlice.getResult()));
        }
    }

    _log.trace("Merge Tile {0} to slice success", tileOp->getLoc());
    return mlir::success();
}

//
// MergeTileWithSlice
//

//                                Input(1x1x80x80)
//                                        |
//                                IE.Tile(1x4x80x80)
//                                        |
//             -----------------------------------------------------------
//             |                   |                   |                 |
//        IE.Slice(1x1x80x80) IE.Slice(1x1x80x80) IE.Slice(1x1x80x80) IE.Slice(1x1x80x80)

// To:

//                                Input(1x1x80x80)
//                                        |

class MergeTileWithSlice final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    MergeTileWithSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TileOp>(ctx), _log(log) {
        setDebugName("MergeTileWithSlice");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp tileOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MergeTileWithSlice::matchAndRewrite(IE::TileOp tileOp, mlir::PatternRewriter& rewriter) const {
    const int64_t rank4D = 4;
    auto tileInShape = getShape(tileOp.getInput());
    auto tileOutShape = getShape(tileOp.getOutput());

    if (tileInShape.size() != rank4D || tileOutShape.size() != rank4D) {
        return matchFailed(_log, rewriter, tileOp, "Input is not 4D");
    }

    auto repeatDim = IE::getSingleDiffAxis(tileInShape, tileOutShape);
    if (!repeatDim.has_value()) {
        return matchFailed(_log, rewriter, tileOp, "Tile on different axis");
    }

    auto areAllUserSliceMeetRequirement = [](IE::TileOp tileOp) {
        auto origShape = getShape(tileOp.getInput());
        for (auto user : tileOp.getOutput().getUsers()) {
            auto sliceOp = mlir::dyn_cast<IE::SliceOp>(user);
            if (sliceOp == nullptr) {
                return false;
            }
            if (getShape(tileOp.getInput()) != getShape(sliceOp.getResult())) {
                return false;
            }

            auto sliceOffset = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
            const auto sliceInShape = getShape(sliceOp.getSource());
            const auto sliceOutShape = getShape(sliceOp.getResult());
            auto sliceDim = IE::getSingleDiffAxis(sliceInShape, sliceOutShape);
            // Check if the slice offset is within the repeated range
            if (sliceOffset[sliceDim.value().ind()] % origShape[sliceDim.value()] != 0) {
                return false;
            }
        }
        return true;
    };

    if (!areAllUserSliceMeetRequirement(tileOp)) {
        return matchFailed(_log, rewriter, tileOp, "Cannot find user Slice op meet requirement");
    }

    for (auto user : tileOp.getOutput().getUsers()) {
        auto sliceOp = mlir::dyn_cast<IE::SliceOp>(user);
        VPUX_THROW_UNLESS(sliceOp != nullptr, "Cannot find user Slice op");
        sliceOp->getResult(0).replaceAllUsesWith(tileOp.getInput());
    }

    _log.trace("Merge Tile {0} to slice success", tileOp->getLoc());
    return mlir::success();
}

}  // namespace

void vpux::IE::registerMergeTileWithSliceRewriters(RewriterRegistry& registry, Logger log) {
    registry.registerRewriterSet("merge-tile-with-slice-set", [&registry, log]() {
        registry.registerRewriter<MergeTileWithReshapeSlice>("merge-tile-with-reshape-slice", log);
        registry.registerRewriter<MergeTileWithSlice>("merge-tile-with-slice", log);
    });
}
