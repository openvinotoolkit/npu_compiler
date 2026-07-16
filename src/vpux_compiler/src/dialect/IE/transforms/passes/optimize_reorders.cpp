//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters/expand_with_layer_rewriter.hpp"
#include "vpux/compiler/dialect/IE/utils/analysis.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZEREORDERS
#define GEN_PASS_DEF_OPTIMIZEREORDERS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// If we can optimize Reorder->GroupConv case
bool DoesReorderWithGroupConvPatternMatch(IE::GroupConvolutionOp origOp) {
    if (!IE::isEltwiseGroupConv(origOp, /*isConstFilter*/ false)) {
        return false;
    }

    auto inReorderOp = origOp.getInput().getDefiningOp<IE::ReorderOp>();
    if (inReorderOp == nullptr || !inReorderOp->hasOneUse()) {
        return false;
    }

    const auto convInOrder = DimsOrder::fromValue(origOp.getInput());
    const auto convOutOrder = DimsOrder::fromValue(origOp.getOutput());
    if (convInOrder != convOutOrder) {
        return false;
    }

    const auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outType.getElementType())) {
        return false;
    }

    auto layerWithPostOp = mlir::cast<IE::LayerWithPostOpInterface>(origOp.getOperation());
    if (layerWithPostOp) {
        const auto postOp = layerWithPostOp.getPostOp();
        if (postOp != nullptr && !postOp.isChannelAgnostic()) {
            return false;
        }
    }

    return true;
}

// If there are nonTrivial Reorders before and after Tile, the two Reorders will be fused after switch Tile and Reorder.

bool isBeneficialReorderFuse(IE::TileOp tileOp) {
    auto inputReorderOp = tileOp.getInput().getDefiningOp<IE::ReorderOp>();

    if (!tileOp.getOutput().hasOneUse()) {
        return false;
    }

    bool isTrivial = inputReorderOp == nullptr ? true : IE::isTrivialReorder(inputReorderOp);

    auto outputReorderOp = mlir::dyn_cast<IE::ReorderOp>(*(tileOp.getOutput().user_begin()));
    isTrivial = isTrivial || (outputReorderOp == nullptr ? true : IE::isTrivialReorder(outputReorderOp));

    return inputReorderOp && outputReorderOp && !isTrivial;
}

// For tileOp tile data in high dim is more efficient than tile date in low dim.
// For example tileOp 1x16x1x1 -> 1x16x100x100 (repeatsValues = [1,1,100,100]), NHWC is more efficient than NCHW layout,
// because NHWC will tile data in the higher dime(H and W of NHWC), the tileOp will convert to no stride DMA. but for
// NCHW will tile data in the lower dim (H and W of NCHW), the tileOp will convert to stride DMA which is low efficient.

bool isBeneficialSwitch(IE::TileOp tileOp, const vpux::DimsOrder& origOrder, const vpux::DimsOrder& switchedOrder) {
    auto outputShape = getShape(tileOp.getOutput());

    if (std::all_of(outputShape.begin(), outputShape.end(), [](auto shape) {
            return shape == 1;
        })) {
        return false;
    }

    if (!tileOp.getRepeatsValues()) {
        return false;
    }
    auto repeatsValues = parseIntArrayAttr<int64_t>(tileOp.getRepeatsValuesAttr());

    SmallVector<int32_t> repeatAxesIndexVal, notRepeatAndHasValueAxesIndexVal;

    // Find the tile dim axes and not tile and has value(size != 1) dim axes.
    // Eg in tileOp 1x16x1x1 -> 1x16x100x100, tile dim axes is [2, 3], not-tile-and-has-value dim axes is [1]. Dim 0
    // size is 1, this dim has no impact on the reorder optimization.
    for (size_t ind = 0; ind < repeatsValues.size(); ++ind) {
        if (repeatsValues[ind] == 1 && outputShape[Dim(ind)] != 1) {
            notRepeatAndHasValueAxesIndexVal.push_back(ind);
        } else if (repeatsValues[ind] > 1) {
            repeatAxesIndexVal.push_back(ind);
        }
    }

    if (notRepeatAndHasValueAxesIndexVal.empty()) {
        return false;
    }

    // Check if all NotRepeatAndHasValueAxes larger than RepeatAxes.
    // Eg in tileOp 1x16x1x1(NHWC) -> 1x16x100x100(NHWC), repeat dim axes is [2, 3], not-repeat-dim axes is [0, 1]. For
    // NHWC layout, the real repeat dim axes is [1, 2], real not-repeat dim axes is [0, 3], and real
    // not-repeat-and-has-value dim axes is [3]. So we need check all of the not-repeat-and-has-value dim axes be larger
    // than the repeat dim axes.
    auto isAllNotRepeatAndHasValueAxesLargerThanRepeatAxes = [&](const vpux::DimsOrder& order,
                                                                 ArrayRef<int32_t> repeatAxes,
                                                                 ArrayRef<int32_t> notRepeatAndHasValueAxes) {
        SmallVector<int32_t> realRepeatAxes, realNotRepeatAndHasValueAxes;
        for (auto notRepeatAndHasValueAxis : notRepeatAndHasValueAxes) {
            realNotRepeatAndHasValueAxes.push_back(order.dimPos(Dim(notRepeatAndHasValueAxis)));
        }
        for (auto repeatAxis : repeatAxes) {
            realRepeatAxes.push_back(order.dimPos(Dim(repeatAxis)));
        }

        std::sort(realNotRepeatAndHasValueAxes.begin(), realNotRepeatAndHasValueAxes.end());
        std::sort(realRepeatAxes.begin(), realRepeatAxes.end());

        return realNotRepeatAndHasValueAxes.front() > realRepeatAxes.back();
    };

    return !isAllNotRepeatAndHasValueAxesLargerThanRepeatAxes(origOrder, repeatAxesIndexVal,
                                                              notRepeatAndHasValueAxesIndexVal) &&
           isAllNotRepeatAndHasValueAxesLargerThanRepeatAxes(switchedOrder, repeatAxesIndexVal,
                                                             notRepeatAndHasValueAxesIndexVal);
}

//
// ReorderWithShapeChange
//
template <class ConcreteOp>
class ReorderWithShapeChange final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ReorderWithShapeChange(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("ReorderWithShapeChange");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origReshapeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// This function is to find groups of axes that are reshaped
// For example, (3,16,8,2)#NCHW -> Reshape -> (1,48,4,4)#NCHW
// The result will be {{N, C}, {H, W}}
SmallVector<SmallVector<Dim>> getReshapedAxes(ShapeRef inShape, ShapeRef outShape, const DimsOrder& order) {
    SmallVector<SmallVector<Dim>> reshapedAxes;
    SmallVector<Dim> reshapedGroup;
    int64_t inProduct = 1;
    int64_t outProduct = 1;
    bool startMatch = false;
    for (const auto& dim : order.toPermutation()) {
        auto inDimSize = inShape[dim];
        auto outDimSize = outShape[dim];
        if (startMatch) {
            // Keep iterating dims until the total element sizes are the same, and those dims form a group
            reshapedGroup.push_back(dim);
            inProduct *= inDimSize;
            outProduct *= outDimSize;
            if (inProduct == outProduct) {
                reshapedAxes.push_back(reshapedGroup);
                startMatch = false;
            }
        } else {
            // Start matching if dim sizes are different
            if (inDimSize != outDimSize) {
                reshapedGroup.assign({dim});
                inProduct = inDimSize;
                outProduct = outDimSize;
                startMatch = true;
            }
        }
    }
    VPUX_THROW_UNLESS(startMatch == false, "Reshape's input {0} and output {1} are not matched", inShape, outShape);

    return reshapedAxes;
}

// Reorder can only propagate if the order of Reshape's reshaped axes are kept the same
// For exmaple, (1,16,8,2)#NHWC -> Reorder -> (1,16,8,2)#NCHW -> Reshape -> (1,16,4,4)#NCHW
// Only H & W are reshaped and their relative order stays the same
// Thus, the Reorder can be propagated through the Reshape like below:
// (1,16,8,2)#NHWC -> Reshape -> (1,16,4,4)#NHWC -> Reorder -> (1,16,4,4)#NCHW
bool isReshapeInImmutableGroup(const SmallVector<SmallVector<Dim>>& reshapedAxes, const DimsOrder& order) {
    for (const auto& group : reshapedAxes) {
        auto dimIter = group.begin();
        auto prevPos = order.dimPos(*dimIter);
        while (++dimIter != group.end()) {
            auto curPos = order.dimPos(*dimIter);
            if (curPos - prevPos != 1) {
                return false;
            }
            prevPos = curPos;
        }
    }

    return true;
}

// Reorder could be propagated when irrelevant shapes are removed (shape=1) and
// the memory permutation of the reshaped tensor remains the same as before the Reorder
// We check two conditions:
// 1. If propagated new Reorder's MemShape is identical with original shapeChangeOp's output MemShape.
// 2. If propagated new Reorder's permutation is identical with original shapeChangeOp's output permutation.
//
// Eg1. (1,1,64,64)#NWHC ->Reorder-> (1,1,64,64)#NCHW ->ShapeChangeOp-> (1,64,64,1)#NCHW
//   MemShape: (1,64,64,1) ->Reorder-> (1,1,64,64) ->ShapeChangeOp-> (1,64,64,1)
//   NormMemShape: (0,1,2,0) ->Reorder-> (0,0,2,1) ->ShapeChangeOp-> (0,2,1,0)
//   MemShape if propagateReorder: (1,64,64,1) ->ShapeCast-> (1,1,64,64) ->Reorder-> (1,64,64,1)
//   NormMemShape if propagateReorder: (0,1,2,0) ->ShapeCast-> (0,0,1,2) ->Reorder-> (0,2,1,0)
//   return ture as (64,64) == (64,64) && (2,1) == (2,1)
//
// Eg2. (1,1,64,64)#NCWH ->Reorder-> (1,1,64,64)#NCHW ->ShapeChangeOp-> (64,64,1,1)#NCHW
//   MemShape: (1,1,64,64) ->Reorder-> (1,1,64,64) ->ShapeChangeOp-> (64,64,1,1)
//   NormMemShape: (0,0,1,2) ->Reorder-> (0,0,2,1) ->ShapeChangeOp-> (2,1,0,0)
//   MemShape if propagateReorder:  (1,1,64,64) ->ShapeCast-> (64,64,1,1) ->Reorder-> (64,64,1,1)
//   NormMemShape if propagateReorder: (0,0,1,2) ->ShapeCast-> (1,2,0,0) ->ReorderOp-> (1,2,0,0)
//   return false as (64,64) == (64,64) && (2,1) != (1,2)
bool isIdenticalMemShapeAndPermutation(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) {
    auto inMemShape = inType.getMemShape();
    const auto inOrder = inType.getDimsOrder();
    const auto newOutType = outType.changeDimsOrder(inOrder);
    auto newOutMemShape = newOutType.getMemShape();

    // return normalized memShape
    // eg1: (1,1,64,64)#NHWC -> (0,1,2,0)
    // eg2: (1,1,64,64)#NCHW -> (0,0,1,2)
    auto getNormalizedMemShape = [](MemShapeRef memShape) -> MemShape {
        SmallVector<int64_t> normalizedVec;
        int64_t nonTrivailNum = 1;
        for (auto& shape : memShape) {
            if (shape == 1) {
                normalizedVec.push_back(0);
            } else {
                normalizedVec.push_back(nonTrivailNum++);
            }
        }
        return MemShape(normalizedVec);
    };

    auto originalReorderPermutation = getPermutationFromOrders(inOrder, outType.getDimsOrder(), inType.getContext());
    // cal original normalized mem shape
    auto inNormMemShape = getNormalizedMemShape(inMemShape);
    auto origReorderNormMemShape = applyPerm(inNormMemShape, originalReorderPermutation);

    // get new reshape's normalized mem permutation
    auto newReshapeNormMemShape = getNormalizedMemShape(newOutMemShape);
    auto newOutNormMemShape = applyPerm(newReshapeNormMemShape, originalReorderPermutation);

    // ignore trivial dims
    origReorderNormMemShape.erase(std::remove(origReorderNormMemShape.begin(), origReorderNormMemShape.end(), 0),
                                  origReorderNormMemShape.end());
    newOutNormMemShape.erase(std::remove(newOutNormMemShape.begin(), newOutNormMemShape.end(), 0),
                             newOutNormMemShape.end());

    // ignore shape is one
    inMemShape.erase(std::remove(inMemShape.begin(), inMemShape.end(), 1), inMemShape.end());
    newOutMemShape.erase(std::remove(newOutMemShape.begin(), newOutMemShape.end(), 1), newOutMemShape.end());

    return inMemShape == newOutMemShape && origReorderNormMemShape == newOutNormMemShape;
}

// if not has immutable reshape axes and identical memshape, Reorder could still
// propagate through reshape when the memshape is continuous
// Example:
//   8000x256x1x1xf16#HNWC   -reorder1-> 8000x256x1x1xf16#NCHW -reshape-> 1x8000x16x16xf16#NCHW
//   Memshape (1,8000,1,256) -reorder1-> (8000,256,1,1)        -reshape-> (1,8000,16,16)
// The memshape keeps continuous, reorder could be transformed to permute and propagated as follow:
//   8000x256x1x1xf16#HNWC   -shapecast-> 8000x16x1x16xf16#HNWC -permute-> 1x8000x16x16xf16#NCHW
//   Memshape (1,8000,1,256) -shapecast-> (1,8000,16,16)        -permute-> (1,8000,16,16)
bool isContinuousMemShape(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) {
    const auto inMemShape = inType.getMemShape();
    const auto outMemShape = outType.getMemShape();

    // convert Memshape into Vector
    auto getMemShapeArray = [](MemShapeRef memShape) -> SmallVector<int64_t> {
        SmallVector<int64_t> memShapeVec(memShape.size(), 1);
        for (const auto ind : irange(memShape.size())) {
            memShapeVec[ind] = memShape[MemDim(ind)];
        }
        return memShapeVec;
    };

    // get the permutation of reorder
    const auto inOrder = inType.getDimsOrder();
    const auto outOrder = outType.getDimsOrder();
    const auto origReorderPermutation = getPermutationFromOrders(inOrder, outOrder, inType.getContext());

    // get the reassociationMap of in/out memshape
    auto inMemShapeVec = getMemShapeArray(inMemShape);
    auto outMemShapeVec = getMemShapeArray(outMemShape);
    auto reassociationMap = IE::getReassociationMap(inMemShapeVec, outMemShapeVec);

    // check if the perm is trivial and the in/out memshape could be reassociated
    // successfully (could be legally affined)
    return isTrivialPermute(inMemShape, origReorderPermutation) && mlir::succeeded(reassociationMap);
}

// Maintain the Reorder -> PermuteCast -> Reorder chain as it can later be reduced to a single operation
bool isMaintainPattern(mlir::Operation* op) {
    if (auto prevPermuteCastOp = op->getOperand(0).getDefiningOp<IE::PermuteCastOp>()) {
        if (auto prevReorderOp = prevPermuteCastOp.getOperand().getDefiningOp<IE::ReorderOp>()) {
            return true;
        }
    }

    return false;
}

template <class ConcreteOp>
mlir::LogicalResult ReorderWithShapeChange<ConcreteOp>::matchAndRewrite(ConcreteOp origReshapeOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    const auto ctx = origReshapeOp.getContext();
    const auto origReshapeInput = origReshapeOp->getOperand(0);

    // Propagate Reorder through Reshape only with pattern: Reorder -> Reshape -> Reorder
    // two Reorders could fuse together
    auto origReorderOp = origReshapeInput.template getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    auto outputQuantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(*(origReshapeOp->getResult(0).user_begin()));
    auto outputReorderOp = outputQuantizeCastOp != nullptr
                                   ? mlir::dyn_cast<IE::ReorderOp>(*(outputQuantizeCastOp.getOutput().user_begin()))
                                   : mlir::dyn_cast<IE::ReorderOp>(*(origReshapeOp->getResult(0).user_begin()));
    if (outputReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got Reorder at '{0}' -> {1} at '{2}' pair", origReorderOp->getLoc(), origReshapeOp->getName(),
               origReshapeOp->getLoc());

    const auto origReshapeOutput = origReshapeOp->getResult(0);
    const auto inOrder = DimsOrder::fromValue(origReorderOp.getInput());
    const auto outOrder = DimsOrder::fromAffineMap(origReorderOp.getDstOrder());
    const auto reshapeOutType = mlir::dyn_cast<vpux::NDTypeInterface>(origReshapeOutput.getType());
    const auto reshapeOutOrder = reshapeOutType.getDimsOrder();
    if (outOrder != reshapeOutOrder) {
        return matchFailed(_log.nest(), rewriter, origReshapeOp,
                           "Reshape's input order {0} and output order {1} are not the same", outOrder,
                           reshapeOutOrder);
    }

    // The coming logic will use shapecast which could not infer the axis info for PerAxis quantize type.
    // So here we disable the case.
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(reshapeOutType.getElementType())) {
        return matchFailed(_log.nest(), rewriter, origReshapeOp, "Could not support PerAxis quantize type");
    }

    auto getMemShapeArray = [](mlir::Value val) -> SmallVector<int64_t> {
        const auto memShape = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType()).getMemShape();
        SmallVector<int64_t> memShapeVec(memShape.size(), 1);
        for (const auto ind : irange(memShape.size())) {
            memShapeVec[ind] = memShape[MemDim(ind)];
        }
        return memShapeVec;
    };

    const auto reshapeOutMemShape = getMemShapeArray(origReshapeOutput);
    const auto reshapeInShape = getShape(origReshapeInput);
    const auto reshapeOutShape = getShape(origReshapeOutput);
    const auto reshapedAxes = getReshapedAxes(reshapeInShape, reshapeOutShape, outOrder);
    const auto origReorderInType = origReorderOp.getInput().getType();
    const auto origReshapeOutType = origReshapeOutput.getType();
    if (isReshapeInImmutableGroup(reshapedAxes, inOrder) ||
        isIdenticalMemShapeAndPermutation(origReorderInType, origReshapeOutType)) {
        auto shapeAttr = getIntArrayAttr(ctx, reshapeOutShape);
        auto shapeCastOp = rewriter.create<IE::ShapeCastOp>(takeOpLoc(origReshapeOp, "shapecast_in"),
                                                            origReorderOp.getInput(), shapeAttr);
        if (outputQuantizeCastOp != nullptr) {
            auto newQuantizeCastOp = rewriter.create<IE::QuantizeCastOp>(
                    takeOpLoc(outputQuantizeCastOp, "quantize_out"), shapeCastOp.getResult(),
                    outputQuantizeCastOp.getDstElemTypeAttr());
            auto newReorderOp = rewriter.replaceOpWithNewOp<IE::ReorderOp>(
                    outputQuantizeCastOp, newQuantizeCastOp.getOutput(), origReorderOp.getDstOrderAttr());
            extendOpLoc(newReorderOp, "reorder");
        }
        auto newReorderOp = rewriter.replaceOpWithNewOp<IE::ReorderOp>(origReshapeOp, shapeCastOp.getResult(),
                                                                       origReorderOp.getDstOrderAttr());
        extendOpLoc(newReorderOp, "reorder");
        return mlir::success();
    } else if (isContinuousMemShape(origReorderInType, origReshapeOutType)) {
        const auto inputOrder =
                mlir::dyn_cast<vpux::NDTypeInterface>(origReorderOp.getInput().getType()).getDimsOrder();
        const auto outputShape = inputOrder.toLogicalOrder(MemShape(reshapeOutMemShape));
        const auto shapeAttr = getIntArrayAttr(ctx, outputShape);
        auto shapeCastOp = rewriter.create<IE::ShapeCastOp>(takeOpLoc(origReshapeOp, "shapecast_in"),
                                                            origReorderOp.getInput(), shapeAttr);

        const auto outputOrder = mlir::dyn_cast<vpux::NDTypeInterface>(origReshapeOutput.getType()).getDimsOrder();
        const auto outputOrderAttr = mlir::AffineMapAttr::get(outputOrder.toAffineMap(ctx));
        const auto memPermAttr =
                mlir::AffineMapAttr::get(DimsOrder::fromNumDims(outputOrder.numDims()).toAffineMap(ctx));
        auto permuteOp = rewriter.create<IE::PermuteCastOp>(takeOpLoc(origReshapeOp, "permute_out"),
                                                            shapeCastOp.getResult(), outputOrderAttr, memPermAttr);

        rewriter.replaceOp(origReshapeOp, permuteOp.getOutput());
        return mlir::success();
    }

    return matchFailed(_log.nest(), rewriter, origReshapeOp,
                       "The orders of reshaped axes {0} are different in input order {1} and output order {2}, and the "
                       "shape change op is not a trivial mem reshape ",
                       reshapedAxes, inOrder, outOrder);
}

//
// ReorderWithSubView
//

class ReorderWithSubView final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    ReorderWithSubView(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("ReorderWithSubView");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SliceOp origSubViewOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithSubView::matchAndRewrite(IE::SliceOp origSubViewOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto origReorderOp = origSubViewOp.getSource().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    if (isMaintainPattern(origReorderOp.getOperation())) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> subview at '{1}' pair", origReorderOp->getLoc(), origSubViewOp->getLoc());

    auto getUserOp = [](mlir::Operation* op) -> mlir::Operation* {
        mlir::Operation* user = op;
        while (mlir::isa<IE::PermuteCastOp, IE::AffineReshapeOp>(user) && user->hasOneUse()) {
            user = *user->getUsers().begin();
        }

        return user;
    };

    // In case "ReorderOp + SliceOp", if ReorderOp has no permutation for last dim (For example
    // affine_map<(d0, d2, d3, d1, d4) -> (d0, d1, d2, d3, d4)>), and the SliceOp output shape size for last dim
    // is 1 (For example, d0xd1xd2xd3x1), then the new Reorder <(d0, d2, d3, d1, 1) -> (d0, d1, d2, d3, 1)>
    // after swap will make it worse from performance perspective as inefficient DMA
    if (!origReorderOp.getResult().hasOneUse()) {
        bool allSlicesUsers = true;
        bool benefitToSwap = true;
        bool hasReorderUser = false;
        for (auto* reorderUser : llvm::make_early_inc_range(origReorderOp->getUsers())) {
            auto reorderUserSliceOp = mlir::dyn_cast<IE::SliceOp>(reorderUser);
            if (reorderUserSliceOp == nullptr) {
                allSlicesUsers = false;
                break;
            }

            auto reorderInputDimsOrder = DimsOrder::fromValue(origReorderOp.getInput());
            auto reorderOutputDimsOrder = DimsOrder::fromValue(origReorderOp.getOutput());
            auto reorderInputPerm = reorderInputDimsOrder.toPermutation();
            auto reorderOutputPerm = reorderOutputDimsOrder.toPermutation();
            auto inputPermEnd = (reorderInputPerm.end() - 1)->ind();
            auto outputPermEnd = (reorderOutputPerm.end() - 1)->ind();
            const auto sliceShape = parseIntArrayAttr<int64_t>(reorderUserSliceOp.getStaticSizes());

            if (inputPermEnd == outputPermEnd && sliceShape[outputPermEnd] == 1) {
                benefitToSwap = false;
            }

            auto sliceUser = *reorderUser->getUsers().begin();
            if (reorderUser->hasOneUse() && sliceUser != nullptr && mlir::isa<IE::ReorderOp>(getUserOp(sliceUser))) {
                // Swap if reorder can be fused with reorder post slice operation
                hasReorderUser = true;
            }
        }

        if (allSlicesUsers && !benefitToSwap && !hasReorderUser) {
            return mlir::failure();
        }
    }

    auto newSubViewOp =
            rewriter.create<IE::SliceOp>(takeOpLoc(origSubViewOp, "slice_in"), origReorderOp.getInput(),
                                         origSubViewOp.getStaticOffsetsAttr(), origSubViewOp.getStaticSizesAttr());
    extendOpLoc(newSubViewOp, "{0}_{1}", origSubViewOp.getStaticOffsets(), origSubViewOp.getStaticSizes());
    auto newLoc = takeOpLoc(origReorderOp, "reorder_out_{0}_{1}", origSubViewOp.getStaticOffsets(),
                            origSubViewOp.getStaticSizes());
    rewriter.replaceOpWithNewOp<IE::ReorderOp>(origSubViewOp, newSubViewOp.getResult(), origReorderOp.getDstOrderAttr())
            ->setLoc(newLoc);
    return mlir::success();
}

//
// ReorderWithTile
//

class ReorderWithTile final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    ReorderWithTile(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TileOp>(ctx), _log(log) {
        setDebugName("ReorderWithTile");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp origTileOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithTile::matchAndRewrite(IE::TileOp origTileOp, mlir::PatternRewriter& rewriter) const {
    if (isBeneficialReorderFuse(origTileOp)) {
        return mlir::failure();
    }

    auto origReorderOp = mlir::dyn_cast<IE::ReorderOp>(*(origTileOp.getOutput().user_begin()));
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }
    // Avoid loop rewriting with ReorderWithLayer
    if (mlir::isa<IE::ReorderOp>(*(origReorderOp.getOutput().user_begin()))) {
        return mlir::failure();
    }

    if (isMaintainPattern(origReorderOp.getOperation())) {
        return mlir::failure();
    }

    _log.trace("Got tile at '{0}' -> reorder at '{1}' pair", origTileOp->getLoc(), origReorderOp->getLoc());

    if (!isBeneficialSwitch(origTileOp, DimsOrder::fromValue(origReorderOp.getInput()),
                            DimsOrder::fromValue(origReorderOp.getOutput()))) {
        return mlir::failure();
    }

    auto newReorderOp = rewriter.create<IE::ReorderOp>(takeOpLoc(origReorderOp, "reorder_in"), origTileOp.getInput(),
                                                       origReorderOp.getDstOrderAttr());

    auto outputType = mlir::cast<vpux::NDTypeInterface>(origTileOp.getOutput().getType());
    auto newOutputType = outputType.changeDimsOrder(DimsOrder::fromAffineMap(newReorderOp.getDstOrder()));

    auto tileOp = rewriter.replaceOpWithNewOp<IE::TileOp>(origReorderOp, newOutputType, newReorderOp.getOutput(),
                                                          origTileOp.getRepeats(), origTileOp.getRepeatsValuesAttr());
    extendOpLoc(tileOp, "tile");

    return mlir::success();
}

//
//  The beneficial pattern:
//
//     input               input
//       |                   |
//     Reorder             Expand
//       |                   |
//     Expand   ==>        Reorder
//       |                   |
//     Slice(s)            Slice(s)
//       |                   |
//     Reorder(s)          Reorder(s)
//       |                   |
//     output              output
//
//  It's worth to swap parent Reorder and Expand,  the swapped Reorder will be handled by follow-up optimizations.
//

bool isBeneficialToSwapExpandReorders(IE::ExpandOp origExpandOp, mlir::Operation* maybeReorderOp) {
    auto origReorderOp = mlir::dyn_cast<IE::ReorderOp>(maybeReorderOp);
    if (origReorderOp == nullptr) {
        return false;
    }
    // If Reorder is not Trivial Permute, will swap
    if (!isTrivialReorder(origReorderOp)) {
        return true;
    }

    if (mlir::isa<mlir::BlockArgument>(origExpandOp.getInput())) {
        return false;
    }

    const auto expandOutput = origExpandOp.getOutput();

    SmallVector<IE::SliceOp> slices;

    for (auto userOp : expandOutput.getUsers()) {
        auto maybeSlice = mlir::dyn_cast_or_null<IE::SliceOp>(*userOp);
        if (maybeSlice == nullptr) {
            return false;
        }
        slices.push_back(maybeSlice);
    }

    if (slices.empty()) {
        return false;
    }

    SmallVector<mlir::Value> reorders;
    for (auto& userOp : slices) {
        auto sliceOutput = userOp.getResult();
        if (!sliceOutput.hasOneUse()) {
            return false;
        }
        auto maybeReorderOp = mlir::dyn_cast_or_null<IE::ReorderOp>(*sliceOutput.getUsers().begin());
        if (maybeReorderOp == nullptr) {
            return false;
        }
        reorders.push_back(maybeReorderOp);
    }

    return !reorders.empty();
}

//
// ReorderWithExpandSlice
//

//  The beneficial pattern:
//
//       input(Not Reorder)   input
//          |                   |
//        Expand              Reorder
//          |                   |
//        Slice(s)   ==>      Expand
//          |                   |
//       Reorder(s)           Slice(s)
//          |                   |
//        output              Output
//  This is a cleanup pattern.
//  Move Reorder before Expand, it is beneficial if size of original Reorder(s) total size is larger than
//  new inserted Reorder.
//  Example:
//  (1,1,1,32032) -> Expand -> (1,16,1,32032) -> Slice -> (1,16,1,31995) -> Reorder
//                                            -> Slice -> (1,16,1,31995) -> Reorder
//  After this pass, the pattern:
//  (1,1,1,32032) -> Reorder -> Expand -> Slice
//                                     -> Slice
//  The size of the Reorders reduce from 16x31995x2 to 32032.

class ReorderWithExpandSlice final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    ReorderWithExpandSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("ReorderWithExpandSlice");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origExpandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithExpandSlice::matchAndRewrite(IE::ExpandOp origExpandOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Expand at '{0}' ", origExpandOp->getLoc());
    if (mlir::isa_and_nonnull<IE::ReorderOp>(origExpandOp.getInput().getDefiningOp())) {
        return mlir::failure();
    }

    const auto expandOutput = origExpandOp.getOutput();
    SmallVector<IE::SliceOp> slices;

    for (auto userOp : expandOutput.getUsers()) {
        auto maybeSlice = mlir::dyn_cast_or_null<IE::SliceOp>(*userOp);
        if (maybeSlice == nullptr) {
            return mlir::failure();
        }
        slices.push_back(maybeSlice);
    }

    if (slices.empty()) {
        return mlir::failure();
    }

    SmallVector<IE::ReorderOp> reorders;
    for (auto& userOp : slices) {
        auto sliceOutput = userOp.getResult();
        if (!sliceOutput.hasOneUse()) {
            return mlir::failure();
        }
        auto maybeReorderOp = mlir::dyn_cast_or_null<IE::ReorderOp>(*sliceOutput.getUsers().begin());
        if (maybeReorderOp == nullptr) {
            return mlir::failure();
        }
        reorders.push_back(maybeReorderOp);
    }

    if (reorders.empty() || slices.size() != reorders.size()) {
        return mlir::failure();
    }

    int64_t subReordersTotalSize = 0;
    // check all the reorder op have the same input and output DimsOrder
    auto reorderOutputDimsOrder = DimsOrder::fromValue(reorders[0].getOutput());
    for (auto& reorderOp : reorders) {
        auto reorderOutputDimsOrderLocal = DimsOrder::fromValue(reorderOp.getOutput());
        if (reorderOutputDimsOrderLocal != reorderOutputDimsOrder) {
            return mlir::failure();
        }
        auto reorderOpOutputType = mlir::cast<vpux::NDTypeInterface>(reorderOp.getOutput().getType());
        subReordersTotalSize += reorderOpOutputType.getTotalAllocSize().count();
    }

    // Only benefit the first inserted Reorder size smaller than subslice total size.
    auto origExpandInputType = mlir::cast<vpux::NDTypeInterface>(origExpandOp.getInput().getType());
    auto origExpandInputSize = origExpandInputType.getTotalAllocSize().count();
    if (subReordersTotalSize <= origExpandInputSize) {
        _log.trace("Expand input Size: '{0}' larger than total size of Reorder(s): '{1}'. ", origExpandInputSize,
                   subReordersTotalSize);
        return mlir::failure();
    }

    auto newReorderOp = rewriter.create<IE::ReorderOp>(takeOpLoc(origExpandOp, "reorder_in"), origExpandOp.getInput(),
                                                       reorders[0].getDstOrderAttr());
    auto newExpandOp = rewriter.create<IE::ExpandOp>(takeOpLoc(origExpandOp, "expand_out"), newReorderOp.getOutput(),
                                                     origExpandOp.getPadsBeginAttr(), origExpandOp.getPadsEndAttr());

    for (size_t index = 0; index < slices.size(); index++) {
        auto subSlice = slices[index];
        auto subReorder = reorders[index];
        auto newSliceOp = rewriter.create<IE::SliceOp>(takeOpLoc(subSlice, "slice_out"),
                                                       subReorder.getOutput().getType(), newExpandOp.getOutput(),
                                                       subSlice.getStaticOffsetsAttr(), subSlice.getStaticSizesAttr());
        rewriter.replaceOp(subSlice, newSliceOp.getOutputs());
        subReorder.replaceAllUsesWith(subReorder.getInput());
        rewriter.eraseOp(subReorder);
    }
    return mlir::success();
}

//
// ReorderWithAffineReshapeTile
//

//  The beneficial pattern:
//
//       Reorder
//          |
//     AffineReshape         ShapeCast
//          |                   |
//         Tile        ==>     Tile
//          |                   |
//     AffineReshape         ShapeCast
//          |
//        Reorder
//  This is a cleanup pattern.
//  It's worth to eliminate input Reorder and output Reorder if the Tile Axis dim position is not changed or in high
//  dim. For tileOp tile data in high dim is more efficient than tile date in low dim. For example tileOp 1x16x1x1 ->
//  1x16x100x1 (repeatsValues = [1,1,100,1]), NHWC is more efficient than NCHW layout, because NHWC will tile data in
//  the higher dim (H of NHWC), the tileOp will convert to no stride DMA. but for NCHW will tile data in the lower dim
//  (H and W of NCHW), the tileOp will convert to stride DMA which is low efficient.
//

class ReorderWithAffineReshapeTile final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    ReorderWithAffineReshapeTile(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::TileOp>(ctx), _log(log) {
        setDebugName("ReorderWithAffineReshapeTile");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp origTileOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithAffineReshapeTile::matchAndRewrite(IE::TileOp origTileOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Tile Op at '{0}' ", origTileOp->getLoc());
    const auto ctx = rewriter.getContext();
    // step 1: check input/output pattern
    auto inputReshapeOp = origTileOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    if (inputReshapeOp == nullptr || !inputReshapeOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto inputReorder = inputReshapeOp.getInput().getDefiningOp<IE::ReorderOp>();
    if (inputReorder == nullptr || !inputReorder.getOutput().hasOneUse()) {
        return mlir::failure();
    }
    if (!origTileOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }
    auto outputReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(*(origTileOp.getOutput().user_begin()));
    if (outputReshapeOp == nullptr) {
        return mlir::failure();
    }
    if (!outputReshapeOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }
    auto outputReorder = mlir::dyn_cast<IE::ReorderOp>(*(outputReshapeOp.getOutput().user_begin()));
    if (outputReorder == nullptr) {
        return mlir::failure();
    }

    // step 2: check repeatsValue in new DimsOrder is efficient.
    const auto repeatsValue = parseIntArrayAttr<int64_t>(origTileOp.getRepeatsValuesAttr());
    const auto greaterThanOne = [](auto dim) {
        return dim > 1;
    };
    // not 1 dim repeats
    if (llvm::count_if(repeatsValue, greaterThanOne) != 1) {
        return mlir::failure();
    }

    // Find original TileOp Axis index.
    auto tileoOpAxis = llvm::find_if(repeatsValue, greaterThanOne);
    auto tileoOpAxisIndex = std::distance(repeatsValue.begin(), tileoOpAxis);

    // Check the changed TileOp Axis dim
    // For example, the original TileOp repeats [1, 1, 83, 1], the Axis loc is 2
    // For NCHW, the Axis locates on H of NCHW (origAxisLoc is 2)
    // If the target DimsOrder is NHWC, the Axis locates on H of NHWC (targetAxisLoc is 1)
    // This check is to ensure the TileOp effciency is not worse, and pattern will benefit
    // from elimination of 2 ReorderOp.
    auto origOrder = DimsOrder::fromValue(origTileOp.getOutput());
    auto targetOrder = DimsOrder::fromValue(inputReorder.getInput());
    auto origAxisLoc = origOrder.dimPos(Dim(tileoOpAxisIndex));
    auto targetAxisLoc = targetOrder.dimPos(Dim(tileoOpAxisIndex));

    if (targetAxisLoc > origAxisLoc) {
        return mlir::failure();
    }

    // step 3: clean up input/output reorders.
    auto inputReshapeOutputShape = getShape(inputReshapeOp.getOutput());
    auto inputShapeAttr = getIntArrayAttr(ctx, inputReshapeOutputShape);
    auto inputShapeCastOp = rewriter.create<IE::ShapeCastOp>(takeOpLoc(origTileOp, "shapecast_in"),
                                                             inputReorder.getInput(), inputShapeAttr);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origTileOp.getOutput().getType());
    auto newOutputType = outputType.changeDimsOrder(targetOrder);

    auto newTileOp =
            rewriter.create<IE::TileOp>(takeOpLoc(origTileOp, "as_tile"), newOutputType, inputShapeCastOp.getResult(),
                                        nullptr, origTileOp.getRepeatsValuesAttr());

    auto outputReshapeOutputShape = getShape(outputReshapeOp.getOutput());
    auto outputShapeAttr = getIntArrayAttr(ctx, outputReshapeOutputShape);
    auto outputShapeCastOp = rewriter.create<IE::ShapeCastOp>(takeOpLoc(origTileOp, "shapecast_out"),
                                                              newTileOp.getOutput(), outputShapeAttr);

    rewriter.replaceOp(outputReorder, outputShapeCastOp);
    return mlir::success();
}

//
// ReorderWithSplit
//

class ReorderWithSplit final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    ReorderWithSplit(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SplitOp>(ctx), _log(log) {
        setDebugName("ReorderWithSplit");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp origSplitOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithSplit::matchAndRewrite(IE::SplitOp origSplitOp, mlir::PatternRewriter& rewriter) const {
    if (origSplitOp.getAxis() != nullptr) {
        return mlir::failure();
    }

    auto origReorderOp = origSplitOp.getInput().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> Split at '{1}' pair", origReorderOp->getLoc(), origSplitOp->getLoc());

    const auto inOrder = DimsOrder::fromValue(origReorderOp.getInput());
    const auto outOrder = DimsOrder::fromValue(origReorderOp.getOutput());
    const auto dstOrderAttr = origReorderOp.getDstOrderAttr();
    auto newSplit = rewriter.create<IE::SplitOp>(takeOpLoc(origSplitOp, "as_split"), origReorderOp.getInput(),
                                                 origSplitOp.getAxis(), origSplitOp.getNumSplitsAttr(),
                                                 origSplitOp.getAxisValueAttr());

    SmallVector<mlir::Value> newOutputs;
    newOutputs.reserve(origSplitOp.getOutputs().size());

    for (auto res : origSplitOp.getOutputs()) {
        if (res.getUses().empty()) {
            newOutputs.push_back(newSplit.getResult(res.getResultNumber()));
            continue;
        }

        _log.trace("Insert reorder '{0}' -> '{1}' for Split output at idx='{2}'.", inOrder, outOrder,
                   res.getResultNumber());
        auto newLoc = takeOpLoc(origSplitOp, "reorder_{0}", res.getResultNumber());
        auto reorder = rewriter.create<IE::ReorderOp>(newLoc, newSplit.getResult(res.getResultNumber()), dstOrderAttr);
        newOutputs.push_back(reorder);
    }

    _log.trace("Replace Split with new output values.");
    rewriter.replaceOp(origSplitOp, newOutputs);

    return mlir::success();
}

//
// ReorderWithConcat
//

void replaceChildReorderWithNewConcat(IE::ConcatOp& origConcatOp, IE::ConcatOp& newConcatOp,
                                      mlir::PatternRewriter& rewriter, Logger log) {
    auto outputConcat = origConcatOp.getOutput();
    auto childReorderOp = mlir::dyn_cast_or_null<IE::ReorderOp>(*outputConcat.getUsers().begin());

    auto newOutputConcat = newConcatOp.getOutput();
    vpux::changeDimsOrder(newOutputConcat, DimsOrder::fromValue(childReorderOp.getOutput()), log);
    childReorderOp.getOutput().replaceAllUsesExcept(newOutputConcat,
                                                    llvm::SmallPtrSet<mlir::Operation*, 1>{newConcatOp});

    rewriter.eraseOp(childReorderOp);
    rewriter.eraseOp(origConcatOp);
}

//
//  The beneficial pattern:
//
//    input1    input2 ...                                           input3
//       |         |                                                    |
//    Reorder  Reorder ...input3                   input1  input2 ...Reorder
//           \     |     /                            \     |     /
//               Concat                 ==>               Concat
//                 |                                        |
//               Reorder                                  output
//                 |
//               output
//
//  It's worth to swap parent Reorder and Concat if the child Reorder can be eliminated.
//  Two cases supported:
//    1. All the inputs of concat are Reorders and share the same input layout
//    2. When the concat has only one Reorder user, and the number of Reorders is reduced after propagation
//       This case requires the existence of an output Reorder because it's the largest and the benefit of
//       reducing input Reorders may not offset the cost of introducing a new output Reorder.

struct BeneficialConcatInfo {
    DimsOrder newConcatLayout;
    bool hasOutputReorderWithTargetLayout;
};

std::optional<BeneficialConcatInfo> getBeneficialConcatLayout(IE::ConcatOp& origConcatOp) {
    const auto outputConcat = origConcatOp.getOutput();
    const auto origConcatLayout = DimsOrder::fromValue(outputConcat);
    auto outputReorder = mlir::dyn_cast_or_null<IE::ReorderOp>(*outputConcat.getUsers().begin());
    bool hasOnlyOneReorderUser = outputConcat.hasOneUse() && outputReorder != nullptr;

    // Collect all layouts from input reorders' inputs and output reorder's output
    DenseMap<uint64_t, size_t> layoutCounts;

    // Count inputs' layouts
    for (const auto& arg : origConcatOp.getInputs()) {
        // Skip Const inputs
        if (mlir::isa_and_nonnull<Const::DeclareOp>(arg.getDefiningOp())) {
            continue;
        }

        DimsOrder inputLayout;
        if (auto argReorderOp = arg.getDefiningOp<IE::ReorderOp>()) {
            inputLayout = DimsOrder::fromValue(argReorderOp.getInput());
        } else {
            inputLayout = DimsOrder::fromValue(arg);
        }
        layoutCounts[inputLayout.code()]++;
    }

    // Beneficial if all input layouts are the same
    if (layoutCounts.size() == 1) {
        auto mostCommonCode = layoutCounts.begin()->first;
        if (mostCommonCode == origConcatLayout.code()) {
            // Optimal layout already
            return std::nullopt;
        }
        if (hasOnlyOneReorderUser) {
            const auto outputLayout = DimsOrder::fromValue(outputReorder.getOutput());
            if (outputLayout.code() == mostCommonCode) {
                return BeneficialConcatInfo{DimsOrder::fromCode(mostCommonCode), true};
            }
        }
        return BeneficialConcatInfo{DimsOrder::fromCode(mostCommonCode), false};
    }

    if (hasOnlyOneReorderUser) {
        // reorder with eltwise group conv could be optimized by ReorderWithGroupConv
        auto user = *outputReorder.getOutput().getUsers().begin();
        auto dwConv = mlir::dyn_cast_or_null<IE::GroupConvolutionOp>(user);
        if (dwConv && DoesReorderWithGroupConvPatternMatch(dwConv)) {
            return std::nullopt;
        }
    } else {
        return std::nullopt;
    }

    // Count output layout
    const auto outputLayout = DimsOrder::fromValue(outputReorder.getOutput());
    layoutCounts[outputLayout.code()]++;

    // Find the most common layout
    uint64_t mostCommonCode = 0;
    size_t maxCount = 0;
    for (const auto& [code, count] : layoutCounts) {
        if (count > maxCount || (count == maxCount && code == outputLayout.code())) {
            maxCount = count;
            mostCommonCode = code;
        }
    }

    if (mostCommonCode == origConcatLayout.code()) {
        // Optimal layout already
        return std::nullopt;
    }

    // Beneficial if the number of Reorders is reduced after propagation
    const auto origCount = layoutCounts[origConcatLayout.code()];
    if (maxCount > origCount) {
        if (outputLayout.code() == mostCommonCode) {
            return BeneficialConcatInfo{DimsOrder::fromCode(mostCommonCode), true};
        }
        return BeneficialConcatInfo{DimsOrder::fromCode(mostCommonCode), false};
    }

    return std::nullopt;
}

class ReorderWithConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ReorderWithConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("ReorderWithConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origConcatOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithConcat::matchAndRewrite(IE::ConcatOp origConcatOp,
                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Concat at '{1}' ", getDebugName(), origConcatOp->getLoc());

    auto beneficialInfo = getBeneficialConcatLayout(origConcatOp);
    if (!beneficialInfo.has_value()) {
        _log.nest().trace("No beneficial layout found");
        return mlir::failure();
    }

    const auto targetLayout = beneficialInfo->newConcatLayout;
    const auto hasOutputReorderWithTargetLayout = beneficialInfo->hasOutputReorderWithTargetLayout;
    _log.nest().trace("Beneficial layout '{0}' found, hasOutputReorder '{1}'", targetLayout,
                      hasOutputReorderWithTargetLayout);

    SmallVector<mlir::Value> initialInputs;
    initialInputs.reserve(origConcatOp.getInputs().size());
    SmallVector<size_t> indexNeedReorder;

    auto constNum = 0;
    mlir::Operation* origReorderOp = nullptr;

    for (const auto& it : origConcatOp.getInputs() | indexed) {
        auto arg = it.value();
        auto argReorderOp = arg.getDefiningOp<IE::ReorderOp>();
        if (argReorderOp == nullptr) {
            indexNeedReorder.push_back(it.index());
            if (auto constOp = arg.getDefiningOp<Const::DeclareOp>()) {
                initialInputs.push_back(constOp.getOutput());
                constNum++;
                continue;
            }

            initialInputs.push_back(arg);
            continue;
        }

        origReorderOp = argReorderOp.getOperation();
        const auto argOrder = DimsOrder::fromValue(argReorderOp.getInput());

        // Track inputs that need reordering to target layout
        if (argOrder != targetLayout) {
            indexNeedReorder.push_back(it.index());
        }
        initialInputs.push_back(argReorderOp.getInput());
    }

    // To avoid affecting multiple branches optimization with reOrders before concat
    // Just skip only one non-const reorder input cases with the reorder-permutecast-reorder pattern
    if ((origConcatOp.getNumOperands() - constNum == 1) && origReorderOp) {
        if (isMaintainPattern(origReorderOp)) {
            _log.nest().trace("Got MaintainPattern");
            return mlir::failure();
        }
    }

    // Insert reorders for ConstOps and inputs with different layout
    for (auto index : indexNeedReorder) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfterValue(initialInputs[index]);
        auto reorderOut = rewriter.createOrFold<IE::ReorderOp>(takeOpLoc(origConcatOp, "reorder_in{0}", index),
                                                               initialInputs[index],
                                                               targetLayout.toAffineMap(rewriter.getContext()));
        initialInputs[index] = reorderOut;
    }

    auto newConcat = rewriter.create<IE::ConcatOp>(takeOpLoc(origConcatOp, "concat_out"), initialInputs,
                                                   origConcatOp.getPerAxisAttr(), origConcatOp.getStaticOffsetsAttr());

    if (hasOutputReorderWithTargetLayout) {
        // Output reorder's output layout matches target, can eliminate child reorder
        replaceChildReorderWithNewConcat(origConcatOp, newConcat, rewriter, _log);
    } else {
        // No output reorder, or the output reorder's output layout doesn't match target, need to add new reorder
        const auto originalConcatOrder = DimsOrder::fromValue(origConcatOp.getOutput());
        auto reorderOp =
                rewriter.replaceOpWithNewOp<IE::ReorderOp>(origConcatOp, origConcatOp.getType(), newConcat.getOutput(),
                                                           originalConcatOrder.toAffineMap(origConcatOp.getContext()));
        extendOpLoc(reorderOp, "reorder_out");
    }

    return mlir::success();
}

//
// ReorderWithQuantCast
//

class ReorderWithQuantCast final : public mlir::OpRewritePattern<IE::QuantizeCastOp> {
public:
    ReorderWithQuantCast(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeCastOp>(ctx), _log(log) {
        setDebugName("ReorderWithQuantCast");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeCastOp origQuantCastOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithQuantCast::matchAndRewrite(IE::QuantizeCastOp origQuantCastOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto origReorderOp = origQuantCastOp.getInput().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> quantize cast at '{1}' pair", origReorderOp->getLoc(),
               origQuantCastOp->getLoc());

    auto newQuantCastOp = rewriter.create<IE::QuantizeCastOp>(
            takeOpLoc(origQuantCastOp, "as_qcast"), origReorderOp.getInput(), origQuantCastOp.getDstElemTypeAttr());

    auto newReorder = rewriter.replaceOpWithNewOp<IE::ReorderOp>(origQuantCastOp, newQuantCastOp.getOutput(),
                                                                 origReorderOp.getDstOrderAttr());
    extendOpLoc(newReorder, "reorder");
    return mlir::success();
}

//
// ReorderWithPermuteCast
//
//               input                                input
//                 |                                    |
//              Reorder                             PermuteCast
//                 |                                    |
//             PermuteCast            ==>            Reorder
//                 |                                    |
//              Reorder                              Reorder
//                 |                                    |
//               output                               output
//
// No benefit for case that the input is a NCE task as it could fuse mem permute
// Not swap for case that two reorder could not be eliminated and the user of nextReorder is concatOp
//

class ReorderWithPermuteCast final : public mlir::OpRewritePattern<IE::PermuteCastOp> {
public:
    ReorderWithPermuteCast(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteCastOp>(ctx), _log(log) {
        setDebugName("ReorderWithPermuteCast");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteCastOp origPermuteCastOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Infer the output logical layout of original permuteCast, which is the same as the new permuteCast
// For example, based on the permutecast output physical layout = NCDHW, and permutecast
// dstOrder: [d0, d4, d1, d2, d3], it could calculate the permutecast logical layout = NDHWC
DimsOrder inferPermuteCastOutLogicalLayout(const vpux::DimsOrder& origReorderDstOrder,
                                           mlir::AffineMap origPermuteCastDstOrder) {
    auto origPermuteCastOutPerm = origReorderDstOrder.toPermutation();
    const auto order = DimsOrder::fromAffineMap(origPermuteCastDstOrder);
    auto targetPermutation = order.toPermutation();
    auto sourcePermutation = order.toPermutation();

    for (auto pIt = origPermuteCastOutPerm.begin(); pIt != origPermuteCastOutPerm.end(); ++pIt) {
        auto dimPosTargetPermutation = origReorderDstOrder.dimPos(Dim(pIt->ind()));
        sourcePermutation[targetPermutation[dimPosTargetPermutation].ind()] = Dim(pIt->ind());
    }

    return DimsOrder::fromPermutation(sourcePermutation);
}

// Infer the dstOrder of the new permuteCast
// For example, If the new permutecast logical layout = NDHWC, and physical layout = NDHWC
// it could calculate the new permuteCast dstOrder: [d0, d1, d2, d3, d4]
DimsOrder inferPermuteCastDstOrder(IE::ReorderOp origReorderOp, IE::PermuteCastOp origPermuteCastOp,
                                   mlir::MLIRContext* ctx) {
    const auto origReorderDstOrder = DimsOrder::fromAffineMap(origReorderOp.getDstOrder());
    const auto origPermuteCastDstOrderAttr = origPermuteCastOp.getDstOrderAttr();

    auto logicalLayout = inferPermuteCastOutLogicalLayout(origReorderDstOrder, origPermuteCastDstOrderAttr.getValue());

    // Calculate the permutation of new permuteCast
    const auto outputOrder = vpux::DimsOrder::fromValue(origReorderOp.getInput());
    const auto memPerm = vpux::getPermutationFromOrders(logicalLayout, outputOrder, ctx);

    return DimsOrder::fromAffineMap(memPerm);
}

mlir::LogicalResult ReorderWithPermuteCast::matchAndRewrite(IE::PermuteCastOp origPermuteCastOp,
                                                            mlir::PatternRewriter& rewriter) const {
    auto origReorderOp = origPermuteCastOp.getInput().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    if (!origReorderOp.getResult().hasOneUse() || !origPermuteCastOp.getResult().hasOneUse()) {
        return mlir::failure();
    }

    auto nextReorderOp = mlir::dyn_cast<IE::ReorderOp>(*origPermuteCastOp.getResult().getUsers().begin());
    if (nextReorderOp == nullptr) {
        return mlir::failure();
    }

    // If Reorder is Trivial Permute, will not swap
    if (isTrivialReorder(origReorderOp)) {
        return mlir::failure();
    }

    const auto origOutOrder = DimsOrder::fromValue(origPermuteCastOp.getOutput());
    const auto numDims = checked_cast<unsigned>(origOutOrder.numDims());
    const auto identityMap = mlir::AffineMap::getMinorIdentityMap(numDims, numDims, rewriter.getContext());
    if (origPermuteCastOp.getMemPerm() != identityMap) {
        return mlir::failure();
    }

    // No benefit for case that NCE tasks could fuse mem permute
    auto layerWithPermute = origReorderOp.getInput().getDefiningOp<IE::LayerWithPermuteInterface>();
    // This condition cause perf regression for InterpolateOp
    if (layerWithPermute != nullptr && layerWithPermute.isSupportedPermutation(origReorderOp) &&
        !mlir::isa_and_present<IE::InterpolateOp>(layerWithPermute)) {
        return mlir::failure();
    }

    // Infer the DstOrder of the new permuteCast
    auto newDstOrder = inferPermuteCastDstOrder(origReorderOp, origPermuteCastOp, rewriter.getContext());
    const auto newDstOrderAttr = mlir::AffineMapAttr::get(newDstOrder.toAffineMap(rewriter.getContext()));
    auto newMemPermAttr = origPermuteCastOp.getMemPermAttr();

    // Not swap in case that the two reorder could not be eliminated and the nextReorder's user is concatOp
    const auto nextReorderOutOrder = DimsOrder::fromAffineMap(nextReorderOp.getDstOrder());
    bool canEliminate = newDstOrder == nextReorderOutOrder;
    for (auto* nextReorderUser : llvm::make_early_inc_range(nextReorderOp.getResult().getUsers())) {
        auto nextConcat = mlir::dyn_cast<IE::ConcatOp>(nextReorderUser);
        if (nextConcat != nullptr && !canEliminate) {
            return mlir::failure();
        }
    }

    _log.trace("Got reorder at '{0}' -> permute cast at '{1}' pair", origReorderOp->getLoc(),
               origPermuteCastOp->getLoc());

    auto newPermuteCastOp = rewriter.create<IE::PermuteCastOp>(
            takeOpLoc(origPermuteCastOp, "as_pcast"), origReorderOp.getInput(), newDstOrderAttr, newMemPermAttr);

    rewriter.replaceOpWithNewOp<IE::ReorderOp>(origPermuteCastOp, newPermuteCastOp.getOutput(),
                                               origPermuteCastOp.getDstOrderAttr());
    return mlir::success();
}

//
// ReorderWithConvert
//

class ReorderWithConvert final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    ReorderWithConvert(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("ReorderWithConvert");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithConvert::matchAndRewrite(IE::ConvertOp convertOp,
                                                        mlir::PatternRewriter& rewriter) const {
    // Note that in this case we replace Convert -> Reorder with Reorder -> Convert
    // This is an opposite behavior, compared to other rewriters
    if (!convertOp.getResult().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, convertOp, "ConvertOp has more then one user");
    }

    auto origReorderOp = mlir::dyn_cast<IE::ReorderOp>(*convertOp.getResult().getUsers().begin());
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    const auto srcType = convertOp.getInput().getType();
    const auto dstElemType = convertOp.getDstElemType();
    auto isODUPermuteSupported = [](mlir::Type elemType) {
        return elemType.isF16();
    };
    if (getElemTypeSize(srcType) >= getElemTypeSize(dstElemType)) {
        return matchFailed(rewriter, convertOp, "Convert doesn't increase data size");
    }

    auto reorderIsNotPerformant =
            !isODUPermuteSupported(srcType.getElementType()) && isODUPermuteSupported(dstElemType);
    if (reorderIsNotPerformant) {
        return matchFailed(rewriter, convertOp,
                           "Reorder can not be converted into ODU permute after move before Convert");
    }

    auto newReorderOp = rewriter.create<IE::ReorderOp>(takeOpLoc(origReorderOp, "as_reorder"), convertOp.getInput(),
                                                       origReorderOp.getDstOrderAttr());

    auto newConvertOp = rewriter.replaceOpWithNewOp<IE::ConvertOp>(
            origReorderOp, origReorderOp.getType(), newReorderOp.getOutput(), convertOp.getDstElemTypeAttr());
    extendOpLoc(newConvertOp, "convert");

    return mlir::success();
}

mlir::FailureOr<mlir::Operation*> getConvertOrReturnOpConsumer(mlir::Operation* op) {
    std::function<bool(mlir::Operation*)> isConvertOrReturnOpFound = [](mlir::Operation* op) -> bool {
        return mlir::isa<IE::ConvertOp, mlir::func::ReturnOp>(op);
    };

    return IE::searchOpConsumers(op, isConvertOrReturnOpFound);
}

mlir::FailureOr<mlir::Operation*> getReturnOpConsumer(mlir::Operation* op) {
    std::function<bool(mlir::Operation*)> isReturnOpFound = [](mlir::Operation* op) -> bool {
        return mlir::isa<mlir::func::ReturnOp>(op);
    };

    return IE::searchOpConsumers(op, isReturnOpFound);
}

bool doesConvertOpIncreaseElemSize(IE::ConvertOp convertOp) {
    const auto srcType = convertOp.getInput().getType();
    const auto dstElemType = convertOp.getDstElemType();
    return getElemTypeSize(srcType) <= getElemTypeSize(dstElemType);
}

//
// ReorderWithLayer
//

class ReorderWithLayer final : public mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface> {
public:
    ReorderWithLayer(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface>(ctx), _log(log) {
        setDebugName("ReorderWithLayer");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayoutInfoOpInterface layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithLayer::matchAndRewrite(IE::LayoutInfoOpInterface layerOp,
                                                      mlir::PatternRewriter& rewriter) const {
    if (mlir::isa<IE::ReorderOp>(layerOp)) {
        return mlir::failure();
    }

    _log.trace("Got layer operation '{0}' at '{1}'", layerOp->getName(), layerOp->getLoc());

    auto argReorderOp = layerOp->getOperand(0).getDefiningOp<IE::ReorderOp>();
    if (argReorderOp == nullptr) {
        return mlir::failure();
    }
    // Skip reorder propagation for below cases:
    // 1.ReorderOp's consumers are pure view like ops with ReturnOp
    //   e.g.Reorder->AffineReshape->PermuteCast->QuantizeCast->ReturnOp
    // 2.ReorderOp's consumers are pure view like ops with ConvertOp and ReturnOp.
    //   Additionally ConvertOp is increasing element type size.
    //   e.g.Reorder->AffineReshape->ConvertOp(F16->F32)->AffineReshape->PermuteCast->QuantizeCast->ReturnOp
    // 3.ReorderOp's consumers is tileOp and the propagation will lead a low efficient layout for tileOp.
    if (auto tileOp = mlir::dyn_cast<IE::TileOp>(&layerOp)) {
        if (!isBeneficialReorderFuse(*tileOp)) {
            if (!isBeneficialSwitch(*tileOp, DimsOrder::fromValue(argReorderOp.getOutput()),
                                    DimsOrder::fromValue(argReorderOp.getInput()))) {
                return mlir::failure();
            }
        }
    }
    // Skip reorder propagation for below case to avoid PermuteOp back infer error:
    // AffineReshape (with changed data rank) -> MemPermute
    if (auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(&layerOp)) {
        auto affineInShape = getShape(affineReshapeOp->getInput());
        auto affineOutShape = getShape(affineReshapeOp->getOutput());
        if (affineInShape.size() != affineOutShape.size()) {
            if (auto memPermOp = mlir::dyn_cast<IE::MemPermuteOp>(*affineReshapeOp->getOutput().getUsers().begin())) {
                return mlir::failure();
            }
        }
    }
    auto getConsumerResult = getConvertOrReturnOpConsumer(argReorderOp);
    if (!mlir::failed(getConsumerResult)) {
        auto convertOrReturnOp = getConsumerResult.value();
        if (mlir::isa<mlir::func::ReturnOp>(convertOrReturnOp)) {
            return mlir::failure();
        } else if (mlir::isa<IE::ConvertOp>(convertOrReturnOp)) {
            auto convertOp = mlir::dyn_cast<IE::ConvertOp>(convertOrReturnOp);
            bool convertOpHasReturnConsumer = !mlir::failed(getReturnOpConsumer(convertOp));
            if (convertOpHasReturnConsumer && doesConvertOpIncreaseElemSize(convertOp)) {
                return mlir::failure();
            }
        } else {
            VPUX_THROW("Unexpected operation '{0}' at '{1}'", convertOrReturnOp->getName(),
                       convertOrReturnOp->getLoc());
        }
    }
    const auto propagatingOrder = DimsOrder::fromValue(argReorderOp.getInput());

    // Propagate first input layout and infer layout info
    auto orderInfo = layerOp.getLayoutInfo();
    orderInfo.setInput(0, propagatingOrder);
    layerOp.inferLayoutInfo(orderInfo);
    if (orderInfo.getInput(0) != propagatingOrder) {
        return matchFailed(_log.nest(), rewriter, layerOp, "Layer doesn't support propagating order {0}",
                           propagatingOrder);
    }
    // Check if additional reorders for other inputs are needed
    for (auto ind : irange<size_t>(1, orderInfo.getNumInputs())) {
        const auto input = layerOp->getOperand(checked_cast<uint32_t>(ind));
        const auto order = DimsOrder::fromValue(input);
        const auto isConstInput = mlir::isa_and_nonnull<Const::DeclareOp>(input.getDefiningOp());
        const auto isReorderInput = mlir::isa_and_nonnull<IE::ReorderOp>(input.getDefiningOp());
        const auto canAddTrivialReorder =
                isTrivialReorder(order, orderInfo.getInput(checked_cast<uint32_t>(ind)), getShape(input));

        if (order != orderInfo.getInput(ind) && !isConstInput && !isReorderInput && !canAddTrivialReorder) {
            return matchFailed(_log.nest(), rewriter, layerOp, "Non-constant inputs require additional Reorders");
        }
    }

    rewriter.startOpModification(layerOp);

    _log.nest(1).trace("Remove Reorder before the first input");
    layerOp->getOpOperand(0).set(argReorderOp.getInput());

    const auto inputs = layerOp->getOpOperands();
    for (auto i : irange<size_t>(1, inputs.size())) {
        auto& input = inputs[i];

        const auto curOrder = DimsOrder::fromValue(input.get());
        const auto supportedOrder = orderInfo.getInput(i);

        _log.nest(1).trace("Process input #{0}", i);
        if (curOrder != supportedOrder) {
            insertReorderForInput(layerOp, input, supportedOrder, rewriter, _log.nest());
        }
    }

    const auto outputs = layerOp->getOpResults();
    for (auto i : irange(outputs.size())) {
        auto output = outputs[i];

        const auto curOrder = DimsOrder::fromValue(output);
        const auto supportedOrder = orderInfo.getOutput(i);

        _log.nest(1).trace("Process output #{0}", i);
        if (curOrder != supportedOrder) {
            changeDimsOrder(output, supportedOrder, _log.nest());
            insertReorderForOutput(layerOp, output, curOrder, rewriter, _log.nest());
        }
    }
    rewriter.finalizeOpModification(layerOp);

    return mlir::success();
}

//
// ReorderWithHWEltwise
//
//  The beneficial pattern:
//
// Reorder    Reorder or Const  Reorder     Reorder
//        \     /                  |           |
//        Eltwise          =>   Reorder     Reorder
//          |                      |           |
//        Reorder             LayoutCast   LayoutCast  or Const(changed dims order)
//          |                         \     /
//        SWOp                        Eltwise
//                                       |
//                                    LayoutCast
//                                       |
//                                      SWOp
// The Dimorder has no impact on the element-wise NCE.Eltwise ADD operation
// In the IE dialect, propagating ReorderOp before AddOp allows the fusion of two ReorderOp.
// This reduces the consumption of DMA resources by the VPU.MempermuteOp converted by ReorderOp and also keep the SWOp
// in optimal layout.
//

template <class EltwiseOp, class ConcreteOp>
class ReorderWithHWEltwise final : public mlir::OpRewritePattern<IE::ReorderOp> {
public:
    ReorderWithHWEltwise(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReorderOp>(ctx), _log(log) {
        this->setDebugName("ReorderWithHWEltwise");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class EltwiseOp, class ConcreteOp>
mlir::LogicalResult ReorderWithHWEltwise<EltwiseOp, ConcreteOp>::matchAndRewrite(
        IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE::ReorderOp at {1}", this->getDebugName(), origOp->getLoc());

    auto parentEltwise = origOp.getInput().getDefiningOp<EltwiseOp>();
    if (parentEltwise == nullptr || !parentEltwise.getResult().hasOneUse() ||
        VPU::NCEInvariant::isSupported(parentEltwise).failed()) {
        return mlir::failure();
    }

    auto parentInput1Reorder = mlir::dyn_cast_or_null<IE::ReorderOp>(parentEltwise.getInput1().getDefiningOp());
    if (parentInput1Reorder == nullptr || !parentInput1Reorder.getResult().hasOneUse()) {
        return mlir::failure();
    }

    // Second input must be Reorder or Constant
    auto parentInput2Reorder = mlir::dyn_cast_or_null<IE::ReorderOp>(parentEltwise.getInput2().getDefiningOp());
    if (parentInput2Reorder != nullptr && !parentInput2Reorder.getResult().hasOneUse()) {
        return mlir::failure();
    }

    auto constInput = mlir::dyn_cast_or_null<Const::DeclareOp>(parentEltwise.getInput2().getDefiningOp());
    if (constInput != nullptr && !constInput.getResult().hasOneUse()) {
        return mlir::failure();
    }

    if (parentInput2Reorder == nullptr && constInput == nullptr) {
        return mlir::failure();
    }

    // E#122076: ReorderWithHWEltwise only supports for HW AddOp (DimOrder::NHWC) who could be converted to
    // NCE.Eltwise.Add ReorderWithHWEltwise should only be a temporary solution. ReorderWithHWEltwise rewriter should be
    // work for any DimOrder.
    const auto targetInOutOrder = DimsOrder::NHWC;
    const auto parentEltwiseOutOrder = DimsOrder::fromValue(parentEltwise.getOutput());
    if (parentEltwiseOutOrder != targetInOutOrder) {
        return mlir::failure();
    }

    mlir::Value reorderInput2 = nullptr;
    if (parentInput2Reorder != nullptr) {
        for (auto consumerOp : origOp.getOutput().getUsers()) {
            if (!mlir::isa<ConcreteOp>(consumerOp)) {
                return mlir::failure();
            }
        }
        reorderInput2 = rewriter.create<IE::ReorderOp>(takeOpLoc(origOp, "reorder_in2"), parentEltwise.getInput2(),
                                                       origOp.getDstOrderAttr());
    } else if (constInput != nullptr) {
        reorderInput2 = rewriter.createOrFold<IE::ReorderOp>(takeOpLoc(origOp, "reorder_in2"), constInput.getResult(),
                                                             origOp.getDstOrderAttr());
    } else {
        VPUX_THROW("Unsupported usecase.");
    }

    const auto nhwcOrderAttr = mlir::AffineMapAttr::get(targetInOutOrder.toAffineMap(origOp.getContext()));

    auto reorderInput1 = rewriter.create<IE::ReorderOp>(takeOpLoc(origOp, "reorder_in1"), parentEltwise.getInput1(),
                                                        origOp.getDstOrderAttr());
    auto newIn1 =
            rewriter.create<IE::LayoutCastOp>(takeOpLoc(parentEltwise, "lcast_in1"), reorderInput1, nhwcOrderAttr);
    auto newIn2 =
            rewriter.create<IE::LayoutCastOp>(takeOpLoc(parentEltwise, "lcast_in2"), reorderInput2, nhwcOrderAttr);

    mlir::IRMapping mapper;
    mapper.map(parentEltwise->getOperands(), SmallVector{newIn1, newIn2});
    auto newEltwiseOp = rewriter.clone(*parentEltwise, mapper);
    rewriter.replaceOp(parentEltwise, newEltwiseOp->getResults());

    _log.trace("New EltwiseOp: {0}", newEltwiseOp);

    const auto orderOutAttr =
            mlir::AffineMapAttr::get(DimsOrder::fromValue(origOp.getOutput()).toAffineMap(origOp.getContext()));

    _log.trace("Replace by IE::LayoutCastOp with DimsOrder: {0}", orderOutAttr);

    rewriter.replaceOpWithNewOp<IE::LayoutCastOp>(origOp, newEltwiseOp->getResult(0), orderOutAttr);
    return mlir::success();
}

//
// ReorderWithHWAddSlice
//
//  The beneficial pattern:
//
// Reorder    Reorder or Const  Reorder     Reorder
//        \     /                  |           |
//         Add             =>   Reorder     Reorder
//          |                      |           |
//   (QuantizeCast)             LayoutCast  LayoutCast  or Const(changed dims order)
//          |                         \     /
//        Slice                         Add
//                                       |
//                                   LayoutCast
//                                       |
//                                 (QuantizeCast)
//                                       |
//                                    Reorder
//                                       |
//                                     Slice

class ReorderWithHWAddSlice final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    ReorderWithHWAddSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        this->setDebugName("ReorderWithHWAddSlice");
    }

    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithHWAddSlice::matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE::AddOp at {1}", this->getDebugName(), origOp->getLoc());

    // E#122076: ReorderWithHWAdd only supports HW AddOp (DimOrder::NHWC) who could be converted to NCE.Eltwise.Add
    // ReorderWithHWAdd should only be a temporary solution. ReorderWithHWAdd rewriter should work for any DimOrder
    const auto targetInOutOrder = DimsOrder::NHWC;
    const auto origDimsOrder = DimsOrder::fromValue(origOp.getOutput());
    if (origDimsOrder != targetInOutOrder) {
        return mlir::failure();
    }

    // Check [Reorder] - Add
    auto input1Op = origOp.getInput1().getDefiningOp();
    auto input2Op = origOp.getInput2().getDefiningOp();
    IE::ReorderOp inputReorder = nullptr;
    if ((inputReorder = mlir::dyn_cast_or_null<IE::ReorderOp>(input1Op))) {
        // Check if input2 is Reorder or Constant
        if (!mlir::isa_and_nonnull<IE::ReorderOp, Const::DeclareOp>(input2Op)) {
            return mlir::failure();
        }
    } else if ((inputReorder = mlir::dyn_cast_or_null<IE::ReorderOp>(input2Op))) {
        // Check if input1 is Constant
        if (!mlir::isa_and_nonnull<Const::DeclareOp>(input1Op)) {
            return mlir::failure();
        }
    } else {
        // Both inputs are not Reorder
        return mlir::failure();
    }

    // Check no branches
    if (!hasOneUniqueUser(input1Op) || !hasOneUniqueUser(input2Op)) {
        return mlir::failure();
    }

    bool bothInputsSame = inputReorder.getResult().hasOneUse() ? false : true;

    // Check Reorder - Add - [(QuantizeCast)]
    auto consumerOp = *(origOp.getResult().getUsers().begin());
    auto quantCast = mlir::dyn_cast<IE::QuantizeCastOp>(consumerOp);
    if (quantCast != nullptr) {
        consumerOp = *(consumerOp->getResult(0).getUsers().begin());
    }

    // Check Reorder - Add - (QuantizeCast) - [Slice]
    if (mlir::dyn_cast<IE::SliceOp>(consumerOp) == nullptr) {
        return mlir::failure();
    }

    auto sliceParent = consumerOp->getOperand(0);
    const auto origShape = getShape(sliceParent);
    const auto newDimsOrder = DimsOrder::fromValue(inputReorder.getInput());
    const auto reorderInputPerm = newDimsOrder.toPermutation();
    const auto reorderOutputPerm = origDimsOrder.toPermutation();

    // Get Slice DMA width
    auto getSliceWidth = [](ShapeRef shapeBeforeSlice, ShapeRef shapeAfterSlice, DimArr permutation) -> int64_t {
        int64_t width = 1;
        for (int i = shapeBeforeSlice.size() - 1; i > 0; i--) {
            auto dim = permutation[i];
            width *= shapeAfterSlice[dim];
            if (shapeBeforeSlice[dim] != shapeAfterSlice[dim]) {  // Slice dimension
                break;
            }
        }
        return width;
    };

    for (auto* user : llvm::make_early_inc_range(sliceParent.getUsers())) {
        // All users must be Slices
        auto userSliceOp = mlir::dyn_cast<IE::SliceOp>(user);
        if (userSliceOp == nullptr) {
            return mlir::failure();
        }

        // Test adding Reorder and make sure the added Reorder-Slice can be further optimized into Slice-PermuteCast
        auto sliceShape = getShape(userSliceOp.getResult());
        if (!isTrivialReorder(newDimsOrder, origDimsOrder, sliceShape)) {
            return mlir::failure();
        }

        // After Reorder-Slice swap, the new slice should not have a smaller DMA width than the original
        int64_t origSliceWidth = getSliceWidth(origShape, sliceShape, reorderOutputPerm);
        int64_t newSliceWidth = getSliceWidth(origShape, sliceShape, reorderInputPerm);
        if (newSliceWidth < origSliceWidth) {
            return mlir::failure();
        }
    }

    // Pattern matched
    const auto origOrderMap = origDimsOrder.toAffineMap(rewriter.getContext());
    const auto newOrderMap = newDimsOrder.toAffineMap(rewriter.getContext());
    auto reorderInput1 =
            rewriter.createOrFold<IE::ReorderOp>(takeOpLoc(origOp, "reorder_in1"), origOp.getInput1(), newOrderMap);
    auto newIn1 = rewriter.create<IE::LayoutCastOp>(takeOpLoc(origOp, "lcast_in1"), reorderInput1, origOrderMap);
    auto newIn2 = newIn1;
    if (!bothInputsSame) {
        auto reorderInput2 =
                rewriter.createOrFold<IE::ReorderOp>(takeOpLoc(origOp, "reorder_in2"), origOp.getInput2(), newOrderMap);
        newIn2 = rewriter.create<IE::LayoutCastOp>(takeOpLoc(origOp, "lcast_in2"), reorderInput2, origOrderMap);
    }
    mlir::Value newAdd = rewriter.create<IE::AddOp>(
            takeOpLoc(origOp, "as_add"), origOp.getType(), newIn1, newIn2, origOp.getAutoBroadcastAttr(),
            origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
    _log.trace("New AddOp: {0}", newAdd);
    auto newOut = rewriter.create<IE::LayoutCastOp>(takeOpLoc(origOp, "lcast_out"), newAdd, newOrderMap);
    auto newReorderOp = rewriter.create<IE::ReorderOp>(takeOpLoc(origOp, "reorder_out"), newOut, origOrderMap);

    if (quantCast != nullptr) {
        auto outputTypeQuantize = mlir::cast<mlir::ShapedType>(quantCast.getType());
        auto outElemType = outputTypeQuantize.getElementType();
        auto newQuantCast = rewriter.create<IE::QuantizeCastOp>(takeOpLoc(origOp, "qcast"), newReorderOp, outElemType);
        sliceParent.replaceAllUsesWith(newQuantCast.getResult());
        _log.trace("Replace by IE::QuantizeCastOp: {0}", newQuantCast);
    } else {
        sliceParent.replaceAllUsesWith(newReorderOp.getResult());
        _log.trace("Replace by IE::ReorderOp {0}", newReorderOp);
    }

    return mlir::success();
}

//
// ReorderWithGroupConv
//

class ReorderWithGroupConv final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    ReorderWithGroupConv(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
        setDebugName("ReorderWithGroupConv");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Value getNewFilter(IE::GroupConvolutionOp origOp, int64_t newChannel, int64_t repeatCnt,
                         mlir::PatternRewriter& rewriter) {
    auto ctx = origOp->getContext();
    SmallVector<int64_t> repeatsShape(getShape(origOp.getFilter()).size(), 1);

    auto filterReorderOp = mlir::dyn_cast_or_null<IE::ReorderOp>(origOp.getFilter().getDefiningOp());
    if (filterReorderOp && filterReorderOp->hasOneUse()) {
        auto filterTileOp = mlir::dyn_cast_or_null<IE::TileOp>(filterReorderOp.getInput().getDefiningOp());
        if (filterTileOp && filterTileOp->hasOneUse()) {
            repeatsShape[Dims4D::Act::N.ind()] = newChannel;
            auto newTileOp = rewriter.create<IE::TileOp>(takeOpLoc(filterTileOp, "as_tile"), filterTileOp.getInput(),
                                                         nullptr, getIntArrayAttr(ctx, Shape(repeatsShape)));
            return rewriter
                    .create<IE::ReorderOp>(takeOpLoc(filterReorderOp, "reorder_out"), newTileOp.getOutput(),
                                           filterReorderOp.getDstOrderAttr())
                    .getOutput();
        }
    }

    repeatsShape[Dims4D::Act::N.ind()] = repeatCnt;
    return rewriter
            .create<IE::TileOp>(takeOpLoc(origOp, "filter_repeats"), origOp.getFilter(), nullptr,
                                getIntArrayAttr(ctx, Shape(repeatsShape)))
            .getOutput();
}

mlir::LogicalResult ReorderWithGroupConv::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp->getContext();
    if (!DoesReorderWithGroupConvPatternMatch(origOp)) {
        return matchFailed(_log, rewriter, origOp, "DWConv pattern doesn't match");
    }

    auto inReorderOp = origOp.getInput().getDefiningOp<IE::ReorderOp>();
    const auto convInOrder = DimsOrder::fromValue(origOp.getInput());

    const auto inOrderAttr = mlir::AffineMapAttr::get(convInOrder.toAffineMap(ctx));
    const auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto origChannel = inType.getShape()[Dims4D::Act::C];
    const auto reorderInType = mlir::cast<vpux::NDTypeInterface>(inReorderOp.getInput().getType());
    const auto reorderInOrder = reorderInType.getDimsOrder();
    const auto reorderInMemShape = reorderInType.getMemShape();
    const auto dimPos = convInOrder.dimPos(Dims4D::Act::C);
    const auto newChannel = reorderInMemShape.raw()[dimPos];
    const auto newBatch = reorderInMemShape.raw()[convInOrder.dimPos(Dims4D::Act::N)];

    // Here we have two solution:
    // 1. LayoutCast, only change the layout, logic shape is the same, so don't need to change filter. LayoutCast
    // will convert to permuteCast in VPUIP dialect, since it could not infer output from input, it may block some
    // optimizaton.
    // 2. PermuteCast, change the layout and logic shape, so we may need to adjust the filter, need extra cost.
    // Experimental threshold data
    constexpr int64_t THRESHOLD_FOR_BENEFICIAL_CONVERSION = 16;
    const auto repeatCnt = newChannel / origChannel;

    // there is a potential possibility that even newChannel is not divisible by origChannel, we could try to use
    // permuteCast, but this need pattern match to tile the weights from the beggining, currently we don't support
    // it, see #E165612
    if (newBatch != 1 || newChannel % origChannel != 0 || repeatCnt > THRESHOLD_FOR_BENEFICIAL_CONVERSION ||
        origOp.getBias()) {
        auto inLayoutCast = rewriter.create<IE::LayoutCastOp>(takeOpLoc(origOp, "in_layoutCast"),
                                                              inReorderOp.getInput(), inOrderAttr);
        origOp->setOperand(0, inLayoutCast.getOutput());
        const auto outOrderAttr = mlir::AffineMapAttr::get(reorderInOrder.toAffineMap(ctx));
        rewriter.setInsertionPointAfter(origOp);
        auto outLayoutCast = rewriter.create<IE::LayoutCastOp>(takeOpLoc(origOp, "out_layoutCast"), origOp.getOutput(),
                                                               outOrderAttr);
        auto newReorder =
                rewriter.create<IE::ReorderOp>(takeOpLoc(origOp, "reorder"), outLayoutCast.getOutput(), inOrderAttr);
        rewriter.replaceAllUsesExcept(origOp.getOutput(), newReorder.getOutput(), {outLayoutCast});
    } else {
        auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(checked_cast<uint32_t>(inType.getRank()), ctx);
        auto inPermuteCast = rewriter.create<IE::PermuteCastOp>(
                takeOpLoc(origOp, "in_permuteCast"), inReorderOp.getInput(), convInOrder.toAffineMap(ctx), identityMap);
        auto newFilter = getNewFilter(origOp, newChannel, repeatCnt, rewriter);
        auto newGroupAttr = getIntAttr(ctx, newChannel);
        auto newGroupConv = rewriter.create<IE::GroupConvolutionOp>(
                origOp->getLoc(), inPermuteCast.getResult(), newFilter, origOp.getBias(), origOp.getStridesAttr(),
                origOp.getPadsBeginAttr(), origOp.getPadsEnd(), origOp.getDilationsAttr(), newGroupAttr,
                origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputPaddingAttr(),
                origOp.getInputPaddingAttr());
        auto origOutputType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        newGroupConv.getOutput().setType(
                mlir::cast<mlir::RankedTensorType>(origOutputType.changeShape(getShape(newGroupConv.getOutput()))));

        auto outPermuteCast =
                rewriter.create<IE::PermuteCastOp>(takeOpLoc(origOp, "out_permuteCast"), newGroupConv.getOutput(),
                                                   reorderInOrder.toAffineMap(ctx), identityMap);
        auto newReorder = rewriter.create<IE::ReorderOp>(takeOpLoc(origOp, "out_reorder"), outPermuteCast.getOutput(),
                                                         inOrderAttr);

        rewriter.replaceOp(origOp, newReorder.getOutput());
    }

    return mlir::success();
}

//
// ReorderWithAddAndGroupConv
//
// Optimize the pattern:
//   input1(NCHW)   input2(NCHW)
//       |               |
//     Reorder        Reorder
//       |               |
//     (NHWC)          (NHWC)
//         \           /
//             Add
//              |
//        GroupConvolution (eltwise)
//              |
//            Reorder
//              |
//          (any order)
//
// Into:
//   input1          input2
//       |               |
//   PermuteCast     PermuteCast
//       |               |
//         \           /
//             Add
//              |
//        GroupConvolution (eltwise)
//              |
//          PermuteCast      <- restores NCHW logical shape
//              |
//           Reorder         <- reaches the original dst order; folded if dst is already NCHW
//              |
//           output
//
// The input PermuteCasts reinterpret memory layout without data movement, eliminating expensive
// Reorders. The output PermuteCast+Reorder pair is canonicalized away when the original chain
// was a NCHW -> NHWC -> NCHW round-trip.
//

class ReorderWithAddAndGroupConv final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    ReorderWithAddAndGroupConv(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
        setDebugName("ReorderWithAddAndGroupConv");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithAddAndGroupConv::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE::GroupConvolutionOp at '{1}'", getDebugName(), origOp->getLoc());

    // --- 1. Pattern topology: Reorder -> Add -> eltwise-GroupConv -> Reorder ---

    if (!IE::isEltwiseGroupConv(origOp, /*isConstFilter=*/true)) {
        return mlir::failure();
    }
    if (!origOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }
    auto outReorderOp = mlir::dyn_cast<IE::ReorderOp>(*origOp.getOutput().getUsers().begin());
    if (outReorderOp == nullptr) {
        return mlir::failure();
    }
    auto addOp = origOp.getInput().getDefiningOp<IE::AddOp>();
    if (addOp == nullptr || !addOp.getResult().hasOneUse()) {
        return mlir::failure();
    }
    auto in1ReorderOp = addOp.getInput1().getDefiningOp<IE::ReorderOp>();
    auto in2ReorderOp = addOp.getInput2().getDefiningOp<IE::ReorderOp>();
    if (in1ReorderOp == nullptr || in2ReorderOp == nullptr) {
        return mlir::failure();
    }
    if (!in1ReorderOp.getResult().hasOneUse() || !in2ReorderOp.getResult().hasOneUse()) {
        return mlir::failure();
    }
    // All Reorders must perform actual data movement (non-trivial) to justify the substitution.
    if (isTrivialReorder(in1ReorderOp) || isTrivialReorder(in2ReorderOp) || isTrivialReorder(outReorderOp)) {
        return mlir::failure();
    }

    // --- 2. Layout consistency: all orders form a coherent round-trip ---

    // Input Reorders must feed the GroupConv layout; GroupConv must preserve it at the output
    // so the output PermuteCast can restore the original order.
    const auto groupConvInOrder = DimsOrder::fromValue(origOp.getInput());
    if (DimsOrder::fromAffineMap(in1ReorderOp.getDstOrder()) != groupConvInOrder ||
        DimsOrder::fromAffineMap(in2ReorderOp.getDstOrder()) != groupConvInOrder) {
        return mlir::failure();
    }
    if (DimsOrder::fromValue(origOp.getOutput()) != groupConvInOrder) {
        return mlir::failure();
    }
    const auto origInOrder = DimsOrder::fromValue(in1ReorderOp.getInput());
    // This optimization targets a fixed round-trip: NCHW -> NHWC -> NCHW.
    // The shape arithmetic below assumes newGroups is taken from original W.
    if (origInOrder != DimsOrder::NCHW || groupConvInOrder != DimsOrder::NHWC) {
        return mlir::failure();
    }

    // Both Add inputs must share the same type to rule out broadcasting: PermuteCast changes
    // the logical shape, so any implicit broadcast could silently shift to a different axis.
    if (in1ReorderOp.getInput().getType() != in2ReorderOp.getInput().getType()) {
        return mlir::failure();
    }

    // --- 3. Semantic constraints: attributes that are axis-sensitive ---

    const auto hasPerAxisQuantElemType = [](mlir::Value val) {
        const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType());
        return ndType != nullptr && mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(ndType.getElementType());
    };
    const auto hasNonChannelAgnosticPostOp = [](mlir::Operation* op) {
        auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
        if (!layerWithPostOp) {
            return false;
        }

        const auto postOp = layerWithPostOp.getPostOp();
        return postOp != nullptr && !postOp.isChannelAgnostic();
    };

    // Per-axis quantization scales are indexed by the channel axis; that axis changes after
    // PermuteCast, so the quantization would apply to the wrong dimension.
    const auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    if (hasPerAxisQuantElemType(origOp.getOutput()) || hasPerAxisQuantElemType(addOp.getInput1()) ||
        hasPerAxisQuantElemType(addOp.getInput2()) || hasPerAxisQuantElemType(addOp.getOutput())) {
        return mlir::failure();
    }
    // GroupConv/Add per-channel post-op parameters are indexed by the channel axis;
    // after the rewrite the channel axis differs, so only channel-agnostic post-ops are safe.
    if (hasNonChannelAgnosticPostOp(origOp.getOperation()) || hasNonChannelAgnosticPostOp(addOp.getOperation())) {
        return mlir::failure();
    }
    // Bias length equals origGroups; after the rewrite groups = W (newGroups), which may differ.
    if (origOp.getBias() != nullptr) {
        return mlir::failure();
    }

    // GroupConv/Add padding attributes are axis-sensitive. The rewrite changes logical axis
    // interpretation, so preserve semantics only when no input/output padding is attached.
    if (origOp.getInputPaddingAttr() != nullptr || origOp.getOutputPaddingAttr() != nullptr ||
        addOp.getInputPaddingAttr() != nullptr || addOp.getOutputPaddingAttr() != nullptr) {
        return mlir::failure();
    }

    // --- 4. Shape arithmetic: the post-PermuteCast channel value must be legal ---

    // After PermuteCast the W dimension of the original tensor becomes the new channel count.
    const auto inType = mlir::cast<vpux::NDTypeInterface>(in1ReorderOp.getInput().getType());
    const auto origInShape = inType.getShape();
    const auto newChannels = origInShape[Dims4D::Act::W];
    // Unaligned channel count would require an Expand later, defeating the purpose.
    auto alignedChannelsIface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    const auto channelAlignment = alignedChannelsIface ? alignedChannelsIface.getInputChannelAlignment() : 1;
    if (newChannels % channelAlignment != 0) {
        return mlir::failure();
    }
    // Filter OC = origGroups = C of the original input; the slice [0, newGroups) must fit.
    const auto origGroups = origInShape[Dims4D::Act::C];
    if (newChannels > origGroups) {
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.getFilter());
    // This rewrite relies on depthwise-like eltwise GroupConv semantics where IC per group is 1.
    // For IC > 1, changing groups from C to W would require remapping IC consistently, which is not
    // performed by this transform.
    if (filterShape[Dims4D::Filter::IC] != 1) {
        return mlir::failure();
    }

    _log.trace("Replacing Reorder-Add-GroupConv-Reorder with PermuteCast-Add-GroupConv-PermuteCast at '{0}'",
               origOp->getLoc());

    const auto ctx = rewriter.getContext();
    const auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(checked_cast<uint32_t>(outType.getRank()), ctx);
    const auto identityMapAttr = mlir::AffineMapAttr::get(identityMap);
    const auto groupConvInOrderAttr = mlir::AffineMapAttr::get(groupConvInOrder.toAffineMap(ctx));

    // PermuteCast: reinterpret memory layout without data movement.
    // Example: NCHW 1x1024x35x256 (mem [1,1024,35,256]) -> NHWC 1x256x1024x35 (same mem, C=256).
    auto in1PermuteCast = rewriter.create<IE::PermuteCastOp>(
            takeOpLoc(origOp, "in1_permuteCast"), in1ReorderOp.getInput(), groupConvInOrderAttr, identityMapAttr);
    auto in2PermuteCast = rewriter.create<IE::PermuteCastOp>(
            takeOpLoc(origOp, "in2_permuteCast"), in2ReorderOp.getInput(), groupConvInOrderAttr, identityMapAttr);

    // The PermuteCast output has a permuted logical shape (e.g. 1x256x1024x35 NHWC).
    // C dimension of the permuted shape becomes the new groups count for the eltwise GroupConv.
    const auto permutedShape = getShape(in1PermuteCast.getOutput());
    const auto newGroups = permutedShape[Dims4D::Act::C];
    const auto newGroupsAttr = getIntAttr(ctx, newGroups);

    // Add operates on the permuted shapes directly.
    const auto origAddElemType = mlir::cast<vpux::NDTypeInterface>(addOp.getType()).getElementType();
    const auto newAddOutType =
            mlir::cast<vpux::NDTypeInterface>(in1PermuteCast.getOutput().getType()).changeElemType(origAddElemType);
    auto newAddOp =
            rewriter.create<IE::AddOp>(addOp->getLoc(), newAddOutType, in1PermuteCast.getOutput(),
                                       in2PermuteCast.getOutput(), addOp.getAutoBroadcastAttr(), addOp.getPostOpAttr(),
                                       addOp.getClampAttr(), addOp.getOutputPaddingAttr(), addOp.getInputPaddingAttr());

    // The GroupConv filter must match the new groups count. Since isEltwiseGroupConv ensures
    // the filter is a splat constant, slicing OC dim from origGroups to newGroups is semantically equivalent.
    SmallVector<int64_t> filterSliceOffsets(filterShape.size(), 0);
    SmallVector<int64_t> filterSliceSizes(filterShape.raw().begin(), filterShape.raw().end());
    filterSliceSizes[Dims4D::Filter::OC.ind()] = newGroups;
    auto newFilter = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, "filter_slice"), origOp.getFilter(),
                                                  getIntArrayAttr(ctx, filterSliceOffsets),
                                                  getIntArrayAttr(ctx, filterSliceSizes));

    const auto newGroupConvOutType = mlir::cast<vpux::NDTypeInterface>(in1PermuteCast.getOutput().getType())
                                             .changeElemType(outType.getElementType());
    auto newGroupConv = rewriter.create<IE::GroupConvolutionOp>(
            origOp->getLoc(), newGroupConvOutType, newAddOp.getOutput(), newFilter.getResult(), /*bias=*/nullptr,
            origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            newGroupsAttr, origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputPaddingAttr(),
            origOp.getInputPaddingAttr());

    // PermuteCast back: NHWC 1x256x1024x35 (mem [1,1024,35,256]) -> origInOrder (NCHW) 1x1024x35x256.
    // Physical memory [1,1024,35,256] interpreted as NCHW [N,C,H,W] gives shape 1x1024x35x256.
    const auto origInOrderAttr = mlir::AffineMapAttr::get(origInOrder.toAffineMap(ctx));
    auto outPermuteCast = rewriter.create<IE::PermuteCastOp>(
            takeOpLoc(origOp, "out_permuteCast"), newGroupConv.getOutput(), origInOrderAttr, identityMapAttr);

    // Always emit the trailing Reorder to reach the original dst order. When the original chain
    // was a round-trip (dst == NCHW == origInOrder) this becomes a trivial Reorder.
    auto newOutReorder = rewriter.create<IE::ReorderOp>(takeOpLoc(origOp, "out_reorder"), outPermuteCast.getOutput(),
                                                        outReorderOp.getDstOrderAttr());
    rewriter.replaceOp(outReorderOp, newOutReorder.getOutput());

    return mlir::success();
}

//
// ReorderWithEltwise
//

template <class ConcreteOp>
class ReorderWithEltwise final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ReorderWithEltwise(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("ReorderWithEltwise");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// This rewriter aims to propagate Reorder through Eltwise-like op with 'illegal' layout,
// which cannot be done by ReorderWithLayer.
// Since Eltwise-like ops do not really care about the layout, so we can insert PermuteCast around
// the op to make an identical layout, then Reorder can be propagated.

template <class ConcreteOp>
mlir::LogicalResult ReorderWithEltwise<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    if (!ConcreteOp::template hasTrait<IE::EltwiseOp>()) {
        return mlir::failure();
    }
    _log.debug("[{0}] Got Eltwise-like Op at {1}", this->getDebugName(), origOp->getLoc());

    auto nestedLog = _log.nest();

    if (origOp->getNumOperands() > 1 || origOp->getNumResults() > 1) {
        nestedLog.trace("Op has more than one input and/or output.");
        return mlir::failure();
    }

    const auto ctx = rewriter.getContext();
    auto inputReorder = origOp.getOperand().template getDefiningOp<IE::ReorderOp>();
    if (inputReorder == nullptr || !inputReorder.getResult().hasOneUse()) {
        nestedLog.trace("Parent Op is not Reorder or the parent op has multiple uses.");
        return mlir::failure();
    }

    const auto inputType = mlir::cast<NDTypeInterface>(inputReorder.getInput().getType());
    const auto origInOrder = inputType.getDimsOrder();

    auto layerOp = mlir::dyn_cast<IE::LayoutInfoOpInterface>(origOp.getOperation());
    if (layerOp == nullptr) {
        nestedLog.trace("Op does not implement LayoutInfoOpInterface.");
        return mlir::failure();
    }
    auto orderInfo = layerOp.getLayoutInfo();
    orderInfo.setInput(0, origInOrder);
    layerOp.inferLayoutInfo(orderInfo);

    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(checked_cast<unsigned>(inputType.getRank()), ctx);
    const auto inPermuteCastType = inferNewTypeWithMemPerm(inputType, identityMap, orderInfo.getInput(0));
    auto inPermuteCast = rewriter.create<IE::PermuteCastOp>(
            appendLoc(origOp->getLoc(), "perm_cast_in"), inPermuteCastType, inputReorder.getInput(),
            mlir::AffineMapAttr::get(orderInfo.getInput(0).toAffineMap(ctx)), mlir::AffineMapAttr::get(identityMap));

    auto origOutElemType = mlir::cast<NDTypeInterface>(origOp->getResult(0).getType()).getElementType();
    auto outType = mlir::cast<NDTypeInterface>(inPermuteCast.getResult().getType()).changeElemType(origOutElemType);

    mlir::IRMapping mapper;
    mapper.map(origOp.getOperand(), inPermuteCast.getResult());
    auto newConcreteOp = mlir::cast<ConcreteOp>(rewriter.clone(*origOp, mapper));
    newConcreteOp.getOutput().setType(mlir::cast<mlir::RankedTensorType>(outType));

    const auto outPermuteCastType = inferNewTypeWithMemPerm(
            mlir::cast<NDTypeInterface>(newConcreteOp.getOutput().getType()), identityMap, origInOrder);
    auto outPermuteCast = rewriter.create<IE::PermuteCastOp>(
            appendLoc(origOp->getLoc(), "perm_cast_out"), outPermuteCastType, newConcreteOp.getOutput(),
            mlir::AffineMapAttr::get(origInOrder.toAffineMap(ctx)), mlir::AffineMapAttr::get(identityMap));

    auto outReorder = rewriter.replaceOpWithNewOp<IE::ReorderOp>(origOp, outPermuteCast.getResult(),
                                                                 inputReorder.getDstOrderAttr());
    extendOpLoc(outReorder, "out_reorder");

    _log.debug("Propagate Reorder through Eltwise-like op by inserting PermuteCast");
    return mlir::success();
}

//
// OptimizeReordersPass
//

class OptimizeReordersPass final : public IE::impl::OptimizeReordersBase<OptimizeReordersPass> {
public:
    explicit OptimizeReordersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeReordersPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReorderWithShapeChange<IE::ReshapeOp>>(&ctx, _log);
    patterns.add<ReorderWithShapeChange<IE::AffineReshapeOp>>(&ctx, _log);
    patterns.add<ReorderWithShapeChange<IE::ShapeCastOp>>(&ctx, _log);
    patterns.add<ReorderWithSubView>(&ctx, _log);
    patterns.add<IE::ExpandWithLayer>(&ctx, isBeneficialToSwapExpandReorders, _log);
    patterns.add<ReorderWithSplit>(&ctx, _log);
    patterns.add<ReorderWithConcat>(&ctx, _log);
    patterns.add<ReorderWithQuantCast>(&ctx, _log);
    patterns.add<ReorderWithTile>(&ctx, _log);
    patterns.add<ReorderWithLayer>(&ctx, _log);
    patterns.add<ReorderWithPermuteCast>(&ctx, _log);
    patterns.add<ReorderWithHWEltwise<IE::AddOp, IE::MVNOp>>(&ctx, _log);
    patterns.add<ReorderWithHWEltwise<IE::AddOp, IE::SubtractOp>>(&ctx, _log);
    patterns.add<ReorderWithHWAddSlice>(&ctx, _log);
    patterns.add<ReorderWithGroupConv>(&ctx, _log);
    patterns.add<ReorderWithAddAndGroupConv>(&ctx, _log);
    patterns.add<ReorderWithEltwise<IE::GeluOp>>(&ctx, _log);
    IE::ReorderOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getOperation();
    // Raise the greedy iteration cap above the shared default. Large models can require more rewrite sweeps to
    // reach a fixpoint than the default allows, causing the pass to fail even though the sequence terminates.
    // The greedy driver stops as soon as it converges, so a larger cap is free for models that converge early
    // and only lets slow-but-terminating cases finish instead of failing prematurely.
    auto config = getDefaultGreedyRewriteConfig();
    config.setMaxIterations(config.getMaxIterations() * 10);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), config))) {
        signalPassFailure();
        return;
    }

    mlir::RewritePatternSet cleanupPatterns(&ctx);
    cleanupPatterns.add<ReorderWithConvert>(&ctx, _log);
    cleanupPatterns.add<ReorderWithExpandSlice>(&ctx, _log);
    cleanupPatterns.add<ReorderWithAffineReshapeTile>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(cleanupPatterns));

    mlir::RewritePatternSet canonPatterns(&ctx);
    IE::ReorderOp::getCanonicalizationPatterns(canonPatterns, &ctx);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(canonPatterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeReordersPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeReordersPass(Logger log) {
    return std::make_unique<OptimizeReordersPass>(log);
}
