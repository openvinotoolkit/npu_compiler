//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
namespace vpux::IE {
#define GEN_PASS_DECL_UNIQUIFYBRANCHES
#define GEN_PASS_DEF_UNIQUIFYBRANCHES
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool doesSliceAndPermutationModifySameAxis(DimsOrder perm, ArrayRef<uint64_t> sliceAxes, Logger log) {
    auto order = to_small_vector(irange(perm.numDims()) | transformed([&](uint64_t idx) {
                                     return checked_cast<uint64_t>(perm.dimAt(idx).ind());
                                 }));

    for (auto axis : sliceAxes) {
        if (order[axis] != axis) {
            log.trace("Layer modifies sliced axis: {0}", axis);
            return true;
        }
    }

    return false;
}

template <typename ConcreteOp>
class MoveLayerBeforeSlice : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MoveLayerBeforeSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    virtual bool isBeneficialTransformation(IE::SliceOp sliceOp, ConcreteOp layerOp,
                                            ArrayRef<ConcreteOp> siblingLayerOps) const;
    virtual bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                                   ConcreteOp layerOp) const = 0;
    virtual bool sameAttributes(ConcreteOp layerOp, ConcreteOp currLayerOp) const = 0;
    virtual mlir::Operation* createNewLayerOp(ConcreteOp layerOp, IE::SliceOp sliceOp,
                                              mlir::PatternRewriter& rewriter) const;
    virtual SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, ConcreteOp layerOp) const = 0;
    virtual SmallVector<int64_t> getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                               ConcreteOp layerOp) const;

protected:
    virtual bool isLegalTransformation(IE::SliceOp sliceOp, ConcreteOp layerOp,
                                       ArrayRef<ConcreteOp> siblingLayerOps) const;
    Logger _log;
};

template <typename ConcreteOp>
bool MoveLayerBeforeSlice<ConcreteOp>::isLegalTransformation(IE::SliceOp sliceOp, ConcreteOp layerOp,
                                                             ArrayRef<ConcreteOp> siblingLayerOps) const {
    const auto sliceAxes = getSliceAxes(sliceOp);
    if (doesSliceAndLayerOpModifySameAxis(sliceOp, sliceAxes, layerOp)) {
        return false;
    }

    if (!isBeneficialTransformation(sliceOp, layerOp, siblingLayerOps)) {
        return false;
    }

    return true;
}

template <typename ConcreteOp>
bool MoveLayerBeforeSlice<ConcreteOp>::isBeneficialTransformation(IE::SliceOp, ConcreteOp, ArrayRef<ConcreteOp>) const {
    return true;
}

template <typename ConcreteOp>
mlir::LogicalResult MoveLayerBeforeSlice<ConcreteOp>::matchAndRewrite(ConcreteOp layerOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto maybeSliceOp = layerOp.getInput().template getDefiningOp<IE::SliceOp>();
    if (maybeSliceOp == nullptr) {
        return mlir::failure();
    }

    auto sliceSrcUsers = maybeSliceOp.getSource().getUsers();

    auto hasAnotherSlice = llvm::any_of(sliceSrcUsers, [&](mlir::Operation* user) {
        auto anotherSlice = mlir::dyn_cast<IE::SliceOp>(user);
        return anotherSlice != nullptr && anotherSlice != maybeSliceOp;
    });

    if (!hasAnotherSlice) {
        return mlir::failure();
    }

    _log.trace("Got layer op: {0}", layerOp);
    _log.trace("Parent slice: {0}", maybeSliceOp);

    SmallVector<ConcreteOp> siblingLayerOps;
    const auto isSameSliceLayerBranch = [&](mlir::Operation* user) {
        auto currSliceOp = mlir::dyn_cast<IE::SliceOp>(user);
        if (currSliceOp == nullptr) {
            return true;
        }

        if (!currSliceOp.getResult().hasOneUse()) {
            return false;
        }

        if (currSliceOp.getStaticSizesAttr() != maybeSliceOp.getStaticSizesAttr()) {
            return false;
        }

        auto currLayerOp = mlir::dyn_cast<ConcreteOp>(*currSliceOp.getResult().getUsers().begin());
        if (currLayerOp == nullptr) {
            return false;
        }

        siblingLayerOps.push_back(currLayerOp);

        return sameAttributes(layerOp, currLayerOp);
    };

    const auto root = maybeSliceOp.getSource();
    for (auto user : root.getUsers()) {
        if (!isSameSliceLayerBranch(user)) {
            return mlir::failure();
        }
    }

    if (!isLegalTransformation(maybeSliceOp, layerOp, siblingLayerOps)) {
        return mlir::failure();
    }

    auto newLayerOp = createNewLayerOp(layerOp, maybeSliceOp, rewriter);
    _log.trace("Create new layer op: {0}", newLayerOp);

    auto newSizes = getNewSizes(maybeSliceOp, layerOp);
    const auto sliceAxes = getSliceAxes(maybeSliceOp);
    for (auto user : root.getUsers()) {
        auto sliceOp = mlir::dyn_cast<IE::SliceOp>(user);
        if (sliceOp == nullptr) {
            continue;
        }

        auto currLayerOp = mlir::dyn_cast<ConcreteOp>(*sliceOp.getResult().getUsers().begin());
        auto newOffsets = getNewOffsets(sliceOp, sliceAxes, currLayerOp);
        auto newSlice = rewriter.create<IE::SliceOp>(sliceOp->getLoc(), newLayerOp->getResult(0),
                                                     getIntArrayAttr(newLayerOp->getContext(), newOffsets),
                                                     getIntArrayAttr(newLayerOp->getContext(), newSizes));
        _log.trace("Create new Slice: {0}", newSlice);

        rewriter.replaceOp(currLayerOp, newSlice.getResult());
    }

    return mlir::success();
}

template <typename ConcreteOp>
mlir::Operation* MoveLayerBeforeSlice<ConcreteOp>::createNewLayerOp(ConcreteOp layerOp, IE::SliceOp sliceOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    mlir::IRMapping mapper;
    mapper.map(layerOp->getOperands(), ArrayRef({sliceOp.getSource()}));

    rewriter.setInsertionPointAfterValue(sliceOp.getSource());
    auto* newLayerOp = rewriter.clone(*layerOp.getOperation(), mapper);
    vpux::inferReturnTypes(newLayerOp, vpux::InferShapedTypeMode::ALL);
    return newLayerOp;
}

template <typename ConcreteOp>
SmallVector<int64_t> MoveLayerBeforeSlice<ConcreteOp>::getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t>,
                                                                     ConcreteOp) const {
    return parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
}

//
// MoveExpandBeforeSlice
//

class MoveExpandBeforeSlice final : public MoveLayerBeforeSlice<IE::ExpandOp> {
public:
    MoveExpandBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<IE::ExpandOp>(ctx, log) {
    }

private:
    bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                           IE::ExpandOp layerOp) const override;
    bool sameAttributes(IE::ExpandOp layerOp, IE::ExpandOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::ExpandOp layerOp) const override;
};

bool MoveExpandBeforeSlice::doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                                              IE::ExpandOp layerOp) const {
    auto sizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
    auto expandedShape = getShape(layerOp.getOutput());

    for (auto axis : sliceAxes) {
        if (sizes[axis] != expandedShape[Dim(axis)]) {
            _log.trace("Expand modifies sliced axis: {0}", axis);
            return true;
        }
    }

    return false;
}

bool MoveExpandBeforeSlice::sameAttributes(IE::ExpandOp layerOp, IE::ExpandOp currLayerOp) const {
    return layerOp.getPadsBeginAttr() == currLayerOp.getPadsBeginAttr() &&
           layerOp.getPadsEndAttr() == currLayerOp.getPadsEndAttr();
}

SmallVector<int64_t> MoveExpandBeforeSlice::getNewSizes(IE::SliceOp sliceOp, IE::ExpandOp layerOp) const {
    auto begin = parseIntArrayAttr<int64_t>(layerOp.getPadsBegin());
    auto end = parseIntArrayAttr<int64_t>(layerOp.getPadsEnd());
    auto expandedShape = getShape(layerOp.getOutput());

    auto sizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());

    SmallVector<int64_t> newSizes(expandedShape.size(), 0);
    for (size_t i = 0; i < expandedShape.size(); i++) {
        newSizes[i] = begin[i] + sizes[i] + end[i];
    }

    return newSizes;
}

//
// MoveReorderBeforeSlice
//

class MoveReorderBeforeSlice final : public MoveLayerBeforeSlice<IE::ReorderOp> {
public:
    MoveReorderBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<IE::ReorderOp>(ctx, log) {
    }

public:
    bool isBeneficialTransformation(IE::SliceOp sliceOp, IE::ReorderOp layerOp,
                                    ArrayRef<IE::ReorderOp> siblingLayerOps) const override;

    bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                           IE::ReorderOp layerOp) const override;
    bool sameAttributes(IE::ReorderOp layerOp, IE::ReorderOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::ReorderOp layerOp) const override;
};

bool MoveReorderBeforeSlice::doesSliceAndLayerOpModifySameAxis(IE::SliceOp, ArrayRef<uint64_t>, IE::ReorderOp) const {
    /*
        IE.Reorder changes tensor layout but does not change tensor logical shape.
        IE.Slice operates with logic shape, not with physical layouts.
        We don't need to check for axes when we try to swap IE.Slice and IE.Reorder.

        For example, if we have a 3x4 tensor (N = 3, C = 4) in NC layout:
            [11, 12, 13, 14]
            [21, 22, 23, 24]
            [31, 32, 33, 34]

        The 1st Case - Reorder after Slice
        step 1: IE.Slice static_offset = [0, 1] and static_shape = [2, 2], we get tensor in below:
            [12, 13]
            [22, 23]

        step 2: IE.Reorder changes layout from NC to CN, we get data in memory as below:
            [12, 22] [13, 23]

        The 2nd Case - Slice after Reorder
        step 1: IE.Reorder changes layout from NC to CN, we get data in memory as below:
            [11, 21, 31] [12, 22, 32] [13, 23, 33] [14, 24, 34]

        step 2: IE.Slice static_offset = [0, 1] and static_shape = [2, 2], we get tensor in below:
            [12, 22] [13, 23]

        The results of the two cases are completely identical.
        This is applicable to higher ranks as well.
    */

    return false;
}

bool MoveReorderBeforeSlice::isBeneficialTransformation(IE::SliceOp sliceOp, IE::ReorderOp layerOp,
                                                        ArrayRef<IE::ReorderOp> siblingLayerOps) const {
    // transformation is beneficial when reorder tensor total size can be decreased
    int64_t parallelReordersTotalSize = 0;
    for (auto reorder : siblingLayerOps) {
        // no need to calculate reoder data size when Reorder is a PermuteCast
        parallelReordersTotalSize +=
                isTrivialReorder(reorder)
                        ? 0
                        : mlir::cast<vpux::NDTypeInterface>(reorder.getOutput().getType()).getTotalAllocSize().count();
    }

    auto root = sliceOp.getSource();
    const auto srcOrder = DimsOrder::fromValue(root);
    const auto dstOrder = DimsOrder::fromValue(layerOp.getOutput());
    const auto rootShape = getShape(root);

    const auto newReorderSize = isTrivialReorder(srcOrder, dstOrder, rootShape)
                                        ? 0
                                        : mlir::cast<vpux::NDTypeInterface>(root.getType()).getTotalAllocSize().count();

    if (newReorderSize > parallelReordersTotalSize) {
        _log.trace("Root tensor size '{0}' is larger than total size of parallel Reorder(s): '{1}'. ", newReorderSize,
                   parallelReordersTotalSize);
        return false;
    }

    return true;
}

bool MoveReorderBeforeSlice::sameAttributes(IE::ReorderOp layerOp, IE::ReorderOp currLayerOp) const {
    return layerOp.getDstOrder() == currLayerOp.getDstOrder();
}

SmallVector<int64_t> MoveReorderBeforeSlice::getNewSizes(IE::SliceOp sliceOp, IE::ReorderOp) const {
    return parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
}

//
// MoveTransposeBeforeSlice
//

class MoveTransposeBeforeSlice final : public MoveLayerBeforeSlice<IE::TransposeOp> {
public:
    MoveTransposeBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<IE::TransposeOp>(ctx, log) {
    }

public:
    bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                           IE::TransposeOp layerOp) const override;
    bool sameAttributes(IE::TransposeOp layerOp, IE::TransposeOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::TransposeOp layerOp) const override;
};

bool MoveTransposeBeforeSlice::doesSliceAndLayerOpModifySameAxis(IE::SliceOp, ArrayRef<uint64_t> sliceAxes,
                                                                 IE::TransposeOp layerOp) const {
    const auto perm = DimsOrder::fromAffineMap(layerOp.getOrderValue().value());
    return doesSliceAndPermutationModifySameAxis(perm, sliceAxes, _log);
}

bool MoveTransposeBeforeSlice::sameAttributes(IE::TransposeOp layerOp, IE::TransposeOp currLayerOp) const {
    return layerOp.getOrderValue() == currLayerOp.getOrderValue();
}

SmallVector<int64_t> MoveTransposeBeforeSlice::getNewSizes(IE::SliceOp sliceOp, IE::TransposeOp layerOp) const {
    const auto perm = DimsOrder::fromAffineMap(layerOp.getOrderValue().value());
    auto originOrder = to_small_vector(irange(perm.numDims()) | transformed([&](uint64_t idx) {
                                           return checked_cast<uint64_t>(perm.dimAt(idx).ind());
                                       }));

    auto sizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
    SmallVector<int64_t> newSizes(originOrder.size(), 0);
    for (size_t i = 0; i < originOrder.size(); i++) {
        newSizes[i] = sizes[originOrder[i]];
    }

    return newSizes;
}

//
// MovePermuteCastBeforeSlice
//

class MovePermuteCastBeforeSlice final : public MoveLayerBeforeSlice<IE::PermuteCastOp> {
public:
    MovePermuteCastBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<IE::PermuteCastOp>(ctx, log) {
    }

public:
    bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                           IE::PermuteCastOp layerOp) const override;
    bool sameAttributes(IE::PermuteCastOp layerOp, IE::PermuteCastOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::PermuteCastOp layerOp) const override;
    SmallVector<int64_t> getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                       IE::PermuteCastOp layerOp) const override;
};

bool MovePermuteCastBeforeSlice::doesSliceAndLayerOpModifySameAxis(IE::SliceOp, ArrayRef<uint64_t> sliceAxes,
                                                                   IE::PermuteCastOp layerOp) const {
    const auto orderPerm = DimsOrder::fromAffineMap(layerOp.getDstOrder());
    const auto memPerm = DimsOrder::fromAffineMap(layerOp.getMemPerm());

    return doesSliceAndPermutationModifySameAxis(orderPerm, sliceAxes, _log) ||
           doesSliceAndPermutationModifySameAxis(memPerm, sliceAxes, _log);
}

bool MovePermuteCastBeforeSlice::sameAttributes(IE::PermuteCastOp layerOp, IE::PermuteCastOp currLayerOp) const {
    return layerOp.getDstOrder() == currLayerOp.getDstOrder() && layerOp.getMemPerm() == currLayerOp.getMemPerm();
}

SmallVector<int64_t> MovePermuteCastBeforeSlice::getNewSizes(IE::SliceOp, IE::PermuteCastOp layerOp) const {
    mlir::SmallVector<mlir::ShapedTypeComponents> inferredReturnShapes;

    inferPermuteReturnTypeComponents(layerOp.getInput(), layerOp.getMemPerm(), layerOp.getDstOrder(),
                                     inferredReturnShapes, false);

    VPUX_THROW_WHEN(inferredReturnShapes.size() != 1, "Should should be 1 but got {0}", inferredReturnShapes.size());

    auto dims = inferredReturnShapes.front().getDims();

    return mlir::SmallVector<int64_t>{dims.begin(), dims.end()};
}

SmallVector<int64_t> MovePermuteCastBeforeSlice::getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t>,
                                                               IE::PermuteCastOp layerOp) const {
    const auto permuteCastInputType = mlir::cast<vpux::NDTypeInterface>(layerOp.getInput().getType());
    const auto permuteCastInputOrder = permuteCastInputType.getDimsOrder();
    const auto permuteCastOutputOrder = DimsOrder::fromAffineMap(layerOp.getDstOrder());
    const auto finalPermutation =
            vpux::getPermutationFromOrders(permuteCastInputOrder, permuteCastOutputOrder, sliceOp->getContext());
    const auto oldOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
    SmallVector<int64_t> newOffsets(oldOffsets.size());
    for (auto dim : irange(oldOffsets.size())) {
        newOffsets[finalPermutation.getDimPosition(dim)] = oldOffsets[dim];
    }

    return newOffsets;
};

//
// MoveAffineReshapeBeforeSlice
//

class MoveAffineReshapeBeforeSlice final : public MoveLayerBeforeSlice<IE::AffineReshapeOp> {
public:
    MoveAffineReshapeBeforeSlice(mlir::MLIRContext* ctx, Logger log)
            : MoveLayerBeforeSlice<IE::AffineReshapeOp>(ctx, log) {
    }

public:
    bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                           IE::AffineReshapeOp layerOp) const override;
    bool isLegalTransformation(IE::SliceOp sliceOp, IE::AffineReshapeOp layerOp,
                               ArrayRef<IE::AffineReshapeOp> siblingLayerOps) const override;
    bool doAffineInputAndOutputShapesMismatch(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                              IE::AffineReshapeOp layerOp) const;
    bool sameAttributes(IE::AffineReshapeOp layerOp, IE::AffineReshapeOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::AffineReshapeOp layerOp) const override;
    SmallVector<int64_t> getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                       IE::AffineReshapeOp layerOp) const override;

    mlir::Operation* createNewLayerOp(IE::AffineReshapeOp layerOp, IE::SliceOp sliceOp,
                                      mlir::PatternRewriter& rewriter) const override;
};

bool MoveAffineReshapeBeforeSlice::isLegalTransformation(IE::SliceOp sliceOp, IE::AffineReshapeOp layerOp,
                                                         ArrayRef<IE::AffineReshapeOp> siblingLayerOps) const {
    if (!MoveLayerBeforeSlice::isLegalTransformation(sliceOp, layerOp, siblingLayerOps)) {
        return false;
    }

    const auto sliceAxes = getSliceAxes(sliceOp);
    if (doAffineInputAndOutputShapesMismatch(sliceOp, sliceAxes, layerOp)) {
        return false;
    }

    return true;
}

bool MoveAffineReshapeBeforeSlice::doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                                                     IE::AffineReshapeOp) const {
    // only move AffineReshape when slice on the highest dimension
    if (sliceAxes.size() != 1) {
        return true;
    }

    auto sliceAxis = sliceAxes[0];
    auto inType = mlir::cast<NDTypeInterface>(sliceOp.getSource().getType());
    auto dimOrder = inType.getDimsOrder();
    auto shape = inType.getShape();

    const auto highestDim = getHighestNonTrivialDim(shape, dimOrder).value_or(Dim(0));
    return checked_cast<uint64_t>(highestDim.ind()) != sliceAxis;
}

bool MoveAffineReshapeBeforeSlice::doAffineInputAndOutputShapesMismatch(IE::SliceOp sliceOp,
                                                                        ArrayRef<uint64_t> sliceAxes,
                                                                        IE::AffineReshapeOp layerOp) const {
    // check if pattern has incompatible tensors for affine reshape given slice dimensions
    auto outType = mlir::cast<NDTypeInterface>(layerOp.getOutput().getType());
    auto outShape = outType.getShape();
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(layerOp.getDimMapping());
    auto mappedSliceDim = dimMapping[sliceAxes[0]];
    auto sliceDim = Dim(mappedSliceDim[0]);
    auto inShape = getShape(sliceOp.getSource());
    auto restSize = outShape.totalSize() / outShape[sliceDim];

    return inShape.totalSize() % restSize != 0;
}

bool MoveAffineReshapeBeforeSlice::sameAttributes(IE::AffineReshapeOp layerOp, IE::AffineReshapeOp currLayerOp) const {
    return layerOp.getDimMapping() == currLayerOp.getDimMapping() &&
           layerOp.getShapeValue() == currLayerOp.getShapeValue();
}

SmallVector<int64_t> MoveAffineReshapeBeforeSlice::getNewSizes(IE::SliceOp, IE::AffineReshapeOp layerOp) const {
    return vpux::parseIntArrayAttr<int64_t>(layerOp.getShapeValue());
}

SmallVector<int64_t> MoveAffineReshapeBeforeSlice::getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                                                 IE::AffineReshapeOp layerOp) const {
    auto sliceOffset = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
    auto sliceSize = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
    VPUX_THROW_UNLESS(sliceAxes.size() == 1, "Unexpected slice axes for {0}", sliceOp);
    auto outType = mlir::dyn_cast<vpux::NDTypeInterface>(layerOp.getOutput().getType());
    auto outShape = outType.getShape();
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(layerOp.getDimMapping());
    auto mappedSliceDim = dimMapping[sliceAxes[0]];
    auto sliceDim = Dim(mappedSliceDim[0]);
    auto sliceSizeVal = 1;
    if (mappedSliceDim.size() > 1) {
        for (size_t index = 1; index < mappedSliceDim.size(); index++) {
            sliceSizeVal = sliceSizeVal * outShape[Dim(mappedSliceDim[index])];
        }
    } else {
        sliceSizeVal = sliceSize[sliceAxes[0]];
    }
    auto sliceOffsetVal = sliceOffset[sliceAxes[0]];

    auto newSizes = getNewSizes(sliceOp, layerOp);
    auto sliceIdx = sliceOffsetVal / sliceSizeVal;

    SmallVector<int64_t> newOffsets(newSizes.size(), 0);
    auto newSliceOffset = mappedSliceDim.size() == 1 ? newSizes[sliceDim.ind()] : 1;
    newOffsets[sliceDim.ind()] = sliceIdx * newSliceOffset;

    return newOffsets;
}

mlir::Operation* MoveAffineReshapeBeforeSlice::createNewLayerOp(IE::AffineReshapeOp layerOp, IE::SliceOp sliceOp,
                                                                mlir::PatternRewriter& rewriter) const {
    auto outType = mlir::dyn_cast<vpux::NDTypeInterface>(layerOp.getOutput().getType());
    auto outShape = outType.getShape();
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(layerOp.getDimMapping());
    const auto sliceAxes = getSliceAxes(sliceOp);
    auto mappedSliceDim = dimMapping[sliceAxes[0]];
    auto sliceDim = Dim(mappedSliceDim[0]);
    auto inShape = getShape(sliceOp.getSource());
    auto restSize = outShape.totalSize() / outShape[sliceDim];
    Shape newOutShape(outShape.raw());
    newOutShape[sliceDim] = inShape.totalSize() / restSize;

    rewriter.setInsertionPointAfterValue(sliceOp.getSource());
    return rewriter.create<IE::AffineReshapeOp>(layerOp.getLoc(), sliceOp.getSource(), layerOp.getDimMapping(),
                                                vpux::getIntArrayAttr(rewriter, to_small_vector(newOutShape)));
}

//
// MoveMemPermuteBeforeSlice
//

class MoveMemPermuteBeforeSlice final : public MoveLayerBeforeSlice<IE::MemPermuteOp> {
public:
    MoveMemPermuteBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<IE::MemPermuteOp>(ctx, log) {
    }

public:
    bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                           IE::MemPermuteOp layerOp) const override;
    bool sameAttributes(IE::MemPermuteOp layerOp, IE::MemPermuteOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::MemPermuteOp layerOp) const override;
};

bool MoveMemPermuteBeforeSlice::doesSliceAndLayerOpModifySameAxis(IE::SliceOp, ArrayRef<uint64_t> sliceAxes,
                                                                  IE::MemPermuteOp layerOp) const {
    const auto orderPerm = DimsOrder::fromAffineMap(layerOp.getDstOrder());
    const auto memPerm = DimsOrder::fromAffineMap(layerOp.getMemPerm());

    return doesSliceAndPermutationModifySameAxis(orderPerm, sliceAxes, _log) ||
           doesSliceAndPermutationModifySameAxis(memPerm, sliceAxes, _log);
}

bool MoveMemPermuteBeforeSlice::sameAttributes(IE::MemPermuteOp layerOp, IE::MemPermuteOp currLayerOp) const {
    return layerOp.getDstOrder() == currLayerOp.getDstOrder() && layerOp.getMemPerm() == currLayerOp.getMemPerm();
}

SmallVector<int64_t> MoveMemPermuteBeforeSlice::getNewSizes(IE::SliceOp, IE::MemPermuteOp layerOp) const {
    mlir::SmallVector<mlir::ShapedTypeComponents> inferredReturnShapes;

    inferPermuteReturnTypeComponents(layerOp.getInput(), layerOp.getMemPerm(), layerOp.getDstOrder(),
                                     inferredReturnShapes, false);

    VPUX_THROW_WHEN(inferredReturnShapes.size() != 1, "Should should be 1 but got {0}", inferredReturnShapes.size());

    auto dims = inferredReturnShapes.front().getDims();

    return mlir::SmallVector<int64_t>{dims.begin(), dims.end()};
}

//
// MoveQuantizeCastBeforeSlice
//

class MoveQuantizeCastBeforeSlice final : public MoveLayerBeforeSlice<IE::QuantizeCastOp> {
public:
    MoveQuantizeCastBeforeSlice(mlir::MLIRContext* ctx, Logger log)
            : MoveLayerBeforeSlice<IE::QuantizeCastOp>(ctx, log) {
    }

public:
    bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                           IE::QuantizeCastOp layerOp) const override;
    bool isBeneficialTransformation(IE::SliceOp, IE::QuantizeCastOp layerOp,
                                    ArrayRef<IE::QuantizeCastOp>) const override;
    bool sameAttributes(IE::QuantizeCastOp layerOp, IE::QuantizeCastOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::QuantizeCastOp layerOp) const override;
};

bool MoveQuantizeCastBeforeSlice::doesSliceAndLayerOpModifySameAxis(IE::SliceOp, ArrayRef<uint64_t> sliceAxes,
                                                                    IE::QuantizeCastOp layerOp) const {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(layerOp.getInput().getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(layerOp.getOutput().getType());

    auto doesSliceModifySameAxis = [&](vpux::NDTypeInterface ndType) {
        if (const auto perAxisType =
                    mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(ndType.getElementType())) {
            const auto axis = perAxisType.getQuantizedDimension();
            return llvm::is_contained(sliceAxes, axis);
        }
        return false;
    };

    return doesSliceModifySameAxis(inType) || doesSliceModifySameAxis(outType);
}

// #E168219 only beneficial for quantile type
bool MoveQuantizeCastBeforeSlice::isBeneficialTransformation(IE::SliceOp, IE::QuantizeCastOp layerOp,
                                                             ArrayRef<IE::QuantizeCastOp>) const {
    auto dstElemType = layerOp.getDstElemType();
    return mlir::isa<mlir::quant::QuantileQuantizedPerAxisType>(dstElemType) ||
           mlir::isa<mlir::quant::QuantileQuantizedType>(dstElemType);
}

bool MoveQuantizeCastBeforeSlice::sameAttributes(IE::QuantizeCastOp layerOp, IE::QuantizeCastOp currLayerOp) const {
    return layerOp.getDstElemType() == currLayerOp.getDstElemType();
}

SmallVector<int64_t> MoveQuantizeCastBeforeSlice::getNewSizes(IE::SliceOp sliceOp, IE::QuantizeCastOp) const {
    return parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
}

//
// MoveReorderBeforeSplit
//
class MoveReorderBeforeSplit final : public mlir::OpRewritePattern<IE::ReorderOp> {
public:
    MoveReorderBeforeSplit(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReorderOp>(ctx), _log(log) {
        setDebugName("MoveReorderBeforeSplit");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReorderOp layerOp, mlir::PatternRewriter& rewriter) const final;

protected:
    Logger _log;
};

mlir::LogicalResult MoveReorderBeforeSplit::matchAndRewrite(IE::ReorderOp layerOp,
                                                            mlir::PatternRewriter& rewriter) const {
    auto maybeSplitOp = layerOp.getInput().template getDefiningOp<IE::SplitOp>();
    if (maybeSplitOp == nullptr) {
        return mlir::failure();
    }

    for (const auto& res : maybeSplitOp.getOutputs()) {
        auto currReorderOp = mlir::dyn_cast_or_null<IE::ReorderOp>(*res.getUsers().begin());
        if (currReorderOp == nullptr || !currReorderOp.getResult().hasOneUse() ||
            layerOp.getDstOrder() != currReorderOp.getDstOrder()) {
            return mlir::failure();
        }
    }

    // From performance perspective, the matching case is the split axis for SplitOp will be in lower memory dim after
    // ReorderOp. For example, SplitOp: split axis = C, ReorderOp: NDHWC -> NCDHW, then split memory dim from d4 to d1
    // will benefit the performance.
    const auto axisInd = maybeSplitOp.getAxisValue().value();
    const auto reorderInOrder = DimsOrder::fromValue(layerOp.getInput());
    const auto reorderOutOrder = DimsOrder::fromValue(layerOp.getOutput());
    const auto memPosInOrder = reorderInOrder.toMemDim(Dim(axisInd));
    const auto memPosOutOrder = reorderOutOrder.toMemDim(Dim(axisInd));
    if (memPosInOrder.ind() < memPosOutOrder.ind()) {
        return mlir::failure();
    }

    // Create new ReorderOp
    const auto dstOrderAttr = layerOp.getDstOrderAttr();
    auto newReorder = rewriter.create<IE::ReorderOp>(maybeSplitOp->getLoc(), maybeSplitOp.getInput(), dstOrderAttr);

    // Create new SplitOp
    auto newSplit = rewriter.create<IE::SplitOp>(maybeSplitOp->getLoc(), newReorder.getOutput(), maybeSplitOp.getAxis(),
                                                 maybeSplitOp.getNumSplitsAttr(), maybeSplitOp.getAxisValueAttr());

    for (const auto& res : maybeSplitOp.getOutputs()) {
        for (auto use : llvm::make_early_inc_range(res.getUsers())) {
            auto reorder = mlir::dyn_cast<IE::ReorderOp>(use);
            rewriter.replaceOp(reorder, newSplit.getResult(res.getResultNumber()));
        }
    }

    return mlir::success();
}

// Convert:

//        Operation
//        |       |
//       Sin     Cos
//        |       |
//     Reorder   Reorder

// To:

//        Operation
//            |
//         Reorder
//         |     |
//        Sin   Cos

//
// MoveReorderBeforeEltwiseOp
//
class MoveReorderBeforeEltwiseOp final : public mlir::OpRewritePattern<IE::ReorderOp> {
public:
    MoveReorderBeforeEltwiseOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ReorderOp>(ctx), _log(log) {
        setDebugName("MoveReorderBeforeEltwiseOp");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReorderOp layerOp, mlir::PatternRewriter& rewriter) const final;

protected:
    Logger _log;
};

mlir::LogicalResult MoveReorderBeforeEltwiseOp::matchAndRewrite(IE::ReorderOp layerOp,
                                                                mlir::PatternRewriter& rewriter) const {
    auto eltwiseOp = layerOp.getInput().getDefiningOp();
    const auto isValidEltwiseOp = [&](mlir::Operation* op) {
        if (op == nullptr) {
            return false;
        }
        if (!op->hasOneUse()) {
            return false;
        }
        if (DimsOrder::fromValue(op->getOperand(0)) != DimsOrder::fromValue(op->getResult(0))) {
            return false;
        }

        return op->hasTrait<IE::EltwiseOp>() && op->getNumOperands() == 1 && op->getNumResults() == 1;
    };

    if (!isValidEltwiseOp(eltwiseOp)) {
        return mlir::failure();
    }

    const auto refInOrder = DimsOrder::fromValue(layerOp.getInput());
    const auto refOutOrder = DimsOrder::fromValue(layerOp.getOutput());
    auto parentOp = eltwiseOp->getOperand(0).getDefiningOp();
    if (parentOp == nullptr) {
        return mlir::failure();
    }
    SmallVector<mlir::Operation*> eltwiseOps;
    SmallVector<IE::ReorderOp> reorderOps;
    for (auto eltwiseUser : parentOp->getUsers()) {
        if (!isValidEltwiseOp(eltwiseUser)) {
            return mlir::failure();
        }

        auto reorderUser = mlir::dyn_cast<IE::ReorderOp>(*eltwiseUser->getUsers().begin());
        if (reorderUser == nullptr) {
            return mlir::failure();
        }
        const auto inOrder = DimsOrder::fromValue(reorderUser.getInput());
        const auto outOrder = DimsOrder::fromValue(reorderUser.getOutput());
        if (refInOrder != inOrder || refOutOrder != outOrder) {
            return mlir::failure();
        }

        auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(eltwiseUser);
        if (iface == nullptr) {
            return mlir::failure();
        }

        // Check if the operation could support the new layout
        auto orderInfo = iface.getLayoutInfo();
        orderInfo.setInput(0, refOutOrder);
        iface.inferLayoutInfo(orderInfo, /*seOpsEnabled=*/false, /*seExperimentalOpsEnabled=*/false);
        if (orderInfo.getInput(0) != refOutOrder || orderInfo.getOutput(0) != refOutOrder) {
            return mlir::failure();
        }
        eltwiseOps.push_back(eltwiseUser);
        reorderOps.push_back(reorderUser);
    }

    if (eltwiseOps.size() < 2) {
        return mlir::failure();
    }

    rewriter.setInsertionPointAfterValue(parentOp->getResult(0));
    auto newReorder =
            rewriter.create<IE::ReorderOp>(layerOp->getLoc(), eltwiseOp->getOperand(0), layerOp.getDstOrderAttr());

    for (auto p : eltwiseOps | indexed) {
        auto reorderOp = reorderOps[p.index()];
        auto eltwiseOp = p.value();
        mlir::IRMapping mapper;
        mapper.map(eltwiseOp->getOperand(0), newReorder.getOutput());
        auto newOp = rewriter.clone(*eltwiseOp, mapper);
        auto newOutType = mlir::cast<NDTypeInterface>(eltwiseOp->getResult(0).getType()).changeDimsOrder(refOutOrder);
        newOp->getResult(0).setType(newOutType);
        rewriter.replaceOp(reorderOp, newOp->getResult(0));
    }

    return mlir::success();
}

//
// UniquifyBranches
//

class UniquifyBranches final : public IE::impl::UniquifyBranchesBase<UniquifyBranches> {
public:
    explicit UniquifyBranches(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void UniquifyBranches::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveExpandBeforeSlice>(&ctx, _log);
    patterns.add<MoveReorderBeforeSlice>(&ctx, _log);
    patterns.add<MoveTransposeBeforeSlice>(&ctx, _log);
    patterns.add<MovePermuteCastBeforeSlice>(&ctx, _log);
    patterns.add<MoveAffineReshapeBeforeSlice>(&ctx, _log);
    patterns.add<MoveMemPermuteBeforeSlice>(&ctx, _log);
    patterns.add<MoveQuantizeCastBeforeSlice>(&ctx, _log);
    patterns.add<MoveReorderBeforeSplit>(&ctx, _log);
    patterns.add<MoveReorderBeforeEltwiseOp>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createUniquifyBranchesPass(Logger log) {
    return std::make_unique<UniquifyBranches>(log);
}
