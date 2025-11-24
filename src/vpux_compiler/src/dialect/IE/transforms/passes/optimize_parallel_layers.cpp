//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <algorithm>

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZEPARALLELLAYERS
#define GEN_PASS_DEF_OPTIMIZEPARALLELLAYERS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool isSliceOnHighestDim(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes) {
    if (sliceAxes.size() != 1) {
        return false;
    }

    auto sliceAxis = sliceAxes[0];
    auto inType = mlir::cast<NDTypeInterface>(sliceOp.getSource().getType());
    auto dimOrder = inType.getDimsOrder();
    auto shape = inType.getShape();

    const auto highestDim = getHighestNonTrivialDim(shape, dimOrder).value_or(Dim(0));
    return checked_cast<uint64_t>(highestDim.ind()) == sliceAxis;
}

bool isSourceFullySlicedWithoutIntervalOrOverlap(mlir::Value source, ArrayRef<IE::SliceOp> sliceOps) {
    if (sliceOps.size() < 2) {
        return false;
    }

    auto firstSlice = sliceOps.front();
    auto firstSliceSizeAttr = firstSlice.getStaticSizesAttr();
    auto hasTheSameSourceAndSize = [&](IE::SliceOp sliceOp) {
        if (sliceOp.getSource() != source) {
            return false;
        }

        if (sliceOp.getStaticSizesAttr() != firstSliceSizeAttr) {
            return false;
        }

        return true;
    };
    if (!llvm::all_of(sliceOps, hasTheSameSourceAndSize)) {
        return false;
    }

    const auto sliceAxes = getSliceAxes(sliceOps.front());
    if (sliceAxes.size() != 1) {
        return false;
    }
    const auto sliceAxis = sliceAxes.front();

    auto compareSliceOps = [sliceAxis](IE::SliceOp a, IE::SliceOp b) {
        auto offsetsA = vpux::parseIntArrayAttr<int64_t>(a.getStaticOffsetsAttr());
        auto offsetsB = vpux::parseIntArrayAttr<int64_t>(b.getStaticOffsetsAttr());

        return offsetsA[sliceAxis] < offsetsB[sliceAxis];
    };

    SmallVector<IE::SliceOp> sortedSliceOps(sliceOps.begin(), sliceOps.end());
    std::sort(sortedSliceOps.begin(), sortedSliceOps.end(), compareSliceOps);

    auto sliceSize = vpux::parseIntArrayAttr<int64_t>(firstSliceSizeAttr)[sliceAxis];
    if (checked_cast<int64_t>(sortedSliceOps.size() * sliceSize) != getShape(source)[Dim(sliceAxis)]) {
        return false;
    }

    for (size_t i = 1; i < sortedSliceOps.size(); ++i) {
        auto currSliceOp = sortedSliceOps[i];
        auto currOffset = vpux::parseIntArrayAttr<int64_t>(currSliceOp.getStaticOffsetsAttr())[sliceAxis];

        auto prevSliceOp = sortedSliceOps[i - 1];
        auto prevOffset = vpux::parseIntArrayAttr<int64_t>(prevSliceOp.getStaticOffsetsAttr())[sliceAxis];
        if (currOffset != prevOffset + sliceSize) {
            return false;
        }
    }

    return true;
}

template <typename ConcreteOp>
class MoveLayerBeforeSlice : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MoveLayerBeforeSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    virtual bool sameAttributes(ConcreteOp layerOp, ConcreteOp currLayerOp) const = 0;
    virtual mlir::Operation* createNewLayerOp(ArrayRef<ConcreteOp> siblingLayerOps, IE::SliceOp sliceOp,
                                              mlir::PatternRewriter& rewriter) const;
    virtual SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, ConcreteOp layerOp) const;
    virtual SmallVector<int64_t> getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                               ConcreteOp layerOp) const;
    virtual bool isLegalTransformation(IE::SliceOp sliceOp, ConcreteOp layerOp,
                                       ArrayRef<ConcreteOp> siblingLayerOps) const = 0;

protected:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult MoveLayerBeforeSlice<ConcreteOp>::matchAndRewrite(ConcreteOp layerOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    SmallVector<IE::SliceOp> sliceOps;
    for (auto operand : layerOp->getOperands()) {
        auto sliceOp = operand.template getDefiningOp<IE::SliceOp>();
        if (sliceOp != nullptr) {
            sliceOps.push_back(sliceOp);
        }
    }
    if (sliceOps.size() != 1) {
        return mlir::failure();
    }

    auto maybeSliceOp = sliceOps.front();

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

    auto newLayerOp = createNewLayerOp(siblingLayerOps, maybeSliceOp, rewriter);
    _log.trace("Create new layer op: {0}", *newLayerOp);

    auto newSizes = getNewSizes(maybeSliceOp, layerOp);
    const auto sliceAxes = getSliceAxes(maybeSliceOp);
    SmallVector<std::pair<ConcreteOp, IE::SliceOp>> opsVec;
    for (auto user : root.getUsers()) {
        auto sliceOp = mlir::dyn_cast<IE::SliceOp>(user);
        if (sliceOp == nullptr) {
            continue;
        }

        auto currLayerOp = mlir::dyn_cast<ConcreteOp>(*sliceOp.getResult().getUsers().begin());
        auto newOffsets = getNewOffsets(sliceOp, sliceAxes, currLayerOp);
        auto newSlice = rewriter.create<IE::SliceOp>(takeOpLoc(currLayerOp, "as_slice"), newLayerOp->getResult(0),
                                                     getIntArrayAttr(newLayerOp->getContext(), newOffsets),
                                                     getIntArrayAttr(newLayerOp->getContext(), newSizes));
        _log.trace("Create new Slice: {0}", newSlice);

        opsVec.push_back({currLayerOp, newSlice});
    }

    for (auto p : opsVec) {
        rewriter.replaceOp(p.first, p.second.getResult());
    }

    return mlir::success();
}

template <typename ConcreteOp>
mlir::Operation* MoveLayerBeforeSlice<ConcreteOp>::createNewLayerOp(ArrayRef<ConcreteOp> siblingLayerOps,
                                                                    IE::SliceOp sliceOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    auto layerOp = siblingLayerOps.front();
    mlir::IRMapping mapper;
    mapper.map(layerOp->getOperands(), ArrayRef({sliceOp.getSource()}));

    rewriter.setInsertionPointAfterValue(sliceOp.getSource());
    auto* newLayerOp = rewriter.clone(*layerOp.getOperation(), mapper);
    extendOpLoc(newLayerOp, "new_layer");
    vpux::inferReturnTypes(newLayerOp, vpux::InferShapedTypeMode::ALL);
    return newLayerOp;
}

template <typename ConcreteOp>
SmallVector<int64_t> MoveLayerBeforeSlice<ConcreteOp>::getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t>,
                                                                     ConcreteOp) const {
    return parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
}

template <typename ConcreteOp>
SmallVector<int64_t> MoveLayerBeforeSlice<ConcreteOp>::getNewSizes(IE::SliceOp sliceOp, ConcreteOp) const {
    return vpux::parseIntArrayAttr<int64_t>(sliceOp.getStaticSizesAttr());
}

//
// MoveEltwiseBeforeSlice
//
template <typename EltwiseOp>
class MoveEltwiseBeforeSlice final : public MoveLayerBeforeSlice<EltwiseOp> {
    using MoveLayerBeforeSlice<EltwiseOp>::_log;

public:
    MoveEltwiseBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<EltwiseOp>(ctx, log) {
    }

public:
    bool isLegalTransformation(IE::SliceOp sliceOp, EltwiseOp layerOp,
                               ArrayRef<EltwiseOp> siblingLayerOps) const override;
    bool sameAttributes(EltwiseOp layerOp, EltwiseOp currLayerOp) const override;
    mlir::Operation* createNewLayerOp(ArrayRef<EltwiseOp> siblingLayerOps, IE::SliceOp sliceOp,
                                      mlir::PatternRewriter& rewriter) const override;

private:
    mlir::Value permuteCastToIdentityOrder(mlir::Value input, mlir::PatternRewriter& rewriter) const;
};

template <typename EltwiseOp>
bool MoveEltwiseBeforeSlice<EltwiseOp>::isLegalTransformation(IE::SliceOp sliceOp, EltwiseOp,
                                                              ArrayRef<EltwiseOp> siblingLayerOps) const {
    const auto sliceAxes = getSliceAxes(sliceOp);
    if (sliceAxes.size() != 1) {
        return false;
    }

    auto isCstAndSplat = [](mlir::Value value) {
        auto cstOp = mlir::dyn_cast_or_null<Const::DeclareOp>(value.getDefiningOp());
        return cstOp != nullptr && cstOp.getContentAttr().isSplat();
    };
    auto getSourceSliceOp = [&](auto eltwiseOp) -> IE::SliceOp {
        auto input1 = eltwiseOp.getInput1();
        auto input2 = eltwiseOp.getInput2();

        IE::SliceOp inputSliceOp = nullptr;
        auto slice1 = mlir::dyn_cast_or_null<IE::SliceOp>(input1.getDefiningOp());
        auto slice2 = mlir::dyn_cast_or_null<IE::SliceOp>(input2.getDefiningOp());
        if (slice1 != nullptr && isCstAndSplat(input2)) {
            inputSliceOp = slice1;
        } else if (slice2 != nullptr && isCstAndSplat(input1)) {
            inputSliceOp = slice2;
        }

        return inputSliceOp;
    };
    // ensure all sibling eltwise ops has sliced source and get the sliceOp list
    SmallVector<IE::SliceOp> sliceOps;
    for (auto eltwiseOp : siblingLayerOps) {
        auto inputSliceOp = getSourceSliceOp(eltwiseOp);
        if (inputSliceOp == nullptr) {
            return false;
        }
        sliceOps.push_back(inputSliceOp);
    }

    return isSourceFullySlicedWithoutIntervalOrOverlap(sliceOp.getSource(), sliceOps);
}

template <typename EltwiseOp>
bool MoveEltwiseBeforeSlice<EltwiseOp>::sameAttributes(EltwiseOp layerOp, EltwiseOp currLayerOp) const {
    return layerOp.getInput1().getType() == currLayerOp.getInput1().getType() &&
           layerOp.getInput2().getType() == currLayerOp.getInput2().getType();
}

template <typename EltwiseOp>
mlir::Value MoveEltwiseBeforeSlice<EltwiseOp>::permuteCastToIdentityOrder(mlir::Value input,
                                                                          mlir::PatternRewriter& rewriter) const {
    const auto ctx = rewriter.getContext();
    const auto inputShape = getShape(input);
    const auto identityOrder = DimsOrder::fromNumDims(inputShape.size());
    const auto identityOrderMap = identityOrder.toAffineMap(ctx);
    return rewriter.createOrFold<IE::PermuteCastOp>(
            appendLoc(input.getLoc(), "_identity_permute_cast"), input, identityOrderMap,
            mlir::AffineMap::getMultiDimIdentityMap(identityOrder.numDims(), ctx));
}

template <typename EltwiseOp>
mlir::Operation* MoveEltwiseBeforeSlice<EltwiseOp>::createNewLayerOp(ArrayRef<EltwiseOp> siblingLayerOps,
                                                                     IE::SliceOp sliceOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();

    auto firstMultiply = siblingLayerOps.front();
    auto source = sliceOp.getSource();

    // Cast source to canonical order
    auto canonicalPermuteCast = permuteCastToIdentityOrder(source, rewriter);

    // Reshape the size of the Slice dimension to a new higher dimension
    auto dimsOrder = mlir::cast<NDTypeInterface>(sliceOp.getResult().getType()).getDimsOrder();
    const auto sliceAxes = getSliceAxes(sliceOp);
    const auto sliceDim = sliceAxes.front();
    const auto sliceDimPos = dimsOrder.dimPos(Dim(sliceDim));
    const auto canonicalSourceShape = getShape(canonicalPermuteCast);
    auto targetShape = to_small_vector(canonicalSourceShape);
    VPUX_THROW_UNLESS(targetShape[sliceDimPos] % checked_cast<int64_t>(siblingLayerOps.size()) == 0,
                      "Size {0} can't be divided by {1}", targetShape[sliceDimPos], siblingLayerOps.size());
    targetShape[sliceDimPos] = targetShape[sliceDimPos] / checked_cast<int64_t>(siblingLayerOps.size());
    targetShape.insert(targetShape.begin() + sliceDimPos, checked_cast<int64_t>(siblingLayerOps.size()));

    auto newLhs = rewriter.create<IE::ReshapeOp>(appendLoc(source.getLoc(), "_reshape_for_lhs"), canonicalPermuteCast,
                                                 nullptr, false,
                                                 getIntArrayAttr(rewriter.getContext(), ShapeRef(targetShape)));

    // To ensure we merge multiply ops in the correct order, we need to sort them by the offset of the slice
    auto compareSliceOps = [sliceDim](IE::SliceOp a, IE::SliceOp b) {
        auto offsetsA = vpux::parseIntArrayAttr<int64_t>(a.getStaticOffsetsAttr());
        auto offsetsB = vpux::parseIntArrayAttr<int64_t>(b.getStaticOffsetsAttr());

        return offsetsA[sliceDim] < offsetsB[sliceDim];
    };
    auto sortedSiblingLayerOps = SmallVector<EltwiseOp>(siblingLayerOps.begin(), siblingLayerOps.end());
    llvm::sort(sortedSiblingLayerOps, [&](EltwiseOp a, EltwiseOp b) {
        auto sliceA = mlir::dyn_cast_or_null<IE::SliceOp>(a.getInput1().getDefiningOp());
        if (sliceA == nullptr) {
            sliceA = mlir::dyn_cast_or_null<IE::SliceOp>(a.getInput2().getDefiningOp());
        }
        auto sliceB = mlir::dyn_cast_or_null<IE::SliceOp>(b.getInput1().getDefiningOp());
        if (sliceB == nullptr) {
            sliceB = mlir::dyn_cast_or_null<IE::SliceOp>(b.getInput2().getDefiningOp());
        }
        return compareSliceOps(sliceA, sliceB);
    });
    // Concat rhs
    SmallVector<mlir::Value> concatRhs;
    for (auto multiply : sortedSiblingLayerOps) {
        auto input2 = multiply.getInput2();
        // Cast rhs to canonical order
        auto rhsPermuteCast = permuteCastToIdentityOrder(input2, rewriter);

        auto input2TargetShape = SmallVector<int64_t>(targetShape.size(), 1);
        // Reshape input2 to handle the case when 2 inputs have different ranks
        auto input2Reshape = rewriter.createOrFold<IE::ReshapeOp>(
                appendLoc(source.getLoc(), "_reshape_rhs"), rhsPermuteCast, nullptr, false,
                getIntArrayAttr(rewriter.getContext(), ShapeRef(input2TargetShape)));

        concatRhs.push_back(input2Reshape);
    }
    auto newRhs =
            rewriter.create<IE::ConcatOp>(appendLoc(source.getLoc(), "_concat_for_rhs"), concatRhs, Dim(sliceDimPos));

    // Create new eltwiseOp
    auto multiply =
            rewriter.create<EltwiseOp>(appendLoc(source.getLoc(), "_merged_eltwise"), newLhs, newRhs,
                                       firstMultiply.getAutoBroadcastAttr(), nullptr, nullptr, nullptr, nullptr);

    // Reshape to the original source MemShape
    auto outputReshape =
            rewriter.create<IE::ReshapeOp>(appendLoc(multiply.getLoc(), "_output_reshape"), multiply, nullptr, false,
                                           getIntArrayAttr(rewriter.getContext(), getMemShape(source)));

    // Cast to the original dims order
    const auto sourceShape = getShape(source);
    auto outPermuteCast = rewriter.createOrFold<IE::PermuteCastOp>(
            appendLoc(multiply.getLoc(), "_output_permute_cast"), outputReshape, dimsOrder.toAffineMap(ctx),
            mlir::AffineMap::getMultiDimIdentityMap(sourceShape.size(), ctx));

    _log.trace("[{0}] Successfully merged parallel Eltwise operations", this->getDebugName());
    return outPermuteCast.getDefiningOp();
}

//
// MoveReshapeBeforeSlice
//

class MoveReshapeBeforeSlice final : public MoveLayerBeforeSlice<IE::ReshapeOp> {
public:
    MoveReshapeBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<IE::ReshapeOp>(ctx, log) {
    }

public:
    bool isLegalTransformation(IE::SliceOp sliceOp, IE::ReshapeOp layerOp,
                               ArrayRef<IE::ReshapeOp> siblingLayerOps) const override;
    bool sameAttributes(IE::ReshapeOp layerOp, IE::ReshapeOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::ReshapeOp layerOp) const override;
    SmallVector<int64_t> getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                       IE::ReshapeOp layerOp) const override;

    mlir::Operation* createNewLayerOp(ArrayRef<IE::ReshapeOp> siblingLayerOps, IE::SliceOp sliceOp,
                                      mlir::PatternRewriter& rewriter) const override;
};

// Check if the shapeA vector ends with the shapeB vector and all preceding elements are 1
bool checkShapes(ArrayRef<int64_t> shapeA, ArrayRef<int64_t> shapeB) {
    if (shapeA.size() < shapeB.size()) {
        return false;
    }

    for (size_t i = 0; i < shapeB.size(); ++i) {
        if (shapeA[shapeA.size() - shapeB.size() + i] != shapeB[i]) {
            return false;
        }
    }

    for (size_t i = 0; i < shapeA.size() - shapeB.size(); ++i) {
        if (shapeA[i] != 1) {
            return false;
        }
    }

    return true;
}

bool MoveReshapeBeforeSlice::isLegalTransformation(IE::SliceOp sliceOp, IE::ReshapeOp layerOp,
                                                   ArrayRef<IE::ReshapeOp>) const {
    if (!layerOp.getShapeValue().has_value()) {
        return false;
    }

    // Transformation is legal when the shapes are changed by either inserting or dropping dimensions of size 1.
    // For example, below 2 cases are both legal:
    // Reshape from 1x2x3 to 2x3
    // Reshape from 2x3 to 1x2x3
    auto inShape = to_small_vector(getShape(layerOp.getInput()));
    auto outShape = to_small_vector(getShape(layerOp.getOutput()));
    if (checkShapes(inShape, outShape)) {
        // Ensure the source shape has enough dimensions of size 1 to strip
        auto sourceShape = getShape(sliceOp.getSource());
        size_t dimOneCnt = std::count_if(sourceShape.begin(), sourceShape.end(), [](int64_t size) {
            return size == 1;
        });
        auto diffRanks = inShape.size() - outShape.size();
        if (diffRanks <= dimOneCnt) {
            return true;
        }
    }

    if (checkShapes(outShape, inShape)) {
        return true;
    }

    return false;
}

bool MoveReshapeBeforeSlice::sameAttributes(IE::ReshapeOp layerOp, IE::ReshapeOp currLayerOp) const {
    return layerOp.getShapeValue() == currLayerOp.getShapeValue();
}

SmallVector<int64_t> MoveReshapeBeforeSlice::getNewSizes(IE::SliceOp, IE::ReshapeOp layerOp) const {
    return vpux::parseIntArrayAttr<int64_t>(layerOp.getShapeValue().value());
}

// Remove n elements with value 1 and record their indices
SmallVector<int64_t> removeOnesAndRecordIndices(SmallVector<int64_t>& vec, size_t n) {
    size_t count = 0;
    auto it = vec.begin();
    int64_t index = 0;
    SmallVector<int64_t> removedIndices;
    while (it != vec.end() && count < n) {
        if (*it == 1) {
            removedIndices.push_back(index);
            it = vec.erase(it);
            count++;
        } else {
            ++it;
        }
        ++index;
    }

    return removedIndices;
}

// Remove elements at specified indices
void removeElementsAtIndices(SmallVector<int64_t>& vec, ArrayRef<int64_t> indices) {
    SmallVector<int64_t> sortedIndices = SmallVector<int64_t>(indices.begin(), indices.end());
    std::sort(sortedIndices.begin(), sortedIndices.end(), std::greater<int64_t>());

    for (int64_t index : sortedIndices) {
        if (index >= 0 && index < checked_cast<int64_t>(vec.size())) {
            vec.erase(vec.begin() + index);
        }
    }
}

SmallVector<int64_t> MoveReshapeBeforeSlice::getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t>,
                                                           IE::ReshapeOp layerOp) const {
    auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());

    auto inShape = getShape(layerOp.getInput());
    auto outShape = getShape(layerOp.getOutput());
    auto sourceShape = to_small_vector(getShape(sliceOp.getSource()));
    if (checkShapes(inShape, outShape)) {
        // Shapes are changed by dropping dimensions of size 1
        auto diffRanks = inShape.size() - outShape.size();
        auto removedIndices = removeOnesAndRecordIndices(sourceShape, diffRanks);
        removeElementsAtIndices(sliceOffsets, removedIndices);
    } else {
        // Shapes are changed by inserting dimensions of size 1
        auto diffRanks = outShape.size() - inShape.size();
        sliceOffsets.insert(sliceOffsets.begin(), diffRanks, 1);
    }

    return sliceOffsets;
}

mlir::Operation* MoveReshapeBeforeSlice::createNewLayerOp(ArrayRef<IE::ReshapeOp> siblingLayerOps, IE::SliceOp sliceOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto layerOp = siblingLayerOps.front();
    auto inShape = getShape(layerOp.getInput());
    auto outShape = getShape(layerOp.getOutput());

    auto targetShape = to_small_vector(getShape(sliceOp.getSource()));

    if (checkShapes(inShape, outShape)) {
        // Shapes are changed by dropping dimensions of size 1
        auto diffRanks = inShape.size() - outShape.size();
        removeOnesAndRecordIndices(targetShape, diffRanks);
    } else {
        // Shapes are changed by inserting dimensions of size 1
        auto diffRanks = outShape.size() - inShape.size();
        targetShape.insert(targetShape.begin(), diffRanks, 1);
    }

    rewriter.setInsertionPointAfterValue(sliceOp.getSource());
    return rewriter.create<IE::ReshapeOp>(takeOpLoc(layerOp, "as_reshape"), sliceOp.getSource(), nullptr, false,
                                          vpux::getIntArrayAttr(rewriter, targetShape));
}

//
// MoveFCBeforeSlice
//

class MoveFCBeforeSlice final : public MoveLayerBeforeSlice<IE::FullyConnectedOp> {
public:
    MoveFCBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<IE::FullyConnectedOp>(ctx, log) {
    }

public:
    bool isLegalTransformation(IE::SliceOp sliceOp, IE::FullyConnectedOp layerOp,
                               ArrayRef<IE::FullyConnectedOp> siblingLayerOps) const override;
    bool sameAttributes(IE::FullyConnectedOp layerOp, IE::FullyConnectedOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::FullyConnectedOp fcOp) const override;
    SmallVector<int64_t> getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                       IE::FullyConnectedOp fcOp) const override;

    mlir::Operation* createNewLayerOp(ArrayRef<IE::FullyConnectedOp> siblingLayerOps, IE::SliceOp sliceOp,
                                      mlir::PatternRewriter& rewriter) const override;

private:
    const size_t VALID_FC_SHAPE_RANK = 2;
};

bool MoveFCBeforeSlice::isLegalTransformation(IE::SliceOp sliceOp, IE::FullyConnectedOp layerOp,
                                              ArrayRef<IE::FullyConnectedOp> siblingLayerOps) const {
    // Only move FullyConnectedOp when slice on the highest dimension
    const auto sliceAxes = getSliceAxes(sliceOp);
    if (!isSliceOnHighestDim(sliceOp, sliceAxes)) {
        return false;
    }

    // SliceOp should be the weights of FullyConnectedOp
    if (layerOp.getWeights().getDefiningOp() != sliceOp) {
        return false;
    }

    auto inputSource = layerOp.getInput();
    SmallVector<IE::SliceOp> sliceOps;
    for (auto fcOp : siblingLayerOps) {
        // Currently, FC with bias is not supported
        if (fcOp.getBias() != nullptr) {
            return false;
        }

        // All FC operations should share the same lhs input
        if (fcOp.getInput() != inputSource) {
            return false;
        }

        // All FC operations's weights should be sliced and have a valid 2D shape
        auto weights = fcOp.getWeights();
        auto weightsSliceOp = weights.getDefiningOp<IE::SliceOp>();
        if (weightsSliceOp == nullptr || getShape(weights).size() != VALID_FC_SHAPE_RANK ||
            getShape(fcOp.getOutput()).size() != VALID_FC_SHAPE_RANK) {
            return false;
        }

        sliceOps.push_back(weightsSliceOp);
    }

    return isSourceFullySlicedWithoutIntervalOrOverlap(sliceOp.getSource(), sliceOps);
}

bool MoveFCBeforeSlice::sameAttributes(IE::FullyConnectedOp, IE::FullyConnectedOp) const {
    return true;
}

SmallVector<int64_t> MoveFCBeforeSlice::getNewSizes(IE::SliceOp, IE::FullyConnectedOp fcOp) const {
    auto fcOutShape = getShape(fcOp.getOutput());
    return to_small_vector(fcOutShape);
}

SmallVector<int64_t> MoveFCBeforeSlice::getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t>,
                                                      IE::FullyConnectedOp fcOp) const {
    const auto sliceAxes = getSliceAxes(sliceOp);
    const auto sliceAxis = sliceAxes.front();
    auto origOffsets = vpux::parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr());
    auto fcOutShape = getShape(fcOp.getOutput());
    SmallVector<int64_t> newOffsets = SmallVector<int64_t>(fcOutShape.size(), 0);
    if (sliceAxis == 1) {
        newOffsets[1] = origOffsets[sliceAxis] / (getShape(sliceOp.getResult())[Dim(sliceAxis)]);
    } else if (sliceAxis == 0) {
        newOffsets[1] = origOffsets[sliceAxis];
    }

    return newOffsets;
}

mlir::Operation* MoveFCBeforeSlice::createNewLayerOp(ArrayRef<IE::FullyConnectedOp> siblingLayerOps,
                                                     IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const {
    auto fcOp = siblingLayerOps.front();
    auto inputSource = fcOp.getInput();
    auto weightsSource = sliceOp.getSource();
    auto weightsSourceShape = getShape(weightsSource);

    const auto sliceAxes = getSliceAxes(sliceOp);
    const auto sliceAxis = sliceAxes.front();

    mlir::Value newWeights;
    if (sliceAxis == 1) {
        auto targetShape = to_small_vector(weightsSourceShape);
        targetShape[0] = checked_cast<int64_t>(siblingLayerOps.size());
        targetShape[1] = weightsSourceShape.back() / checked_cast<int64_t>(siblingLayerOps.size());
        newWeights = rewriter.create<IE::ReshapeOp>(weightsSource.getLoc(), weightsSource, nullptr, false,
                                                    getIntArrayAttr(rewriter.getContext(), ShapeRef(targetShape)));
    } else if (sliceAxis == 0) {
        newWeights = weightsSource;
    }

    auto newFullyConnected =
            rewriter.create<IE::FullyConnectedOp>(takeOpLoc(fcOp, "merged"), inputSource, newWeights, fcOp.getBias());

    return newFullyConnected;
}

//
// MoveTanhBeforeSlice
//

class MoveTanhBeforeSlice final : public MoveLayerBeforeSlice<IE::TanhOp> {
public:
    MoveTanhBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<IE::TanhOp>(ctx, log) {
    }

public:
    bool isLegalTransformation(IE::SliceOp sliceOp, IE::TanhOp layerOp,
                               ArrayRef<IE::TanhOp> siblingLayerOps) const override;
    bool sameAttributes(IE::TanhOp layerOp, IE::TanhOp currLayerOp) const override;
};

bool MoveTanhBeforeSlice::isLegalTransformation(IE::SliceOp, IE::TanhOp, ArrayRef<IE::TanhOp>) const {
    return true;
}

bool MoveTanhBeforeSlice::sameAttributes(IE::TanhOp, IE::TanhOp) const {
    return true;
}

template <typename ConcreteOp>
class MoveLayerAfterConcat : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    MoveLayerAfterConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    virtual bool validateConcat(IE::ConcatOp) const = 0;
    virtual SmallVector<ConcreteOp> getValidInputs(IE::ConcatOp concatOp) const = 0;
    virtual void createNewSubgraphAndReplace(ArrayRef<ConcreteOp> siblingLayerOps, IE::ConcatOp origOp,
                                             mlir::PatternRewriter& rewriter) const = 0;

protected:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult MoveLayerAfterConcat<ConcreteOp>::matchAndRewrite(IE::ConcatOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    if (!validateConcat(origOp)) {
        return mlir::failure();
    }

    auto siblingOps = getValidInputs(origOp);
    if (siblingOps.empty()) {
        return mlir::failure();
    }

    createNewSubgraphAndReplace(siblingOps, origOp, rewriter);

    return mlir::success();
}

//
// MoveReshapeAfterConcat
//
class MoveReshapeAfterConcat final : public MoveLayerAfterConcat<IE::ReshapeOp> {
public:
    MoveReshapeAfterConcat(mlir::MLIRContext* ctx, Logger log): MoveLayerAfterConcat<IE::ReshapeOp>(ctx, log) {
    }

public:
    bool validateConcat(IE::ConcatOp) const override;
    SmallVector<IE::ReshapeOp> getValidInputs(IE::ConcatOp concatOp) const override;
    void createNewSubgraphAndReplace(ArrayRef<IE::ReshapeOp> siblingLayerOps, IE::ConcatOp origOp,
                                     mlir::PatternRewriter& rewriter) const override;
};

bool MoveReshapeAfterConcat::validateConcat(IE::ConcatOp concatOp) const {
    auto concatAxis = IE::getConcatAxis(concatOp);
    return concatAxis.has_value();
}

SmallVector<IE::ReshapeOp> MoveReshapeAfterConcat::getValidInputs(IE::ConcatOp concatOp) const {
    auto origConcatAxis = IE::getConcatAxis(concatOp);
    SmallVector<IE::ReshapeOp> reshapeOps;
    for (auto input : concatOp.getInputs()) {
        auto inputOp = input.getDefiningOp();
        auto reshapeOp = mlir::dyn_cast_or_null<IE::ReshapeOp>(inputOp);
        if (reshapeOp == nullptr || !reshapeOp->hasOneUse()) {
            return {};
        }

        reshapeOps.push_back(reshapeOp);
    }

    if (reshapeOps.size() < 2) {
        return {};
    }

    // All Reshape ops should be the same
    auto firstReshape = reshapeOps.front();
    auto areTheSameReshape = [&](IE::ReshapeOp reshapeOp) {
        return getShape(reshapeOp.getInput()) == getShape(firstReshape.getInput()) &&
               getShape(reshapeOp.getOutput()) == getShape(firstReshape.getOutput());
    };
    if (!llvm::all_of(reshapeOps, areTheSameReshape)) {
        return {};
    }

    // Transformation is supported when the shapes are changed by either inserting or dropping dimensions of size 1.
    const auto inShape = to_small_vector(getShape(firstReshape.getInput()));
    const auto outShape = to_small_vector(getShape(firstReshape.getOutput()));
    if (!checkShapes(inShape, outShape) && !checkShapes(outShape, inShape)) {
        return {};
    }

    // Ensure we can find a new concat axis on the original input.
    // For example, consider the following scenario:
    // - The original input tensor has a shape of 1x2.
    // - Each input is reshaped to a new shape of 1x1x1x2.
    // - The original concat axis is d1.
    // The new concat axis is calculated by origConcatAxis - diffRanks, which is -1 in this case.
    // Therefore, the transformation is illegal because we can't find a valid new concat axis on the original input.
    //      1x2        1x2
    //       |          |
    //    Reshape    Reshape
    //       |          |
    //   1x1x1x2     1x1x1x2
    //      \          /
    //          Concat
    //             |
    //          1x2x1x2
    //              |
    if (checkShapes(outShape, inShape)) {
        auto diffRanks = outShape.size() - inShape.size();
        if (origConcatAxis.value().ind() < checked_cast<int64_t>(diffRanks)) {
            return {};
        }
    }

    return reshapeOps;
}

void MoveReshapeAfterConcat::createNewSubgraphAndReplace(ArrayRef<IE::ReshapeOp> siblingLayerOps, IE::ConcatOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> concatInputs;
    for (auto reshapeOp : siblingLayerOps) {
        concatInputs.push_back(reshapeOp.getInput());
    }

    auto origConcatAxis = IE::getConcatAxis(origOp);
    auto newConcatAxis = origConcatAxis.value();

    auto firstReshape = siblingLayerOps.front();
    auto inShape = getShape(firstReshape.getInput());
    auto outShape = getShape(firstReshape.getOutput());
    SmallVector<int64_t> targetShape = to_small_vector(getShape(origOp.getOutput()));
    if (checkShapes(inShape, outShape)) {
        // Shapes are changed by dropping dimensions of size 1
        auto diffRanks = inShape.size() - outShape.size();
        newConcatAxis = Dim(origConcatAxis.value().ind() + diffRanks);
    } else {
        // Shapes are changed by inserting dimensions of size 1
        auto diffRanks = outShape.size() - inShape.size();
        VPUX_THROW_WHEN(origConcatAxis.value().ind() < checked_cast<int64_t>(diffRanks),
                        "Incompatible Concat {0} with Reshape {1}", origOp, firstReshape);
        newConcatAxis = Dim(origConcatAxis.value().ind() - diffRanks);
    }
    auto newConcat = rewriter.create<IE::ConcatOp>(origOp->getLoc(), concatInputs, newConcatAxis);

    auto newReshape = rewriter.create<IE::ReshapeOp>(takeOpLoc(firstReshape, "new_reshape"), newConcat, nullptr, false,
                                                     getIntArrayAttr(rewriter.getContext(), ShapeRef(targetShape)));

    rewriter.replaceOp(origOp, newReshape.getOutput());
}

//
// MoveFCAfterConcat
//
class MoveFCAfterConcat final : public MoveLayerAfterConcat<IE::FullyConnectedOp> {
public:
    MoveFCAfterConcat(mlir::MLIRContext* ctx, Logger log): MoveLayerAfterConcat<IE::FullyConnectedOp>(ctx, log) {
    }

public:
    bool validateConcat(IE::ConcatOp) const override;
    SmallVector<IE::FullyConnectedOp> getValidInputs(IE::ConcatOp concatOp) const override;
    void createNewSubgraphAndReplace(ArrayRef<IE::FullyConnectedOp> siblingLayerOps, IE::ConcatOp origOp,
                                     mlir::PatternRewriter& rewriter) const override;

private:
    const size_t VALID_FC_SHAPE_RANK = 2;
};

bool MoveFCAfterConcat::validateConcat(IE::ConcatOp concatOp) const {
    auto concatAxis = IE::getConcatAxis(concatOp);
    return concatAxis.has_value();
}

bool isLhsConcatBeneficial(IE::FullyConnectedOp fcOp) {
    // This transformation moves parallel FullyConnected layers after Concat by concatenating the FC lhs input.
    // It is beneficial when new concat size is smaller.
    auto outputShape = getShape(fcOp.getOutput());
    auto inputShape = getShape(fcOp.getInput());
    if (inputShape.totalSize() < outputShape.totalSize()) {
        return true;
    }

    // When the FC lhs input is a Softmax with a single non-one dimension, it is impossible to fully utilize ACT-SHAVEs.
    // Therefore, it is beneficial to move FC and Softmax after Concat.
    IE::SoftMaxOp softmaxOp = nullptr;
    if (auto maybeReshapeOp = fcOp.getInput().getDefiningOp<IE::ReshapeOp>()) {
        softmaxOp = maybeReshapeOp.getInput().getDefiningOp<IE::SoftMaxOp>();
    } else {
        softmaxOp = fcOp.getInput().getDefiningOp<IE::SoftMaxOp>();
    }

    if (softmaxOp == nullptr) {
        return false;
    }

    auto softmaxShape = getShape(softmaxOp.getOutput());
    auto nonOneDimCnt = std::count_if(softmaxShape.begin(), softmaxShape.end(), [](int64_t size) {
        return size > 1;
    });

    return nonOneDimCnt == 1;
}

SmallVector<IE::FullyConnectedOp> MoveFCAfterConcat::getValidInputs(IE::ConcatOp concatOp) const {
    SmallVector<IE::FullyConnectedOp> fcOps;
    mlir::Value rhsRoot = nullptr;
    std::optional<ShapeRef> lhsShape = std::nullopt;
    for (auto input : concatOp.getInputs()) {
        auto inputOp = input.getDefiningOp();
        auto fcOp = mlir::dyn_cast_or_null<IE::FullyConnectedOp>(inputOp);
        if (fcOp == nullptr || !fcOp->hasOneUse()) {
            return {};
        }

        if (fcOp.getBias() != nullptr) {
            return {};
        }

        // All FC ops should have the same weights
        if (rhsRoot == nullptr) {
            rhsRoot = fcOp.getWeights();
        } else {
            if (rhsRoot != fcOp.getWeights()) {
                return {};
            }
        }

        // All FC ops should have the same lhs shape
        if (!lhsShape.has_value()) {
            lhsShape = getShape(fcOp.getInput());
            if (lhsShape.value().size() != VALID_FC_SHAPE_RANK) {
                return {};
            }

            if (!isLhsConcatBeneficial(fcOp)) {
                return {};
            }
        } else {
            if (lhsShape != getShape(fcOp.getInput())) {
                return {};
            }
        }

        fcOps.push_back(fcOp);
    }

    if (fcOps.size() < 2) {
        return {};
    }

    return fcOps;
}

void MoveFCAfterConcat::createNewSubgraphAndReplace(ArrayRef<IE::FullyConnectedOp> siblingLayerOps, IE::ConcatOp origOp,
                                                    mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> concatInputs;
    for (auto fcOp : siblingLayerOps) {
        concatInputs.push_back(fcOp.getInput());
    }
    auto newLhs = rewriter.create<IE::ConcatOp>(origOp.getLoc(), concatInputs, Dim(0));

    auto firstFC = siblingLayerOps.front();
    const auto rhsRoot = firstFC.getWeights();
    auto newFullyConnected =
            rewriter.create<IE::FullyConnectedOp>(takeOpLoc(firstFC, "new_fc"), newLhs, rhsRoot, firstFC.getBias());

    auto targetShape = getShape(origOp.getOutput());
    auto outputReshape = rewriter.createOrFold<IE::ReshapeOp>(
            appendLoc(origOp->getLoc(), "_output_reshape"), newFullyConnected.getOutput(), nullptr, false,
            getIntArrayAttr(rewriter.getContext(), ShapeRef(targetShape)));

    rewriter.replaceOp(origOp, outputReshape);
}

//
// MoveSoftmaxAfterConcat
//
class MoveSoftmaxAfterConcat final : public MoveLayerAfterConcat<IE::SoftMaxOp> {
public:
    MoveSoftmaxAfterConcat(mlir::MLIRContext* ctx, Logger log): MoveLayerAfterConcat<IE::SoftMaxOp>(ctx, log) {
    }

public:
    bool validateConcat(IE::ConcatOp) const override;
    SmallVector<IE::SoftMaxOp> getValidInputs(IE::ConcatOp concatOp) const override;
    void createNewSubgraphAndReplace(ArrayRef<IE::SoftMaxOp> siblingLayerOps, IE::ConcatOp origOp,
                                     mlir::PatternRewriter& rewriter) const override;
};

bool MoveSoftmaxAfterConcat::validateConcat(IE::ConcatOp concatOp) const {
    auto concatAxis = IE::getConcatAxis(concatOp);
    return concatAxis.has_value();
}

SmallVector<IE::SoftMaxOp> MoveSoftmaxAfterConcat::getValidInputs(IE::ConcatOp concatOp) const {
    auto concatAxis = IE::getConcatAxis(concatOp);
    if (!concatAxis.has_value()) {
        return {};
    }

    SmallVector<IE::SoftMaxOp> softmaxOps;
    std::optional<int64_t> softmaxAxis = std::nullopt;
    for (auto input : concatOp.getInputs()) {
        auto inputOp = input.getDefiningOp();
        auto softmax = mlir::dyn_cast_or_null<IE::SoftMaxOp>(inputOp);
        if (softmax == nullptr) {
            return {};
        }

        // All SoftMax ops should have the same axis
        // Softmax axis must be different with Concat axis
        auto inputType = mlir::cast<vpux::NDTypeInterface>(softmax.getInput().getType());
        if (softmaxAxis == std::nullopt) {
            softmaxAxis = vpux::getPositiveAxisInd(softmax.getAxisIndAttr(), inputType.getRank());
            if (softmaxAxis.value() == concatAxis.value().ind()) {
                return {};
            }
        } else {
            if (softmaxAxis != vpux::getPositiveAxisInd(softmax.getAxisIndAttr(), inputType.getRank())) {
                return {};
            }
        }

        softmaxOps.push_back(softmax);
    }

    if (softmaxOps.size() < 2) {
        return {};
    }

    return softmaxOps;
}

void MoveSoftmaxAfterConcat::createNewSubgraphAndReplace(ArrayRef<IE::SoftMaxOp> siblingLayerOps, IE::ConcatOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> newConcatInputs;
    for (auto softmax : siblingLayerOps) {
        newConcatInputs.push_back(softmax.getInput());
    }

    auto concatAxis = IE::getConcatAxis(origOp);
    auto newConcat = rewriter.create<IE::ConcatOp>(origOp->getLoc(), newConcatInputs, concatAxis.value());

    auto firstSoftmaxOp = siblingLayerOps.front();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(firstSoftmaxOp.getInput().getType());
    auto softmaxAxis = vpux::getPositiveAxisInd(firstSoftmaxOp.getAxisIndAttr(), inputType.getRank());
    auto newSoftmax = rewriter.create<IE::SoftMaxOp>(takeOpLoc(firstSoftmaxOp, "new_softmax"), newConcat.getOutput(),
                                                     getIntAttr(rewriter.getContext(), softmaxAxis),
                                                     firstSoftmaxOp.getPadSizeAttr());

    rewriter.replaceOp(origOp, newSoftmax.getOutput());
}

//
// MoveAddAfterConcat
//
class MoveAddAfterConcat final : public MoveLayerAfterConcat<IE::AddOp> {
public:
    MoveAddAfterConcat(mlir::MLIRContext* ctx, Logger log): MoveLayerAfterConcat<IE::AddOp>(ctx, log) {
    }

    bool validateConcat(IE::ConcatOp) const override;
    SmallVector<IE::AddOp> getValidInputs(IE::ConcatOp concatOp) const override;
    void createNewSubgraphAndReplace(ArrayRef<IE::AddOp> siblingLayerOps, IE::ConcatOp origOp,
                                     mlir::PatternRewriter& rewriter) const override;
};

bool MoveAddAfterConcat::validateConcat(IE::ConcatOp concatOp) const {
    auto concatAxis = IE::getConcatAxis(concatOp);
    return concatAxis.has_value();
}

SmallVector<IE::AddOp> MoveAddAfterConcat::getValidInputs(IE::ConcatOp concatOp) const {
    SmallVector<IE::AddOp> addOps;
    for (auto input : concatOp.getInputs()) {
        auto inputOp = input.getDefiningOp();
        auto add = mlir::dyn_cast_or_null<IE::AddOp>(inputOp);
        if (add == nullptr) {
            return {};
        }

        addOps.push_back(add);
    }

    // All Add ops should share single input
    auto firstAddOp = addOps.front();
    auto input1 = firstAddOp.getInput1();
    auto input2 = firstAddOp.getInput2();
    auto outputType = firstAddOp.getOutput().getType();

    SmallVector<int64_t> sharedInputIndVec;
    auto shareSingleInput = [&](IE::AddOp addOp) {
        auto currentInput1 = addOp.getInput1();
        auto currentInput2 = addOp.getInput2();
        auto shareInput1 = (currentInput1 == input1) && (currentInput2 != input2);
        auto shareInput2 = (currentInput1 != input1) && (currentInput2 == input2);

        if (shareInput1) {
            sharedInputIndVec.push_back(1);
            return currentInput2.getType() == input2.getType() && outputType == input2.getType();
        }

        if (shareInput2) {
            sharedInputIndVec.push_back(2);
            return currentInput1.getType() == input1.getType() && outputType == input1.getType();
        }

        return false;
    };

    if (addOps.size() < 2) {
        return {};
    }

    SmallVector<IE::AddOp> addOpsWithoutFirst = SmallVector<IE::AddOp>(addOps.begin() + 1, addOps.end());
    if (!llvm::all_of(addOpsWithoutFirst, shareSingleInput)) {
        return {};
    }

    auto haveTheSameSharedInputInd = [&](int64_t ind) {
        return ind == sharedInputIndVec.front();
    };
    if (!llvm::all_of(sharedInputIndVec, haveTheSameSharedInputInd)) {
        return {};
    }

    return addOps;
}

void MoveAddAfterConcat::createNewSubgraphAndReplace(ArrayRef<IE::AddOp> siblingLayerOps, IE::ConcatOp origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    auto firstAddOp = siblingLayerOps.front();
    auto lastAddOp = siblingLayerOps.back();
    auto shareInput1 = firstAddOp.getInput1() == lastAddOp.getInput1();
    SmallVector<mlir::Value> newConcatInputs;
    for (auto addOp : siblingLayerOps) {
        if (shareInput1) {
            newConcatInputs.push_back(addOp.getInput2());
        } else {
            newConcatInputs.push_back(addOp.getInput1());
        }
    }

    auto concatAxis = IE::getConcatAxis(origOp);
    auto newConcat = rewriter.create<IE::ConcatOp>(origOp->getLoc(), newConcatInputs, concatAxis.value());
    IE::AddOp newAdd;
    if (shareInput1) {
        newAdd = rewriter.create<IE::AddOp>(takeOpLoc(origOp, "as_add"), firstAddOp.getInput1(), newConcat.getOutput(),
                                            firstAddOp.getAutoBroadcast(), nullptr, nullptr, nullptr, nullptr);
    } else {
        newAdd = rewriter.create<IE::AddOp>(takeOpLoc(origOp, "as_add"), newConcat.getOutput(), firstAddOp.getInput2(),
                                            firstAddOp.getAutoBroadcast(), nullptr, nullptr, nullptr, nullptr);
    }

    rewriter.replaceOp(origOp, newAdd);
}

//
// OptimizeParallelLayers
//

class OptimizeParallelLayers final : public IE::impl::OptimizeParallelLayersBase<OptimizeParallelLayers> {
public:
    explicit OptimizeParallelLayers(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void OptimizeParallelLayers::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patternsWithSlice(&ctx);
    patternsWithSlice.add<MoveEltwiseBeforeSlice<IE::MultiplyOp>>(&ctx, _log);
    patternsWithSlice.add<MoveEltwiseBeforeSlice<IE::AddOp>>(&ctx, _log);
    patternsWithSlice.add<MoveReshapeBeforeSlice>(&ctx, _log);
    patternsWithSlice.add<MoveFCBeforeSlice>(&ctx, _log);
    patternsWithSlice.add<MoveTanhBeforeSlice>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patternsWithSlice),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    mlir::RewritePatternSet patternsWithConcat(&ctx);
    patternsWithConcat.add<MoveReshapeAfterConcat>(&ctx, _log);
    patternsWithConcat.add<MoveFCAfterConcat>(&ctx, _log);
    patternsWithConcat.add<MoveSoftmaxAfterConcat>(&ctx, _log);
    patternsWithConcat.add<MoveAddAfterConcat>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patternsWithConcat),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeParallelLayersPass(Logger log) {
    return std::make_unique<OptimizeParallelLayers>(log);
}
