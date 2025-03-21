//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters/propagate_transpose_affine_reshape_common.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <optional>

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEAFFINERESHAPE
#define GEN_PASS_DEF_PROPAGATEAFFINERESHAPE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr Byte DMA_DATA_PATH_LEN_BYTE = Byte(32);

int64_t accumulateSizeBeforeDim(MemShapeRef memShape, MemDim dim) {
    return std::accumulate(memShape.begin(), memShape.begin() + dim.ind(), 1, std::multiplies<int64_t>());
}

//
// MoveThroughLayer
//

template <typename ConcreteOp>
class MoveThroughLayer : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MoveThroughLayer(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

protected:
    virtual mlir::DenseSet<int64_t> getModifiedAxis(ConcreteOp origOp) const = 0;
    virtual SmallVector<mlir::Attribute> getNewAttrs(ConcreteOp origOp, IE::AffineReshapeOp affineReshape) const = 0;
    virtual void updateAttrs(mlir::Operation* origOp, ArrayRef<mlir::Attribute> newAttrs) const = 0;

protected:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult MoveThroughLayer<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    auto maybeAffineReshape = origOp.getInput().template getDefiningOp<IE::AffineReshapeOp>();
    if (maybeAffineReshape == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got layer: '{0}'", origOp);
    _log.trace("Parent AffineReshape: '{0}'", maybeAffineReshape);

    const auto affineInShape = getShape(maybeAffineReshape.getInput());
    const auto affineOutShape = getShape(maybeAffineReshape.getOutput());

    const auto modifiedAxes = getModifiedAxis(origOp);
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(maybeAffineReshape.getDimMapping());

    if (IE::areModifiedAxesSplitOrMerged(dimMapping, affineInShape, affineOutShape, modifiedAxes, false, _log)) {
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    const SmallVector<mlir::Value> inputsToMap = {maybeAffineReshape.getInput()};
    mapper.map(origOp->getOperands(), ArrayRef(inputsToMap));
    auto* newLayerOp = rewriter.clone(*origOp.getOperation(), mapper);

    auto newAttrs = getNewAttrs(origOp, maybeAffineReshape);
    _log.trace("New attributes: '{0}'", newAttrs);

    updateAttrs(newLayerOp, newAttrs);

    vpux::inferReturnTypes(newLayerOp, vpux::InferShapedTypeMode::ALL);
    _log.trace("Create new layer: '{0}'", newLayerOp->getLoc());

    const auto outputShape = origOp.getType().getShape();
    const auto outShapeAttr = getIntArrayAttr(newLayerOp->getContext(), outputShape);

    auto newAffineReshape = rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(
            origOp, newLayerOp->getResult(0), maybeAffineReshape.getDimMappingAttr(), outShapeAttr);
    _log.trace("Replace current layer op with new AffineReshape: '{0}'", newAffineReshape);

    return mlir::success();
}

//
// MoveThroughTranspose
//

class MoveThroughTranspose final : public MoveThroughLayer<IE::TransposeOp> {
public:
    MoveThroughTranspose(mlir::MLIRContext* ctx, Logger log): MoveThroughLayer<IE::TransposeOp>(ctx, log) {
    }

private:
    mlir::DenseSet<int64_t> getModifiedAxis(IE::TransposeOp origOp) const override;
    SmallVector<mlir::Attribute> getNewAttrs(IE::TransposeOp origOp, IE::AffineReshapeOp affineReshape) const override;
    void updateAttrs(mlir::Operation* origOp, ArrayRef<mlir::Attribute> newAttrs) const override;
};

mlir::DenseSet<int64_t> MoveThroughTranspose::getModifiedAxis(IE::TransposeOp origOp) const {
    const auto originPerm = DimsOrder::fromAffineMap(origOp.getOrderValue().value());
    const auto order = to_small_vector(irange(originPerm.numDims()) | transformed([&](uint64_t idx) {
                                           return checked_cast<uint64_t>(originPerm.dimAt(idx).ind());
                                       }));

    mlir::DenseSet<int64_t> modifiedAxes;
    for (size_t i = 0; i < order.size(); i++) {
        if (order[i] != i) {
            modifiedAxes.insert(i);
        }
    }

    return modifiedAxes;
}

SmallVector<mlir::Attribute> MoveThroughTranspose::getNewAttrs(IE::TransposeOp origOp,
                                                               IE::AffineReshapeOp affineReshape) const {
    const auto affineInShape = getShape(affineReshape.getInput());
    const auto affineOutShape = getShape(affineReshape.getOutput());
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshape.getDimMapping());
    const auto invertedDimMapping =
            IE::invertDimMappingWithAxesNotSplitOrMerged(dimMapping, affineInShape, affineOutShape);

    SmallVector<unsigned> newPerm(affineInShape.size(), 0);
    const auto originPerm = DimsOrder::fromAffineMap(origOp.getOrderValue().value());
    const auto order = to_small_vector(irange(originPerm.numDims()) | transformed([&](uint64_t idx) {
                                           return checked_cast<uint64_t>(originPerm.dimAt(idx).ind());
                                       }));

    for (size_t i = 0; i < newPerm.size(); i++) {
        newPerm[i] = i;
    }

    for (size_t outDim = 0; outDim < order.size(); outDim++) {
        if (order[outDim] != outDim) {
            auto inDimIdx = invertedDimMapping[outDim];
            if (newPerm[inDimIdx] == inDimIdx) {
                newPerm[inDimIdx] = invertedDimMapping[order[outDim]];
            }
        }
    }

    const auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(newPerm, origOp->getContext()));
    return SmallVector<mlir::Attribute>{orderAttr};
}

void MoveThroughTranspose::updateAttrs(mlir::Operation* origOp, ArrayRef<mlir::Attribute> newAttrs) const {
    origOp->setAttr("order_value", newAttrs[0]);
}

//
// MoveThroughPermuteCast
//

// The pattern like below usually converted from decomposed matmul, and due to the AffineReshape -> PermuteCast ->
// AffineReshape chain, the concat will introduce extra DDR->DDR copies (if cannot concat on CMX), which is low
// efficient. Convert pattern from:
//      Ops                      Ops
//       |                        |
// AffineReshape            AffineReshape
//       |                        |
//  PermuteCast              PermuteCast
//       |                        |
// AffineReshape    ...     AffineReshape
//       \           |           /
//                Concat
// To:
//     Ops         ...         Ops
//      \           |           /
//                Concat
//                  |
//            AffineReshape
//                  |
//             PermuteCast
//                  |
//            AffineReshape

class MoveAffineReshapePermuteCastThroughConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    MoveAffineReshapePermuteCastThroughConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        this->setDebugName("MoveAffineReshapePermuteCastThroughConcat");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveAffineReshapePermuteCastThroughConcat::matchAndRewrite(IE::ConcatOp origOp,
                                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto ctx = rewriter.getContext();

    const auto axis = getConcatAxesFromOffsets(origOp, getShape(origOp.getOutput()));
    if (axis.size() != 1) {
        _log.nest().trace("[{0}]: Not only concat on 1 axis", getDebugName());
        return mlir::failure();
    }
    const auto concatAxis = (*axis.begin()).ind();

    auto concatInputs = origOp.getInputs();
    SmallVector<mlir::Value> newConcatInputs;
    IE::AffineReshapeOp firstAffineReshapeOp = nullptr;
    IE::PermuteCastOp permuteCastOp = nullptr;
    IE::AffineReshapeOp secondAffineReshapeOp = nullptr;
    for (const auto& input : concatInputs) {
        secondAffineReshapeOp = input.getDefiningOp<IE::AffineReshapeOp>();

        if (!secondAffineReshapeOp) {
            _log.nest().trace("[{0}]: There is no child AffineReshape op found for concat", getDebugName());
            return mlir::failure();
        }

        // Check the second affinereshape
        auto affineInShape = getShape(secondAffineReshapeOp.getInput());
        auto affineOutShape = getShape(secondAffineReshapeOp.getOutput());
        if (affineInShape.size() != 4 || affineOutShape.size() != 4) {
            _log.nest().trace("[{0}]: Only 4D rand AffineReshape supported", getDebugName());
            return mlir::failure();
        }
        if (affineInShape[Dims4D::Act::H] != 1 || affineInShape[Dims4D::Act::W] != 1 ||
            affineOutShape[Dims4D::Act::N] != 1 || affineOutShape[Dims4D::Act::C] != 1) {
            _log.nest().trace("[{0}]: Input AffineReshape is not eligible", getDebugName());
            return mlir::failure();
        }

        permuteCastOp = secondAffineReshapeOp.getInput().getDefiningOp<IE::PermuteCastOp>();
        if (!permuteCastOp) {
            _log.nest().trace("[{0}]: There is no PermuteCast op found for concat", getDebugName());
            return mlir::failure();
        }

        const auto memPerm = DimsOrder::fromAffineMap(permuteCastOp.getMemPerm());
        // Check Concat dim is N after permutecast
        if (memPerm.dimAt(0).ind() != concatAxis) {
            _log.nest().trace("[{0}]: Concat axis is not N after permute", getDebugName());
            return mlir::failure();
        }

        firstAffineReshapeOp = permuteCastOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
        if (!firstAffineReshapeOp) {
            _log.nest().trace("[{0}]: There is no parent AffineReshape op found for concat", getDebugName());
            return mlir::failure();
        }

        auto inputShape = getShape(firstAffineReshapeOp.getOutput());
        // If non one dim size is bigger than 2 dims, the mempermute is not trivial after propagated.
        SmallVector<Dim> nonOneDims = getNonOneDim(inputShape);
        if (nonOneDims.size() > 2) {
            return mlir::failure();
        }

        newConcatInputs.push_back(firstAffineReshapeOp.getInput());
    }

    auto concatOutputShape = getShape(origOp.getOutput());
    auto reshape1OutputShape = getShape(firstAffineReshapeOp.getOutput());
    auto reshape1InputShape = getShape(firstAffineReshapeOp.getInput());

    const auto cmxMemSize = VPU::getTotalCMXSize(origOp.getOperation());
    auto inNDInterface = mlir::cast<vpux::NDTypeInterface>(secondAffineReshapeOp.getInput().getType());
    const auto elementSize = inNDInterface.getCompactAllocSize().count() / reshape1InputShape.totalSize();

    auto affreshape1Producer = firstAffineReshapeOp.getInput().getDefiningOp();
    auto totalProducerInputSize = [&]() {
        int64_t totalSize = 0;
        if (affreshape1Producer == nullptr) {
            return totalSize;  // tensor should be blockarg, so only need concat size
        }

        for (auto input : affreshape1Producer->getOperands()) {
            totalSize += mlir::cast<vpux::NDTypeInterface>(input.getType()).getTotalAllocSize().count();
        }
        return totalSize;
    };

    auto concatTotalSize = totalProducerInputSize() + concatOutputShape.totalSize();
    // If concat op cannot fix into CMX, it's usually need DDR concat which is low efficient, otherwise no need to
    // propagate and it also introduce accuracy issue for this case. Experimental constraint to ensure this optimization
    // only process the DDR->DDR copies are the bottleneck.
    constexpr float threshold = 0.7f;
    if (concatTotalSize * elementSize * threshold < cmxMemSize.count()) {
        return mlir::failure();
    }

    auto newConcat = rewriter.create<IE::ConcatOp>(origOp.getLoc(), newConcatInputs, Dims4D::Act::N);

    SmallVector<int64_t> newShape1 = {concatOutputShape[Dims4D::Act::C], reshape1OutputShape[Dims4D::Act::C],
                                      reshape1OutputShape[Dims4D::Act::H], 1};
    auto newAffineReshapeOp = rewriter.create<IE::AffineReshapeOp>(
            appendLoc(origOp.getLoc(), "_reshape_1"), newConcat.getOutput(), firstAffineReshapeOp.getDimMappingAttr(),
            getIntArrayAttr(ctx, newShape1));

    auto dstOrder = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    const auto memPerm = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    auto newPermuteCastOp = rewriter.create<IE::PermuteCastOp>(appendLoc(origOp.getLoc(), "_permute_output"),
                                                               newAffineReshapeOp.getOutput(), dstOrder, memPerm);

    SmallVector<SmallVector<int64_t>> outDimMapping{{Dims4D::Act::N.ind(), Dims4D::Act::C.ind()},
                                                    {Dims4D::Act::H.ind()},
                                                    {Dims4D::Act::W.ind()},
                                                    {Dims4D::Act::W.ind()}};
    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(origOp, newPermuteCastOp.getOutput(),
                                                     getIntArrayOfArray(ctx, outDimMapping),
                                                     getIntArrayAttr(ctx, to_small_vector(concatOutputShape)));

    return mlir::success();
}

//
// MoveThroughExpand
//

class MoveThroughExpand final : public MoveThroughLayer<IE::ExpandOp> {
public:
    MoveThroughExpand(mlir::MLIRContext* ctx, Logger log): MoveThroughLayer<IE::ExpandOp>(ctx, log) {
    }

private:
    SmallVector<mlir::Attribute> getNewAttrs(IE::ExpandOp origOp, IE::AffineReshapeOp affineReshape) const override;
    mlir::DenseSet<int64_t> getModifiedAxis(IE::ExpandOp origOp) const override;
    void updateAttrs(mlir::Operation* origOp, ArrayRef<mlir::Attribute> newAttrs) const override;
};

mlir::DenseSet<int64_t> MoveThroughExpand::getModifiedAxis(IE::ExpandOp origOp) const {
    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());

    mlir::DenseSet<int64_t> modifiedAxes;
    for (size_t i = 0; i < padsBegin.size(); i++) {
        if (padsBegin[i] != 0 || padsEnd[i] != 0) {
            modifiedAxes.insert(i);
        }
    }

    return modifiedAxes;
}

SmallVector<mlir::Attribute> MoveThroughExpand::getNewAttrs(IE::ExpandOp origOp,
                                                            IE::AffineReshapeOp affineReshape) const {
    const auto affineInShape = getShape(affineReshape.getInput());
    const auto affineOutShape = getShape(affineReshape.getOutput());

    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshape.getDimMapping());
    SmallVector<int64_t> invertedDimMapping(affineOutShape.size(), 0);

    for (size_t inDim = 0; inDim < dimMapping.size(); inDim++) {
        auto dimsArr = dimMapping[inDim];
        for (size_t i = 0; i < dimsArr.size(); i++) {
            auto outDim = dimsArr[i];
            if (affineInShape[Dim(inDim)] == affineOutShape[Dim(outDim)]) {
                invertedDimMapping[dimsArr[i]] = inDim;
                break;
            }
        }
    }

    SmallVector<int64_t> newPadsBegin(affineInShape.size(), 0);
    SmallVector<int64_t> newPadsEnd(affineInShape.size(), 0);

    auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());

    for (size_t outDim = 0; outDim < padsBegin.size(); outDim++) {
        auto inDimIdx = invertedDimMapping[outDim];
        if (padsBegin[outDim] != 0) {
            newPadsBegin[inDimIdx] = padsBegin[outDim];
        }
        if (padsEnd[outDim] != 0) {
            newPadsEnd[inDimIdx] = padsEnd[outDim];
        }
    }

    mlir::Builder builder(origOp->getContext());
    auto newBeginPadsAttr = builder.getI64ArrayAttr(newPadsBegin);
    auto newEndPadsAttr = builder.getI64ArrayAttr(newPadsEnd);

    return SmallVector<mlir::Attribute>{newBeginPadsAttr, newEndPadsAttr};
}

void MoveThroughExpand::updateAttrs(mlir::Operation* origOp, ArrayRef<mlir::Attribute> newAttrs) const {
    origOp->setAttr("pads_begin", newAttrs[0]);
    origOp->setAttr("pads_end", newAttrs[1]);
}

//
// MoveThroughConcat
//

class MoveThroughConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    MoveThroughConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::ArrayAttr getConcatOffsetsParameters(mlir::ArrayAttr oldOffsets, mlir::ArrayAttr dimsMappingAttr,
                                           SmallVector<mlir::Value> oldInputs, SmallVector<mlir::Value> newInputs) {
    const auto oldOffsetsList = parseIntArrayOfArrayAttr<int64_t>(oldOffsets);
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(dimsMappingAttr);

    size_t currentIndex = 0;
    SmallVector<SmallVector<int64_t>> newOffsetsList;
    newOffsetsList.reserve(oldOffsetsList.size());

    for (const auto& [oldInput, newInput] : zip(oldInputs, newInputs)) {
        const auto inReshapeShape = getShape(newInput).raw();
        const auto outputReshapeShape = getShape(oldInput).raw();

        SmallVector<int64_t> newOffset(inReshapeShape.size(), 0);
        const auto oldOffset = oldOffsetsList[currentIndex];
        int64_t prevDim = -1;
        int64_t prevOffset = -1;

        for (const auto index : irange(newOffset.size())) {
            const auto inputReshapeSize = inReshapeShape[index];

            const auto& dims = dimMapping[index];
            for (const auto& dim : dims) {
                if (inputReshapeSize != outputReshapeShape[dim]) {
                    continue;
                } else {
                    auto dimIt = llvm::find_if(dims, [&](int64_t elem) {
                        return (outputReshapeShape[elem] != 1 && outputReshapeShape[elem] != inputReshapeSize);
                    });
                    if (dimIt != dims.end()) {
                        return nullptr;
                    }

                    newOffset[index] = oldOffset[dim];

                    // To handle the case of expanding to multiple 1, and concat on this dimension
                    // eg: 2 x ([1] -> [1, 1, 1]) -- Concat --> [1, 2, 1] {offset = [0, 0, 0], [0, 1, 0], [0, 2, 0]}
                    auto dimOneIt = llvm::find_if(dims, [&](int64_t elem) {
                        return (outputReshapeShape[elem] == 1 && oldOffset[elem] != 0);
                    });
                    if (dimOneIt != dims.end()) {
                        newOffset[index] = oldOffset[*dimOneIt];
                    }

                    if (index > 0 && newOffset[index] == prevOffset && dim == prevDim) {
                        newOffset[index] = 0;
                    } else {
                        prevOffset = newOffset[index];
                    }

                    prevDim = dim;
                    break;
                }
            }
        }

        newOffsetsList.push_back(newOffset);
        ++currentIndex;
    }

    return getIntArrayOfArray(dimsMappingAttr.getContext(), ArrayRef(newOffsetsList));
}

mlir::LogicalResult MoveThroughConcat::matchAndRewrite(IE::ConcatOp origConcatOp,
                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}]: Rewriting {1}", getDebugName(), origConcatOp->getLoc());

    if (origConcatOp.getStaticOffsetsAttr() == nullptr) {
        return matchFailed(rewriter, origConcatOp, "Incorrect Concat parameters");
    }

    auto inputs = origConcatOp.getInputs();

    if (inputs.size() < 2) {
        _log.trace("[{0}]: Invalid inputs", getDebugName());
        return mlir::failure();
    }

    SmallVector<mlir::Value> newInputs;
    newInputs.reserve(inputs.size());
    mlir::ArrayAttr dimsMapping;
    const auto modifiedAxises = IE::getConcatModifiedAxis(origConcatOp);

    if (modifiedAxises.empty()) {
        return mlir::failure();
    }

    ShapeRef shapeBeforeAffineReshape;
    auto getDifferentNums = [](ShapeRef shape1, ShapeRef shape2) -> int64_t {
        int64_t differentNums = 0;
        for (size_t i = 0; i < shape1.size(); i++) {
            if (shape1[Dim(i)] != shape2[Dim(i)]) {
                differentNums++;
            }
        }
        return differentNums;
    };

    for (auto input : inputs) {
        auto parentOp = input.getDefiningOp<IE::AffineReshapeOp>();

        if (parentOp == nullptr) {
            _log.trace("[{0}]: Input {1} is not AffineReshape result", getDebugName(), input.getLoc());
            return mlir::failure();
        }

        if (!newInputs.empty()) {
            auto prevInput = newInputs.back();

            if (getShape(prevInput).size() != getShape(parentOp.getInput()).size()) {
                _log.trace("[{0}]: Input {1} has different shape than others", getDebugName(), parentOp.getLoc());
                return mlir::failure();
            }
        }

        if (dimsMapping != nullptr) {
            if (parentOp.getDimMapping() != dimsMapping) {
                _log.trace("[{0}]: Input {1} has different mapping from others", getDebugName(), parentOp.getLoc());
                return mlir::failure();
            }
        } else {
            dimsMapping = parentOp.getDimMapping();
        }

        if (shapeBeforeAffineReshape.empty()) {
            shapeBeforeAffineReshape = getShape(parentOp.getInput());
        } else {
            auto curShapeBeforeAffineReshape = getShape(parentOp.getInput());
            auto differentNums = getDifferentNums(curShapeBeforeAffineReshape, shapeBeforeAffineReshape);
            if (differentNums > modifiedAxises.size()) {
                _log.trace("[{0}]: Input {1} has different shape of non concat axis from others", getDebugName(),
                           parentOp.getLoc());
                return mlir::failure();
            }
        }

        const auto affineInputShape = getShape(parentOp.getInput());
        const auto affineOutputShape = getShape(parentOp.getOutput());

        const auto dimMappingList = parseIntArrayOfArrayAttr<int64_t>(dimsMapping);
        if (IE::areModifiedAxesSplitOrMerged(dimMappingList, affineInputShape, affineOutputShape, modifiedAxises, false,
                                             _log.nest())) {
            return mlir::failure();
        }

        newInputs.push_back(parentOp.getInput());
    }

    VPUX_THROW_WHEN(dimsMapping == nullptr, "Cannot get mapping from Reshapes");

    auto newOffsetsAttr =
            getConcatOffsetsParameters(origConcatOp.getStaticOffsetsAttr(), dimsMapping, inputs, newInputs);

    if (newOffsetsAttr == nullptr) {
        _log.trace("[{0}]: Concat parameters couldn't be calculated", getDebugName(), origConcatOp.getLoc());
        return mlir::failure();
    }

    auto newConcat = rewriter.create<IE::ConcatOp>(origConcatOp.getLoc(), newInputs, nullptr, newOffsetsAttr);

    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(
            origConcatOp, newConcat, dimsMapping,
            getIntArrayAttr(origConcatOp.getContext(), getShape(origConcatOp).raw()));

    return mlir::success();
}

//
// MoveThroughSoftmax
//

class MoveThroughSoftmax final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    MoveThroughSoftmax(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SoftMaxOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughSoftmax");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveThroughSoftmax::matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto affineReshapeOp = origOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    auto newSoftmaxAxis = getNewSoftmaxAxisAfterSwappingWithAffineReshape(origOp, affineReshapeOp, _log);
    if (!newSoftmaxAxis.has_value()) {
        return mlir::failure();
    }

    auto newSoftmaxAxisValue = newSoftmaxAxis.value();
    auto newSoftmaxOp = rewriter.create<IE::SoftMaxOp>(
            origOp.getLoc(), affineReshapeOp.getInput().getType(), affineReshapeOp.getInput(),
            getIntAttr(getContext(), newSoftmaxAxisValue), origOp.getPadSizeAttr());
    auto newAffineReshapeOp =
            rewriter.create<IE::AffineReshapeOp>(affineReshapeOp.getLoc(), newSoftmaxOp.getOutput(),
                                                 affineReshapeOp.getDimMapping(), affineReshapeOp.getShapeValue());
    origOp.replaceAllUsesWith(newAffineReshapeOp.getOutput());

    return mlir::success();
}

//
// MoveThroughMVN
//

class MoveThroughMVN final : public mlir::OpRewritePattern<IE::MVNOp> {
public:
    MoveThroughMVN(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MVNOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughMVN");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveThroughMVN::matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto preAffineReshapeOp = origOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    if (preAffineReshapeOp == nullptr || !preAffineReshapeOp->hasOneUse()) {
        _log.trace("Previous AffineReshapeOp not found or has multiple uses");
        return mlir::failure();
    }

    auto isAcrossChannels = origOp.getAcrossChannels();
    const auto mvnInShape = getShape(origOp.getInput());
    const auto preAffineReshapeInShape = getShape(preAffineReshapeOp.getInput());
    const int64_t rank4D = 4;
    if (preAffineReshapeInShape.size() != rank4D) {
        _log.trace("Previous AffineReshapeOp input shape size is not 4");
        return mlir::failure();
    }

    auto inChSize = mvnInShape[Dims4D::Act::H] * mvnInShape[Dims4D::Act::W] *
                    (isAcrossChannels ? mvnInShape[Dims4D::Act::C] : 1);
    auto outChSize = preAffineReshapeInShape[Dims4D::Act::H] * preAffineReshapeInShape[Dims4D::Act::W] *
                     (isAcrossChannels ? preAffineReshapeInShape[Dims4D::Act::C] : 1);
    if (inChSize != outChSize) {
        if (!isAcrossChannels && mvnInShape[Dims4D::Act::C] == 1 &&
            mvnInShape[Dims4D::Act::N] == preAffineReshapeInShape[Dims4D::Act::N]) {
            // For example, it is equivalent to propagate
            // across_channels = false, 1x1x1280x1 -> across_channels = true, 1x1280x1x1
            isAcrossChannels = true;
        } else {
            _log.trace("AffineReshapeOp not suitable to propagate");
            return mlir::failure();
        }
    }

    // Create new MVNOp
    auto newMvnOp = rewriter.create<IE::MVNOp>(origOp->getLoc(), preAffineReshapeOp.getInput(),
                                               mlir::BoolAttr::get(getContext(), isAcrossChannels),
                                               origOp.getNormalizeVarianceAttr(), origOp.getEpsAttr());

    // Create new AffineReshapeOp
    auto newAffineReshapeOp = rewriter.create<IE::AffineReshapeOp>(preAffineReshapeOp.getLoc(), newMvnOp.getOutput(),
                                                                   preAffineReshapeOp.getDimMapping(),
                                                                   preAffineReshapeOp.getShapeValue());

    origOp.replaceAllUsesWith(newAffineReshapeOp.getOutput());

    return mlir::success();
}

//
// MoveThroughEltwiseGeneric
//

using VerifyCb = FuncRef<bool(mlir::Operation*)>;

template <class ConcreteOp>
class MoveThroughEltwiseGeneric final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MoveThroughEltwiseGeneric(mlir::MLIRContext* ctx, Logger log, VerifyCb verifyFunc = nullptr)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log), _verifyFunc(verifyFunc) {
        this->setDebugName("MoveThroughEltwiseGeneric");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VerifyCb _verifyFunc;
};

template <class ConcreteOp>
mlir::LogicalResult MoveThroughEltwiseGeneric<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    VPUX_THROW_UNLESS(origOp->getNumResults() == 1 && origOp->getNumOperands() == 1,
                      "Not a single input & output operation");

    auto inputAffineReshape = origOp.getInput().template getDefiningOp<IE::AffineReshapeOp>();
    if (inputAffineReshape == nullptr || !inputAffineReshape->hasOneUse()) {
        return mlir::failure();
    }

    const auto reshapeInputRank = getShape(inputAffineReshape.getInput()).size();
    const auto geluInputRank = getShape(origOp.getInput()).size();
    if (geluInputRank != reshapeInputRank) {
        return mlir::failure();
    }

    if ((_verifyFunc) && !_verifyFunc(origOp.getOperation())) {
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperand(0), inputAffineReshape.getInput());
    auto newOp = rewriter.clone(*origOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);
    // Input layout should be kept
    auto dimsOrder = mlir::cast<NDTypeInterface>(newOp->getOperand(0).getType()).getDimsOrder();
    auto newOutType = mlir::cast<NDTypeInterface>(newOp->getResult(0).getType()).changeDimsOrder(dimsOrder);
    newOp->getResult(0).setType(newOutType);

    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(origOp, newOp->getResult(0),
                                                     inputAffineReshape.getDimMappingAttr(),
                                                     inputAffineReshape.getShapeValueAttr());

    return mlir::success();
}

//
// MoveThroughMultiply
//

class MoveThroughMultiply final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    MoveThroughMultiply(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughMultiply");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

    mlir::LogicalResult processMultiplyOpWithBroadCastConstInput(IE::MultiplyOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const;

    bool isConstInput(mlir::Value value) const;

private:
    Logger _log;
};

mlir::LogicalResult MoveThroughMultiply::matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto hasConstInput = llvm::any_of(origOp.getInputs(), [&](auto input) {
        return isConstInput(input);
    });
    if (hasConstInput) {
        return processMultiplyOpWithBroadCastConstInput(origOp, rewriter);
    }

    auto inputAffineReshape1 = origOp.getInput1().getDefiningOp<IE::AffineReshapeOp>();
    if (inputAffineReshape1 == nullptr || !inputAffineReshape1->hasOneUse() ||
        IE::doesAffineReshapeChangeRank(inputAffineReshape1)) {
        return mlir::failure();
    }

    auto inputAffineReshape2 = origOp.getInput2().getDefiningOp<IE::AffineReshapeOp>();
    if (inputAffineReshape2 == nullptr || !inputAffineReshape2->hasOneUse() ||
        IE::doesAffineReshapeChangeRank(inputAffineReshape2)) {
        return mlir::failure();
    }

    if (inputAffineReshape1.getDimMapping() != inputAffineReshape2.getDimMapping()) {
        _log.nest().trace("AffineReshape operations have different dim-mapping");
        return mlir::failure();
    }

    if (inputAffineReshape1.getInput().getType() != inputAffineReshape2.getInput().getType()) {
        _log.nest().trace("AffineReshape operations have different input types");
        return mlir::failure();
    }

    auto newMultiply = rewriter.create<IE::MultiplyOp>(
            origOp.getLoc(), inputAffineReshape1.getInput().getType(), inputAffineReshape1.getInput(),
            inputAffineReshape2.getInput(), origOp.getAutoBroadcastAttr(), origOp.getPostOpAttr(),
            origOp.getClampAttr(), origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());
    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(origOp, newMultiply.getOutput(),
                                                     inputAffineReshape1.getDimMappingAttr(),
                                                     inputAffineReshape1.getShapeValueAttr());

    _log.trace("Successfully move MultiplyOp through AffineReshape");

    return mlir::success();
}

mlir::LogicalResult MoveThroughMultiply::processMultiplyOpWithBroadCastConstInput(
        IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const {
    /* Convert pattern
                      Input
                       |
          Const      AffineReshape           New Const   Input
             \        /                          \        /
              Multiply                     ->     Multiply
                 |                                    |
               Output                             AffineReshape
                                                      |
                                                    Output

    */
    auto nonConstInputIter = llvm::find_if(origOp.getInputs(), [&](auto input) {
        auto parentOp = mlir::dyn_cast_or_null<IE::AffineReshapeOp>(input.getDefiningOp());
        return parentOp != nullptr && parentOp->hasOneUse() && !doesAffineReshapeChangeRank(parentOp);
    });
    if (nonConstInputIter == origOp.getInputs().end()) {
        return mlir::failure();
    }
    auto input = *nonConstInputIter;
    auto affineReshapeOp = input.getDefiningOp<IE::AffineReshapeOp>();
    auto affineReshapeInShape = getShape(affineReshapeOp.getInput());
    auto affineReshapeOutShape = getShape(affineReshapeOp.getOutput());
    auto origOpOutShape = getShape(origOp.getOutput());
    if (affineReshapeOutShape != origOpOutShape) {
        return mlir::failure();
    }

    auto constInputIter = llvm::find_if(origOp.getInputs(), [&](auto input) {
        return isConstInput(input);
    });

    VPUX_THROW_WHEN(constInputIter == origOp.getInputs().end(), "Const input not found");
    auto constInput = *constInputIter;
    auto constInShape = getShape(constInput);

    const auto isScalar = llvm::all_of(constInShape, [](auto dim) {
        return dim == 1;
    });
    const auto isVector = llvm::count(constInShape, 1) == static_cast<int64_t>(constInShape.size() - 1);
    if (!isVector && !isScalar) {
        return mlir::failure();
    }

    if (isVector) {
        // Get the nonbroadcast dim
        const auto dimOrder = DimsOrder::fromValue(constInput);
        const auto constInMemShape = dimOrder.toMemoryOrder(constInShape);
        const auto nonBroadCastDimIdx =
                std::distance(constInShape.begin(), llvm::find_if(constInShape, [](const auto& dim) {
                                  return dim != 1;
                              }));

        const auto nonBroadCastDimSize = constInShape[Dim(nonBroadCastDimIdx)];
        VPUX_THROW_UNLESS(affineReshapeOutShape[Dim(nonBroadCastDimIdx)] == nonBroadCastDimSize,
                          "Unsupported broadcast at '{0}'", origOp->getLoc());

        // Need to check the dim keeps unchanged after affine reshape.
        auto nonConstInMemShape = dimOrder.toMemoryOrder(affineReshapeOutShape);
        auto nonBroadCastMemDim = dimOrder.toMemDim(Dim(nonBroadCastDimIdx));
        const auto sizeBeforeNonBroadCastDim = accumulateSizeBeforeDim(nonConstInMemShape, nonBroadCastMemDim);
        const auto memShapeBeforeReshape = dimOrder.toMemoryOrder(affineReshapeInShape);

        auto dimRange = irange(memShapeBeforeReshape.size()) | reversed;
        auto iter = llvm::find_if(dimRange, [&](const auto& dim) {
            return accumulateSizeBeforeDim(memShapeBeforeReshape, MemDim(dim)) == sizeBeforeNonBroadCastDim;
        });

        if (iter == dimRange.end()) {
            return mlir::failure();
        }

        const auto nonBroadCastDimBeforeReshape = dimOrder.toDim(MemDim(*iter));
        if (affineReshapeInShape[nonBroadCastDimBeforeReshape] != nonBroadCastDimSize) {
            // the broadcast dim size is changed after affine reshape
            return mlir::failure();
        }
        auto newConstInputShape = Shape(affineReshapeOutShape.size(), 1);
        newConstInputShape[nonBroadCastDimBeforeReshape] = nonBroadCastDimSize;
        constInput = rewriter.createOrFold<IE::ReshapeOp>(constInput.getLoc(), constInput, nullptr, false,
                                                          getIntArrayAttr(origOp->getContext(), newConstInputShape));
    }
    auto newMultiply = rewriter.create<IE::MultiplyOp>(origOp.getLoc(), affineReshapeOp.getInput(), constInput,
                                                       origOp.getAutoBroadcastAttr(), origOp.getPostOpAttr(),
                                                       origOp.getClampAttr(), origOp.getOutputChannelsAttr(),
                                                       origOp.getInputChannelsAttr());
    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(
            origOp, newMultiply.getOutput(), affineReshapeOp.getDimMappingAttr(), affineReshapeOp.getShapeValueAttr());
    return mlir::success();
}

bool MoveThroughMultiply::isConstInput(mlir::Value value) const {
    return mlir::isa_and_nonnull<Const::DeclareOp>(value.getDefiningOp());
}

//
// MoveThroughAdd
//
// We may get AffineReshape between Convolution and Add after input shape adjustment for Convolution.
// Convert below pattern:
//
//      Input          Conv
//         |            |
//  [ViewLikeOps]   AffineReshape
//          \          /
//              Add
//               |
//
// to:
//
//      Input          Conv
//         |            |
//  [ViewLikeOps]       |
//         |            |
//      ShapeCast       |
//          \          /
//              Add
//               |
//          AffineReshape
//               |
//
class MoveThroughAdd final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    MoveThroughAdd(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughAdd");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveThroughAdd::matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (origOp.getInput1().getType() != origOp.getInput2().getType()) {
        _log.nest().trace("Add inputs have different input types");
        return mlir::failure();
    }

    if (IE::isPerAxisQuant(origOp.getOutput()) || IE::isPerAxisQuant(origOp.getInput1())) {
        return mlir::failure();
    }

    auto affineReshapeInput = origOp.getInput1();
    auto anotherInput = origOp.getInput2();
    auto inputAffineReshapeOp = affineReshapeInput.getDefiningOp<IE::AffineReshapeOp>();
    if (inputAffineReshapeOp == nullptr) {
        affineReshapeInput = origOp.getInput2();
        anotherInput = origOp.getInput1();
        inputAffineReshapeOp = affineReshapeInput.getDefiningOp<IE::AffineReshapeOp>();
    }

    if (inputAffineReshapeOp == nullptr || !inputAffineReshapeOp->hasOneUse()) {
        return mlir::failure();
    }

    // TODO: E#139356
    // Consider moving this rewriter into VPU dialect.
    // It will be easier to check if operation can be closed to VPU::VerticalFusionOpInterface.
    // This is a better approach to ensure beneficial propagation for generic optimization.
    auto affineReshapeParentOp = inputAffineReshapeOp.getInput().getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::AddOp>(affineReshapeParentOp)) {
        return mlir::failure();
    }

    auto affineReshapeInType = inputAffineReshapeOp.getInput().getType().cast<NDTypeInterface>();
    if (affineReshapeInType.getRank() != 4) {
        return mlir::failure();
    }
    const auto alignment = VPU::NCEInvariant::getAlignment(affineReshapeInType.getElementType());
    const auto affineReshapeInShape = affineReshapeInType.getShape();
    if (affineReshapeInShape[Dims4D::Act::C] % alignment != 0 || affineReshapeInShape[Dims4D::Act::N] > 1) {
        return mlir::failure();
    }

    auto ctx = rewriter.getContext();

    auto inputShape = getShape(inputAffineReshapeOp.getInput());
    auto newInputShapeCast =
            rewriter.create<IE::ShapeCastOp>(anotherInput.getLoc(), anotherInput, getIntArrayAttr(ctx, inputShape));

    auto newInput1 =
            affineReshapeInput == origOp.getInput1() ? inputAffineReshapeOp.getInput() : newInputShapeCast.getResult();
    auto newInput2 =
            affineReshapeInput == origOp.getInput1() ? newInputShapeCast.getResult() : inputAffineReshapeOp.getInput();
    auto origOutputType = origOp.getOutput().getType().cast<NDTypeInterface>();
    auto newOutputType = origOutputType.changeShape(inputShape);
    auto newAddOp = rewriter.create<IE::AddOp>(
            origOp.getLoc(), newOutputType, newInput1, newInput2, origOp.getAutoBroadcastAttr(), origOp.getPostOpAttr(),
            origOp.getClampAttr(), origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());

    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(origOp, origOutputType, newAddOp.getOutput(),
                                                     inputAffineReshapeOp.getDimMappingAttr(),
                                                     inputAffineReshapeOp.getShapeValueAttr());

    _log.trace("Successfully move AddOp through AffineReshape");

    return mlir::success();
}

//
// ConcatReshapeConcat
//

class ConcatReshapeConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ConcatReshapeConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        this->setDebugName("ConcatReshapeConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// Move AffineReshape before Concat
// to support the possible FuseConcat in the following canonicalization
//   Concat                          AffineReshape
//      |                                 |
// AffineReshape            ->         Concat
//      |                                 |
//   Concat                            Concat
mlir::LogicalResult ConcatReshapeConcat::matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got ConcatOp at '{0}'", origOp->getLoc());
    // Check the pattern
    if (!origOp->hasOneUse()) {
        return mlir::failure();
    }
    if (origOp.getStaticOffsetsAttr() == nullptr) {
        return matchFailed(rewriter, origOp, "Incorrect Concat parameters");
    }

    auto reshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(*origOp.getOutput().getUsers().begin());
    if (reshapeOp == nullptr || !reshapeOp->hasOneUse()) {
        return matchFailed(rewriter, origOp, "Pattern mismatch");
    }
    auto outConcatOp = mlir::dyn_cast<IE::ConcatOp>(*reshapeOp.getOutput().getUsers().begin());
    if (outConcatOp == nullptr) {
        return matchFailed(rewriter, origOp, "Pattern mismatch");
    }
    auto finalOutType = outConcatOp.getOutput().getType().dyn_cast<NDTypeInterface>();
    auto memShape = finalOutType.getMemShape();
    auto getNonOneDims = [](MemShapeRef shape) {
        Shape resultShape;
        llvm::copy_if(shape, std::back_inserter(resultShape), [](int64_t elem) {
            return elem != 1;
        });
        return resultShape;
    };
    auto innerDimLengthByte = finalOutType.getElemTypeSize().to<Byte>() * getNonOneDims(memShape).back();
    // E-91195: only when inner dim size is greater than 32 bytes, the optimization shows positive effect
    if (innerDimLengthByte < Byte(DMA_DATA_PATH_LEN_BYTE)) {
        _log.trace("memShape {0}, nonOneShape {1}", memShape, getNonOneDims(memShape));
        return matchFailed(rewriter, origOp, "Not benefit to Swap");
    }

    const auto affineInShape = getShape(reshapeOp.getInput());
    const auto affineOutShape = getShape(reshapeOp.getOutput());

    const auto modifiedAxes = IE::getConcatModifiedAxis(origOp);
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(reshapeOp.getDimMapping());

    if (IE::areModifiedAxesSplitOrMerged(dimMapping, affineInShape, affineOutShape, modifiedAxes, true, _log)) {
        return matchFailed(rewriter, origOp, "Modified Axes split or merged");
    }

    const auto inputs = origOp.getInputs();
    SmallVector<mlir::Value> newInputs;
    SmallVector<vpux::ShapeRef> newInputShapes;
    newInputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        SmallVector<int64_t> newShapeVec =
                IE::calculateInputShapeAfterSwitchConcatAndAffineReshape(input, origOp, reshapeOp);
        const auto outputShapeAttr = getIntArrayAttr(rewriter.getContext(), Shape(newShapeVec));
        auto newAffineReshapeOp = rewriter.create<IE::AffineReshapeOp>(reshapeOp.getLoc(), input,
                                                                       reshapeOp.getDimMapping(), outputShapeAttr);
        newInputs.push_back(newAffineReshapeOp.getOutput());
        newInputShapes.push_back(getShape(newAffineReshapeOp.getOutput()));
    }

    auto newOffsetsAttr = IE::getNewConcatOffsetsParameters(origOp.getStaticOffsetsAttr(), reshapeOp.getDimMapping(),
                                                            inputs, newInputShapes, affineOutShape, modifiedAxes);

    _log.trace("Swapped Concat-AffineReshape pattern");
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(reshapeOp, newInputs, nullptr, newOffsetsAttr);
    rewriter.eraseOp(origOp);
    return mlir::success();
}

//
// MoveThroughSlice
//

class MoveThroughSlice final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    MoveThroughSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::DenseSet<int64_t> getModifiedAxis(IE::AffineReshapeOp origOp) const;
    Logger _log;
};

mlir::DenseSet<int64_t> MoveThroughSlice::getModifiedAxis(IE::AffineReshapeOp origOp) const {
    mlir::DenseSet<int64_t> modifiedAxes;
    for (auto user : origOp.getResult().getUsers()) {
        if (auto userOp = mlir::dyn_cast<IE::SliceOp>(user)) {
            const auto inputShape = getShape(userOp.getSource()).raw();
            const auto staticSizes = parseIntArrayAttr<int64_t>(userOp.getStaticSizesAttr());
            for (size_t i = 0; i < staticSizes.size(); i++) {
                if (staticSizes[i] != inputShape[i] && !modifiedAxes.contains(i)) {
                    modifiedAxes.insert(i);
                }
            }
        }
    }
    return modifiedAxes;
}

mlir::LogicalResult MoveThroughSlice::matchAndRewrite(IE::SliceOp origSliceOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}]: Rewriting {1}", getDebugName(), origSliceOp->getLoc());
    if (origSliceOp.getStaticOffsetsAttr() == nullptr || origSliceOp.getStaticSizesAttr() == nullptr) {
        return matchFailed(rewriter, origSliceOp, "Incorrect Slice parameters");
    }

    auto affineReshapeOp = origSliceOp.getOperand().getDefiningOp<IE::AffineReshapeOp>();
    if (affineReshapeOp == nullptr) {
        return mlir::failure();
    }

    mlir::ArrayAttr dimsMapping = affineReshapeOp.getDimMapping();
    const auto affineInputShape = getShape(affineReshapeOp.getInput());
    const auto affineOutputShape = getShape(affineReshapeOp.getOutput());

    const auto modifiedAxises = getModifiedAxis(affineReshapeOp);
    if (modifiedAxises.empty() || modifiedAxises.size() > 1) {
        _log.trace("[{0}]: {1}'s user has more than one dim sliced or empty, size: {2}", getDebugName(),
                   origSliceOp.getLoc(), modifiedAxises.size());
        return mlir::failure();
    }

    const auto dimMappingList = parseIntArrayOfArrayAttr<int64_t>(dimsMapping);
    if (IE::areModifiedAxesSplitOrMerged(dimMappingList, affineInputShape, affineOutputShape, modifiedAxises, false,
                                         _log.nest())) {
        _log.trace("[{0}]: slice operation {1} areModifiedAxesSplitOrMerged in affineReshape op {2}", getDebugName(),
                   origSliceOp.getLoc(), affineReshapeOp.getLoc());
        return mlir::failure();
    }

    const auto invertedDimMapping =
            IE::invertDimMappingWithAxesNotSplitOrMerged(dimMappingList, affineInputShape, affineOutputShape);

    const auto newSliceAxis = invertedDimMapping[*modifiedAxises.begin()];
    SmallVector<int64_t> newStaticOffset(affineInputShape.size(), 0);
    SmallVector<int64_t> newStaticSize = to_small_vector(affineInputShape);

    const auto staticOffset = parseIntArrayAttr<int64_t>(origSliceOp.getStaticOffsetsAttr());
    newStaticOffset[newSliceAxis] = staticOffset[*modifiedAxises.begin()];
    const auto staticSize = parseIntArrayAttr<int64_t>(origSliceOp.getStaticSizesAttr());
    newStaticSize[newSliceAxis] = staticSize[*modifiedAxises.begin()];
    auto newStaticOffsetAttr = getIntArrayAttr(rewriter.getContext(), newStaticOffset);
    auto newStaticSizeAttr = getIntArrayAttr(rewriter.getContext(), newStaticSize);

    mlir::IRMapping mapper;
    const SmallVector<mlir::Value> inputsToMap = {affineReshapeOp.getInput()};
    mapper.map(origSliceOp->getOperands(), ArrayRef(inputsToMap));
    auto* newLayerOp = rewriter.clone(*origSliceOp.getOperation(), mapper);
    newLayerOp->setAttr("static_offsets", newStaticOffsetAttr);
    newLayerOp->setAttr("static_sizes", newStaticSizeAttr);
    vpux::inferReturnTypes(newLayerOp, vpux::InferShapedTypeMode::ALL);

    const auto outputShape = origSliceOp.getResult().getType().cast<NDTypeInterface>().getShape();
    const auto outShapeAttr = getIntArrayAttr(newLayerOp->getContext(), outputShape);

    auto newAffineReshape = rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(
            origSliceOp, newLayerOp->getResult(0), affineReshapeOp.getDimMappingAttr(), outShapeAttr);
    _log.trace("Replace current layer op with new AffineReshape: '{0}'", newAffineReshape);
    return mlir::success();
}

//
// MoveThroughOneInputEltwise
//

class MoveThroughOneInputEltwise final : public mlir::OpTraitRewritePattern<IE::EltwiseOp> {
public:
    MoveThroughOneInputEltwise(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpTraitRewritePattern<IE::EltwiseOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughOneInputEltwise");
    }

private:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* eltwiseOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveThroughOneInputEltwise::matchAndRewrite(mlir::Operation* eltwiseOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), eltwiseOp->getName(), eltwiseOp->getLoc());

    if (eltwiseOp->getNumOperands() != 1 || eltwiseOp->getNumResults() != 1) {
        return matchFailed(_log, rewriter, eltwiseOp, "EltwiseOp is not a single input & output operation");
    }

    auto affineReshapeOp = eltwiseOp->getOperand(0).getDefiningOp<IE::AffineReshapeOp>();
    if (affineReshapeOp == nullptr || !affineReshapeOp->hasOneUse()) {
        return matchFailed(_log, rewriter, eltwiseOp, "AffineReshapeOp not found or has multiple uses");
    }

    if (IE::isPerAxisQuant(eltwiseOp->getOperand(0)) || IE::isPerAxisQuant(eltwiseOp->getResult(0))) {
        return mlir::failure();
    }

    // ConvertShapeTo4D could generate this subgraph
    //   AffineReshape (2D->4D) -> EltwiseOp (4D) -> AffineReshape (4D->2D)
    // Do not change it into
    //   EltwiseOp (2D) -> AffineReshape (2D->4D) -> AffineReshape (4D->2D)
    if (IE::doesAffineReshapeChangeRank(affineReshapeOp)) {
        return matchFailed(_log, rewriter, eltwiseOp, "AffineReshapeOp changes rank");
    }

    // No benefit to propagate AffineReshapeOp through, e.g., ConvertOp {f16 -> f32}
    const auto srcType = eltwiseOp->getOperand(0).getType();
    const auto dstType = eltwiseOp->getResult(0).getType();
    if (getElemTypeSize(srcType) < getElemTypeSize(dstType)) {
        return matchFailed(_log, rewriter, eltwiseOp,
                           "Input element type size is smaller than output element type size");
    }

    // Do not change
    //   TransposeOp -> AffineReshapeOp -> EltwiseOp
    // into
    //   TransposeOp -> EltwiseOp -> AffineReshapeOp
    // because TransposeOp -> AffineReshapeOp results in better performance
    auto transposeOp = affineReshapeOp->getOperand(0).getDefiningOp<IE::TransposeOp>();
    if (transposeOp != nullptr && !transposeOp->hasOneUse()) {
        return matchFailed(_log, rewriter, eltwiseOp, "Input TransposeOp has more than one uses");
    }

    _log.trace("[{0}] Propagate '{1}' at '{2}' through  '{3}' at '{4}'", this->getDebugName(),
               affineReshapeOp->getName(), affineReshapeOp->getLoc(), eltwiseOp->getName(), eltwiseOp->getLoc());

    mlir::IRMapping eltwiseMapper;
    eltwiseMapper.map(eltwiseOp->getOperand(0), affineReshapeOp.getInput());
    auto newEltwiseOp = rewriter.clone(*eltwiseOp, eltwiseMapper);

    // Optimally, just one function call should be enough:
    //     vpux::inferReturnTypes(newEltwiseOp, vpux::InferShapedTypeMode::SHAPE | vpux::InferShapedTypeMode::LAYOUT)
    // However, inferReturnTypes has an unexpected side effect.
    // Since all the IE EltwiseOps' inferReturnTypeComponents do not forward the layout info, e.g.
    //     vpux::IE::GeluOp::inferReturnTypeComponents
    // inferReturnTypes(Gelu, LAYOUT) will overwrite the original layout with empty layout data,
    // which ultimately removes the original layout.
    // Therefore, manually set the output layout is required.
    vpux::inferReturnTypes(newEltwiseOp, vpux::InferShapedTypeMode::SHAPE);
    auto dimsOrder = mlir::cast<NDTypeInterface>(newEltwiseOp->getOperand(0).getType()).getDimsOrder();
    auto newOutType = mlir::cast<NDTypeInterface>(newEltwiseOp->getResult(0).getType()).changeDimsOrder(dimsOrder);
    newEltwiseOp->getResult(0).setType(newOutType);

    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(eltwiseOp, newEltwiseOp->getResult(0),
                                                     affineReshapeOp.getDimMappingAttr(),
                                                     affineReshapeOp.getShapeValueAttr());

    return mlir::success();
}

//
// PropagateAffineReshape
//

class PropagateAffineReshape final : public IE::impl::PropagateAffineReshapeBase<PropagateAffineReshape> {
public:
    explicit PropagateAffineReshape(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PropagateAffineReshape::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto verifyAvgPool = [](mlir::Operation* op) {
        auto avgPoolOp = mlir::dyn_cast<IE::AvgPoolOp>(op);
        return (avgPoolOp != nullptr) && (IE::isEltwisePooling<IE::AvgPoolOp>(avgPoolOp));
    };

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveThroughTranspose>(&ctx, _log);
    patterns.add<MoveThroughExpand>(&ctx, _log);
    patterns.add<MoveThroughConcat>(&ctx, _log);
    patterns.add<MoveThroughSoftmax>(&ctx, _log);
    patterns.add<MoveThroughMVN>(&ctx, _log);
    patterns.add<MoveThroughEltwiseGeneric<IE::AvgPoolOp>>(&ctx, _log, verifyAvgPool);
    patterns.add<MoveThroughMultiply>(&ctx, _log);
    patterns.add<MoveThroughSlice>(&ctx, _log);
    patterns.add<IE::MoveTransposeAffineReshapeThroughAdd>(&ctx, vpux::benefitHigh, _log);
    patterns.add<MoveAffineReshapePermuteCastThroughConcat>(&ctx, _log);
    patterns.add<MoveThroughAdd>(&ctx, _log);
    patterns.add<MoveThroughOneInputEltwise>(&ctx, _log);
    IE::ReshapeOp::getCanonicalizationPatterns(patterns, &ctx);
    IE::AffineReshapeOp::getCanonicalizationPatterns(patterns, &ctx);
    IE::ShapeCastOp::getCanonicalizationPatterns(patterns, &ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    // ConcatReshapeConcat pattern is doing the opposite propagation comparing to MoveThroughConcat.
    // So we need a seperated pattern set, otherwise we might result in infinite loop between
    // ConcatReshapeConcat and MoveThroughConcat
    mlir::RewritePatternSet patternsBackward(&ctx);
    patternsBackward.add<ConcatReshapeConcat>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patternsBackward),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateAffineReshapePass(Logger log) {
    return std::make_unique<PropagateAffineReshape>(log);
}
