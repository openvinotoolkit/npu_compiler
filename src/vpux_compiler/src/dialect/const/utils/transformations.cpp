//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/utils/transformations.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"
#include "vpux/compiler/dialect/const/utils/mem_permute_optimized.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <numeric>

using namespace vpux;

namespace vpux::Const::details {

namespace {
struct OffsetShapePair {
    SmallVector<int64_t> offset;
    SmallVector<int64_t> shape;
};

// Helps identify whether the Reshape -> SubView transformations can be swapped by checking whether the parameters
// of SubView can be deduced when placed in front of Reshape. It does this by trying to find the matching dimension
// before the reshape that corresponds to a sliced dimension, where the lower dimensions have the same number of
// elements. For example:
//  - reshapeInput  = [1, 16, 8]
//  - reshapeOutput = [1, 16, 2, 4]
//  - subViewOffset = [0, 0, 0, 0]
//  - subViewShape  = [1, 8, 2, 4]
// The function identifies dimension 1 of reshapeOutput as a sliced dimension. It has the same size as dimension 1
// in reshapeInput and the same number of elements in the lower dimensions (8 == 2x4), making it possible to slice
// reshapeInput directly.
// Since there are no other sliced dimensions, the function returns offset [0, 0, 0] and shape [1, 8, 8], which will
// represent the parameters of SubView after swapping the transformations into SubView -> Reshape.
mlir::FailureOr<OffsetShapePair> sliceInputShape(ArrayRef<int64_t> reshapeInput, ArrayRef<int64_t> reshapeOutput,
                                                 ArrayRef<int64_t> subViewOffset, ArrayRef<int64_t> subViewShape) {
    SmallVector<int64_t> newOffset(reshapeInput.size(), 0);
    SmallVector<int64_t> newShape(reshapeInput);

    int64_t inputLowerDimsSize = 1;
    int64_t outputLowerDimsSize = 1;

    auto inputDim = static_cast<int64_t>(reshapeInput.size() - 1);
    for (auto outputDim = static_cast<int64_t>(reshapeOutput.size() - 1); outputDim >= 0; --outputDim) {
        const auto isDimSliced = subViewOffset[outputDim] > 0 || subViewShape[outputDim] < reshapeOutput[outputDim];
        if (!isDimSliced) {
            outputLowerDimsSize *= reshapeOutput[outputDim];
            continue;
        }
        for (; inputDim >= 0; --inputDim) {
            if (reshapeInput[inputDim] == reshapeOutput[outputDim] && inputLowerDimsSize == outputLowerDimsSize) {
                break;
            }
            inputLowerDimsSize *= reshapeInput[inputDim];
        }
        if (inputDim < 0 || reshapeInput[inputDim] != reshapeOutput[outputDim] ||
            inputLowerDimsSize != outputLowerDimsSize) {
            return mlir::failure();
        }
        newOffset[inputDim] = subViewOffset[outputDim];
        newShape[inputDim] = subViewShape[outputDim];

        outputLowerDimsSize *= reshapeOutput[outputDim];
    }
    return OffsetShapePair{std::move(newOffset), std::move(newShape)};
}

//
// MoveSubViewBefore
//

template <typename Attr>
void prepareCastElemTypeSwap(optimization::TransformAttrPos castElemTypeAttrIt) {
    const auto castElemTypeAttr = mlir::cast<Attr>(*castElemTypeAttrIt);
    const auto perAxisType =
            mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(castElemTypeAttr.getElemType());
    if (perAxisType == nullptr) {
        return;
    }
    const auto subViewAttr = mlir::cast<Const::SubViewAttr>(*(castElemTypeAttrIt + 1));
    const Shape offset(parseIntArrayAttr<int64_t>(subViewAttr.getOffset()));
    const Shape shape(parseIntArrayAttr<int64_t>(subViewAttr.getShape()));
    const auto newElemType = tileScalesAndZP(perAxisType, shape, offset);
    *castElemTypeAttrIt = Attr::get(newElemType);
}

void prepareTransposeSwap(optimization::TransformAttrPos transposeAttrIt) {
    const auto transposeAttr = mlir::cast<Const::TransposeAttr>(*transposeAttrIt);
    const auto order = DimsOrder::fromAffineMap(transposeAttr.getOrder().getValue());

    const auto subViewAttr = mlir::cast<Const::SubViewAttr>(*(transposeAttrIt + 1));
    const Shape offset(parseIntArrayAttr<int64_t>(subViewAttr.getOffset()));
    const Shape shape(parseIntArrayAttr<int64_t>(subViewAttr.getShape()));
    SmallVector<int64_t> newOffset(offset.size());
    SmallVector<int64_t> newShape(shape.size());
    for (size_t idx = 0; idx < newShape.size(); idx++) {
        newOffset[order.dimAt(idx).ind()] = offset.raw()[idx];
        newShape[order.dimAt(idx).ind()] = shape.raw()[idx];
    }
    *(transposeAttrIt + 1) = Const::SubViewAttr::get(getIntArrayAttr(transposeAttr.getContext(), newOffset),
                                                     getIntArrayAttr(transposeAttr.getContext(), newShape));
}

mlir::LogicalResult prepareReshapeSwap(optimization::TransformAttrPos reshapeAttrIt, NDTypeInterface reshapeInputType) {
    const auto reshapeAttr = mlir::cast<Const::ReshapeAttr>(*reshapeAttrIt);
    const auto reshapeInputShape = reshapeInputType.getShape();
    const auto reshapeOutputShape = parseIntArrayAttr<int64_t>(reshapeAttr.getShape());

    const auto subViewAttr = mlir::cast<Const::SubViewAttr>(*(reshapeAttrIt + 1));
    const Shape offset(parseIntArrayAttr<int64_t>(subViewAttr.getOffset()));
    const Shape shape(parseIntArrayAttr<int64_t>(subViewAttr.getShape()));
    const auto newSubViewOutput =
            sliceInputShape(reshapeInputShape.raw(), reshapeOutputShape, offset.raw(), shape.raw());
    if (mlir::failed(newSubViewOutput)) {
        return mlir::failure();
    }
    *reshapeAttrIt = Const::ReshapeAttr::get(subViewAttr.getShape());
    const auto newOffset = getIntArrayAttr(reshapeAttr.getContext(), newSubViewOutput->offset);
    const auto newShape = getIntArrayAttr(reshapeAttr.getContext(), newSubViewOutput->shape);
    *(reshapeAttrIt + 1) = Const::SubViewAttr::get(newOffset, newShape);
    return mlir::success();
}

void prepareMemPermuteSwap(optimization::TransformAttrPos memPermuteAttrIt, NDTypeInterface memPermuteInputType) {
    const auto memPermuteAttr = mlir::cast<Const::MemPermuteAttr>(*memPermuteAttrIt);

    // SubView is applied to logical shape. Since the goal is to move SubView before MemPermute we should infer offsets
    // and sizes for input logical shape of MemPermute. To do this, we perform the steps from MemPermute's type
    // inference procedure in reverse order, namely:
    // - Apply output layout (to get offsets/shapes applying output memory shape)
    // - Apply reverse of MemPermute (to get offsets/shapes applying input memory shape)
    // - Apply reverse of input layout (to get offsets/shapes applying input logical shape)
    // Or in a different representation we go from right to left ( apply reverse permutation if arrow direction is
    // opposite) input logical shape -> (Apply input layout) -> Input Memory Shape -> (ApplyMemPerm) ->   Output
    // MemoryLayout <- (ApplyOutputLayout) <- Output Logical Shape

    // Combination of the 3 permutation listed above is applied to get final permutation.

    const auto outputLayoutPermutation = memPermuteAttr.getDstOrder().getValue();

    const auto memPerm = memPermuteAttr.getMemPerm().getValue();
    const auto reverseMemPermLayout = inversePermutation(memPerm);

    const auto inputLayout = memPermuteInputType.getDimsOrder().toAffineMap(memPermuteInputType.getContext());
    const auto reverseInputLayout = inversePermutation(inputLayout);

    // Reverse order is required with compose and it is equivalent to result =
    // reverseInputLayoutTransform(reverseMemPermLayout(reverseMemPermLayout))
    const auto combinedPermutation = reverseInputLayout.compose(reverseMemPermLayout).compose(outputLayoutPermutation);

    const auto permutationOrder = DimsOrder::fromAffineMap(combinedPermutation);
    const auto subViewAttr = mlir::cast<Const::SubViewAttr>(*(memPermuteAttrIt + 1));
    const Shape offset(parseIntArrayAttr<int64_t>(subViewAttr.getOffset()));
    const Shape shape(parseIntArrayAttr<int64_t>(subViewAttr.getShape()));
    SmallVector<int64_t> newOffset(offset.size());
    SmallVector<int64_t> newShape(shape.size());
    for (size_t idx = 0; idx < newShape.size(); idx++) {
        newOffset[idx] = offset.raw()[permutationOrder.dimAt(idx).ind()];
        newShape[idx] = shape.raw()[permutationOrder.dimAt(idx).ind()];
    }
    *(memPermuteAttrIt + 1) = Const::SubViewAttr::get(getIntArrayAttr(memPermuteAttr.getContext(), newOffset),
                                                      getIntArrayAttr(memPermuteAttr.getContext(), newShape));
}

mlir::LogicalResult prepareChangeShapeSwap(optimization::TransformAttrPos changeShapeAttrIt,
                                           NDTypeInterface changeShapeInputType) {
    const auto changeShapeAttr = mlir::cast<Const::ChangeShapeAndElemTypeAttr>(*changeShapeAttrIt);
    const auto changeShapeInput = changeShapeInputType.getShape();
    const auto changeShapeOutput = parseIntArrayAttr<int64_t>(changeShapeAttr.getShape());

    const auto subViewAttr = mlir::cast<Const::SubViewAttr>(*(changeShapeAttrIt + 1));
    const Shape offset(parseIntArrayAttr<int64_t>(subViewAttr.getOffset()));
    const Shape shape(parseIntArrayAttr<int64_t>(subViewAttr.getShape()));
    const auto newSubViewOutput = sliceInputShape(changeShapeInput.raw(), changeShapeOutput, offset.raw(), shape.raw());
    if (mlir::failed(newSubViewOutput)) {
        return mlir::failure();
    }

    {
        auto outputType = changeShapeAttr.inferOutputType(changeShapeInputType);
        outputType = subViewAttr.inferOutputType(outputType);
        const auto newShape = getIntArrayAttr(changeShapeAttr.getContext(), outputType.getShape().raw());
        const auto newElemType = outputType.getElementType();
        *changeShapeAttrIt = Const::ChangeShapeAndElemTypeAttr::get(newShape, newElemType);
    }
    {
        const auto newOffset = getIntArrayAttr(changeShapeAttr.getContext(), newSubViewOutput->offset);
        const auto newShape = getIntArrayAttr(changeShapeAttr.getContext(), newSubViewOutput->shape);
        *(changeShapeAttrIt + 1) = Const::SubViewAttr::get(newOffset, newShape);
    }
    return mlir::success();
}

mlir::LogicalResult prepareRelocateWeightsTableSwap(optimization::TransformAttrPos relocateAttrIt,
                                                    NDTypeInterface relocateInputType) {
    const auto relocateAttr = mlir::cast<Const::RelocateWeightsTableAttr>(*relocateAttrIt);
    const auto subViewAttr = mlir::cast<Const::SubViewAttr>(*(relocateAttrIt + 1));
    const Shape offset(parseIntArrayAttr<int64_t>(subViewAttr.getOffset()));
    const Shape shape(parseIntArrayAttr<int64_t>(subViewAttr.getShape()));

    // More than one channel must be present for the transformation to deduce the weights pointer step
    const auto subviewSize = shape.front();
    if (subviewSize <= 1) {
        return mlir::failure();
    }

    const auto relocateOutputType = relocateAttr.inferOutputType(relocateInputType);
    const auto relocateOutputShape = relocateOutputType.getShape();

    const auto isSlicedOverFirstDim = [&]() {
        for (auto outputDim = static_cast<int64_t>(shape.size() - 1); outputDim >= 0; --outputDim) {
            const auto isDimSliced =
                    offset.raw()[outputDim] > 0 || shape.raw()[outputDim] < relocateOutputShape.raw()[outputDim];
            if (isDimSliced) {
                if (outputDim != 0) {
                    return false;
                }
            }
        }
        return true;
    }();
    if (!isSlicedOverFirstDim) {
        return mlir::failure();
    }

    const auto totalChannels = relocateOutputShape.front();
    const auto weightsTableByteSize = relocateAttr.getWeightsTableSize().getInt();
    const auto weightsTableNumElems = weightsTableByteSize / sizeof(int32_t);
    const auto tableEntrySize = weightsTableNumElems / totalChannels;

    const auto subviewOffset = offset.front();

    const auto clusterOffsets = parseIntArrayAttr<int64_t>(relocateAttr.getOffsets());
    const auto areClustersDifferent = std::adjacent_find(clusterOffsets.begin(), clusterOffsets.end(),
                                                         std::not_equal_to<>()) != clusterOffsets.end();
    SmallVector<int32_t> newWeightsPtrs = {};
    SmallVector<int64_t> newClusterOffsets(clusterOffsets);
    mlir::IntegerAttr newChannelOffsetAttr;
    const auto origChannelOffset =
            relocateAttr.getChannelOffset() != nullptr ? relocateAttr.getChannelOffset().getInt() : 0;
    if (areClustersDifferent) {
        size_t clusterIdx = 0;

        // Ensure only the values for one cluster are sliced
        bool onlyOneClusterSliced = [&]() {
            const auto offsetIt = llvm::find(clusterOffsets, subviewOffset);
            if (offsetIt == clusterOffsets.end()) {
                return false;
            }
            if (offsetIt + 1 != clusterOffsets.end()) {
                const auto nextOffset = *(offsetIt + 1);
                if ((nextOffset - subviewOffset) != subviewSize) {
                    return false;
                }
            }
            clusterIdx = static_cast<size_t>(std::distance(clusterOffsets.begin(), offsetIt));
            return true;
        }();
        if (!onlyOneClusterSliced) {
            return mlir::failure();
        }

        const auto weightsPtrs = parseIntArrayAttr<int32_t>(relocateAttr.getWeightsPtr());
        if (clusterIdx >= weightsPtrs.size()) {
            return mlir::failure();
        }
        newWeightsPtrs = {weightsPtrs[clusterIdx]};
        newClusterOffsets = {0};
        newChannelOffsetAttr = getIntAttr(relocateAttr.getContext(), origChannelOffset);
    } else {
        newWeightsPtrs = parseIntArrayAttr<int32_t>(relocateAttr.getWeightsPtr());
        newChannelOffsetAttr = getIntAttr(relocateAttr.getContext(), origChannelOffset + subviewOffset);
    }

    const auto newWeightsPtrsAttr = getIntArrayAttr(relocateAttr.getContext(), newWeightsPtrs);
    const auto newClusterOffsetsAttr = getIntArrayAttr(relocateAttr.getContext(), newClusterOffsets);

    const auto newTableByteSize = tableEntrySize * subviewSize * sizeof(int32_t);
    const auto newTableByteSizeAttr = getIntAttr(relocateAttr.getContext(), newTableByteSize);

    const auto newWeightsElemBitSizeAttr = relocateAttr.getWeightsElemBitSize();

    auto newWeightsCompressionAttr = relocateAttr.getWeightsCompression();
    // Don't slice sparsity compression when slicing same clusters since RelocateWeightsTable transform
    // will need whole numElems to determine correct weightPtr.
    if (newWeightsCompressionAttr != nullptr && areClustersDifferent) {
        if (newWeightsCompressionAttr.getAxis().getInt() != 0) {
            return mlir::failure();
        }
        const auto numElems = to_small_vector(newWeightsCompressionAttr.getNumElems().getValues<int64_t>());
        const auto newNumElems =
                SmallVector<int64_t>(numElems.begin() + subviewOffset, numElems.begin() + subviewOffset + subviewSize);
        const auto numElemsType = mlir::RankedTensorType::get({static_cast<int64_t>(newNumElems.size())},
                                                              getInt64Type(relocateAttr.getContext()));
        const auto newNumElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(newNumElems));
        newWeightsCompressionAttr =
                VPUIP::SparsityCompressionAttr::get(relocateAttr.getContext(), newWeightsCompressionAttr.getAxis(),
                                                    newNumElemsAttr, newWeightsCompressionAttr.getAlignment());
    }

    *relocateAttrIt = Const::RelocateWeightsTableAttr::get(
            newWeightsPtrsAttr, relocateAttr.getSparsityPtr(), newClusterOffsetsAttr, newTableByteSizeAttr,
            newWeightsElemBitSizeAttr, newWeightsCompressionAttr, newChannelOffsetAttr, relocateAttr.getOriginalOC());

    return mlir::success();
}

mlir::LogicalResult preparePadWithZeroSwap(optimization::TransformAttrPos padWithZeroAttrIt,
                                           NDTypeInterface padWithZeroInputType) {
    const auto padWithZeroAttr = mlir::cast<Const::PadWithZeroAttr>(*padWithZeroAttrIt);
    const auto subViewAttr = mlir::cast<Const::SubViewAttr>(*(padWithZeroAttrIt + 1));
    auto inputShape = padWithZeroInputType.getShape().raw();
    auto offset(parseIntArrayAttr<int64_t>(subViewAttr.getOffset()));
    auto shape(parseIntArrayAttr<int64_t>(subViewAttr.getShape()));
    SmallVector<int64_t> newOffset(offset.size());
    SmallVector<int64_t> newShape(shape.size());
    auto padBefore = parseIntArrayAttr<int64_t>(padWithZeroAttr.getPadBefore());
    auto padAfter = parseIntArrayAttr<int64_t>(padWithZeroAttr.getPadAfter());
    SmallVector<int64_t> newPadBefore(padBefore.size());
    SmallVector<int64_t> newPadAfter(padAfter.size());

    for (size_t idx = 0; idx < newShape.size(); idx++) {
        const auto subviewUpperBound = offset[idx] + shape[idx];
        const auto subviewLowerBound = offset[idx];

        const auto paddedShape = padBefore[idx] + inputShape[idx] + padAfter[idx];

        // check if subview is only to the zero-padded region, fail in that case
        if (subviewLowerBound >= padBefore[idx] + inputShape[idx] || subviewUpperBound <= padBefore[idx]) {
            return mlir::failure();
        }

        newPadAfter[idx] = padAfter[idx];
        newPadBefore[idx] = padBefore[idx];
        if (subviewLowerBound < padBefore[idx]) {
            newPadBefore[idx] -= offset[idx];
            newOffset[idx] = 0;
        } else {
            newOffset[idx] = offset[idx] - padBefore[idx];
            newPadBefore[idx] = 0;
        }

        if (subviewUpperBound > padBefore[idx] + inputShape[idx]) {
            newPadAfter[idx] -= (paddedShape - subviewUpperBound);
        } else {
            newPadAfter[idx] = 0;
        }

        newShape[idx] = shape[idx] - newPadBefore[idx] - newPadAfter[idx];
    }

    *(padWithZeroAttrIt + 1) = Const::SubViewAttr::get(getIntArrayAttr(padWithZeroAttr.getContext(), newOffset),
                                                       getIntArrayAttr(padWithZeroAttr.getContext(), newShape));
    *padWithZeroAttrIt = Const::PadWithZeroAttr::get(getIntArrayAttr(subViewAttr.getContext(), newPadBefore),
                                                     getIntArrayAttr(subViewAttr.getContext(), newPadAfter));
    return mlir::success();
}

//
// MoveReshapeBefore
//

mlir::LogicalResult prepareCastElemTypeSwap(optimization::TransformAttrPos castElemTypeAttrIt,
                                            NDTypeInterface castElemTypeInputType) {
    const auto castElemTypeAttr = mlir::cast<Const::CastElemTypeAttr>(*castElemTypeAttrIt);
    // Keep the logic consistent with the original code. Don't make the swap if there is behavior like QuantCast
    if (mlir::isa<mlir::quant::QuantizedType>(castElemTypeInputType.getElementType()) ||
        mlir::isa<mlir::quant::QuantizedType>(castElemTypeAttr.getElemType())) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult prepareDequantizeSwap(optimization::TransformAttrPos dequantizeAttrIt,
                                          NDTypeInterface dequantizeInputType) {
    auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(dequantizeInputType.getElementType());
    if (perAxisType == nullptr) {
        return mlir::success();
    }

    const auto getNewQuantizationDim = [](int32_t dim, ArrayRef<int64_t> inputShape,
                                          ArrayRef<int64_t> outputShape) -> mlir::FailureOr<int32_t> {
        const auto inputLowerDimsSize =
                std::accumulate(inputShape.begin() + dim, inputShape.end(), int64_t(1), std::multiplies<int64_t>());
        int64_t outputLowerDimsSize = 1;
        auto newDim = static_cast<int32_t>(outputShape.size() - 1);
        for (; newDim >= 0; --newDim) {
            outputLowerDimsSize *= outputShape[newDim];
            if (outputLowerDimsSize >= inputLowerDimsSize) {
                break;
            }
        }
        if (outputLowerDimsSize == inputLowerDimsSize && outputShape[newDim] == inputShape[dim]) {
            return newDim;
        }
        return mlir::failure();
    };

    // If the input is quantized per-axis, check whether the axis is compatible with the new shape
    const auto dim = perAxisType.getQuantizedDimension();
    const auto inputShape = dequantizeInputType.getShape();
    const auto reshapeAttr = mlir::cast<Const::ReshapeAttr>(*(dequantizeAttrIt + 1));
    const auto outputShape = parseIntArrayAttr<int64_t>(reshapeAttr.getShape());
    auto newDim = getNewQuantizationDim(dim, inputShape.raw(), outputShape);
    if (mlir::failed(newDim)) {
        return mlir::failure();
    }

    const auto newPerAxisType = changeAxis(perAxisType, newDim.value());
    *(dequantizeAttrIt + 1) = Const::ChangeShapeAndElemTypeAttr::get(reshapeAttr.getShape(), newPerAxisType);

    return mlir::success();
}

Const::CastElemTypeAttr tryFusingConsecutiveCasts(Const::CastElemTypeAttr currTransformation) {
    const bool castToQuantizedType = mlir::isa<mlir::quant::QuantizedType>(currTransformation.getElemType());
    // E#151161: fusing cast-to-quantized-type is complicated: it requires (at
    // least) expressed type modification which doesn't always agree with
    // further transformations (e.g. one could have fused casts, followed by
    // dequantize which "restores" the previous type that is now changed to
    // something else). ultimately, this can cause invalid IR. as a workaround,
    // just ignore this problem altogether by not fusing cast-to-quantized-type.
    return castToQuantizedType ? nullptr : currTransformation;
}

}  // namespace

//
// FuseConsecutiveTransformations
//

std::pair<optimization::TransformAttrPos, bool> fuseConsecutiveTransformations(
        SmallVector<Const::TransformAttrInterface>& transformations, optimization::TransformAttrPos& currPos,
        NDTypeInterface baseType) {
    if (currPos == transformations.begin() || currPos == transformations.end()) {
        return {currPos, false};
    }

    auto currTransformation = *(currPos);
    auto prevTransformation = *(currPos - 1);

    Const::TransformAttrInterface newTransformation = nullptr;
    const auto areRerorderAndMemPermute = [&](Const::TransformAttrInterface firstAttr,
                                              Const::TransformAttrInterface secondAttr) {
        return mlir::isa<Const::ReorderAttr>(firstAttr) && mlir::isa<Const::MemPermuteAttr>(secondAttr);
    };

    const auto getPrevTransformationInType = [&]() {
        auto prevTransformations = ArrayRef(transformations).drop_back((transformations.end() - currPos) + 1);
        return Const::inferFinalType(baseType, prevTransformations);
    };

    if (mlir::isa<Const::SubViewAttr>(prevTransformation) && mlir::isa<Const::SubViewAttr>(currTransformation)) {
        auto firstAttr = mlir::cast<Const::SubViewAttr>(prevTransformation);
        auto secondAttr = mlir::cast<Const::SubViewAttr>(currTransformation);
        auto firstOffset = parseIntArrayAttr<int64_t>(firstAttr.getOffset());
        auto newOffset = parseIntArrayAttr<int64_t>(secondAttr.getOffset());
        for (auto i : irange(newOffset.size())) {
            newOffset[i] += firstOffset[i];
        }
        newTransformation =
                Const::SubViewAttr::get(getIntArrayAttr(firstAttr.getContext(), newOffset), secondAttr.getShape());
    } else if (mlir::isa<Const::AddAttr>(prevTransformation) && mlir::isa<Const::AddAttr>(currTransformation)) {
        auto firstAttr = mlir::cast<Const::AddAttr>(prevTransformation);
        auto secondAttr = mlir::cast<Const::AddAttr>(currTransformation);
        auto newBias = firstAttr.getBias().getValueAsDouble() + secondAttr.getBias().getValueAsDouble();
        newTransformation = Const::AddAttr::get(getFPAttr(firstAttr.getContext(), newBias));
    } else if (mlir::isa<Const::RescaleAttr>(prevTransformation) && mlir::isa<Const::RescaleAttr>(currTransformation)) {
        auto firstAttr = mlir::cast<Const::RescaleAttr>(prevTransformation);
        auto secondAttr = mlir::cast<Const::RescaleAttr>(currTransformation);
        auto newScale = firstAttr.getScale().getValueAsDouble() * secondAttr.getScale().getValueAsDouble();
        newTransformation = Const::RescaleAttr::get(getFPAttr(firstAttr.getContext(), newScale));
    } else if ((mlir::isa<Const::ReshapeAttr>(prevTransformation) &&
                mlir::isa<Const::ReshapeAttr>(currTransformation)) ||
               (mlir::isa<Const::ReorderAttr>(prevTransformation) &&
                mlir::isa<Const::ReorderAttr>(currTransformation))) {
        newTransformation = currTransformation;
    } else if (areRerorderAndMemPermute(prevTransformation, currTransformation) ||
               areRerorderAndMemPermute(currTransformation, prevTransformation)) {
        auto reorderInType = getPrevTransformationInType();

        mlir::AffineMapAttr newMemPermAttr;
        mlir::AffineMapAttr lastOrder;

        const auto getReorderPerm = [](NDTypeInterface reorderInType, ReorderAttr reorderAttr) {
            auto inOrder = reorderInType.getDimsOrder();
            auto outOrder = DimsOrder::fromAffineMap(reorderAttr.getOrder().getValue());
            return getPermutationFromOrders(inOrder, outOrder, reorderAttr.getContext());
        };

        if (mlir::isa<MemPermuteAttr>(prevTransformation)) {
            auto memPermAttr = mlir::cast<MemPermuteAttr>(prevTransformation);
            auto reorderAttr = mlir::cast<ReorderAttr>(currTransformation);

            auto memPerm = memPermAttr.getMemPerm().getValue();

            reorderInType = memPermAttr.inferOutputType(reorderInType);
            auto reorderPerm = getReorderPerm(reorderInType, reorderAttr);

            newMemPermAttr = mlir::AffineMapAttr::get(reorderPerm.compose(memPerm));
            lastOrder = reorderAttr.getOrder();
        } else {
            auto reorderAttr = mlir::cast<ReorderAttr>(prevTransformation);
            auto memPermAttr = mlir::cast<MemPermuteAttr>(currTransformation);

            auto memPerm = memPermAttr.getMemPerm().getValue();
            auto reorderPerm = getReorderPerm(reorderInType, reorderAttr);

            newMemPermAttr = mlir::AffineMapAttr::get(memPerm.compose(reorderPerm));
            lastOrder = memPermAttr.getDstOrder();
        }

        newTransformation = MemPermuteAttr::get(lastOrder, newMemPermAttr);
    } else if (auto currT = mlir::dyn_cast<Const::CastElemTypeAttr>(currTransformation);
               mlir::isa<Const::CastElemTypeAttr>(prevTransformation) && currT != nullptr) {
        newTransformation = tryFusingConsecutiveCasts(currT);
    }

    if (newTransformation != nullptr) {
        *(currPos - 1) = newTransformation;
        return {transformations.erase(currPos) - 1, true};
    }

    return {currPos, false};
}

//
// FoldTransformation
//

std::pair<optimization::TransformAttrPos, bool> foldTransformation(
        SmallVector<Const::TransformAttrInterface>& transformations, optimization::TransformAttrPos& currPos,
        NDTypeInterface baseType) {
    if (currPos == transformations.end()) {
        return {currPos, false};
    }

    const auto getCurrTransformationInType = [&]() {
        auto prevTransformations = ArrayRef(transformations).drop_back(transformations.end() - currPos);
        return Const::inferFinalType(baseType, prevTransformations);
    };

    if (auto reorderAttr = mlir::dyn_cast<ReorderAttr>(*currPos)) {
        auto currTransformationInType = getCurrTransformationInType();
        auto reorderOutType = reorderAttr.inferOutputType(currTransformationInType);

        if (reorderOutType == currTransformationInType) {
            return {transformations.erase(currPos), true};
        }
    } else if (auto memPermAttr = mlir::dyn_cast<MemPermuteAttr>(*currPos)) {
        auto currTransformationInType = getCurrTransformationInType();
        auto memPermOutType = memPermAttr.inferOutputType(currTransformationInType);
        auto memPerm = memPermAttr.getMemPerm().getValue();

        if (memPermOutType == currTransformationInType && memPerm.isIdentity()) {
            return {transformations.erase(currPos), true};
        }
    } else if (auto transposeAttr = mlir::dyn_cast<TransposeAttr>(*currPos)) {
        auto currTransformationInType = getCurrTransformationInType();
        auto transposeOutType = transposeAttr.inferOutputType(currTransformationInType);
        auto memPerm = transposeAttr.getOrder().getValue();

        // Note: identity check is important here since transposing e.g. 2x2
        // tensor does not modify the type, but still performs the
        // transposition.
        if (transposeOutType == currTransformationInType && memPerm.isIdentity()) {
            return {transformations.erase(currPos), true};
        }
    }

    return {currPos, false};
}

//
// MoveSubViewBefore
//

std::pair<optimization::TransformAttrPos, bool> moveSubViewBefore(
        SmallVector<Const::TransformAttrInterface>& transformations, optimization::TransformAttrPos& currPos,
        NDTypeInterface baseType) {
    if (currPos == transformations.begin() || currPos == transformations.end()) {
        return {currPos, false};
    }

    auto currTransformation = *(currPos);
    auto prevTransformation = *(currPos - 1);

    if (!mlir::isa<Const::SubViewAttr>(currTransformation) ||
        !mlir::isa<Const::AddAttr, Const::RescaleAttr, Const::CastElemTypeAttr, Const::DequantizeAttr,
                   Const::ReorderAttr, Const::MemPermuteAttr, Const::ConvertElemTypeAttr, Const::TransposeAttr,
                   Const::ReshapeAttr, Const::ChangeShapeAndElemTypeAttr, Const::RelocateWeightsTableAttr,
                   Const::PadWithZeroAttr>(prevTransformation)) {
        return {currPos, false};
    }

    const auto swapTransformations =
            [&](optimization::TransformAttrPos prevIt,
                optimization::TransformAttrPos currentIt) -> std::pair<optimization::TransformAttrPos, bool> {
        std::iter_swap(prevIt, currentIt);

        return {prevIt, true};
    };

    NDTypeInterface prevTransformationInType = baseType;
    for (auto tmpIt = transformations.begin(); tmpIt < (currPos - 1); tmpIt++) {
        prevTransformationInType = (*tmpIt).inferOutputType(prevTransformationInType);
    }

    auto result =
            llvm::TypeSwitch<Const::TransformAttrInterface, std::pair<optimization::TransformAttrPos, bool>>(
                    prevTransformation)
                    .Case<Const::AddAttr, Const::RescaleAttr, Const::DequantizeAttr, Const::ReorderAttr>(
                            [&](Const::TransformAttrInterface /*transformation*/) {
                                return swapTransformations(currPos - 1, currPos);
                            })
                    .Case<Const::ConvertElemTypeAttr>([&](Const::ConvertElemTypeAttr) {
                        prepareCastElemTypeSwap<Const::ConvertElemTypeAttr>(currPos - 1);
                        return swapTransformations(currPos - 1, currPos);
                    })
                    .Case<Const::CastElemTypeAttr>([&](Const::CastElemTypeAttr) {
                        prepareCastElemTypeSwap<Const::CastElemTypeAttr>(currPos - 1);
                        return swapTransformations(currPos - 1, currPos);
                    })
                    .Case<Const::TransposeAttr>([&](Const::TransposeAttr) {
                        prepareTransposeSwap(currPos - 1);
                        return swapTransformations(currPos - 1, currPos);
                    })
                    .Case<Const::MemPermuteAttr>([&](Const::MemPermuteAttr) {
                        prepareMemPermuteSwap(currPos - 1, prevTransformationInType);
                        return swapTransformations(currPos - 1, currPos);
                    })
                    .Case<Const::ReshapeAttr>([&](Const::ReshapeAttr) {
                        if (mlir::failed(prepareReshapeSwap(currPos - 1, prevTransformationInType))) {
                            return std::pair<optimization::TransformAttrPos, bool>{currPos, false};
                        }
                        return swapTransformations(currPos - 1, currPos);
                    })
                    .Case<Const::ChangeShapeAndElemTypeAttr>([&](Const::ChangeShapeAndElemTypeAttr) {
                        if (mlir::failed(prepareChangeShapeSwap(currPos - 1, prevTransformationInType))) {
                            return std::pair<optimization::TransformAttrPos, bool>{currPos, false};
                        }
                        return swapTransformations(currPos - 1, currPos);
                    })
                    .Case<Const::RelocateWeightsTableAttr>([&](Const::RelocateWeightsTableAttr) {
                        if (mlir::failed(prepareRelocateWeightsTableSwap(currPos - 1, prevTransformationInType))) {
                            return std::pair<optimization::TransformAttrPos, bool>{currPos, false};
                        }
                        return swapTransformations(currPos - 1, currPos);
                    })
                    .Case<Const::PadWithZeroAttr>([&](Const::PadWithZeroAttr) {
                        if (mlir::failed(preparePadWithZeroSwap(currPos - 1, prevTransformationInType))) {
                            return std::pair<optimization::TransformAttrPos, bool>{currPos, false};
                        }
                        return swapTransformations(currPos - 1, currPos);
                    })
                    .Default([&](Const::TransformAttrInterface /*transformation*/) {
                        return std::pair<optimization::TransformAttrPos, bool>{currPos, false};
                    });

    if (result.second && mlir::isa<Const::PadWithZeroAttr>(prevTransformation)) {
        // When SubView is swapped with PadWithZero it is possible that PadWithZero has all pad dimensions
        // set to 0 and SubView is covering entire input constant. Following code detects such redundant
        // transformations and removes them from the list
        SmallVector<Const::TransformAttrInterface> optimizedTransformations;

        auto subViewPos = currPos - 1;
        auto padWithZeroPos = currPos;
        bool subViewFolded = false;

        auto subViewAttr = mlir::cast<Const::SubViewAttr>(*subViewPos);
        auto outputType = subViewAttr.inferOutputType(prevTransformationInType);
        if (outputType == prevTransformationInType) {
            padWithZeroPos = transformations.erase(subViewPos);
            prevTransformationInType = outputType;
            subViewFolded = true;
        }

        auto padWithZeroAttr = mlir::cast<Const::PadWithZeroAttr>(*padWithZeroPos);
        outputType = padWithZeroAttr.inferOutputType(prevTransformationInType);
        if (outputType == prevTransformationInType) {
            padWithZeroPos = transformations.erase(padWithZeroPos);
        }

        if (subViewFolded) {
            return {transformations.end(), true};
        }

        return {padWithZeroPos - 1, true};
    }

    return result;
}

//
// MoveReshapeBefore
//

std::pair<optimization::TransformAttrPos, bool> moveReshapeBefore(
        SmallVector<Const::TransformAttrInterface>& transformations, optimization::TransformAttrPos& currPos,
        NDTypeInterface baseType) {
    if (currPos == transformations.begin() || currPos == transformations.end()) {
        return {currPos, false};
    }

    auto currTransformation = *(currPos);
    auto prevTransformation = *(currPos - 1);

    if (!mlir::isa<Const::ReshapeAttr>(currTransformation) ||
        !mlir::isa<Const::AddAttr, Const::RescaleAttr, Const::CastElemTypeAttr, Const::DequantizeAttr>(
                prevTransformation)) {
        return {currPos, false};
    }

    const auto swapTransformations =
            [&](optimization::TransformAttrPos prevIt,
                optimization::TransformAttrPos currentIt) -> std::pair<optimization::TransformAttrPos, bool> {
        std::iter_swap(prevIt, currentIt);

        return {prevIt, true};
    };

    auto prevTransformations = ArrayRef(transformations).drop_back((transformations.end() - currPos) + 1);
    auto prevTransformationInType = Const::inferFinalType(baseType, prevTransformations);

    auto result =
            llvm::TypeSwitch<Const::TransformAttrInterface, std::pair<optimization::TransformAttrPos, bool>>(
                    prevTransformation)
                    .Case<Const::AddAttr, Const::RescaleAttr>([&](Const::TransformAttrInterface /*transformation*/) {
                        return swapTransformations(currPos - 1, currPos);
                    })
                    .Case<Const::CastElemTypeAttr>([&](Const::CastElemTypeAttr) {
                        if (mlir::failed(prepareCastElemTypeSwap(currPos - 1, prevTransformationInType))) {
                            return std::pair<optimization::TransformAttrPos, bool>{currPos, false};
                        }
                        return swapTransformations(currPos - 1, currPos);
                    })
                    .Case<Const::DequantizeAttr>([&](Const::DequantizeAttr) {
                        if (mlir::failed(prepareDequantizeSwap(currPos - 1, prevTransformationInType))) {
                            return std::pair<optimization::TransformAttrPos, bool>{currPos, false};
                        }
                        return swapTransformations(currPos - 1, currPos);
                    })
                    .Default([&](Const::TransformAttrInterface /*transformation*/) {
                        return std::pair<optimization::TransformAttrPos, bool>{currPos, false};
                    });

    return result;
}

mlir::FailureOr<Const::FuseWeightsAttr> moveSubViewIntoFuse(Const::FuseWeightsAttr fuseAttr,
                                                            Const::SubViewAttr subViewAttr) {
    // Can't move subview into sparse fused constant as the content setup logic will
    // move subview before sparsify transformation without any adjustment(sparsify is preferred last).
    if (fuseAttr.getSparsity() != nullptr) {
        return mlir::failure();
    }
    // Below code assumes that all fused constants are of flat shape 1x1x1xn. Skip fused constants
    // of different shape to avoid any issues.
    // Similarly subview can only slice 4th dimension of a fused constant
    const auto fusedShape = fuseAttr.getFusedType().getShape();
    const int64_t flatDimensionIndex = 3;
    const int64_t fusedConstantShapeSize = 4;
    const auto subViewOffset(parseIntArrayAttr<int64_t>(subViewAttr.getOffset()));
    const auto subViewShape(parseIntArrayAttr<int64_t>(subViewAttr.getShape()));
    if ((fusedShape.size() != fusedConstantShapeSize) || (subViewOffset.size() != fusedConstantShapeSize) ||
        (subViewShape.size() != fusedConstantShapeSize)) {
        return mlir::failure();
    }
    for (int64_t dim = 0; dim < flatDimensionIndex; dim++) {
        if (fusedShape[dim] != 1 || subViewOffset[dim] != 0 || subViewShape[dim] != 1) {
            return mlir::failure();
        }
    }

    int64_t currentOffsetInFusedBuffer = 0;
    const auto fusedElemTypeSize = vpux::Byte(getElemTypeSize(fuseAttr.getFusedType().getElementType())).count();
    const int64_t subViewStart = subViewOffset[flatDimensionIndex] * fusedElemTypeSize;
    const int64_t subViewEnd = subViewStart + (subViewShape[flatDimensionIndex] * fusedElemTypeSize);

    auto getOverlapRegionBounds = [](const int64_t constantStart, const int64_t constantEnd, const int64_t subViewStart,
                                     const int64_t subViewEnd) -> std::pair<int64_t, int64_t> {
        int subViewConstantEnd = 0;
        int subViewConstantStart = 0;
        if ((constantEnd > constantStart) && (subViewEnd > subViewStart) &&
            ((subViewStart >= constantStart && subViewStart < constantEnd) ||
             (subViewEnd > constantStart && subViewEnd < constantEnd))) {
            subViewConstantStart = subViewStart >= constantStart ? subViewStart : constantStart;
            subViewConstantEnd = subViewEnd >= constantEnd ? constantEnd : subViewEnd;
        }
        return std::make_pair(subViewConstantStart, subViewConstantEnd);
    };

    auto getNonFlatSubViewOffsetAndShape =
            [](ArrayRef<int64_t> originalShape, int flatOffset,
               int flatNumOfElems) -> std::tuple<SmallVector<int64_t>, SmallVector<int64_t>, int64_t> {
        SmallVector<int64_t> offset(originalShape.size(), 0);
        SmallVector<int64_t> shape(originalShape);
        int64_t previousDimsStride = 1;
        int64_t numOfElems = 1;
        int64_t coveredOffset = 0;
        int64_t dimensionToSlice = shape.size() - 1;
        for (; dimensionToSlice >= 0; dimensionToSlice--) {
            numOfElems *= (originalShape[dimensionToSlice]);
            if (numOfElems >= flatNumOfElems + flatOffset) {
                break;
            }
            previousDimsStride = numOfElems;
        }
        offset[dimensionToSlice] =
                static_cast<int64_t>(std::floor(static_cast<double>(flatOffset) / previousDimsStride));
        coveredOffset = offset[dimensionToSlice] * previousDimsStride;
        auto coveredDims =
                static_cast<int64_t>(std::ceil(static_cast<double>(flatNumOfElems + flatOffset) / previousDimsStride));
        shape[dimensionToSlice] = coveredDims - offset[dimensionToSlice];
        for (int64_t dim = dimensionToSlice - 1; dim >= 0; dim--) {
            shape[dim] = 1;
        }
        return std::make_tuple(offset, shape, coveredOffset);
    };

    auto getNewConstant = [&](Const::ContentAttr& constant, int64_t& constantStartInFusedBuffer) -> Const::ContentAttr {
        const int64_t constantStart = constantStartInFusedBuffer;
        const int64_t constantEnd = constantStart + constant.getType().getTotalAllocSize().count();
        auto constantElemType = constant.getType().getElementType();
        // If subview is slicing quantized axis, number of scales per quantized axis might mismatch dimension
        // after subview which will trigger errors during verification. To avoid this below code casts quantized
        // type into storage type. This is safe as nobody is expected to dequantize weights that were already fused
        // and after fused constant folding we would get storage type in fused buffer anyway.
        if (auto quantizedType = mlir::dyn_cast_or_null<mlir::quant::QuantizedType>(constantElemType)) {
            constant = constant.transform().castElemType(normalizeQuantStorageType(quantizedType)).get();
            constantElemType = constant.getType().getElementType();
        }

        auto constantElemTypeSize = vpux::Bit(getElemTypeSize(constantElemType)).count();
        auto [constantSubViewStart, constantSubViewEnd] =
                getOverlapRegionBounds(constantStart, constantEnd, subViewStart, subViewEnd);
        if (constantSubViewStart == 0 && constantSubViewEnd == 0) {
            constantStartInFusedBuffer = constantEnd;
            return Const::ContentAttr{};
        }

        constexpr int64_t bitsInByte = 8;
        auto flatSubViewOffset =
                (constantSubViewStart - constantStartInFusedBuffer) * bitsInByte / constantElemTypeSize;
        auto flatSubViewShape = (constantSubViewEnd - constantSubViewStart) * bitsInByte / constantElemTypeSize;
        auto [newSubViewOffset, newSubViewShape, flatOffsetCorrection] = getNonFlatSubViewOffsetAndShape(
                constant.getType().getShape().raw(), flatSubViewOffset, flatSubViewShape);
        auto subview = Const::SubViewAttr::get(getIntArrayAttr(fuseAttr.getContext(), newSubViewOffset),
                                               getIntArrayAttr(fuseAttr.getContext(), newSubViewShape));
        if (constant.getType() != subview.inferOutputType(constant.getType())) {
            constant = constant.transform().addTransformation(subview).get();
        }

        // Flatten the constant and insert SubView into flat constant if needed. This is done
        // to account for the case when non-flat SubView size doesn't match the flat SubView size.
        // Such cases might arise when flat SubView is not aligned to the stride of final dimension
        // of a constant
        int flatNumOfElems = constant.getType().getTotalAllocSize().count() * bitsInByte / constantElemTypeSize;
        if (!((flatNumOfElems == flatSubViewShape) && (flatSubViewOffset - flatOffsetCorrection == 0))) {
            SmallVector<int64_t> flatShape({1, 1, 1, flatNumOfElems});
            constant = constant.transform()
                               .reshape(ShapeRef(flatShape))
                               .subview(ShapeRef{0, 0, 0, flatSubViewOffset - flatOffsetCorrection},
                                        ShapeRef{1, 1, 1, flatSubViewShape})
                               .get();
        }
        constantStartInFusedBuffer = constantEnd;
        return constant;
    };

    Const::ContentAttr newWeightsTable{};
    Const::ContentAttr newWeights{};
    Const::ContentAttr newActivations{};
    auto weightsTable = fuseAttr.getWeightsTable();
    if (weightsTable != nullptr) {
        newWeightsTable = getNewConstant(weightsTable, currentOffsetInFusedBuffer);
    }
    auto weights = fuseAttr.getWeights();
    if (weights != nullptr) {
        newWeights = getNewConstant(weights, currentOffsetInFusedBuffer);
    }
    auto activations = fuseAttr.getActivations();
    if (activations != nullptr) {
        newActivations = getNewConstant(activations, currentOffsetInFusedBuffer);
    }
    auto newFusedType = mlir::cast<mlir::RankedTensorType>(subViewAttr.inferOutputType(fuseAttr.getFusedType()));
    return Const::FuseWeightsAttr::get(fuseAttr.getContext(), newFusedType, std::move(newWeightsTable),
                                       std::move(newWeights), fuseAttr.getSparsity(), std::move(newActivations));
}

//
// moveTransformationIntoFuse
//

std::pair<optimization::TransformAttrPos, bool> moveTransformationIntoFuse(
        SmallVector<Const::TransformAttrInterface>& transformations, optimization::TransformAttrPos& currPos) {
    if (currPos == transformations.begin() || currPos == transformations.end()) {
        return {currPos, false};
    }

    auto currTransformation = *(currPos);
    auto prevTransformation = *(currPos - 1);

    Const::TransformAttrInterface newFuse = nullptr;
    if (auto fuseAttr = mlir::dyn_cast<Const::FuseWeightsAttr>(prevTransformation)) {
        if (auto relocateAttr = mlir::dyn_cast<Const::RelocateWeightsTableAttr>(currTransformation)) {
            auto newWT = fuseAttr.getWeightsTable().transform().addTransformation(relocateAttr).get();
            newFuse = Const::FuseWeightsAttr::get(fuseAttr.getContext(), fuseAttr.getFusedType(), std::move(newWT),
                                                  fuseAttr.getWeights(), fuseAttr.getSparsity(),
                                                  fuseAttr.getActivations());
        } else if (auto subViewAttr = mlir::dyn_cast<Const::SubViewAttr>(currTransformation)) {
            auto newFuseOrFailure = moveSubViewIntoFuse(fuseAttr, subViewAttr);
            if (mlir::failed(newFuseOrFailure)) {
                return {currPos, false};
            }
            newFuse = newFuseOrFailure.value();
        }
    }

    if (newFuse != nullptr) {
        *(currPos - 1) = newFuse;
        return {transformations.erase(currPos) - 1, true};
    }

    return {currPos, false};
}

int64_t getValueRangeOffset(mlir::quant::QuantizedType inType, mlir::quant::QuantizedType outType) {
    const bool isSupportedConversion =
            mlir::isa<mlir::IntegerType>(inType.getStorageType()) &&
            mlir::isa<mlir::IntegerType>(outType.getStorageType()) &&
            inType.getStorageType().getIntOrFloatBitWidth() == outType.getStorageType().getIntOrFloatBitWidth();
    VPUX_THROW_UNLESS(isSupportedConversion, "Unsupported conversion: {0} -> {1}", inType, outType);

    const auto inZp = extractSingleZeroPoint(inType);
    const auto outZp = extractSingleZeroPoint(outType);
    VPUX_THROW_UNLESS(inZp.has_value() && outZp.has_value(), "Unsupported conversion: {0} -> {1}", inType, outType);

    // Note: the assumption here is that when quantized type is converted to
    // another quantized type, in reality only value range is shifted. in this
    // case, zero-point of the type must also be updated accordingly (since
    // zero-point shifts with the whole range). thus, the reverse is also true:
    // shift in zero-point is the shift of the value range.
    const auto offset = outZp.value() - inZp.value();
    return offset;
}

}  // namespace vpux::Const::details

//
// memPermuteTransformation
//

Const::Content Const::details::memPermuteTransformation(vpux::Const::Content& input, vpux::NDTypeInterface outType,
                                                        mlir::AffineMap memPerm) {
    const auto inOrder = input.getType().getDimsOrder();
    const auto outOrder = outType.getDimsOrder();
    const auto permOrder = DimsOrder::fromAffineMap(memPerm);
    VPUX_THROW_UNLESS(inOrder.numDims() == outOrder.numDims(), "Can't reorder from '{0}' to '{1}'", inOrder, outOrder);
    VPUX_THROW_UNLESS(inOrder.numDims() == permOrder.numDims(), "Can't reorder from '{0}' to '{1}'", inOrder,
                      permOrder);

    auto inMemShape = inOrder.toMemoryOrder(input.getType().getShape());
    if (input.isSplat() || isTrivialPermute(inMemShape, memPerm)) {
        return Const::Content::moveBuffer(outType, std::move(input));
    } else {
        auto output = Const::Content::allocTempBuffer(outType, input.getStorageElemType(), input.isSplat());
        auto outBuf = output.getRawTempBuf();
        const auto inBuf = input.getRawStorageBuf();
        VPUX_THROW_UNLESS(outBuf.size() == inBuf.size(), "Storage buffer size mismatch in 'memPermuteTransformation'");

        const Byte elemSize = getElemTypeSize(input.getStorageElemType());
        const auto inShape = input.getType().getShape();
        const auto inMemShape = inOrder.toMemoryOrder(inShape);

        // Check capability to use specific solution. Most transforms are between NCHW and NHWC layouts, so they
        // implemented separatly
        if (Const::details::isOptimizedTransformationSupported(input, outType)) {
            Const::details::memPermuteTransformationOptimized(input, output);
        } else {
            // Use generic algorithm
            const auto outShape = outType.getShape();
            const auto outMemShape = outOrder.toMemoryOrder(outShape);

            loop_1d(LoopExecPolicy::Parallel, memPerm.getContext(), input.getType().getNumElements(),
                    [&](int64_t inMemInd1D) {
                        const auto inMemIndND = getMemIndexND(inMemInd1D, inMemShape);
                        const auto outMemIndND = permOrder.toMemoryOrder(ShapeRef(inMemIndND.raw()));
                        const auto outMemInd1D = getMemIndex1D(outMemIndND, outMemShape);

                        const auto inMemRawInd = checked_cast<size_t>(inMemInd1D * elemSize.count());
                        VPUX_THROW_UNLESS(inMemRawInd < inBuf.size(),
                                          "Out-of-bound access in 'memPermuteTransformation'");

                        const auto outMemRawInd = checked_cast<size_t>(outMemInd1D * elemSize.count());
                        VPUX_THROW_UNLESS(outMemRawInd < outBuf.size(),
                                          "Out-of-bound access in 'memPermuteTransformation'");

                        std::copy_n(inBuf.data() + inMemRawInd, checked_cast<size_t>(elemSize.count()),
                                    outBuf.data() + outMemRawInd);
                    });
        }
        return output;
    }
}
