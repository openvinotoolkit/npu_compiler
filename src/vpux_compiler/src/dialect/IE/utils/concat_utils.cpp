//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

namespace vpux {
namespace IE {

mlir::DenseSet<int64_t> getConcatAxes(IE::ConcatOp concatOp) {
    auto outputShape = getShape(concatOp.getResult());
    mlir::DenseSet<int64_t> affectedAxes;

    for (auto input : concatOp.getInputs()) {
        auto inputShape = getShape(input);
        for (size_t ind = 0; ind < outputShape.size(); ++ind) {
            const auto dim = Dim(ind);
            if (inputShape[dim] != outputShape[dim]) {
                affectedAxes.insert(dim.ind());
            }
        }
    }

    return affectedAxes;
}

std::optional<std::pair<Dim, Shape>> inferOutputShapeAfterAffineReshapeBeforeConcat(mlir::Value curInput,
                                                                                    IE::ConcatOp concatOp,
                                                                                    IE::AffineReshapeOp reshapeOp) {
    const auto concatAxes = getConcatAxes(concatOp);
    if (concatAxes.size() != 1) {
        return std::nullopt;
    }

    const auto curInputShape = getShape(curInput);
    const auto concatAxis = Dim(*concatAxes.begin());
    const auto affineOutShape = getShape(reshapeOp.getOutput());
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(reshapeOp.getDimMapping());

    const auto concatDimMapping = dimMapping[concatAxis.ind()];
    auto newConcatAxis = Dim(concatDimMapping[0]);
    size_t newConcatAxisSize = 1;
    auto newAffineOutShape = Shape(affineOutShape.raw());

    // There are three scenarios for the "Concat -> AffineReshape" pattern:
    // Pattern 1. Concat axis size remains unchanged after AffineReshape:
    //    - Concat example: [2, 3, 4] + [3, 3, 4] = [5, 3, 4]
    //    - AffineReshape example: [[0], [1], [1]] results in [5, 12]
    //    - newConcatAxis: Dim(0); newAffineOutShape: [2, 12] + [3, 12]
    // Pattern 2. Concat axis size is merged after AffineReshape:
    //    - Concat example: [3, 3, 4] + [3, 6, 4] = [3, 9, 4]
    //    - AffineReshape example: [[0], [1], [1]] results in [3, 36]
    //    - newConcatAxis: Dim(1); newAffineOutShape: [3, 12] + [3, 24]
    // Pattern 3. Concat axis size is split after AffineReshape:
    //    - Concat example: [3, 4] + [6, 4] = [9, 4]
    //    - AffineReshape example: [[0, 1], [2]] results in [3, 3, 4]
    //    - newConcatAxis: Dim(1); newAffineOutShape: [3, 1, 4] + [3, 2, 4]
    if (concatDimMapping.size() > 1) {
        size_t accumuVal = 1;
        bool axisFound = false;

        for (size_t idx = 0; idx < concatDimMapping.size(); ++idx) {
            const auto dim = Dim(concatDimMapping[idx]);
            if (!axisFound && affineOutShape[dim] > 1) {
                axisFound = true;
                newConcatAxis = dim;
            } else {
                accumuVal *= affineOutShape[dim];
            }
        }

        // Pattern 3 negative scenario
        // - Concat example: [2, 4] + [7, 4] = [9, 4]
        // - AffineReshape example: [[0, 1], [2]] results in [3, 3, 4]
        if (curInputShape[concatAxis] % accumuVal != 0) {
            return std::nullopt;
        }

        newConcatAxisSize = curInputShape[concatAxis] / accumuVal;
    } else {
        // Handle Pattern 1 and Pattern 2
        for (size_t dimIdx = 0; dimIdx < dimMapping.size(); ++dimIdx) {
            const auto& curDimMapping = dimMapping[dimIdx];
            if (curDimMapping.size() == 1 && curDimMapping[0] == newConcatAxis.ind()) {
                newConcatAxisSize *= curInputShape[Dim(dimIdx)];
            }
        }
    }

    newAffineOutShape[newConcatAxis] = newConcatAxisSize;
    return std::make_pair(newConcatAxis, std::move(newAffineOutShape));
}

mlir::ArrayAttr inferConcatOffsets(ArrayRef<ShapeRef> concatInShapes, const Dim concatDim, mlir::MLIRContext* ctx) {
    SmallVector<SmallVector<int64_t>> offsetsList(concatInShapes.size(),
                                                  SmallVector<int64_t>(concatInShapes[0].size(), 0));
    int64_t currentOffset = 0;

    for (size_t idx = 0; idx < concatInShapes.size(); idx++) {
        auto shape = concatInShapes[idx];
        int64_t dimSize = shape[concatDim];
        offsetsList[idx][concatDim.ind()] = currentOffset;
        currentOffset += dimSize;
    }

    return getIntArrayOfArray(ctx, ArrayRef(offsetsList));
}

mlir::Value createPaddingConstForConcat(ArrayRef<int64_t> constShape, mlir::Location loc,
                                        vpux::NDTypeInterface inputType, double padValue,
                                        mlir::PatternRewriter& rewriter) {
    const auto origElemType = inputType.getElementType();
    const auto padDataStorageType =
            mlir::RankedTensorType::get(constShape, mlir::Float32Type::get(rewriter.getContext()));
    const auto padDataStorage = Const::createConstContent(padDataStorageType, ArrayRef(static_cast<float>(padValue)));

    const auto padDataType = mlir::RankedTensorType::get(constShape, origElemType);
    auto padDataAttr = Const::ContentAttr::get(
            padDataStorage, Const::ContentSetup(padDataStorage, padDataStorageType).castElemType(origElemType));

    auto constant = rewriter.create<Const::DeclareOp>(loc, padDataType, std::move(padDataAttr));

    const auto dataOrder = inputType.getDimsOrder();
    const auto orderMap = dataOrder.toAffineMap(rewriter.getContext());
    return rewriter.createOrFold<IE::ReorderOp>(loc, constant.getOutput(), orderMap);
}

const mlir::ArrayAttr inferOffsetsAttrWithAxis(IE::ConcatOp origOp, int64_t& axis) {
    auto rank = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getRank();

    SmallVector<SmallVector<int64_t>> finalOffsets;
    finalOffsets.reserve(origOp.getInputs().size());
    finalOffsets.push_back(SmallVector<int64_t>(rank, 0));
    if (axis < 0) {
        axis += rank;
    }

    auto inputs = llvm::drop_end(origOp.getInputs(), 1);
    for (auto input : llvm::enumerate(inputs)) {
        auto inputShape = getShape(input.value());
        auto offsets = SmallVector<int64_t>(rank, 0);
        offsets[axis] = inputShape[Dim(axis)] + finalOffsets.back()[axis];
        finalOffsets.push_back(offsets);
    }

    return getIntArrayOfArray(origOp.getContext(), finalOffsets);
}

std::optional<vpux::Dim> getConcatAxis(IE::ConcatOp concatOp) {
    if (concatOp.getPerAxisAttr()) {
        if (concatOp.getPerAxisAttr().getStride()) {
            return std::nullopt;
        }
        return Dim(concatOp.getPerAxisAttr().getAxis().getValue().getSExtValue());
    }

    const auto concatAxes =
            vpux::IE::getDiffInOutSizeDims(getShape(concatOp.getOperands()[0]), getShape(concatOp.getResult()));
    if (concatAxes.empty() || concatAxes.size() != 1) {
        return std::nullopt;
    }

    const auto concatAxis = concatAxes.front();
    // Should to ensure there is no data overlapped
    VPUX_THROW_UNLESS(concatOp.getStaticOffsetsAttr() != nullptr, "Cannot get StaticOffsetsAttr");
    const auto allOffsets = concatOp.getStaticOffsetsAttr().getAsRange<mlir::ArrayAttr>();

    int64_t accumulator = 0;
    for (const auto& p : zip(concatOp.getInputs(), allOffsets)) {
        const auto inputShape = getShape(std::get<0>(p));
        const auto offsets = parseIntArrayAttr<int64_t>(std::get<1>(p));

        if (accumulator != offsets[concatAxis.ind()]) {
            return std::nullopt;
        }
        accumulator += inputShape[concatAxis];
    }

    if (accumulator != getShape(concatOp.getResult())[concatAxis]) {
        return std::nullopt;
    }

    return concatAxis;
}

mlir::FailureOr<SmallVector<Dim>> getConcatDimWithShape1(IE::ConcatOp concatOp, bool supportAdjacentDims) {
    const auto concatStaticOffsets = concatOp.getStaticOffsets().value();
    if (concatStaticOffsets.size() != concatOp.getInputs().size()) {
        return mlir::failure();
    }

    const auto concatInputType = mlir::cast<vpux::NDTypeInterface>(concatOp.getInputs()[0].getType());
    const auto concatOutputType = mlir::cast<vpux::NDTypeInterface>(concatOp.getOutput().getType());
    const auto concatInShape = concatInputType.getShape();
    const auto concatOutShape = concatOutputType.getShape();
    if (concatInShape.size() != concatOutShape.size()) {
        return mlir::failure();
    }

    SmallVector<Dim> concatDims;
    for (const auto& idx : irange(concatInShape.size())) {
        if (concatInShape[Dim(idx)] != concatOutShape[Dim(idx)]) {
            concatDims.push_back(Dim(idx));
        }
    }

    if (concatDims.empty() || concatDims.size() > 1) {
        return mlir::failure();
    }

    for (const auto& input : concatOp.getInputs()) {
        const auto inputShape = getShape(input);
        if (supportAdjacentDims) {
            SmallVector<Dim> adjustDims;
            if (concatDims[0].ind() - 1 > 0) {
                adjustDims.push_back(Dim(concatDims[0].ind() - 1));
            }
            if (concatDims[0].ind() + 1 < checked_cast<int32_t>(concatInShape.size())) {
                adjustDims.push_back(Dim(concatDims[0].ind() + 1));
            }

            for (auto dim : adjustDims) {
                if (inputShape[dim] == 1) {
                    concatDims[0] = dim;
                    break;
                }
            }

            if (inputShape[concatDims[0]] != 1) {
                return mlir::failure();
            }
        }
    }

    return concatDims;
}

}  // namespace IE
}  // namespace vpux
