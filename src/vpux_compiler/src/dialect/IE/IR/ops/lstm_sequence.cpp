//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::LSTMSequenceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LSTMSequenceOpAdaptor lstm(operands, attrs, prop);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto inDataType = lstm.getInputData().getType();
    const auto inDataShape = mlir::cast<vpux::NDTypeInterface>(inDataType).getShape();

    const auto initialHiddenStateType = mlir::cast<vpux::NDTypeInterface>(lstm.getInitialHiddenState().getType());
    const auto initialHiddenStateShape = initialHiddenStateType.getShape();
    const auto elementType = initialHiddenStateType.getElementType();

    const auto batchSize = initialHiddenStateShape[Dim(0)];
    const auto numDirections = initialHiddenStateShape[Dim(1)];
    const auto hiddenSize = initialHiddenStateShape.back();

    const auto lengthIndex = Dim(inDataShape.size() - 2);
    const auto sequenceLength = inDataShape[lengthIndex];

    const SmallVector<int64_t> outputHiddenValuesShape{batchSize, numDirections, sequenceLength, hiddenSize};

    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(inDataType)) {
        auto outHVBounds = SmallVector<int64_t>(outputHiddenValuesShape.size());
        const auto inBounds = boundedType.getBounds();

        for (size_t i = 0; i < outputHiddenValuesShape.size(); i++) {
            if (outputHiddenValuesShape[i] == mlir::ShapedType::kDynamic) {
                outHVBounds[i] = inBounds[lengthIndex];
            } else {
                outHVBounds[i] = outputHiddenValuesShape[i];
            }
        }
        auto outDesc = vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outputHiddenValuesShape.size()), nullptr,
                                           BoundsRef(outHVBounds));
        inferredReturnShapes.emplace_back(outputHiddenValuesShape, elementType, outDesc);  // outputHiddenValues

    } else {
        inferredReturnShapes.emplace_back(outputHiddenValuesShape, elementType);  // outputHiddenValues
    }

    inferredReturnShapes.emplace_back(initialHiddenStateShape, elementType);  // outputHiddenState
    inferredReturnShapes.emplace_back(initialHiddenStateShape, elementType);  // outputCellState

    return mlir::success();
}

mlir::LogicalResult vpux::IE::LSTMSequenceOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    SmallVector<mlir::OpFoldResult> shapes;
    const auto loc = getLoc();
    const auto inputShapedType = mlir::cast<mlir::ShapedType>(getInputData().getType());
    const auto initialHiddenStateType = mlir::cast<mlir::ShapedType>(getInitialHiddenState().getType());
    const auto outputHiddenValuesType = mlir::cast<mlir::ShapedType>(getOutputHiddenValues().getType());

    const auto seqLengthIdx = inputShapedType.getRank() - 2;
    size_t initialHiddenStateIdx = 0;
    for (const auto dimIdx : irange(outputHiddenValuesType.getRank())) {
        if (outputHiddenValuesType.isDynamicDim(dimIdx)) {
            mlir::OpFoldResult dimOp = builder.createOrFold<mlir::tensor::DimOp>(loc, getInputData(), seqLengthIdx);
            shapes.push_back(mlir::getValueOrCreateConstantIndexOp(builder, loc, dimOp));
        } else {
            shapes.push_back(builder.getIndexAttr(initialHiddenStateType.getDimSize(initialHiddenStateIdx++)));
        }
    }
    reifiedReturnShapes.emplace_back(std::move(shapes));

    SmallVector<mlir::OpFoldResult> outputHiddenStateShapes;
    SmallVector<mlir::OpFoldResult> outputCellStateShapes;
    for (const auto dimIdx : irange(initialHiddenStateType.getRank())) {
        if (initialHiddenStateType.isDynamicDim(dimIdx)) {
            // Dynamic dimensions are not supported for hidden states
            return mlir::failure();
        } else {
            outputHiddenStateShapes.push_back(builder.getIndexAttr(initialHiddenStateType.getDimSize(dimIdx)));
            outputCellStateShapes.push_back(builder.getIndexAttr(initialHiddenStateType.getDimSize(dimIdx)));
        }
    }
    reifiedReturnShapes.emplace_back(std::move(outputHiddenStateShapes));
    reifiedReturnShapes.emplace_back(std::move(outputCellStateShapes));
    return mlir::success();
}
