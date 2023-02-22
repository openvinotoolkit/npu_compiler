//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::GRUSequenceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::GRUSequenceOpAdaptor gru(operands, attrs);
    if (mlir::failed(gru.verify(loc))) {
        return mlir::failure();
    }

    const auto outputStateType = gru.initial_hidden_state().getType().cast<mlir::ShapedType>();
    const auto outputStateShape = outputStateType.getShape();
    const auto seqLength = gru.seq_length();
    SmallVector<int64_t> middleStateShape = {outputStateShape[0], outputStateShape[1], seqLength, outputStateShape[2]};

    inferredReturnShapes.emplace_back(middleStateShape, outputStateType.getElementType());
    inferredReturnShapes.emplace_back(outputStateShape, outputStateType.getElementType());

    return mlir::success();
}
