//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::CTCGreedyDecoderSeqLenOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::CTCGreedyDecoderSeqLenOpAdaptor ctc(operands, attrs);
    if (mlir::failed(ctc.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = ctc.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    if (inShape.size() != 3) {
        return errorAt(loc, "First input tensor should have 3 dimensions");
    }

    const auto outElemType = ctc.sequenceLength().getType().cast<mlir::ShapedType>().getElementType();

    SmallVector<int64_t> outputShape{inShape[0], inShape[1]};
    SmallVector<int64_t> outputLengthShape{inShape[0]};

    inferredReturnShapes.emplace_back(outputShape, outElemType);
    inferredReturnShapes.emplace_back(outputLengthShape, outElemType);

    return mlir::success();
}
