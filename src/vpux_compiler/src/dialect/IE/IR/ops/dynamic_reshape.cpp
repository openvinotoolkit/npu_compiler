//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/core/IR/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LogicalResult.h>

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::DynamicReshapeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DynamicReshapeOpAdaptor reshape(operands, attrs, prop);
    if (mlir::failed(reshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = parseIntArrayAttr<int64_t>(reshape.getOutputShape());
    const auto outBounds = reshape.getOutputBoundsAttr();
    const auto inType = reshape.getInput().getType().cast<mlir::RankedTensorType>();

    const auto outDesc =
            vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape.size()), vpux::getMemorySpace(inType), outBounds);

    inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);
    return mlir::success();
}

mlir::LogicalResult vpux::IE::DynamicReshapeOp::verify() {
    if (!IE::hasDynamicTensors(getOperation())) {
        return errorAt(getLoc(), "Operation must have dynamic tensors");
    }

    return mlir::success();
}
