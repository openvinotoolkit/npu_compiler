//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;
mlir::LogicalResult vpux::IE::MVNOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::MVNOpAdaptor mvn(operands, attrs);
    if (mlir::failed(mvn.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mvn.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();
    if (inShape.size() != 4 && inShape.size() != 5) {
        return errorAt(loc, "First input tensor should have 4 or 5 dimensions");
    }

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}
