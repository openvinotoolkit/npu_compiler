//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/Builders.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::ShapeOfOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ShapeOfOpAdaptor shapeOf(operands, attrs, prop);
    if (mlir::failed(shapeOf.verify(loc))) {
        return mlir::failure();
    }

    // Outputs:
    // 1D tensor that is equal to input tensor shape.
    // Number of elements is equal to input tensor rank.
    // Can be empty 1D tensor if input tensor is a scalar, that means 0-dimensional tensor.

    const auto inType = mlir::cast<mlir::ShapedType>(shapeOf.getInput().getType());
    const auto inRank = inType.getRank();
    const SmallVector<int64_t> outShape = {inRank};
    const auto outElemType = shapeOf.getDstElemType();
    inferredReturnShapes.emplace_back(ArrayRef(outShape), outElemType);

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ShapeOfOp::fold(FoldAdaptor) {
    auto shape = getShape(getInput());
    if (shape.isDynamic()) {
        return nullptr;
    }

    mlir::OpBuilder builder(getOperation());

    auto elemType = getDstElemType();
    auto constantShape = SmallVector<int64_t>{checked_cast<int64_t>(shape.size())};
    auto type = mlir::RankedTensorType::get(constantShape, elemType);

    if (elemType.isSignedInteger(32)) {
        auto values = SmallVector<int32_t>(shape.begin(), shape.end());
        return Const::createConst(builder, appendLoc(getLoc(), "shape_of"), type, ArrayRef(values));
    } else if (elemType.isSignedInteger(64)) {
        auto values = SmallVector<int64_t>(shape.begin(), shape.end());
        return Const::createConst(builder, appendLoc(getLoc(), "shape_of"), type, ArrayRef(values));
    }

    VPUX_THROW("Unsupported data type: {0}", elemType);
}
