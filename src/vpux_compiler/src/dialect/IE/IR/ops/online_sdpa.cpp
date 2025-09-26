//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::OnlineSDPAOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::OnlineSDPAOpAdaptor onlineSdpa(operands, attrs, prop);
    if (mlir::failed(onlineSdpa.verify(loc))) {
        return mlir::failure();
    }

    const auto qShape = getShape(onlineSdpa.getQuery());
    const auto vShape = getShape(onlineSdpa.getValue());

    const auto vEmbedding = vShape.back();
    auto outShape = to_small_vector(qShape);
    outShape.back() = vEmbedding;

    auto qType = mlir::cast<mlir::RankedTensorType>(onlineSdpa.getQuery().getType());
    inferredReturnShapes.emplace_back(outShape, qType.getElementType());

    return mlir::success();
}
