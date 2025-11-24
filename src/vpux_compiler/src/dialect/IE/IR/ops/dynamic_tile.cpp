//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::DynamicTileOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DynamicTileOpAdaptor tile(operands, attrs, prop);
    if (mlir::failed(tile.verify(loc))) {
        return mlir::failure();
    }

    if (tile.getRepeats() != nullptr && tile.getRepeatsValues().has_value()) {
        return errorAt(loc, "Ambiguous repeats representation");
    }

    const auto outShape = parseIntArrayAttr<int64_t>(tile.getOutputShape());

    bool hasDynamic = any_of(outShape, [](int64_t dim) {
        return dim == mlir::ShapedType::kDynamic;
    });
    const auto outBounds = hasDynamic ? parseIntArrayAttr<int64_t>(tile.getOutputBoundsAttr()) : SmallVector<int64_t>{};

    const auto inType = mlir::cast<mlir::RankedTensorType>(tile.getInput().getType());

    const auto outDesc = vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape.size()), vpux::getMemorySpace(inType),
                                             BoundsRef(outBounds));
    inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);

    return mlir::success();
}
