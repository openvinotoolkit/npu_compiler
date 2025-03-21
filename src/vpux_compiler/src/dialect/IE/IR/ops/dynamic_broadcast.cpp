//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DynamicBroadcastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DynamicBroadcastOpAdaptor dynamicBroadcast(operands, attrs, prop);
    if (mlir::failed(dynamicBroadcast.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = parseIntArrayAttr<int64_t>(dynamicBroadcast.getOutputShape());
    const auto outBounds = dynamicBroadcast.getOutputBoundsAttr();

    auto inType = mlir::cast<mlir::RankedTensorType>(dynamicBroadcast.getInput().getType());

    TensorAttr outDesc = nullptr;
    if (outShape == parseIntArrayAttr<int64_t>(outBounds)) {
        outDesc = vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape.size()), vpux::getMemorySpace(inType));
    } else {
        outDesc = vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape.size()), vpux::getMemorySpace(inType),
                                      outBounds);
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);
    return mlir::success();
}
