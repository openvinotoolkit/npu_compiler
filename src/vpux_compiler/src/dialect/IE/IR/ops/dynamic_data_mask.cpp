//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult IE::DynamicDataMaskOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DynamicDataMaskOpAdaptor dynamicDataMask(operands, attrs, prop);
    if (mlir::failed(dynamicDataMask.verify(loc))) {
        return mlir::failure();
    }

    auto outTensorType = mlir::cast<NDTypeInterface>(dynamicDataMask.getOutputTensorType());
    const auto shapeInfo = ShapeInfo::fromNDType(outTensorType);

    ArrayRef<int64_t> outBounds = {};
    if (!shapeInfo.bounds.empty()) {
        outBounds = shapeInfo.bounds;
    }

    const auto outDesc =
            vpux::getTensorAttr(ctx, outTensorType.getDimsOrder(), /*memSpace=*/nullptr, BoundsRef(outBounds));
    inferredReturnShapes.emplace_back(shapeInfo.shape, outTensorType.getElementType(), outDesc);

    return mlir::success();
}
