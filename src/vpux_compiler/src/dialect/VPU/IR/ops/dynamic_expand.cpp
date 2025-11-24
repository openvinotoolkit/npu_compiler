//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/Support/LogicalResult.h>

mlir::LogicalResult vpux::VPU::DynamicExpandOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DynamicExpandOpAdaptor dynExpand(operands, attrs, prop);
    if (mlir::failed(dynExpand.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(dynExpand.getInput().getType());

    auto inShapeInfo = ShapeInfo::fromNDType(inType);

    // For upper bounded tensor interpretation the shape should be equal to the bounds
    // Tensor with dynamic_dims_mask has upper bounds set as its shape
    auto outShape = inShapeInfo.bounds.empty() ? inShapeInfo.shape : inShapeInfo.bounds;

    inferredReturnTypes.push_back(mlir::RankedTensorType::get(outShape, inType.getElementType()));

    return mlir::success();
}
