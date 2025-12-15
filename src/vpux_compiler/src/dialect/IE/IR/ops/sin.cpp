//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::SinOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SinOpAdaptor sine(operands, attrs, prop);
    if (mlir::failed(sine.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(sine.getInput().getType());
    const auto inShapeInfo = ShapeInfo::fromNDType(inType);

    const auto outDesc =
            vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace(), BoundsRef(inShapeInfo.bounds));
    inferredReturnShapes.emplace_back(inShapeInfo.shape, inType.getElementType(), outDesc);

    return mlir::success();
}
