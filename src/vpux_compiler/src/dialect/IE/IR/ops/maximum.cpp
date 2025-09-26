//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::MaximumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MaximumOpAdaptor maximum(operands, attrs, prop);
    if (mlir::failed(maximum.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<vpux::NDTypeInterface>(maximum.getInput1().getType());
    const auto in2Type = mlir::cast<vpux::NDTypeInterface>(maximum.getInput2().getType());

    auto outShapeInfo = inferEltwiseOutputShapeInfo(ShapeInfo::fromNDType(in1Type), ShapeInfo::fromNDType(in2Type),
                                                    maximum.getAutoBroadcast(), loc);

    const auto outDesc = vpux::getTensorAttr(ctx, inferOrder(in1Type, in2Type), /*memSpace=*/nullptr,
                                             BoundsRef(outShapeInfo.bounds));
    inferredReturnShapes.emplace_back(outShapeInfo.shape, in1Type.getElementType(), outDesc);

    return mlir::success();
}

mlir::LogicalResult vpux::IE::MaximumOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                           mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();

    auto outShape = reifyEltwiseTensors(builder, getInput1(), getInput2(), getAutoBroadcast(), loc);

    if (mlir::failed(outShape)) {
        return outShape;
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}
