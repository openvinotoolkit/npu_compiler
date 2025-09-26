//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::LessOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LessOpAdaptor less(operands, attrs, prop);
    if (mlir::failed(less.verify(loc))) {
        return mlir::failure();
    }
    const auto in1Type = mlir::cast<vpux::NDTypeInterface>(less.getInput1().getType());
    const auto in2Type = mlir::cast<vpux::NDTypeInterface>(less.getInput2().getType());

    const auto outShapeInfo = inferEltwiseOutputShapeInfo(ShapeInfo::fromNDType(in1Type),
                                                          ShapeInfo::fromNDType(in2Type), less.getAutoBroadcast(), loc);

    const auto outDesc = vpux::getTensorAttr(ctx, inferOrder(in1Type, in2Type), /*memSpace=*/nullptr,
                                             BoundsRef(outShapeInfo.bounds));

    // Explicitly set the output type to boolean since input and output types are not the same
    inferredReturnShapes.emplace_back(outShapeInfo.shape, getBool8Type(ctx), outDesc);

    return mlir::success();
}

mlir::LogicalResult vpux::IE::LessOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                        mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();

    auto outShape = reifyEltwiseTensors(builder, getInput1(), getInput2(), getAutoBroadcast(), loc);

    if (mlir::failed(outShape)) {
        return outShape;
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}
