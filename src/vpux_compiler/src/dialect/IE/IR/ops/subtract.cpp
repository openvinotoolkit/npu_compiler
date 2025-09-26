//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::SubtractOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SubtractOpAdaptor subtract(operands, attrs, prop);
    if (mlir::failed(subtract.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<vpux::NDTypeInterface>(subtract.getInput1().getType());
    const auto in2Type = mlir::cast<vpux::NDTypeInterface>(subtract.getInput2().getType());

    auto outShapeInfo = inferEltwiseOutputShapeInfo(ShapeInfo::fromNDType(in1Type), ShapeInfo::fromNDType(in2Type),
                                                    subtract.getAutoBroadcast(), loc);

    const auto outDesc = vpux::getTensorAttr(ctx, inferOrder(in1Type, in2Type), /*memSpace=*/nullptr,
                                             BoundsRef(outShapeInfo.bounds));
    inferredReturnShapes.emplace_back(outShapeInfo.shape, in1Type.getElementType(), outDesc);

    return mlir::success();
}

mlir::LogicalResult vpux::IE::SubtractOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                            mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();

    auto outShape = reifyEltwiseTensors(builder, getInput1(), getInput2(), getAutoBroadcast(), loc);

    if (mlir::failed(outShape)) {
        return outShape;
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}

mlir::OpFoldResult vpux::IE::SubtractOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 2, "Wrong number of operands : {0}", operands.size());

    const bool shapeChanges = getShape(getInput1()) != getShape(getOutput());
    if (shapeChanges) {
        return nullptr;
    }

    const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[1]);
    if (attr == nullptr || !attr.isSplat()) {
        return nullptr;
    }

    const auto content = static_cast<Const::ContentAttr>(attr).fold();
    return isDoubleEqual(content.getSplatValue<double>(), 0.0f) ? getInput1() : nullptr;
}
