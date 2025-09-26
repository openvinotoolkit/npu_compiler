//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::AddOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::AddOpAdaptor add(operands, attrs, prop);
    if (mlir::failed(add.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<mlir::RankedTensorType>(add.getInput1().getType());
    const auto in2Type = mlir::cast<mlir::RankedTensorType>(add.getInput2().getType());

    auto in1Shape = SmallVector<int64_t>(in1Type.getShape());
    auto in2Shape = SmallVector<int64_t>(in2Type.getShape());
    if (mlir::failed(IE::unpadInputShape(in1Shape, add.getInputPaddingAttr(), loc))) {
        return mlir::failure();
    }
    if (mlir::failed(IE::unpadInputShape(in2Shape, add.getInputPaddingAttr(), loc))) {
        return mlir::failure();
    }

    auto outShapeRes = IE::broadcastEltwiseShape(in1Shape, in2Shape, add.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        auto outShape = outShapeRes.value();
        if (mlir::failed(IE::padOutputShape(outShape, add.getOutputPaddingAttr(), loc))) {
            return mlir::failure();
        }
        const auto outBounds =
                inferOutputBounds(add.getInput1(), add.getInput2(), ShapeRef(outShape), add.getAutoBroadcast());

        const auto outDesc =
                vpux::getTensorAttr(ctx, inferOrder(in1Type, in2Type), /*memSpace=*/nullptr, BoundsRef(outBounds));
        inferredReturnShapes.emplace_back(outShape, in1Type.getElementType(), outDesc);
    }

    return outShapeRes;
}

mlir::OpFoldResult vpux::IE::AddOp::fold(FoldAdaptor adaptor) {
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

mlir::LogicalResult vpux::IE::AddOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                       mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();

    auto outShape = reifyEltwiseTensors(builder, getInput1(), getInput2(), getAutoBroadcast(), loc);

    if (mlir::failed(outShape)) {
        return outShape;
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}

//
// ShaveCodeGenSupportedOpInterface
//

bool vpux::IE::AddOp::shouldJITCompile() {
    return true;
}
