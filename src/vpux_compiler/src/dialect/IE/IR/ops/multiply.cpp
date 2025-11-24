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

mlir::LogicalResult vpux::IE::MultiplyOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MultiplyOpAdaptor multiply(operands, attrs, prop);
    if (mlir::failed(multiply.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<mlir::RankedTensorType>(multiply.getInput1().getType());
    const auto in2Type = mlir::cast<mlir::RankedTensorType>(multiply.getInput2().getType());

    auto in1Shape = SmallVector<int64_t>(in1Type.getShape());
    auto in2Shape = SmallVector<int64_t>(in2Type.getShape());
    if (mlir::failed(IE::unpadInputShape(in1Shape, multiply.getInputPaddingAttr(), loc))) {
        return mlir::failure();
    }
    if (mlir::failed(IE::unpadInputShape(in2Shape, multiply.getInputPaddingAttr(), loc))) {
        return mlir::failure();
    }

    const auto outShapeRes = IE::broadcastEltwiseShape(in1Shape, in2Shape, multiply.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        auto outShape = outShapeRes.value();
        if (mlir::failed(IE::padOutputShape(outShape, multiply.getOutputPaddingAttr(), loc))) {
            return mlir::failure();
        }
        const auto outBounds = inferOutputBounds(multiply.getInput1(), multiply.getInput2(), ShapeRef(outShape),
                                                 multiply.getAutoBroadcast());
        const auto outOrder =
                in1Type.getRank() >= in2Type.getRank() ? vpux::getOrder(in1Type) : vpux::getOrder(in2Type);

        const auto tensorAttr = getTensorAttr(ctx, outOrder, getMemorySpace(in1Type), BoundsRef(outBounds));
        inferredReturnShapes.emplace_back(outShape, in1Type.getElementType(), tensorAttr);
    }

    return mlir::success();
}

mlir::OpFoldResult vpux::IE::MultiplyOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 2, "Wrong number of operands : {0}", operands.size());

    const auto lhsInputShape = getShape(getInput1());
    const auto rhsInputShape = getShape(getInput2());
    const auto outputShape = getShape(getOutput());

    const bool shapeChanges = (lhsInputShape != outputShape);
    if (shapeChanges) {
        return nullptr;
    }

    auto rhsAttr = mlir::dyn_cast_or_null<vpux::Const::ContentAttr>(operands[1]);
    if (rhsAttr && rhsAttr.isSplat()) {
        const auto folded = rhsAttr.fold();
        const double splatVal = folded.getSplatValue<double>();
        if (isDoubleEqual(splatVal, 1.0)) {
            return getInput1();
        }
        return nullptr;
    }

    auto lhsAttr = mlir::dyn_cast_or_null<vpux::Const::ContentAttr>(operands[0]);
    // Do not fold constants with different shapes
    const bool equalShapes = (lhsInputShape == rhsInputShape);
    if (!equalShapes) {
        return nullptr;
    }

    if (lhsAttr && rhsAttr) {
        return lhsAttr.transform().rescale(rhsAttr).get();
    }
    return nullptr;
}

mlir::LogicalResult vpux::IE::MultiplyOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                            mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();

    auto outShape = reifyEltwiseTensors(builder, getInput1(), getInput2(), getAutoBroadcast(), loc);

    if (mlir::failed(outShape)) {
        return outShape;
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}
