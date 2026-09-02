//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::SelectOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SelectOpAdaptor select(operands, attrs, prop);
    if (mlir::failed(select.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<mlir::ShapedType>(select.getInput1().getType());
    const auto in2Type = mlir::cast<mlir::ShapedType>(select.getInput2().getType());
    const auto in3Type = mlir::cast<mlir::ShapedType>(select.getInput3().getType());

    const auto outShapeRes = IE::broadcastEltwiseShape({in1Type.getShape(), in2Type.getShape(), in3Type.getShape()},
                                                       select.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        inferredReturnShapes.emplace_back(outShapeRes.value(), in2Type.getElementType());
    }

    return outShapeRes;
}

mlir::LogicalResult vpux::IE::SelectOp::verify() {
    const auto thenType = mlir::cast<vpux::NDTypeInterface>(getInput2().getType()).getElementType();
    const auto elseType = mlir::cast<vpux::NDTypeInterface>(getInput3().getType()).getElementType();
    const auto outType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType()).getElementType();

    if (thenType != elseType || elseType != outType) {
        return emitOpError("requires uniform element types across input2, input3 and output, got ")
               << "then=" << thenType << ", else=" << elseType << ", out=" << outType;
    }

    return mlir::success();
}
