//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::LogicalNotOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LogicalNotOpAdaptor logicalNot(operands, attrs, prop);
    if (mlir::failed(logicalNot.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<mlir::ShapedType>(logicalNot.getInput1().getType());

    inferredReturnShapes.emplace_back(in1Type.getShape(), in1Type.getElementType());

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::LogicalNotOp::fold(FoldAdaptor adaptor) {
    const auto outType = mlir::cast<mlir::RankedTensorType>(getOutput().getType());
    if (!outType.hasStaticShape()) {
        return nullptr;
    }

    const auto inputAttr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getInput1());
    if (inputAttr == nullptr) {
        return nullptr;
    }

    return inputAttr.transform().logicalNot().get();
}
