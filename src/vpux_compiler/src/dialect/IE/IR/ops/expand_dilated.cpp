//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dilated_utils.hpp"

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ExpandDilatedOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ExpandDilatedOpAdaptor expandDilated(operands, attrs, prop);
    if (mlir::failed(expandDilated.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::dyn_cast<vpux::NDTypeInterface>(expandDilated.getInput().getType());
    if (!inType) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(expandDilated.getDilations());
    const auto newType = mlir::cast<mlir::RankedTensorType>(getDilatedType(inType, ShapeRef(dilations)));
    inferredReturnShapes.emplace_back(newType.getShape(), newType.getElementType(), newType.getEncoding());

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ExpandDilatedOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    VPUX_THROW_WHEN(operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        const auto dilationsVal = parseIntArrayAttr<int64_t>(getDilations());
        return static_cast<Const::ContentAttr>(attr).transform().expandDilated(ShapeRef(dilationsVal)).get();
    }

    return nullptr;
}
