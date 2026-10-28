//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::GatherElementsOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GatherElementsOpAdaptor gatherElements(operands, attrs, prop);
    if (mlir::failed(gatherElements.verify(loc))) {
        return mlir::failure();
    }

    const auto inIndicesType = mlir::cast<mlir::ShapedType>(gatherElements.getIndices().getType());
    const auto inInputType = mlir::cast<mlir::ShapedType>(gatherElements.getInput().getType());

    inferredReturnShapes.emplace_back(inIndicesType.getShape(), inInputType.getElementType());
    return mlir::success();
}

mlir::OpFoldResult vpux::IE::GatherElementsOp::fold(FoldAdaptor adaptor) {
    auto inputAttr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getInput());
    auto indicesContentAttr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getIndices());
    if (inputAttr == nullptr || indicesContentAttr == nullptr) {
        return nullptr;
    }

    const auto inputType = mlir::cast<mlir::ShapedType>(getInput().getType());
    const auto rank = inputType.getRank();

    int64_t axis = getAxis();
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        return nullptr;
    }

    const auto indicesContent = indicesContentAttr.fold();
    const auto indicesType = mlir::cast<mlir::ShapedType>(getIndices().getType());
    auto indicesAttr = mlir::DenseElementsAttr::getFromRawBuffer(indicesType, indicesContent.getRawStorageBuf());
    if (indicesAttr == nullptr) {
        return nullptr;
    }

    // GatherElementsAttr requires signed i64 indices; cast if the stored type is narrower (e.g. i32).
    if (!indicesType.getElementType().isInteger(64)) {
        const auto i64Type = mlir::IntegerType::get(getContext(), 64, mlir::IntegerType::Signed);
        indicesAttr =
                mlir::cast<mlir::DenseIntElementsAttr>(indicesAttr).mapValues(i64Type, [](const llvm::APInt& val) {
                    return val.sext(64);
                });
    }

    auto axisAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 64), axis);
    return inputAttr.transform().gatherElements(axisAttr, indicesAttr).get();
}
