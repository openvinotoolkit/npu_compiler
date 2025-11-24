//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::FakeConvertOp::verify() {
    const auto dstType = getDstType();
    if (!mlir::isa<mlir::Float8E4M3FNType>(dstType) && !mlir::isa<mlir::Float8E5M2Type>(dstType)) {
        return errorAt(*this, "Unsupported FakeConvert destination type {0}", dstType);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::FakeConvertOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::FakeConvertOpAdaptor cvt(operands, attrs, prop);
    if (mlir::failed(cvt.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<mlir::RankedTensorType>(cvt.getInput().getType());

    inferredReturnShapes.emplace_back(inputType.getShape(), inputType.getElementType());

    return mlir::success();
}
