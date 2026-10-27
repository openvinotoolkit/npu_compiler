//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::GroupQuantCorrectionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GroupQuantCorrectionOpAdaptor groupQuantCorrection(operands, attrs, prop);
    if (mlir::failed(groupQuantCorrection.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(groupQuantCorrection.getActOut().getType());
    const auto inShape = to_small_vector(inType.getShape());
    const auto outDesc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), /*memSpace=*/nullptr);
    inferredReturnShapes.emplace_back(inShape, inType.getElementType(), outDesc);

    return mlir::success();
}

mlir::LogicalResult vpux::IE::GroupQuantCorrectionOp::verify() {
    const auto actOutShape = mlir::cast<vpux::NDTypeInterface>(getActOut().getType()).getShape();
    if (actOutShape[Dims4D::Act::H] != 1 || actOutShape[Dims4D::Act::W] != 1) {
        return errorAt(*this, "act_out must have shape [G, C, 1, 1].");
    }

    const auto actInShape = mlir::cast<vpux::NDTypeInterface>(getActIn().getType()).getShape();
    if (actInShape[Dims4D::Act::N] != actOutShape[Dims4D::Act::N] || actInShape[Dims4D::Act::C] != 1 ||
        actInShape[Dims4D::Act::H] != 1) {
        return errorAt(*this, "act_in must have shape [G, 1, 1, GS] with G matching act_out.");
    }

    const auto scalesShape = mlir::cast<vpux::NDTypeInterface>(getScales().getType()).getShape();
    if (scalesShape != actOutShape) {
        return errorAt(*this, "scales must have shape [G, C, 1, 1] matching act_out.");
    }

    const auto zpShape = mlir::cast<vpux::NDTypeInterface>(getZp().getType()).getShape();
    if (zpShape != actOutShape) {
        return errorAt(*this, "zp must have shape [G, C, 1, 1] matching act_out.");
    }

    return mlir::success();
}
