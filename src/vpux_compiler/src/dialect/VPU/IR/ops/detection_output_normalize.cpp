//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::VPU::DetectionOutputNormalizeOp::verify() {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(getPriorBoxes().getType());
    const auto inputShape = inType.getShape();

    if (inputShape.size() != 4) {
        return errorAt(*this, "Input shape tensor must be 4D, got {0}", inputShape);
    }

    const auto nonNormalizedBoxWithPadding = 5;
    if (inputShape[Dims4D::Act::W] != nonNormalizedBoxWithPadding) {
        return errorAt(*this, "Input shape tensor must have width == 5, got {0}", inputShape);
    }

    return mlir::success();
}

//
// inferReturnTypes
//

mlir::LogicalResult VPU::DetectionOutputNormalizeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputNormalizeOpAdaptor normalize(operands, attrs, prop);
    if (mlir::failed(normalize.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(normalize.getPriorBoxes().getType());

    const auto normalizedBoxSize = 4;
    auto outputShape = inputType.getShape().toValues();
    outputShape[Dims4D::Act::W] = normalizedBoxSize;

    const auto outputType = inputType.changeShape(outputShape);
    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}
