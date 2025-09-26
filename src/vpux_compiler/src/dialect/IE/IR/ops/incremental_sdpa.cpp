//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::IncrementalSDPAOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::IncrementalSDPAOpAdaptor incrementalSdpa(operands, attrs, prop);
    if (mlir::failed(incrementalSdpa.verify(loc))) {
        return mlir::failure();
    }

    auto intermediateInputsType = SmallVector<mlir::Type>{incrementalSdpa.getInputPartialOutput().getType(),
                                                          incrementalSdpa.getInputRunningMax().getType(),
                                                          incrementalSdpa.getInputRunningSum().getType()};

    for (auto intermediateInputType : intermediateInputsType) {
        const auto shapedType = mlir::cast<mlir::ShapedType>(intermediateInputType);
        inferredReturnShapes.emplace_back(shapedType.getShape(), shapedType.getElementType());
    }

    return mlir::success();
}
