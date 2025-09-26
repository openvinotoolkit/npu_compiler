//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
using namespace vpux;

mlir::LogicalResult vpux::VPU::GroupNormalizationOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GroupNormalizationOpAdaptor groupNorm(operands, attrs, prop);
    if (mlir::failed(groupNorm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(groupNorm.getInput().getType());
    const auto outType = inType.changeShape(inType.getShape());
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
