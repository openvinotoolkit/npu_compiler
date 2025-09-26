//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DynamicDataMaskOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DynamicDataMaskOpAdaptor dynamicDataMask(operands, attrs, prop);
    if (mlir::failed(dynamicDataMask.verify(loc))) {
        return mlir::failure();
    }

    inferredReturnTypes.emplace_back(dynamicDataMask.getOutputTensorType());

    return mlir::success();
}
