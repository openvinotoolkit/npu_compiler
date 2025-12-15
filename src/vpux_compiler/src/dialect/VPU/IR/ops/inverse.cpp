//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#include <mlir/Support/LogicalResult.h>
#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"

mlir::LogicalResult vpux::VPU::InverseOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::InverseOpAdaptor inverse(operands, attrs, prop);
    if (mlir::failed(inverse.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = inverse.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}
