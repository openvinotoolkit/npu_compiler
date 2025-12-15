//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::AsinOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::AsinOpAdaptor asin(operands, attrs, prop);
    if (mlir::failed(asin.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = asin.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}
