//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::SquaredDifferenceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SquaredDifferenceOpAdaptor squaredDiff(operands, attrs, prop);
    if (mlir::failed(squaredDiff.verify(loc))) {
        return mlir::failure();
    }

    return inferEltwiseReturnTypes(inferredReturnTypes, loc, squaredDiff.getInput1(), squaredDiff.getInput2(),
                                   squaredDiff.getAutoBroadcast());
}
