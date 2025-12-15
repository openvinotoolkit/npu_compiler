//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"

using namespace vpux;
using namespace mlir;

mlir::LogicalResult vpux::VPU::ConditionalCopyOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ConditionalCopyOpAdaptor conditionalCopyOp(operands, attrs, prop);
    if (mlir::failed(conditionalCopyOp.verify(loc))) {
        return mlir::failure();
    }
    auto inType = conditionalCopyOp.getInput1().getType();
    inferredReturnTypes.emplace_back(inType);
    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::ConditionalCopyOp::verify() {
    const auto input1Type = getInput1().getType();
    const auto input2Type = getInput2().getType();

    if (input1Type != input2Type) {
        return errorAt(*this, "Input1 should have same Type as Input2");
    }

    return mlir::success();
}
