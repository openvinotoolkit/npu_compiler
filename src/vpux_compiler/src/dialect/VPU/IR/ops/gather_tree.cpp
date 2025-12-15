//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GatherTreeOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GatherTreeOpAdaptor gatherTree(operands, attrs, prop);
    if (mlir::failed(gatherTree.verify(loc))) {
        return mlir::failure();
    }

    const auto stepIdsType = gatherTree.getStepIds().getType();
    inferredReturnTypes.push_back(stepIdsType);

    return mlir::success();
}
