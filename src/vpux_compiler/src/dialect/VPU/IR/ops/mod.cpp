//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ModOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                       mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                       mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                       mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ModOpAdaptor mod(operands, attrs, prop);
    if (mlir::failed(mod.verify(loc))) {
        return mlir::failure();
    }

    return inferEltwiseReturnTypes(inferredReturnTypes, loc, mod.getInput1(), mod.getInput2(), mod.getAutoBroadcast());
}
