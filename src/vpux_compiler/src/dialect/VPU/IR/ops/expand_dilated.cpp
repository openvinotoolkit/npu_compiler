//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dilated_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ExpandDilatedOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ExpandDilatedOpAdaptor expandDilated(operands, attrs, prop);
    if (mlir::failed(expandDilated.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::dyn_cast<vpux::NDTypeInterface>(expandDilated.getInput().getType());
    if (!inType) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(expandDilated.getDilations());
    const auto outType = getDilatedType(inType, ShapeRef(dilations));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
