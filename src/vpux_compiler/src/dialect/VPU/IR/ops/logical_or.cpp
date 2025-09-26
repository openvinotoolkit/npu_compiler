//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LogicalOrOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LogicalOrOpAdaptor logicalor(operands, attrs, prop);
    if (mlir::failed(logicalor.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<vpux::NDTypeInterface>(logicalor.getInput1().getType());
    const auto in2Type = mlir::cast<vpux::NDTypeInterface>(logicalor.getInput2().getType());

    const auto outShapeRes = IE::broadcastEltwiseShape(in1Type.getShape().raw(), in2Type.getShape().raw(),
                                                       logicalor.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        const auto outType = in1Type.changeShape(ShapeRef(outShapeRes.value()));
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}
