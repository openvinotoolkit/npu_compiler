//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::FakeConvertOp::verify() {
    const auto dstType = getDstType();

    if (!mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(dstType)) {
        return errorAt(*this, "Unsupported dstType {0}", dstType);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::FakeConvertOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               std::optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::OpaqueProperties prop, mlir::RegionRange,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::FakeConvertOpAdaptor fc(operands, attrs, prop);
    if (mlir::failed(fc.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(fc.getInput().getType());
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}
