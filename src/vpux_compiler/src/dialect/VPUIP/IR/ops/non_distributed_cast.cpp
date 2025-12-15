//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// ViewLikeOpInterface
//

mlir::Value VPUIP::NonDistributedCastOp::getViewSource() {
    return getInput();
}

//
// verify
//

mlir::LogicalResult vpux::VPUIP::NonDistributedCastOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };

    const auto inDistributedType = mlir::cast<vpux::VPUIP::DistributedBufferType>(getInput().getType());
    const auto mode = inDistributedType.getDistribution().getMode().getValue();
    if (!VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) &&
        mode != (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
        logCb(formatv("Unsupported mode for input"));
        return mlir::failure();
    }
    const auto outType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    if (inDistributedType.getShape() != outType.getShape()) {
        logCb(formatv("Mismatch between input and output shape"));
        return mlir::failure();
    }
    if (inDistributedType.getElementType() != outType.getElementType()) {
        logCb(formatv("Mismatch between input and output element type"));
        return mlir::failure();
    }
    if (inDistributedType.getMemoryKind() != outType.getMemoryKind()) {
        logCb(formatv("Mismatch between input and output memory kind"));
        return mlir::failure();
    }

    if (inDistributedType.getStrides() != outType.getStrides()) {
        logCb(formatv("Mismatch between input and output strides"));
        return mlir::failure();
    }

    if (inDistributedType.getDimsOrder() != outType.getDimsOrder()) {
        logCb(formatv("Mismatch between input and output dim order"));
        return mlir::failure();
    }
    return mlir::success();
}
