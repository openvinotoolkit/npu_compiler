//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// ViewLikeOpInterface
//

mlir::Value VPUIP::DistributedCastOp::getViewSource() {
    return getInput();
}

//
// fold
//

mlir::OpFoldResult VPUIP::DistributedCastOp::fold(FoldAdaptor) {
    return getInput().getType() == getOutput().getType() ? getInput() : mlir::TypedValue<mlir::MemRefType>{nullptr};
}

//
// verify
//

mlir::LogicalResult vpux::VPUIP::DistributedCastOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };

    if (auto sparseBufferInput = mlir::dyn_cast<vpux::VPUIP::SparseBufferType>(getInput().getType())) {
        if (auto sparseBufferOutput = mlir::dyn_cast<vpux::VPUIP::SparseBufferType>(getOutput().getType())) {
            const auto inputData = mlir::cast<vpux::VPUIP::DistributedBufferType>(sparseBufferInput.getData());
            const auto outputData = mlir::cast<vpux::VPUIP::DistributedBufferType>(sparseBufferOutput.getData());
            return VPU::isDistributedCastCompatible(inputData, outputData, logCb);
        }

        logCb(formatv("Mismatch between types for input and output. "
                      "If input is SparseBufferType then output must be of same type."));
        return mlir::failure();
    }

    const auto outType = getOutput().getType();
    if (mlir::isa<VPUIP::SparseBufferType>(outType)) {
        logCb(formatv("Mismatch between types for input and output. "
                      "If output is SparseBufferType then input must be of same type."));
        return mlir::failure();
    }

    const auto inDistributedType = mlir::cast<vpux::VPUIP::DistributedBufferType>(getInput().getType());
    const auto outDistributedType = mlir::cast<vpux::VPUIP::DistributedBufferType>(getOutput().getType());

    return VPU::isDistributedCastCompatible(inDistributedType, outDistributedType, logCb);
}
