//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// fold
//

mlir::OpFoldResult VPU::DistributedCastOp::fold(FoldAdaptor) {
    return getInput().getType() == getOutput().getType() ? getInput() : mlir::TypedValue<mlir::TensorType>{nullptr};
}

//
// verify
//

mlir::LogicalResult vpux::VPU::DistributedCastOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };

    const auto inDistributedTypeInterface = mlir::cast<vpux::VPU::DistributedTypeInterface>(getInput().getType());
    const auto outDistributedTypeInterface = mlir::cast<vpux::VPU::DistributedTypeInterface>(getOutput().getType());

    auto inDistributedTypes = inDistributedTypeInterface.getDistributedTypes();
    auto outDistributedTypes = outDistributedTypeInterface.getDistributedTypes();
    if (inDistributedTypes.size() == 0 || inDistributedTypes.size() != outDistributedTypes.size()) {
        return mlir::failure();
    }
    auto isCompatible = [&logCb](mlir::Type type1, mlir::Type type2) -> bool {
        return isDistributedCastCompatible(mlir::cast<vpux::VPU::DistributedTensorType>(type1),
                                           mlir::cast<vpux::VPU::DistributedTensorType>(type2), logCb)
                .succeeded();
    };

    return std::equal(inDistributedTypes.begin(), inDistributedTypes.end(), outDistributedTypes.begin(), isCompatible)
                   ? mlir::success()
                   : mlir::failure();
}
