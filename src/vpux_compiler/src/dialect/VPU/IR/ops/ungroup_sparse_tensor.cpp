//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"

using namespace vpux;

mlir::Value VPU::UngroupSparseTensorOp::getViewSource(ptrdiff_t /*resultInd*/) {
    return getInput();
}

mlir::LogicalResult VPU::UngroupSparseTensorOp::inferReturnTypes(mlir::MLIRContext*, std::optional<mlir::Location>,
                                                                 mlir::ValueRange operands,
                                                                 mlir::DictionaryAttr /*attrs*/, mlir::OpaqueProperties,
                                                                 mlir::RegionRange /*ranges*/,
                                                                 SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    VPUX_THROW_UNLESS(mlir::isa<vpux::VPU::SparseTensorType>(operands[0].getType()),
                      "Operand of type {0} is not a sparse tensor", operands[0].getType());
    const auto sparseTensorType = mlir::cast<vpux::VPU::SparseTensorType>(operands[0].getType());

    inferredReturnTypes.push_back(sparseTensorType.getData());
    if (sparseTensorType.getSparsityMap() != nullptr) {
        inferredReturnTypes.push_back(sparseTensorType.getSparsityMap());
    }
    if (sparseTensorType.getStorageElementTable() != nullptr) {
        inferredReturnTypes.push_back(sparseTensorType.getStorageElementTable());
    }

    return mlir::success();
}
