//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"

using namespace vpux;
using namespace VPU;

bool vpux::VPU::isDepthwiseOp(mlir::Operation* op) {
    return mlir::isa<VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>(op);
}

bool VPU::isNCEWithInt4Weights(mlir::Operation* op) {
    auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr) {
        return false;
    }

    auto weights = nceOp.getWeightsOperand();
    if (weights == nullptr) {
        return false;
    }

    auto weightsElemType = mlir::cast<vpux::NDTypeInterface>(weights.getType()).getElementType();
    if (const auto quantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(weightsElemType)) {
        return quantizedType.getStorageTypeIntegralWidth() == 4;
    }

    return false;
}

bool VPU::isNCEWithSEPActivation(mlir::Operation* op) {
    auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr) {
        return false;
    }
    auto sparseTensorActivation = nceOp->getOperand(0).getDefiningOp<VPU::GroupSparseTensorOp>();
    if (sparseTensorActivation == nullptr) {
        return false;
    }
    return sparseTensorActivation.getStorageElementTable() != nullptr;
}
