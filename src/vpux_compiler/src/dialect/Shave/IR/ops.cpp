//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/SymbolTable.h>

using namespace vpux;

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/Shave/ops/meta-ops.cpp.inc>

//
// ExternalKernelOp
//

//
// CustomOpIOTensorConfigOp
//

llvm::SmallVector<Shave::TensorRank> Shave::CustomOpIOTensorConfigOp::getSupportedRankValues() {
    llvm::SmallVector<Shave::TensorRank> result;
    const auto supportedRankAttr = getSupportedRankAttr();
    for (auto attr : supportedRankAttr.getValue()) {
        result.push_back(mlir::cast<Shave::TensorRankAttr>(attr).getValue());
    }
    return result;
}

llvm::SmallVector<Shave::TensorDataType> Shave::CustomOpIOTensorConfigOp::getSupportedDtypeValues() {
    llvm::SmallVector<Shave::TensorDataType> result;
    const auto supportedDtypeAttr = getSupportedDtypeAttr();
    for (auto attr : supportedDtypeAttr.getValue()) {
        result.push_back(mlir::cast<Shave::TensorDataTypeAttr>(attr).getValue());
    }
    return result;
}
