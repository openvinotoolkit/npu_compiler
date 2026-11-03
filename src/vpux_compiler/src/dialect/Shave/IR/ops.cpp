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

Shave::CustomOpConfigOp Shave::ExternalKernelOp::getCustomOpConfig() {
    auto configAttr = getConfigAttr();
    if (!configAttr) {
        return nullptr;
    }

    auto moduleOp = (*this)->template getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
        return nullptr;
    }

    return mlir::dyn_cast_if_present<Shave::CustomOpConfigOp>(mlir::SymbolTable::lookupSymbolIn(moduleOp, configAttr));
}

llvm::SmallVector<Shave::ScratchLocation> Shave::ExternalKernelOp::getScratchLocationValues() {
    llvm::SmallVector<Shave::ScratchLocation> result;
    auto locAttr = getScratchLocationsAttr();
    if (!locAttr) {
        return result;
    }
    for (auto attr : locAttr.getValue()) {
        result.push_back(mlir::cast<Shave::ScratchLocationAttr>(attr).getValue());
    }
    return result;
}

//
// AuxiliaryBufferOpInterface
//

vpux::SmallVector<mlir::OpOperand*> Shave::ExternalKernelOp::getAuxiliaryBuffers() {
    SmallVector<mlir::OpOperand*> scratchBuffs{};

    for (auto& cmxScratchBuffer : getScratchBuffersMutable()) {
        scratchBuffs.emplace_back(&cmxScratchBuffer);
    }

    return scratchBuffs;
}

//
// SWOpInterface
//

bool Shave::ExternalKernelOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> /*buffers*/, Byte /*reservedMem*/) {
    return false;
}

bool Shave::ExternalKernelOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> /*buffers*/) {
    return false;
}

bool Shave::ExternalKernelOp::supportCycleCostCalculation() {
    return false;
}

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
