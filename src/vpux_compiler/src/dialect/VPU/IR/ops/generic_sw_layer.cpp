//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <iterator>

using namespace vpux;

// E#152917 Analyze & settle on GenericSwLayerOp integration & vpux interface usage

//
// SymbolUserOpInterface
//

mlir::LogicalResult VPU::GenericSwLayerOp::verifySymbolUses(mlir::SymbolTableCollection&) {
    // E#152917 : Add proper implementation for interface method
    return mlir::success();
}

//
// CallOpInterface
//

mlir::CallInterfaceCallable VPU::GenericSwLayerOp::getCallableForCallee() {
    return getOperation()->getAttrOfType<mlir::SymbolRefAttr>("callee");
}

void VPU::GenericSwLayerOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
    setCalleeAttr(mlir::cast<mlir::SymbolRefAttr>(callee));
}

mlir::Operation::operand_range VPU::GenericSwLayerOp::getArgOperands() {
    return {operand_begin(), operand_end()};
}

mlir::MutableOperandRange VPU::GenericSwLayerOp::getArgOperandsMutable() {
    return mlir::MutableOperandRange(this->getOperation());
}
