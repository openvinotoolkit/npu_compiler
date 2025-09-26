//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUASM_ATTRS
#define VPUX_COMPILER_DIALECT_VPUASM_ATTRS

include "mlir/IR/OpBase.td"

/// A TypeAttr for memref types.
def VPUASM_BufferTypeAttr
    : TypeAttrBase<"::vpux::VPUASM::BufferType", "VPU ASM Buffer type attribute"> {
    let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

def VPUASM_IndexTypeAttr
    : TypeAttrBase<"::vpux::VPURegMapped::IndexType", "TypeAttr for Mapped Inference IndexType" > {
    let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

def KernelParamsProp: Property<"vpux::VPUASM::KernelParamsProperty">{
    let hashProperty = "hash_value($_storage)";
    let storageType = "vpux::VPUASM::KernelParamsProperty";
    let writeToMlirBytecode = writeMlirBytecodeWithConvertToAttribute;
    let readFromMlirBytecode = readMlirBytecodeUsingConvertFromAttribute;
}

#endif
