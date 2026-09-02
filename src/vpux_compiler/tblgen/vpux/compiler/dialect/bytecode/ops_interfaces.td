//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def Bytecode_VersionedOpInterface : OpInterface<"VersionedOpInterface"> {
    let description = "Versioning interface for bytecode operations";

    let cppNamespace = "vpux::bytecode";

    let methods = [
        InterfaceMethod<
            "Get the minimum version of the operation (i.e. the version in which the operation was introduced)",
            "intel_npu::vm::Version", "getMinVersion"
        >,
    ];
}

def Bytecode_SerializableOpInterface : OpInterface<"SerializableOpInterface"> {
    let description = "Serialization interface for bytecode operations";

    let cppNamespace = "vpux::bytecode";

    let methods = [
        InterfaceMethod<
            "Serialize operation to bytecode format",
            "void", "serialize", (ins "vpux::bytecode::BytecodeWriter&":$writer)
        >,
        InterfaceMethod<
            "Get the binary size of the operation when serialized, in bytes",
            "size_t", "getBinarySize"
        >,
    ];
}

def Bytecode_InstructionOpInterface : OpInterface<"InstructionOpInterface"> {
    let description = "Interface for instruction operations";

    let cppNamespace = "vpux::bytecode";

    let methods = [
        InterfaceMethod<
            "Get the operation's opcode",
            "intel_npu::vm::OpCode", "getOpcode"
        >,
        InterfaceMethod<
            "Get the operation's addressing mode",
            "uint16_t", "getAddressingMode", (ins), [{}],
            [{
                return static_cast<uint16_t>(0x00);  // Default addressing mode (i.e. register mode)
            }]
        >,
        InterfaceMethod<
            "Return true if the given operand is a register that this instruction "
            "writes to. Ops with no write destination (branches, assert, "
            "buffer.store, cmd_list.close/exec) use the default which returns false.",
            "bool", "isDstOperand",
            (ins "::mlir::OpOperand&":$operand), [{}],
            [{ return false; }]
        >,
    ];
}

#endif
