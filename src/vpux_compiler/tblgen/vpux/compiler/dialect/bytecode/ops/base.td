//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_BASE
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_BASE

include "vpux/compiler/dialect/bytecode/dialect.td"
include "vpux/compiler/dialect/bytecode/ops_interfaces.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/OpBase.td"

class Bytecode_Op<string mnemonic, list<Trait> traits = []> :
        Op<Bytecode_Dialect, mnemonic, traits> {
}

class Bytecode_SerializableOp<string mnemonic, list<Trait> traits = [], string minVersion = ""> :
        Bytecode_Op<mnemonic, [
                DeclareOpInterfaceMethods<Bytecode_VersionedOpInterface>,
                DeclareOpInterfaceMethods<Bytecode_SerializableOpInterface>
            ] # traits> {
    code serializableExtraClassDefinition = [{
        intel_npu::vm::Version $cppClass::getMinVersion() {
            return intel_npu::vm::Version(}] # !subst(".", ", ", minVersion) # [{);
        }
    }];
    let extraClassDefinition = serializableExtraClassDefinition;
}

class Bytecode_InstructionOp<string mnemonic, list<Trait> traits = [], string minVersion = "", string opcode = ""> :
        Bytecode_SerializableOp<mnemonic, [
                DeclareOpInterfaceMethods<Bytecode_InstructionOpInterface>
            ] # traits,
            minVersion> {
    let extraClassDefinition = [{
        intel_npu::vm::OpCode $cppClass::getOpcode() {
            return }] # opcode # [{;
        }
        size_t $cppClass::getBinarySize() {
            return intel_npu::vm::getStaticInstructionSize(getOpcode()).value();
        }
    }] # serializableExtraClassDefinition;
}

class Bytecode_ZeroOperandsInstructionBase<string mnemonic, list<Trait> traits = [], string minVersion, string opcode>
        : Bytecode_InstructionOp<mnemonic, traits, minVersion, opcode> {
    string opDescription = "";
    let description = opDescription;

    let assemblyFormat = [{
        attr-dict
    }];
}

class Bytecode_OneOperandInstructionBase<string mnemonic, list<Trait> traits = [], string minVersion, string opcode>
        : Bytecode_InstructionOp<mnemonic, traits, minVersion, opcode> {
    string opDescription = "";
    let description = opDescription # [{
        The operation supports one addressing mode:
        1. Register mode: `reg` is a register
    }];

    let arguments = (ins
        Bytecode_RegisterType:$reg
    );

    let assemblyFormat = [{
        $reg attr-dict
    }];
}


class Bytecode_TwoOperandsInstructionBase<string mnemonic, list<Trait> traits = [], string minVersion, string opcode>
        : Bytecode_InstructionOp<mnemonic, traits, minVersion, opcode> {
    string opDescription = "";
    let description = opDescription # [{
        The operation supports one addressing mode:
        1. Register mode: `dst` and `src` are registers
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        Bytecode_RegisterType:$src
    );

    let assemblyFormat = [{
        $dst `,` $src attr-dict
    }];
}

class Bytecode_ThreeOperandsInstructionBase<string mnemonic, list<Trait> traits = [], string minVersion, string opcode>
        : Bytecode_InstructionOp<mnemonic, traits, minVersion, opcode> {
    string opDescription = "";
    let description = opDescription # [{
        The operation supports one addressing mode:
        1. Register mode: `lhs` and `rhs` are registers
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        Bytecode_RegisterType:$lhs,
        Bytecode_RegisterType:$rhs
    );

    let assemblyFormat = [{
        $dst `,` $lhs `,` $rhs attr-dict
    }];
}

class Bytecode_OneOperandsPlusImmInstructionBase<string mnemonic, list<Trait> traits = [], string minVersion, string opcode>
        : Bytecode_InstructionOp<mnemonic, traits, minVersion, opcode> {
    string opDescription = "";
    let description = opDescription # [{
        The operation supports one addressing mode:
        1. Register mode: `src` is a register; `flag` is a 16-bit immediate value
    }];

    let arguments = (ins
        Bytecode_RegisterType:$src,
        I16Attr:$flag
    );

    let assemblyFormat = [{
        $src `,` $flag attr-dict
    }];
}

class Bytecode_TwoOperandsPlusImmInstructionBase<string mnemonic, list<Trait> traits = [], string minVersion, string opcode>
        : Bytecode_InstructionOp<mnemonic, traits, minVersion, opcode> {
    string opDescription = "";
    let description = opDescription # [{
        The operation supports one addressing mode:
        1. Register mode: `src` is a register; `flag` is a 16-bit immediate value
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        Bytecode_RegisterType:$src,
        I16Attr:$flag
    );

    let assemblyFormat = [{
        $dst `,` $src `,` $flag attr-dict
    }];
}

class Bytecode_ThreeOperandsPlusImmInstructionBase<string mnemonic, list<Trait> traits = [], string minVersion, string opcode>
        : Bytecode_InstructionOp<mnemonic, traits, minVersion, opcode> {
    string opDescription = "";
    let description = opDescription # [{
        The operation supports one addressing mode:
        1. Register mode: `lhs` and `rhs` are registers; `flag` is a 16-bit immediate value
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        Bytecode_RegisterType:$lhs,
        Bytecode_RegisterType:$rhs,
        I16Attr:$flag
    );

    let assemblyFormat = [{
        $dst `,` $lhs `,` $rhs `,` $flag attr-dict
    }];
}

class Bytecode_VariadicOperandsInstructionOp<string mnemonic, list<Trait> traits = [], string minVersion = "", string opcode = ""> :
        Bytecode_SerializableOp<mnemonic, [
                DeclareOpInterfaceMethods<Bytecode_InstructionOpInterface>
            ] # traits,
            minVersion> {
    let extraClassDefinition = [{
        intel_npu::vm::OpCode $cppClass::getOpcode() {
            return }] # opcode # [{;
        }
    }] # serializableExtraClassDefinition;
}

#endif
