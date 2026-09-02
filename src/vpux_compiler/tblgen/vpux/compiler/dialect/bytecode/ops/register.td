//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_REGISTER
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_REGISTER

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"

def Bytecode_GeneralRegisterOp : Bytecode_Op<"general_register", []> {
    let summary = "General register operation";

    let description = [{
        This operation represents a general register, which is identified by a register number.

        Example:
        ```mlir
        bytecode.general_register 0  // Represents the general register identified by number 0
        ```
    }];

    let arguments = (ins
        I16Attr:$regNum
    );

    let results = (outs
        Bytecode_RegisterType:$result
    );

    let assemblyFormat = [{
        $regNum attr-dict
    }];
}

def Bytecode_VirtualGeneralRegisterOp : Bytecode_Op<"virtual_general_register", []> {
    let summary = "Virtual general register operation";

    let description = [{
        This operation represents a virtual general register.

        Example:
        ```mlir
        bytecode.virtual_general_register  // Represents one virtual general register of a function's invocation
        ```

        In the call frame of a function, the register space of general and parameter registers are shared,
        and the general registers are located at the beginning of the register space (e.g. a function with three general registers
        and two parameter registers would have the general registers assigned numbers 0, 1, and 2).
        The distinction between the general and parameter registers is important for the calling convention,
        as parameter registers are used to pass arguments to functions.
        The register allocator will replace the virtual general registers with the actual general registers.
    }];

    let results = (outs
        Bytecode_RegisterType:$result
    );

    let builders = [
        OpBuilder<(ins)>,
    ];

    let assemblyFormat = [{
        attr-dict
    }];
}

def Bytecode_VirtualParameterRegisterOp : Bytecode_Op<"virtual_parameter_register", []> {
    let summary = "Virtual parameter register operation";

    let description = [{
        This operation represents a virtual parameter register, which is identified by a register index.
        Every parameter of a function is mapped to a virtual parameter register (e.g. first parameter is mapped to the first virtual parameter register).

        Example:
        ```mlir
        bytecode.virtual_parameter_register 0  // Represents the first virtual parameter register of a function's invocation
        ```

        In the call frame of a function, the register space of general and parameter registers are shared,
        and the parameter registers are located at the end of the register space (e.g. a function with three general registers
        and two parameter registers would have the parameter registers assigned numbers 3 and 4).
        The distinction between the general and parameter registers is important for the calling convention,
        as parameter registers are used to pass arguments to functions.
        The register allocator will replace the virtual parameter registers with the actual general registers.
    }];

    let arguments = (ins
        I16Attr:$paramIndex
    );

    let results = (outs
        Bytecode_RegisterType:$result
    );

    let builders = [
        OpBuilder<(ins
            "uint16_t":$paramIndex
        )>,
    ];

    let assemblyFormat = [{
        $paramIndex attr-dict
    }];
}

def Bytecode_SetOp : Bytecode_TwoOperandsInstructionBase<"set", [], "1.0.0", "intel_npu::vm::OpCode::SET"> {
    let summary = "Set operation";

    let description = [{
        This operation sets the value of a source register into a destination register.

        Examples:
        ```mlir
        bytecode.set %dst, %src  // Set the value from source register to destination register
        ```
    }];
}

def Bytecode_SetImmOp : Bytecode_InstructionOp<"set_imm", [], "1.0.0", "intel_npu::vm::OpCode::SET_IMM"> {
    let extraClassDeclaration = [{
        bool isDstOperand(::mlir::OpOperand& operand) {
            return operand.get() == getDst();
        }
    }];
    let summary = "Set immediate operation";

    let description = [{
        This operation sets a 64-bit immediate value into a destination register.

        Examples:
        ```mlir
        bytecode.set_imm %dst, 42  // Set the immediate value 42 into the destination register
        ```

        Note: Compared to other instructions, `set_imm` has the immediate value operand stored as a 64-bit value.
        This differs from the rest of the instructions, which have the operands encoded as 16-bit values.
        This is done to allow any value to be stored into the destination register.
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        I64Attr:$immValue
    );

    let assemblyFormat = [{
        $dst `,` $immValue attr-dict
    }];
}

def Bytecode_SetImmIdxOp : Bytecode_InstructionOp<"set_imm_idx",
        [DeclareOpInterfaceMethods<SymbolUserOpInterface>],
        "1.0.0", "intel_npu::vm::OpCode::SET_IMM"> {
    let summary = "Set symbol index operation";

    let description = [{
        This operation resolves a symbol reference to its positional index in the function section and writes the
        result into a destination register.

        Example:
        ```mlir
        bytecode.set_imm_idx %dst, @func_section::@my_function
        ```

        This is an intermediate compiler-only operation. During serialization it lowers to the regular `set.imm`
        instruction with the computed integer index.
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        SymbolRefAttr:$callee
    );

    let assemblyFormat = [{
        $dst `,` $callee attr-dict
    }];
}

def Bytecode_ImmRegisterOp : Bytecode_Op<"imm_register", []> {
    let summary = "Materialize a 64-bit constant into a virtual register";

    let description = [{
        Intermediate op: materialize a 64-bit immediate value as a virtual register.
        Lowered to (VirtualGeneralRegisterOp + SetImmOp) by DeduplicateAndLowerImmRegisterOps
        before register allocation. That pass deduplicates same-value source uses: entry-block
        materializations are reused across all successor blocks; within non-entry blocks,
        same-value source uses share a single VGR. A fresh (VGR + SetImmOp) is always created
        for each destination use, ensuring shared VGRs are never overwritten.

        Example:
        ```mlir
        %reg = bytecode.imm_register 1
        ```
    }];

    let arguments = (ins I64Attr:$value);
    let results   = (outs Bytecode_RegisterType:$result);

    let builders = [
        OpBuilder<(ins "mlir::IntegerAttr":$value), [{
            build($_builder, $_state,
                  vpux::bytecode::RegisterType::get($_builder.getContext()),
                  value);
        }]>
    ];

    let assemblyFormat = "$value attr-dict";
}

#endif
