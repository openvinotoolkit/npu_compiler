//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_CONTROL_FLOW
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_CONTROL_FLOW

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/BuiltinAttributes.td"

def Bytecode_RetOp : Bytecode_ZeroOperandsInstructionBase<"ret", [Terminator], "1.0.0", "intel_npu::vm::OpCode::RET"> {
    let summary = "Return operation";

    let opDescription = [{
        This operation represents a return operation in the bytecode format, and it is used to terminate the execution of a function.
        It does not return any value.

        Example:
        ```mlir
        bytecode.ret
        ```
    }];
}

def Bytecode_RetVOp : Bytecode_VariadicOperandsInstructionOp<"retv", [Terminator], "1.0.0", "intel_npu::vm::OpCode::RETV"> {
    let summary = "Return operation with values";

    let description = [{
        This operation represents a return operation in the bytecode format, and it is used to terminate the execution of a function.
        It returns values from the function.

        Example:
        ```mlir
        bytecode.retv %src1, %src2  // Returns the values in src1 and src2 from the function
        ```
    }];

    let arguments = (ins
        Variadic<Bytecode_RegisterType>:$regs
    );

    let assemblyFormat = [{
        $regs attr-dict
    }];
}

def Bytecode_CallOp : Bytecode_VariadicOperandsInstructionOp<"call", [AttrSizedOperandSegments], "1.0.0", "intel_npu::vm::OpCode::CALL"> {
    let summary = "Call a function by index";

    let description = [{
        Calls the function whose index is held in `func_idx`. Creates a new call frame, copies
        the argument register values into the callee's parameter registers, and transfers control
        to the callee's first instruction. `results_dst` is reserved for return-value mapping;
        value return handling is introduced with return-value instructions (for example, `retv`/'ret').
        Control returns to the instruction following the `call`.

        Signature: call rs, N, rN..., M, rM...

        Example:
        ```mlir
        bytecode.call %func_idx, results(%result0, %result1 : !bytecode.Register, !bytecode.Register),
                                 args(%arg0, %arg1 : !bytecode.Register, !bytecode.Register)
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$func_idx,
        Variadic<Bytecode_RegisterType>:$results_dst,
        Variadic<Bytecode_RegisterType>:$args
    );

    let assemblyFormat = [{
        $func_idx `,` `results``(` ($results_dst^ `:` type($results_dst))? `)` `,`
                      `args``(` ($args^ `:` type($args))? `)`
        attr-dict
    }];
}

def Bytecode_AssertOp : Bytecode_InstructionOp<"assert", [], "1.0.0", "intel_npu::vm::OpCode::ASSERT"> {
    let summary = "Assert operation";

    let description = [{
        This operation represents an assert operation in the bytecode format, and it is used to terminate the execution of a function if a condition is not met.
        The message operand corresponds with the index of a string within the string section.

        Example:
        ```mlir
        bytecode.assert %condition, %message  // Asserts that the condition is true, and if not, terminates the function execution with the provided message
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$condition,
        SymbolNameAttr:$msg_sym
    );

    let assemblyFormat = [{
        $condition `,` $msg_sym attr-dict
    }];
}

def Bytecode_JmpOp : Bytecode_InstructionOp<"jmp", [Terminator], "1.0.0", "intel_npu::vm::OpCode::JMP"> {
    let summary = "Unconditional jump to a successor block";
    let hasVerifier = 1;
    let description = [{
        PC-relative unconditional jump to the target `dest` block. The PC-relative int64 byte
        offset is computed during serialization and embedded directly in the instruction as an operand.

        The `dest` block must be within the same region as the `jmp` instruction,
        to ensure it is emitted as a valid jump target during serialization.

        Example:
        ```mlir
        bytecode.jmp ^dest
        ```
    }];

    let successors = (successor AnySuccessor:$dest);

    let assemblyFormat = [{ $dest attr-dict }];
}

def Bytecode_JEOp : Bytecode_InstructionOp<"je", [Terminator], "1.0.0", "intel_npu::vm::OpCode::JE"> {
    let summary = "Jump to trueDest if two registers are equal, else fall to falseDest";
    let hasVerifier = 1;
    let description = [{
        PC-relative conditional jump. If x[`lhs`] == x[`rhs`], jumps to `trueDest`; otherwise falls
        through to `falseDest`. The PC-relative int64 byte offset to `trueDest` is computed during
        serialization and embedded directly in the instruction as an operand.
        `falseDest` must be the physically next block in the region so the not-taken path falls through.

        The `trueDest` block must be within the same region as the `je` instruction,
        to ensure it is emitted as a valid jump target during serialization.

        Example:
        ```mlir
        bytecode.je %lhs, %rhs, ^trueDest, ^falseDest
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$lhs,
        Bytecode_RegisterType:$rhs
    );

    let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

    let assemblyFormat = [{ $lhs `,` $rhs `,` $trueDest `,` $falseDest attr-dict }];
}

def Bytecode_JNEOp : Bytecode_InstructionOp<"jne", [Terminator], "1.0.0", "intel_npu::vm::OpCode::JNE"> {
    let summary = "Jump to trueDest if two registers are not equal, else fall to falseDest";
    let hasVerifier = 1;
    let description = [{
        PC-relative conditional jump. If x[`lhs`] != x[`rhs`], jumps to `trueDest`; otherwise falls
        through to `falseDest`. The PC-relative int64 byte offset to `trueDest` is computed during
        serialization and embedded directly in the instruction as an operand.
        `falseDest` must be the physically next block in the region so the not-taken path falls through.

        The `trueDest` block must be within the same region as the `jne` instruction,
        to ensure it is emitted as a valid jump target during serialization.

        Example:
        ```mlir
        bytecode.jne %lhs, %rhs, ^trueDest, ^falseDest
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$lhs,
        Bytecode_RegisterType:$rhs
    );

    let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

    let assemblyFormat = [{ $lhs `,` $rhs `,` $trueDest `,` $falseDest attr-dict }];
}

#endif
