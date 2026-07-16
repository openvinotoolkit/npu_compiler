//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_BITWISE
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_BITWISE

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/BuiltinAttributes.td"

def Bytecode_And64Op : Bytecode_ThreeOperandsInstructionBase<"and.64", [], "1.0.0", "intel_npu::vm::OpCode::AND_64"> {
    let summary = "Compute the bitwise AND of two 64-bit values";

    let opDescription = [{
        This operation computes the bitwise AND of two 64-bit values (`lhs`, `rhs`) and stores the result in `dst`.

        Examples:
        ```mlir
        bytecode.and.64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_Not64Op : Bytecode_TwoOperandsInstructionBase<"not.64", [], "1.0.0", "intel_npu::vm::OpCode::NOT_64"> {
    let summary = "Compute the bitwise NOT of a 64-bit value";

    let opDescription = [{
        This operation computes the bitwise NOT of a 64-bit value (`src`) and stores the result in `dst`.

        Examples:
        ```mlir
        bytecode.not.64 %dst, %src  // register mode
        ```
    }];
}

def Bytecode_Or64Op : Bytecode_ThreeOperandsInstructionBase<"or.64", [], "1.0.0", "intel_npu::vm::OpCode::OR_64"> {
    let summary = "Compute the bitwise OR of two 64-bit values";

    let opDescription = [{
        This operation computes the bitwise OR of two 64-bit values (`lhs`, `rhs`) and stores the result in `dst`.

        Examples:
        ```mlir
        bytecode.or.64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_Xor64Op : Bytecode_ThreeOperandsInstructionBase<"xor.64", [], "1.0.0", "intel_npu::vm::OpCode::XOR_64"> {
    let summary = "Compute the bitwise XOR of two 64-bit values";

    let opDescription = [{
        This operation computes the bitwise XOR of two 64-bit values (`lhs`, `rhs`) and stores the result in `dst`.

        Examples:
        ```mlir
        bytecode.xor.64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_Sll64Op : Bytecode_ThreeOperandsInstructionBase<"sll.64", [], "1.0.0", "intel_npu::vm::OpCode::SLL_64"> {
    let summary = "Logical shift left of a 64-bit value";

    let opDescription = [{
        This operation shifts `lhs` left by the number of bits specified in `rhs` and stores the result in `dst`.

        Both operands are interpreted as 64-bit values. The shift count is masked to the range [0, 63].

        Examples:
        ```mlir
        bytecode.sll.64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_Srl64Op : Bytecode_ThreeOperandsInstructionBase<"srl.64", [], "1.0.0", "intel_npu::vm::OpCode::SRL_64"> {
    let summary = "Logical shift right of a 64-bit value";

    let opDescription = [{
        This operation shifts `lhs` right by the number of bits specified in `rhs` and stores the result in `dst`.

        Both operands are interpreted as 64-bit values. The shift count is masked to the range [0, 63].

        Examples:
        ```mlir
        bytecode.srl.64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_Sra64Op : Bytecode_ThreeOperandsInstructionBase<"sra.64", [], "1.0.0", "intel_npu::vm::OpCode::SRA_64"> {
    let summary = "Arithmetic shift right of a 64-bit value";

    let opDescription = [{
        This operation shifts `lhs` right by the number of bits specified in `rhs` and stores the result in `dst`.

        Both operands are interpreted as signed 64-bit values. The shift count is masked to the range [0, 63].
        The sign bit is preserved.

        Examples:
        ```mlir
        bytecode.sra.64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

#endif
