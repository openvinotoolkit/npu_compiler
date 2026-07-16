//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_ARITHMETIC
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_ARITHMETIC

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/BuiltinAttributes.td"

def Bytecode_AddI64Op : Bytecode_ThreeOperandsInstructionBase<"add.i64", [], "1.0.0", "intel_npu::vm::OpCode::ADD_I64"> {
    let summary = "Add two 64-bit integers";

    let opDescription = [{
        This operation adds two 64-bit integers (`lhs`, `rhs`) and produces a 64-bit integer result (`dst`).

        The integers are expected to use two's complement representation, making this instruction appropriate for both signed and unsigned integer addition.
        In case of overflow, the result silently wraps around (e.g., adding 1 to INT64_MAX results in INT64_MIN).

        Examples:
        ```mlir
        bytecode.add.i64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_MulI64Op : Bytecode_ThreeOperandsInstructionBase<"mul.i64", [], "1.0.0", "intel_npu::vm::OpCode::MUL_I64"> {
    let summary = "Multiply two 64-bit integers";

    let opDescription = [{
        This operation multiplies two 64-bit integers (`lhs`, `rhs`) and produces a 64-bit integer result (`dst`).

        The integers are expected to use two's complement representation, making this instruction appropriate for both signed and unsigned integer multiplication.
        In case of overflow, the result silently wraps around (e.g., multiplying INT64_MAX by 2 results in -2).

        Examples:
        ```mlir
        bytecode.mul.i64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_MinI64Op : Bytecode_ThreeOperandsInstructionBase<"min.i64", [], "1.0.0", "intel_npu::vm::OpCode::MIN_I64"> {
    let summary = "Compute the signed minimum of two 64-bit integers";

    let opDescription = [{
        This operation computes the signed minimum of two 64-bit integers (`lhs`, `rhs`) and stores the result in `dst`.

        Both operands are interpreted as two's complement signed 64-bit integers.

        Examples:
        ```mlir
        bytecode.min.i64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_MaxI64Op : Bytecode_ThreeOperandsInstructionBase<"max.i64", [], "1.0.0", "intel_npu::vm::OpCode::MAX_I64"> {
    let summary = "Compute the signed maximum of two 64-bit integers";

    let opDescription = [{
        This operation computes the signed maximum of two 64-bit integers (`lhs`, `rhs`) and stores the result in `dst`.

        Both operands are interpreted as two's complement signed 64-bit integers.

        Examples:
        ```mlir
        bytecode.max.i64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_SubI64Op : Bytecode_ThreeOperandsInstructionBase<"sub.i64", [], "1.0.0", "intel_npu::vm::OpCode::SUB_I64"> {
    let summary = "Subtract two 64-bit integers";

    let opDescription = [{
        This operation subtracts two 64-bit integers (`lhs`, `rhs`) and produces a 64-bit integer result (`dst`).

        The integers are expected to use two's complement representation, making this instruction appropriate for both signed and unsigned integer subtraction.
        In case of overflow, the result silently wraps around (e.g., subtracting 1 from INT64_MIN results in INT64_MAX).

        Examples:
        ```mlir
        bytecode.sub.i64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_DivI64Op : Bytecode_ThreeOperandsInstructionBase<"div.i64", [], "1.0.0", "intel_npu::vm::OpCode::DIV_I64"> {
    let summary = "Divide two 64-bit integers";

    let opDescription = [{
        This operation divides two 64-bit integers (`lhs`, `rhs`) and produces a 64-bit integer result (`dst`).

        The integers are expected to use two's complement representation.

        In case of division overflow (INT64_MIN / -1), the result is defined to be INT64_MIN to avoid undefined behavior.
        When dividing by zero, the execution is halted.

        Examples:
        ```mlir
        bytecode.div.i64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_DivU64Op : Bytecode_ThreeOperandsInstructionBase<"div.u64", [], "1.0.0", "intel_npu::vm::OpCode::DIV_U64"> {
    let summary = "Divide two 64-bit unsigned integers";

    let opDescription = [{
        This operation divides two 64-bit unsigned integers (`lhs`, `rhs`) and produces a 64-bit unsigned integer result (`dst`).

        The integers are interpreted as unsigned values.

        When dividing by zero, the execution is halted.

        Examples:
        ```mlir
        bytecode.div.u64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_MinU64Op : Bytecode_ThreeOperandsInstructionBase<"min.u64", [], "1.0.0", "intel_npu::vm::OpCode::MIN_U64"> {
    let summary = "Compute the unsigned minimum of two 64-bit integers";

    let opDescription = [{
        This operation computes the unsigned minimum of two 64-bit integers (`lhs`, `rhs`) and stores the result in `dst`.

        Both operands are interpreted as unsigned 64-bit integers.

        Examples:
        ```mlir
        bytecode.min.u64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_AddU64Op : Bytecode_ThreeOperandsInstructionBase<"add.u64", [], "1.0.0", "intel_npu::vm::OpCode::ADD_U64"> {
    let summary = "Add two unsigned 64-bit integers";

    let opDescription = [{
        This operation adds two unsigned 64-bit integers (`lhs`, `rhs`) and produces an unsigned 64-bit integer result (`dst`).

        Overflow wraps around.

        Examples:
        ```mlir
        bytecode.add.u64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_MaxU64Op : Bytecode_ThreeOperandsInstructionBase<"max.u64", [], "1.0.0", "intel_npu::vm::OpCode::MAX_U64"> {
    let summary = "Compute the unsigned maximum of two 64-bit integers";

    let opDescription = [{
        This operation computes the unsigned maximum of two 64-bit integers (`lhs`, `rhs`) and stores the result in `dst`.

        Both operands are interpreted as unsigned 64-bit integers.

        Examples:
        ```mlir
        bytecode.max.u64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_MulU64Op : Bytecode_ThreeOperandsInstructionBase<"mul.u64", [], "1.0.0", "intel_npu::vm::OpCode::MUL_U64"> {
    let summary = "Multiply two unsigned 64-bit integers";

    let opDescription = [{
        This operation multiplies two unsigned 64-bit integers (`lhs`, `rhs`) and produces an unsigned 64-bit integer result (`dst`).

        Overflow wraps around.

        Examples:
        ```mlir
        bytecode.mul.u64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_RemU64Op : Bytecode_ThreeOperandsInstructionBase<"rem.u64", [], "1.0.0", "intel_npu::vm::OpCode::REM_U64"> {
    let summary = "Compute remainder of dividing two unsigned 64-bit integers";

    let opDescription = [{
        This operation computes the remainder of the division between two unsigned 64-bit integers (`lhs`, `rhs`) and produces an unsigned 64-bit integer result (`dst`).

        When dividing by zero, the execution is halted.

        Examples:
        ```mlir
        bytecode.rem.u64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_SubU64Op : Bytecode_ThreeOperandsInstructionBase<"sub.u64", [], "1.0.0", "intel_npu::vm::OpCode::SUB_U64"> {
    let summary = "Subtract two unsigned 64-bit integers";

    let opDescription = [{
        This operation subtracts two unsigned 64-bit integers (`lhs`, `rhs`) and produces an unsigned 64-bit integer result (`dst`).

        Underflow wraps around.

        Examples:
        ```mlir
        bytecode.sub.u64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_RemI64Op : Bytecode_ThreeOperandsInstructionBase<"rem.i64", [], "1.0.0", "intel_npu::vm::OpCode::REM_I64"> {
    let summary = "Compute remainder of dividing two 64-bit integers";

    let opDescription = [{
        This operation computes the remainder of the division between two 64-bit integers (`lhs`, `rhs`) and produces a 64-bit integer result (`dst`).

        The integers are expected to use two's complement representation.

        In case of division overflow (INT64_MIN / -1), the result is defined to be 0 to avoid undefined behavior.
        When dividing by zero, the execution is halted.

        Examples:
        ```mlir
        bytecode.rem.i64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_AbsI64Op : Bytecode_TwoOperandsInstructionBase<"abs.i64", [], "1.0.0", "intel_npu::vm::OpCode::ABS_I64"> {
    let summary = "Compute absolute value of a 64-bit integer";

    let opDescription = [{
        This operation computes the absolute value of a 64-bit integer (`src`) and stores the result in `dst`.

        The integer is expected to use two's complement representation.

        In case of INT64_MIN, which cannot be represented as a positive 64-bit integer, the result is defined to be INT64_MIN.

        Examples:
        ```mlir
        bytecode.abs.i64 %dst, %src  // register mode
        ```
    }];
}

def Bytecode_AddF64Op : Bytecode_ThreeOperandsInstructionBase<"add.f64", [], "1.0.0", "intel_npu::vm::OpCode::ADD_F64"> {
    let summary = "Add two 64-bit floating-point numbers";

    let opDescription = [{
        This operation adds two 64-bit floating-point numbers (`lhs`, `rhs`) and produces a 64-bit floating-point result (`dst`).

        The floating-point values are expected to be stored in the IEEE 754 double-precision format.
        In case of overflow, the standard IEEE 754 rules apply.

        Examples:
        ```mlir
        bytecode.add.f64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_SubF64Op : Bytecode_ThreeOperandsInstructionBase<"sub.f64", [], "1.0.0", "intel_npu::vm::OpCode::SUB_F64"> {
    let summary = "Subtract two 64-bit floating-point numbers";

    let opDescription = [{
        This operation subtracts two 64-bit floating-point numbers (`lhs`, `rhs`) and produces a 64-bit floating-point result (`dst`).

        The floating-point values are expected to be stored in the IEEE 754 double-precision format.
        In case of overflow, the standard IEEE 754 rules apply.

        Examples:
        ```mlir
        bytecode.sub.f64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_MulF64Op : Bytecode_ThreeOperandsInstructionBase<"mul.f64", [], "1.0.0", "intel_npu::vm::OpCode::MUL_F64"> {
    let summary = "Multiply two 64-bit floating-point numbers";

    let opDescription = [{
        This operation multiplies two 64-bit floating-point numbers (`lhs`, `rhs`) and produces a 64-bit floating-point result (`dst`).

        The floating-point values are expected to be stored in the IEEE 754 double-precision format.
        In case of overflow, the standard IEEE 754 rules apply.

        Examples:
        ```mlir
        bytecode.mul.f64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_DivF64Op : Bytecode_ThreeOperandsInstructionBase<"div.f64", [], "1.0.0", "intel_npu::vm::OpCode::DIV_F64"> {
    let summary = "Divide two 64-bit floating-point numbers";

    let opDescription = [{
        This operation divides two 64-bit floating-point numbers (`lhs`, `rhs`) and produces a 64-bit floating-point result (`dst`).

        The floating-point values are expected to be stored in the IEEE 754 double-precision format.
        In case of overflow, the standard IEEE 754 rules apply.

        Examples:
        ```mlir
        bytecode.div.f64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_RemF64Op : Bytecode_ThreeOperandsInstructionBase<"rem.f64", [], "1.0.0", "intel_npu::vm::OpCode::REM_F64"> {
    let summary = "Remainder of division of two 64-bit floating-point numbers";

    let opDescription = [{
        This operation computes the remainder of the division of two 64-bit floating-point numbers (`lhs`, `rhs`) and produces a 64-bit floating-point result (`dst`).

        The floating-point values are expected to be stored in the IEEE 754 double-precision format.
        In case of overflow, the standard IEEE 754 rules apply.

        Examples:
        ```mlir
        bytecode.rem.f64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_MaxF64Op : Bytecode_ThreeOperandsInstructionBase<"max.f64", [], "1.0.0", "intel_npu::vm::OpCode::MAX_F64"> {
    let summary = "Maximum of two 64-bit floating-point numbers";

    let opDescription = [{
        This operation computes the maximum value between two 64-bit floating-point numbers (`lhs`, `rhs`) and produces a 64-bit floating-point result (`dst`).

        The floating-point values are expected to be stored in the IEEE 754 double-precision format.
        In case one of the operands is NaN, the result is defined to be the other operand (i.e., max(NaN, x) = x).

        Examples:
        ```mlir
        bytecode.max.f64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_MinF64Op : Bytecode_ThreeOperandsInstructionBase<"min.f64", [], "1.0.0", "intel_npu::vm::OpCode::MIN_F64"> {
    let summary = "Minimum of two 64-bit floating-point numbers";

    let opDescription = [{
        This operation computes the minimum value between two 64-bit floating-point numbers (`lhs`, `rhs`) and produces a 64-bit floating-point result (`dst`).

        The floating-point values are expected to be stored in the IEEE 754 double-precision format.
        In case one of the operands is NaN, the result is defined to be the other operand (i.e., min(NaN, x) = x).

        Examples:
        ```mlir
        bytecode.min.f64 %dst, %lhs, %rhs  // register mode
        ```
    }];
}

def Bytecode_AbsF64Op : Bytecode_TwoOperandsInstructionBase<"abs.f64", [], "1.0.0", "intel_npu::vm::OpCode::ABS_F64"> {
    let summary = "Absolute value of a 64-bit floating-point number";

    let opDescription = [{
        This operation computes the absolute value of a 64-bit floating-point number (`src`) and stores the result in `dst`.

        Examples:
        ```mlir
        bytecode.abs.f64 %dst, %src  // register mode
        ```
    }];
}

def Bytecode_NegF64Op : Bytecode_TwoOperandsInstructionBase<"neg.f64", [], "1.0.0", "intel_npu::vm::OpCode::NEG_F64"> {
    let summary = "Negation of a 64-bit floating-point number";

    let opDescription = [{
        This operation negates a 64-bit floating-point number (`src`) and stores the result in `dst`.

        Examples:
        ```mlir
        bytecode.neg.f64 %dst, %src  // register mode
        ```
    }];
}

def Bytecode_CeilF64Op : Bytecode_TwoOperandsInstructionBase<"ceil.f64", [], "1.0.0", "intel_npu::vm::OpCode::CEIL_F64"> {
    let summary = "Ceiling of a 64-bit floating-point number";

    let opDescription = [{
        This operation rounds up a 64-bit floating-point number (`src`) to the nearest integer and stores the result in `dst`.

        Examples:
        ```mlir
        bytecode.ceil.f64 %dst, %src  // register mode
        ```
    }];
}

def Bytecode_FloorF64Op : Bytecode_TwoOperandsInstructionBase<"floor.f64", [], "1.0.0", "intel_npu::vm::OpCode::FLOOR_F64"> {
    let summary = "Floor of a 64-bit floating-point number";

    let opDescription = [{
        This operation rounds down a 64-bit floating-point number (`src`) to the nearest integer and stores the result in `dst`.

        Examples:
        ```mlir
        bytecode.floor.f64 %dst, %src  // register mode
        ```
    }];
}

def Bytecode_RoundF64Op : Bytecode_TwoOperandsPlusImmInstructionBase<"round.f64", [], "1.0.0", "intel_npu::vm::OpCode::ROUND_F64"> {
    let summary = "Round a 64-bit floating-point number to nearest integer";

    let opDescription = [{
        This operation rounds a 64-bit floating-point number (`src`) to the nearest integer according to the rounding mode specified by the `flag` immediate and stores the result in `dst`.

        Rounding modes: RNE=0x0, RNA=0x1, RDN=0x2, RUP=0x3, RTZ=0x4.

        In case the rounding mode is not recognized, the execution is halted.

        Examples:
        ```mlir
        bytecode.round.f64 %dst, %src, 3  // round up (RUP)
        ```
    }];
}

#endif
