//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_COMPARISON
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_COMPARISON

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/BuiltinAttributes.td"

def Bytecode_CmpI64Op : Bytecode_ThreeOperandsPlusImmInstructionBase<"cmp.i64", [], "1.0.0", "intel_npu::vm::OpCode::CMP_I64"> {
    let summary = "Compare two 64-bit integers";

    let description = [{
        This operation compares two 64-bit integers (`lhs`, `rhs`) and stores 1 (true) or 0 (false) in `dst`.

        The `flag` attribute is a 16-bit immediate encoding the comparison predicate and sign:
        - bits 7-0 (CMP): 0x00=EQ, 0x01=NE, 0x02=GT, 0x03=GTE, 0x04=LT, 0x05=LTE
        - bit 8 (SIGN): 1=signed, 0=unsigned

        The sign bit determines whether the numbers should be treated as signed or unsigned.
        For example, a bit pattern like 0xFFFFFFFFFFFFFFFF would be -1 in signed comparison and 18446744073709551615 (2^64-1) in unsigned comparison.

        Examples:
        ```mlir
        bytecode.cmp.i64 %dst, %lhs, %rhs, 260  // signed less-than: flag = (1<<8)|0x04 = 0x104 = 260
        bytecode.cmp.i64 %dst, %lhs, %rhs, 0    // unsigned equal:   flag = 0x000 = 0
        ```
    }];
}


def Bytecode_CmpF64Op : Bytecode_ThreeOperandsPlusImmInstructionBase<"cmp.f64", [], "1.0.0", "intel_npu::vm::OpCode::CMP_F64"> {
    let summary = "Compare two 64-bit floating-point numbers";

    let description = [{
        This operation compares two 64-bit IEEE 754 floating-point numbers (`lhs`, `rhs`) and stores 1 (true) or 0 (false) in `dst`.

        The `flag` attribute is a 16-bit immediate encoding the comparison predicate:
        - bits 7-0 (CMP): 0x00=EQ, 0x01=NE, 0x02=GT, 0x03=GTE, 0x04=LT, 0x05=LTE

        For comparisons involving NaNs, the result is always false, except for NE which is true if either one or both operands are NaN.
        Positive zero and negative zero are considered equal.

        Examples:
        ```mlir
        bytecode.cmp.f64 %dst, %lhs, %rhs, 2  // greater-than: flag = 0x02 = 2
        ```
    }];
}

#endif
