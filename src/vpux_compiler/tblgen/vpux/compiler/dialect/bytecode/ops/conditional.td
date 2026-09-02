//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_CONDITIONAL
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_CONDITIONAL

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"

def Bytecode_SelectOp : Bytecode_InstructionOp<"select", [], "1.0.0", "intel_npu::vm::OpCode::SELECT"> {
    let summary = "Conditionally select one of two values";

    let description = [{
        This operation conditionally assigns the value of `trueVal` or `falseVal` into `dst`, based
        on the value of `cond`. The condition is considered true if `cond` is non-zero.

        Example:
        ```mlir
        bytecode.select %dst, %cond, %trueVal, %falseVal
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        Bytecode_RegisterType:$cond,
        Bytecode_RegisterType:$trueVal,
        Bytecode_RegisterType:$falseVal
    );

    let assemblyFormat = [{
        $dst `,` $cond `,` $trueVal `,` $falseVal attr-dict
    }];

    let extraClassDeclaration = [{
        bool isDstOperand(::mlir::OpOperand& operand) {
            return operand.get() == getDst();
        }
    }];
}

#endif
