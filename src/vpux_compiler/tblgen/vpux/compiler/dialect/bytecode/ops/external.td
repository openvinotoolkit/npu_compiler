//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_EXTERNAL
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_EXTERNAL

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"

def Bytecode_ExtAssertOp : Bytecode_Op<"ext.assert", []> {
    let summary = "External assert operation";

    let description = [{
        This operation represents an intermediary stage of an assert instruction, which stores the string message as an attribute.
        This message is external data, which will be moved to the string section when lowering to the final assert instruction operation.

        Example:
        ```mlir
        bytecode.ext.assert %condition, "Assert message"
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$condition,
        StrAttr:$message
    );

    let assemblyFormat = [{
        $condition `,` $message attr-dict
    }];
}

def Bytecode_ExtFuncOp : Bytecode_Op<"ext.func", [HasParent<"vpux::bytecode::FuncSectionOp">, IsolatedFromAbove, Symbol]> {
    let summary = "External function operation";

    let description = [{
        This operation represents an intermediary stage of a function, which stores the function type as an attribute.
        This type is external data, which will be decomposed into bytecode type attributes and moved to the type section
        when lowering to the final function operation.

        Example:
        ```mlir
        bytecode.ext.func @my_function (i64, i64) -> (i64) {
            // Bytecode instructions for the function body
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type
    );

    let regions = (region
        AnyRegion:$body
    );

    let assemblyFormat = [{
        $sym_name $function_type attr-dict-with-keyword
        $body
    }];
}

def Bytecode_ExtBufferViewOp : Bytecode_Op<"ext.buffer.view", [SameVariadicOperandSize]> {
    let summary = "External buffer.view operation";

    let description = [{
        Intermediary stage of a buffer.view instruction. Carries the new element type as an MLIR
        type attribute alongside already-materialized shape and stride registers. During
        ConvertIntermediateBytecodeOps, this op is rewritten to bytecode.buffer.view with a
        FlatSymbolRefAttr for the element type;

        Example:
        ```mlir
        bytecode.ext.buffer.view %dst, %src, f32 offset(%byteOff) shape(%sz0, %sz1) strides(%st0, %st1)
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        Bytecode_RegisterType:$src,
        Bytecode_RegisterType:$byte_offset,
        TypeAttr:$elem_type,
        Variadic<Bytecode_RegisterType>:$shape,
        Variadic<Bytecode_RegisterType>:$strides
    );

    let assemblyFormat = [{
        $dst `,` $src `,` $elem_type `offset` `(` $byte_offset `)` `shape` `(` $shape `)` `strides` `(` $strides `)` attr-dict
    }];

    let hasVerifier = 1;
}

def Bytecode_ExtBufferCreateOp : Bytecode_Op<"ext.buffer.create", []> {
    let summary = "External buffer.create operation";

    let description = [{
        This operation represents an intermediary stage of a buffer.create instruction, which carries an MLIR memref
        type as an attribute. When lowering to the final buffer.create instruction, the memref type is decomposed
        into element-type/shape/strides entries in the type section, and the operands are materialized as virtual
        registers loaded with the corresponding immediate values.

        The destination register is supplied as an operand to keep the bytecode form free of use-def chains: every
        instruction reads and writes registers passed in from the surrounding scope.

        `dynamic_sizes` supplies one register per dynamic dimension of `buffer_type` (in dimension order, matching
        `memref.alloc` semantics). Fully-static allocations carry no `dynamic_sizes`; the lowering then materializes
        every extent as an immediate.

        Example:
        ```mlir
        bytecode.ext.buffer.create %dst, memref<2x3xf32>
        bytecode.ext.buffer.create %dst, memref<?xi8> sizes(%n)
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        TypeAttrOf<Builtin_MemRef>:$buffer_type,
        Variadic<Bytecode_RegisterType>:$dynamic_sizes
    );

    let assemblyFormat = [{
        $dst `,` $buffer_type (`sizes` `(` $dynamic_sizes^ `)`)? attr-dict
    }];

    let hasVerifier = 1;
}


#endif
