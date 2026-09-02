//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_BUFFER
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_BUFFER

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"

def Bytecode_BufferCreateOp : Bytecode_VariadicOperandsInstructionOp<"buffer.create",
        [SameVariadicOperandSize, DeclareOpInterfaceMethods<SymbolUserOpInterface>],
        "1.0.0", "intel_npu::vm::OpCode::BUFFER_CREATE"> {
    let summary = "Allocate a managed buffer";

    let description = [{
        Allocate a new buffer.

        The `elem_type` attribute is a symbol reference to a `bytecode.type` entry in the type
        section that describes the element type of the buffer. During serialization the symbol is
        resolved to its positional index in the type section, which is encoded directly in the
        instruction stream to avoid discrepancies caused by type-section reordering (e.g. by the
        `ConvertIntermediateBytecodeOps` pass). The operation stores the new buffer handle into
        the `dst` register. The `shape` and `strides` operand groups describe the buffer per
        dimension and must contain the same number of registers. The resulting buffer owns its
        memory and has read-write permissions.

        Example:
        ```mlir
        bytecode.buffer.create %dst, @type_section::@i64_type shape(%size0, %size1) strides(%stride0, %stride1)
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        SymbolRefAttr:$elem_type,
        Variadic<Bytecode_RegisterType>:$shape,
        Variadic<Bytecode_RegisterType>:$strides
    );

    let assemblyFormat = [{
        $dst `,` $elem_type `shape` `(` $shape `)` `strides` `(` $strides `)` attr-dict
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool isDstOperand(::mlir::OpOperand& operand) {
            return operand.get() == getDst();
        }
    }];
}

def Bytecode_BufferGetDimOp : Bytecode_ThreeOperandsInstructionBase<"buffer.get_dim", [], "1.0.0", "intel_npu::vm::OpCode::BUFFER_GET_DIM"> {
    let summary = "Get a dimension from a buffer's shape";

    let description = [{
        Read a shape dimension from an existing buffer.

        The `buffer` register contains the source buffer handle. The `dim_index` register selects
        the shape entry to read. The operation stores the selected shape value into the `dim_value`
        register.

        Example:
        ```mlir
        bytecode.buffer.get_dim %dim_value, %buffer, %dim_index
        ```
    }];
}

def Bytecode_BufferSubviewOp : Bytecode_VariadicOperandsInstructionOp<"buffer.subview", [SameVariadicOperandSize], "1.0.0", "intel_npu::vm::OpCode::BUFFER_SUBVIEW"> {
    let summary = "Create a strided subview of a buffer";

    let description = [{
        Create a view into a region of an existing buffer.

        The `src` register contains the parent buffer handle. The operation stores the new subview
        handle into the `dst` register. The `offsets`, `sizes` and `strides` operand groups describe
        the subview per dimension and must contain the same number of registers. The offsets, sizes and
        strides values are measured in elements. The resulting buffer shares the parent buffer memory
        and inherits the parent buffer permissions.

        Example:
        ```mlir
        bytecode.buffer.subview %dst, %src offsets(%offset0, %offset1) sizes(%size0, %size1) strides(%stride0, %stride1)
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        Bytecode_RegisterType:$src,
        Variadic<Bytecode_RegisterType>:$offsets,
        Variadic<Bytecode_RegisterType>:$sizes,
        Variadic<Bytecode_RegisterType>:$strides
    );

    let assemblyFormat = [{
        $dst `,` $src `offsets` `(` $offsets `)` `sizes` `(` $sizes `)` `strides` `(` $strides `)` attr-dict
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool isDstOperand(::mlir::OpOperand& operand) {
            return operand.get() == getDst();
        }
    }];
}

def Bytecode_BufferViewOp : Bytecode_VariadicOperandsInstructionOp<"buffer.view",
        [SameVariadicOperandSize, DeclareOpInterfaceMethods<SymbolUserOpInterface>],
        "1.0.0", "intel_npu::vm::OpCode::BUFFER_VIEW"> {
    let summary = "Create a buffer with a new element type, shape and strides";

    let description = [{
        Create a view of an existing buffer with a reinterpreted element type, shape and strides.

        The `src` register contains the source buffer handle. The `byte_offset` register holds the
        byte offset into the source buffer at which the view begins. The operation stores the new
        buffer handle into the `dst` register. The `elem_type` attribute is a symbol reference
        naming the new element type in the type section. The `shape` and `strides` operand groups
        describe the view per dimension and must contain the same number of registers. The underlying
        memory is shared with the source buffer. The VM validates that the byte offset plus the
        computed view size does not exceed the source buffer's memory range. The lifecycle of this
        buffer is tied to the parent buffer.

        Example:
        ```mlir
        bytecode.buffer.view %dst, %src, @type_section::@i64_type offset(%byteOff) shape(%size0, %size1) strides(%stride0, %stride1)
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        Bytecode_RegisterType:$src,
        Bytecode_RegisterType:$byte_offset,
        SymbolRefAttr:$elem_type,
        Variadic<Bytecode_RegisterType>:$shape,
        Variadic<Bytecode_RegisterType>:$strides
    );

    let assemblyFormat = [{
        $dst `,` $src `,` $elem_type `offset` `(` $byte_offset `)` `shape` `(` $shape `)` `strides` `(` $strides `)` attr-dict
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool isDstOperand(::mlir::OpOperand& operand) {
            return operand.get() == getDst();
        }
    }];
}

def Bytecode_BufferStoreOp : Bytecode_VariadicOperandsInstructionOp<"buffer.store", [], "1.0.0", "intel_npu::vm::OpCode::BUFFER_STORE"> {
    let summary = "Write a value into a buffer element";

    let description = [{
        Store a register value into an existing buffer at the given indices.

        The `buffer` register contains the target buffer handle. The `value` register holds the
        payload; the runtime writes the buffer's element-type byte width from the least-significant
        bytes of this register, regardless of the register's own width. The `indices` operand group
        describes the element coordinate; the operand count must match the runtime buffer rank.
        The operation produces no SSA result.

        Example:
        ```mlir
        bytecode.buffer.store %buffer, %value indices(%i0, %i1)
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$buffer,
        Bytecode_RegisterType:$value,
        Variadic<Bytecode_RegisterType>:$indices
    );

    let assemblyFormat = [{
        $buffer `,` $value `indices` `(` $indices `)` attr-dict
    }];

    let hasVerifier = 1;
}


def Bytecode_BufferLoadOp : Bytecode_VariadicOperandsInstructionOp<"buffer.load", [], "1.1.0", "intel_npu::vm::OpCode::BUFFER_LOAD"> {
    let summary = "Read a value from a buffer element";

    let description = [{
        Load a register value from an existing buffer at the given indices.

        The `buffer` register contains the source buffer handle. The runtime copies `elemByteSize`
        bytes from the addressed buffer element into the least-significant bytes of the destination
        register. The `indices` operand group describes the element coordinate; the operand count
        must match the runtime buffer rank.

        Example:
        ```mlir
        bytecode.buffer.load %dst, %buffer indices(%i0, %i1)
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        Bytecode_RegisterType:$buffer,
        Variadic<Bytecode_RegisterType>:$indices
    );

    let assemblyFormat = [{
        $dst `,` $buffer `indices` `(` $indices `)` attr-dict
    }];

    let hasVerifier = 1;
}

#endif
