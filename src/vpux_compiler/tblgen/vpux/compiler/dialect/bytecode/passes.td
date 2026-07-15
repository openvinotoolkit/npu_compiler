//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_PASSES
#define VPUX_COMPILER_DIALECT_BYTECODE_PASSES

include "mlir/Pass/PassBase.td"

//
// ConvertIntermediateBytecodeOps
//

def ConvertIntermediateBytecodeOps : PassBase<"convert-intermediate-bytecode-ops", "vpux::ModulePass"> {
    let summary = "Converts intermediate bytecode operations to bytecode instructions";

    let description = [{
        The pass converts `bytecode.ext.*` operations to final bytecode operations and materializes referenced data into
        bytecode sections. It creates constant, string and type sections next to the function section before rewriting
        if those sections are not already present in the module.

        `bytecode.ext.func` becomes `bytecode.func` with symbol references to function name and function type entries.
        `bytecode.ext.assert` moves the assert message to the string section and emits `bytecode.assert`.
        `bytecode.ext.buffer.create` registers the buffer and element types, materializes element type index, shape and
        strides as immediate registers, and emits `bytecode.buffer.create`.
    }];

    let constructor = "vpux::bytecode::createConvertIntermediateBytecodeOpsPass()";

    let dependentDialects = [
        "vpux::bytecode::BytecodeDialect"
    ];
}

//
// AllocateBytecodeRegisters
//

def AllocateBytecodeRegisters : PassBase<"allocate-bytecode-registers", "vpux::ModulePass"> {
    let summary = "Allocate concrete bytecode registers for virtual register operations";

    let description = [{
        The pass replaces every `bytecode.virtual_general_register` and `bytecode.virtual_parameter_register`
        operation with a corresponding `bytecode.general_register` operation whose register number follows the
        bytecode calling convention: general registers are numbered `[0..G-1]` and parameter registers are
        numbered `[G..G+P-1]`, where `G` is the number of general registers used by the function and `P` is
        the number of parameters of the function.

        General registers are assigned using the linear scan algorithm (Poletto, Sarkar): the live interval
        of each virtual general register is `[first_use, last_use]` in lexical op order; intervals are
        processed in start-point order and share a free pool of returned register numbers. Parameter
        registers with the same index are collapsed to a single `bytecode.general_register` via CSE.

        The call-frame size `G + P` is conveyed implicitly via the emitted `bytecode.general_register`
        operations: bytecode serialization derives the `num_general_registers` header field from the
        highest register number present in the function body (plus one). To keep this count consistent
        with the calling convention when some (or all) trailing parameters are unreferenced by the
        function body, the pass always materializes a `bytecode.general_register` at slot `G + P - 1`
        whenever `P > 0`.
    }];

    let constructor = "vpux::bytecode::createAllocateBytecodeRegistersPass()";

    let dependentDialects = [
        "vpux::bytecode::BytecodeDialect"
    ];
}

//
// SerializeKernelsToBytecode
//

def SerializeKernelsToBytecode : PassBase<"serialize-kernels-to-bytecode", "vpux::ModulePass"> {
    let summary = "Convert HostExec::BinaryOp operations into bytecode kernel section";

    let description = [{
        This pass converts HostExec::BinaryOp operations (containing serialized ELF binary data)
        into bytecode kernel_section/kernel operations.

        It walks HostExec::BinaryOp instances, extracts the binary data from the contained
        HostExec::BinaryDataOp, and creates bytecode.kernel ops in a bytecode.kernel_section.
        The BinaryOp is then erased and replaced with a stub mlir::ModuleOp containing a
        function declaration, to preserve symbol resolution for Core::NestedCallOp references.
    }];

    let constructor = "vpux::bytecode::createSerializeKernelsToBytecodePass()";

    let dependentDialects = [
        "vpux::bytecode::BytecodeDialect",
        "vpux::HostExec::HostExecDialect"
    ];
}

//
// InjectBytecodeMetadata
//

def InjectBytecodeMetadata : PassBase<"inject-bytecode-metadata", "vpux::ModulePass"> {
    let summary = "Inject network metadata into bytecode module";

    let description = [{
        Reads I/O descriptors from net.NetworkInfoOp and creates a bytecode.MetadataSectionOp
        in the module containing bytecode.NetworkMetadataOp, bytecode.InputMetadataOp,
        bytecode.OutputMetadataOp and bytecode.ProfilingOutputMetadataOp entries.

        The resulting metadata section is used in the Interpreter backend to construct
        NetworkMetadata without going through ELF serialization.

        Ordering requirement: must run after ConvertHostcodeToBytecodePass. The num_streams
        field in bytecode.NetworkMetadataOp is derived by counting bytecode.cmd_list.create
        ops present in the module at the time this pass runs. The pass signals failure if no
        such ops are found.
    }];

    let constructor = "vpux::bytecode::createInjectBytecodeMetadataPass()";

    let dependentDialects = [
        "vpux::bytecode::BytecodeDialect",
        "vpux::net::NetDialect"
    ];
}

#endif
