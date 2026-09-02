//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_KERNEL_SUBMISSION
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_KERNEL_SUBMISSION

include "vpux/compiler/dialect/bytecode/ops/base.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/SymbolInterfaces.td"

def Bytecode_CmdListCreateOp : Bytecode_OneOperandInstructionBase<"cmd_list.create", [], "1.0.0", "intel_npu::vm::OpCode::CMD_LIST_CREATE"> {
    let summary = "Create a command list";

    let description = [{
        Creates a new command list and stores its opaque handle into the `reg` register.
        The command list is initially empty and is intended to be populated with commands,
        then finalized and submitted by subsequent command-list operations.

        Example:
        ```mlir
        bytecode.cmd_list.create %cmd_list  // creates a command list; handle stored in %cmd_list
        ```
    }];

    let extraClassDeclaration = [{
        bool isDstOperand(::mlir::OpOperand& operand) { return operand.get() == getReg(); }
    }];
}

def Bytecode_CmdListCloseOp : Bytecode_OneOperandInstructionBase<"cmd_list.close", [], "1.0.0", "intel_npu::vm::OpCode::CMD_LIST_CLOSE"> {
    let summary = "Close a command list";

    let description = [{
        Closes the command list identified by the opaque handle in `reg`.
        After this operation, the command list is finalized and ready to be submitted
        for execution by subsequent command-list operations.

        Example:
        ```mlir
        bytecode.cmd_list.close %cmd_list  // closes command list identified by %cmd_list
        ```
    }];
}

def Bytecode_CmdListExecOp : Bytecode_OneOperandsPlusImmInstructionBase<"cmd_list.exec", [], "1.0.0", "intel_npu::vm::OpCode::CMD_LIST_EXEC"> {
    let summary = "Execute a command list";

    let description = [{
        Submits the command list identified by the opaque handle in `reg` to the
        command queue for execution. Execution is asynchronous and this operation
        does not wait for completion.

        Example:
        ```mlir
        bytecode.cmd_list.exec %cmd_list %flag // submits command list identified by %cmd_list
        ```
    }];
}

def Bytecode_KernelCreateOp : Bytecode_VariadicOperandsInstructionOp<"kernel.create",
        [AttrSizedOperandSegments, DeclareOpInterfaceMethods<SymbolUserOpInterface>],
        "1.0.0", "intel_npu::vm::OpCode::KERNEL_CREATE"> {
    let summary = "Create a kernel object with input and output buffer handles";

    let description = [{
        Creates a kernel dispatch object from a `bytecode.kernel` symbol reference and a set of
        input/output buffer handle registers. The resulting handle is stored in `dst`.

        The `kernel` attribute names a `bytecode.kernel` op inside a `bytecode.kernel_section`
        in the parent module. During serialization the symbol is resolved to its positional index
        in the kernel section; that index is encoded as a 16-bit immediate in the binary stream,
        so no extra register is required to carry the value.

        Binary encoding:
        ```
        [opcode:2] [dst:2] [kernel_idx_imm:2] [N:2] [N×input_reg:2] [M:2] [M×output_reg:2]
        ```

        Example:
        ```mlir
        bytecode.kernel.create %dst, @kernel_section::@my_kernel, inputs(%in0, %in1), outputs(%out0)
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        SymbolRefAttr:$kernel,
        Variadic<Bytecode_RegisterType>:$inputs,
        Variadic<Bytecode_RegisterType>:$outputs
    );

    let assemblyFormat = [{
        $dst `,` $kernel `,` `inputs` `(` $inputs `)` `,` `outputs` `(` $outputs `)` attr-dict
    }];

    let extraClassDeclaration = [{
        bool isDstOperand(::mlir::OpOperand& operand) {
            return operand.get() == getDst();
        }
    }];
}

def Bytecode_CmdListAddKernelOp : Bytecode_VariadicOperandsInstructionOp<
        "cmd_list.add_kernel", [AttrSizedOperandSegments], "1.0.0", "intel_npu::vm::OpCode::CMD_LIST_ADD_KERNEL"> {
    let summary = "Add a kernel to a command list";

    let description = [{
        Appends a kernel command to the command list identified by `cmd_list`.
        `kernel` is the opaque kernel handle to append. `signal_events` and `wait_events`
        are variadic register lists of event handles associated with this kernel command.

        Signature: cmd_list.add_kernel rs1, rs2, N, rN..., M, rM...

        Example:
        ```mlir
        bytecode.cmd_list.add_kernel %cmd_list, %kernel,
                                     (%signal0 : !bytecode.Register),
                                     (%wait0, %wait1 : !bytecode.Register, !bytecode.Register)
        ```
    }];

    let arguments = (ins
        Bytecode_RegisterType:$cmd_list,
        Bytecode_RegisterType:$kernel,
        Variadic<Bytecode_RegisterType>:$signal_events,
        Variadic<Bytecode_RegisterType>:$wait_events
    );

    let assemblyFormat = [{
        $cmd_list `,` $kernel `,`
        `(` ($signal_events^ `:` type($signal_events))? `)` `,`
        `(` ($wait_events^ `:` type($wait_events))? `)`
        attr-dict
    }];
}

#endif
