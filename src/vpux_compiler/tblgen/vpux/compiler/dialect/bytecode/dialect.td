//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE
#define VPUX_COMPILER_DIALECT_BYTECODE

include "mlir/IR/OpBase.td"

def Bytecode_Dialect : Dialect {
    let summary = "Bytecode Dialect";

    let description = [{
        The `Bytecode` dialect models a low-level, host-executable bytecode format
        for the NPU compiler. The format contains a register-based instruction set,
        which includes opcodes for the following computations: arithmetic, bitwise,
        comparison, type-conversion, register and buffer access, control flow,
        and kernel submission. Operations are versioned and map directly to
        bytecode instructions.

        The dialect is serialized into a binary blob and executed by the host
        virtual machine / interpreter. It is primarily used by the `HostCompile`
        pipeline.
    }];


    let name = "bytecode";

    let cppNamespace = "vpux::bytecode";

    let dependentDialects = [
        "vpux::Core::CoreDialect"
    ];

    let extraClassDeclaration = [{
        void registerAttributes();
        void registerTypes();
        static void setupExtraInterfaces(mlir::DialectRegistry& registry);
    }];

    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

#endif
