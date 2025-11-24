//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUASM_PASSES
#define VPUX_COMPILER_DIALECT_VPUASM_PASSES

include "mlir/Pass/PassBase.td"

//
// HoistInputOutputs
//

def HoistInputOutputs : PassBase<"hoist-input-outputs", "vpux::ModulePass"> {
    let summary = "Hoist Input Outputs to Module declaration";

    let description = [{
        Initially our Input and Outputs are providied as func arguments, that positionally represent the original network
        I/O's as defined by the ABI with the Plugin. This has the consequence that if we want to
        have OPS that reference sub-parts of these I/O's we have to add extra subview (or similar) ops that represent
        the transfer of a particular view of the I/O.
        As these "View" operations are not something we will want to execute, it is desired to remove them at a certain
        stage of compilation (where we would tend to keep only the ops that are of interenst to the RUNTIME)

        For this to happen, we will have to introduce *alias* buffer declarations that declare the exact desired buffer
        that we wish to use.

        This leads to a state of the IR where both the alias buffer declarations and function arguments are permitted
        to represent an I/O.

        This pass will effectively remove any funcArgument, and will hoist up their declarations to a global ModuleOp
        (similarliy with net.NetworkInfoOp with the critical difference that it will hold buffer/memref semantics)

        Any reference within the function body to the funcArguments will be replaced by the aliased declarations
    }];

    let constructor = "vpux::VPUASM::createHoistInputOutputsPass()";

    let dependentDialects = [
        "vpux::VPUASM::VPUASMDialect",
        "vpux::VPURT::VPURTDialect",
        "vpux::net::NetDialect"
    ];
}

//
// AddProfilingSection
//

def AddProfilingSection : PassBase<"add-profiling-section", "vpux::ModulePass"> {
    let summary = "Adds the profiling metadata section OP";

    let constructor = "vpux::VPUASM::createAddProfilingSectionPass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect",
        "vpux::VPUASM::VPUASMDialect"
    ];
}

//
// AddCompilerHash
//

def AddCompilerHash : PassBase<"add-compiler-hash", "vpux::FunctionPass"> {
    let summary = "Add compiler hash information to IR";

    let description = [{
        This pass adds the latest git commit hash info of the compiler into the IR.
        It can be used for tracking which compiler version is used to generate the binary,
        and for checking commit hash info by the ELF loader during compatibility check.
    }];

    let constructor = "vpux::VPUASM::createAddCompilerHashPass()";

    let dependentDialects = [
        "vpux::VPUASM::VPUASMDialect",
        "vpux::ELF::ELFDialect"
    ];
}

#endif // VPUX_COMPILER_DIALECT_VPUASM_PASSES
