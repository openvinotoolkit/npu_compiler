//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_SHAVE_PASSES
#define VPUX_COMPILER_DIALECT_SHAVE_PASSES

include "mlir/Pass/PassBase.td"

//
// LoadExternalKernelResources
//

def LoadExternalKernelResources : PassBase<"load-external-kernel-resources", "vpux::FunctionPass"> {
    let summary = "Loads the implementations of the Shave.ExternalKernelOps into ShaveBinary resources";

    let description = [{
        This pass processes all Shave.ExternalKernelOp operations in the function and loads their corresponding
        kernel implementations from external ELF binary files into the ShaveBinaryResources infrastructure.

        For each ExternalKernelOp, the pass:
        1. Retrieves the kernel path and unique identifier from the operation attributes
        2. Reads the ELF binary file from the specified path
        3. Validates that the file is a valid ELF binary by checking the ELF magic bytes
        4. Loads the binary data into the Shave binary resources registry
        5. Associates the binary with the target architecture for proper compilation and deployment

        The pass maintains a set of already-loaded kernel identifiers to avoid redundant processing when multiple
        operations reference the same external kernel. This optimization reduces memory overhead and compilation time
        in scenarios where external kernels are reused across different invocations.

        This pass is essential for enabling integration of pre-compiled, hand-optimized, or third-party kernel
        implementations into the VPUX compilation pipeline, providing a bridge between externally developed kernels
        and the compiler's internal resource management system.
    }];

    let constructor = "vpux::Shave::createLoadExternalKernelResourcesPass()";

    let dependentDialects = [
        "vpux::Shave::ShaveDialect"
    ];
}

#endif
