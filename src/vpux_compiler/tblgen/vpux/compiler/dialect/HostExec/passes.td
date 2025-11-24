//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_HOSTEXEC_PASSES
#define VPUX_COMPILER_DIALECT_HOSTEXEC_PASSES

include "mlir/Pass/PassBase.td"

//
// SerializeELFToBinary
//

def SerializeELFToBinary : PassBase<"serialize-elf-to-binary", "vpux::FunctionPass"> {
    let summary = "Serialize ELF module to binary";

    let description = [{
        The pass serializes NPU code in the ELF dialect into a binary and stores the binary in IR as a HostExec.Binary op.
        An extern func op is created with the call signature the same as the entry point into the ELF binary.

        The resultant ELF files will be used within the host function.
    }];

    let constructor = "vpux::HostExec::createSerializeELFToBinaryPass()";

    let dependentDialects = [
        "vpux::HostExec::HostExecDialect"
    ];
}

//
// ConvertToLLVMUMDCalls
//

def ConvertToLLVMUMDCalls : PassBase<"convert-to-llvm-umd-calls", "vpux::ModulePass"> {
    let summary = "Converts MLIR operations to LLVM UMD runtime API calls";

    let description = [{
        This pass converts higher-level MLIR operations into LLVM dialect operations that make calls
        to the UMD (User Mode Driver) runtime API. The resulting code can be executed by the host CPU
        and interfaces with the NPU device through the UMD wrapper library.

        The pass handles (through level zero wrapper):
         - Converting memref.alloc operations to Level Zero memory allocations
         - Converting memref.copy operations to Level Zero memory copies
         - Converting async operations (Execute, Await, AwaitAll, CreateGroup, AddToGroup) to appropriate
           Level Zero command list operations
         - Managing command list creation, submission, and synchronization
    }];

    let constructor = "vpux::HostExec::createConvertToLLVMUMDCallsPass()";

    let dependentDialects = [
        "vpux::HostExec::HostExecDialect",
        "mlir::memref::MemRefDialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::func::FuncDialect",
        "mlir::async::AsyncDialect"
    ];
}

//
// PrepareHostFuncForAsyncExecution
//

def PrepareHostFuncForAsyncExecution : PassBase<"prepare-host-function-for-async-execution", "vpux::FunctionPass"> {
    let summary = "Prepares host function for async execution";

    let description = [{
        This pass wraps CallOpInterface operations in async.execute regions, creates async.group operations
        for scf.for loops to enable parallel execution, removes unused results from scf.index_switch operations
        to simplify control flow, and ensures proper async token management throughout the function.

        The special case of the pass is handling scf.for operations. Every scf.for operation assigned with an async group.
        It is expected that scf.for region contains at least one CallOpInterface operation.
        Those ops are wrapped into async regions and tokens produced by async.execute are added to the group.
        After the loop is finished, the async group is awaited by async.await_all.

        The pass is designed to be used to express execution based on Level Zero submission model. It is expected that
        async ops will be replaced with the corresponding Level Zero calls. Some ops like async.await and async_add_to_group
        can be mapped to no-op while lowering to Level Zero as they do not have a direct equivalent in the Level Zero
        and needed just to satisfy the async dialect requirements.
    }];

    let constructor = "vpux::HostExec::createPrepareHostFuncForAsyncExecutionPass()";

    let dependentDialects = [
        "mlir::async::AsyncDialect"
    ];
}

//
// OptimizeMemRefCopies
//

def OptimizeMemRefCopies : PassBase<"optimize-memref-copies", "vpux::FunctionPass"> {
    let summary = "Optimize MemRef copies";

    let description = [{
        The pass optimizes memref.copy operations by removing unnecessary copies. Two patterns are supported:
            1. memref.copy copies data from a call argument to a subview
            2. memref.copy copies data from an scf.index_switch to a subview
            3. memref.copy copies data from an alloc to a function argument
    }];

    let constructor = "vpux::HostExec::createOptimizeMemRefCopiesPass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::memref::MemRefDialect"
    ];
}

//
// ReplaceAllocsWithSingleAllocAndViews
//

def ReplaceAllocsWithSingleAllocAndViews : PassBase<"replace-allocs-with-single-alloc-and-views", "vpux::FunctionPass"> {
    let summary = "Replace multiple allocs with a single alloc and views";

    let description = [{
        The pass replaces multiple `memref.alloc` operations within a function with a single allocation of a large buffer.
        For each original allocation, a view into the large buffer is created to preserve semantics.
        This optimization reduces memory overhead and can improve performance by minimizing the number of allocations.

        Additionally, this approach enables further lowering, such as deferring the actual allocation
        until it is needed (e.g., when the host function receives input data larger than in previous invocations).

        In case of dynamic shapes, it is recommended to run this pass after
        `mlir::memref::createResolveShapedTypeResultDimsPass`, as it allows to simplify dependencies among dynamic dimensions.
    }];

    let constructor = "vpux::HostExec::createReplaceAllocsWithSingleAllocAndViewsPass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::memref::MemRefDialect",
        "vpux::HostExec::HostExecDialect"
    ];
}

//
// SerializeNetworkMetadata
//

def SerializeNetworkMetadata : PassBase<"serialize-network-metadata", "vpux::ModulePass"> {
    let summary = "Converts MLIR operations to LLVM UMD runtime API calls";

    let description = [{
        This pass construct network metadata and serialize it. Which is required
        as "main" function is llvm function to coordinate inference execution of
        subgraphs of a given model.
    }];

    let constructor = "vpux::HostExec::createSerializeNetworkMetadataPass()";

    let dependentDialects = [
        "vpux::HostExec::HostExecDialect",
        "mlir::func::FuncDialect",
    ];
}

#endif
