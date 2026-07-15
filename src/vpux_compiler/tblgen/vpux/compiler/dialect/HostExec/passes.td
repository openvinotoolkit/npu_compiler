//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_HOSTEXEC_PASSES
#define VPUX_COMPILER_DIALECT_HOSTEXEC_PASSES

include "mlir/Pass/PassBase.td"

//
// SerializeELFToBinary
//

def SerializeELFToBinary : PassBase<"serialize-elf-to-binary", "vpux::ModulePass"> {
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
// InlineMainBatchingCalls
//

def InlineMainBatchingCalls : PassBase<"inline-main-batching-calls", "vpux::ModulePass"> {
    let summary = "Inlines main_batching calls before bytecode conversion";
    let description = [{
        The pass finds calls to main_batching* functions, inlines them at their call sites, and removes callees that
        become dead after inlining.
    }];
    let constructor = "vpux::HostExec::createInlineMainBatchingCallsPass()";
    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::cf::ControlFlowDialect"
    ];
}

//
// PrepareHostFuncForAsyncExecution
//

def PrepareHostFuncForAsyncExecution : PassBase<"prepare-host-function-for-async-execution", "vpux::ModulePass"> {
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
        async ops will be replaced with the corresponding Level Zero calls. Some ops like async.await and async.add_to_group
        can be mapped to no-op while lowering to Level Zero as they do not have a direct equivalent in the Level Zero
        and needed just to satisfy the async dialect requirements.

        When the `remove-return-values` option is enabled, the pass strips all return values from host functions
        across the entire module: every func.return in affected functions is replaced with a void return,
        then all func.call sites targeting those functions are updated to drop their results,
        and finally all affected function types are updated to return no results.
    }];

    let constructor = "vpux::HostExec::createPrepareHostFuncForAsyncExecutionPass()";

    let dependentDialects = [
        "mlir::async::AsyncDialect"
    ];

    let options = [
        Option<
            "removeReturnValues", "remove-return-values",
            "bool", "true",
            "Remove return values from host functions"
        >
    ];
}

//
// OptimizeMemRefCopies
//

def OptimizeMemRefCopies : PassBase<"optimize-memref-copies", "vpux::FunctionPass"> {
    let summary = "Optimize MemRef copies";

    let description = [{
        The pass optimizes memref.copy operations by removing unnecessary copies.
        Each pattern eliminates the copy by having the producer write directly into the
        target buffer. Three patterns are supported:
            - memref.copy copies data from a call result to a subview
            - memref.copy copies data from an scf.index_switch to a subview
            - memref.copy copies data from an alloc to a function argument
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

//
// ExtractReturnShapes
//

def ExtractReturnShapes : PassBase<"extract-return-shapes", "vpux::ModulePass"> {
    let summary = "Extract shapes info from return op operands of the main func";

    let description = [{
        This pass looks for all return op operands of the main func and creates `tensor.dim` ops
        for each dimension of the current operand with dynamic output shape. To avoid unused tensor/arith ops elimination,
        `tensor.from_elements` ops are created with `tensor.dim` ops as arguments and returned in the main func.
        Later in the pipeline in the pass `outline-dim-operations` these tensor and arith operations will be moved
        to a separate function (`func.func @output_shape`) and removed from the main func's return op and from `NetworkInfo`.
        If the outlined shape-computation chain still contains unresolved ops whose result shapes cannot be reified
        through MLIR interfaces such as `ReifyRankedShapedTypeOpInterface`, `outline-dim-operations` will fail.

        BEFORE:
        ```MLIR
            func.func @main(%arg: tensor<1x2x3x?xf32, {bounds = []}>) -> tensor<1x4x3x?xf32, {bounds = []}> {
                ...
                %0 = some_op(%arg) : tensor<1x2x3x?xf32, {bounds = []}> -> tensor<1x4x3x?xf32, {bounds = []}>
                ...
                return %0 : tensor<1x4x3x?xf32, {bounds = []}>
            }
        ```

        AFTER:
        ```MLIR
            func.func @main(%arg: tensor<1x2x3x?xf32, {bounds = []}>) -> (tensor<1x4x3x?xf32, {bounds = []}>, tensor<4xsi64>) {
                ...
                %0 = some_op(%arg) : tensor<1x2x3x?xf32, {bounds = []}> -> tensor<1x4x3x?xf32, {bounds = []}>
                ...

                %IDX_0 = arith.constant 0 : index
                %DIM_0 = tensor.dim %0, %IDX_0 : !BoundedType // '1'
                %IDX_CAST_0 = arith.index_cast %DIM_0 : index to i64

                %IDX_1 = arith.constant 1 : index
                %DIM_1 = tensor.dim %0, %IDX_1 : !BoundedType // '4'
                %IDX_CAST_1 = arith.index_cast %DIM_1 : index to i64

                %IDX_2 = arith.constant 2 : index
                %DIM_2 = tensor.dim %0, %IDX_2 : !BoundedType // '3'
                %IDX_CAST_2 = arith.index_cast %DIM_2 : index to i64

                %IDX_3 = arith.constant 3 : index
                %DIM_3 = tensor.dim %0, %IDX_3 : !BoundedType // '?'
                %IDX_CAST_3 = arith.index_cast %DIM_3 : index to i64

                %FROM_ELEMENTS = tensor.from_elements [[IDX_CAST_0]], [[IDX_CAST_1]], [[IDX_CAST_2]], [[IDX_CAST_3]] : tensor<4xi64>

                return %0, %FROM_ELEMENTS : tensor<1x4x3x?xf32, {bounds = []}>, tensor<4xsi64>
            }
        ```
    }];

    let constructor = "vpux::HostExec::createExtractReturnShapesPass()";

    let dependentDialects = [
        "vpux::HostExec::HostExecDialect",
        "mlir::func::FuncDialect",
        "mlir::tensor::TensorDialect",
        "mlir::arith::ArithDialect"
    ];
}

//
// OutlineDimOperations
//

def OutlineDimOperations : PassBase<"outline-dim-operations", "vpux::ModulePass"> {
    let summary = "Outline tensor shape operations to a separate function";

    let description = [{
        This pass collects all tensor and arith operations that calculate return shapes and outlines them
        to a separate `output_shape` func. No `func.call` op is inserted that is differenct from other outlining passes.

        BEFORE:
        ```MLIR
            func.func @main(%arg: tensor<1x4x3x?xf32, {bounds = []}>) -> (tensor<1x2x3x?xf32, {bounds = []}>, tensor<4xsi64>) {
                ...
                %0 = some_op(%arg) : tensor<1x4x3x?xf32, {bounds = []}> -> tensor<1x2x3x?xf32, {bounds = []}>
                ...

                %CST_1_i64 = arith.constant 1 : i64
                %CST_2_i64 = arith.constant 2 : i64
                %CST_3_i64 = arith.constant 3 : i64

                %CST_2 = arith.constant 2 : index
                %IDX = arith.constant 3 : index
                %DIM = tensor.dim %0, %IDX : !BoundedType   // '?'
                %DIV = arith.divsi %DIM, %CST_2 : index     // this operation comes from `resolve-shaped-type-result-dims`
                                                            // pass that resolves `tensor.dim` of result of operations
                %IDX_CAST = arith.index_cast %DIV : index to i64

                %FROM_ELEMENTS = tensor.from_elements [[CST_1_i64]], [[CST_2_i64]], [[CST_3_i64]], [[IDX_CAST]] : tensor<4xi64>

                return %0, %FROM_ELEMENTS : tensor<1x2x3x?xf32, {bounds = []}>, tensor<4xsi64>
            }
        ```

        AFTER:
        ```MLIR
            func.func @output_shape(%arg: tensor<1x4x3x?xf32, {bounds = []}>) -> tensor<4xsi64> {
                %CST_1_i64 = arith.constant 1 : i64
                %CST_2_i64 = arith.constant 2 : i64
                %CST_3_i64 = arith.constant 3 : i64

                %CST_2 = arith.constant 2 : index
                %IDX = arith.constant 3 : index
                %DIM = tensor.dim %0, %IDX : !BoundedType // '?'
                %DIV = arith.divsi %DIM, %CST_2 : index
                %IDX_CAST = arith.index_cast %DIV : index to i64

                %FROM_ELEMENTS = tensor.from_elements [[CST_1_i64]], [[CST_2_i64]], [[CST_3_i64]], [[IDX_CAST]] : tensor<4xi64>
                return %FROM_ELEMENTS : tensor<4xsi64>
            }

            func.func @main(%arg: tensor<1x4x3x?xf32, {bounds = []}>) -> tensor<1x2x3x?xf32, {bounds = []}> {
                ...
                %0 = some_op(%arg) : tensor<1x4x3x?xf32, {bounds = []}> -> tensor<1x2x3x?xf32, {bounds = []}>
                ...
                return %0 : tensor<1x2x3x?xf32, {bounds = []}>
            }
        ```
    }];

    let constructor = "vpux::HostExec::createOutlineDimOperationsPass()";

    let dependentDialects = [
        "vpux::HostExec::HostExecDialect",
        "mlir::func::FuncDialect",
        "mlir::tensor::TensorDialect",
        "mlir::arith::ArithDialect"
    ];
}

//
// GenerateExecutionContextFuncs
//

def GenerateExecutionContextFuncs : PassBase<"generate-execution-context-funcs", "vpux::ModulePass"> {
    let summary = "Generates functions to handle an execution context for L0 wrapper";

    let description = [{
        This pass generates LLVM functions to create, reset, destroy an execution context and update mutable commandlists
    }];

    let constructor = "vpux::HostExec::createGenerateExecutionContextFuncsPass()";

    let dependentDialects = [
        "vpux::HostExec::HostExecDialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::func::FuncDialect"
    ];
}

//
// WrapFuncCall
//

def WrapFuncCall : PassBase<"wrap-func-call", "vpux::ModulePass"> {
    let summary = "Goes through an entire module and replaces all callOps of a wrapped function with callOps of a wrapper function";

    let description = [{
        Each original function call operation is replaced with a corresponding call to a new function.
        A function whose call operations are replaced is referred to as the wrapped function.
        A function whose call operations are inserted in its place is referred to as the function wrapper.
        The function wrapper and the wrapped function must have identical argument lists and the same declaration.
        This pass can be used to provide a wrapper for an external function in order to inject troubleshooting information at the call site.
        It can also be used to facilitate decoupling a function from its calls during the execution of a complex pipeline,
        where the function definition and its uses belong to different modules in the IR hierarchy and therefore cannot be processed simultaneously.

        BEFORE: We had a diverged pipeline, which processed @topModule and @nestedModule in different ways:
        ```MLIR
            module @topModule {
              func.func private @mock_foo(%arg: tensor<1x4x3x?xf32, {bounds = []}>) -> (tensor<1x4x3x?xf32, {bounds = []}>) attributes {wrapFunctionAttr = ["wrapper=real_foo", "deleteWrapped=true"]}

              func.func @main(%arg: tensor<1x4x3x?xf32, {bounds = []}>) -> (tensor<1x4x3x?xf32, {bounds = []}>) {
                %ret = call @mock_foo(%arg) : (tensor<1x4x3x?xf32, {bounds = []}>) -> tensor<1x4x3x?xf32, {bounds = []}>
                return %ret : tensor<1x4x3x?xf32, {bounds = []}>
              }

              module @nestedModule {
                func.func @real_foo(%arg: memref<1x4x3x?xf32>) -> (memref<1x4x3x?xf32>) {
                    # actual processing
                }
              }
            }
        ```

        THEN: transformations were applied to @topModule lowering `tensor` to `memref`:
        ```MLIR
            module @topModule {
              func.func private @mock_foo(%arg: memref<1x4x3x?xf32>) -> (memref<1x4x3x?xf32>) attributes {wrapFunctionAttr = ["wrapper=real_foo", "deleteWrapped=true"]}

              func.func @main(%arg: memref<1x4x3x?xf32>) -> (memref<1x4x3x?xf32>) {
                %ret = call @mock_foo(%arg) : (memref<1x4x3x?xf32>) -> memref<1x4x3x?xf32>
                return %ret : memref<1x4x3x?xf32>
              }

              module @nestedModule {
                func.func @real_foo(%arg: memref<1x4x3x?xf32>) -> (memref<1x4x3x?xf32>) {
                    # actual processing
                }
              }
            }
        ```

        NOW modules are unpacked into the unified @topModule and WrapFuncCall is executed:
        ```MLIR
            module @topModule {
              func.func @main(%arg: memref<1x4x3x?xf32>) -> (memref<1x4x3x?xf32>) {
                %ret = call @real_foo(%arg) : (memref<1x4x3x?xf32>) -> memref<1x4x3x?xf32>
                return %ret : memref<1x4x3x?xf32>
              }

              func.func @real_foo(%arg: memref<1x4x3x?xf32>) -> (memref<1x4x3x?xf32>) {
                    # actual processing
                }
            }
        ```

    }];

    let constructor = "vpux::HostExec::createWrapFuncCallPass()";
}

//
// PropagateDynamicShapes
//

def PropagateDynamicShapes : PassBase<"propagate-dynamic-shapes", "vpux::ModulePass"> {
    let summary = "Propagate dynamic shapes from host to device functions";

    let description = [{
        This pass propagates dynamic shape information from host functions to NPU device functions.
        For each dynamic memref input to a nested call, it calculates the complete runtime shape and
        writes the dimension values into existing shape buffer operands at the call site.

        Algorithm:
            1. Classify each callee input as either:
                - Dynamic data source: memref with at least one dynamic dimension.
                - Shape buffer: memref<rank x si32> that must be filled.
            2. Pair each dynamic source with its corresponding shape buffer by position.
                Data and shape buffer positions are preserved after ungrouping.
            3. Emit stores that populate each shape buffer:
                - Dynamic dimensions: memref.dim(source, i)
                - Static dimensions: arith.constant
                - Cast path: index -> i32 -> si32, then memref.store

        Example for input shape [?x1x548]:
            %d0 = memref.dim(%input, 0)           // runtime extent
            %shape[2] = cast(%d0)                 // store runtime value
            %shape[1] = arith.constant(1)         // store static value
            %shape[0] = arith.constant(548)       // store static value

        This follows the same core idea as Reify/scf.for-based host materialization:
        extract runtime extents via memref.dim and pass shape buffers as regular data operands
        to execute_graph after lowering to LLVM/UMD.

        Input shape buffers are populated in reverse dimension order. SW kernels expect shape data
        in this order and reverse the index internally when performing shape inference.
    }];

    let constructor = "vpux::HostExec::createPropagateDynamicShapesPass()";

    let dependentDialects = [
        "vpux::HostExec::HostExecDialect",
        "mlir::arith::ArithDialect",
        "mlir::func::FuncDialect",
        "mlir::memref::MemRefDialect",
        "vpux::VPUIP::VPUIPDialect"
    ];
}

#endif
