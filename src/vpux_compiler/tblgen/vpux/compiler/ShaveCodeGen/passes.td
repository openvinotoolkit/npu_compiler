//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_SHAVE_CODE_GEN_PASSES
#define VPUX_COMPILER_SHAVE_CODE_GEN_PASSES

include "mlir/Pass/PassBase.td"

//=================================================================================
// ShaveCodeGen Passes
//=================================================================================

//
// OutlineCodeGenCapsules
//

def OutlineCodeGenCapsules : PassBase<"outline-codegen-capsules", "vpux::ModulePass"> {
    let summary = "Outline the CodeGenCapsules contents to external functions";

    let description = [{
        The pass outlines the Shave-directed CodeGenCapsules to external functions.
        In place of the capsules, it introduces VPU::GenericSwLayer ops that now represent calls to the shave functions.
    }];

    let constructor = "vpux::ShaveCodeGen::createOutlineCodeGenCapsulesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::func::FuncDialect"
    ];
}

//
// FoldUnitDimReshapes
//

def FoldUnitDimReshapes : PassBase<"fold-unit-dim-reshapes", "vpux::FunctionPass"> {
    let summary = "Removes reshape/view chains that only affect unit dimensions";

    let description = [{
        AI-generated summary

        Eliminates reshape operations that only add or remove unit (size-1) dimensions from
        tensors. Folds trivial reshapes where input and output have the same number of non-unit
        elements, simplifying the IR before SHAVE kernel code generation.
    }];

    let constructor = "vpux::ShaveCodeGen::createFoldUnitDimReshapesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::tensor::TensorDialect",
        "mlir::linalg::LinalgDialect"
    ];
}

//
// LowerMathToShaveIntrinsics
//

def LowerMathToShaveIntrinsics : PassBase<"lower-math-to-shave-intrinsics", "vpux::ModulePass"> {
    let summary = "Lower math operations to shave intrinsics";

    let description = [{
        Some math operations directly map to shave intrinsics in cases where there are no
        equivalent generic LLVM operations (for example tanh). In these cases there is no
        generic lowering path to the LLVM dialect.
        This pass inserts the required declarations for the shave intrisics in the software
        module and replaces the math operations with function calls.
    }];

    let constructor = "vpux::ShaveCodeGen::createLowerMathToShaveIntrinsicsPass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::math::MathDialect"
    ];
}

def LinalgTileAndFuseSwLayers : PassBase<"linalg-tile-and-fuse-sw-layers", "vpux::ModulePass"> {
    let summary = "In-kernel tile and fuse pass";

    let description = [{
       The purpose is to remove (ideally all) of the temporary buffers. The resulting tiles
       will be promotable to allocas which in turn will later be promoted to registers.
       This pass should run after all of the various other tiling passes.
    }];

    let constructor = "vpux::ShaveCodeGen::createLinalgTileAndFuseSwLayersPass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::scf::SCFDialect"
    ];
}

def ShaveStackAllocation : PassBase<"shave-stack-allocation", "vpux::ModulePass"> {
    let summary = "Performs stack optimizations for shave";

    let description = [{
        AI-generated summary

        Optimizes memory allocation in SHAVE kernel functions:
        - Runs buffer hoisting and PromoteBuffersToStack to move small allocations to stack
        - Replaces remaining memref.alloc operations with function arguments (scratch buffers)
        - Updates the kernel FuncOp signature and all GenericSwLayerOp call sites to pass the
          new scratch buffer operands
    }];

    let constructor = "vpux::ShaveCodeGen::createShaveStackAllocationPass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::bufferization::BufferizationDialect"
    ];
}


//
// AdaptLLVMFuncsForShave
//

def AdaptLLVMFuncsForShave : PassBase<"adapt-LLVM-funcs-for-shave", "vpux::ModulePass"> {
    let summary = "Adapt LLVM Functions for the Shave calling convention";

    let description = [{
        Shave has some expectations for the functions it calls via the management kernel.
        This pass adjusts the generated LLVM functions to be compatible with this calling convention.
        More exactly:
            1. Packs all the function arguments into a singular pointer type which is all-encapsulating
                Note: To accomplish this, pointer arithmetic is introduced in order to retrieve
                      the necessary data from the opaque pointer argument.
            2. Removes function return values
    }];

    let constructor = "vpux::ShaveCodeGen::createAdaptLLVMFuncsForShavePass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::math::MathDialect"
    ];
}

//
// EncapsulateCodeGenOps
//

def EncapsulateCodeGenOps : PassBase<"encapsulate-codegen-ops", "vpux::FunctionPass"> {
    let summary = "Encapsulate ShaveCodeGen supported ops into IE.CodeGenCapsule operations";

    let description = [{
        This pass aims at encapsulating all the ShaveCodeGen supported operations within
        IE::CodeGenCapsule ops. The triage for the encapsulation is done based on the
        ShaveCodeGenSupportedOpInterface. All ops implementing this interface will be wrapped.

        Additionally, this pass also constructs a FusionChainAnalysis which can be used in a
        succeding fusion pass. The construction of this analysis is also required for the
        well-functioning of the encapsulation logic.
    }];

    let constructor = "vpux::ShaveCodeGen::createEncapsulateCodeGenOpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// EarlyCodeGenCapsuleFusion
//

def EarlyCodeGenCapsuleFusion : PassBase<"early-codegen-capsule-fusion", "vpux::FunctionPass"> {
    let summary = "Fuse compatible CodeGenCapsules";

    let description = [{
        This pass aims at fusing compatible CodeGenCapsules such that the compute-abstracting
        operation within these capsules can be further fused.

        The fusion is done by using the FusionChainAnalysis and querying the pre-computed capsule chains.
        Because of this, EncapsulateCodeGenOps & the ahead-of-time construction of the FusionChainAnalysis
        is a hard requirement for the well-functioning of this pass.
    }];

    let constructor = "vpux::ShaveCodeGen::createEarlyCodeGenCapsuleFusionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

def ShaveKernelSimplify : PassBase<"shave-kernel-simplify", "vpux::ModulePass"> {
    let summary = "Applies peephole optimizations to the software module";

    let description = [{
        AI-generated summary

        Simplifies SHAVE kernel function bodies through canonical pattern application:
        - Bubbles extract_slice ops upward through linalg operations
          (linalg::populateBubbleUpExtractSliceOpPatterns)
        - Swaps extract_slice with fill patterns
          (linalg::populateSwapExtractSliceWithFillPatterns)
        - Folds redundant tensor.empty operations
          (tensor::populateFoldTensorEmptyPatterns)
        Reduces temporary allocations and simplifies linalg/tensor IR before lowering.
    }];

    let constructor = "vpux::ShaveCodeGen::createShaveKernelSimplifyPass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::tensor::TensorDialect"
    ];
}

def FlattenEltwiseKernel : PassBase<"flatten-eltwise-kernel", "vpux::ModulePass"> {
    let summary = "Flatten element-wise kernels";

    let description = [{
        AI-generated summary

        Flattens multi-dimensional tensor access patterns in element-wise SHAVE kernels into
        linear 1D iteration. Replaces nested loop structures over N-dimensional memrefs with a
        single loop over total element count, reducing loop overhead and enabling better
        vectorization for element-wise operations like ReLU or Add.
    }];

    let constructor = "vpux::ShaveCodeGen::createFlattenEltwiseKernelPass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::linalg::LinalgDialect"
    ];
}

def MoveKernelResultsToArguments : PassBase<"move-kernel-results-to-arguments", "vpux::ModulePass"> {
    let summary = "Moves kernel results to the arguments list for outlined ShaveCodeGen functions";

    let description = [{
        Enforce the calling convention for returned tensors in outlined software kernels.
        The memory for the returned tensors is allocated by the caller, and the callee
        receives memrefs as additional arguments (similar to a structure return).
    }];

    let constructor = "vpux::ShaveCodeGen::createMoveKernelResultsToArgumentsPass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::bufferization::BufferizationDialect"
    ];
}

def OneShotBufferizeSWKernels : PassBase<"one-shot-bufferize-sw-kernels", "vpux::ModulePass"> {
    let summary = "Bufferize outlined software kernels";

    let description = [{
        AI-generated summary

        Applies one-shot bufferization to SHAVE software kernel functions, converting
        tensor-based IR into memref-based IR. Transforms tensor.empty, tensor.insert_slice,
        and other tensor operations into their buffer equivalents (memref.alloc, memref.copy),
        preparing kernel bodies for direct memory access code generation.
    }];

    let constructor = "vpux::ShaveCodeGen::createOneShotBufferizeSWKernelsPass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::bufferization::BufferizationDialect",
    ];
}

def ParametricCapsuleTiling : PassBase<"scg-parametric-capsule-tiling", "vpux::FunctionPass"> {
    let summary = "Capsule parametric tiling";

    let description = [{
        Performs parametric tiling of clusters of tileable ops inside CodeGenCapsule ops.
        Tiling is performed in a parallel fashion. Tiling axes are chosen such that the
        maximal number of ops are being fused while still having the possibility to tile
        and fit into CMX.
    }];

    let constructor = "vpux::ShaveCodeGen::createParametricCapsuleTilingPass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "vpux::IE::IEDialect",
        "vpux::Shave::ShaveDialect",
        "mlir::tensor::TensorDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::scf::SCFDialect",
        "mlir::arith::ArithDialect",
        "mlir::affine::AffineDialect"
    ];
}

#endif
