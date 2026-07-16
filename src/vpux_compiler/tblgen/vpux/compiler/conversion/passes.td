//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_CONVERSION_PASSES
#define VPUX_COMPILER_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//=================================================================================
// LowerIE2VPU
//=================================================================================

//
// ConvertIEToVPUNCE
//

def ConvertIEToVPUNCE : PassBase<"convert-IE-to-VPU-NCE", "vpux::FunctionPass"> {
    let summary = "Convert supported IE layers to VPU NCE operations";

    let description = [{
        AI-generated summary

        Converts NCE-compatible IE operations (Convolution, AveragePool,
        PermuteQuantize, GroupConvolution, Eltwise, MatMul) to VPU NCE counterparts
        (NCEConvolution, NCEAveragePool, NCEPermute, NCEDepthConvolution, NCEEltwise,
        NCEMatMul). Generates aligned weight tensors, weights tables, and PPE
        configuration attributes. Does not change memory spaces or insert Copy
        operations.
    }];

    let constructor = "vpux::createConvertIEToVPUNCEPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// AddBuffersForNetResults
//

def AddBuffersForNetResults : PassBase<"add-buffers-for-net-results", "vpux::ModulePass"> {
    let summary = "Add network results in Function parameters";

    let description = [{
        This pass adds buffers to the function parameters to copy the network result to them.
        In contrast to MLIR standard analogue pass, this pass preserves the function results to utilize use-def chains on bufferized IR.
        The `return` operation will take an aliases of output buffers from the function arguments.
    }];

    let constructor = "vpux::createAddBuffersForNetResults()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "mlir::func::FuncDialect",
        "mlir::memref::MemRefDialect"
    ];

    let options = [
        Option<
            "useMemrefForHostFunctionBufferization", "use-memref-for-host-function-bufferization",
            "bool", "false",
            "Flag to use memref copy instead of VPUIP copy in host function."
        >
    ];
}

//=================================================================================
// Lowering VPU to VPUIP
//=================================================================================

//
// InPlaceBufferizationAnalyze
//

def InPlaceBufferizationAnalyze : PassBase<"in-place-bufferization-analyze", "vpux::ModulePass"> {
    let summary = "One shot in place analyze ops from VPU to VPUIP";

    let description = [{
        Analyze if operations could be bufferized in-place. This pass must be called before
        one-shot bufferize action is performed(done by OneShotBufferizeVPU2VPUIP).
    }];

    let constructor = "vpux::createInPlaceBufferizationAnalyzePass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// OneShotBufferizeVPU2VPUIP
//

def OneShotBufferizeVPU2VPUIP : PassBase<"one-shot-bufferize-VPU-to-VPUIP", "vpux::ModulePass"> {
    let summary = "One shot bufferize ops from VPU to VPUIP";

    let description = [{
        One shot bufferize ops from VPU to VPUIP dialect, include converting:
        (1) software ops from VPU to VPUIP::SWKernel ops. (37XX+)
        (2) ops which can be executed on NCE to VPUIP::NCEClusterTask ops. (37XX+)
    }];

    let constructor = "vpux::createOneShotBufferizeVPU2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect",
        "mlir::memref::MemRefDialect",
        "mlir::bufferization::BufferizationDialect",
        "mlir::scf::SCFDialect",
        "mlir::tensor::TensorDialect"
    ];
}

//
// AdjustDynamicOpsBeforeBufferization
//

def AdjustDynamicOpsBeforeBufferization : PassBase<"adjust-dynamic-ops-before-bufferization", "vpux::ModulePass"> {
    let summary = "Prepare operation with dynamic tensors to be properly bufferized";

    let description = [{
        AI-generated summary

        Prepares dynamic-shape operations for bufferization by converting operations with
        dynamic dimensions into forms that carry explicit shape information.

        Registered rewrite patterns:
        - UnsqueezeRewrite: replaces VPU.UnsqueezeOp (when it has dynamic tensors) with
          VPU.DynamicReshapeOp plus a constant shape tensor that encodes the output dimensions
    }];

    let constructor = "vpux::createAdjustDynamicOpsBeforeBufferizationPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ConvertEltwiseLayers2Math
//

def ConvertEltwiseLayers2Math : PassBase<"convert-eltwise-layers-to-math", "vpux::FunctionPass"> {
    let summary = "Convert Eltwise IE Dialect SW Layers to Math dialect";

    let description = [{
        Some IE SW Layers are simple elementwise operations that can be directly mapped to Math dialect
        equivalent ops. This pass defines conversion patterns for such operations.
    }];

    let constructor = "vpux::ShaveCodeGen::createConvertEltwiseLayers2MathPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::arith::ArithDialect",
        "mlir::func::FuncDialect",
        "mlir::scf::SCFDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::tensor::TensorDialect",
        "mlir::math::MathDialect"
    ];
}

def DecomposeAggregateOps : PassBase<"decompose-aggregate-ops", "vpux::ModulePass"> {
    let summary = "Decomposes aggregate ops (mainly softmax)";

    let description = [{
        Decomposes aggregate ops in the SW module. The purpose is to lower
        linalg.softmax (and any other aggregate ops).
    }];

    let constructor = "vpux::ShaveCodeGen::createDecomposeAggregateOpsPass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::arith::ArithDialect",
        "mlir::math::MathDialect",
        "mlir::tensor::TensorDialect",
        "mlir::linalg::LinalgDialect"
    ];
}

//
// ExpandLayers
//

def ExpandLayers : PassBase<"expand-layers", "vpux::ModulePass"> {
    let summary = "Expand high-level math operations into polynomial approximations for SHAVE kernels";

    let description = [{
        AI-generated summary

        Lowers math operations (sin, cos, erf, round, asin, acos, asinh, acosh, fma, negf)
        from the VPU.SW module generated functions into polynomial approximation sequences
        and bitwise operations suitable for SHAVE execution. Handles f16-specific rounding
        without f32 promotion and supports both scalar and vector operands.
    }];

    let constructor = "vpux::ShaveCodeGen::createExpandLayersPass()";

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::math::MathDialect"
    ];
}


//
// ConvertAffine2LLVM
//

def ConvertAffine2LLVM : PassBase<"convert-Affine-to-LLVM", "vpux::ModulePass"> {
    let summary = "Convert Affine and SCF control flow to LLVM IR for SHAVE kernel compilation";

    let description = [{
        AI-generated summary

        Converts affine.for loops and structured control flow (SCF) to standard LLVM
        control flow, then lowers all operations (arith, memref, func, math) to LLVM
        dialect. Propagates statically known memref properties after lowering to compensate
        for any information lost during argument conversion. Adjusts index arguments to
        keep the calling convention compatible with the caller. Analyzes memref argument
        aliases across all SW kernel invocations to add LLVM noalias attributes, enabling
        aggressive vectorization optimizations in generated SHAVE code.
    }];

    let constructor = "vpux::ShaveCodeGen::createConvertAffine2LLVMPass()";

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::affine::AffineDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::memref::MemRefDialect",
        "mlir::func::FuncDialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::scf::SCFDialect",
        "mlir::math::MathDialect",
        "mlir::index::IndexDialect"
    ];
}

//
// ConvertVPUIP2VPUMI37XX
//

def ConvertVPUIP2VPUMI37XX : PassBase<"convert-VPUIP-to-VPUMI37XX", "vpux::ModulePass"> {
    let summary = "Convert VPUIP dialect to VPUMI37XX dialect";

    let description = [{
        AI-generated summary

        Lowers VPUIP dialect operations containing DMA and NCE computations to VPUMI37XX
        register-mapped operations (DPUInvariant, DPUVariant, NNDMA). Handles distributed
        buffer unrolling for multi-cluster scenarios, barrier legalization, and task
        dependency chaining. Generates MappedInference operation encapsulating all tasks
        into executable kernel blocks.
    }];

    let constructor = "vpux::createConvertVPUIP2VPUMI37XXPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect",
        "vpux::VPUMI37XX::VPUMI37XXDialect",
        "vpux::VPURegMapped::VPURegMappedDialect",
        "vpux::ELFNPU37XX::ELFNPU37XXDialect"
    ];
}

//
// ConvertVPUIP2VPUMI40XX
//

def ConvertVPUIP2VPUMI40XX : PassBase<"convert-VPUIP-to-VPUMI40XX", "vpux::FunctionPass"> {
    let summary = "Convert VPUIP dialect to VPUMI40XX dialect";

    let description = [{
        AI-generated summary

        Lowers VPUIP dialect operations to VPUMI40XX register-mapped instructions (DMA, DPU,
        barrier operations). Supports Workload Management Extension (WLM) for multi-engine
        scheduling, SHAVE stack allocation, memory-side caching, and profiling metadata
        generation. Enumerates and chains tasks by executor locality with proper barrier
        dependency tracking.
    }];

    let constructor = "vpux::createConvertVPUIP2VPUMI40XXPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect",
        "vpux::VPUMI40XX::VPUMI40XXDialect",
        "vpux::VPURegMapped::VPURegMappedDialect"
    ];

    let options = [
        Option<
            "allocateDDRStackFrames", "allocate-ddr-stack-frames",
            "bool", "false",
            "Flag to enable shave stack frames allocation in DDR."
        >
    ];
}

//
// ConvertVPUMI40XX2VPUASM
//

def ConvertVPUMI40XX2VPUASM : PassBase<"convert-VPUMI40XX-to-VPUASM", "vpux::ModulePass"> {
    let summary = "Convert VPUMI40XX dialect to VPUASM";

    let description = [{
        VPUMI40XX is the last level of representation where we still employ op-result-operand relationships.
        After this conversion pass expect only symbolic relationships between the OPS. (with the occasional exception,
        and helped by *VPUASM_SymbolizeValueOp* we can still permit non-symbolic ops to be present in the IR, having
        an explicit conversion OP, that can still be present as information containers).

        Concrete example for DistributetBuffers that have no VPUASM counterpart, we can still hook them into VPUASM.Invariant
        by using SymbolizeValueOp to create a symbol for it.
    }];

    let constructor = "vpux::createConvertVPUMI40XX2VPUASMPass()";
    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect",
        "vpux::VPUMI40XX::VPUMI40XXDialect",
        "vpux::VPURegMapped::VPURegMappedDialect",
        "vpux::VPUASM::VPUASMDialect",
        "vpux::VPUIPDPU::VPUIPDPUDialect",
        "vpux::ELFNPU37XX::ELFNPU37XXDialect"
    ];

}

//
// ConvertVPUMI37XX2ELF
//

def ConvertVPUMI37XX2ELF : PassBase<"convert-VPUMI37XX-to-ELF", "vpux::ModulePass"> {
    let summary = "Convert VPUMI37XX register-mapped operations to ELF binary structure";

    let description = [{
        AI-generated summary

        Converts VPUMI37XX operations (DPUInvariant, DPUVariant, NNDMA, MappedInference)
        into typed ELF sections (.text.DPUInvariants, .text.DPUVariants, .data.Buffers,
        etc.), symbol tables, and relocation records. Creates symbol tables for network I/O
        and runtime CMX address mapping. Generates relocations to bind abstract task indices
        to concrete addresses at load time.
    }];

    let constructor = "vpux::createConvertVPUMI37XX2ELFPass()";

    let dependentDialects = [
        "vpux::VPURT::VPURTDialect", // E#47129
        "vpux::VPUMI37XX::VPUMI37XXDialect",
        "vpux::ELFNPU37XX::ELFNPU37XXDialect"
    ];
}

//=================================================================================
// Lower VPUASM2ELF
//=================================================================================

//
// ConvertVPUASM2NPUReg40XX
//

def ConvertVPUASM2NPUReg40XX : PassBase<"convert-VPUASM-to-NPUReg40XX", "vpux::ModulePass"> {
    let summary = "Convert VPUASM dialect to NPUReg40XX";

    let description = [{
        VPUASM -> NPUReg40XX lowering that happens before relocation passes. Intended to be exapanded
        with more operations gradually over time. Eventually, planned to be replaced with regular
        VPUASM -> NPUReg40XX lowering.
    }];

    let constructor = "vpux::createConvertVPUASM2NPUReg40XXPass()";
    let dependentDialects = [
        "vpux::VPURegMapped::VPURegMappedDialect",
        "vpux::VPUASM::VPUASMDialect",
        "vpux::NPUReg40XX::NPUReg40XXDialect",
        "vpux::ELF::ELFDialect"
    ];
}

//
// ConvertVPUASM2NPUReg50XX
//

def ConvertVPUASM2NPUReg50XX : PassBase<"convert-VPUASM-to-NPUReg50XX", "vpux::ModulePass"> {
    let summary = "Convert VPUASM dialect to NPUReg50XX";

    let description = [{
        VPUASM -> NPUReg50XX lowering that happens before relocation passes. Intended to be exapanded
        with more operations gradually over time. Eventually, planned to be replaced with regular
        VPUASM -> NPUReg50XX lowering.
    }];

    let constructor = "vpux::createConvertVPUASM2NPUReg50XXPass()";
    let dependentDialects = [
        "vpux::VPURegMapped::VPURegMappedDialect",
        "vpux::VPUASM::VPUASMDialect",
        "vpux::NPUReg50XX::NPUReg50XXDialect",
        "vpux::ELF::ELFDialect"
    ];
}
//
// ConvertVPUIPDPU2NPUReg40XX
//

def ConvertVPUIPDPU2NPUReg40XX : PassBase<"convert-VPUIPDPU-to-NPUReg40XX", "vpux::ModulePass"> {
    let summary = "Convert VPUIPDPU dialect to NPUReg40XX";

    let description = [{
        VPUIPDPU is a logical representation of the DPU descriptor hierarchy, following the HW functionality specific for each architecture generation.
        NPUReg40XX dialect represents ops in terms of hardware registers.
    }];

    let constructor = "vpux::createConvertVPUIPDPU2NPUReg40XXPass()";
    let dependentDialects = [
        "vpux::VPURegMapped::VPURegMappedDialect",
        "vpux::VPUIPDPU::VPUIPDPUDialect",
        "vpux::NPUReg40XX::NPUReg40XXDialect",
        "vpux::ELF::ELFDialect"
    ];

    let options = [
        Option<
            "dpuDryRun", "dpu-dry-run",
            "std::string", "\"none\"",
            "Flag which identifies whether to dry run DPU operations"
        >
    ];

}

//
// ConvertVPUASM2NPUReg50XX
//

def ConvertVPUIPDPU2NPUReg50XX : PassBase<"convert-VPUIPDPU-to-NPUReg50XX", "vpux::ModulePass"> {
    let summary = "Convert VPUIPDPU dialect to NPUReg50XX";

    let description = [{
        VPUIPDPU is a logical representation of the DPU descriptor hierarchy, following the HW functionality specific for each architecture generation.
        NPUReg50XX dialect represents ops in terms of hardware registers.
    }];

    let constructor = "vpux::createConvertVPUIPDPU2NPUReg50XXPass()";
    let dependentDialects = [
        "vpux::VPURegMapped::VPURegMappedDialect",
        "vpux::VPUIPDPU::VPUIPDPUDialect",
        "vpux::NPUReg50XX::NPUReg50XXDialect",
        "vpux::ELF::ELFDialect"
    ];

    let options = [
        Option<
            "npu5PPEBackwardsCompatibilityModeOpt", "npu5-ppe-backwards-compatibility-mode",
            "vpux::VPURegMapped::NPU5PPEBackwardsCompatibilityMode", "",
            "Flag which controls NPU5 PPE Backwards compatibility mode",
            [{::llvm::cl::values(
            clEnumValN(vpux::VPURegMapped::NPU5PPEBackwardsCompatibilityMode::DISABLED, "DISABLED", "NPU5 PPE Backwards Compatibility Mode DISABLED"),
            clEnumValN(vpux::VPURegMapped::NPU5PPEBackwardsCompatibilityMode::ENABLED, "ENABLED", "NPU5 PPE Backwards Compatibility Mode ENABLED")
          )}]
        >
    ];
}
//
// ConvertHostcodeToBytecode
//

def ConvertHostcodeToBytecode : PassBase<"convert-hostcode-to-bytecode", "vpux::ModulePass"> {
    let summary = "Convert operations that execute on host to Bytecode";

    let description = [{
        Converts host-compile functions (func.func with the HostCompile compilation mode attribute)
        to the Bytecode dialect.

        In Bytecode dialect, there are multiple sections that exist (e.g. constant section, string section).
        This pass only creates the function section, so the other sections will be inserted by later passes.
        As a result, operations that use data from other sections will be converted to intermediate operations in the Bytecode dialect,
        and the later passes will convert them to the final Bytecode operations.
        For example, `cf.assert` will be converted to `bytecode.ext.assert` in this pass and will store the message string as an attribute.
        Another pass will lower `bytecode.ext.assert` to `bytecode.assert` and move the message string to the string section, and replace
        the attribute with a symbol from the string section. See the `ConvertIntermediateBytecodeOps` pass for details.
        The pass also lowers static `memref.alloc` to `bytecode.ext.buffer.create` and static non-rank-reducing
        `memref.subview` directly to `bytecode.buffer.subview` with per-dimension offsets/sizes/strides materialized
        as immediate registers. `memref.dim` is folded when both the source dimension and the index are statically
        known; the residual cases (dynamic source dim or dynamic index) lower to `bytecode.buffer.get_dim`, which
        reads the dimension from the runtime buffer descriptor. `memref.store` lowers to `bytecode.buffer.store`
        when the stored value and all indices are converted to bytecode registers.

        The following dialect operations are lowered:
        - `arith` arithmetic and comparison ops -> bytecode arithmetic/comparison ops
        - `cf.br` -> `bytecode.jmp`
        - `cf.cond_br` -> `bytecode.je` with dedicated setup blocks for the true and false edges
        - `cf.switch` -> a chain of `bytecode.je` ops (one per case value) with a default fallback block
        - empty `func.return` -> `bytecode.ret`; non-empty `func.return` -> `bytecode.retv`
        - `func.call` -> `bytecode.call`
        - `async.create_group`, `async.add_to_group`, `async.await_all`, `async.execute` -> bytecode async ops
        - `cf.assert` -> `bytecode.ext.assert` (string payload kept as attribute; resolved by ConvertIntermediateBytecodeOps)

        Block arguments of non-entry blocks are eliminated: a pre-pass allocates one
        `bytecode.virtual_general_register` per non-entry block argument at the entry block.
        Branch rewriters emit `bytecode.set` ops before each jump to copy values into those registers.
        After conversion, all uses of the original block arguments are replaced with the canonical registers.

        The pre-pass also allocates a shared const-1 register (used by all `cf.cond_br` JE comparisons)
        and one comparison register per `cf.switch` op, both at the entry block.

        The converted function body is placed inside a `bytecode.func_section`. The original
        `func.func` op is erased. String, type, and constant sections are not created by this pass;
        they are inserted by subsequent passes.
    }];

    let constructor = "vpux::bytecode::createConvertHostcodeToBytecodePass()";

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::async::AsyncDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::func::FuncDialect",
        "mlir::math::MathDialect",
        "mlir::memref::MemRefDialect",
        "mlir::tensor::TensorDialect",
        "vpux::bytecode::BytecodeDialect",
        "vpux::Core::CoreDialect"
    ];
}
//
// ConvertLayers2VPU
//

def ConvertLayers2VPU : PassBase<"convert-layers-to-VPU", "vpux::FunctionPass"> {
    let summary = "Convert IE layer operations to VPU dialect for hardware acceleration";

    let description = [{
        AI-generated summary

        Converts IE dialect operations to VPU dialect equivalents via rewrite patterns.
        Uses auto-generated 1:1 conversion patterns for the majority of operations. Special
        cases include IfRewrite (inlines IE.IfOp branches and replaces with
        VPU.ConditionalCopyOp) and LogSoftmaxTopK/PeakRewrite (1:1 conversion of
        pre-decomposed LogSoftmax sub-operations).
    }];

    let constructor = "vpux::createConvertLayers2VPUPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertDynamicQuantToVPUNCE
//

def ConvertDynamicQuantToVPUNCE : PassBase<"convert-dynamic-quant-to-VPU-NCE", "vpux::FunctionPass"> {
    let summary = "Convert IE convolutions with dynamic quantized weights to NCE operations";

    let description = [{
        Converts IE.DynamicDequantize -> IE.Convolution pattern to
        VPU.PopulateWeightTable -> VPU.NCE.Convolution pattern.

        IE.Convolution to VPU.NCE.Convolution lowering is identical to the general case
        (see convert-IE-to-VPU-NCE), but the weights table is generated from runtime scales
        and is represented by VPU.PopulateWeightsTable op, instead of a Const.
    }];

    let constructor = "vpux::createConvertDynamicQuantToVPUNCEPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "vpux::IE::IEDialect"
    ];
}

#endif
