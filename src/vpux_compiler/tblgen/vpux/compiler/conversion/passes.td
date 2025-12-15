//
// Copyright (C) 2022-2025 Intel Corporation.
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
        The pass converts VPUNCE-friendly operations (such as Conv, Pool, Eltwise)

        It doesn't change the memory spaces or add any new operations such as Copy
    }];

    let constructor = "vpux::createConvertIEToVPUNCEPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ConvertIEToVPUM2I
//

def ConvertIEToVPUM2I : PassBase<"convert-IE-to-VPU-M2I", "vpux::FunctionPass"> {
    let summary = "Convert supported IE layers to VPU M2I operations";

    let constructor = "vpux::createConvertIEToVPUM2IPass()";

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
        Replace dynamic operations with their alternative version that can properly handle dynamic tensors.
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

//
// ExpandLayers
//

def ExpandLayers : PassBase<"expand-layers", "vpux::ModulePass"> {
    let summary = "Expand IE Dialect SW Layers to Math dialect";

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
    let summary = "Convert Affine and Std dialects to LLVM dialect";

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
    let summary = "Convert VPUIP Dialect to VPUMI37XX Dialect";

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
// ConvertVPUMI37XX2VPUASM
//

def ConvertVPUMI37XX2VPUASM : PassBase<"convert-VPUMI37XX-to-VPUASM", "vpux::ModulePass"> {
    let summary = "Convert VPUMI37XX dialect to VPUASM";

    let description = [{
        VPUMI37XX is the last level of representation where we still employ op-result-operand relationships.
        After this conversion pass expect only symbolic relationships between the OPS. (with the occasional exception,
        and helped by *VPUASM_SymbolizeValueOp* we can still permit non-symbolic ops to be present in the IR, having
        an explicit conversion OP, that can still be present as information containers).

        Concrete example for DistributetBuffers that have no VPUASM counterpart, we can still hook them into VPUASM.Invariant
        by using SymbolizeValueOp to create a symbol for it.
    }];

    let constructor = "vpux::createConvertVPUMI37XX2VPUASMPass()";
    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect",
        "vpux::VPUMI37XX::VPUMI37XXDialect",
        "vpux::VPURegMapped::VPURegMappedDialect",
        "vpux::VPUASM::VPUASMDialect",
        "vpux::ELFNPU37XX::ELFNPU37XXDialect"
    ];
}

//
// ConvertVPUIP2VPUMI40XX
//

def ConvertVPUIP2VPUMI40XX : PassBase<"convert-VPUIP-to-VPUMI40XX", "vpux::FunctionPass"> {
    let summary = "Convert VPUIP Dialect to VPUMI40XX Dialect";

    let constructor = "vpux::createConvertVPUIP2VPUMI40XXPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect",
        "vpux::VPUMI40XX::VPUMI40XXDialect",
        "vpux::VPURegMapped::VPURegMappedDialect"
    ];

    let options = [
        Option<
            "allocateShaveStackFrames", "allocate-shave-stack-frames",
            "bool", "false",
            "Flag to enable shave stack frame allocation."
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

    let options = [
        Option<
            "enablePWLMOpt", "workload-management-enable",
            "bool", "true",
            "Flag which identifies enablement of partial workload management"
        >
    ];
}

//
// ConvertVPUMI37XX2ELF
//

def ConvertVPUMI37XX2ELF : PassBase<"convert-VPUMI37XX-to-ELF", "vpux::ModulePass"> {
    let summary = "Convert VPUMI37XX Dialect to ELF Dialect";

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
        VPUIPDPU is a logical representation of the DPU config., following the HW functionality specific for each architecture generation.
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
// ConvertVPUASM2NPUReg50XX
//

def ConvertVPUIPDPU2NPUReg50XX : PassBase<"convert-VPUIPDPU-to-NPUReg50XX", "vpux::ModulePass"> {
    let summary = "Convert VPUIPDPU dialect to NPUReg50XX";

    let description = [{
        VPUIPDPU is a logical representation of the DPU config., following the HW functionality specific for each architecture generation.
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
// ConvertLayers2VPU
//

def ConvertLayers2VPU : PassBase<"convert-layers-to-VPU", "vpux::FunctionPass"> {
    let summary = "Convert Layers Operations to VPU Dialect (SW and View layers)";

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
    let summary = "Convert IE.DynamicDequantize -> IE.Convolution pattern to VPU.PopulateWeightTable-> VPU.NCE.Convolution pattern";

    let constructor = "vpux::createConvertDynamicQuantToVPUNCEPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "vpux::IE::IEDialect"
    ];
}

#endif
