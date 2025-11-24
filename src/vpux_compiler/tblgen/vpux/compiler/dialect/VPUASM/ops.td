//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUASM_OPS
#define VPUX_COMPILER_DIALECT_VPUASM_OPS

include "vpux/compiler/dialect/VPUASM/attributes.td"
include "vpux/compiler/dialect/VPUASM/ops_interfaces.td"
include "vpux/compiler/dialect/VPUASM/dialect.td"

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/types.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.td"

include "vpux/compiler/dialect/VPUMI40XX/attributes.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "vpux/compiler/dialect/VPURegMapped/types.td"

//
// Base classes
//

class VPUASM_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPUASM_Dialect,
            mnemonic,
            traits
        >;

class VPUASM_Instruction<string mnemonic, list<Trait> traits = []> :
        VPUASM_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                    DeclareOpInterfaceMethods<ELF_PredefinedPurposeMemoryOpInterface>,
                    Symbol,
                    IsolatedFromAbove
                ]
            )
        > {
}

class VPUASM_Memory<string mnemonic, list<Trait> traits = []> :
        VPUASM_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    IsolatedFromAbove,
                    Symbol
                ]
            )
        > {

}

//Separate the interface that represents the nature that an OP may reference another symbolic OP (aka it has symbolRef's)

//
// Buffers
//

//
// ConstOp
//

def VPUASM_ConstBufferOp :
        VPUASM_Memory<"ConstBuffer",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BufferLocationInterface, ["getMemorySection"]>
            ]
        > {
    let summary = "Constant R/O buffer symbolic declaration";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_BufferTypeAttr:$buffer_type,
        Const_ContentAttr:$content
    );

    let assemblyFormat = [{
        $sym_name
        attr-dict
        $buffer_type `=` custom<ContentAttr>($content)
    }];

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "::mlir::StringAttr":$sym_name, "::vpux::VPUASM::BufferType":$buffer_type,
            "::vpux::Const::ContentAttr":$content
        )>
    ];

}

//
// DeclareBufferOp
//

def VPUASM_DeclareBufferOp :
        VPUASM_Memory<"DeclareBuffer",
            [
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getMemoryOffset", "setMemoryOffset", "getAlignmentRequirements"]>,
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BufferLocationInterface, ["getMemorySection"]>
            ]
        > {
    let summary = "RunTime dynamic R/W buffer symbolic declaration";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_BufferTypeAttr:$buffer_type
    );

    let assemblyFormat = [{
        $sym_name
        attr-dict
        $buffer_type
    }];
}

def VPUASM_DeclareTaskBufferOp :
        VPUASM_Memory<"DeclareTaskBuffer",
            [
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getMemoryOffset", "setMemoryOffset", "getAlignmentRequirements"]>,
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BufferLocationInterface, ["getMemorySection"]>,
                VPUASM_IndexedTaskInterface
            ]
        > {

    let summary = "Buffer Op representing a discrete location of a task in runtime metadata Buffer";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,
        VPURegMapped_TaskTypeAttr:$task_type,
        OptionalAttr<UI64Attr>:$offset
    );

    let assemblyFormat = [{
        $sym_name
        `idx` `(` $task_index `)`
        $task_type
        attr-dict
    }];
}

//
// Ops
//

//
// IOBindings
//

def VPUASM_IOBindingsOp:
        VPUASM_Op<"IOBindings",
            [
                IsolatedFromAbove,
                HasParent<"mlir::ModuleOp">,
                NoRegionArguments
            ]
            #GraphRegionNoTerminator.traits
        > {
    let summary = "Mapped inference level I/O Network description";

    let description = [{
        This operation is bound to the main MLIR module and explicitly encapsulates the I/O information that the network will
        work with at buffer and location level.
    }];

    let regions = (region
        SizedRegion<1>:$inputDeclarations,
        SizedRegion<1>:$outputDeclarations,
        SizedRegion<1>:$profilingBuffDeclarations
    );

    let extraClassDeclaration = [{
        size_t getNetInputsCount();
        vpux::SmallVector<vpux::VPUASM::DeclareBufferOp, 1> getInputDeclarationsOps();

        size_t getNetOutputsCount();
        vpux::SmallVector<vpux::VPUASM::DeclareBufferOp, 1> getOutputDeclarationsOps();

        size_t getProfilingBuffsCount();
        vpux::SmallVector<vpux::VPUASM::DeclareBufferOp, 1> getProfilingBuffDeclarationsOps();

        static VPUASM::IOBindingsOp getFromModule(
            mlir::ModuleOp module
        );
    }];

    let builders = [
        OpBuilder<
            (ins)
        >
    ];

    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        attr-dict
        `inputDeclarations` `:` $inputDeclarations
        `outputDeclarations` `:` $outputDeclarations
        `profilingBuffDeclarations` `:` $profilingBuffDeclarations
    }];
}

//
// MappedInferenceOp_37XX
//

def VPUASM_MappedInferenceOp_37XX :
        VPUASM_Instruction<"MappedInference_37XX",
            [
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>
            ]
        > {
    let summary = "Task representing the MappedInference structure on 37XX";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<SymbolRefArrayAttr>:$dmaTasks,
        OptionalAttr<SymbolRefAttr>:$invariantTasks,
        OptionalAttr<SymbolRefAttr>:$variantTasks,
        OptionalAttr<SymbolRefAttr>:$actKernelRanges,
        OptionalAttr<SymbolRefAttr>:$actKernelInvocations,
        OptionalAttr<SymbolRefAttr>:$barrierTasks,
        OptionalAttr<SymbolRefAttr>:$actShaveRt,
        OptionalAttr<SymbolRefArrayAttr>:$actShaveStacks,
        I64ArrayAttr:$dmaCount,
        UI32Attr:$invariantCount,
        UI32Attr:$variantCount,
        UI32Attr:$actKernelRangesCount,
        UI32Attr:$actKernelInvocationsCount,
        UI32Attr:$barrierCount
    );

    let assemblyFormat = [{
    attr-dict
        $sym_name `:`
        (`dmas` `(` $dmaTasks^ `)`)?
        (`invariants` `(` $invariantTasks^ `)`)?
        (`variants` `(` $variantTasks^ `)`)?
        (`actKernelRanges` `(` $actKernelRanges^ `)`)?
        (`actKernelInvocations` `(` $actKernelInvocations^ `)`)?
        (`barriers` `(` $barrierTasks^ `)`)?
        (`actShaveRt` `(` $actShaveRt^ `)`)?
        (`actShaveStacks` `(` $actShaveStacks^ `)`)?
        `dmaCount` `(` $dmaCount `)`
        `invariantCount` `(` $invariantCount `)`
        `variantCount` `(` $variantCount `)`
        `actKernelRangesCount` `(` $actKernelRangesCount `)`
        `actKernelInvocationsCount` `(` $actKernelInvocationsCount `)`
        `barrierCount` `(` $barrierCount `)`
    }];
}

//
// MappedInferenceOp
//

def VPUASM_MappedInferenceOp :
        VPUASM_Instruction<"MappedInference",
            [
            ]
        > {
    let summary = "Task representing the MappedInference structure";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<SymbolRefArrayOfArraysAttr>:$dmaTasks,
        OptionalAttr<SymbolRefArrayAttr>:$invariantTasks,
        OptionalAttr<SymbolRefArrayAttr>:$variantTasks,
        OptionalAttr<SymbolRefArrayAttr>:$actKernelRanges,
        OptionalAttr<SymbolRefArrayAttr>:$actKernelInvocations,
        OptionalAttr<SymbolRefAttr>:$mediaTasks,
        OptionalAttr<SymbolRefAttr>:$barrierTasks,
        OptionalAttr<SymbolRefAttr>:$actShaveRt,
        OptionalAttr<SymbolRefArrayAttr>:$actShaveStacks,
        OptionalAttr<SymbolRefAttr>:$managedMappedInference,
        I64ArrayOfArraysAttr:$dmaCount,
        I64ArrayAttr:$invariantCount,
        I64ArrayAttr:$variantCount,
        I64ArrayAttr:$actKernelRangesCount,
        I64ArrayAttr:$actKernelInvocationsCount,
        I64Attr:$mediaCount,
        I64Attr:$barrierCount,
        OptionalAttr<SymbolRefAttr>:$dmaHwpBase,
        OptionalAttr<SymbolRefAttr>:$hwpWorkpointCfg,
        SymbolRefAttr:$mappedInferenceVersion
    );

    let assemblyFormat = [{
    attr-dict
        $sym_name `:`
        (`dmas` `(` $dmaTasks^ `)`)?
        (`invariants` `(` $invariantTasks^ `)`)?
        (`variants` `(` $variantTasks^ `)`)?
        (`actKernelRanges` `(` $actKernelRanges^ `)`)?
        (`actKernelInvocations` `(` $actKernelInvocations^ `)`)?
        (`mediaTasks` `(` $mediaTasks^ `)`)?
        (`barriers` `(` $barrierTasks^ `)`)?
        (`actShaveRt` `(` $actShaveRt^ `)`)?
        (`actShaveStacks` `(` $actShaveStacks^ `)`)?
        (`managedMappedInference` `(` $managedMappedInference^ `)`)?
        `dmaCount` `(` $dmaCount `)`
        `invariantCount` `(` $invariantCount `)`
        `variantCount` `(` $variantCount `)`
        `actKernelRangesCount` `(` $actKernelRangesCount `)`
        `actKernelInvocationsCount` `(` $actKernelInvocationsCount `)`
        `mediaCount` `(` $mediaCount `)`
        `barrierCount` `(` $barrierCount `)`
        `mappedInferenceVersion` `(` $mappedInferenceVersion `)`
        (`dmaHwpBase` `(` $dmaHwpBase^ `)`)?
        (`hwpWorkpointCfg` `(` $hwpWorkpointCfg^ `)`)?
    }];
}

def VPUASM_NNrtConfigOp :
        VPUASM_Instruction<"nnrtConfig",
            [
            ]
        > {
    let summary = "Task representing the MappedInference structure";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        UnitAttr:$isActKernelInvocations,
        OptionalAttr<SymbolRefAttr>:$actShaveRt,
        OptionalAttr<SymbolRefArrayAttr>:$actShaveStacks,
        OptionalAttr<SymbolRefAttr>:$dmaHwpBase,
        OptionalAttr<SymbolRefAttr>:$hwpWorkpointCfg
    );

    let assemblyFormat = [{
    attr-dict
        $sym_name `:`
        (`dmaHwpBase` `(` $dmaHwpBase^ `)`)?
        (`hwpWorkpointCfg` `(` $hwpWorkpointCfg^ `)`)?
    }];
}



//
// ManagedMappedInferenceOp
//

def VPUASM_ManagedMappedInferenceOp :
        VPUASM_Instruction<"ManagedMappedInference",[]
        > {

    let summary = [{Task representing the alternative "Managed Mapped Inference" data structure}];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<SymbolRefArrayOfArraysAttr>:$dmaTasks,
        OptionalAttr<SymbolRefAttr>:$workItems,
        OptionalAttr<SymbolRefAttr>:$barrierTasks,
        OptionalAttr<SymbolRefAttr>:$bootstrapBarriers,
        SymbolRefAttr:$nnrtConfig,
        OptionalAttr<SymbolRefAttr>:$barrierConfigurationTasks,
        OptionalAttr<SymbolRefAttr>:$numOfBarrierReprogrammings,

        I64ArrayOfArraysAttr:$dmaCount,
        I64Attr:$workItemsCount,
        I64Attr:$barrierCount,
        IntAttr:$final_barrier_id,
        I64Attr:$bootstrapBarriersCount,
        I64Attr:$bootsrapWorkItemsCount,
        I64Attr:$barrierConfigurationTasksCount,
        I64Attr:$barriersReprogrammingCount,
        I64Attr:$barrierConfigurationStride,
        UI8Attr:$actshv_used,
        UI8Attr:$dpu_used,
        UI8Attr:$media_used,
        UI8Attr:$dma_from_ddr_used,
        UI8Attr:$dma_from_cmx_used,
        SymbolRefAttr:$mappedInferenceVersion,
        VPURegMapped_WorkloadManagementBarrierProgrammingModeAttr:$workloadManagementBarrierProgrammingMode,
        BoolAttr:$disableDmaSwFifo
        );

    let assemblyFormat = [{
        $sym_name `:`
        `dmas`  $dmaTasks
        `workItems` `(` $workItems `)`
        `barrierTasks` `(` $barrierTasks `)`
        (`bootstrapBarriers` `(` $bootstrapBarriers^ `)`)?
        `nnrtConfig` `(` $nnrtConfig `)`
        `mappedInferenceVersion` `(` $mappedInferenceVersion `)`
        (`barrierConfigurationTasks` `(` $barrierConfigurationTasks^ `)`)?
        (`numOfBarrierReprogrammings` `(` $numOfBarrierReprogrammings^ `)`)?

        attr-dict
    }];
}

//
// ConfigureBarrierOp
//

def VPUASM_ConfigureBarrierOp :
        VPUASM_Instruction<"ConfigureBarrier",
            [
                VPUASM_IndexedTaskInterface
            ]
        > {
    let summary = "A task to configure the setup for a barrier";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        OptionalAttr<VPUASM_IndexTypeAttr>:$work_item_idx,
        UI8Attr:$id,
        SI64Attr:$next_same_id,
        UI8Attr:$producer_count,
        UI8Attr:$consumer_count
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        $sym_name
        `idx` `(` $task_index `)`
        (`workItemIdx` `(` $work_item_idx^ `)` )?
        `(` $id `)`
        `=` `>` `(` $next_same_id `)`
        `counts` `(` $producer_count `:` $consumer_count `)`
        attr-dict
    }];
}

//
// BootstrapOp
//

def VPUASM_BootstrapOp :
        VPUASM_Instruction<"Bootstrap",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                VPUASM_IndexedTaskInterface
            ]
        > {
    let summary = "A task to configure the setup for a barrier";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        UI32Attr:$barrier_id
    );

    let assemblyFormat = [{
        $sym_name
        attr-dict
    }];
}


//
// ManagedBarrierOp
//

def VPUASM_ManagedBarrierOp :
        VPUASM_Instruction<"ManagedBarrier",
            [
                VPUASM_IndexedTaskInterface
            ]
        > {
    let summary = "A task to configure the setup for a barrier, with workItem attached";
    let description = [{Virtually the same as ConfigureBarrierOp, however since this will be managed barrier, it will have
                        a different memory footprint in the serialized form. Intended as temporary until we can figure out how
                        to unify representations}];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        OptionalAttr<VPUASM_IndexTypeAttr>:$work_item_idx,
        DefaultValuedAttr<UI32Attr, "0">:$work_item_count,
        UI8Attr:$id,
        SI64Attr:$next_same_id,
        UI8Attr:$producer_count,
        UI8Attr:$consumer_count
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        $sym_name
        `idx` `(` $task_index `)`
        (`workItemIdx` `(` $work_item_idx^ `)` )?
        `(` $id `)`
        `=` `>` `(` $next_same_id `)`
        `counts` `(` $producer_count `:` $consumer_count `)`
        attr-dict
    }];
}

//
// WorkItemOp
//

def VPUASM_WorkItemOp :
        VPUASM_Instruction<"WorkItem",
            [
                VPUASM_IndexedTaskInterface
            ]
        > {

    let summary = "Work Item Task";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index, //workitem index in the list of tasks
        VPUASM_IndexTypeAttr:$real_task_index, // index of the actual task dpu or dma or shave
        OptionalAttr<VPUASM_IndexTypeAttr>:$next_workitem_idx, // next linked workitem idx on the same barrier
        VPURegMapped_TaskTypeAttr:$task_type,
        SymbolRefAttr:$first_task,
        I64Attr:$task_count
    );

    let assemblyFormat = [{
        $sym_name
        `idx` `(` $task_index `)`
        `real_task_index` `(` $real_task_index `)`
        (`next_workitem_idx` `(` $next_workitem_idx^ `)` )?
        `task_type` `(` $task_type `)`
        `first_task` `(` $first_task `)`
        `task_count` `(` $task_count `)`
        attr-dict
    }];
}

//
// NNDMAOp
//

def VPUASM_NNDMAOp :
        VPUASM_Instruction<"NNDMA",
            [
                DeclareOpInterfaceMethods<ELF_DDRMemoryAccessingOpInterface>,
                VPUASM_IndexedTaskInterface
            ]
        > {
    let summary = "NN DMA task";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        OptionalAttr<SymbolRefAttr>:$task_location,
        OptionalAttr<SymbolRefAttr>:$next_link,
        SymbolRefAttr:$input,
        SymbolRefArrayAttr:$output_buffs,
        UI8ArrayAttr:$waitBarriers,
        UI8ArrayAttr:$updateBarriers,
        UI64Attr:$start_after,
        UI64Attr:$clean_after,
        DefaultValuedAttr<VPUIP_DMAAccModeAttr, "VPUIP::DMAAccMode::DISABLE">:$acceleration_mode,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        UnitAttr:$enable_msc,
        OptionalAttr<SymbolRefAttr>:$act_compression_size_entry,
        OptionalAttr<SymbolRefAttr>:$act_compression_sparsity_map,
        OptionalAttr<VPUMI40XX_DMATransactionAttr>:$dma_transaction,
        // dma_descriptor is now optional to allow DMA refactoring to be completed on NPU4
        // Its functionality is now partially replaced by dma_transaction
        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<I64ArrayAttr>:$tile_indexes,
        OptionalAttr<SymbolRefAttr>:$indices,
        OptionalAttr<VPUIP_GatherAddressingMode>:$addressing_mode
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$src, "mlir::Value":$dst)
        >
    ];

    let extraClassDeclaration = [{
        static vpux::VPURegMapped::TaskType getTaskType() {
            return vpux::VPURegMapped::TaskType::DMA;
        }
    }];

    let assemblyFormat = [{
        $sym_name
        `idx` `(` $task_index `)`
        (`taskLocation` `(` $task_location^ `)`)?
        (`links` `(` $next_link^ `)`)?
        `input` `(` $input `)`
        `outputs` `(` $output_buffs `)`
        `waits` `(` $waitBarriers `)`
        `updates` `(` $updateBarriers `)`
        `start_after` `(` $start_after `)`
        `clean_after` `(` $clean_after `)`
        (`dma_transaction` `(` $dma_transaction^ `)`)?
        (`dma_descriptor` `(` $dma_descriptor^ `)`)?
        `acceleration_mode` `(` $acceleration_mode `)`
        (`act_compression_size_entry` `(` $act_compression_size_entry^ `)`)?
        (`act_compression_sparsity_map` `(` $act_compression_sparsity_map^ `)`)?
        (`is_out_of_order` `(` $is_out_of_order^ `)`)?
        (`is_critical` `(` $is_critical^ `)`)?
        (`tile_indexes` `(` $tile_indexes^ `)`)?
        (`indices` `(` $indices^ `)`)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// Activation Kernel Ops
//

//
// ActShaveRtOp
//

def VPUASM_ActShaveRtOp :
        VPUASM_Op<"ActShaveRt",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_PredefinedPurposeMemoryOpInterface>,
                Symbol,
                IsolatedFromAbove
            ]
        > {
    let summary = "Declaration of Act Shave Management Kernel";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        StrAttr:$kernel_path
    );

    let assemblyFormat = [{
        $sym_name attr-dict `kernel` `(` $kernel_path `)`
    }];
}

//
// DeclareKernelTextOp
//

def VPUASM_DeclareKernelTextOp :
        VPUASM_Instruction<"DeclareKernelText",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>
            ]
        > {
    let summary = "Declaration of Software Kernel .text ";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        StrAttr:$kernel_path
    );

    let assemblyFormat = [{
        $sym_name attr-dict `:` $kernel_path
    }];
}


//
// DeclareKernelDataOp
//


def VPUASM_DeclareKernelDataOp :
        VPUASM_Instruction<"DeclareKernelData",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>
            ]
        > {
    let summary = "Declaration of Software Kernel .data";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        StrAttr:$kernel_path
    );

    let assemblyFormat = [{
        $sym_name attr-dict `:` $kernel_path
    }];
}

//
// ShaveStackFrameOp
//

def VPUASM_ShaveStackFrameOp :
        VPUASM_Memory<"ActShaveRtStack",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySizeCached", "getAlignmentRequirements"]>,
                DeclareOpInterfaceMethods<ELF_PredefinedPurposeMemoryOpInterface>,
                DeclareOpInterfaceMethods<ELF_BufferLocationInterface, ["getMemorySection"]>
            ]
        > {
    let summary = "Declaration of Act Shave stack frames buffers";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        UI64Attr:$stackSize
    );

    let assemblyFormat = [{
         $sym_name attr-dict `:` $stackSize
    }];

}

//
// DeclareKernelEntryOp
//

def VPUASM_DeclareKernelEntryOp :
        VPUASM_Op<"DeclareKernelEntry",
            [
                IsolatedFromAbove
            ]
        > {
    let summary = "Declaration of Kernel Entry";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        StrAttr:$kernel_path
    );

    let assemblyFormat = [{
        $sym_name attr-dict `:` $kernel_path
    }];

    let extraClassDeclaration = [{
        uint32_t getKernelEntry();
    }];

}

//
// KernelParamsOp
//

def VPUASM_KernelParamsOp :
        VPUASM_Instruction<"KernelParams",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serializeCached"]>,
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySizeCached", "getAlignmentRequirements"]>,
                DeclareOpInterfaceMethods<ELF_DDRMemoryAccessingOpInterface>
            ]
        > {
    let summary = "Kernel Params";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        SymbolRefArrayAttr:$inputs,
        SymbolRefArrayAttr:$outputs,
        SymbolRefArrayAttr:$dynamicInputShapes,
        SymbolRefArrayAttr:$dynamicOutputShapes,
        StrAttr:$kernel_type,
        SmallVectorU8Prop:$kernel_params,
        UnitAttr:$is_output_broadcasted,
        UnitAttr:$isJitCompiled
    );

    let builders = [
        OpBuilder<(ins
            "mlir::StringAttr":$sym_name,
            "mlir::ArrayAttr":$inputs,
            "mlir::ArrayAttr":$outputs,
            "mlir::ArrayAttr":$dynamicInputShapes,
            "mlir::ArrayAttr":$dynamicOutputShapes,
            "mlir::StringAttr":$kernel_type,
            "SmallVector<uint8_t>&&":$kernel_params,
            "mlir::UnitAttr":$is_output_broadcasted,
            "mlir::UnitAttr":$isJitCompiled
        )>
    ];

    let extraClassDeclaration = [{
        size_t getParamsStructSize();
    }];

    let assemblyFormat = [{
        $sym_name
        `inputs` `(` $inputs `)`
        `outputs` `(` $outputs `)`
        `dynamicInputShapes` `(` $dynamicInputShapes `)`
        `dynamicOutputShapes` `(` $dynamicOutputShapes `)`
        `kernel_type` `(` $kernel_type `)`
        attr-dict
        prop-dict
    }];
}

//
// ActKernelRangeOp
//

def VPUASM_ActKernelRangeOp :
        VPUASM_Instruction<"ActKernelRange",
            [
                VPUASM_IndexedTaskInterface
            ]
        > {
    let summary = "Activation Kernel Range";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        OptionalAttr<SymbolRefAttr>:$task_location,
        OptionalAttr<SymbolRefAttr>:$kernel_text,
        OptionalAttr<SymbolRefAttr>:$kernel_entry,
        SymbolRefAttr:$kernelTaskType
    );

    let assemblyFormat = [{
        $sym_name
        `idx` `(` $task_index `)`
        (`taskLocation` `(` $task_location^ `)`)?
        `kernelTaskType` `(` $kernelTaskType `)`
        (`calls` $kernel_text^ )?
        (`:` $kernel_entry^ )?
        attr-dict
    }];
}

//
// ActKernelInvocationOp
//

def VPUASM_ActKernelInvocationOp :
        VPUASM_Instruction<"ActKernelInvocation",
            [
                VPUASM_IndexedTaskInterface
            ]
        > {
    let summary = "Activation Kernel Invocation";

    let description = [{
        next_link attributes refers to next invocation in the linked list. Since at least non-WLM
        does not distinguish linked lists for invocations range inside mapped inference and having
        2 linked lists of invocations per tile, next_link for ActKernelInvocationOp refers not to
        immediately next task, but one after that if exists (round-robin distribution:
        invo#0 is linked to invo#2, invo#1 is linked to invo#3, invo#2 is linked to invo#4, etc)
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        OptionalAttr<SymbolRefAttr>:$task_location,
        OptionalAttr<SymbolRefAttr>:$next_link,
        SymbolRefAttr:$kernel_range,
        OptionalAttr<SymbolRefAttr>:$kernel_data,
        SymbolRefAttr:$kernel_params,
        UI8ArrayAttr:$waitBarriers,
        UI8ArrayAttr:$updateBarriers,
        OptionalAttr<SymbolRefAttr>:$profiling_data,
        DefaultValuedAttr<UI64Attr, "0">:$tile,
        UI64Attr:$start_after,
        UI64Attr:$clean_after,
        UI64Attr:$range_index
    );

    let extraClassDeclaration = [{
        static vpux::VPURegMapped::TaskType getTaskType() {
            return vpux::VPURegMapped::TaskType::ActKernelInvocation;
        }
    }];

    let assemblyFormat = [{
        $sym_name
        `idx` `(` $task_index `)`
        `taskLocation` `(` $task_location `)`
        `->` $kernel_range `(` (`kernel_data` `:` $kernel_data^ `,`)? `kernel_params` `:` $kernel_params  `)`
        (`profiling_data` `(` $profiling_data^ `)`)?
        `waits` `(` $waitBarriers `)`
        `updates` `(` $updateBarriers `)`
        `tile` `(` $tile `)`
        `start_after` `(` $start_after `)`
        `clean_after` `(` $clean_after `)`
        `range_index` `(` $range_index `)`
        attr-dict
    }];
}

//
// DPU Variant Op
//

def VPUASM_DPUVariantOp :
        VPUASM_Instruction<"DPUVariant",
            [
                VPUASM_IndexedTaskInterface
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "DPU Variant Op";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        OptionalAttr<SymbolRefAttr>:$task_location,
        OptionalAttr<SymbolRefAttr>:$next_link,
        SymbolRefAttr:$invariant,
        SymbolRefAttr:$invariant_task_location,

        OptionalAttr<SymbolRefAttr>:$weights,

        OptionalAttr<SymbolRefAttr>:$weight_table,
        OptionalAttr<SymbolRefAttr>:$weight_table_data_ptr,
        OptionalAttr<SymbolRefAttr>:$weight_table_sp_ptr,
        OptionalAttr<SymbolRefAttr>:$weight_table_scale,
        OptionalAttr<SymbolRefAttr>:$weight_table_bias,
        OptionalAttr<SymbolRefAttr>:$weight_zero_points,

        VPUIP_NCETaskTypeAttr:$nce_task_type,
        I64ArrayAttr:$inStart,
        I64ArrayAttr:$inEnd,
        I64ArrayAttr:$start,
        I64ArrayAttr:$end,
        VPU_PaddingAttr:$pad,
        VPU_MPEModeAttr:$mpe_mode,
        OptionalAttr<IntAttr>:$cluster_id,
        OptionalAttr<DPUHaloRegionArrayAttr>: $haloRegions,
        OptionalAttr<IntAttr>:$workload_id,
        UnitAttr:$spr_lut_read,
        UnitAttr:$pallet_lut_read,
        UnitAttr:$force_inv_read,
        OptionalAttr<I64ArrayAttr>:$variant_primitive_id // IDs for variant primitives A and B
    );

    let assemblyFormat = [{
        $sym_name
        `idx` `(` $task_index `)`
        `taskLocation` `(` $task_location `)`
        (`nextLink` `(` $next_link^ `)`)?
        `invariant` $invariant
        `calls` $invariant_task_location
        (`weights` `(` $weights^ `)`)?
        (`weight_table` `(` $weight_table^ `)`)?
        (`weight_table_data_ptr` `(` $weight_table_data_ptr^ `)`)?
        (`weight_table_sp_ptr` `(` $weight_table_sp_ptr^ `)`)?
        (`weight_table_scale` `(` $weight_table_scale^ `)`)?
        (`weight_table_bias` `(` $weight_table_bias^ `)`)?
        (`weight_zero_points` `(` $weight_zero_points^ `)`)?
        attr-dict
    }];
}

def VPUASM_PPETaskOp : VPU_PPETaskOpBase<
    VPUASM_Dialect,
    "PPETask",
    [ParentOneOf<["vpux::VPUASM::DPUInvariantOp", "vpux::VPUASM::DPUInvariantOp_37XX"]>]
> {}

//
// DPU Invariant Op
//

def VPUASM_DPUInvariantOp :
        VPUASM_Instruction<"DPUInvariant",
            [
                VPUASM_IndexedTaskInterface
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "DPU Invariant Op";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        OptionalAttr<SymbolRefAttr>:$task_location,
        OptionalAttr<SymbolRefAttr>:$input,
        OptionalAttr<SymbolRefAttr>:$input_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$input_storage_element_table,

        OptionalAttr<SymbolRefAttr>:$weights,
        OptionalAttr<SymbolRefAttr>:$weights_sparsity_map,

        OptionalAttr<SymbolRefAttr>:$weight_table,
        OptionalAttr<SymbolRefAttr>:$weight_table_data_ptr,
        OptionalAttr<SymbolRefAttr>:$weight_table_sp_ptr,
        OptionalAttr<SymbolRefAttr>:$weight_table_scale,
        OptionalAttr<SymbolRefAttr>:$weight_table_bias,
        OptionalAttr<SymbolRefAttr>:$weight_zero_points,

        OptionalAttr<SymbolRefAttr>:$spr_lookup_table,
        OptionalAttr<SymbolRefAttr>:$pallet_lookup_table,

        OptionalAttr<SymbolRefAttr>:$output,
        OptionalAttr<SymbolRefAttr>:$output_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$profiling_data,
        OptionalAttr<SymbolRefAttr>:$dynamic_sequence_length,

        OptionalAttr<SymbolRefAttr>:$max_per_xy,
        OptionalAttr<SymbolRefAttr>:$min_per_xy,
        OptionalAttr<SymbolRefArrayAttr>:$min_max_per_tensor,


        OptionalAttr<VPUASM_BufferTypeAttr>:$output_type_continued,

        UI8ArrayAttr:$waitBarriers,
        UI8ArrayAttr:$updateBarriers,

        VPUIP_NCETaskTypeAttr:$nce_task_type,
        OptionalAttr<VPU_EltwiseTypeAttr>:$eltwise_type,
        VPU_MPEModeAttr:$mpe_frequent_mode,
        OptionalAttr<VPU_MPEEngineAttr>:$mpe_engine,

        OptionalAttr<I64ArrayAttr>:$kernel_size,
        OptionalAttr<I64ArrayAttr>:$kernel_strides,
        OptionalAttr<VPU_PaddingAttr>:$kernel_padding,
        UnitAttr:$is_continued,
        OptionalAttr<IntAttr>:$cm_sp_pattern,
        UnitAttr:$input_channels_compression,
        UnitAttr:$is_zero_offset_weights_table,
        OptionalAttr<IntAttr>:$out_channel_offset,
        UnitAttr:$is_superdense,
        OptionalAttr<BoolAttr>:$is_inplace,
        OptionalAttr<IntAttr>:$input_se_size,
        OptionalAttr<IntAttr>:$output_se_size,
        UnitAttr:$is_permute_quantize,
        UnitAttr:$is_small_kernel_optimized,
        OptionalAttr<VPUIP_DynamicScaleConfigAttr>:$dynamic_scale_config,
        OptionalAttr<VPUIP_LocalRegionAttr>:$local_region,

        UI64Attr:$start_after,
        UI64Attr:$clean_after,

        UI64Attr:$variant_count,
        UI32Attr:$first_variant_index,
        UI32Attr:$last_variant_index
    );

    let regions = (region
        SizedRegion<1>:$ppe
    );

    let extraClassDeclaration = [{
        static vpux::VPURegMapped::TaskType getTaskType() {
            return vpux::VPURegMapped::TaskType::DPUInvariant;
        }
    }];

    let assemblyFormat = [{
        $sym_name
        `idx` `(` $task_index `)`
        `taskLocation` `(` $task_location `)`
        (`input` `(` $input^ `)`)?
        (`input_sparsity_map` `(` $input_sparsity_map^ `)`)?
        (`input_storage_element_table` `(` $input_storage_element_table^ `)`)?
        (`weights` `(` $weights^ `)`)?
        (`weights_sparsity_map` `(` $weights_sparsity_map^ `)`)?
        (`weight_table` `(` $weight_table^ `)`)?
        (`weight_table_data_ptr` `(` $weight_table_data_ptr^ `)`)?
        (`weight_table_sp_ptr` `(` $weight_table_sp_ptr^ `)`)?
        (`weight_table_scale` `(` $weight_table_scale^ `)`)?
        (`weight_table_bias` `(` $weight_table_bias^ `)`)?
        (`weight_zero_points` `(` $weight_zero_points^ `)`)?
        (`spr_lookup_table` `(` $spr_lookup_table^ `)`)?
        (`pallet_lookup_table` `(` $pallet_lookup_table^ `)`)?
        (`output` `(` $output^ `)`)?
        (`output_sparsity_map` `(` $output_sparsity_map^ `)`)?
        (`profiling_data` `(` $profiling_data^ `)`)?
        (`max_per_xy` `(` $max_per_xy^ `)`)?
        (`min_per_xy` `(` $min_per_xy^ `)`)?
        (`min_max_per_tensor` `(` $min_max_per_tensor^ `)`)?
        `waits` `(` $waitBarriers `)`
        `updates` `(` $updateBarriers `)`
        attr-dict
        `PPE` `:` $ppe
    }];
}

//
// DPU Variant Op 37XX
//

def VPUASM_DPUVariantOp_37XX :
        VPUASM_Instruction<"DPUVariant_37XX",
            [
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                VPUASM_IndexedTaskInterface
            ]
        > {

    let summary = "DPU Variant Op_37XX";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        OptionalAttr<SymbolRefAttr>:$task_location,
        SymbolRefAttr:$invariant,
        I64ArrayAttr:$start,
        I64ArrayAttr:$end,
        VPU_PaddingAttr:$pad,
        VPU_MPEModeAttr:$mpe_mode,
        OptionalAttr<IntAttr>:$cluster_id
    );

    let assemblyFormat = [{
        $sym_name
        `idx` `(` $task_index `)`
        `taskLocation` `(` $task_location `)`
        `calls` $invariant
        attr-dict
    }];
}

//
// DPU Invariant Op 37XX
//

def VPUASM_DPUInvariantOp_37XX :
        VPUASM_Instruction<"DPUInvariant_37XX",
            [
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                VPUASM_IndexedTaskInterface
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "DPU Invariant Op 37XX";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        OptionalAttr<SymbolRefAttr>:$task_location,
        SymbolRefAttr:$input,
        OptionalAttr<SymbolRefAttr>:$input_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$input_storage_element_table,

        OptionalAttr<SymbolRefAttr>:$weights,
        OptionalAttr<SymbolRefAttr>:$weights_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$weight_table,

        //Parent I/O's needed until we fully integrate VPUIPDPU
        SymbolRefAttr:$parent_input,
        OptionalAttr<SymbolRefAttr>:$parent_input_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$parent_input_storage_element_table,

        SymbolRefAttr:$parent_output,

        SymbolRefArrayAttr:$output_buffs,

        OptionalAttr<SymbolRefAttr>:$profiling_data,

        UI8ArrayAttr:$waitBarriers,
        UI8ArrayAttr:$updateBarriers,

        VPUIP_NCETaskTypeAttr:$nce_task_type,
        OptionalAttr<VPU_EltwiseTypeAttr>:$eltwise_type,
        VPU_MPEModeAttr:$mpe_frequent_mode,

        OptionalAttr<I64ArrayAttr>:$kernel_size,
        OptionalAttr<I64ArrayAttr>:$kernel_strides,
        OptionalAttr<VPU_PaddingAttr>:$kernel_padding,
        UnitAttr:$is_continued,
        OptionalAttr<IntAttr>:$cm_sp_pattern,
        UnitAttr:$input_channels_compression,
        UnitAttr:$is_zero_offset_weights_table,
        UnitAttr:$is_segmented,
        OptionalAttr<IntAttr>:$out_channel_offset,

        UI64Attr:$start_after,
        UI64Attr:$clean_after
    );

    //PPE needed until we fully integrate VPUIPDPU
    let regions = (region
        SizedRegion<1>:$ppe

    );

    let extraClassDeclaration = [{
        static vpux::VPURegMapped::TaskType getTaskType() {
            return vpux::VPURegMapped::TaskType::DPUInvariant;
        }
    }];

    let assemblyFormat = [{
        $sym_name
        `idx` `(` $task_index `)`
        `taskLocation` `(` $task_location `)`
        `input` `(` $input `)`
        (`input_sparsity_map` `(` $input_sparsity_map^ `)`)?
        (`input_storage_element_table` `(` $input_storage_element_table^ `)`)?
        (`weights` `(` $weights^ `)`)?
        (`weights_sparsity_map` `(` $weights_sparsity_map^ `)`)?
        (`weight_table` `(` $weight_table^ `)`)?
        `parent_input` `(` $parent_input `)`
        (`parent_input_sparsity_map` `(` $parent_input_sparsity_map^ `)`)?
        (`parent_input_storage_element_table` `(` $parent_input_storage_element_table^ `)`)?
        `parent_output` `(` $parent_output `)`
        `outputs` `(` $output_buffs `)`
        (`profiling_data` `(` $profiling_data^ `)`)?
        `waits` `(` $waitBarriers `)`
        `updates` `(` $updateBarriers `)`
        attr-dict
        `PPE` `:` $ppe
    }];
}

//
// SymbolizeValueOp
//

def VPUASM_SymbolizeValueOp :
    VPUASM_Op<"SymbolizeValueOp",
        [
            Symbol
        ]
    > {

    let summary = "Symbolize any SSA value";

    let description = [{
        In cases where mixing operations that return SSA values into symbolic use-def-chains.
        Basically this op is  a "materialization op" that materializez any type into a "Symbolic type"
    }];

    let arguments = (ins
        AnyType:$input,
        SymbolNameAttr:$sym_name
    );

    let assemblyFormat = [{
        $sym_name `(` $input `:` type($input) `)` attr-dict
    }];
}

//
// Network Metadtata Op
//

def VPUASM_NetworkMetadataOp :
        VPUASM_Op<"NetworkMetadata",
            [
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                VPUASM_IndexedTaskInterface,
                IsolatedFromAbove,
                Symbol
            ]
        > {
    let summary = "Network Metadata Op";

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let extraClassDeclaration = [{
        void serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection, ::elf::NetworkMetadata& metadata);
    }];

    let assemblyFormat = [{
        $sym_name
        attr-dict
    }];
}

//
// Profiling Metadata Op
//

def VPUASM_ProfilingMetadataOp :
        VPUASM_Op<"ProfilingMetadata",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                IsolatedFromAbove,
                Symbol
            ]
        > {
    let summary = "Profiling Metadata Op";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        ElementsAttr:$metadata
    );

    let assemblyFormat = [{
        $sym_name
        attr-dict
    }];
}

//
// M2IOp
//

def VPUASM_M2IOp :
        VPUASM_Instruction<"M2I",
            [
                VPUASM_IndexedTaskInterface
            ]
        > {
    let summary = "M2I task op";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        OptionalAttr<SymbolRefAttr>:$taskLocation,
        OptionalAttr<SymbolRefAttr>:$next_link,
        SymbolRefAttr:$input,
        SymbolRefAttr:$output_buff,
        OptionalAttr<SymbolRefAttr>:$profiling_data,

        UnitAttr:$do_csc,
        UnitAttr:$do_norm,
        VPU_M2iColorFmtAttr:$inFmt,
        VPU_M2iColorFmtAttr:$outFmt,

        UnitAttr:$chroma_in_reverse_channels,
        UnitAttr:$chroma_out_reverse_channels,
        UnitAttr:$luma_in_reverse_channels,
        UnitAttr:$luma_out_reverse_channels,
        UI32Attr:$scale_factor_x,
        UI32Attr:$scale_factor_y,
        OptionalAttr<F64ArrayAttr>:$norm,
        OptionalAttr<UI32Attr>:$tile_offset_x,
        OptionalAttr<UI32Attr>:$tile_offset_y,
        DefaultValuedAttr<VPU_M2iInterpAttr, "vpux::VPU::M2iInterp::NEAREST">:$interp,

        UI8ArrayAttr:$waitBarriers,
        UI8ArrayAttr:$updateBarriers,
        UI64Attr:$start_after,
        UI64Attr:$clean_after
    );

    let extraClassDefinition = [{
        static vpux::VPURegMapped::TaskType getTaskType() {
            return vpux::VPURegMapped::TaskType::M2I;
        }
    }];

    let assemblyFormat = [{
         $sym_name
        `idx` `(` $task_index `)`
        (`taskLocation` `(` $taskLocation^ `)`)?
        (`links` `(` $next_link^ `)`)?
        `inputs` `(` $input `)`
        `outputs` `(` $output_buff `)`
        (`profiling_data` `(` $profiling_data^ `)`)?
        attr-dict
    }];
}

//
// MappedInferenceVersionOp
//

def MappedInferenceVersionOp :
        VPUASM_Op<"MappedInferenceVersion",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                Symbol
            ]
        > {
    let summary = "Represents a versioning mechanism for Mapped Inference. Only trivially defined at this level.";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        UI32Attr:$major,
        UI32Attr:$minor,
        UI32Attr:$patch
    );

    let assemblyFormat = [{
        $sym_name
        `(` $major `_` $minor `_` $patch `)`
        attr-dict
        }];
}

//
// PlatformInfoOp
//

def PlatformInfoOp :
        VPUASM_Op<"PlatformInfo",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                Symbol
            ]
        > {
    let summary = "Represents an op describing various platform information, such as arch, revision, etc.";

    let description = [{
        PlatformInfoOp is an operation that is meant to encapsulate various information about the NPU. For
        the moment, this information is limited to the archKind (taken from the module), but can
        easily be extended with other information such as SKU info, etc.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let assemblyFormat = [{
        attr-dict
    }];

    let builders = [
        OpBuilder<(ins)>,
    ];
}

//
// CompilerHashOp
//

def CompilerHashOp :
        VPUASM_Op<"CompilerHash",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                Symbol
            ]
        > {
    let summary = "Represents a versioning mechanism for the latest compiler hash value";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        StrAttr:$compiler_hash
    );

    let extraClassDeclaration = [{
        std::vector<uint8_t> getSerializedCompilerHash();
    }];

    let assemblyFormat = [{
        $sym_name
        `compiler_hash` `(` $compiler_hash `)`
        attr-dict
    }];
}

#endif
