//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI37XX_OPS
#define VPUX_COMPILER_DIALECT_VPUMI37XX_OPS

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/VPUIP/ops_interfaces.td"
include "vpux/compiler/dialect/VPUMI37XX/attributes.td"
include "vpux/compiler/dialect/VPUMI37XX/dialect.td"
include "vpux/compiler/dialect/VPUMI37XX/ops_interfaces.td"
include "vpux/compiler/dialect/VPUMI37XX/types.td"
include "vpux/compiler/dialect/VPURT/types.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/config/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/types.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/ELFNPU37XX/ops_interfaces.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"
include "vpux/compiler/dialect/VPURegMapped/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//
// Base classes
//

class VPUMI37XX_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPUMI37XX_Dialect,
            mnemonic,
            traits
        >;

class VPUMI37XX_TaskOp<string mnemonic, list<Trait> traits = []> :
        VPUMI37XX_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>,
                    DeclareOpInterfaceMethods<ELFNPU37XX_GetOffsetOfOpInterface>,
                    VPUMI37XX_SingleOutputAsIndexOp
                ]
            )
        > {
}

//
// ConfigureBarrierOp
//

def VPUMI37XX_ConfigureBarrierOp :
        VPUMI37XX_Op<"ConfigureBarrier",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>,
                VPUMI37XX_SingleOutputAsIndexOp,
                DeclarationOp
            ]
        > {
    let summary = "A task to configure the setup for a barrier";

    let arguments = (ins
        UI8Attr:$id,
        SI64Attr:$next_same_id,
        Optional<VPURegMapped_IndexType>:$previousSameId,
        OptionalAttr<UI8Attr>:$producer_count,
        OptionalAttr<UI8Attr>:$consumer_count
    );

    let results = (outs
        VPURegMapped_IndexType:$barrier
    );

    let assemblyFormat = [{
        attr-dict
        `<` $id `,` $next_same_id `>`
        (`previousSameId` `(` $previousSameId^ `:` type($previousSameId) `)`)?
        `->` type(results)
    }];
}

//
// MappedInferenceOp
//
def VPUMI37XX_MappedInferenceOp :
        VPUMI37XX_Op<"MappedInference",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>,
                DeclareOpInterfaceMethods<ELFNPU37XX_GetOffsetOfOpInterface>,
                VPUMI37XX_SingleOutputAsIndexOp,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "Task representing the MappedInference structure";

    let arguments = (ins
        Variadic<VPURegMapped_IndexType>:$dmaTasks,
        Optional<VPURegMapped_IndexType>:$invariantTasks,
        Optional<VPURegMapped_IndexType>:$variantTasks,
        Optional<VPURegMapped_IndexType>:$actKernelRanges,
        Optional<VPURegMapped_IndexType>:$actKernelInvocations,
        Optional<VPURegMapped_IndexType>:$barrierTasks,
        Optional<VPURegMapped_IndexType>:$actShaveRt,
        Variadic<AnyMemRef>:$actShaveStacks,
        I64ArrayAttr:$dmaCount,
        UI32Attr:$invariantCount,
        UI32Attr:$variantCount,
        UI32Attr:$actKernelRangesCount,
        UI32Attr:$actKernelInvocationsCount,
        UI32Attr:$barrierCount
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
    attr-dict
        (`dmas` `(` $dmaTasks^ `:` type($dmaTasks) `)`)?
        (`invariants` `(` $invariantTasks^ `:` type($invariantTasks) `)`)?
        (`variants` `(` $variantTasks^ `:` type($variantTasks) `)`)?
        (`actKernelRanges` `(` $actKernelRanges^ `:` type($actKernelRanges) `)`)?
        (`actKernelInvocations` `(` $actKernelInvocations^ `:` type($actKernelInvocations) `)`)?
        (`barriers` `(` $barrierTasks^ `:` type($barrierTasks) `)` )?
        (`actShaveRt` `(` $actShaveRt^ `:` type($actShaveRt) `)` )?
        (`actShaveStacks` `(` $actShaveStacks^ `:` type($actShaveStacks) `)`)?
        `dmaCount` `(` $dmaCount `)`
        `invariantCount` `(` $invariantCount `)`
        `variantCount` `(` $variantCount `)`
        `actKernelRangesCount` `(` $actKernelRangesCount `)`
        `actKernelInvocationsCount` `(` $actKernelInvocationsCount `)`
        `barrierCount` `(` $barrierCount `)`
        `->` type(results)
    }];
}


//
// NNDMAOp
//

def VPUMI37XX_NNDMAOp :
        VPUMI37XX_TaskOp<"NNDMA",
            [
                DeclareOpInterfaceMethods<VPUMI37XX_ExecutableTaskOpInterface>,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "NN DMA task";

    let arguments = (ins
        Optional<VPURegMapped_IndexType>:$taskLocation,
        AnyMemRef:$input,
        Variadic<AnyMemRef>:$output_buffs,
        Optional<VPURegMapped_IndexType>:$nextDMAIdx,
        Variadic<VPURegMapped_IndexType>:$waitBarriers,
        Variadic<VPURegMapped_IndexType>:$updateBarriers,
        UI64Attr:$start_after,
        UI64Attr:$clean_after,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        IntAttr:$port,
        VPUIP_DMAAccModeAttr:$acceleration_mode,
        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$src, "mlir::Value":$dst)
        >,
        OpBuilder<
            (ins "mlir::Type":$index, "mlir::Value":$input, "mlir::ValueRange":$output_buffs,
            "mlir::Value":$nextDMAIdx, "mlir::ValueRange":$waitBarriers, "mlir::ValueRange":$updateBarriers,
            "uint64_t":$start_after, "uint64_t":$clean_after, "bool":$is_out_of_order,
            "bool":$is_critical, "int64_t":$port, "vpux::VPUIP::DMAAccMode":$acceleration_mode,
            "vpux::VPUIP::DMADescriptorAttr":$dma_descriptor)
        >
    ];

    let assemblyFormat = [{
        attr-dict
        (`taskLocation` `(` $taskLocation^ `:` type($taskLocation) `)`)?
        `inputs` `(` $input `:` type($input) `)`
        (`outputs` `(` $output_buffs^ `:` type($output_buffs) `)`)?
        ( `nextDMAIdx` `(` $nextDMAIdx^ `:` type($nextDMAIdx) `)`)?
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `start_after` `(` $start_after `)`
        `clean_after` `(` $clean_after `)`
        `acceleration_mode` `(` $acceleration_mode `)`
        `->` type(results)
    }];
}

//
// Activation Kernel Ops
//

//
// ActShaveRtOp
//

def VPUMI37XX_ActShaveRtOp :
        VPUMI37XX_Op<"ActShaveRt",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>,
                VPUMI37XX_SingleOutputAsIndexOp,
                DeclarationOp
            ]
        > {
    let summary = "Declaration of Act Shave Management Kernel";

    let arguments = (ins
        StrAttr:$kernel_path
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict `kernel` `(` $kernel_path `)` `->` type(results)
    }];

    let extraClassDeclaration = [{
        uint32_t getKernelEntry();
        uint32_t getVersion();
    }];
}

//
// DeclareKernelTextOp
//

def VPUMI37XX_DeclareKernelTextOp :
        VPUMI37XX_Op<"DeclareKernelText",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>,
                VPUMI37XX_SingleOutputAsIndexOp,
                DeclarationOp
            ]
        > {
    let summary = "Declaration of Software Kernel .text ";

    let arguments = (ins
        StrAttr:$kernel_path
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict `kernel_path` `(` $kernel_path `)` `->` type(results)
    }];
}


//
// DeclareKernelArgsOp
//

def VPUMI37XX_DeclareKernelArgsOp :
        VPUMI37XX_Op<"DeclareKernelArgs",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>,
                VPUMI37XX_SingleOutputAsIndexOp,
                DeclarationOp
            ]
        > {
    let summary = "Declaration of Software Kernel .args";

    let arguments = (ins
        StrAttr:$kernel_path
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict `kernel_path` `(` $kernel_path `)` `->` type(results)
    }];
}

//
// DeclareKernelEntryOp
//

def VPUMI37XX_DeclareKernelEntryOp :
        VPUMI37XX_Op<"DeclareKernelEntry",
            [
                VPUMI37XX_SingleOutputAsIndexOp,
                DeclarationOp
            ]
        > {
    let summary = "Declaration of Kernel Entry";

    let arguments = (ins
        StrAttr:$kernel_path
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict `kernel_path` `(` $kernel_path `)` `->` type(results)
    }];

    let extraClassDeclaration = [{
        uint32_t getKernelEntry();
    }];

}

//
// KernelParamsOp
//

def VPUMI37XX_KernelParamsOp :
        VPUMI37XX_Op<"KernelParams",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>,
                DeclareOpInterfaceMethods<ELFNPU37XX_GetOffsetOfOpInterface>,
                VPUMI37XX_SingleOutputAsIndexOp,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "Kernel Params";

    let description = [{
        This operation contains description of parameters for software kernel.
        It can be used to cover dynamic and static shapes. Dynamic shapes are represented by
        dynamicInputShapes and dynamicOutputShapes arguments. If shape does not require to be dynamic the corresponding argument
        can be omitted. The shape is resolved via kernel_params for this case.

        When isCompiled is true it means that the kernel is obtained through lowering to the LLVM dialect,
        else it is built from C++.
    }];

    let arguments = (ins
        Variadic<AnyMemRef>:$inputs,
        Variadic<AnyMemRef>:$outputs,
        VariadicOfVariadic<AnyMemRef, "dynamicInputShapesSize">:$dynamicInputShapes,
        VariadicOfVariadic<AnyMemRef, "dynamicOutputShapesSize">:$dynamicOutputShapes,
        DenseI32ArrayAttr:$dynamicInputShapesSize,
        DenseI32ArrayAttr:$dynamicOutputShapesSize,
        StrAttr:$kernel_type,
        ElementsAttr:$kernel_params,
        DefaultValuedAttr<BoolAttr, "false">:$isCompiled
    );

    let results = (outs
        VPURegMapped_IndexType:$index  // TODO: modify to VPUIndexType
    );

    let extraClassDeclaration = [{
        size_t getParamsStructSize();
    }];

    let assemblyFormat = [{
        attr-dict
        (`inputs` `(` $inputs^ `:` type($inputs) `)`)?
        (`outputs` `(` $outputs^ `:` type($outputs) `)`)?
        (`dynamicInputShapes` `(` $dynamicInputShapes^ `:` type($dynamicInputShapes) `)`)?
        (`dynamicOutputShapes` `(` $dynamicOutputShapes^ `:` type($dynamicOutputShapes) `)`)?
        `kernel_type` `(` $kernel_type `)`
        `kernel_params` `(` $kernel_params `)`
        `->` type(results)
    }];
}

//
// ActKernelRangeOp
//

def VPUMI37XX_ActKernelRangeOp : VPUMI37XX_TaskOp<"ActKernelRange",
        [AttrSizedOperandSegments]
    > {
    let summary = "Activation Kernel Range";

    let arguments = (ins
        Optional<VPURegMapped_IndexType>:$taskLocation,
        Optional<VPURegMapped_IndexType>:$kernel_text_index,
        Optional<VPURegMapped_IndexType>:$kernel_args_index,
        Optional<VPURegMapped_IndexType>:$kernel_entry_index,
        SymbolRefAttr:$kernelTaskType
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict
        (`taskLocation` `(` $taskLocation^ `:` type($taskLocation) `)`)?
        (`kernel_text_index` `(` $kernel_text_index^ `:` type($kernel_text_index) `)`)?
        (`kernel_args_index` `(` $kernel_args_index^ `:` type($kernel_args_index) `)`)?
        (`kernel_entry_index` `(` $kernel_entry_index^ `:` type($kernel_entry_index) `)`)?
        `kernelTaskType` `(` $kernelTaskType `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// ActKernelInvocationOp
//

def VPUMI37XX_ActKernelInvocationOp :
        VPUMI37XX_TaskOp<"ActKernelInvocation",
            [
                DeclareOpInterfaceMethods<VPUMI37XX_ExecutableTaskOpInterface>,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "Activation Kernel Invocation";

    let arguments = (ins
        Optional<VPURegMapped_IndexType>:$taskLocation,
        Variadic<VPURegMapped_IndexType>:$waitBarriers,
        Variadic<VPURegMapped_IndexType>:$updateBarriers,
        VPURegMapped_IndexType:$range_index,
        Optional<VPURegMapped_IndexType>:$params_index,
        Optional<MemRefOf<[UI32]>>:$profiling_data,
        DefaultValuedAttr<UI64Attr, "0">:$tile,
        UI64Attr:$start_after,
        UI64Attr:$clean_after
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict
        (`taskLocation` `(` $taskLocation^ `:` type($taskLocation) `)`)?
        `range_index` `(` $range_index `:` type($range_index) `)`
        (`params_index` `(` $params_index^ `:` type($params_index) `)`)?
        (`profiling_data` `(` $profiling_data^ `:` type($profiling_data) `)`)?
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `tile` `(` $tile `)`
        `start_after` `(` $start_after `)`
        `clean_after` `(` $clean_after `)`
        `->` type(results)
    }];
}

//
// DPU Variant Op
//

def VPUMI37XX_DPUVariantOp : VPUMI37XX_TaskOp<"DPUVariant", []> {
    let summary = "DPU Variant Op";

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let arguments = (ins
        Optional<VPURegMapped_IndexType>:$taskLocation,
        VPURegMapped_IndexType:$Invariant,
        I64ArrayAttr:$start,
        I64ArrayAttr:$end,
        VPU_PaddingAttr:$pad,
        VPU_MPEModeAttr:$mpe_mode,
        OptionalAttr<IntAttr>:$cluster_id,
        OptionalAttr<IntAttr>:$workload_id
    );
}

def VPUMI37XX_PPETaskOp : VPU_PPETaskOpBase<
    VPUMI37XX_Dialect,
    "PPETask",
    [ParentOneOf<["vpux::VPUMI37XX::DPUInvariantOp"]>]
> {}

//
// DPU Invariant Op
//

def VPUMI37XX_DPUInvariantOp :
        VPUMI37XX_TaskOp<"DPUInvariant",
            [
                DeclareOpInterfaceMethods<VPUMI37XX_ExecutableTaskOpInterface>,
                AttrSizedOperandSegments
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "DPU Invariant Op";

    let arguments = (ins
        Optional<VPURegMapped_IndexType>:$taskLocation,
        MemRefOf<[F16, BF16, quant_QuantizedType]>:$input,
        Optional<MemRefOf<[I1]>>:$input_sparsity_map,
        Optional<MemRefOf<[I32]>>:$input_storage_element_table,
        Optional<MemRefOf<[F16, BF16, quant_QuantizedType]>>:$weights,
        Optional<MemRefOf<[I1]>>:$weights_sparsity_map,
        Optional<MemRefOf<[SI32]>>:$weight_table,

        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$parent_input,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer]>>:$parent_input_sparsity_map,
        Optional<AnyTypeOf<[MemRefOf<[I32]>, VPUIP_DistributedBuffer]>>:$parent_input_storage_element_table,

        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$parent_output,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer]>>:$parent_output_sparsity_map,

        Variadic<MemRefOf<[F16, F32, BF16, quant_QuantizedType]>>:$output_buffs,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer]>>:$output_sparsity_map_buff,
        Optional<AnyTypeOf<[MemRefOf<[UI64]>, VPUIP_DistributedBuffer]>>:$profiling_data,

        VPUIP_NCETaskTypeAttr:$nce_task_type,
        OptionalAttr<VPU_EltwiseTypeAttr>:$eltwise_type,
        VPU_MPEModeAttr:$mpe_frequent_mode,

        OptionalAttr<I64ArrayAttr>:$kernel_size,
        OptionalAttr<I64ArrayAttr>:$kernel_strides,
        OptionalAttr<VPU_PaddingAttr>:$kernel_padding,
        UnitAttr:$is_continued,
        OptionalAttr<IntAttr>:$cm_sp_pattern,
        UnitAttr:$is_segmented,
        UnitAttr:$input_channels_compression,
        UnitAttr:$is_zero_offset_weights_table,
        OptionalAttr<IntAttr>:$out_channel_offset,
        UnitAttr:$is_superdense,
        OptionalAttr<BoolAttr>:$is_inplace,
        OptionalAttr<IntAttr>:$input_se_size,
        OptionalAttr<IntAttr>:$output_se_size,
        UnitAttr:$is_permute_quantize,

        Variadic<VPURegMapped_IndexType>:$waitBarriers,
        Variadic<VPURegMapped_IndexType>:$updateBarriers,
        UI64Attr:$start_after,
        UI64Attr:$clean_after
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let regions = (region
        SizedRegion<1>:$ppe
    );

    let extraClassDeclaration = [{
        std::size_t getBarrierHitsCount() {
            auto users = getResult().getUsers();
            return static_cast<std::size_t>(std::count_if(std::begin(users), std::end(users), [](auto user) { return mlir::isa<vpux::VPUMI37XX::DPUVariantOp>(user); }));
        }
    }];

    let assemblyFormat = [{
        attr-dict
        (`taskLocation` `(` $taskLocation^ `:` type($taskLocation) `)`)?
        `input` `(` $input  `:` type($input) `)`
        (`input_sparsity_map` `(` $input_sparsity_map^  `:` type($input_sparsity_map) `)`)?
        (`input_storage_element_table` `(` $input_storage_element_table^  `:` type($input_storage_element_table) `)`)?
        (`weights` `(` $weights^  `:` type($weights) `)`)?
        (`weights_sparsity_map` `(` $weights_sparsity_map^  `:` type($weights_sparsity_map) `)`)?
        (`weight_table` `(` $weight_table^  `:` type($weight_table) `)`)?
        `parent_input` `(` $parent_input `:` type($parent_input) `)`
        (`parent_input_sparsity_map` `(` $parent_input_sparsity_map^  `:` type($parent_input_sparsity_map) `)`)?
        (`parent_input_storage_element_table` `(` $parent_input_storage_element_table^  `:` type($parent_input_storage_element_table) `)`)?
        `parent_output` `(` $parent_output `:` type($parent_output) `)`
        (`parent_output_sparsity_map` `(` $parent_output_sparsity_map^  `:` type($parent_output_sparsity_map) `)`)?
        (`outputs` `(` $output_buffs^ `:` type($output_buffs) `)`)?
        (`output_sparsity_map_buff` `(` $output_sparsity_map_buff^  `:` type($output_sparsity_map_buff) `)`)?
        (`profiling_data` `(` $profiling_data^  `:` type($profiling_data) `)`)?
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type($index)
        `PPE` `:` $ppe
    }];

}

//
// Network Metadata Op
//

def VPUMI37XX_NetworkMetadataOp :
        VPUMI37XX_Op<"NetworkMetadata",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>
            ]
        > {
    let summary = "Network Metadata Op";

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let extraClassDeclaration = [{
        void serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection, ::elf::NetworkMetadata& metadata);
    }];

    let assemblyFormat = [{
        attr-dict `->` type(results)
    }];
}

//
// PerformanceMetrics
//

def VPUMI37XX_PerformanceMetricsOp :
        VPUMI37XX_Op<"PerformanceMetrics",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>
            ]
        > {
    let summary = "Performance Metrics Op";

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict `->` type(results)
    }];
}

//
// Profiling Metadata Op
//

def VPUMI37XX_ProfilingMetadataOp :
        VPUMI37XX_Op<"ProfilingMetadata",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>
            ]
        > {
    let summary = "Profiling Metadata Op";

    let arguments = (ins
        ElementsAttr:$metadata
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict `->` type(results)
    }];
}

//
// MappedInferenceVersionOp
//

def MappedInferenceVersionOp :
        VPUMI37XX_Op<"MappedInferenceVersion",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>
            ]
        > {
    let summary = "Represents a versioning mechanism for Mapped Inference";

    let arguments = (ins
          UI32Attr:$major,
          UI32Attr:$minor,
          UI32Attr:$patch
        );

    let results = (outs);

    let assemblyFormat = [{
        `(` $major `_` $minor `_` $patch `)`
        attr-dict}];

    let builders = [
        OpBuilder<(ins)>
        ];
}

//
// PlatformInfoOp
//

def PlatformInfoOp :
        VPUMI37XX_Op<"PlatformInfo",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>
            ]
        > {
    let summary = "Represents an op describing various platform information, such as arch, revision, etc.";

    let arguments = (ins
          Config_ArchKindAttr:$archKind
        );

    let assemblyFormat = [{
        attr-dict
        }];

    let builders = [
        OpBuilder<(ins)>
        ];
}


//
// DeclareTaskBufferOp
//

def VPUMI37XX_DeclareTaskBufferOp :
        VPUMI37XX_Op<"DeclareTaskBuffer", [
        ]> {
    let summary = "Buffer Op representing a discrete location of a task in runtime metadataBuffer";

    let arguments = (ins
        VPURegMapped_TaskTypeAttr:$task_type,
        OptionalAttr<UI64Attr>:$offset
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict
        $task_type `->` type(results)
    }];

    let extraClassDeclaration = [{
        uint64_t getMemoryOffset() {
            return getOffset().value_or(0);
        }

        void setMemoryOffset(mlir::IntegerAttr offset) {
            setOffsetAttr(offset);
        }
    }];
}

#endif
