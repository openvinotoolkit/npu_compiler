//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_OPS
#define VPUX_COMPILER_DIALECT_VPUIP_OPS

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/core/types.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPUIP/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/types.td"
include "vpux/compiler/dialect/VPURT/types.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//
// Base classes
//

class VPUIP_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPUIP_Dialect,
            mnemonic,
            traits
        >;

class VPUIP_TaskOp<int numResult, string mnemonic, list<Trait> traits = []> :
        VPUIP_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    InferTypeOpInterface,
                    DeclareOpInterfaceMethods<VPUIP_LayerOpInterface>,
                    VPUIP_AsyncLayerOpInterface,
                    DeclareOpInterfaceMethods<VPUIP_TaskOpInterface>,
                    MemoryEffectsOpInterface
                ]
            )
        > {
    code commonExtraClassDeclaration = VPUIP_InferLayerReturnTypes<numResult>.impl;

    let extraClassDeclaration = commonExtraClassDeclaration;
}

//
// NNDMAOp
//

def VPUIP_NNDMAOp :
        VPUIP_TaskOp<1, "NNDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder,
                VPUIP_DMATypeOpInterface,
                DotInterface
            ]
        > {
    let summary = "NN DMA task";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_ITIBuffer, VPUIP_BoundedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$spillId,
        OptionalAttr<UnitAttr>:$compress_candidate,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata,
        UnitAttr:$split_candidate,
        UnitAttr:$profiling_buffer_mgmt,
        OptionalAttr<IntAttr>:$fusionId
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port, "int32_t":$dma_hwp_id)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$port, "mlir::UnitAttr":$is_out_of_order, "mlir::UnitAttr":$is_critical, "mlir::IntegerAttr":$spillId)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical, "mlir::IntegerAttr":$spillId)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical, "mlir::IntegerAttr":$spillId, "mlir::UnitAttr":$compress_candidate)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical, "mlir::IntegerAttr":$spillId, "mlir::UnitAttr":$compress_candidate, "bool":$split_candidate, "mlir::IntegerAttr":$fusionId)
        >,
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = mlir::cast<vpux::NDTypeInterface>(getType()).getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// PermuteDMAOp
//

def VPUIP_PermuteDMAOp :
        VPUIP_TaskOp<1, "PermuteDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = "NN DMA task which enables permutation";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<AffineMapAttr>:$mem_perm,
        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata,
        OptionalAttr<VPUIP_InternalDataFlow>:$internalDataFlow
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::AffineMapAttr":$mem_perm,
                 "VPUIP::DMADescriptorAttr":$dma_descriptor)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::AffineMapAttr":$mem_perm,
                  "VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::IntegerAttr":$port)
        >
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = mlir::cast<vpux::NDTypeInterface>(getType()).getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// GatherDMAOp
//

def VPUIP_GatherDMAOp :
        VPUIP_TaskOp<1,"GatherDMA",
            [
                InferTypeOpInterface,
                ViewLikeOpInterface,
                VPUIP_DMATypeOpInterface,
                DotInterface
            ]
        > {
    let summary = [{
        Represents a gather operation using gather addressing mode of DMA for VPU4.
        See scatter-gather addressing in VPU4 HAS for more information.
    }];

    let arguments = (ins
        AnyMemRef:$input,
        AnyTypeOf<[MemRefOf<[I64]>, VPUIP_DistributedBuffer]>:$indices,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$outputBuff,


        IntAttr:$elementSize,
        IntAttr:$padding,

        OptionalAttr<IntAttr>:$port,

        UnitAttr:$isOutOfOrder,
        UnitAttr:$isCritical,

        OptionalAttr<VPUIP_DMADescriptorAttr>:$DMADescriptor,
        OptionalAttr<IntAttr>:$dmaHwpId,

        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata,
        OptionalAttr<VPUIP_GatherAddressingMode>:$addressingMode
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$indices, "mlir::Value":$outputBuff,
                 "mlir::IntegerAttr":$elementSize, "mlir::IntegerAttr":$padding,
                 "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$indices, "mlir::Value":$outputBuff,
                 "int64_t":$elementSize, "int64_t":$padding,
                 "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$indices, "mlir::Value":$outputBuff,
                 "int64_t":$elementSize, "int64_t":$padding,
                 "int64_t":$port, "VPUIP::GatherAddressingMode":$addressingMode)
        >
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = mlir::cast<vpux::NDTypeInterface>(getType()).getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `indices` `(` $indices `:` type($indices) `)`
        `outputs` `(` $outputBuff `:` type($outputBuff) `)`
        `->` type(results)
    }];
}

//
// ConvertDMAOp
//

def VPUIP_ConvertDMAOp :
        VPUIP_TaskOp<1, "ConvertDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameInOutDimsOrder,
                VPUIP_DMATypeOpInterface,
                DotInterface
            ]
        > {
    let summary = "NN DMA task which enables data conversion";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_ITIBuffer, VPUIP_BoundedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port)
        >,
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = mlir::cast<vpux::NDTypeInterface>(getType()).getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// DecompressDMAOp
//

def VPUIP_DecompressDMAOp :
        VPUIP_TaskOp<1, "DecompressDMAOp",
            [
                ViewLikeOpInterface,
                DotInterface,
                VPUIP_DMATypeOpInterface,
                AttrSizedOperandSegments
            ]
        > {
    let summary = [{
        NN DMA task for BitCompactor weight compression & Activation Spill Decompression.
        In case of BitCompactor weight compression: use BitCompactor library to compress weights and decompress with the HW feature in the DMA (BTC)
        In case of Activation Spill Decompression: decompress with the HW feature in the DMA (BTC)
        Omits VPUIP_SameShape trait & VPUIP_SameElementType (verifier for input/output same element type)}];

    let arguments = (ins
        AnyMemRef:$input,
        Optional<AnyMemRef>:$act_compression_size_entry,
        Optional<AnyMemRef>:$act_compression_sparsity_map,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$port, "mlir::UnitAttr":$is_out_of_order, "mlir::UnitAttr":$is_critical)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$output_buff, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$act_compression_sparsity_map, "mlir::Value":$output_buff, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$output_buff, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$output_buff, "mlir::IntegerAttr":$port, "mlir::UnitAttr":$is_out_of_order, "mlir::UnitAttr":$is_critical)
        >,
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = mlir::cast<vpux::NDTypeInterface>(getType()).getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`act_compression_size_entry` `(` $act_compression_size_entry^  `:` type($act_compression_size_entry) `)`)?
        (`act_compression_sparsity_map` `(` $act_compression_sparsity_map^  `:` type($act_compression_sparsity_map) `)`)?
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// CompressDMAOp
//

def VPUIP_CompressDMAOp :
        VPUIP_TaskOp<1, "CompressDMAOp",
            [
                ViewLikeOpInterface,
                DotInterface,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = [{
        NN DMA task for Activation Spill Compression.
        In case of Activation Spill Compression: compress with the HW feature in the DMA (BTC).
        Omits VPUIP_SameShape trait & VPUIP_SameElementType (verifier for input/output same element type)}];

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$act_compression_size_entry,
        Optional<AnyMemRef>:$act_compression_sparsity_map,
        AnyMemRef:$output_buff,

        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyMemRef:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$output_buff, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$act_compression_sparsity_map, "mlir::Value":$output_buff, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$output_buff, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$output_buff, "mlir::IntegerAttr":$port, "mlir::UnitAttr":$is_out_of_order, "mlir::UnitAttr":$is_critical)
        >,
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = mlir::cast<vpux::NDTypeInterface>(getType()).getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `act_compression_size_entry` `(` $act_compression_size_entry  `:` type($act_compression_size_entry) `)`
        (`act_compression_sparsity_map` `(` $act_compression_sparsity_map^  `:` type($act_compression_sparsity_map) `)`)?
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// SyncDMAOp
//

def VPUIP_SyncDMAOp :
        VPUIP_TaskOp<1, "SyncDMA",
            [
                ViewLikeOpInterface,
                VPUIP_DMATypeOpInterface,
                DotInterface
            ]
        > {
    let summary = "Sync DMA task";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,
        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyMemRef:$output
    );

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// ReadOnlyDMAOp
//

def VPUIP_ReadOnlyDMAOp :
        VPUIP_TaskOp<1, "ReadOnlyDMA",
            [
                ViewLikeOpInterface,
                DotInterface,
                DeclareOpInterfaceMethods<VPUIP_DMATypeOpInterface, ["getOutput","getOutputBuff"]>
            ]
        > {


    let summary = "This operation represents a Read-Only DMA task, modeled as a singular DMA operation without an output. "
                "The input is read and then discarded, with no result produced. The Read-Only mode is enabled via "
                "`SWIDTH > 0 && DWIDTH == 0` at the descriptor level, handled in the VPUMI → VPUASM rewriter based on "
                "output presence. As a result, certain code paths explicitly check for a missing output—either ignoring it "
                "or throwing an error if an output is unexpectedly used. A verifier ensures no users consume this operation's (nonexistent) result."
                "Possible usecases: DMA read-only support enables data reads without usage, aiding debugging, silicon workarounds, FIFO draining, DDR performance analysis, and MSC backup fill as a pseudo-prefetch.";


    let arguments = (ins
        AnyMemRef:$input,
        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyMemRef:$results
    );

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::GREEN;
        }


    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`

    }];

    let hasVerifier = 1;
}

//
// DPUTaskOp
//

def VPUIP_DPUTaskOp :
        VPUIP_Op<
            "DPUTask",
            [
                HasParent<"vpux::VPUIP::NCEClusterTaskOp">,
                DeclareOpInterfaceMethods<VPUIP_CycleCostInterface>

            ]
        > {
    let summary = "This object represents workload for a single DPU tile";

    let arguments = (ins
        I64ArrayAttr:$outStart,
        I64ArrayAttr:$outEnd,
        OptionalAttr<I64ArrayAttr>:$inStart,
        OptionalAttr<I64ArrayAttr>:$inEnd,
        VPU_PaddingAttr:$pad,
        VPU_MPEModeAttr:$mpe_mode,
        OptionalAttr<IntAttr>:$cluster_id,
        OptionalAttr<DPUHaloRegionArrayAttr>: $haloRegions,
        OptionalAttr<IntAttr>:$workload_id,
        OptionalAttr<I64ArrayAttr>:$variant_primitive_id // IDs for variant primitives A and B
    );

    let builders = [
        OpBuilder<(ins
            "mlir::ArrayAttr":$outStart,
            "mlir::ArrayAttr":$outEnd,
            "VPU::PaddingAttr":$pad,
            "VPU::MPEMode":$mpeMode
        )>,
        OpBuilder<(ins
            "mlir::ArrayAttr":$outStart,
            "mlir::ArrayAttr":$outEnd,
            "VPU::PaddingAttr":$pad,
            "VPU::MPEMode":$mpeMode,
            "mlir::IntegerAttr":$clusterId
        )>,
        OpBuilder<(ins
            "mlir::ArrayAttr":$outStart,
            "mlir::ArrayAttr":$outEnd,
            "mlir::ArrayAttr":$inStart,
            "mlir::ArrayAttr":$inEnd,
            "VPU::PaddingAttr":$pad,
            "VPU::MPEMode":$mpeMode
        )>,
        OpBuilder<(ins
            "mlir::ArrayAttr":$outStart,
            "mlir::ArrayAttr":$outEnd,
            "mlir::ArrayAttr":$inStart,
            "mlir::ArrayAttr":$inEnd,
            "VPU::PaddingAttr":$pad,
            "VPU::MPEMode":$mpeMode,
            "mlir::IntegerAttr":$clusterId
        )>,
        OpBuilder<(ins
            "mlir::ArrayAttr":$outStart,
            "mlir::ArrayAttr":$outEnd,
            "mlir::ArrayAttr":$inStart,
            "mlir::ArrayAttr":$inEnd,
            "VPU::PaddingAttr":$pad,
            "VPU::MPEMode":$mpeMode,
            "mlir::IntegerAttr":$clusterId,
            "mlir::ArrayAttr":$haloRegions
        )>
    ];

    let assemblyFormat = [{
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// PPETaskOp
//

def VPUIP_PPETaskOp :
        VPU_PPETaskOpBase<
            VPUIP_Dialect,
            "PPETask",
            [
                HasParent<"vpux::VPUIP::NCEClusterTaskOp">
            ]
         > {
    let builders = [
        OpBuilder<(ins
            "VPU::PPEAttr":$ppe
        )>
    ];
}

//
// NCEClusterTaskOp
//

def VPUIP_NCEClusterTaskOp :
        VPUIP_TaskOp<1, "NCEClusterTask",
            [
                AttrSizedOperandSegments,
                AttrSizedResultSegments,
                MultiViewOpInterface,
                NoRegionArguments,
                DeclareOpInterfaceMethods<VPUIP_CycleCostInterface>,
                OpAsmOpInterface
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "NCE Cluster Task Operation";

    let description = [{
        This operation defines NCE cluster task which describes single cluster of 5 DPUs. It is
        comprised of two argument categories:

        * Variants - describes the attributes for an individual DPU within the cluster.
        * Invariants - describes the collective attributes of the cluster.
        * is_superdense - a kind of ODU mode: allow OC = 4 and avoid channel alignment for IC <= 4 case.

        The variants argument takes on a region argument and up to 5 DPUTaskOps. The invariants
        take on a variety of argument types.

        Attribute superdense can be used for DPU permuteQuantize of NCHW FP16 input data.

        The NCEClusterTaskOp also supports fixed PPE functions as well as generic PPE instruction
        lists. The generic PPE instruction list argument needs to be described as a region of PPE
        supported ops. Single fixed PPE functions and generic PPE instruciton list usage is
        mutually exclusive.
    }];

    let arguments = (ins
        Optional<AnyTypeOf<[MemRefOf<[F32, SI32, F16, BF16, quant_QuantizedType]>, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>>:$input,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer]>>:$input_sparsity_map,
        Optional<AnyTypeOf<[MemRefOf<[I32]>, VPUIP_DistributedBuffer]>>:$input_storage_element_table,
        Optional<AnyTypeOf<[MemRefOf<[F16, BF16, quant_QuantizedType]>, VPUIP_DistributedBuffer]>>:$weights,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer]>>:$weights_sparsity_map,

        Optional<AnyTypeOf<[MemRefOf<[SI32]>, VPUIP_DistributedBuffer]>>:$weight_table,
        Optional<AnyTypeOf<[MemRefOf<[SI32]>, VPUIP_DistributedBuffer]>>:$weight_table_data_ptr,
        Optional<AnyTypeOf<[MemRefOf<[SI32]>, VPUIP_DistributedBuffer]>>:$weight_table_sp_ptr,
        Optional<AnyTypeOf<[MemRefOf<[F32, F8E5M2, F8E4M3FN]>, VPUIP_DistributedBuffer]>>:$weight_table_scale,
        Optional<AnyTypeOf<[MemRefOf<[F32]>, VPUIP_DistributedBuffer]>>:$weight_table_bias,
        Optional<AnyTypeOf<[MemRefOf<[I8, I4]>, VPUIP_DistributedBuffer]>>:$weight_zero_points,

        Optional<AnyTypeOf<[MemRefOf<[UI16]>, VPUIP_DistributedBuffer]>>:$spr_lookup_table,
        Optional<AnyTypeOf<[MemRefOf<[UI16]>, VPUIP_DistributedBuffer]>>:$pallet_lookup_table,

        Optional<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>>:$parent_input,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer]>>:$parent_input_sparsity_map,
        Optional<AnyTypeOf<[MemRefOf<[I32]>, VPUIP_DistributedBuffer]>>:$parent_input_storage_element_table,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>:$parent_output,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>>: $parent_output_sparsity_map,
        Variadic<VPUIP_ITIBuffer>:$output_ITI_buff,

        AnyTypeOf<[MemRefOf<[F16, F32, BF16, quant_QuantizedType]>, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>:$output_buff,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>>:$output_sparsity_map_buff,
        Optional<AnyTypeOf<[MemRefOf<[UI64]>, VPUIP_DistributedBuffer]>>:$profiling_data,
        Optional<AnyTypeOf<[MemRefOf<[I32]>, VPUIP_DistributedBuffer]>>:$dynamic_sequence_length,

        Optional<AnyMemRef>:$max_per_xy,
        Optional<AnyMemRef>:$min_per_xy,
        Variadic<AnyMemRef>:$min_max_per_tensor,

        VPUIP_NCETaskTypeAttr:$task_type,
        OptionalAttr<VPU_EltwiseTypeAttr>:$eltwise_type,
        OptionalAttr<I64ArrayAttr>:$kernel_size,
        OptionalAttr<I64ArrayAttr>:$kernel_strides,
        OptionalAttr<VPU_PaddingAttr>:$kernel_padding,
        UnitAttr:$is_continued,
        OptionalAttr<IntAttr>:$cm_sp_pattern,
        UnitAttr:$is_segmented,
        OptionalAttr<IntAttr>:$out_channel_offset,
        UnitAttr:$input_channels_compression,
        UnitAttr:$is_zero_offset_weights_table,
        UnitAttr:$is_superdense,
        OptionalAttr<BoolAttr>:$is_inplace,
        OptionalAttr<IntAttr>:$input_se_size,
        OptionalAttr<IntAttr>:$output_se_size,
        UnitAttr:$is_permute_quantize,
        UnitAttr:$is_small_kernel_optimized,
        OptionalAttr<VPUIP_DpuProfilingMetadataAttr>:$profilingMetadata,
        OptionalAttr<VPU_MPEEngineAttr>:$mpe_engine,
        OptionalAttr<VPUIP_DynamicScaleConfigAttr>:$dynamic_scale_config,
        OptionalAttr<VPUIP_LocalRegionAttr>:$local_region,
        OptionalAttr<VPUIP_S2DConfigAttr>:$s2d_config
    );

    let results = (outs
        AnyTypeOf<[MemRefOf<[F16, F32, BF16, quant_QuantizedType]>, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>:$output,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>>:$output_sparsity_map,
        Optional<AnyTypeOf<[MemRefOf<[UI64]>, VPUIP_DistributedBuffer]>>:$profiling_output
    );

    let regions = (region
        SizedRegion<1>:$variants,
        SizedRegion<1>:$ppe
    );

    let skipDefaultBuilders = 0;

    let builders = [
        // used by profiling pass to keep op attributes intact
        OpBuilder<(ins
            "NCEClusterTaskOp":$task,
            "mlir::Type":$profilingOutput
        )>,

        OpBuilder<(ins
            "mlir::Value":$input, "mlir::Value":$weights, "mlir::Value":$weight_table,
            "mlir::Value":$weight_table_data_ptr, "mlir::Value":$weight_table_sp_ptr,
            "mlir::Value":$weight_table_scale, "mlir::Value":$weight_table_bias,
            "mlir::Value":$weight_zero_points,
            "mlir::Value":$spr_lookup_table, "mlir::Value":$pallet_lookup_table,
            "mlir::Value":$parent_input, "mlir::Value":$parent_output,
            "mlir::Value":$output_buff,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_zero_offset_weights_table,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized,
            CArg<"VPU::MPEEngineAttr", "nullptr">:$mpe_engine,
            CArg<"VPU::EltwiseTypeAttr", "nullptr">:$eltwise_type
        )>,

        OpBuilder<(ins
            "mlir::Type":$output,
            "mlir::Value":$input, "mlir::Value":$weights, "mlir::Value":$weight_table,
            "mlir::Value":$weight_table_data_ptr, "mlir::Value":$weight_table_sp_ptr,
            "mlir::Value":$weight_table_scale, "mlir::Value":$weight_table_bias,
            "mlir::Value":$weight_zero_points,
            "mlir::Value":$spr_lookup_table, "mlir::Value":$pallet_lookup_table,
            "mlir::Value":$parent_input, "mlir::Value":$parent_output,
            "mlir::Value":$output_buff,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_zero_offset_weights_table,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized,
            CArg<"VPU::MPEEngineAttr", "nullptr">:$mpe_engine,
            CArg<"VPU::EltwiseTypeAttr", "nullptr">:$eltwise_type
        )>,

        OpBuilder<(ins
            "mlir::Value":$input, "mlir::Value":$weights, "mlir::Value":$weight_table,
            "mlir::Value":$weight_table_data_ptr, "mlir::Value":$weight_table_sp_ptr,
            "mlir::Value":$weight_table_scale, "mlir::Value":$weight_table_bias,
            "mlir::Value":$weight_zero_points,
            "mlir::Value":$spr_lookup_table, "mlir::Value":$pallet_lookup_table,
            "mlir::Value":$parent_input, "mlir::Value":$parent_output,
            "mlir::Value":$output_buff, "mlir::Value":$profiling_data,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_zero_offset_weights_table,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized,
            CArg<"VPU::MPEEngineAttr", "nullptr">:$mpe_engine,
            CArg<"VPU::EltwiseTypeAttr", "nullptr">:$eltwise_type
        )>,

        OpBuilder<(ins
            "mlir::Type":$output, "mlir::Type":$profiling_output,
            "mlir::Value":$input, "mlir::Value":$weights, "mlir::Value":$weight_table,
            "mlir::Value":$weight_table_data_ptr, "mlir::Value":$weight_table_sp_ptr,
            "mlir::Value":$weight_table_scale, "mlir::Value":$weight_table_bias,
            "mlir::Value":$weight_zero_points,
            "mlir::Value":$spr_lookup_table, "mlir::Value":$pallet_lookup_table,
            "mlir::Value":$parent_input, "mlir::Value":$parent_output,
            "mlir::Value":$output_buff, "mlir::Value":$profiling_data,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_zero_offset_weights_table,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized,
            CArg<"VPU::MPEEngineAttr", "nullptr">:$mpe_engine,
            CArg<"VPU::EltwiseTypeAttr", "nullptr">:$eltwise_type
        )>,

        OpBuilder<(ins
            "mlir::Value":$input, "mlir::Value":$input_sparsity_map, "mlir::Value":$input_storage_element_table,
            "mlir::Value":$weights, "mlir::Value":$weights_sparsity_map,
            "mlir::Value":$weight_table,
            "mlir::Value":$weight_table_data_ptr, "mlir::Value":$weight_table_sp_ptr,
            "mlir::Value":$weight_table_scale, "mlir::Value":$weight_table_bias,
            "mlir::Value":$weight_zero_points,
            "mlir::Value":$spr_lookup_table, "mlir::Value":$pallet_lookup_table,
            "mlir::Value":$parent_input, "mlir::Value":$parent_input_sparsity_map, "mlir::Value":$parent_input_storage_element_table,
            "mlir::Value":$parent_output, "mlir::Value":$parent_output_sparsity_map,
            "mlir::Value":$output_buff, "mlir::Value":$output_sparsity_map_buff, "mlir::Value":$profiling_data,
            "mlir::Value":$dynamic_sequence_length,
            "mlir::Value":$max_per_xy, "mlir::Value":$min_per_xy,"mlir::ValueRange":$min_max_per_tensor,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_zero_offset_weights_table,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized,
            CArg<"VPU::MPEEngineAttr", "nullptr">:$mpe_engine,
            CArg<"VPU::EltwiseTypeAttr", "nullptr">:$eltwise_type,
            CArg<"VPUIP::DynamicScaleConfigAttr", "nullptr">:$dynamic_scale_config,
            CArg<"VPUIP::LocalRegionAttr", "nullptr">:$local_region,
            CArg<"VPUIP::S2DConfigAttr", "nullptr">:$s2d_config
        )>,

        OpBuilder<(ins
            "mlir::Type":$output, "mlir::Type":$output_sparsity_map, "mlir::Type":$profiling_output,
            "mlir::Value":$input, "mlir::Value":$input_sparsity_map, "mlir::Value":$input_storage_element_table,
            "mlir::Value":$weights, "mlir::Value":$weights_sparsity_map,
            "mlir::Value":$weight_table,
            "mlir::Value":$weight_table_data_ptr, "mlir::Value":$weight_table_sp_ptr,
            "mlir::Value":$weight_table_scale, "mlir::Value":$weight_table_bias,
            "mlir::Value":$weight_zero_points,
            "mlir::Value":$spr_lookup_table, "mlir::Value":$pallet_lookup_table,
            "mlir::Value":$parent_input, "mlir::Value":$parent_input_sparsity_map, "mlir::Value":$parent_input_storage_element_table,
            "mlir::Value":$parent_output, "mlir::Value":$parent_output_sparsity_map,
            "mlir::Value":$output_buff, "mlir::Value":$output_sparsity_map_buff, "mlir::Value":$profiling_data,
            "mlir::Value":$dynamic_sequence_length,
            "mlir::Value":$max_per_xy, "mlir::Value":$min_per_xy,"mlir::ValueRange":$min_max_per_tensor,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_zero_offset_weights_table,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized,
            CArg<"VPU::MPEEngineAttr", "nullptr">:$mpe_engine,
            CArg<"VPU::EltwiseTypeAttr", "nullptr">:$eltwise_type,
            CArg<"VPUIP::DynamicScaleConfigAttr", "nullptr">:$dynamic_scale_config,
            CArg<"VPUIP::LocalRegionAttr", "nullptr">:$local_region,
            CArg<"VPUIP::S2DConfigAttr", "nullptr">:$s2d_config
        )>,

        OpBuilder<(ins
            "mlir::Type":$output, "mlir::Type":$output_sparsity_map, "mlir::Type":$profiling_output,
            "mlir::Value":$input, "mlir::Value":$input_sparsity_map, "mlir::Value":$input_storage_element_table,
            "mlir::Value":$weights, "mlir::Value":$weights_sparsity_map,
            "mlir::Value":$weight_table,
            "mlir::Value":$weight_table_data_ptr, "mlir::Value":$weight_table_sp_ptr,
            "mlir::Value":$weight_table_scale, "mlir::Value":$weight_table_bias, "mlir::Value":$weight_zero_points,
            "mlir::Value":$spr_lookup_table, "mlir::Value":$pallet_lookup_table,
            "mlir::Value":$parent_input, "mlir::Value":$parent_input_sparsity_map, "mlir::Value":$parent_input_storage_element_table,
            "mlir::Value":$parent_output, "mlir::Value":$parent_output_sparsity_map, "mlir::ValueRange":$output_ITI_buff,
            "mlir::Value":$output_buff, "mlir::Value":$output_sparsity_map_buff, "mlir::Value":$profiling_data,
            "mlir::Value":$dynamic_sequence_length,
            "mlir::Value":$max_per_xy, "mlir::Value":$min_per_xy,"mlir::ValueRange":$min_max_per_tensor,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_zero_offset_weights_table,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized,
            CArg<"VPU::MPEEngineAttr", "nullptr">:$mpe_engine,
            CArg<"VPU::EltwiseTypeAttr", "nullptr">:$eltwise_type,
            CArg<"VPUIP::DynamicScaleConfigAttr", "nullptr">:$dynamic_scale_config,
            CArg<"VPUIP::LocalRegionAttr", "nullptr">:$local_region,
            CArg<"VPUIP::S2DConfigAttr", "nullptr">:$s2d_config
        )>
    ];

    let assemblyFormat = [{
        attr-dict
        `input` `(` $input  `:` type($input) `)`
        (`input_sparsity_map` `(` $input_sparsity_map^  `:` type($input_sparsity_map) `)`)?
        (`input_storage_element_table` `(` $input_storage_element_table^  `:` type($input_storage_element_table) `)`)?
        (`weights` `(` $weights^  `:` type($weights) `)`)?
        (`weights_sparsity_map` `(` $weights_sparsity_map^  `:` type($weights_sparsity_map) `)`)?
        (`weight_table` `(` $weight_table^  `:` type($weight_table) `)`)?
        (`weight_table_data_ptr` `(` $weight_table_data_ptr^  `:` type($weight_table_data_ptr) `)`)?
        (`weight_table_sp_ptr` `(` $weight_table_sp_ptr^  `:` type($weight_table_sp_ptr) `)`)?
        (`weight_table_scale` `(` $weight_table_scale^  `:` type($weight_table_scale) `)`)?
        (`weight_table_bias` `(` $weight_table_bias^  `:` type($weight_table_bias) `)`)?
        (`weight_zero_points` `(` $weight_zero_points^  `:` type($weight_zero_points) `)`)?
        (`spr_lookup_table` `(` $spr_lookup_table^  `:` type($spr_lookup_table) `)`)?
        (`pallet_lookup_table` `(` $pallet_lookup_table^  `:` type($pallet_lookup_table) `)`)?
        `parent_input` `(` $parent_input `:` type($parent_input) `)`
        (`parent_input_sparsity_map` `(` $parent_input_sparsity_map^  `:` type($parent_input_sparsity_map) `)`)?
        (`parent_input_storage_element_table` `(` $parent_input_storage_element_table^  `:` type($parent_input_storage_element_table) `)`)?
        `parent_output` `(` $parent_output `:` type($parent_output) `)`
        (`parent_output_sparsity_map` `(` $parent_output_sparsity_map^  `:` type($parent_output_sparsity_map) `)`)?
        (`output_ITI_buff` `(` $output_ITI_buff^ `:` type($output_ITI_buff) `)`)?
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`output_sparsity_map` `(` $output_sparsity_map_buff^  `:` type($output_sparsity_map_buff) `)`)?
        (`profiling_data` `(` $profiling_data^  `:` type($profiling_data) `)`)?
        (`dynamic_sequence_length` `(` $dynamic_sequence_length^  `:` type($dynamic_sequence_length) `)`)?
        (`max_per_xy` `(` $max_per_xy^  `:` type($max_per_xy) `)`)?
        (`min_per_xy` `(` $min_per_xy^  `:` type($min_per_xy) `)`)?
        (`min_max_per_tensor` `(` $min_max_per_tensor^  `:` type($min_max_per_tensor) `)`)?
        `->` type($output) `` custom<OptionalResultTypes>(type($output_sparsity_map), type($profiling_output)) ` `
        `variants` `:` $variants
        `PPE` `:` $ppe
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        vpux::VPUIP::DPUTaskOp addDPUTask(mlir::OpBuilder& builder,
                                          mlir::ArrayAttr outStart, mlir::ArrayAttr outEnd,
                                          mlir::ArrayAttr inStart, mlir::ArrayAttr inEnd,
                                          vpux::VPU::PaddingAttr pad, vpux::VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId = nullptr);
        vpux::VPUIP::DPUTaskOp addDPUTask(mlir::OpBuilder& builder,
                                          mlir::ArrayAttr outStart, mlir::ArrayAttr outEnd,
                                          mlir::ArrayAttr inStart, mlir::ArrayAttr inEnd,
                                          vpux::VPU::PaddingAttr pad, vpux::VPU::MPEMode mpeMode, mlir::ArrayAttr variant_primitive_id);
        vpux::VPUIP::DPUTaskOp addDPUTask(mlir::OpBuilder& builder,
                                          mlir::ArrayAttr outStart, mlir::ArrayAttr outEnd,
                                          mlir::ArrayAttr inStart, mlir::ArrayAttr inEnd,
                                          vpux::VPU::PaddingAttr pad, vpux::VPU::MPEMode mpeMode,
                                          mlir::IntegerAttr clusterId, mlir::ArrayAttr haloRegions);
        vpux::VPUIP::DPUTaskOp addDPUTask(mlir::OpBuilder& builder,
                                          mlir::ArrayAttr outStart, mlir::ArrayAttr outEnd,
                                          vpux::VPU::PaddingAttr pad, vpux::VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId = nullptr);

        template <typename... Args>
        vpux::VPUIP::PPETaskOp addPPETask(mlir::OpBuilder& builder, Args&&... args);

        int64_t getNumVariants();

        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DPU;
        }

        static mlir::StringRef getDefaultDialect() {
            return "VPUIP";
        }

        static mlir::LogicalResult inferReturnTypes(
                mlir::MLIRContext*,
                std::optional<mlir::Location>,
                mlir::ValueRange operands,
                mlir::DictionaryAttr,
                mlir::OpaqueProperties,
                mlir::RegionRange,
                llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes);
    }];
}

//
// VPUIP_SwKernelRun
//

def VPUIP_SwKernelRun
    : VPUIP_Op<"SW.Kernel.run"> {

    let arguments  = (ins
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>>:$args,
        OptionalAttr<ArrayAttr>:$attrs
    );

    let assemblyFormat = [{
        attr-dict
        ( `(` $args^ `)` `:` type($args) )?
    }];
}

//
// VPUIP_SwKernelOp
//

def VPUIP_SwKernelOp :
        VPUIP_TaskOp<1, "SW.Kernel",
            [
                MultiViewOpInterface,
                IsolatedFromAbove,
                AttrSizedOperandSegments,
                AttrSizedResultSegments,
                DeclareOpInterfaceMethods<VPUIP_CycleCostInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "Software Layer Task";

    let description = [{
        This operation defines Activation shave task.
        There are two different modes or handling inputs with dynamic shapes
        - In the first mode, dynamic inputs are accepted as BoundedBuffers, which combine data and dynamic shape into one type.
        dynamicInputShapes and dynamicInputShapesMap are not used in that case:
                inputs[3]: [BoundedBuffer, MemRef, BoundedBuffer]
                dynamicInputShapes[]: []
                dynamicInputShapesMap[]: nullptr
        - In the second mode all BoundedBuffer's are unrolled by ungroup-bounded-buffers pass: E#111348.
        Separate handling of dynamic data and dynamic shape is needed because feasible allocation doesn't support multiple root buffers per input
                inputs[3]: [MemRef, MemRef, MemRef]
                dynamicInputShapes[2]: [MemRef, MemRef]
                dynamicInputShapesMap[3]: [0, -1, 1]
        For outputs, the same applies.
    }];

    let arguments = (ins
        SymbolRefAttr:$kernelFunction,
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>>:$inputs,
        Variadic<AnyTypeOf<[MemRefOf<[SI32]>, VPUIP_DistributedBuffer]>>:$dynamicInputShapes,
        OptionalAttr<DenseI32ArrayAttr>:$dynamicInputShapesMap,
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>>:$output_buffs,
        Variadic<AnyTypeOf<[MemRefOf<[SI32]>, VPUIP_DistributedBuffer]>>:$dynamicOutputShapeBuffs,
        OptionalAttr<DenseI32ArrayAttr>:$dynamicOutputShapesMap,
        Optional<AnyTypeOf<[MemRefOf<[UI32]>, VPUIP_DistributedBuffer]>>:$profiling_data,
        OptionalAttr<IntAttr>:$tileIndex,
        OptionalAttr<IntAttr>:$listIndex,
        OptionalAttr<I64ArrayOfArraysAttr>:$inputStrides,
        OptionalAttr<I64ArrayOfArraysAttr>:$outputStrides,
        OptionalAttr<VPUIP_SwProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>>:$results,
        Variadic<AnyTypeOf<[MemRefOf<[SI32]>,VPUIP_DistributedBuffer]>>:$dynamicOutputShapes,
        Optional<AnyTypeOf<[MemRefOf<[UI32]>, VPUIP_DistributedBuffer]>>:$profiling_output
    );

    let regions = (region
        SizedRegion<1>:$body
    );

    let builders = [
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::Value":$profiling_data,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex,
            "mlir::ArrayAttr":$outputStrides
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex,
            "mlir::ArrayAttr":$inputStrides,
            "mlir::ArrayAttr":$outputStrides
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::Value":$profiling_data,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex,
            "mlir::ArrayAttr":$inputStrides,
            "mlir::ArrayAttr":$outputStrides
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::ValueRange":$dynamicInputShapes,
            "llvm::ArrayRef<int32_t>":$dynamicInputShapesMap,
            "mlir::ValueRange":$dynamicOutputShapeBuffs,
            "llvm::ArrayRef<int32_t>":$dynamicOutputShapesMap,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::ValueRange":$dynamicInputShapes,
            "llvm::ArrayRef<int32_t>":$dynamicInputShapesMap,
            "mlir::ValueRange":$dynamicOutputShapeBuffs,
            "llvm::ArrayRef<int32_t>":$dynamicOutputShapesMap,
            "mlir::Value":$profiling_data,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::ValueRange":$dynamicInputShapes,
            "llvm::ArrayRef<int32_t>":$dynamicInputShapesMap,
            "mlir::ValueRange":$dynamicOutputShapeBuffs,
            "llvm::ArrayRef<int32_t>":$dynamicOutputShapesMap,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex,
            "mlir::ArrayAttr":$outputStrides
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::ValueRange":$dynamicInputShapes,
            "llvm::ArrayRef<int32_t>":$dynamicInputShapesMap,
            "mlir::ValueRange":$dynamicOutputShapeBuffs,
            "llvm::ArrayRef<int32_t>":$dynamicOutputShapesMap,
            "mlir::Value":$profiling_data,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex,
            "mlir::ArrayAttr":$outputStrides
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::ValueRange":$dynamicInputShapes,
            "llvm::ArrayRef<int32_t>":$dynamicInputShapesMap,
            "mlir::ValueRange":$dynamicOutputShapeBuffs,
            "llvm::ArrayRef<int32_t>":$dynamicOutputShapesMap,
            "mlir::Value":$profiling_data,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex,
            "mlir::ArrayAttr":$inputStrides,
            "mlir::ArrayAttr":$outputStrides
        )>,

        // used by profiling pass to keep op attributes intact
        OpBuilder<(ins
            "SwKernelOp":$swOp,
            "mlir::Value":$profilingOutput
        )>
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::SHAVE_ACT;
        }

        static mlir::LogicalResult inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> loc,
                                                    mlir::ValueRange operands, mlir::DictionaryAttr attrs, mlir::OpaqueProperties,
                                                    mlir::RegionRange regions,
                                                    mlir::SmallVectorImpl<mlir::Type>& inferredTypes);

        static vpux::VPUIP::KernelInfo getKernelInfo(mlir::Operation* origOp);
        static vpux::VPUIP::KernelInfo getDummyKernelInfo();

        void print(::mlir::OpAsmPrinter& p);
        static ::mlir::ParseResult parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result);
    }];

    let hasVerifier = 1;
}

//
// SpaceToDepthDMAOp
//

def VPUIP_SpaceToDepthDMAOp :
        VPUIP_TaskOp<1, "SpaceToDepthDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = "A NNDMA task which performs a functionality equivalent DMA transfer to a SpaceToDepth task";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        IntAttr:$block_size,
        IE_SpaceToDepthModeAttr:$mode,
        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata,
        OptionalAttr<VPUIP_InternalDataFlow>:$internalDataFlow
    );

    let builders = [
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$block_size,
            "vpux::IE::SpaceToDepthModeAttr":$mode, "VPUIP::DMADescriptorAttr":$dma_descriptor)
        >,
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$block_size,
            "vpux::IE::SpaceToDepthModeAttr":$mode, "VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::IntegerAttr":$port)
        >
    ];

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = mlir::cast<vpux::NDTypeInterface>(getType()).getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// PerAxisTileDMAOp
//

def VPUIP_PerAxisTileDMAOp :
        VPUIP_TaskOp<1, "PerAxisTileDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameInOutDimsOrder,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = "A NNDMA task which performs a functionality equivalent DMA transfer to a PerAxisTileOp task";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        OptionalAttr<IntAttr>:$axis,
        OptionalAttr<IntAttr>:$tiles,
        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let builders = [
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$axis,
            "mlir::IntegerAttr":$tiles, "VPUIP::DMADescriptorAttr":$dma_descriptor)
        >,
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$block_size,
            "mlir::IntegerAttr":$tiles, "VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::IntegerAttr":$port)
        >
    ];

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = mlir::cast<vpux::NDTypeInterface>(getType()).getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// DepthToSpaceDMAOp
//

def VPUIP_DepthToSpaceDMAOp :
        VPUIP_TaskOp<1, "DepthToSpaceDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = "A NNDMA task which performs a functionality equivalent DMA transfer to a DepthToSpace task";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        IntAttr:$block_size,
        IE_DepthToSpaceModeAttr:$mode,
        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor,
        OptionalAttr<IE_ChannelPaddingAttr>:$padded_channels,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata,
        OptionalAttr<VPUIP_InternalDataFlow>:$internalDataFlow
    );

    let builders = [
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$block_size,
            "vpux::IE::DepthToSpaceModeAttr":$mode, "VPUIP::DMADescriptorAttr":$dma_descriptor,
            "vpux::IE::ChannelPaddingAttr":$padded_channels)
        >,
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$block_size,
            "vpux::IE::DepthToSpaceModeAttr":$mode, "VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::IntegerAttr":$port,
            "vpux::IE::ChannelPaddingAttr":$padded_channels)
        >
    ];

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = mlir::cast<vpux::NDTypeInterface>(getType()).getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// M2ITaskOp
//

def VPUIP_M2ITaskOp :
        VPUIP_TaskOp<1,"M2ITask",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "M2I task op";

    let arguments = (ins
        MemRefOf<[UI8, F16]>:$input,
        MemRefOf<[UI8, F16]>:$output_buff,
        Optional<MemRefOf<[UI8]>>:$profiling_data,

        BoolAttr:$do_csc,
        BoolAttr:$do_norm,
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
        OptionalAttr<VPUIP_M2IProfilingMetadataAttr>:$profilingMetadata,
        DefaultValuedAttr<VPU_M2iInterpAttr, "vpux::VPU::M2iInterp::NEAREST">:$interp
    );

    let results = (outs
        MemRefOf<[UI8, F16]>:$output,
        Optional<MemRefOf<[UI8]>>:$profiling_output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`profiling_data` `(` $profiling_data^  `:` type($profiling_data) `)`)?
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::M2I;
        }

        static mlir::LogicalResult inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> loc,
                                                    mlir::ValueRange operands, mlir::DictionaryAttr attrs, mlir::OpaqueProperties,
                                                    mlir::RegionRange regions,
                                                    mlir::SmallVectorImpl<mlir::Type>& inferredTypes);
    }];
}

//
// DistributedCastOp
//

def VPUIP_DistributedCastOp :
        VPUIP_Op<
            "DistributedCast",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure
            ]
        > {
    let summary = "Operation that casts one DistributedBuffer type to another.";

    let arguments = (ins
        AnyTypeOf<[VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$input
    );

    let results = (outs
        AnyTypeOf<[VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` qualified(type($input)) `)`
        `->` qualified(type(results))
    }];

    let hasFolder = 1;
    let hasVerifier = 1;
}

//
// NonDistributedCast
//

def VPUIP_NonDistributedCastOp:
        VPUIP_Op<
            "NonDistributedCastOp",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure
            ]
        > {
    let summary = "Operation that casts one DistributedBuffer type to non-distributed buffer type.";

    let arguments = (ins
        AnyTypeOf<[VPUIP_DistributedBuffer]>:$input
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` qualified(type($input)) `)`
        `->` qualified(type(results))
    }];

    let hasVerifier = 1;
}


//
// SubViewOp
//

def VPUIP_SubViewOp :
        VPUIP_Op<
            "SubView",
            [
                Pure,
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface>,
                DotInterface
            ]
        > {
    let summary = "Extract single subview from buffer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$source,

        I64ArrayAttr:$static_offsets,
        I64ArrayAttr:$static_sizes,
        OptionalAttr<I64ArrayAttr>:$static_strides,
        OptionalAttr<I64ArrayOfArraysAttr>:$explicit_output_shapes
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$result
    );

    let assemblyFormat = [{
        $source $static_offsets $static_sizes ($static_strides^)?
        attr-dict `:` type($source) `to` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }

        Byte getByteOffset();

        static bool isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
            return vpux::areTypesCompatible(lhs, rhs,
                vpux::IE::TypeComparisonMode::STRICT_EQUAL,
                /*checkInferredDimsOrder=*/true,
                /*checkInferredMemSpace=*/true
            );
        }
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ShapeRef":$static_offsets, "vpux::ShapeRef":$static_sizes)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ArrayRef<int64_t>":$static_offsets, "vpux::ArrayRef<int64_t>":$static_sizes)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "mlir::ArrayAttr":$static_offsets, "mlir::ArrayAttr":$static_sizes)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "mlir::ArrayAttr":$static_offsets, "mlir::ArrayAttr":$static_sizes, "mlir::ArrayAttr":$static_strides)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ShapeRef":$static_offsets, "vpux::ShapeRef":$static_sizes, "vpux::ShapeRef":$static_strides)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ArrayRef<int64_t>":$static_offsets, "vpux::ArrayRef<int64_t>":$static_sizes, "vpux::ArrayRef<int64_t>":$static_strides)
        >
    ];

    let hasFolder = 1;
    let hasCanonicalizer = 1;
    let hasVerifier = 1;
}

//
// ConcatViewOp
//

def VPUIP_ConcatViewOp :
        VPUIP_Op<
            "ConcatView",
            [
                Pure,
                InferTypeOpInterface,
                ViewLikeOpInterface
            ]
        > {
    let summary = "VPUIP ConcatView layer. Dummy operation to maintain use-def chains.";

    let arguments = (ins
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_SparseBuffer, VPUIP_BoundedBuffer]>>:$inputs,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_SparseBuffer, VPUIP_BoundedBuffer]>:$output_buff
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_SparseBuffer, VPUIP_BoundedBuffer]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputs `:` type($inputs) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        mlir::Value getViewSource() {
            return getOutputBuff();
        }
    }] # VPUIP_InferLayerReturnTypes<1>.impl;
}

//
// PermuteCastOp
//

def VPUIP_PermuteCastOp :
        VPUIP_Op<
            "PermuteCast",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure
            ]
        > {
    let summary = "Operation that changes the layout information of a buffer";

    let description = [{
        The op changes layout information in the following way:
            * dst_order: order of output buffer is set to value of this arg
            * mem_perm: describes the permutation applied on input buffer's memory shape
                        to obtain the memory shape of the output buffer.
    }];

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer, VPUIP_DistributedBuffer]>:$source,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer, VPUIP_DistributedBuffer]>:$result
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $source `:` type($source) `)`
        `->` type(results)
    }];

    let hasFolder = 1;
}

//
// QuantizeCastOp
//

def VPUIP_QuantizeCastOp :
        VPUIP_Op<
            "QuantizeCast",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure
            ]
        > {
    let summary = "Operation that changes the quantization information of a buffer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$input
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

//
// GenericReshape
//

def VPUIP_GenericReshapeOp :
        VPUIP_Op<
            "GenericReshape",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure,
                VPUIP_SameOperandsAndResultElementType
            ]
        > {
    let summary = "Generic Reshape layer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer, VPUIP_DistributedBuffer]>:$input
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer, VPUIP_DistributedBuffer]>:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

//
// StaticAllocOp
//

def VPUIP_StaticAllocOp :
        VPUIP_Op<
            "StaticAlloc",
            [
                DeclarationOp,
                DotInterface,
                Pure
            ]
        > {
    let summary = "Static buffer allocation";

    let arguments = (ins
        IntAttr:$offset
    );

    let results = (outs
        AnyMemRef:$memory
    );

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }
    }];

    let assemblyFormat = [{
        `<` $offset `>` attr-dict `->` type(results)
    }];
}

//
// CopyOp
//

def VPUIP_CopyOp :
        VPUIP_Op<
            "Copy",
            [
                ViewLikeOpInterface,
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<VPUIP_LayerOpInterface>,
                MemoryEffectsOpInterface,
                DeclareOpInterfaceMethods<VPUIP_CycleCostInterface>,
                DotInterface
            ]
        > {
    let summary = "Copy layer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$output_buff
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::GREEN;
        }
    }] # VPUIP_InferLayerReturnTypes<1>.impl;
}

//
// TimestampOp
//

def VPUIP_TimestampOp :
        VPUIP_Op<
            "Timestamp",
            [
                ViewLikeOpInterface,
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<VPUIP_LayerOpInterface>,
                MemoryEffectsOpInterface,
                SameOperandsAndResultElementType,
                DotInterface
            ]
        > {
    let summary = "Get timer timestamp operation";

    let description = [{
        Get timer timestamp operation
    }];

    let arguments = (ins
        MemRefOf<[UI32, UI64]>:$output_buff
    );

    let results = (outs
        MemRefOf<[UI32, UI64]>:$output
    );

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::BLUE;
        }
    }] # VPUIP_InferLayerReturnTypes<1>.impl;

    let assemblyFormat = [{
        attr-dict
        `(` $output_buff `:` type($output_buff) `)`
        `->` type($output)
    }];
}

//
// ShapeCastOp
//

def VPUIP_ShapeCastOp :
        VPUIP_Op<
            "ShapeCast",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure,
                DeclareOpInterfaceMethods<InferTypeOpInterface>
            ]
        > {
    let summary = "Operation that changes the shape information of a buffer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$source,
        I64ArrayAttr:$shape,
        OptionalAttr<I64ArrayOfArraysAttr>:$explicit_output_shapes,
        OptionalAttr<I64ArrayOfArraysAttr>:$explicit_output_offsets
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$result
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $source `:` type($source) `)`
        `->` type(results)
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ShapeRef":$shape)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ArrayRef<int64_t>":$shape)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "mlir::ArrayAttr":$shape)
        >,
    ];
}

//
// GroupSparseBufferOp
//

def VPUIP_GroupSparseBufferOp :
        VPUIP_Op<
            "GroupSparseBuffer",
            [
                Pure,
                DotInterface,
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<GroupedViewOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
            ]
        > {
    let summary = "Group sparsity data and metadata into a value";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$data,
        Optional<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>>:$sparsityMap,
        Optional<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>>:$storageElementTable,

        UnitAttr:$is_weights,
        OptionalAttr<VPUIP_SparsityCompressionAttr>:$sparsity_compression,

        OptionalAttr<VPU_SEAttr>:$seAttr
    );

    let results = (outs
        VPUIP_SparseBuffer:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$data,
                CArg<"bool", "{}">:$is_weights, CArg<"VPUIP::SparsityCompressionAttr", "{}">:$sparsity_compression)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap,
                CArg<"bool", "{}">:$is_weights, CArg<"VPUIP::SparsityCompressionAttr", "{}">:$sparsity_compression)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap, "mlir::Value":$storageElementTable,
                CArg<"bool", "{}">:$is_weights, CArg<"VPUIP::SparsityCompressionAttr", "{}">:$sparsity_compression)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap, "mlir::Value":$storageElementTable,
                CArg<"VPU::SEAttr", "{}">:$seAttr)
        >
    ];

    let assemblyFormat = [{
        `(` $data
            (`,` $sparsityMap^ `` custom<OptionalTypes>(type($sparsityMap)))?
            (`,` $storageElementTable^ `` custom<OptionalTypes>(type($storageElementTable)))?
        `)`
        attr-dict
        `` custom<OptionalTypes>(type($data))
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }
    }];
}

//
// UngroupSparseBufferOp
//

def VPUIP_UngroupSparseBufferOp :
        VPUIP_Op<
            "UngroupSparseBuffer",
            [
                Pure,
                DotInterface,
                AttrSizedResultSegments,
                DeclareOpInterfaceMethods<MultiViewOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
            ]
        > {
    let summary = "Ungroups sparse value into data and metadata buffers";

    let arguments = (ins
        VPUIP_SparseBuffer:$input
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$data,
        Optional<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>>:$sparsityMap,
        Optional<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>>:$storageElementTable
    );

    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilder<(ins
            "mlir::Type":$data, "mlir::Type":$sparsityMap, "mlir::Type":$storageElementTable,
            "mlir::Value":$input
        )>,

        OpBuilder<(ins
            "mlir::Value":$input
        )>,
    ];

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        `` custom<OptionalTypes>(type($input))
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }
    }];
}

//
// StorageElementTable
//

def VPUIP_StorageElementTableOp :
        VPUIP_Op<
            "StorageElementTable",
            [
                Pure,
                DeclareOpInterfaceMethods<InferTypeOpInterface>,
            ]
        > {
    let summary = "Declares a Storage Element Pointers table";

    let description = [{
        A Storage Element represents a 1x1xN volume that contains sparse data, where N
        represents the number of channels stored. The Storage Element Table is comprised
        of pointers to these Storage Elements, which have the following structure:

        31-29 28                            9 8         0
        -------------------------------------------------
        | xx |           DATA_PTR            | BASE_PTR |
        -------------------------------------------------

        The DATA_PTR represents the offset to a Storage Element in relation to the start of
        the input data. BASE_PTR is used to decide what base address is added to DATA_PTR
        in order to find the location of the Storage Element in memory during inference.

        This operation represents the Storage Element Table in relation to the input data,
        on top of which transformations can be applied. This operation will later get
        converted to a constant, where the pointers are generated based on the information
        contained in this operation.

        The following information is contained:
        - dataShape, dataElemType, dataStrides: information about the input data that
          is associated with this Storage Element Table
        - seSize: the size of a Storage Element; seSize can be either:
            * an integer, in which case every seDepth is the same size, equal to the seSize value;
            * an array of integers, in which case each seDepth might have a different size; currently
              the only use case is having different seSize per cluster, therefore the seSize array should
              ultimately have as many values as there are clusters used for the DPU op this SETable belongs to.
        - seDepth: the number of Storage Elements per depth
        - seAttr: information on how the input data is transformed
        - basePtrs: base pointers associated with each Storage Element pointer
    }];

    let arguments = (ins
        I64ArrayAttr:$dataShape,
        TypeAttr:$dataElemType,
        I64ArrayAttr:$seSize,
        IntAttr:$seDepth,
        OptionalAttr<VPU_SEAttr>:$seAttr,
        OptionalAttr<I64ArrayAttr>:$dataStrides,
        OptionalAttr<I32ElementsAttr>:$basePtrs
    );

    let results = (outs
        MemRefOf<[I32]>:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let assemblyFormat = [{
         attr-dict `->` type(results)
    }];

    let builders = [
        OpBuilder<(ins
            CArg<"llvm::ArrayRef<int64_t>">:$dataShape,
            CArg<"mlir::Type">:$dataElemType,
            CArg<"llvm::ArrayRef<int64_t>">:$seSize,
            CArg<"int64_t">:$seDepth,
            CArg<"VPU::SEAttr">:$seAttr
        )>
    ];

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
            return vpux::areTypesCompatible(lhs, rhs,
                vpux::IE::TypeComparisonMode::STRICT_EQUAL,
                /*checkInferredDimsOrder=*/true,
                /*checkInferredMemSpace=*/true
            );
        }
    }];
}

//
// StubOp
//

def VPUIP_StubOp :
        VPUIP_Op<
            "Stub",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure
            ]
        > {
    let summary = "Substitute operation for stubbing.";

    let arguments = (ins
        Variadic<AnyMemRef>:$inputs
    );

    let results = (outs
        Variadic<AnyMemRef>:$outputs
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputs `:` type($inputs) `)`
        `->` type(results)
    }];
}

//
// ViewOp
//

def VPUIP_ViewOp :
        VPUIP_Op<
            "ViewOp",
            [
                Pure,
                DeclareOpInterfaceMethods<ViewLikeOpInterface>
            ]
        > {
    let summary = "Create a new view for buffer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$source
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$result
    );

    let assemblyFormat = [{
        $source
        attr-dict `:` type($source) `to` type(results)
    }];

}

//
// ExpandOp
//

def VPUIP_ExpandOp :
        VPUIP_Op<
            "Expand",
            [
                ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPUIP_LayerOpInterface>,
                InferTypeOpInterface
            ]
        > {
    let summary = "Expand layer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer]>:$output_buff,

        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }
    }] # VPUIP_InferLayerReturnTypes<1>.impl;
}

//
// ExpandDMAOp
//

def VPUIP_ExpandDMAOp :
        VPUIP_TaskOp<1, "ExpandDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = "A NNDMA task which expands tensor by copying values from original buffer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,

        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,

        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor,
        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let builders = [
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::ArrayAttr":$pads_begin,
            "mlir::ArrayAttr":$pads_end, "VPUIP::DMADescriptorAttr":$dma_descriptor)
        >,
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::ArrayAttr":$pads_begin,
            "mlir::ArrayAttr":$pads_end, "VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::IntegerAttr":$port)
        >
    ];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = mlir::cast<vpux::NDTypeInterface>(getType()).getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// UpsamplingDMAOp
//

def VPUIP_UpsamplingDMAOp :
        VPUIP_TaskOp<1, "UpsamplingDMAOp",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = "A NNDMA task which performs a functionality equivalent DMA transfer to a UpsamplingDMA task";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,
        I64ArrayAttr:$upsampling_factor,

        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor,
        OptionalAttr<I64ArrayAttr>:$expand,
        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::ArrayAttr":$upsampling_factor, "vpux::VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::ArrayAttr":$expand)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::ArrayAttr":$upsampling_factor, "vpux::VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::ArrayAttr":$expand, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::ArrayAttr":$upsampling_factor, "vpux::VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::ArrayAttr":$expand, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical, "mlir::IntegerAttr":$dma_hwp_id, "vpux::VPUIP::DmaProfilingMetadataAttr":$profilingMetadata)
        >,
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = mlir::cast<vpux::NDTypeInterface>(getType()).getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ExtractFlatSlice
//

def VPUIP_ExtractFlatSliceOp :
        VPUIP_Op<
            "ExtractFlatSlice",
            [
                Pure,
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface>
            ]
        > {
    let summary = "Operation that extracts flat slice from DistributedBuffer";

    let description = [{
        This operation provides view to single slice from compact DistributedBuffer on the tiling dim.
        In contrast to SubView, this operation can take buffer slice from single cluster.
        For example, !VPUIP.DistributedBuffer<1x32x16x128xf16, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]>
        And offset is 20, which effectively means, that view will be memref<1x1x16x128xf16, [@CMX_NN, 2]>, because
        dims are distributed as 8+8+8+8 and 20 is 8+8+4, which means slice is located in 2 cluster
    }];

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$source,
        IntAttr:$offset,
        DefaultValuedAttr<IntAttr, "1">:$length
    );

    let results = (outs
        AnyMemRef:$result
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $source `:` type($source) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        Byte getByteOffset();
    }];

}

//
// ProfilingSectionInfoOp
//

def VPUIP_ProfilingSectionOp :
        VPUIP_Op<
            "ProfilingSection",
            [
                IsolatedFromAbove
            ]
        > {
    let summary = "Information about memory offset and size of engine specific profiling section";

    let description = [{
        This operation is bound to `net.DataInfo` operation and holds information about profiling section:
          * Engine name
          * Start offset(bytes)
          * Section size(bytes)
    }];

    let arguments = (ins
        IntAttr:$sectionType,
        IntAttr:$offset,
        IntAttr:$size
    );

    let assemblyFormat = [{
        `type` $sectionType `:` $size `bytes` `from` $offset
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// GroupBoundedBufferOp
//
def VPUIP_GroupBoundedBufferOp :
        VPUIP_Op<"GroupBoundedBuffer",
            [
                Pure,
                DotInterface,
                DeclareOpInterfaceMethods<GroupedViewOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
            ]
        > {
    let summary = "Creates a BoundedBuffer type by combining upper-bounded data buffer and a buffer with dynamic shape";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$data,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$dynamicShape
    );

    let results = (outs
        VPUIP_BoundedBuffer:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// UngroupBoundedBufferOp
//

def VPUIP_UngroupBoundedBufferOp :
        VPUIP_Op<
            "UngroupBoundedBuffer",
            [
                Pure,
                DotInterface,
                DeclareOpInterfaceMethods<MultiViewOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
            ]
        > {
    let summary = "Unpacks BoundedBuffer type into upper-bounded buffer and its dynamic shape buffer";

    let arguments = (ins
        VPUIP_BoundedBuffer:$input
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$data,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$dynamicShape
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// VPUIP_UpsamplingOp
//

def VPUIP_UpsamplingOp :
        VPUIP_Op<"Upsampling",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<VPUIP_LayerOpInterface>,
                MemoryEffectsOpInterface
            ]
        > {
    let summary = "Upsampling VPUIP layer";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        I64ArrayAttr:$upsampling_factor,
        OptionalAttr<IE_UpsamplingPadAttr>:$pad
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::SHAVE_NN;
        }
    }] # VPUIP_InferLayerReturnTypes<1>.impl;
}


//
// BarProgDMAOp
//

def VPUIP_BarProgDMAOp :
        VPUIP_TaskOp<1, "BarProgDMA",
            [
                ViewLikeOpInterface,
                VPUIP_DMATypeOpInterface,
                DotInterface
            ]
        > {
    let summary = "Barrier programming DMA task";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,
        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata,
        VPUIP_PhysicalBarrierRangeAttr: $physical_barrier_range
    );

    let results = (outs
        AnyMemRef:$output
    );

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `physical_barrier_range` `(` $physical_barrier_range `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// FetchDMAOp
//

def VPUIP_FetchDMAOp :
        VPUIP_TaskOp<1, "FetchDMA",
            [
                ViewLikeOpInterface,
                VPUIP_DMATypeOpInterface,
                DotInterface
            ]
        > {
    let summary = "Fetch task descriptors DMA task";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,
        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata,
        VPUIP_FetchDMAAttr:$fetch_dma
    );

    let results = (outs
        AnyMemRef:$output
    );

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `fetch_dma` `(` $fetch_dma `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// EnqueueDMAOp
//

def VPUIP_EnqueueDMAOp :
        VPUIP_TaskOp<1, "EnqueueDMA",
            [
                ViewLikeOpInterface,
                VPUIP_DMATypeOpInterface,
                DotInterface
            ]
        > {
    let summary = "Task enqueue DMA operation";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,
        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata,
        VPUIP_EnqueueDMAAttr: $enqueue_dma_attr
    );

    let results = (outs
        AnyMemRef:$output
    );

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `enqueue_dma_attr` `(` $enqueue_dma_attr `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

#endif
