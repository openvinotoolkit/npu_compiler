//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPUIP_ATTRIBUTES

include "vpux/compiler/dialect/VPUIP/attr_interfaces.td"
include "vpux/compiler/dialect/core/attr_interfaces.td"
include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"

//
// Base classes
//

class VPUIP_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 0;
}

class VPUIP_I64BitEnumAttr <string name, string summary, list<I64BitEnumAttrCase> cases> :
        I64BitEnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 0;
}

class VPUIP_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<VPUIP_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

class VPUIP_Attr<string name, list<Trait> traits = []> :
        AttrDef<VPUIP_Dialect, name, traits> {
    let mnemonic = name # "Attr";
    let assemblyFormat = "`<` struct(params) `>`";
}

//
// PoolLayerType
//

def VPUIP_PoolLayerType :
    VPUIP_I64EnumAttr<
        "PoolLayerType",
        "Type of Pooling layer",
        [
            I64EnumAttrCase<"MAX", 0>,
            I64EnumAttrCase<"AVG", 1>,
        ]
    > {
}

def VPUIP_PoolLayerTypeAttr : VPUIP_EnumAttr<VPUIP_PoolLayerType, "pool_layer_type">;

//
// NCETaskType
//

def VPUIP_NCETaskType :
    VPUIP_I64EnumAttr<
        "NCETaskType",
        "NCE task type",
        [
            I64EnumAttrCase<"CONV",             0>,
            I64EnumAttrCase<"DWCONV",           1>,
            I64EnumAttrCase<"MAXPOOL",          2>,
            I64EnumAttrCase<"AVEPOOL",          3>,
            I64EnumAttrCase<"ELTWISE",          4>,
            I64EnumAttrCase<"IDENTITY",         5>,
            I64EnumAttrCase<"REDUCEMEAN",       6>,
            I64EnumAttrCase<"REDUCESUMSQUARE",  7>,
            I64EnumAttrCase<"REDUCESUM",        8>,
        ]
    > {
}

def VPUIP_NCETaskTypeAttr : VPUIP_EnumAttr<VPUIP_NCETaskType, "nce_task_type">;

//
// SparsityCompressionAttr
//

def VPUIP_SparsityCompressionAttr : VPUIP_Attr<"SparsityCompression", [DeclareAttrInterfaceMethods<HwSpecificMemRefFieldInterface>]> {
    let description = [{
        Represents the compression as the number of elements along a specified axis.

        For example, a two-dimensional type with the shape 4x30 might be compressed
        along axis 0 into with the number of elements [12, 15, 30, 3].

        In case the compression is over the entire data (instead of a specified axis),
        the `axis` attribute can be set to null with the `numElems` as a splat value.

        The `alignment` attribute can be used to represent a required alignment for
        each set of elements on the given axis. For example, in case the compression
        for weights sparsity is represented by this attribute, the compression will
        be over the output channel axis and each weight set (i.e. ICxKYxKX - set of
        values for each output channel) has to be aligned to 16 bytes.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$axis,
        "mlir::ElementsAttr":$numElems,
        "mlir::IntegerAttr":$alignment
    );

    let extraClassDeclaration = [{
        int64_t getTotalNumElems() const;
        int64_t getNumElemsInRange(int64_t startIdx, int64_t size) const;
        Byte getAllocSize(mlir::Type elemType) const;
    }];

    let assemblyFormat = "`<` struct(params) `>`";
}

def VPUIP_SwizzlingSchemeAttr : VPUIP_Attr<"SwizzlingScheme", [DeclareAttrInterfaceMethods<HwSpecificMemRefFieldInterface>]> {
    let description = [{
        This contains information about all the swizzling related requirements for the buffer

        The 'key' is the swizzling key to be used, these keys refers to 5 swizzling patterns that determines how much striping is applied
        Higher swizzling shows higher performance but needs larger alignment/padding size for each allocation depending on the architecture
        The 'sizeAlignment' is the requirement for the buffer size dictated by the architecture
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$key,
        "mlir::IntegerAttr":$sizeAlignment
    );
    let assemblyFormat = "`<` struct(params) `>`";
}

//
// DMADescriptorAttr
//

def VPUIP_DMADescriptorAttr : VPUIP_Attr<"DMADescriptor"> {
    let parameters = (ins
        "mlir::IntegerAttr":$numPlanes,
        "mlir::IntegerAttr":$len,
        "mlir::IntegerAttr":$srcWidth,
        "mlir::IntegerAttr":$srcStride,
        "mlir::IntegerAttr":$srcPlaneStride,
        "mlir::IntegerAttr":$dstWidth,
        "mlir::IntegerAttr":$dstStride,
        "mlir::IntegerAttr":$dstPlaneStride
    );
}

//
// DmaChannelType
//

def VPUIP_DmaChannelType :
        I64EnumAttr<
            "DmaChannelType",
            "DMA channel type",
            [
                I64EnumAttrCase<"DDR", 0>,
                I64EnumAttrCase<"CMX", 1>,
                I64EnumAttrCase<"NOT_SPECIFIED", 2>
            ]
        > {
    let cppNamespace = "vpux::VPUIP";

    let description = [{
      In VPU different DMA channel receives data movement job from link agent.
      Different channel is used based on transaction source:
       - DDR channel - DMA SRC is DDR
       - CMX channel - DMA SRC is CMX or HW register"
    }];
}

//
// GatherAddressingMode
//

def VPUIP_GatherAddressingMode :
        I64EnumAttr<
            "GatherAddressingMode",
            "Gather Addressing Mode",
            [
                I64EnumAttrCase<"ABSOLUTE", 0>,
                I64EnumAttrCase<"INDEXED", 1>
            ]
        > {
    let cppNamespace = "vpux::VPUIP";

    let description = [{
      Gather operation done by DMA engine can have two addressing modes:
        - Absolute addressing - each element in the indices list represents the raw address value
                                that is used by the DMA for accessing memory
        - Indexed addressing - each element in the indices list represents the index
                                to access into a Virtual Array in memory
    }];
}

//
// DPUHaloRegionAttr
//

def VPUIP_DPUHaloRegionAttr : VPUIP_Attr<"DPUHaloRegion"> {
    let parameters = (ins
        "mlir::IntegerAttr":$xStart,
        "mlir::IntegerAttr":$xEnd,
        "mlir::IntegerAttr":$yStart,
        "mlir::IntegerAttr":$yEnd,
        "mlir::IntegerAttr":$zStart,
        "mlir::IntegerAttr":$zEnd,
        "mlir::IntegerAttr":$targetOffset,
        "mlir::ArrayAttr":$targetClusters,
        OptionalParameter<"mlir::IntegerAttr">:$sparsityOffset,
        "mlir::IntegerAttr":$targetWidth
    );
}

def DPUHaloRegionArrayAttr : TypedArrayAttrBase<VPUIP_DPUHaloRegionAttr, "Array of DPU Halo Regions">;

//
// HaloRegionAttr
//

def VPUIP_HaloRegionAttr : VPUIP_Attr<"HaloRegion"> {
    let parameters = (ins
        "mlir::ArrayAttr":$shape,
        "mlir::ArrayAttr":$offset,
        "mlir::IntegerAttr":$cluster_id
    );
}

def HaloRegionArrayAttr : TypedArrayAttrBase<VPUIP_HaloRegionAttr, "Array of Halo Regions">;

//
// OutwardHaloRegionAttr
//

def VPUIP_OutwardHaloRegionAttr : AttrDef<VPUIP_Dialect,"OutwardHaloRegion",[]> {

    let mnemonic = "OutwardHaloRegionAttr";

    let parameters = (ins
        "mlir::ArrayAttr":$shape,
        "mlir::ArrayAttr":$offset,
        "mlir::IntegerAttr":$cluster_id,
        "mlir::ArrayAttr":$inwardHaloRegions
    );

    let hasCustomAssemblyFormat = 1;
}

//
// LocalRegionAttr
//

def VPUIP_LocalRegionAttr : VPUIP_Attr<"LocalRegion"> {
    let description = [{
        Attribute attached to NCEClusterTask operation. Represents a region that is outputted only on the local tile,
        and only what falls into the region gets written. Output can also be fully suppressed on the local tile.
        Used in case of SOK/SOH/SOW/SOHW transitions.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$xStart,
        "mlir::IntegerAttr":$xEnd,
        "mlir::IntegerAttr":$yStart,
        "mlir::IntegerAttr":$yEnd,
        OptionalParameter<"mlir::UnitAttr">:$suppressOutput
    );
}

//
// DMA Acceleration mode
//

def VPUIP_DMAAccMode :
    VPUIP_I64EnumAttr<
        "DMAAccMode",
        "DMA Acceleration Mode",
        [
            I64EnumAttrCase<"DISABLE",       0>,
            I64EnumAttrCase<"COMPRESSION",   1>,
            I64EnumAttrCase<"DECOMPRESSION", 2>,
            I64EnumAttrCase<"WEIGHT_PREP",   3>
        ]
    > {
}

def VPUIP_DMAAccModeAttr : VPUIP_EnumAttr<VPUIP_DMAAccMode, "dma_cc_mode">;

//
// DPU Profiling Metadata
//

def VPUIP_DpuProfilingMetadataAttr : VPUIP_Attr<"DpuProfilingMetadata", [
        DeclareAttrInterfaceMethods<VPUIP_ProfilingMetadataAttrInterface>
    ] > {
    let description = [{
        Attribute attached to NCEClusterTask operation. Represent related to the task profiling memory layout
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$bufferId,
        "mlir::IntegerAttr":$taskId,
        "mlir::IntegerAttr":$maxVariants,
        OptionalParameter<"mlir::IntegerAttr">:$numVariants,
        OptionalParameter<"mlir::IntegerAttr">:$clusterId
    );

    let assemblyFormat = "`<` struct(params) `>`";
}

//
// DMA Profiling Metadata
//

def VPUIP_DmaProfilingMetadataAttr : VPUIP_Attr<"DmaProfilingMetadata", [
        DeclareAttrInterfaceMethods<VPUIP_ProfilingMetadataAttrInterface>
    ] > {
    let description = [{
        Attribute attached to operation to keep DMA profiling information
    }];

    let parameters = (ins
        OptionalParameter<"mlir::IntegerAttr">:$dataIndex,  // The position in the profiling output binary
        OptionalParameter<"mlir::UnitAttr">:$profBegin      // Marker of profBegin task
    );

    let assemblyFormat = "`<` ( `dataIndex` `=` $dataIndex^ ):(`profBeginFlag` $profBegin)? `>`";
}

//
// ActProfilingMetadata
//

def VPUIP_SwProfilingMetadataAttr : VPUIP_Attr<"SwProfilingMetadata", [
        DeclareAttrInterfaceMethods<VPUIP_ProfilingMetadataAttrInterface>
    ] > {
    let description = [{
        Attribute attached to opearion to keep Act profiling information
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$bufferId,                      // ID of high level buffer
        "mlir::IntegerAttr":$bufferOffset,                  // Offset from beginning of profiling buffer to current
        "mlir::IntegerAttr":$clusterSize,                   // Size of one CMX tile corresponding to this buffer
        "mlir::IntegerAttr":$dataIndex,                     // Index of data in current buffer
        OptionalParameter<"mlir::IntegerAttr">:$tileId,     // Index of tile(similar to workload_id)
        OptionalParameter<"mlir::IntegerAttr">:$clusterId   // clusterId
    );

    let assemblyFormat = "`<` struct(params) `>`";
}

//
// M2IProfilingMetadata
//

def VPUIP_M2IProfilingMetadataAttr : VPUIP_Attr<"M2IProfilingMetadata", [
        DeclareAttrInterfaceMethods<VPUIP_ProfilingMetadataAttrInterface>
    ] > {
    let description = [{
        Attribute attached to opearion to keep M2I profiling information
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$bufferId,                      // ID of high level buffer
        "mlir::IntegerAttr":$bufferOffset                   // Offset from beginning of profiling buffer to current
    );

    let assemblyFormat = "`<` struct(params) `>`";
}

def VPUIP_CompressionState : VPUIP_I64EnumAttr<"CompressionState", "Buffer compression state", [
    I64EnumAttrCase<"NoCompression", 0>,
    I64EnumAttrCase<"CompressionCandidate", 1>,
    I64EnumAttrCase<"RuntimeCompressed", 2>,
    I64EnumAttrCase<"CompiletimeCompressed", 3>
    ]>{}

def VPUIP_CompressionStateAttr : VPUIP_EnumAttr<VPUIP_CompressionState, "Compression", [
        DeclareAttrInterfaceMethods<HwSpecificMemRefFieldInterface>
    ]> {
        let description = [{
            This attribute is attached to the MemRef type to indicate if it was compressed.
            Distinction is made between compile-time compression and runtime compression to account
            for worst case scenario during runtime.
        }];
    }


//
// VPUIPInlinerDispatch
//

// This attribute controls the inlining behaviour for 'func' operations. A 'func' operation can be annotated
// with an attribute named "inliner_dispatch": {"inliner_dispatch" = #VPUIP.VPUIPInlinerDispatchAttr}.
// See 'InlinerDispatchAttr' for more information.
def VPUIPInlinerDispatchAttr : InlinerDispatchAttr<VPUIP_Dialect, "VPUIPInlinerDispatch">;

//
// VPUIPObjectAttr
//

def VPUIP_ObjectAttr : VPUIP_Attr<"Object"> {
    let description = [{
        VPUIP object holds the hex string of serialized NPU compute functions in the ELF dialect.
    }];
    let parameters = (ins
        "mlir::StringAttr":$object
    );
}

//
// VPUIPInternalDataFlow
//

// This attribute is used to describe the internal data flow of an op while leaving its input and output operands untouched.
// This is useful when the operands need to remain in their original form for other ops.
// For example, a SpaceToDepthDMAOp with mode = BLOCKS_FIRST and block_size = 2
//  - input: memref<1x4x2x12xf16, #NHWC>
//  - output: memref<1x16x1x6xf16, #NHWC>
// behaves internally like a permute DMA with
//  - input: memref<1x4x1x2x6x2xf16, #map0>, where #map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
//  - output: memref<1x2x2x4x1x6xf16, #map1> where #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
//
// The mapping order describes how each logical dim from the input maps to a logical dim from the output.
// For the above example, we have
//  - mappingOrder: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>
//
// The loop order describes the order in which the input logical dims should be iterated over.
// For the above example, for a simple continuous iteration, we have
//  - loopOrder: #map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>

def VPUIP_InternalDataFlow : VPUIP_Attr<"InternalDataFlow"> {
    let description = [{
        Attribute describing internal data flow of an Op.
    }];

    let parameters = (ins
        "mlir::Type":$inputType,
        "mlir::Type":$outputType,
        "mlir::AffineMapAttr":$mappingOrder,    // Order representing the transpose from input to output
        "mlir::AffineMapAttr":$loopOrder        // Order in which dimensions from input and output will be iterated over
    );

    let assemblyFormat = "`<` struct(params) `>`";
}

//
// VPUIP_PhysicalBarrierRangeAttr
//

def VPUIP_PhysicalBarrierRangeAttr : VPUIP_Attr<"PhysicalBarrierRange"> {
    let description = [{
        "Represents a range of PIDs: [start, end]"
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$start,
        "mlir::IntegerAttr":$end
    );

    let extraClassDeclaration = [{
        /// Returns the number of PIDs in this range (inclusive).
        int64_t getPidCount() const {
            return getEnd().getInt() - getStart().getInt() + 1;
        };

        /// Returns the first PID in this range
        int64_t getFirstPid() const {
            return getStart().getInt();
        };
    }];

    let assemblyFormat = "`<` $start `to` $end `>`";
}

def VPUIP_DynamicSkipMode :
    VPUIP_I64EnumAttr<
        "DynamicSkipMode",
        "Dynamic skip mode",
        [
            I64EnumAttrCase<"DISABLED",  0>,
            I64EnumAttrCase<"SPATIAL_ONLY", 1>,
            I64EnumAttrCase<"SPATIAL_AND_OUTPUT_CHANNELS",  2>,
            I64EnumAttrCase<"SPATIAL_AND_INPUT_CHANNELS",   3>
        ]
    > {
}

def VPUIP_DynamicSkipModeAttr : VPUIP_EnumAttr<VPUIP_DynamicSkipMode, "dynamic_skip_mode">;

def VPUIP_DynamicAdjustMode :
        VPUIP_I64BitEnumAttr<
            "DynamicAdjustMode",
            "Dynamic adjust mode",
            [
                I64BitEnumAttrCase<"DISABLED",          0x0>,
                I64BitEnumAttrCase<"OUTPUT_CHANNELS",   0x1>,
                I64BitEnumAttrCase<"INPUT_CHANNELS",    0x2>
            ]
        > {
}

def VPUIP_DynamicAdjustModeAttr : VPUIP_EnumAttr<VPUIP_DynamicAdjustMode, "dynamic_adjust_mode">;

def VPUIP_DynamicScaleConfigAttr : VPUIP_Attr<"DynamicScaleConfig"> {
    let description = [{
        Config for NCE dynamic scaling of compute

        dynamicSkipMode: config of how the tasks should be skipped; either on a invariant or variant basis
        dynamicAdjustMode: config of how the tasks should be adjusted; mostly on a invariant level

        ppeOverwriteData: if present, data which is still computed yet invalid, will be overwritten with
        this pattern
        invariantPrimitiveId: if present, use primitives and dynamic scaling feature on invariant level
    }];

    let parameters = (ins
        "DynamicSkipModeAttr":$dynamicSkipMode,
        "DynamicAdjustModeAttr":$dynamicAdjustMode,

        OptionalParameter<"mlir::IntegerAttr">:$ppeOverwriteData,
        OptionalParameter<"mlir::ArrayAttr">:$invariantPrimitiveId // IDs for invariant primitives A and B
    );
    let assemblyFormat = "`<` struct(params) `>`";
}

//
// VPUIP_FetchDMAAttr
//

def VPUIP_FetchDMAAttr : VPUIP_Attr<"FetchDMA"> {
    let description = [{
        "Represents attributes required to correctly lower with NNDMAs for fetch task descriptors"
    }];

    let parameters = (ins
        "vpux::VPU::ExecutorKindAttr":$targetExecutorKindAttr,
        "mlir::IntegerAttr":$tileIdx,
        "mlir::IntegerAttr":$listIdx,
        "mlir::IntegerAttr":$execGroupIdx
    );

    let assemblyFormat = [{
        `<`
            $targetExecutorKindAttr `,`
            `tile` `=` $tileIdx `,`
            `list` `=` $listIdx `,`
            `group` `=` $execGroupIdx
        `>`
    }];
}

//
// VPUIP_EnqueueDMAAttr
//

def VPUIP_EnqueueDMAAttr : VPUIP_Attr<"EnqueueDMA"> {
    let description = [{
        "Represents attributes required for identifying what operation is to be enqueued using this DMA"
    }];

    let parameters = (ins
        "vpux::VPU::ExecutorKindAttr":$targetExecutorKindAttr,
        "mlir::IntegerAttr":$tileIdx,
        "mlir::IntegerAttr":$listIdx,
        "mlir::IntegerAttr":$startTaskIdx,
        "mlir::IntegerAttr":$endTaskIdx
    );

    let assemblyFormat = [{
        `<` $targetExecutorKindAttr `,`
            `tile` `=` $tileIdx `,`
            `list` `=` $listIdx `,`
            `startTask` `=` $startTaskIdx `,`
            `endTask` `=` $endTaskIdx `>`
    }];
}

#endif
