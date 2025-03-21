//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPU_ATTRIBUTES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/VPU/attr_interfaces.td"
include "vpux/compiler/dialect/VPU/dialect.td"

//
// Base classes
//

class VPU_Attr<string name, list<Trait> traits = []> : AttrDef<VPU_Dialect, name, traits> {
    let mnemonic = name;
    let assemblyFormat = "`<` struct(params) `>`";
}

class VPU_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 0;
}

class VPU_I64BitEnumAttr <string name, string summary, list<I64BitEnumAttrCase> cases> :
        I64BitEnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 0;
}

class VPU_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<VPU_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

//
// ArchKind
//

def VPU_ArchKind :
        VPU_I64EnumAttr<
            "ArchKind",
            "Represents VPU architecture generation",
            [
                I64EnumAttrCase<"UNKNOWN", 0>,
                I64EnumAttrCase<"NPU37XX", 3>,
                I64EnumAttrCase<"NPU40XX", 4>,
            ]
        > {
}

def VPU_ArchKindAttr : VPU_EnumAttr<VPU_ArchKind, "arch_kind">;

//
// MemoryKind
//

def VPU_MemoryKind :
        VPU_I64EnumAttr<
            "MemoryKind",
            "Represents the actual hardware memory hierarchy",
            [
                I64EnumAttrCase<"DDR",      0>,
                I64EnumAttrCase<"CSRAM",    1>,
                I64EnumAttrCase<"CMX_NN",   2>,
                I64EnumAttrCase<"Register", 3>
            ]
        > {
}

def VPU_MemoryKindAttr : VPU_EnumAttr<VPU_MemoryKind, "memory_kind">;

//
// ActShaveTaskType
//

def VPU_ActShaveTaskType:
        VPU_I64EnumAttr<
            "ActShaveTaskType",
            "Activation SHAVE Task Type",
            [
                I64EnumAttrCase<"COMPUTE",                0>,
                I64EnumAttrCase<"CACHE_FLUSH",            1>,
                I64EnumAttrCase<"CACHE_INVALIDATE",       2>,
                I64EnumAttrCase<"CACHE_FLUSH_INVALIDATE", 3>,
                I64EnumAttrCase<"CACHE_PREFETCH",         4>
            ]
        > {
}

def VPU_ActShaveTaskTypeAttr : VPU_EnumAttr<VPU_ActShaveTaskType, "task_type">;

//
// ExecutorKind
//

def VPU_ExecutorKind :
        VPU_I64EnumAttr<
            "ExecutorKind",
            "Representd hardware executror resources",
            [
                I64EnumAttrCase<"DMA_NN",    0>,
                I64EnumAttrCase<"NCE",       1>,
                I64EnumAttrCase<"DPU",       2>,
                I64EnumAttrCase<"SHAVE_NN",  3>,
                I64EnumAttrCase<"SHAVE_ACT", 4>,
                I64EnumAttrCase<"M2I",       5>,
                I64EnumAttrCase<"UNKNOWN",   6>,
            ]
        > {
}

def VPU_ExecutorKindAttr : VPU_EnumAttr<VPU_ExecutorKind, "executor_kind">;

//
// CompilationMode
//

def VPU_CompilationMode  :
        VPU_I64EnumAttr<
            "CompilationMode",
            "Compilation Mode",
            [
                I64EnumAttrCase<"ReferenceSW",  0>,
                I64EnumAttrCase<"DefaultHW",    1>,
                I64EnumAttrCase<"ShaveCodeGen", 2>
            ]
        > {
}

def VPU_CompilationModeAttr : VPU_EnumAttr<VPU_CompilationMode, "compilation_mode">;

//
// RevisionID
//

def VPU_RevisionID  :
        VPU_I64EnumAttr<
            "RevisionID",
            "Revision ID",
            [
                I64EnumAttrCase<"REVISION_A0",  0>,
                I64EnumAttrCase<"REVISION_A1",  1>,
                I64EnumAttrCase<"REVISION_A3",  2>,
                I64EnumAttrCase<"REVISION_B",   3>,
                I64EnumAttrCase<"REVISION_C",   4>,
                I64EnumAttrCase<"REVISION_D",   5>,
                I64EnumAttrCase<"REVISION_K",   6>,
                I64EnumAttrCase<"REVISION_NONE",   7>
            ]
        > {
}

def VPU_RevisionIDAttr : VPU_EnumAttr<VPU_RevisionID, "revision_id">;

//
// VFScenario
//

def VPU_VFScenario :
        VPU_I64EnumAttr<
            "VFScenario",
            "Scenarios for VF scheduling",
            [
                I64EnumAttrCase<"MINIMAL", 0>,
                I64EnumAttrCase<"LASTOP_PREFETCHING", 1>,
                I64EnumAttrCase<"WEIGHTS_PREFETCHING", 3>,
                I64EnumAttrCase<"FULL_PREFETCHING", 4>,
                I64EnumAttrCase<"VF_PIPELINING", 5>
            ]
        > {
}

def VPU_VFScenarioAttr : VPU_EnumAttr<VPU_VFScenario, "vf_scenario">;

//
// SparsitySupport
//

def VPU_SparsitySupport :
        I32BitEnumAttr<
            "SparsitySupport",
            "Sparsity support of an operation",
            [
                I32BitEnumAttrCase<"NONE",           0x0>,
                I32BitEnumAttrCase<"SPARSE_INPUTS",  0x1>,
                I32BitEnumAttrCase<"SPARSE_OUTPUTS", 0x2>,
                I32BitEnumAttrCase<"SPARSE_WEIGHTS", 0x4>
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// ActivationSparsityProfile
//

def VPU_ActivationSparsityProfile :
        VPU_I64EnumAttr<
            "ActivationSparsityProfile",
            "Represents desired activation sparsity profile",
            [
                I64EnumAttrCase<"S0", 0>,      // Only for ops where runtime sparsity is possible
                I64EnumAttrCase<"S1", 1>       // As much as possible
            ]
        > {
}

def VPU_ActivationSparsityProfileAttr : VPU_EnumAttr<VPU_ActivationSparsityProfile, "activation_sparsity_profile">;

//
// WeightsSparsityHeuristic
//

def VPU_WeightsSparsityHeuristic :
        VPU_I64EnumAttr<
            "WeightsSparsityHeuristic",
            "Selects the weights sparsity heuristic which compares the sparse values ration to a threshold",
            [
                I64EnumAttrCase<"RATIO", 0>,    // Fixed threshold based on the element type
                I64EnumAttrCase<"CMX",   1>     // Threshold is decided based on the CMX usage of the weights
            ]
        > {
}

def VPU_WeightsSparsityHeuristicAttr : VPU_EnumAttr<VPU_WeightsSparsityHeuristic, "weights_sparsity_heuristic">;

//
// EltwiseType
//

def VPU_EltwiseType :
        VPU_I64EnumAttr<
            "EltwiseType",
            "Type of Eltwise operation",
            [
                I64EnumAttrCase<"ADD",           0>,
                I64EnumAttrCase<"SUBTRACT",      1>,
                I64EnumAttrCase<"MULTIPLY",      2>,
                I64EnumAttrCase<"DIVIDE",        3>,
                I64EnumAttrCase<"SQUARED_DIFF",  4>,
                I64EnumAttrCase<"POWER",         5>,
                I64EnumAttrCase<"FLOOR_MOD",     6>,
                I64EnumAttrCase<"MOD",           7>,
                I64EnumAttrCase<"MIN",           8>,
                I64EnumAttrCase<"MAX",           9>,
                I64EnumAttrCase<"AND",           10>,
                I64EnumAttrCase<"EQUAL",         11>,
                I64EnumAttrCase<"LESS",          12>,
                I64EnumAttrCase<"LESS_EQUAL",    13>,
                I64EnumAttrCase<"NOT_EQUAL",     14>,
                I64EnumAttrCase<"GREATER",       15>,
                I64EnumAttrCase<"GREATER_EQUAL", 16>,
                I64EnumAttrCase<"LOGICAL_NOT",   17>,
                I64EnumAttrCase<"LOGICAL_OR",    18>,
                I64EnumAttrCase<"LOGICAL_XOR",   19>
            ]
        > {
}

def VPU_EltwiseTypeAttr : VPU_EnumAttr<VPU_EltwiseType, "eltwise_type">;

//
// ReduceType
//

def VPU_ReduceType :
        VPU_I64EnumAttr<
            "ReduceType",
            "Type of Reduce operation",
            [
                I64EnumAttrCase<"MEAN",         0>,
                I64EnumAttrCase<"SUM",          1>,
                I64EnumAttrCase<"MAX",          2>,
                I64EnumAttrCase<"MIN",          3>,
                I64EnumAttrCase<"SUMOFSQUARES", 4>
            ]
        > {
}

def VPU_ReduceTypeAttr : VPU_EnumAttr<VPU_ReduceType, "reduce_type">;

//
// PaddingAttr
//

def VPU_PaddingAttr : VPU_Attr<"Padding"> {
    let parameters = (ins
        "mlir::IntegerAttr":$left,
        "mlir::IntegerAttr":$right,
        "mlir::IntegerAttr":$top,
        "mlir::IntegerAttr":$bottom
    );
}

//
// MPEMode
//

def VPU_MPEMode :
        VPU_I64EnumAttr<
            "MPEMode",
            "MPE Mode",
            [
                I64EnumAttrCase<"VECTOR",       0>,
                I64EnumAttrCase<"MATRIX",       1>,
                I64EnumAttrCase<"VECTOR_FP16",  2>,
                I64EnumAttrCase<"CUBOID_16x16", 3>,
                I64EnumAttrCase<"CUBOID_8x16",  4>,
                I64EnumAttrCase<"CUBOID_4x16",  5>,
                I64EnumAttrCase<"DCIM",         6>,
                I64EnumAttrCase<"NOP",          7>
            ]
        > {
}

def VPU_MPEModeAttr : VPU_EnumAttr<VPU_MPEMode, "mpe_mode">;

//
// PPEMode
//

def VPU_PPEMode :
        VPU_I64EnumAttr<
            "PPEMode",
            "Post Processing Element Type",
            [
                // Low-level instructions
                I64EnumAttrCase<"STORE",   0>,
                I64EnumAttrCase<"LOAD",    1>,
                I64EnumAttrCase<"CLEAR",   2>,
                I64EnumAttrCase<"NOOP",    3>,
                I64EnumAttrCase<"HALT",    4>,

                // Element-Wise Operations
                I64EnumAttrCase<"ADD",     5>,
                I64EnumAttrCase<"SUB",     6>,
                I64EnumAttrCase<"MULT",    7>,
                I64EnumAttrCase<"MAXIMUM", 8>,
                I64EnumAttrCase<"MINIMUM", 9>,
                I64EnumAttrCase<"AND",     10>,
                I64EnumAttrCase<"OR",      11>,
                I64EnumAttrCase<"XOR",     12>,

                // Activations
                I64EnumAttrCase<"LRELU",   13>,
                I64EnumAttrCase<"LRELUX",  14>,
                I64EnumAttrCase<"LPRELU",  15>,
                I64EnumAttrCase<"CEIL",    16>,
                I64EnumAttrCase<"FLOOR",   17>,
                I64EnumAttrCase<"POW",     18>,
                I64EnumAttrCase<"EXP",     19>,
                I64EnumAttrCase<"SIGMOID", 20>,
                I64EnumAttrCase<"TANH",    21>,
                I64EnumAttrCase<"SQRT",    22>,
                I64EnumAttrCase<"RSQRT",   23>,
                I64EnumAttrCase<"FLEXARB", 24>,
                I64EnumAttrCase<"NOT",     25>,
                I64EnumAttrCase<"ABS",     26>,
                I64EnumAttrCase<"NEG",     27>,
                I64EnumAttrCase<"SIN",     28>,
                I64EnumAttrCase<"COS",     29>
            ]
        > {
}

def VPU_PPEModeAttr : VPU_EnumAttr<VPU_PPEMode, "ppe_mode">;

//
// PPEAttr
//

class VPU_PPEAttrBase<Pred condition, string summary> : Attr<condition, summary> {
    let storageType = [{ vpux::VPU::PPEAttr }];
    let returnType = [{ vpux::VPU::PPEAttr }];
    let convertFromStorage = "$_self";
}

def VPU_PPEAttr : VPU_PPEAttrBase<CPred<"mlir::isa<vpux::VPU::PPEAttr>($_self)">,
                                      "PPE attribute"> {
        string cppType = "vpux::VPU::PPEAttr";
}

def VPU_PPEStubAttr : VPU_Attr<"PPEStub", [
        DeclareAttrInterfaceMethods<VPU_PPEAttrInterface>
    ] > {
    let description = [{
        Empty PPE attribute.
        Useful for testing and other situations where setting a concrete PPE attribute is unfeasible.
    }];

    let assemblyFormat = "`<` struct(params) `>`";
}

def VPU_PPEIntAttr : VPU_Attr<"PPEInt", [
        DeclareAttrInterfaceMethods<VPU_PPEAttrInterface>
    ] > {
    let description = [{
        Legacy mode for VPU PPE attribute, storing parameters for both int and float PPE operations.
        Since the PPE attribute maps directly to the HW config table, some missing fields are later replaced by mathematically neutral values.

        Neutral values:
        - `mode`: NOOP
        - (`clamp_low`, `clamp_high`): (std::numeric_limits<int_32>::min(), std::numeric_limits<int_32>::max())
        - (`lrelu_mult`, `lrelu_shift`): (1.0, 0.0)
        - `fp_lrelu_alpha`: 1.0

        The exact usage of the parameters inside PPE HW can be found in the PPE section of the compiler docs.
    }];
    let parameters = (ins
        "vpux::VPU::PPEModeAttr":$mode,
        OptionalParameter<"mlir::IntegerAttr">:$clamp_low,
        OptionalParameter<"mlir::IntegerAttr">:$clamp_high,
        OptionalParameter<"mlir::IntegerAttr">:$lrelu_mult,
        OptionalParameter<"mlir::IntegerAttr">:$lrelu_shift,
        OptionalParameter<"mlir::ArrayAttr">:$quant_scale,
        OptionalParameter<"mlir::ArrayAttr">:$quant_mult,
        OptionalParameter<"mlir::ArrayAttr">:$quant_shift,
        OptionalParameter<"mlir::IntegerAttr">:$quant_post_shift,
        OptionalParameter<"mlir::ArrayAttr">:$in1_quant_mult,
        OptionalParameter<"mlir::ArrayAttr">:$in2_quant_mult,
        OptionalParameter<"mlir::FloatAttr">:$fp_prelu_alpha
    );

    let assemblyFormat = "`<` struct(params) `>`";
}

def VPU_PPEFpAttr : VPU_Attr<"PPEFp", [
        DeclareAttrInterfaceMethods<VPU_PPEAttrInterface>
    ] > {
    let description = [{
        Stores parameters for the float/non-backwards-compatible PPE (Post Processing Engine) pipeline.

        Parameters:
        - `mode`: Specifies the post-op type which may influence how the other fields are used. Default: `NOOP`
        - `clamp_low`: Specifies the low clamp applied on the output. Default: `std::numeric_limits<float>::lowest()`
        - `clamp_high`: Specifies the high clamp applied on the output. Default: `std::numeric_limits<float>::max()`
        - `scale`: Specifies the per-tensor scale applied on the PPE input, in case of per-channel scale (stored in the WT) this field is set to null.
        - `prelu_alpha`: Specifies the leaky-relu alpha applied to negative values, may also be used as a per-tensor multiplier in certain modes. Default: `{1.0}`
        - `bias`: Specifies the per-tensor bias applied on the PPE input, in case of per-channel bias (stored in the WT) this field is set to null.
        - `adder`: Specifies the per-tensor zero-point applied on the PPE output. Default: `0.0`
        - `in1_mult`/`in2_mult`: Specifies the scales applied to the inputs of eltwise ops by the IDU. Default: `1.0`
        - `sprlut`: Enables and stores the SprLUT configuration table for SprLUT supported post-ops. Default: `null`

        The exact usage of the parameters inside PPE HW can be found in the PPE section of the compiler docs.
    }];
    let parameters = (ins
        "vpux::VPU::PPEModeAttr":$mode,
        "mlir::FloatAttr":$clamp_low,
        "mlir::FloatAttr":$clamp_high,
        OptionalParameter<"mlir::FloatAttr">:$scale,
        "mlir::ArrayAttr":$prelu_alpha,
        OptionalParameter<"mlir::FloatAttr">:$bias,
        "mlir::FloatAttr":$adder,
        OptionalParameter<"mlir::ArrayAttr">:$in1_mult,
        OptionalParameter<"mlir::ArrayAttr">:$in2_mult,
        OptionalParameter<"mlir::ElementsAttr">:$sprlut
    );

    let assemblyFormat = "`<` struct(params) `>`";
}

//
// MPEEngineAttr
//

class VPU_MPEEngineAttrBase<Pred condition, string summary> : Attr<condition, summary> {
    let storageType = [{ vpux::VPU::MPEEngineAttr }];
    let returnType = [{ vpux::VPU::MPEEngineAttr }];
    let convertFromStorage = "$_self";
}

def VPU_MPEEngineAttr : VPU_MPEEngineAttrBase<CPred<"$_self.isa<vpux::VPU::MPEEngineAttr>()">,
                                      "MPE Engine attribute"> {
        string cppType = "vpux::VPU::MPEEngineAttr";
    }

include "vpux/compiler/NPU37XX/dialect/VPU/attributes.td"

//
// MultiClusterStrategy
//

def VPU_MultiClusterStrategy :
        VPU_I64EnumAttr<
            "MultiClusterStrategy",
            "MultiCluster Strategy",
            [
                I64EnumAttrCase<"SplitOverHeight",           0>,
                I64EnumAttrCase<"SplitOverKernel",           1>,
                I64EnumAttrCase<"SplitOverWidth",            2>,
                I64EnumAttrCase<"Clustering",                3>,
                I64EnumAttrCase<"SplitOverHeightOverlapped", 4>,
                I64EnumAttrCase<"HKSwitch",                  5>,
                I64EnumAttrCase<"SplitOverHeightKernel",     6>,
                I64EnumAttrCase<"SplitOverHeightWidth",      7>,
                I64EnumAttrCase<"SplitOverBatch",            8>,
                I64EnumAttrCase<"SplitOverGroup",            9>,
            ]
        > {
}

def VPU_MultiClusterStrategyAttr : VPU_EnumAttr<VPU_MultiClusterStrategy, "multi_cluster_strategy">;

//
// MultiClusterOptimizationScope
//

def VPU_MCOptimizationScope :
        VPU_I64EnumAttr<
            "MCOptimizationScope",
            "Selects the scope of optimization to perform on multicluster strategy selection",
            [
                I64EnumAttrCase<"LOCAL",    0>,    // Greedy layer-wise strategy applied
                I64EnumAttrCase<"SUBGRAPH", 1>     // Subgraph optimization based on dynamically built subgraphs
            ]
        > {
}

def VPU_MCOptimizationScopeAttr : VPU_EnumAttr<VPU_MCOptimizationScope, "mc_optimization_scope">;

//
// NCE Interpolate
//

def VPU_NCEInterpolateMode :
        VPU_I64EnumAttr<
            "NCEInterpolateMode",
            "Specifies type of interpolation",
            [
                I64EnumAttrCase<"NEAREST",  0>,
                I64EnumAttrCase<"BILINEAR", 1>
            ]
        > {
}

def VPU_NCEInterpolateModeAttr : VPU_EnumAttr<VPU_NCEInterpolateMode, "nce_interpolate_mode">;

//
// DistributionMode
//

def VPU_DistributionMode :
        VPU_I64BitEnumAttr<
            "DistributionMode",
            "Tensor distribution mode",
            [
                I64BitEnumAttrCase<"NONE",         0x0>,
                I64BitEnumAttrCase<"OVERLAPPED",   0x1>,
                I64BitEnumAttrCase<"DUPLICATED",   0x2>,
                I64BitEnumAttrCase<"SEGMENTED",    0x4>,
                I64BitEnumAttrCase<"MULTICASTED",  0x8>
            ]
        > {
}

def VPU_DistributionModeAttr : VPU_EnumAttr<VPU_DistributionMode, "tensor_distribution_mode">;

//
// DistributionInfoAttr
//

/////////////////////////////
//
// mode - hint for how data and compute are distributed across clusters
//      * SEGMENTED - data and compute is split in a simple manner across clusters
//      not taking into account the overlap lines. This is possible for
//      input tensors in VPUX3XXX since HW can read the overlap lines from
//      neighboring clusters.
//          * for NPU40XX we continue to use SEGMENTED yet we take into account the overlap
//          lines, this is intended to be improve in the future, for easier understanding
//      * OVERLAPPED - data is split taking into account the overlap lines in
//      memory, compute does not differ from SEGMENTED. This is done for VPUX3XXX for the
//      layers which can't perform intercluster read accesses, and for all case in NPU40XX
//      and following.
//      * DUPLICATED - both data and compute are duplicated on the specified clusters.
//      * SEGMENTED | DUPLICATED - such a combination means compute is segmented across clusters
//      but the data will be duplicated with the feature of broadcast.
//      * SEGMENTED | MULTICASTED - similar to SEGMENTED | DUPLICATED but used for cases when we
//      broadcast the data and segment across the channel dimension. In future this will be removed
//      in favor of only SEGMENTED | DUPLICATED
// num_tiles - shape for tiling data and compute across clusters
//      * size of num_tiles is equal to the data shape size
//      * amount of tiling per one axis is usually equal to num_clusters
//      unless tiling is done on multiple axes
// kernel/pads/strides - parameters used to compute overlap lines in the case of
//      OVERLAP mode, logic works by infering a balanced split of the result buffer
//      by taking into account the current buffer as input followed by a backinfer
//      of the input shapes per cluster with the mentioned overlap params, such that
//      the will be produced accordingly
// num_clusters - amount of clusters over which the data and compute is segmented
//      * this is not necessarily always equal to the full amount of clusters available
//      during a compilation, for end-end performance reasons
// alignment - shape describing how the per cluster tiled shapes should be aligned
//      thus the per cluster segmentation is done with this alignment in mind
// uniform_distributed_segments - boolean controling the approach of how data and compute are split
//      across clusters.
//      * for example splitting 10 compute lines across 4 clusters is done like:
//          * [3, 3, 2, 2] when uniform_distributed_segments = true
//          * [3, 3, 3, 1] when uniform_distributed_segments = false
//      * in absence of explicit reasons or limitations, uniform_distributed_segments = true
//      is preferred since it generates a more uniformly distribution of compute and data
// compute_shapes - array of compute shapes per cluster used with OVERLAP mode.
//      * represents exact shapes found in each cluster
//      * defines the result produced by a compute op
//      * mutually exclusive with kernel/pad/strides
// compute_offsets - array of compute offsets per cluster used with OVERLAP mode.
//      * represents exact offsets found in each cluster
//      * defines the result produced by a compute op
//      * mutually exclusive with kernel/pad/strides
// memory_shapes - array of memory shapes per cluster used with OVERLAP mode.
//      * represents exact shapes found in each cluster
//      * defines what is actually in memory
//      * mutually exclusive with kernel/pad/strides
// memory_offsets - array of memory offsets per cluster used with OVERLAP mode.
//      * represents exact offsets found in each cluster
//      * defines what is actually in memory
//      * mutually exclusive with kernel/pad/strides
// equal_memory_and_compute_view - used with OVERLAP mode
//      * indicates compute view should be obtained by applying memory view calculations
//      * necessary when having output OVERLAPPED, but without proper halo support
//
/////////////////////////////

def VPU_DistributionInfoAttr : VPU_Attr<"DistributionInfo"> {
    let parameters = (ins
        "vpux::VPU::DistributionModeAttr":$mode,
        OptionalParameter<"mlir::ArrayAttr">:$num_tiles,
        OptionalParameter<"mlir::ArrayAttr">:$kernel,
        OptionalParameter<"vpux::VPU::PaddingAttr">:$pads,
        OptionalParameter<"mlir::ArrayAttr">:$strides,
        "mlir::IntegerAttr":$num_clusters,
        OptionalParameter<"mlir::ArrayAttr">:$alignment,
        OptionalParameter<"mlir::UnitAttr">:$uniform_distributed_segments,
        OptionalParameter<"mlir::ArrayAttr">:$compute_shapes,
        OptionalParameter<"mlir::ArrayAttr">:$compute_offsets,
        OptionalParameter<"mlir::ArrayAttr">:$memory_shapes,
        OptionalParameter<"mlir::ArrayAttr">:$memory_offsets,
        OptionalParameter<"mlir::UnitAttr">:$equal_memory_and_compute_view
    );
}

//
// SparsityCompressionAttr
//

def VPU_SparsityCompressionAttr : VPU_Attr<"SparsityCompression"> {
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


//
// SEAttr
//

class VPU_SEAttrBase<Pred condition, string summary> : Attr<condition, summary> {
    let storageType = [{ vpux::VPU::SEAttr }];
    let returnType = [{ vpux::VPU::SEAttr }];
    let convertFromStorage = "$_self";
}

def VPU_SEAttr : VPU_SEAttrBase<CPred<"$_self.isa<vpux::VPU::SEAttr>()">,
                                      "Storage Element attribute"> {
        string cppType = "vpux::VPU::SEAttr";
    }

//
// SEInterpolateAttr
//

def VPU_SEInterpolateAttr : VPU_Attr<"SEInterpolate", [
        DeclareAttrInterfaceMethods<VPU_SEAttrInterface>
    ] > {
    let description = [{
        This attribute contains parameters for HW interpolate which is implemented
        by means of Storage Element table. It describes how the Storage Element table
        is generated to pick elements from the initial data.

        The attribute is intended to be used with the sparse input type for NCE
        operations, as well as in the Storage Element Table operation itself.

        The attribute contains the following parameters:
        - `mode`: describes the type of data duplication that will be applied over
          the input data in order to generate the effective data that will be read
          by the IDU. It can take the following values: `NEAREST` and `BILINEAR`
        - `coordinate_transformation_mode`: describes how the coordinate of the
          interpolated output can be transformed into a coordinate of the input data
        - `scale`: an array with the same rank as the input data. It describes how
          many times is the output greater than the input
        - `nearest_mode`: applicable only for `NEAREST` mode above; it will decide
          how back-inferred float coordinates of the input data will be rounded to
          actual coordinates
        - `offsets` / `sizes`: describes what slice of the effective output data
          represents the final output. The parameters above define how the input data
          will be duplicated based on a set formula. However, not all of the resulting
          data might be required for the instance of sparse data (e.g. after tiling).
          These two parameters allow extracting only a slice of the duplicated data.
        - `initial_input_shape` / `initial_output_shape`: maintains the information
          about the input and output data of the original Interpolate operation (e.g.
          before tiling), which is necessary for some modes, such as ALIGN_CORNERS.
          It is also help to infer input coordinate for bilinear interpolate when tiling.

        An example:
        ```
        Bilinear Interpolate with PYTORCH_HALF_PIXEL coordMode
        Starting from the following input data:
            1 2 3
            4 5 6
            7 8 9
        With the following configuration:
          - scale: [1, 1, 2, 2]
        Following parameters can be infered:
          - factors: [4, 4]
            The `factors` represents the number of times each element is copied on H and W separately
          - pads: [1, 1, 1, 1]
            The `pads` represents the number of times boundary data is expanded in
            the [left, top, right, bottom] respectively
        The data is duplicated using the Storage Element pointers into:
            1 1 1 1 1 2 2 2 2 3 3 3 3 3
            1 1 1 1 1 2 2 2 2 3 3 3 3 3
            1 1 1 1 1 2 2 2 2 3 3 3 3 3
            1 1 1 1 1 2 2 2 2 3 3 3 3 3
            1 1 1 1 1 2 2 2 2 3 3 3 3 3
            4 4 4 4 4 5 5 5 5 6 6 6 6 6
            4 4 4 4 4 5 5 5 5 6 6 6 6 6
            4 4 4 4 4 5 5 5 5 6 6 6 6 6
            4 4 4 4 4 5 5 5 5 6 6 6 6 6
            7 7 7 7 7 8 8 8 8 9 9 9 9 9
            7 7 7 7 7 8 8 8 8 9 9 9 9 9
            7 7 7 7 7 8 8 8 8 9 9 9 9 9
            7 7 7 7 7 8 8 8 8 9 9 9 9 9
            7 7 7 7 7 8 8 8 8 9 9 9 9 9
        ```
    }];
    let parameters = (ins
        "vpux::VPU::NCEInterpolateModeAttr":$mode,
        "vpux::IE::InterpolateCoordModeAttr":$coordinate_transformation_mode,
        "mlir::ArrayAttr":$scale,
        OptionalParameter<"vpux::IE::InterpolateNearestModeAttr">:$nearest_mode,
        OptionalParameter<"mlir::ArrayAttr">:$offsets,
        OptionalParameter<"mlir::ArrayAttr">:$sizes,
        OptionalParameter<"mlir::ArrayAttr">:$initial_input_shape,
        OptionalParameter<"mlir::ArrayAttr">:$initial_output_shape
    );

    let assemblyFormat = "`<` struct(params) `>`";
    let genVerifyDecl = 1;
}

//
// SERollAttr
//

def VPU_SERollAttr : VPU_Attr<"SERoll", [
        DeclareAttrInterfaceMethods<VPU_SEAttrInterface>
    ] > {
    let description = [{
        Describes how the input data can be rolled on the spatial dimensions
        by a Storage Element Table. The attribute is intended to be used with
        the sparse input type for NCE operations, as well as in the Storage Element
        Table operation itself.

        The attribute contains the following parameters:
        - `shift`: [H, W] - describes how many lines/columns should be rolled on
          each spatial dimension
        - `axes`: [H, W] - describes on which dimensions to roll
        - `offsets` / `sizes`: [N, C, H, W] - describes what slice of the effective
          output data represents the final output. The parameters above define how
          the input data will be expanded. However, not all of the resulting data
          might be required for the instance of sparse data (e.g. after tiling).
          These two parameters allow extracting only a slice of the duplicated data.

        An example:
        ```
        Starting from the following input data with shape [N, C, 3, 3]:
            1 2 3
            4 5 6
            7 8 9
        With the following configuration:
          - shift: [1, 2]
          - axes: [2, 3]
        The data is rolled using the Storage Element pointers into:
            8 9 7
            2 3 1
            5 6 4
        The following sparsity map is utilized:
            1 1 1
            1 1 1
            1 1 1
        In order to generate the final data, this data can be further
        sliced by using:
          - offsets: [0, 0, 0, 0]
          - sizes:   [N, C, 3, 3]
        Resulting in:
            8 9 7
            2 3 1
            5 6 4

        Another example is extracting tile with output tile configuration:
          - offsets: [0, 0, 0, 0]
          - sizes:   [N, C, 2, 2]
        The output tile data:
            8 9
            2 3
        Then got the input tile:
            2 3
            5 6
            8 9
        The data is rolled with configuration:
          - shift: [1, 0]
          - axes: [2, 3]
        The rolled output data:
            8 9
            2 3
            5 6
        Then sliced by using:
          - offsets: [0, 0, 0, 0]
          - sizes:   [N, C, 2, 2]
        Resulting in:
            8 9
            2 3
        ```
    }];
    let parameters = (ins
        "mlir::ArrayAttr":$shift,
        "mlir::ArrayAttr":$axes,
        OptionalParameter<"mlir::ArrayAttr">:$offsets,
        OptionalParameter<"mlir::ArrayAttr">:$sizes
    );

    let assemblyFormat = "`<` struct(params) `>`";
    let genVerifyDecl = 1;
}

//
// SEUpsamplingAttr
//

def VPU_SEUpsamplingAttr : VPU_Attr<"SEUpsampling", [
        DeclareAttrInterfaceMethods<VPU_SEAttrInterface>
    ] > {
    let description = [{
        Describes how the input data can be upsampled on the spatial dimensions
        by a Storage Element Table. The attribute is intended to be used with
        the sparse input type for NCE operations, as well as in the Storage Element
        Table operation itself.

        The attribute contains the following parameters:
        - `factors`: [H, W] - describes how many zero-valued elements should be
          introduced between two consecutive elements on each spatial dimension
        - `padding`: [left, top, right, bottom] - describes how many zero-valued
          pad values should be introduced on each margin
        - `offsets` / `sizes`: [N, C, H, W] - describes what slice of the effective
          output data represents the final output. The parameters above define how
          the input data will be expanded. However, not all of the resulting data
          might be required for the instance of sparse data (e.g. after tiling).
          These two parameters allow extracting only a slice of the duplicated data.

        An example:
        ```
        Starting from the following input data:
            1 2 3
            4 5 6
            7 8 9
        With the following configuration:
          - factors: [1, 2]
          - padding: [1, 1, 1, 1]
        The data is duplicated using the Storage Element pointers into:
            1 1 1 1 2 2 2 3 3
            1 1 1 1 2 2 2 3 3
            1 1 1 1 2 2 2 3 3
            4 4 4 4 5 5 5 6 6
            4 4 4 4 5 5 5 6 6
            7 7 7 7 8 8 8 9 9
            7 7 7 7 8 8 8 9 9
        The following sparsity map is utilized:
            0 0 0 0 0 0 0 0 0
            0 1 0 0 1 0 0 1 0
            0 0 0 0 0 0 0 0 0
            0 1 0 0 1 0 0 1 0
            0 0 0 0 0 0 0 0 0
            0 1 0 0 1 0 0 1 0
            0 0 0 0 0 0 0 0 0
        In order to generate the final upsampled data:
            0 0 0 0 0 0 0 0 0
            0 1 0 0 2 0 0 3 0
            0 0 0 0 0 0 0 0 0
            0 4 0 0 5 0 0 6 0
            0 0 0 0 0 0 0 0 0
            0 7 0 0 8 0 0 9 0
            0 0 0 0 0 0 0 0 0
        This data can be further sliced by using:
          - offsets: [0, 0, 2, 1]
          - sizes:   [N, C, 3, 5]
        Resulting in:
            0 0 0 0 0
            4 0 0 5 0
            0 0 0 0 0
        ```
    }];
    let parameters = (ins
        "mlir::ArrayAttr":$factors,
        OptionalParameter<"mlir::ArrayAttr">:$padding,
        OptionalParameter<"mlir::ArrayAttr">:$offsets,
        OptionalParameter<"mlir::ArrayAttr">:$sizes
    );

    let assemblyFormat = "`<` struct(params) `>`";
    let genVerifyDecl = 1;
}

//
// SEPaddingAttr
//

def VPU_SEPaddingAttr : VPU_Attr<"SEPadding", [
        DeclareAttrInterfaceMethods<VPU_SEAttrInterface>
    ] > {
    let description = [{
        Describes how the input data can be padded on the spatial dimensions
        by a Storage Element Table. The attribute is intended to be used with
        the sparse input type for NCE operations, as well as in the Storage Element
        Table operation itself.

        The attribute contains the following parameters:
        - `mode`: describes how to generate new element padding values. It can take
          the following values: `CONSTANT`, `EDGE`, `REFLECT`, and `SYMMETRIC`
        - `padding`: [left, top, right, bottom] - describes how many pad values
          should be introduced on each margin
        - `offsets` / `sizes`: [N, C, H, W] - describes what slice of the effective
          output data represents the final output. The parameters above define how
          the input data will be expanded. However, not all of the resulting data
          might be required for the instance of sparse data (e.g. after tiling).
          These two parameters allow extracting only a slice of the duplicated data.

        An example:
        ```
        Starting from the following input data:
            1 2 3
            4 5 6
            7 8 9
        With the following configuration:
          - mode: `REFLECT`
          - padding: [1, 2, 2, 1]
        The data is duplicated using the Storage Element pointers into:
            8 7 8 9 8 7
            5 4 5 6 5 4
            2 1 2 3 2 1
            5 4 5 6 5 4
            8 7 8 9 8 7
            5 4 5 6 5 4
        This data can be further sliced by using last three lines:
            5 4 5 6 5 4
            8 7 8 9 8 7
            5 4 5 6 5 4
        The input data are used:
            4 5 6
            7 8 9
        The data is duplicated using the Storage Element pointers into:
            8 7 8 9 8 7
            8 7 8 9 8 7
            5 4 5 6 5 4
            8 7 8 9 8 7
            5 4 5 6 5 4
        This data can be further sliced by using:
          - offsets: [0, 0, 2, 0]
          - sizes:   [N, C, 3, 6]
        ```
    }];
    let parameters = (ins
        "vpux::IE::PadModeAttr":$mode,
        "mlir::ArrayAttr":$padding,
        OptionalParameter<"mlir::ArrayAttr">:$offsets,
        OptionalParameter<"mlir::ArrayAttr">:$sizes
    );

    let extraClassDeclaration = [{
        std::tuple<Shape, Shape, Shape> inferPadInputTileParams(ShapeRef outputTileOffset, ShapeRef outputTileShape, ShapeRef inputShape) const;
    }];

    let assemblyFormat = "`<` struct(params) `>`";
    let genVerifyDecl = 1;
}

//
// SEDilatedConvAttr
//

def VPU_SEDilatedConvAttr : VPU_Attr<"SEDilatedConv", [
        DeclareAttrInterfaceMethods<VPU_SEAttrInterface>
    ] > {
    let description = [{
        Describes how input data can be modified to emulate dilated convolution behavior
        with normal convolution. Using this attribute implies that original dilated
        convolution is decomposed into a graph of normal convolutions with sparse input.
        Number of convolutions in the graph is D^2. Each convolution should process only
        a part of input tensor and this part is defined by Storage Element Table.
        Output of all convolution then should be interleaved using strided concat.

        The attribute contains the following parameters:

        - `dilation`: [H, W] - describes how many elements should be skipped on each
          spatial dimension, it is dilation of ofiginal kernel

        - `kernelStride`: [H, W] - original convolution kernel stride

        - `kernelSize`: [H, W] - original convolution kernel size

        - `dataOffset`: [N, C, H, W] - starting point of input tensor used for every
          sub-convolution

        - `dataSizes`: [N, C, H, W] - size of original input sub-tensor used for
          every sub-convolution

        - `offsets` / `sizes`: [N, C, H, W] - describes what slice of the effective
          output data represents the final output. The parameters above define how
          the input data will be expanded. Attribute methods may provide more data
          than necessary and these parameters can help to further slice it.

        Example:
        ```
        Original convoltion has:
            - input     = 1x16x6x6
            - kernel    = 3x3
            - dilations = 2x2
            - stride    = 1x1

        Original input data [1, 16, 6, 6]:
            1  2  3  4  5  6
            7  8  9 10  11 12
            13 14 15 16 17 18
            19 20 21 22 23 24
            25 26 27 28 29 30
            31 32 33 34 35 36

        It can be decomposed into a subgraph of 4 sub-convolutions.
        The first one has the following configuration for SET attribute:

            - dilation:     [2, 2]
            - kernelStride: [1, 1]
            - kernelSize:   [3, 3]
            - dataOffset:   [0, 0, 0, 0]
            - dataSizes:    [1, 16, 5, 5]

        The data is fetched using the Storage Element pointers:
            1  3  5
            13 15 17
            25 27 29

        The second one has the following configuration for SET attribute:

            - dilation:     [2, 2]
            - kernelStride: [1, 1]
            - kernelSize:   [3, 3]
            - dataOffset:   [0, 0, 0, 1]
            - dataSizes:    [1, 16, 5, 5]

        The data is fetched using the Storage Element pointers:
            2  4  6
            14 16 18
            26 28 30

        The other 2 convolutions have similar parameters, but with different dataOffset.
        All 4 outputs have to be interleaved to get the final output.
        ```
    }];
    let parameters = (ins
        "mlir::ArrayAttr":$dilation,

        "mlir::ArrayAttr":$kernelStride,
        "mlir::ArrayAttr":$kernelSize,

        "mlir::ArrayAttr":$dataOffset,
        "mlir::ArrayAttr":$dataSizes,

        OptionalParameter<"mlir::ArrayAttr">:$offsets,
        OptionalParameter<"mlir::ArrayAttr">:$sizes
    );

    let assemblyFormat = "`<` struct(params) `>`";
    let genVerifyDecl = 1;
}

//
// M2I Color Formats
//  PL_ = planar
//  SP_ = semi-planar
//  IL_ = interleaved
//
// PL_YUV444_8 = 0x00, // Planar 4:4:4 format (8-bit per component)
// PL_YUV420_8 = 0x01, // Planar 8-bit 4:2:0 format
// PL_RGB24 = 0x02,    // Planar 24-bit RGB data (8-bit per component)
// PL_RGB30 = 0x03,    // Planar 30-bit RGB data (10-bit per component)
// PL_GRAY8 = 0x04,    // 8-bit grayscale
// PL_FP16_RGB = 0x05, // Planar RGB fp16 data
// PL_FP16_YUV = 0x06, // Planar YUV fp16 data
// PL_YUV422_8 = 0x07, // Planar 8 bit
// SP_NV12_8 = 0x08,   // 8-bit YUV 4:2:0 semi planar
// SP_NV12_10 = 0x09,  // 10-bit YUV 4:2:0 semi planar
// SP_PO10 = 0x0A,     // 10-bit YUV 4:2:0 semi planar
// IL_YUV422_8 = 0x18, // Interleaved 8 bit
// IL_RGB8888 = 0x19,  // Interleaved 32-bit RGB data (8-bit per component)
// IL_RGB888 = 0x1A,   // Interleaved 24-bit RGB data (8-bit per component)
// IL_RGB30 = 0x1B     // Interleaved 30-bit RGB data (10-bit per component)
//

def VPU_M2iColorFmt :
        VPU_I64EnumAttr<
            "M2iColorFmt",
            "M2I color formats",
            [
                I64EnumAttrCase<"PL_YUV444_8", 0>,
                I64EnumAttrCase<"PL_YUV420_8", 1>,
                I64EnumAttrCase<"PL_RGB24",    2>,
                I64EnumAttrCase<"PL_RGB30",    3>,
                I64EnumAttrCase<"PL_GRAY8",    4>,
                I64EnumAttrCase<"PL_FP16_RGB", 5>,
                I64EnumAttrCase<"PL_FP16_YUV", 6>,
                I64EnumAttrCase<"PL_YUV422_8", 7>,
                I64EnumAttrCase<"SP_NV12_8",   8>,
                I64EnumAttrCase<"SP_NV12_10",  9>,
                I64EnumAttrCase<"SP_P010",     10>,
                I64EnumAttrCase<"IL_YUV422_8", 24>,
                I64EnumAttrCase<"IL_RGB8888",  25>,
                I64EnumAttrCase<"IL_RGB888",   26>,
                I64EnumAttrCase<"IL_RGB30",    27>
            ]
        > {
}

def VPU_M2iColorFmtAttr : VPU_EnumAttr<VPU_M2iColorFmt, "m2i_color_fmt">;

def VPU_M2iInterp :
        VPU_I64EnumAttr<
            "M2iInterp",
            "M2I resize interpolation type",
            [
                I64EnumAttrCase<"NEAREST",  0>,
                I64EnumAttrCase<"BILINEAR", 1>
            ]
        > {
}

def VPU_M2iInterpAttr : VPU_EnumAttr<VPU_M2iInterp, "m2i_interp">;

//
// TaskType
//

def VPU_TaskType :
    VPU_I64EnumAttr<
        "TaskType",
        "The type of processing code relevant to the VPU task",
        [
            I64EnumAttrCase<"DMA",                 0>,
            I64EnumAttrCase<"ActKernelInvocation", 1>,
            I64EnumAttrCase<"ActKernelRange",      2>,
            I64EnumAttrCase<"DPUInvariant",        3>,
            I64EnumAttrCase<"DPUVariant",          4>,
            I64EnumAttrCase<"M2I",                 5>
        ]
    > {
}

def VPU_TaskTypeAttr : VPU_EnumAttr<VPU_TaskType, "reg_task_type">;

#endif
