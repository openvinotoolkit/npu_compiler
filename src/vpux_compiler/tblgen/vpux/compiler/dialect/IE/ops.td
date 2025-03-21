//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS
#define VPUX_COMPILER_DIALECT_IE_OPS

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/core/types.td"
include "vpux/compiler/dialect/IE/dialect.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//
// Base classes
//

class IE_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            IE_Dialect,
            mnemonic,
            traits
        >;

class IE_LayerOp<string mnemonic, list<Trait> traits = []> :
        IE_Op<
            mnemonic,
            [
                Pure,
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                DeclareOpInterfaceMethods<IE_LayerOpInterface>
            ] # traits
        > {
    list<string> elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    bit checkInferredDimsOrder = 0;
    bit checkInferredMemSpace = 0;

    code baseExtraClassDeclaration = [{
        static bool isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
            return vpux::areTypesCompatible(lhs, rhs,
                }] # !interleave(elemComparisonModes, "|") # [{,
                static_cast<bool>(}] # checkInferredDimsOrder # [{),
                static_cast<bool>(}] # checkInferredMemSpace # [{)
            );
        }
    }];

    let extraClassDeclaration = baseExtraClassDeclaration;

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// CNNNetworkOp
//

def IE_CNNNetworkOp :
        IE_Op<
            "CNNNetwork",
            [
                IsolatedFromAbove,
                HasParent<"mlir::ModuleOp">,
                NoRegionArguments,
                DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                OpAsmOpInterface
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "InferenceEngine CNN Network description";

    let description = [{
        This operation is bound to MLIR Module and holds extra information about InferenceEngine CNN Network:

          * Precision and layout for user-provided inputs.
          * Precision and layout for user-provided outputs.
          * Layout for output profiling data(optional).
          * Entry point (Function name) for the network inference.
          * Model inference time by DPU cycle unit.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$entryPoint,
        OptionalAttr<IntAttr>:$inferenceTiming
    );

    let regions = (region
        SizedRegion<1>:$inputs_info,
        SizedRegion<1>:$outputs_info,
        VariadicRegion<SizedRegion<1>>:$profiling_outputs_info
    );

    let extraClassDeclaration = [{
        size_t getNetInputsCount();
        vpux::SmallVector<vpux::IE::DataInfoOp, 1> getInputsDataInfo();

        size_t getNetOutputsCount();
        vpux::SmallVector<vpux::IE::DataInfoOp, 1> getOutputsDataInfo();

        size_t getProfilingOutputsCount();
        vpux::SmallVector<vpux::IE::DataInfoOp, 1> getProfilingOutputsDataInfo();

        static void getFromModule(
                mlir::ModuleOp module,
                vpux::IE::CNNNetworkOp& netInfo,
                mlir::func::FuncOp& netFunc);

        static mlir::StringRef getDefaultDialect() {
            return "IE";
        }
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::FlatSymbolRefAttr":$entryPoint, "bool":$withProfiling)
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `entryPoint` `:` $entryPoint
        `inputsInfo` `:` $inputs_info
        `outputsInfo` `:` $outputs_info
        (`profilingOutputsInfo` `:` $profiling_outputs_info^)?
    }];

    let hasVerifier = 1;
}

//
// DataInfoOp
//

def IE_DataInfoOp :
        IE_Op<
            "DataInfo",
            [
                IsolatedFromAbove,
                NoRegionArguments,
                OpAsmOpInterface,
                HasParent<"vpux::IE::CNNNetworkOp">
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Information about InferenceEngine CNN Network input/output Data object";

    let description = [{
        This operation is bound to `IE.CNNNetwork` Operation and holds information about Data object:

          * Name - primary name of Data object
          * Actual shape - OV node output tensor shape after batching
          * Original shape - OV node output tensor shape before batching
          * Friendly name - OV node friendly name
          * Input name - legacy name of Parameter or Result tensor
          * Tensor names - OV node tensor names
          * User-defined shape
          * User-defined precision (element type)
          * User-defined layout
    }];

    let arguments = (ins
        StrAttr:$name,
        TypeAttr:$userType,
        OptionalAttr<TypeAttr>:$originalShape,
        OptionalAttr<StrAttr>:$friendlyName,
        OptionalAttr<StrAttr>:$inputName,
        OptionalAttr<ArrayAttr>:$tensorNames
    );

    let regions = (region
        VariadicRegion<SizedRegion<1>>:$sections
    );

    let extraClassDeclaration = [{
        vpux::DimsOrder getDimsOrder();
    }];

    let assemblyFormat = [{
        $name (`friendlyName` `=` $friendlyName^)? (`inputName` `=` $inputName^)? (`tensorNames` `=` $tensorNames^)? ($sections^)? `:` $userType (`originalShape` `=` $originalShape^)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// IE_PipelineOptionsOp
//

def IE_PipelineOptionsOp :
        IE_Op<
            "PipelineOptions",
            [
                SymbolTable,
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Parallel to config options used in the compiler to be exposed globally";

    let description = [{
        Options can include another PipelineOptionsOp or OptionsOp
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let regions = (region
        SizedRegion<1>:$options
    );

    let assemblyFormat = [{
        attr-dict
        $sym_name
        custom<OptionalBlockRegion>($options)
    }];
}

//
// IE_OptionOp
//

def IE_OptionOp :
        IE_Op<
            "Option",
            [
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Option to be added to the model/IR";

    let description = [{
        Concrete option values that will be added to IR, must have a sym_name for lookup
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        IntAttr:$optionValue
    );

    let assemblyFormat = [{
        $sym_name `:` $optionValue
        attr-dict
    }];
}

//
// SparsityStatisticsOp
//

def IE_SparsityStatisticsOp :
        IE_Op<
            "SparsityStatistics",
            [
                IsolatedFromAbove,
                HasParent<"mlir::ModuleOp">,
                NoRegionArguments,
                OpAsmOpInterface
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Sparsity statistics for inputs of operations";

    let description = [{
        This operation is bound to MLIR Module and holds extra information about input sparsity ratios collected from dataset:

          * NGraph layer name.
          * Input id.
          * Sparsity ratio.

    }];

    let regions = (region
        SizedRegion<1>:$sparsityInfo
    );

    let assemblyFormat = [{
        attr-dict
        `sparsityInfo` `:` $sparsityInfo
    }];

    let hasVerifier = 0;
}

//
// SparsityInfoOp
//

def IE_SparsityInfoOp :
        IE_Op<
            "SparsityInfo",
            [
                IsolatedFromAbove,
                HasParent<"vpux::IE::SparsityStatisticsOp">
            ]
        > {
    let summary = "Information about estimated sparsity ratio for input of specific layer";

    let description = [{
        This operation is bound to `IE.SparsityStatistics` Operation and holds information about SparsityInfo object:

          * Node name
          * Input ID
          * Sparsity ratio
    }];

    let arguments = (ins
        StrAttr:$name,
        IntAttr:$inputId,
        F64Attr:$ratio
    );

    let assemblyFormat = [{
        $ratio `at` `input` $inputId `of` $name
        attr-dict
    }];

    let hasVerifier = 0;
}


//
// MemoryResourceOp
//

def IE_MemoryResourceOp :
        IE_Op<
            "MemoryResource",
            [
                IERT_ResourceOpInterface,
                Symbol
            ]
        > {
    let summary = "Information about memory resource";

    let description = [{
        The memory resource is defined by the following attributes:

          * sym_name - kind of memory space.
          * byteSize - size in bytes of memory space.
          * offset - optional offset attribute if this resource is part of larger memory space
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        IntAttr:$byteSize,
        OptionalAttr<IntAttr>:$offset
    );

    let extraClassDeclaration = [{
        vpux::Byte size() { return vpux::Byte(getByteSize()); }

        static mlir::StringRef getDefaultDialect() {
            return "IE";
        }
    }];

    let assemblyFormat = [{
        $byteSize `bytes` `of` $sym_name (`offset` $offset^)?
        attr-dict
    }];
}

//
// ExecutorResourceOp
//

def IE_ExecutorResourceOp :
        IE_Op<
            "ExecutorResource",
            [
                IERT_ResourceOpInterface,
                IERT_ComputeResourceOpInterface,
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Information about executor resource";

    let description = [{
        The executor resource is defined by the following attributes:

          * sym_name - kind of the executor.
          * count - number of executor units.
          * activity_factor - power activity factor.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        IntAttr:$count,
        OptionalAttr<F64Attr>:$activity_factor
    );

    let extraClassDeclaration = [{
        static mlir::StringRef getDefaultDialect() {
            return "IE";
        }
    }];

    let assemblyFormat = [{
        attr-dict
        $count `of` $sym_name
    }];
}


//
// TileResourceOp
//

def IE_TileResourceOp :
        IE_Op<
            "TileResource",
            [
                IERT_ResourceOpInterface,
                IERT_ComputeResourceOpInterface,
                SymbolTable,
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Information about tile resource";

    let description = [{
        The tile resource is defined by the following attributes:
          * sym_name - kind of the executor.
          * count - number of executor units.
          * proc_freq - processor frequency (in MHz).
          * activity_factor - power activity factor.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        IntAttr:$count,
        OptionalAttr<F64Attr>:$proc_freq,
        OptionalAttr<F64Attr>:$activity_factor
    );

    let regions = (region
        SizedRegion<1>:$subExecutors
    );

    let extraClassDeclaration = [{
        mlir::FloatAttr getProcessorFrequency() {
            return getProcFreqAttr();
        }

        void setProcessorFrequency(mlir::FloatAttr procFreqAttr) {
            setProcFreqAttr(procFreqAttr);
        }

        bool hasProcessorFrequency() {
            return getProcFreqAttr() != nullptr;
        }

        vpux::IE::ExecutorResourceOp addSubExecutor(mlir::SymbolRefAttr executor, size_t count);
        bool hasSubExecutor(mlir::SymbolRefAttr executor);
        vpux::IE::ExecutorResourceOp getSubExecutor(mlir::SymbolRefAttr executor);
        vpux::IE::MemoryResourceOp addAvailableMemory(mlir::SymbolRefAttr memSpace, Byte size);

        bool hasAvailableMemory(mlir::SymbolRefAttr memSpace);

        vpux::IE::MemoryResourceOp getAvailableMemory(mlir::SymbolRefAttr memSpace);


        template <typename Enum, typename OutT = vpux::IE::ExecutorResourceOp>
        using exec_resource_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

        template <typename Enum, typename OutT = vpux::IE::MemoryResourceOp>
        using mem_resource_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

        template <typename Enum>
        exec_resource_if<Enum> addSubExecutor(Enum kind, size_t count) {
            return addSubExecutor(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)), count);
        }

        template <typename Enum, typename OutT = bool>
        using bool_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

        template <typename Enum>
        bool_if<Enum> hasSubExecutor(Enum kind) {
            return hasSubExecutor(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)));
        }

        template <typename Enum>
        exec_resource_if<Enum> getSubExecutor(Enum kind) {
            return getSubExecutor(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)));
        }

        template <typename Enum>
        mem_resource_if<Enum> addAvailableMemory(Enum kind, Byte size) {
            return addAvailableMemory(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)), size);
        }

        template <typename Enum>
        bool_if<Enum> hasAvailableMemory(Enum kind) {
            return hasAvailableMemory(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)));
        }

        template <typename Enum>
        mem_resource_if<Enum> getAvailableMemory(Enum kind) {
            return getAvailableMemory(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)));
        }

        static mlir::StringRef getDefaultDialect() {
            return "IE";
        }
    }];

    let assemblyFormat = [{
        attr-dict
        $count `of` $sym_name (`at` $proc_freq^`MHz`)?
        custom<OptionalBlockRegion>($subExecutors)
    }];
}

//
// FakeConvertOp
//

def IE_FakeConvertOp :
        IE_LayerOp<
            "FakeConvert",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine FakeConvert layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$scale,
        Optional<RankedTensorOf<[F16, F32]>>:$shift,

        TypeAttr:$dst_type
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasVerifier = 1;
}

//
// ConvertOp
//

def IE_ConvertOp :
        IE_LayerOp<
            "Convert",
            [
                DeclareOpInterfaceMethods<CastOpInterface>,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Convert layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let hasVerifier = 1;
}

//
// ConvertLikeOp
//

def IE_ConvertLikeOp:
        IE_LayerOp<
            "ConvertLike"
        > {
    let summary = "InferenceEngine ConvertLike layer";

    let description = [{
        This operator returns a tensor with the same shape as the data input
        and same type as the like input
    }];

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$like
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// ReorderOp
//

def IE_ReorderOp :
        IE_LayerOp<
            "Reorder"
        > {
    let summary = "InferenceEngine Reorder layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        AffineMapAttr:$dstOrder
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 0;
}

//
// SoftMaxOp
//

def IE_SoftMaxOp :
        IE_LayerOp<
            "SoftMax",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "InferenceEngine SoftMax layer";

    let description = [{
        This operation is intended to run Softmax filter with extra capability
        to padding with 0 part of output on axis attribute direction.
        - padSize - attribute that allow to specify how much from axis dimension
        to not be taken in consideration from input and to be wrote with 0 on output.
        This extra functionality was added in order to cover,
        with minimum cost, align required from neighbor filters.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        IntAttr:$axisInd,
        OptionalAttr<IntAttr>:$padSize
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// LogSoftmaxOp
//

def IE_LogSoftmaxOp :
        IE_LayerOp<
            "LogSoftmax"
        > {
    let summary = "InferenceEngine LogSoftmax layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        IntAttr:$axisInd
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasFolder = 1;
}

//
// TileOp
//

def IE_TileOp :
        IE_LayerOp<
            "Tile"
        > {
    let summary = "InferenceEngine Tile layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$repeats,
        OptionalAttr<I64ArrayAttr>:$repeats_values
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// DynamicTileOp
//

def IE_DynamicTileOp :
        IE_LayerOp<
            "DynamicTile"
        > {
    let summary = "InferenceEngine Tile layer with dynamic operands";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$target_shape,

        Optional<RankedTensorOf<[AnyInteger]>>:$repeats,
        OptionalAttr<I64ArrayAttr>:$repeats_values,

        I64ArrayAttr:$output_shape,
        I64ArrayAttr:$output_bounds
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// PerAxisTileOp
//

def IE_PerAxisTileOp :
        IE_LayerOp<
            "PerAxisTile"
        > {
    let summary = "InferenceEngine per axis Tile layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        IntAttr:$axis,
        IntAttr:$tiles
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ReLUOp
//

def IE_ReLUOp :
        IE_LayerOp<
            "ReLU",
            [
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                ShapeBoundOp
            ]
        > {
    let summary = "InferenceEngine ReLU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// SplitOp
//

def IE_SplitOp :
        IE_LayerOp<
            "Split"
        > {
    let summary = "InferenceEngine Split layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<AnyRankedTensor>:$axis,

        IntAttr:$num_splits,
        OptionalAttr<IntAttr>:$axis_value
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );

    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// PowerOp
//

def IE_PowerOp :
        IE_LayerOp<
            "Power",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Power layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64]>:$input1,
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// AddOp
//

def IE_AddOp :
        IE_LayerOp<
            "Add",
            [
                Commutative,
                IE_EltwiseOp,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                ShapeBoundOp
            ]
        > {
    let summary = "InferenceEngine Add layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$input1,
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<IntAttr>:$output_channels,
        OptionalAttr<IntAttr>:$input_channels
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$output
    );

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION];
}

//
// DivideOp
//

def IE_DivideOp :
        IE_LayerOp<
            "Divide",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Divide layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64]>:$input1,
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64]>:$output
    );
}

//
// SquaredDiffOp
//

def IE_SquaredDifferenceOp :
        IE_LayerOp<
            "SquaredDiff",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine SquaredDiff layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI64]>:$output
    );
}

//
// RollOp
//

def IE_RollOp :
        IE_LayerOp<
            "Roll"
        > {
    let summary = "InferenceEngine Roll layer";

    let arguments = (ins
        AnyRankedTensor:$data,
        1DTensorOf<[SI32, SI64]>:$shift,
        1DTensorOf<[SI32, SI64]>:$axes
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// FloorModOp
//

def IE_FloorModOp :
        IE_LayerOp<
            "FloorMod",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine FloorMod layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ModOp
//

def IE_ModOp :
        IE_LayerOp<
            "Mod",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Mod layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// LessOp
//

def IE_LessOp :
        IE_LayerOp<
            "Less",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Less layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[Bool8]>:$output
    );
}

//
// LessEqualOp
//

def IE_LessEqualOp :
        IE_LayerOp<
            "LessEqual",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine LessEqual layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[Bool8]>:$output
    );
}

//
// GreaterOp
//

def IE_GreaterOp :
        IE_LayerOp<
            "Greater",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Greater layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[Bool8]>:$output
    );
}

//
// GreaterEqualOp
//

def IE_GreaterEqualOp :
        IE_LayerOp<
            "GreaterEqual",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine GreaterEqual layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI64]>:$output
    );
}

//
// LogicalOrOp
//

def IE_LogicalOrOp :
        IE_LayerOp<
            "LogicalOr",
            [
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine LogicalOr layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, SI32, SI64]>:$input1,
        RankedTensorOf<[I8, F16, F32, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, SI32, SI64]>:$output
    );
}

//
// LogicalNotOp
//

def IE_LogicalNotOp :
        IE_LayerOp<
            "LogicalNot",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Logical Not layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, SI32, SI64]>:$input1
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, SI32, SI64]>:$output
    );
}

//
// LogicalXorOp
//

def IE_LogicalXorOp :
        IE_LayerOp<
            "LogicalXor",
            [
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine LogicalXor layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, SI32, SI64]>:$input1,
        RankedTensorOf<[I8, F16, F32, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, SI32, SI64]>:$output
    );
}

//
// MultiplyOp
//

def IE_MultiplyOp :
        IE_LayerOp<
            "Multiply",
            [
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Multiply layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$input1,
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<IntAttr>:$output_channels,
        OptionalAttr<IntAttr>:$input_channels
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$output
    );

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION];
}

//
// AndOp
//

def IE_AndOp :
        IE_LayerOp<
            "And",
            [
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine And layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, SI32, SI64]>:$input1,
        RankedTensorOf<[I8, F16, F32, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast

    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, SI32, SI64, quant_QuantizedType]>:$output
    );
}

//
// BitwiseAndOp
//

def IE_BitwiseAndOp :
        IE_LayerOp<
            "BitwiseAnd",
            [
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine BitwiseAndOp layer";

    let arguments = (ins
        RankedTensorOf<[AnyInteger]>:$input1,
        RankedTensorOf<[AnyInteger]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[AnyInteger]>:$output
    );
}

//
// BitwiseOrOp
//

def IE_BitwiseOrOp :
        IE_LayerOp<
            "BitwiseOr",
            [
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine BitwiseOrOp layer";

    let arguments = (ins
        RankedTensorOf<[AnyInteger]>:$input1,
        RankedTensorOf<[AnyInteger]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[AnyInteger]>:$output
    );
}

//
// BitwiseXorOp
//

def IE_BitwiseXorOp :
        IE_LayerOp<
            "BitwiseXor",
            [
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine BitwiseXorOp layer";

    let arguments = (ins
        RankedTensorOf<[AnyInteger]>:$input1,
        RankedTensorOf<[AnyInteger]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[AnyInteger]>:$output
    );
}

//
// BitwiseNotOp
//

def IE_BitwiseNotOp :
        IE_LayerOp<
            "BitwiseNot",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine BitwiseNotOp layer";

    let arguments = (ins
        RankedTensorOf<[AnyInteger]>:$input1
    );

    let results = (outs
        RankedTensorOf<[AnyInteger]>:$output
    );
}

//
// ConvolutionOp
//

def IE_ConvolutionOp :
        IE_LayerOp<
            "Convolution",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                ShapeBoundOp
            ]
        > {
    let summary = "InferenceEngine Convolution layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32]>:$input,
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32]>:$filter,
        Optional<RankedTensorOf<[F16, F32, SI32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<F32Attr>:$static_scale, // constant applied to the weights table
        OptionalAttr<IntAttr>:$output_channels,
        OptionalAttr<IntAttr>:$input_channels
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32]>:$output
    );

    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// GroupConvolutionOp
//

def IE_GroupConvolutionOp :
        IE_LayerOp<
            "GroupConvolution"
        > {
    let summary = "InferenceEngine GroupConvolution layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32]>:$input,
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32]>:$filter,
        Optional<RankedTensorOf<[F16, F32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        OptionalAttr<IntAttr>:$groups,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<IntAttr>:$output_channels,
        OptionalAttr<IntAttr>:$input_channels
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32]>:$output
    );

    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// GroupNormalizationOp
//

def IE_GroupNormalizationOp :
        IE_LayerOp<
            "GroupNormalization"
        > {
    let summary = "InferenceEngine GroupNormalization layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$scale,
        RankedTensorOf<[F16, F32]>:$bias,

        I32Attr:$num_groups,
        F32Attr:$epsilon

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AvgPoolOp
//

def IE_AvgPoolOp :
        IE_LayerOp<
            "AvgPool"
        > {
    let summary = "InferenceEngine AvgPool layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,
        UnitAttr:$exclude_pads,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<F32Attr>:$static_scale,
        OptionalAttr<IntAttr>:$output_channels,
        OptionalAttr<IntAttr>:$input_channels
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$output
    );

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// MaxPoolOp
//

def IE_MaxPoolOp :
        IE_LayerOp<
            "MaxPool",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                ShapeBoundOp
            ]
        > {
    let summary = "InferenceEngine MaxPool layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<IntAttr>:$output_channels,
        OptionalAttr<IntAttr>:$input_channels
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$output
    );

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// MaxPool8Op
//

def IE_MaxPool8Op :
        IE_LayerOp<
            "MaxPool8"
        > {
    let summary = "InferenceEngine MaxPool8 layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,
        TypeAttr:$index_element_type,

        IntAttr:$axis
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32, SI64, SI8, UI8]>:$output,
        RankedTensorOf<[SI32, SI64]>:$output_index
    );

}


//
// AdaptiveAvgPoolOp
//

def IE_AdaptiveAvgPoolOp :
        IE_LayerOp<
            "AdaptiveAvgPool"
        > {
    let summary = "InferenceEngine AdaptiveAvgPool layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        1DTensorOf<[SI32, SI64]>:$pooled_spatial_shape
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AdaptiveMaxPoolOp
//

def IE_AdaptiveMaxPoolOp :
        IE_LayerOp<
            "AdaptiveMaxPool"
        > {
    let summary = "InferenceEngine AdaptiveMaxPool layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        1DTensorOf<[SI32, SI64]>:$pooled_spatial_shape,
        TypeAttr:$index_element_type
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output,
        RankedTensorOf<[SI32, SI64]>:$output_index

    );
}

//
// ShuffleChannelsOp
//

def IE_ShuffleChannelsOp :
        IE_LayerOp<
            "ShuffleChannels",
            [
                SameOperandsAndResultType,
                SameOperandsAndResultShape
            ]
        > {
    let summary = "InferenceEngine ShuffleChannels layer";

    let arguments = (ins
        4DTensorOf<[F16, F32]>:$input,

        IntAttr:$axis,
        IntAttr:$group
    );

    let results = (outs
        4DTensorOf<[F16, F32]>:$output
    );
}

//
// GatherOp
//

def IE_GatherOp :
        IE_LayerOp<
            "Gather"
        > {
    let summary = "InferenceEngine Gather layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        Optional<RankedTensorOf<[AnyInteger]>>:$axis,
        OptionalAttr<IntAttr>:$axis_value,
        IntAttr:$batch_dims,

        OptionalAttr<IntAttr>:$indices_rank
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
}

//
// GatherNDOp
//

def IE_GatherNDOp :
        IE_LayerOp<
            "GatherND"
        > {
    let summary = "InferenceEngine GatherND layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,

        IntAttr:$batch_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
}

//
// GatherElementsOp
//

def IE_GatherElementsOp :
        IE_LayerOp<
              "GatherElements"
        > {
    let summary = "InferenceEngine GatherElements layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,

        IntAttr:$axis
    );

    let results = (outs
        AnyRankedTensor:$output
    );

}

//
// GatherTreeOp
//

def IE_GatherTreeOp :
        IE_LayerOp<
            "GatherTree"
        > {
    let summary = "InferenceEngine GatherTree layer";

    let arguments = (ins
        AnyRankedTensor:$stepIds,
        AnyRankedTensor:$parentIds,
        AnyRankedTensor:$maxSeqLen,
        AnyRankedTensor:$endToken
    );

    let results = (outs
        AnyRankedTensor:$finalIds
    );

    let hasVerifier = 1;
}

//
// YieldOp
//

def IE_YieldOp :
        IE_Op<
            "Yield",
            [
                HasParent<"IfOp">,
                DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
                Pure,
                Terminator
            ]
        > {
    let summary = "Terminator for wrapping operation";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$operands
    );

    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

    let hasVerifier = 1;
}

//
// IfOp
//

def IE_IfOp :
        IE_LayerOp<
            "If",
            [
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                                ["inferReturnTypeComponents"]>,
                SingleBlockImplicitTerminator<"YieldOp">,
                RecursiveMemoryEffects
            ]
        > {
    let summary = "InferenceEngine Conditional if layer";

    let description = [{
    Evaluates a Boolean condition and then takes one of two distinct execution
    paths. This implements the semantic If-then-else structure.
    }];

    let arguments = (ins
        1DTensorOf<[Bool8, SI8]>:$cond,
        Variadic<AnyRankedTensor>:$inputs
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$output
    );

    let regions = (region
        SizedRegion<1>:$then_branch,
        SizedRegion<1>:$else_branch
    );

    let assemblyFormat = [{
        `then_branch` `:` $then_branch
        `else_branch` `:` $else_branch
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let hasVerifier = 1;
}
//
// ScatterNDUpdateOp
//

def IE_ScatterNDUpdateOp :
        IE_LayerOp<
            "ScatterNDUpdate"
        > {
    let summary = "InferenceEngine ScatterNDUpdate layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        AnyRankedTensor:$updates
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ScatterUpdateOp
//

def IE_ScatterUpdateOp :
        IE_LayerOp<
            "ScatterUpdate"
        > {
    let summary = "InferenceEngine ScatterUpdate layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        AnyRankedTensor:$updates,
        Optional<AnyRankedTensor>:$axis,
        OptionalAttr<IntAttr>:$axis_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

     let hasCanonicalizer = 1;
}

//
// ScatterElementsUpdateOp
//

def IE_ScatterElementsUpdateOp :
        IE_LayerOp<
            "ScatterElementsUpdate"
        > {
    let summary = "InferenceEngine ScatterElementsUpdate layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        AnyRankedTensor:$updates,
        Optional<RankedTensorOf<[AnyInteger]>>:$axis,
        OptionalAttr<IntAttr>:$axis_value,
        IE_ScatterElementsUpdateReductionTypeAttr:$reduction,
        BoolAttr:$use_init_val
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//
// ClampOp
//

def IE_ClampOp :
        IE_LayerOp<
            "Clamp",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Clamp layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        F64Attr:$min,
        F64Attr:$max
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//
// EluOp
//

def IE_EluOp :
        IE_LayerOp<
            "Elu",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Elu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$x
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// ReshapeOp
//

def IE_ReshapeOp :
        IE_LayerOp<
            "Reshape",
            [
                IE_ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine Reshape layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$shape,

        UnitAttr:$special_zero,
        OptionalAttr<I64ArrayAttr>:$shape_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// SqueezeOp
//

def IE_SqueezeOp :
        IE_LayerOp<
            "Squeeze",
            [
                IE_ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine Squeeze layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// UnsqueezeOp
//

def IE_UnsqueezeOp :
        IE_LayerOp<
            "Unsqueeze",
            [
                IE_ViewLikeOpInterface
            ]
        > {

    let summary = "InferenceEngine Unsqueeze layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// SigmoidOp
//

def IE_SigmoidOp :
        IE_LayerOp<
            "Sigmoid",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sigmoid layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// GridSampleOp
//

def IE_GridSampleOp :
        IE_LayerOp<
            "GridSample"
        > {
    let summary = "InferenceEngine GridSample layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$grid,

        UnitAttr:$align_corners,
        OptionalAttr<IE_GridSampleModeAttr>:$mode,
        OptionalAttr<IE_GridSamplePaddingModeAttr>:$padding_mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// LRNOp
//

def IE_LRNOp :
        IE_LayerOp<
            "LRN"
        > {
    let summary = "InferenceEngine LRN layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<1DTensorOf<[SI32, SI64]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$size
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// LRN_IEOp
//

def IE_LRN_IEOp :
        IE_LayerOp<
            "LRN_IE"
        > {
    let summary = "InferenceEngine LRN_IE layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$size,
        IE_LRN_IERegionAttr:$region
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// BroadcastOp
//

def IE_BroadcastOp :
        IE_LayerOp<
            "Broadcast"
        > {
    let summary = "InferenceEngine Broadcast layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        1DTensorOf<[AnyInteger]>:$target_shape,
        Optional<1DTensorOf<[AnyInteger]>>:$axes_mapping,

        OptionalAttr<IE_BroadcastTypeAttr>:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

//
// BucketizeOp
//

def IE_BucketizeOp :
        IE_LayerOp<
            "Bucketize"
        > {
    let summary = "InferenceEngine Bucketize layer";

    let arguments = (ins
        AnyRankedTensor:$data,
        1DTensorOf<[AnyInteger, AnyFloat]>:$buckets,

        TypeAttr:$output_type,
        UnitAttr:$with_right_bound
    );

    let results = (outs
        RankedTensorOf<[SI32, SI64]>:$output
    );

    let hasVerifier = 1;
}

//
// ReduceMaxOp
//

def IE_ReduceMaxOp :
        IE_LayerOp<
            "ReduceMax"

        > {
    let summary = "InferenceEngine ReduceMax layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<TensorRankOf<[AnyInteger], [0, 1]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// ReduceMeanOp
//

def IE_ReduceMeanOp :
        IE_LayerOp<
            "ReduceMean"
        > {
    let summary = "InferenceEngine ReduceMean Layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<TensorRankOf<[AnyInteger], [0, 1]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// ReduceLogicalOrOp
//

def IE_ReduceLogicalOrOp :
        IE_LayerOp<
            "ReduceLogicalOr"
        > {
    let summary = "InferenceEngine ReduceLogicalOr layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<TensorRankOf<[AnyInteger], [0, 1]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// ReduceLogicalAndOp
//

def IE_ReduceLogicalAndOp :
        IE_LayerOp<
            "ReduceLogicalAnd"
        > {
    let summary = "InferenceEngine  ReduceLogicalAnd layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<TensorRankOf<[AnyInteger], [0, 1]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// ReduceProdOp
//

def IE_ReduceProdOp :
        IE_LayerOp<
            "ReduceProd"
        > {
    let summary = "InferenceEngine ReduceProd layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<TensorRankOf<[AnyInteger], [0, 1]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// ReduceSumOp
//

def IE_ReduceSumOp :
        IE_LayerOp<
            "ReduceSum"
        > {
    let summary = "InferenceEngine ReduceSum layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<TensorRankOf<[AnyInteger], [0, 1]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
    let hasVerifier = 1;
}

//
// ReduceMinOp
//

def IE_ReduceMinOp :
        IE_LayerOp<
            "ReduceMin"
        > {
    let summary = "InferenceEngine ReduceMin layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<TensorRankOf<[AnyInteger], [0, 1]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// ReduceL1Op
//

def IE_ReduceL1Op :
        IE_LayerOp<
            "ReduceL1"
        > {
    let summary = "InferenceEngine ReduceL1 layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<TensorRankOf<[AnyInteger], [0, 1]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// ReduceL2Op
//

def IE_ReduceL2Op :
        IE_LayerOp<
            "ReduceL2"
        > {
    let summary = "InferenceEngine ReduceL2 layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<TensorRankOf<[AnyInteger], [0, 1]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// MinimumOp
//

def IE_MinimumOp :
        IE_LayerOp<
            "Minimum",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Minimum layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// MaximumOp
//

def IE_MaximumOp :
        IE_LayerOp<
            "Maximum",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Maximum layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// FakeQuantizeOp
//

def IE_FakeQuantizeOp :
        IE_LayerOp<
            "FakeQuantize",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine FakeQuantize layer";

    let description = [{
        The operation works in two modes:
         * integral quantization: specified by the 'levels' attribute
         * floating-point quantization: specified by the 'low_fp_type' attribute, [f8E4M3FN | f8E5M2]

        Only one of these attributes should be provided.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$input_low,
        RankedTensorOf<[F16, F32]>:$input_high,
        RankedTensorOf<[F16, F32]>:$output_low,
        RankedTensorOf<[F16, F32]>:$output_high,

        OptionalAttr<IntAttr>:$levels,
        OptionalAttr<TypeAttr>:$low_fp_type,
        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
    let hasVerifier = 1;
}

//
// QuantizeOp
//

def IE_QuantizeOp :
        IE_LayerOp<"Quantize",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Quantize layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[quant_QuantizedType]>:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

//
// DequantizeOp
//

def IE_DequantizeOp :
        IE_LayerOp<"Dequantize",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Dequantize layer";

    let arguments = (ins
        RankedTensorOf<[quant_QuantizedType]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// DynamicQuantizeOp
//

def IE_DynamicQuantizeOp :
        IE_LayerOp<"DynamicQuantize"
        > {
    let summary = "InferenceEngine Dynamic-Quantize layer";

    let arguments = (ins
        RankedTensorOf<[F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[UI8]>:$output,
        1DTensorOf<[F32]>:$scale,
        1DTensorOf<[UI8]>:$zero_point
    );
}

//
// QuantizeCastOp
//

def IE_QuantizeCastOp :
        IE_LayerOp<
            "QuantizeCast",
            [
                IE_ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine Quantize Cast layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
    let hasVerifier = 1;
}

//
// MatMul
//

def IE_MatMulOp:
        IE_LayerOp<
            "MatMul",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                ShapeBoundOp
            ]
        > {
    let summary = "InferenceEngine MatMul layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, quant_QuantizedType]>:$input1,
        RankedTensorOf<[F16, F32, SI32, quant_QuantizedType]>:$input2,

        UnitAttr:$transpose_a,
        UnitAttr:$transpose_b,

        OptionalAttr<IE_PostOpAttr>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, quant_QuantizedType]>:$output
    );

    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION];
}

//
// TanhOp
//

def IE_TanhOp :
        IE_LayerOp<
            "Tanh",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Tanh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// SinOp
//

def IE_SinOp :
        IE_LayerOp<
            "Sin",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sin layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// CosOp
//

def IE_CosOp :
        IE_LayerOp<
            "Cos",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Cos layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// TanOp
//

def IE_TanOp :
        IE_LayerOp<
            "Tan",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Tan layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// SqrtOp
//

def IE_SqrtOp :
        IE_LayerOp<
            "Sqrt",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sqrt layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// SinhOp
//

def IE_SinhOp :
        IE_LayerOp<
            "Sinh",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sinh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// CoshOp
//

def IE_CoshOp :
        IE_LayerOp<
            "Cosh",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Cosh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AsinhOp
//

def IE_AsinhOp :
        IE_LayerOp<
            "Asinh",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Asinh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AcoshOp
//

def IE_AcoshOp :
        IE_LayerOp<
            "Acosh",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Acosh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// HardSigmoidOp
//

def IE_HardSigmoidOp :
        IE_LayerOp<
            "HardSigmoid",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine HardSigmoid layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$alpha,
        Optional<RankedTensorOf<[F16, F32]>>:$beta,

        OptionalAttr<F64Attr>:$alpha_value,
        OptionalAttr<F64Attr>:$beta_value
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//
// EmbeddingBagOffsetsSumOp
//

def IE_EmbeddingBagOffsetsSumOp :
        IE_LayerOp<
            "EmbeddingBagOffsetsSum",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine EmbeddingBagOffsetsSum layer";

    let arguments = (ins
        AnyRankedTensor:$emb_table,

        Optional<1DTensorOf<[SI64, SI32]>>:$indices,
        Optional<1DTensorOf<[SI64, SI32]>>:$offsets,
        Optional<RankedTensorOf<[SI64, SI32]>>:$default_index,
        Optional<1DTensorOf<[AnyInteger, AnyFloat]>>:$per_sample_weights,

        OptionalAttr<I64ArrayAttr>:$indices_value,
        OptionalAttr<I64ArrayAttr>:$offsets_value,
        OptionalAttr<IntAttr>:$default_index_value,
        OptionalAttr<F64ArrayAttr>:$per_sample_weights_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//
// EmbeddingSegmentsSumOp
//

def IE_EmbeddingSegmentsSumOp :
        IE_LayerOp<
            "EmbeddingSegmentsSum",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine EmbeddingSegmentsSum layer";

    let arguments = (ins
        AnyRankedTensor:$emb_table,
        Optional<1DTensorOf<[SI64, SI32]>>:$indices,
        Optional<1DTensorOf<[SI64, SI32]>>:$segment_ids,
        Optional<RankedTensorOf<[SI64, SI32]>>:$num_segments,
        Optional<RankedTensorOf<[SI64, SI32]>>:$default_index,
        Optional<1DTensorOf<[AnyInteger, AnyFloat]>>:$per_sample_weights,

        OptionalAttr<I64ArrayAttr>:$indices_value,
        OptionalAttr<I64ArrayAttr>:$segment_ids_value,
        OptionalAttr<IntAttr>:$num_segments_value,
        OptionalAttr<IntAttr>:$default_index_value,
        OptionalAttr<F64ArrayAttr>:$per_sample_weights_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//
// EmbeddingBagPackedSumOp
//

def IE_EmbeddingBagPackedSumOp :
        IE_LayerOp<
            "EmbeddingBagPackedSum"
        > {
    let summary = "InferenceEngine EmbeddingBagPackedSum layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$emb_table,
        2DTensorOf<[SI32, SI64]>:$indices,
        Optional<2DTensorOf<[F16, F32]>>:$per_sample_weights
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AbsOp
//

def IE_AbsOp :
        IE_LayerOp<
            "Abs",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Abs layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AtanOp
//

def IE_AtanOp :
        IE_LayerOp<
            "Atan",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Atan layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AsinOp
//

def IE_AsinOp :
        IE_LayerOp<
            "Asin",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Asin layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AcosOp
//

def IE_AcosOp :
        IE_LayerOp<
            "Acos",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Acos layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AtanhOp
//

def IE_AtanhOp :
        IE_LayerOp<
            "Atanh",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Atanh layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// HSigmoidOp
//

def IE_HSigmoidOp :
        IE_LayerOp<
            "HSigmoid",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine HSigmoid layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// LogOp
//

def IE_LogOp :
        IE_LayerOp<
            "Log",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Log layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// SeluOp
//

def IE_SeluOp :
        IE_LayerOp<
            "Selu",
            [
                IE_EltwiseOp,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine Selu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$data,
        Optional<RankedTensorOf<[F16, F32]>>:$alpha,
        Optional<RankedTensorOf<[F16, F32]>>:$lambda,

        OptionalAttr<F64Attr>:$alphaValue,
        OptionalAttr<F64Attr>:$lambdaValue
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}


//
// GeluOp
//

def IE_GeluOp :
        IE_LayerOp<
            "Gelu",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Gelu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// ExpOp
//

def IE_ExpOp :
        IE_LayerOp<
            "Exp",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Exp layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// HSwishOp
//

def IE_HSwishOp :
        IE_LayerOp<
            "HSwish",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine HSwish layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// FloorOp
//

def IE_FloorOp :
        IE_LayerOp<
            "Floor",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Floor layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// RoundOp
//

def IE_RoundOp :
        IE_LayerOp<
            "Round",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Round layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        IE_RoundModeAttr:$mode
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// MishOp
//

def IE_MishOp :
        IE_LayerOp<
            "Mish",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Mish layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// ErfOp
//

def IE_ErfOp :
        IE_LayerOp<
            "Erf",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Erf layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// TransposeOp
//

def IE_TransposeOp :
        IE_LayerOp<
            "Transpose",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "InferenceEngine Transpose layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$order,

        OptionalAttr<AffineMapAttr>:$order_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 0;
}

//
// ProposalOp
//

def IE_ProposalOp :
        IE_LayerOp<
            "Proposal"
        > {
    let summary = "InferenceEngine Proposal layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$class_probs,
        RankedTensorOf<[F16, F32]>:$bbox_deltas,
        RankedTensorOf<[F16, F32]>:$image_shape,

        IE_ProposalAttr:$proposal_attrs
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output,
        RankedTensorOf<[F16, F32]>:$probs
    );
}

//
// InterpolateOp
//

def IE_InterpolateOp :
        IE_LayerOp<
            "Interpolate",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine Interpolate layer";

    let arguments = (ins
        RankedTensorOf<[UI8, F16, F32, quant_QuantizedType]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$sizes,
        Optional<RankedTensorOf<[F16, F32]>>:$scales,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$sizes_attr,
        OptionalAttr<F64ArrayAttr>:$scales_attr,
        OptionalAttr<I64ArrayAttr>:$axes_attr,
        OptionalAttr<F64ArrayAttr>:$tile_offset_attr,
        OptionalAttr<I64ArrayAttr>:$initial_input_dims_attr,
        OptionalAttr<I64ArrayAttr>:$initial_output_dims_attr,

        IE_InterpolateAttr:$attr,
        OptionalAttr<IntAttr>:$output_channels
    );

    let results = (outs
        RankedTensorOf<[UI8, F16, F32, quant_QuantizedType]>:$output
    );
    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// TopKOp
//

def IE_TopKOp :
        IE_LayerOp<
            "TopK"
        > {
    let summary = "InferenceEngine TopK layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$k,
        OptionalAttr<IntAttr>:$k_value,

        IntAttr:$axis,
        IE_TopKModeAttr:$mode,
        IE_TopKSortTypeAttr:$sort,
        TypeAttr:$element_type
    );

    let results = (outs
        AnyRankedTensor:$output_values,
        AnyRankedTensor:$target_shape
    );

    let hasCanonicalizer = 1;
    let hasVerifier = 1;
}

//
// RegionYoloOp
//

def IE_RegionYoloOp :
        IE_LayerOp<
            "RegionYolo"
        > {
    let summary = "InferenceEngine RegionYolo layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$input,

        IntAttr:$coords,
        IntAttr:$classes,
        IntAttr:$num_regions,
        BoolAttr:$do_softmax,
        I64ArrayAttr:$mask,
        IntAttr:$axis,
        IntAttr:$end_axis,
        F64ArrayAttr:$anchors
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ReorgYoloOp
//

def IE_ReorgYoloOp :
        IE_LayerOp<
            "ReorgYolo"
        > {
    let summary = "InferenceEngine ReorgYolo layer";

    let arguments = (ins
        4DTensorOf<[AnyInteger, AnyFloat]>:$input,

        IntAttr:$stride
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// DetectionOutputOp
//

def IE_DetectionOutputOp :
        IE_LayerOp<
            "DetectionOutput",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine DetectionOutput layer";

    let arguments = (ins
        2DTensorOf<[AnyFloat]>:$in_box_logits,
        2DTensorOf<[AnyFloat]>:$in_class_preds,
        3DTensorOf<[AnyFloat]>:$in_proposals,
        Optional<2DTensorOf<[AnyFloat]>>:$in_additional_preds,
        Optional<2DTensorOf<[AnyFloat]>>:$in_additional_proposals,

        IE_DetectionOutputAttr:$attr
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// NormalizeL2Op
//

def IE_NormalizeL2Op :
        IE_LayerOp<
            "NormalizeL2"
        > {
    let summary = "InferenceEngine NormalizeL2 layer";

    let arguments = (ins
        AnyRankedTensor:$data,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,
        OptionalAttr<ArrayAttr>:$axes_value,

        F64Attr:$eps,
        IE_EpsModeAttr:$eps_mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//
// NormalizeIEOp
//

def IE_NormalizeIEOp :
        IE_LayerOp<
            "NormalizeIE"
        > {
    let summary = "InferenceEngine NormalizeIE layer";

    let arguments = (ins
        AnyRankedTensor:$data,
        AnyRankedTensor:$weights,

        F64Attr:$eps,
        BoolAttr:$across_spatial,
        BoolAttr:$channel_shared
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// CumSumOp
//

def IE_CumSumOp:
        IE_LayerOp<
            "CumSum"
        > {
    let summary = "InferenceEngine CumSum layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axis,

        OptionalAttr<IntAttr>:$axis_value,
        UnitAttr:$exclusive,
        UnitAttr:$reverse
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//
// EyeOp
//

def IE_EyeOp:
        IE_LayerOp<
            "Eye",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine Eye layer";

    let arguments = (ins
        Optional<1DTensorOf<[SI32, SI64]>>:$num_rows,
        Optional<1DTensorOf<[SI32, SI64]>>:$num_columns,
        1DTensorOf<[SI32, SI64]>:$diagonal_index,
        Optional<1DTensorOf<[SI32, SI64]>>:$batch_shape,

        OptionalAttr<IntAttr>:$num_rows_value,
        OptionalAttr<IntAttr>:$num_columns_value,
        OptionalAttr<I64ArrayAttr>:$batch_shape_value,

        TypeAttr:$outputType
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
}

//
// MVNOp
//

def IE_MVNOp :
        IE_LayerOp<
            "MVN"
        > {
    let summary = "InferenceEngine MVN1 layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        BoolAttr:$across_channels,
        BoolAttr:$normalize_variance,
        F64Attr:$eps,
        OptionalAttr<I64ArrayAttr>:$internal_reshape
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::BoolAttr":$across_channels,
            "::mlir::BoolAttr":$normalize_variance,
            "::mlir::FloatAttr":$eps
        )>
    ];

    let hasCanonicalizer = 1;
}

//
// MVN6Op
//

def IE_MVN6Op :
        IE_LayerOp<
            "MVN6",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine MVN6 layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<AnyRankedTensor>:$scale,
        Optional<AnyRankedTensor>:$bias,
        Optional<1DTensorOf<[SI32, SI64]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value,
        BoolAttr:$normalize_variance,
        F64Attr:$eps,
        IE_MvnEpsModeAttr:$eps_mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
}

//
// ConcatOp
//

def IE_ConcatOp :
        IE_LayerOp<
            "Concat",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "InferenceEngine Concat layer";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,

        OptionalAttr<IE_ConcatAttr>:$per_axis,
        OptionalAttr<I64ArrayOfArraysAttr>:$static_offsets
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "vpux::IE::ConcatAttr":$per_axis)
        >,
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "mlir::IntegerAttr":$axis,
                 CArg<"mlir::IntegerAttr", "{}">:$offset, CArg<"mlir::IntegerAttr", "{}">:$stride)
        >,
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "int64_t":$axis, CArg<"int64_t", "0">:$offset, CArg<"int64_t", "1">:$stride)
        >,
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "vpux::Dim":$axis, CArg<"int64_t", "0">:$offset, CArg<"int64_t", "1">:$stride)
        >,

        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "mlir::ArrayAttr":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::ArrayRef<vpux::Shape>":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::ArrayRef<vpux::ShapeRef>":$static_offsets)
        >,
    ];

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// ROIPoolingOp
//

def IE_ROIPoolingOp :
        IE_LayerOp<
            "ROIPooling"
        > {
    let summary = "InferenceEngine ROIPooling layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$coords,

        I64ArrayAttr:$output_size,
        F64Attr:$spatial_scale,
        IE_ROIPoolingMethodAttr:$method
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// PSROIPoolingOp
//

def IE_PSROIPoolingOp :
        IE_LayerOp<
            "PSROIPooling"
        > {
    let summary = "InferenceEngine PSROIPooling layer";

    let arguments = (ins
        4DTensorOf<[F16, F32]>:$input,
        2DTensorOf<[F16, F32]>:$coords,

        IntAttr:$output_dim,
        F64Attr:$spatial_scale,
        IntAttr:$group_size,
        OptionalAttr<IntAttr>:$spatial_bins_x,
        OptionalAttr<IntAttr>:$spatial_bins_y,
        OptionalAttr<IE_PSROIPoolingModeAttr>:$mode
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// ROIAlignOp
//

def IE_ROIAlignOp :
        IE_LayerOp<
            "ROIAlign",
            [
               ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine ROIAlign layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$coords,
        1DTensorOf<[AnyInteger]>:$roisIdx,

        IntAttr:$pooled_h,
        IntAttr:$pooled_w,
        IntAttr:$sampling_ratio,
        F64Attr:$spatial_scale,
        IE_ROIAlignMethodAttr:$poolingMode,
        IE_ROIAlignAlignedMethodAttr:$alignedMode
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// ExperimentalDetectronROIFeatureExtractorOp
//

def IE_ExperimentalDetectronROIFeatureExtractorOp :
        IE_LayerOp<
            "ExperimentalDetectronROIFeatureExtractor"
        > {
    let summary = "InferenceEngine ExperimentalDetectronROIFeatureExtractor layer";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,

        IE_ExperimentalDetectronROIFeatureExtractorAttr:$attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output,
        RankedTensorOf<[F16, F32]>:$outputROIs
    );
}


//
// StridedSliceOp
//

def IE_StridedSliceOp :
        IE_LayerOp<
            "StridedSlice",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<IE_ElemTypeInfoOpInterface>
            ]
        > {
    let summary = "InferenceEngine StridedSlice layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<1DTensorOf<[AnyInteger]>>:$begins,
        Optional<1DTensorOf<[AnyInteger]>>:$ends,
        Optional<1DTensorOf<[AnyInteger]>>:$strides,

        OptionalAttr<I64ArrayAttr>:$begins_attr,
        OptionalAttr<I64ArrayAttr>:$ends_attr,
        OptionalAttr<I64ArrayAttr>:$strides_attr,

        I64ArrayAttr:$begin_mask,
        I64ArrayAttr:$end_mask,
        I64ArrayAttr:$new_axis_mask,
        I64ArrayAttr:$shrink_axis_mask,
        I64ArrayAttr:$ellipsis_mask
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        bool isSimplified();
    }];

    let hasVerifier = 1;
}

//
// ConvolutionBackpropDataOp
//

def IE_ConvolutionBackpropDataOp:
        IE_LayerOp<
            "ConvolutionBackpropData"
        > {
    let summary = "InferenceEngine ConvolutionBackpropData layer";

    let description = [{
        Represents a transposed convolution, which consumes an input and filter tensor
        and generates an output larger than the input.

        Operands:
        - `input`: the input tensor of the operation; 4D layout [N, C_IN, Y, X]
        - `filter`: the convolutional kernel tensor; expected layout [C_IN, C_OUT, KY, KX]
        - (optional) `output_shape`: specifies the spatial shape of the output;
          expected values `[Y, X]`

        Attributes:
        - `strides`: represents the distance in pixels to slide the filter on the output
          tensor; expected values `[SY, SX]`
        - `pads_begin`: represents the number of pixels to remove from the beginning of
          each axis in the output; expected values `[PAD_TOP, PAD_LEFT]`
        - `pads_end`: represents the number of pixels to remove from the end of each axis
          in the output; expected values `[PAD_BOTTOM, PAD_RIGHT]`
        - `dilations`: has the same definition as dilations for a regular Convolution but
          applied in the backward way, for the output tensor; expected values `[DY, DX]`
        - `output_padding`: adds additional amount of paddings per each spatial axis in
          the output tensor; expected values `[PY; PX]`

        Results:
        - `output`: the output tensor of the operation; 4D layout [N, C_OUT, Y, X]
    }];

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$filter,
        Optional<1DTensorOf<[AnyInteger]>>:$output_shape,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$output_padding
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// GroupConvolutionBackpropData
//

def IE_GroupConvolutionBackpropDataOp:
        IE_LayerOp<
            "GroupConvolutionBackpropData"
        > {
    let summary = "InferenceEngine GroupConvolutionBackpropData layer";

    let description = [{
        Represents a grouped transposed convolution, which consumes an input and filter tensor
        and generates an output larger than the input.

        Compared to the ConvolutionBackpropData operation, the input and filter tensors are
        split into multiple groups whose results are computed independently and then concatenated.
        The number of groups can be deduced from the shape of the filter.

        Operands:
        - `input`: the input tensor of the operation; 4D layout [N, GROUPS * C_IN, Y, X]
        - `filter`: the convolutional kernel tensor; expected layout [GROUPS, C_IN, C_OUT, KY, KX]
        - (optional) `output_shape`: specifies the spatial shape of the output;
          expected values `[Y, X]`

        Attributes:
        - `strides`: represents the distance in pixels to slide the filter on the output
          tensor; expected values `[SY, SX]`
        - `pads_begin`: represents the number of pixels to remove from the beginning of
          each axis in the output; expected values `[PAD_TOP, PAD_LEFT]`
        - `pads_end`: represents the number of pixels to remove from the end of each axis
          in the output; expected values `[PAD_BOTTOM, PAD_RIGHT]`
        - `dilations`: has the same definition as dilations for a regular Convolution but
          applied in the backward way, for the output tensor; expected values `[DY, DX]`
        - `output_padding`: adds additional amount of paddings per each spatial axis in
          the output tensor; expected values `[PAD_Y; PAD_X]`

        Results:
        - `output`: the output tensor of the operation; 4D layout [N, GROUPS * C_OUT, Y, X]
    }];

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$filter,
        Optional<1DTensorOf<[AnyInteger]>>:$output_shape,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$output_padding
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// TransposedConvolutionOp
//

def IE_TransposedConvolutionOp:
        IE_LayerOp<
            "TransposedConvolution",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "TransposedConvolution layer";

    let description = [{
        Represents a transposed convolution, which consumes an input and filter tensor
        and generates an output larger than the input.

        Compared to the ConvolutionBackpropData operation, the filter expects the data
        in the [C_OUT, C_IN, KY, KX] format, similar to forward convolutions.

        Operands:
        - `input`: the input tensor of the operation; 4D layout [N, C_IN, Y, X]
        - `filter`: the convolutional kernel tensor; expected layout [C_OUT, C_IN, KY, KX]
        - (optional) `output_shape`: specifies the spatial shape of the output;
          expected values `[Y, X]`
        - (optional) `bias`: the bias tensor; 4D layout [N, C, H, W]

        Attributes:
        - `strides`: represents the distance in pixels to slide the filter on the output
          tensor; expected values `[SY, SX]`
        - `pads_begin`: represents the number of pixels to remove from the beginning of
          each axis in the output; expected values `[PAD_TOP, PAD_LEFT]`
        - `pads_end`: represents the number of pixels to remove from the end of each axis
          in the output; expected values `[PAD_BOTTOM, PAD_RIGHT]`
        - `dilations`: has the same definition as dilations for a regular Convolution but
          applied in the backward way, for the output tensor; expected values `[DY, DX]`
        - `output_padding`: adds additional amount of paddings per each spatial axis in
          the output tensor; expected values `[PY; PX]`

        Results:
        - `output`: the output tensor of the operation; 4D layout [N, C_OUT, Y, X]
    }];

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$filter,
        Optional<1DTensorOf<[AnyInteger]>>:$output_shape,
        Optional<RankedTensorOf<[F16, F32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$output_padding,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<IntAttr>:$output_channels,
        OptionalAttr<IntAttr>:$input_channels
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// GroupTransposedConvolutionOp
//

def IE_GroupTransposedConvolutionOp:
        IE_LayerOp<
            "GroupTransposedConvolution"
        > {
    let summary = "GroupTransposedConvolution layer";

    let description = [{
        Represents a grouped transposed convolution, which consumes an input and filter tensor
        and generates an output larger than the input.

        Compared to the TransposedConvolution operation, the input and filter tensors are
        split into multiple groups whose results are computed independently and then concatenated.
        The number of groups can be deduced from the shape of the filter.

        Additionally, compared to the GroupConvolutionBackpropData operation, the filter expects
        the data in the [GROUPS, C_OUT, C_IN, KY, KX] format, similar to forward group convolutions.

        Operands:
        - `input`: the input tensor of the operation; 4D layout [N, GROUPS * C_IN, Y, X]
        - `filter`: the convolutional kernel tensor; expected layout [GROUPS, C_OUT, C_IN, KY, KX]
        - (optional) `output_shape`: specifies the spatial shape of the output;
          expected values `[Y, X]`

        Attributes:
        - `strides`: represents the distance in pixels to slide the filter on the output
          tensor; expected values `[SY, SX]`
        - `pads_begin`: represents the number of pixels to remove from the beginning of
          each axis in the output; expected values `[PAD_TOP, PAD_LEFT]`
        - `pads_end`: represents the number of pixels to remove from the end of each axis
          in the output; expected values `[PAD_BOTTOM, PAD_RIGHT]`
        - `dilations`: has the same definition as dilations for a regular Convolution but
          applied in the backward way, for the output tensor; expected values `[DY, DX]`
        - `output_padding`: adds additional amount of paddings per each spatial axis in
          the output tensor; expected values `[PAD_Y; PAD_X]`

        Results:
        - `output`: the output tensor of the operation; 4D layout [N, GROUPS * C_OUT, Y, X]
    }];

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$filter,
        Optional<1DTensorOf<[AnyInteger]>>:$output_shape,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$output_padding,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<IntAttr>:$output_channels,
        OptionalAttr<IntAttr>:$input_channels
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// PReluOp
//

def IE_PReluOp :
        IE_LayerOp<
            "PRelu",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine PRelu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$negative_slope
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// LeakyReluOp
//

def IE_LeakyReluOp :
        IE_LayerOp<
            "LeakyRelu",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine LeakyRelu layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32, SI8, UI8]>:$input,

        F64Attr:$negative_slope
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType, SI32, SI8, UI8]>:$output
    );
    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// SwishOp
//

def IE_SwishOp :
        IE_LayerOp<
            "Swish",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Swish layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$beta,

        OptionalAttr<F64Attr>:$beta_value
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// ScaleShiftOp
//

def IE_ScaleShiftOp :
        IE_LayerOp<
            "ScaleShift",
            [
                AttrSizedOperandSegments,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine ScaleShift layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$weights,
        Optional<RankedTensorOf<[F16, F32]>>:$biases
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// UpsamplingOp
//

def IE_UpsamplingOp :
        IE_LayerOp<
            "Upsampling"
        > {
    let summary = "InferenceEngine Upsampling layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input,
        I64ArrayAttr:$upsampling_factor,
        OptionalAttr<IE_UpsamplingPadAttr>:$pad,
        OptionalAttr<IntAttr>:$output_channels
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );
}


//
// GRNOp
//

def IE_GRNOp :
        IE_LayerOp<
            "GRN"
        > {
    let summary = "InferenceEngine GRN layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$bias
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// NegativeOp
//

def IE_NegativeOp :
        IE_LayerOp<
            "Negative",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Negative layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI64]>:$output
    );
}

//
// SignOp
//

def IE_SignOp :
        IE_LayerOp<
            "Sign",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Sign layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// FullyConnected
//

def IE_FullyConnectedOp:
        IE_LayerOp<
            "FullyConnected",
            [
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                ShapeBoundOp
            ]
        > {
    let summary = "InferenceEngine FullyConnected layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input,
        RankedTensorOf<[F16, F32, SI32]>:$weights,
        Optional<RankedTensorOf<[F16, F32, SI32]>>:$bias
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// CTCGreedyDecoderOp
//

def IE_CTCGreedyDecoderOp :
        IE_LayerOp<
            "CTCGreedyDecoder"
        > {
    let summary = "InferenceEngine CTCGreedyDecoder layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$sequenceLengths,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// CTCGreedyDecoderSeqLenOp
//

def IE_CTCGreedyDecoderSeqLenOp :
        IE_LayerOp<
            "CTCGreedyDecoderSeqLen"
        > {
    let summary = "InferenceEngine CTCGreedyDecoderSeqLen layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[SI32, SI64]>:$sequenceLength,
        Optional<RankedTensorOf<[SI32, SI64]>>:$blankIndex,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        RankedTensorOf<[SI32, SI64]>:$output,
        RankedTensorOf<[SI32, SI64]>:$outputLength
    );
}

//
// PadOp
//

def IE_PadOp :
        IE_LayerOp<
            "Pad",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine Pad layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$pads_begin,
        Optional<RankedTensorOf<[AnyInteger]>>:$pads_end,
        Optional<RankedTensorOf<[AnyInteger, AnyFloat]>>:$pad_value,

        OptionalAttr<I64ArrayAttr>:$pads_begin_attr,
        OptionalAttr<I64ArrayAttr>:$pads_end_attr,
        OptionalAttr<F64Attr>:$pad_value_attr,

        IE_PadModeAttr:$mode,
        OptionalAttr<IntAttr>:$output_channels
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let assemblyFormat = [{
        `(` $input `)` (`[` $pads_begin^ `,` $pads_end (`,` $pad_value^)? `]`)? attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// ExpandOp
//

def IE_ExpandOp :
        IE_LayerOp<
            "Expand"
        > {
    let summary = "Expand tensor with uninitialized values";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "std::optional<vpux::ShapeRef>":$pads_begin, "std::optional<vpux::ShapeRef>":$pads_end)
        >
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
    let hasVerifier = 1;
}

//
// LSTMCellOp
//

def IE_LSTMCellOp :
        IE_LayerOp<
            "LSTMCell",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine LSTMCell layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$inputData,
        RankedTensorOf<[F16, F32]>:$initialHiddenState,
        RankedTensorOf<[F16, F32]>:$initialCellState,
        Optional<RankedTensorOf<[F16, F32]>>:$weights,
        RankedTensorOf<[F16, F32]>:$recurrenceWeights,
        Optional<RankedTensorOf<[F16, F32]>>:$biases,

        IntAttr:$hiddenSize
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$outputHiddenState,
        RankedTensorOf<[F16, F32]>:$outputCellState
    );
}

//
// LSTMGatesOp
//

def IE_LSTMGatesOp :
        IE_LayerOp<
            "LSTMGates"
        > {
    let summary = "Computes LSTM activation functions";

    let description = [{
        This operation is intended to be run as a software stage after computing and adding LSTM matrix multiplications.

        - **gatesInput** - tensor of shape **[batchSize, 4 * hiddenSize]** or **[1, 1, batchSize, 4 * hiddenSize]**. Formula:
            ```
            gatesInput = (inputData * weights) + (initialHiddenState * recurrenceWeights) + biases
            * - Matrix multiplication
            + - Element-wise add
            ```
        - The meaning of other operands are identical to those in LSTMCell operation.
    }];

    let arguments = (ins
        AnyTypeOf<[2DTensorOf<[F16, F32]>, 4DTensorOf<[F16, F32]>]>:$gatesInput,
        AnyTypeOf<[2DTensorOf<[F16, F32]>, 4DTensorOf<[F16, F32]>]>:$initialCellState
    );

    let results = (outs
        AnyTypeOf<[2DTensorOf<[F16, F32]>, 4DTensorOf<[F16, F32]>]>:$outputHiddenState,
        AnyTypeOf<[2DTensorOf<[F16, F32]>, 4DTensorOf<[F16, F32]>]>:$outputCellState
    );

    let hasVerifier = 1;
}

//
// SliceOp
//

def IE_SliceOp :
        IE_LayerOp<
            "Slice"
        > {
    let summary = "Extract single slice from tensor";

    let arguments = (ins
        AnyRankedTensor:$source,

        I64ArrayAttr:$static_offsets,
        I64ArrayAttr:$static_sizes
    );

    let results = (outs
        AnyRankedTensor:$result
    );

    let assemblyFormat = [{
        $source $static_offsets $static_sizes
        attr-dict `:` type($source) `to` type(results)
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ShapeRef":$static_offsets, "vpux::ShapeRef":$static_sizes)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ArrayRef<int64_t>":$static_offsets, "vpux::ArrayRef<int64_t>":$static_sizes)
        >
    ];

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// SubtractOp
//

def IE_SubtractOp :
        IE_LayerOp<
            "Subtract",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Subtract layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$input1,
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<IntAttr>:$output_channels,
        OptionalAttr<IntAttr>:$input_channels
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI64, UI8, UI16, UI32, UI64, quant_QuantizedType]>:$output
    );

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION];
}

//
// LSTMSequenceOp
//

def IE_LSTMSequenceOp :
        IE_LayerOp<
            "LSTMSequence",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine LSTMSequence layer";

    let description = [{
        This operation is for InferenceEngine LSTMSequence layer:

          * The sequenceLength attribute is not mandatory, which allows us to support dynamic
            shape cases where the sequenceLength is initially unknown. In such cases, we will
            derive the necessary information from the inputData and keep the sequenceLength
            attribute empty.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$inputData,
        RankedTensorOf<[F16, F32]>:$initialHiddenState,
        RankedTensorOf<[F16, F32]>:$initialCellState,
        Optional<RankedTensorOf<[F16, F32]>>:$weights,
        RankedTensorOf<[F16, F32]>:$reccurenceWeights,
        Optional<RankedTensorOf<[F16, F32]>>:$biases,

        OptionalAttr<IntAttr>:$sequenceLength,
        IE_RNNSequenceDirectionAttr:$direction
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$outputHiddenValues,
        RankedTensorOf<[F16, F32]>:$outputHiddenState,
        RankedTensorOf<[F16, F32]>:$outputCellState
    );
}

//
// MemPermuteOp
//

def IE_MemPermuteOp :
        IE_LayerOp<
            "MemPermute"
        > {
    let summary = "InferenceEngine MemPermute layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 0;
}

//
// PermuteCastOp
//

def IE_PermuteCastOp :
        IE_LayerOp<
            "PermuteCast",
            [
                IE_ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine PermuteCast layer";

    let description = [{
        The op changes layout information in the following way:
            * dst_order: layout attribute of result is set to value of this arg
            * mem_perm: describes the permutation applied on the input value's memory shape
                        to obtain the memory shape of the output value.
    }];

    let arguments = (ins
        AnyRankedTensor:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// CeilingOp
//

def IE_CeilingOp :
        IE_LayerOp<
            "Ceiling",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Ceiling layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// EqualOp
//

def IE_EqualOp :
        IE_LayerOp<
            "Equal",
            [
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Equal layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[Bool8]>:$output
    );
}

//
// SelectOp
//

def IE_SelectOp :
        IE_LayerOp<
            "Select",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Select layer";

    let arguments = (ins
        AnyRankedTensor:$input1,
        RankedTensorOf<[SI32, SI64, F16, F32]>:$input2,
        RankedTensorOf<[SI32, SI64, F16, F32]>:$input3,
        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// SpaceToDepthOp
//

def IE_SpaceToDepthOp :
        IE_LayerOp<
            "SpaceToDepthOp"
        > {
    let summary = "InferenceEngine SpaceToDepthOp layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        DefaultValuedAttr<IntAttr, "1">:$block_size,
        IE_SpaceToDepthModeAttr:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

//
// SpaceToBatch
//

def IE_SpaceToBatch :
        IE_LayerOp<
            "SpaceToBatch",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine SpaceToBatch layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<1DTensorOf<[AnyInteger]>>:$block_shape,
        Optional<1DTensorOf<[AnyInteger]>>:$pads_begin,
        Optional<1DTensorOf<[AnyInteger]>>:$pads_end,

        OptionalAttr<I64ArrayAttr>:$block_shape_value,
        OptionalAttr<I64ArrayAttr>:$pads_begin_value,
        OptionalAttr<I64ArrayAttr>:$pads_end_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
}

//
// BatchToSpace
//

def IE_BatchToSpace :
        IE_LayerOp<
            "BatchToSpace",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine BatchToSpace layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<1DTensorOf<[AnyInteger]>>:$block_shape,
        Optional<1DTensorOf<[AnyInteger]>>:$crops_begin,
        Optional<1DTensorOf<[AnyInteger]>>:$crops_end,

        OptionalAttr<I64ArrayAttr>:$block_shape_value,
        OptionalAttr<I64ArrayAttr>:$crops_begin_value,
        OptionalAttr<I64ArrayAttr>:$crops_end_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
}

//
//ReverseOp
//

def IE_ReverseOp :
        IE_LayerOp<
            "Reverse"
        > {

    let summary = "Reverse operations reverse specified axis in an input tensor.";

    let arguments = (ins
        AnyRankedTensor:$input,
        Optional<1DTensorOf<[AnyInteger]>>:$axis,

        OptionalAttr<I64ArrayAttr>:$axis_value,

        IE_ReverseModeAttr:$mode
    );

    let results = (outs
    AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;

}

//
// AffineReshapeOp
//

def IE_AffineReshapeOp :
        IE_LayerOp<
            "AffineReshape",
            [
                IE_ViewLikeOpInterface
            ]
        > {
    let summary = "AffineReshape layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayOfArraysAttr:$dim_mapping,
        I64ArrayAttr:$shape_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;

    let hasVerifier = 1;
}

//
// NotEqualOp
//

def IE_NotEqualOp :
        IE_LayerOp<
            "NotEqual",
            [
                Commutative,
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine NotEqual layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input1,
        RankedTensorOf<[F16, F32, SI32, SI64]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[Bool8]>:$output
    );
}

//
// ReverseSequenceOp
//

def IE_ReverseSequenceOp :
        IE_LayerOp<
            "ReverseSequence"
        > {
    let summary = "Reverse variable length sequence operation";

    let arguments = (ins
        AnyRankedTensor:$data,
        1DTensorOf<[AnyInteger]>:$seq_length,

        IntAttr:$seq_axis,
        IntAttr:$batch_axis
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

//
// DepthToSpaceOp
//

def IE_DepthToSpaceOp :
        IE_LayerOp<
            "DepthToSpace"
        > {
    let summary = "InferenceEngine DepthToSpace layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        IntAttr:$block_size,
        IE_DepthToSpaceModeAttr:$mode,
        OptionalAttr<IE_ChannelPaddingAttr>:$padded_channels
    );

    let builders = [
    OpBuilder<(ins
            "mlir::Value":$input, "int64_t":$block_size, "IE::DepthToSpaceMode":$mode
        )>,
      OpBuilder<(ins
            "mlir::Value":$input, "mlir::IntegerAttr":$block_size, "IE::DepthToSpaceModeAttr":$mode
        )>,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::Value":$input, "mlir::IntegerAttr":$block_size, "IE::DepthToSpaceModeAttr":$mode)
        >,
    ];

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

//
// SoftPlusOp
//

def IE_SoftPlusOp :
        IE_LayerOp<
            "SoftPlus",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine SoftPlus layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// CopyOp
//

def IE_CopyOp :
        IE_LayerOp<
            "Copy"
        > {
    let summary = "InferenceEngine Copy layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        OptionalAttr<IndexedSymbolAttr>:$out_mem_space
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// ExtractImagePatchesOp
//

def IE_ExtractImagePatchesOp :
        IE_LayerOp<
            "ExtractImagePatches"
        > {
    let summary = "InferenceEngine ExtractImagePatches layer";

    let arguments = (ins
        4DTensorOf<[AnyType]>:$data,

        I64ArrayAttr:$sizes,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$rates,
        IE_PadTypeAttr:$autoPad
    );

    let results = (outs
        4DTensorOf<[AnyType]>:$output
    );
}

// YuvToRgbOp
//  Conversions:
//   NV12toRGB, NV12toBGR,
//   I420toRGB, I420toBGR
//

def IE_YuvToRgbOp :
        IE_LayerOp<
            "YuvToRgb",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine NV12/I420 to RGB/BGR layer";

    let arguments = (ins
                 4DTensorOf<[UI8, F16, F32]> :$input1,
        Optional<4DTensorOf<[UI8, F16, F32]>>:$input2,
        Optional<4DTensorOf<[UI8, F16, F32]>>:$input3,

        IE_ColorFmtAttr:$inFmt,
        IE_ColorFmtAttr:$outFmt
    );

    let results = (outs
        4DTensorOf<[UI8, F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// RandomUniformOp
//

def IE_RandomUniformOp :
        IE_LayerOp<
            "RandomUniform"
        > {
    let summary = "InferenceEngine RandomUniform layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$min,
        RankedTensorOf<[F16, F32, SI32]>:$max,

        I64ArrayAttr:$output_shape,
        TypeAttr:$outputType,
        IntAttr:$global_seed,
        IntAttr:$op_seed
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
    );

    let hasVerifier = 1;
}

//
// OneHotOp
//

def IE_OneHotOp :
        IE_LayerOp<
            "OneHot",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine OneHot layer";

    let arguments = (ins
        RankedTensorOf<[SI32, SI64]>:$input,
        Optional<RankedTensorOf<[SI32, SI64]>>:$depth,
        Optional<RankedTensorOf<[AnyType]>>:$on_value,
        Optional<RankedTensorOf<[AnyType]>>:$off_value,

        OptionalAttr<IntAttr>:$depth_attr,
        OptionalAttr<F64Attr>:$on_value_attr,
        OptionalAttr<F64Attr>:$off_value_attr,

        IntAttr:$axis_attr,
        TypeAttr:$outputType
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//
// BatchNormInferenceOp
//

def IE_BatchNormInferenceOp :
        IE_LayerOp<
            "BatchNormInference",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine BatchNormInference layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        Optional<RankedTensorOf<[F16, F32]>>:$gamma,
        Optional<RankedTensorOf<[F16, F32]>>:$beta,
        Optional<RankedTensorOf<[F16, F32]>>:$mean,
        Optional<RankedTensorOf<[F16, F32]>>:$variance,

        OptionalAttr<F64ArrayAttr>:$gamma_value,
        OptionalAttr<F64ArrayAttr>:$beta_value,
        OptionalAttr<F64ArrayAttr>:$mean_value,
        OptionalAttr<F64ArrayAttr>:$variance_value,

        F64Attr:$eps
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasCanonicalizer = 1;
}

//
// ExpandDilatedOp
//

def IE_ExpandDilatedOp :
        IE_LayerOp<
            "ExpandDilated"
        > {
    let summary = "Expand tensor with uninitialized values according to dilations";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayAttr:$dilations
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

//
// StubOp
//

def IE_StubOp :
        IE_Op<
            "Stub",
            [
                Pure
            ]
        > {
    let summary = "Substitute operation for stubbing.";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// AssignOp
//

def IE_AssignOp :
        IE_Op<
            "Assign",
            [
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                DeclareOpInterfaceMethods<IE_LayerOpInterface>
            ]
        > {
    let summary = "InferenceEngine Assign layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        StrAttr:$name
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// ReadValueOp
//

def IE_ReadValueOp :
        IE_Op<
            "ReadValue",
            [
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                DeclareOpInterfaceMethods<IE_LayerOpInterface>
            ]
        > {
    let summary = "InferenceEngine ReadValue layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        StrAttr:$name
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// GRUCellOp
//

def IE_GRUCellOp :
        IE_LayerOp<
            "GRUCell"
        > {
    let summary = "InferenceEngine GRUCell layer";

    let arguments = (ins
        2DTensorOf<[F16, F32]>:$input_data,
        2DTensorOf<[F16, F32]>:$initial_hidden_state,
        2DTensorOf<[F16, F32]>:$weights,
        2DTensorOf<[F16, F32]>:$recurrence_weights,
        Optional<1DTensorOf<[F16, F32]>>:$biases,

        IntAttr:$hidden_size,
        UnitAttr:$should_linear_before_reset,
        F64Attr:$clip
    );

    let results = (outs
        2DTensorOf<[F16, F32]>:$output_hidden_state
    );
}

//
// GRUSequence
//

def IE_GRUSequenceOp :
        IE_LayerOp<
            "GRUSequence"
        > {
    let summary = "InferenceEngine GRUSequence layer";

    let arguments = (ins
        3DTensorOf<[F16, F32]>:$input_data,
        3DTensorOf<[F16, F32]>:$initial_hidden_state,
        3DTensorOf<[F16, F32]>:$weights,
        3DTensorOf<[F16, F32]>:$recurrence_weights,
        2DTensorOf<[F16, F32]>:$biases,

        IntAttr:$hidden_size,
        IntAttr:$seq_length,
        IE_RNNSequenceDirectionAttr:$direction,
        UnitAttr:$should_linear_before_reset,
        F64Attr:$clip
    );

    let results = (outs
        4DTensorOf<[F16, F32]>:$middle_hidden_state,
        3DTensorOf<[F16, F32]>:$output_hidden_state
    );
}

//
// GRUSequenceLastPart
//

def IE_GRUSequenceLastPartOp :
        IE_LayerOp<
            "GRUSequenceLastPart"
        > {
    let summary = "GRUSequence can be split into two parts, GRUSequenceFirstPart and GRUSequenceLastPart.";

    let description = [{
        GRUSequenceOp has five inputs and two outputs, and it can be split into GRUSequenceFirstPart and
        GRUSequenceLastPart. The calculation method of GRUSequenceFirstPart is similar to matrix multiply,
        and GRUSequenceLastPart will use output of GRUSequenceFirstPart and other inputs of GRUSequence to
        get outputs of GRUSequence.

        The split solution can be used for performance improvement since GRUSequenceFirstPart can be converted
        to HW convolution to utilize hardware feature of NPU, and can be used to meet CMX requirement when
        feasible tiling strategy can't be generated.
    }];

    let arguments = (ins
        4DTensorOf<[F16, F32]>:$first_part_output,
        3DTensorOf<[F16, F32]>:$initial_hidden_state,
        3DTensorOf<[F16, F32]>:$recurrence_weights,
        2DTensorOf<[F16, F32]>:$biases,

        IntAttr:$hidden_size,
        IntAttr:$seq_length,
        IE_RNNSequenceDirectionAttr:$direction,
        UnitAttr:$should_linear_before_reset,
        F64Attr:$clip
    );

    let results = (outs
        4DTensorOf<[F16, F32]>:$middle_hidden_state,
        3DTensorOf<[F16, F32]>:$output_hidden_state
    );
}

//
// DeformablePSROIPoolingOp
//

def IE_DeformablePSROIPoolingOp :
        IE_LayerOp<
            "DeformablePSROIPooling"
        > {
    let summary = "InferenceEngine DeformablePSROIPooling layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$input_score_maps,
        2DTensorOf<[AnyFloat]>:$input_rois,
        Optional<4DTensorOf<[AnyFloat]>>:$input_transformations,

        IntAttr:$output_dim,
        F64Attr:$spatial_scale,
        OptionalAttr<IntAttr>:$group_size,
        OptionalAttr<IntAttr>:$spatial_bins_x,
        OptionalAttr<IntAttr>:$spatial_bins_y,
        OptionalAttr<F64Attr>:$trans_std,
        OptionalAttr<IntAttr>:$part_size,
        OptionalAttr<IE_DeformablePSROIPoolingModeAttr>:$mode
    );

    let results = (outs
        4DTensorOf<[AnyFloat]>:$output
    );
}

//
// NonMaxSuppressionOp
//

def IE_NonMaxSuppressionOp:
        IE_LayerOp<
            "NonMaxSuppression",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine NonMaxSuppression layer";

    let arguments = (ins
        3DTensorOf<[F16, F32]>:$in_box_coords,
        3DTensorOf<[F16, F32]>:$in_box_scores,
        Optional<1DTensorOf<[SI32, SI64]>>:$max_output_boxes_per_class,
        Optional<1DTensorOf<[F16, F32]>>:$iou_threshold,
        Optional<1DTensorOf<[F16, F32]>>:$score_threshold,
        Optional<1DTensorOf<[F16, F32]>>:$soft_nms_sigma,

        IE_BoxEncodingTypeAttr:$box_encoding,
        UnitAttr:$sort_result_descending,

        OptionalAttr<IntAttr>:$max_output_boxes_per_class_value,
        OptionalAttr<F64Attr>:$iou_threshold_value,
        OptionalAttr<F64Attr>:$score_threshold_value,
        OptionalAttr<F64Attr>:$soft_nms_sigma_value
    );

    let results = (outs
        2DTensorOf<[SI32]>:$out_selected_indices,
        2DTensorOf<[F16, F32]>:$out_selected_scores,
        1DTensorOf<[SI32]>:$out_valid_outputs
    );

    let hasCanonicalizer = 1;
}

//
// PermuteQuantizeOp
//

def IE_PermuteQuantizeOp :
        IE_LayerOp<
            "PermuteQuantize"
        > {
    let summary = "InferenceEngine PermuteQuantize layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm,
        TypeAttr:$dstElemType,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

//
// ShapeCastOp
//

def IE_ShapeCastOp :
        IE_LayerOp<
            "ShapeCast",
            [
                IE_ViewLikeOpInterface
            ]
        > {
    let summary = "ShapeCast layer";

    let arguments = (ins
        AnyRankedTensor:$source,
        I64ArrayAttr:$shape
    );

    let results = (outs
        AnyRankedTensor:$result
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $source `:` type($source) `)`
        `->` type(results)
    }];

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// LayoutCastOp
//

def IE_LayoutCastOp :
        IE_LayerOp<
            "LayoutCast",
            [
                IE_ViewLikeOpInterface
            ]
        > {
    let summary = "This layer overrides layout of a given tensor.";

    let arguments = (ins
        AnyRankedTensor:$input,
        AffineMapAttr:$dst_order
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// LoopTerminatorOp
//

def IE_LoopTerminatorOp :
        IE_Op<
            "LoopTerminator",
            [
                ParentOneOf<["TensorIteratorOp", "LoopOp"]>,
                DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
                Pure,
                Terminator
            ]
        > {
    let summary = "Terminator for wrapping TensorIterator or Loop body module";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$operands
    );

    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

    let hasVerifier = 1;
}

//
// TensorIteratorOp
//

def IE_TensorIteratorOp :
        IE_LayerOp<
            "TensorIterator",
            [
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                                ["inferReturnTypeComponents"]>,
                SingleBlockImplicitTerminator<"LoopTerminatorOp">,
                RecursiveMemoryEffects
            ]
        > {
    let summary = "InferenceEngine Tensor Iterator layer";

    let description = [{
    TensorIterator layer performs recurrent execution of the network, which is described in the body, iterating through the data along axis if given.
    }];

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,
        IntAttr:$num_iterations,
        OptionalAttr<ArrayAttr>:$slice_input_descs,
        OptionalAttr<ArrayAttr>:$invariant_input_descs,
        OptionalAttr<ArrayAttr>:$feedback_input_descs,
        OptionalAttr<ArrayAttr>:$concat_output_descs,
        OptionalAttr<ArrayAttr>:$invariant_output_descs
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$output
    );

    let regions = (region
        SizedRegion<1>:$body_module
    );

    let assemblyFormat = [{
        `body_module` `:` $body_module `\n`
        `num_iterations` `:` $num_iterations `\n`
        `slice_input_descs` `:` $slice_input_descs `\n`
        `invariant_input_descs` `:` $invariant_input_descs `\n`
        `feedback_input_descs` `:` $feedback_input_descs `\n`
        `concat_output_descs` `:` $concat_output_descs `\n`
        `invariant_output_descs` `:` $invariant_output_descs `\n`
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// LoopOp
//

def IE_LoopOp :
        IE_LayerOp<
            "Loop",
            [
                DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                                ["inferReturnTypeComponents"]>,
                SingleBlockImplicitTerminator<"LoopTerminatorOp">,
                RecursiveMemoryEffects
            ]
        > {
    let summary = "InferenceEngine Loop layer";

    let description = [{
    Loop layer performs recurrent execution of the network, which is described in the body, iterating through the data along axis if given.
    Besides, it has current_iter and execution_cond params to determine if the loop go on or terminate.

    Inputs and Output are operands and results of Loop Op, i.e. op input and op output.
    Specially, for Loop Op, its inputs is: [trip_cpunt, condition, input1,  input2,  input3, ... ]
    trip_count: max number of iteration, usually connected to a constant node, but can be a uncertain parameter
    condition: the boolean to decide whether this loop will be executed or not in the first iteration

    The region body_module is the network which will be executed repeatedly by Loop.
    It has own input and output, i.e. body input and body output.

    Attributes:
    num_iterations is the number of iterations, which is inferred according to given trip_count(max number of iteration), execution_conditon
    (cosntant true or false will lead to a certain number of iterations) at it was built by Openvino.

    current_iter_index is the index of one special param of body input.
    It is the current iteration. If it is set as -1, that means there is no such param.

    exec_cond_index is the index of one special result of body output.
    It is the next iteration execution condition. It is required, cannot be -1 (i.e. unset).

    descs is a vector of descriptions on `op I/O index` and `body I/O index` mappings.
    For example, invariant_input_descs contains one or more IE::InvariantInputPortMapAttr attributes.
    Given one IE::InvariantInputPortMapAttr with externalPortId = 3 and internalLayerId = 5, it
    means op input[3] will be transformed to body input[5].
    All of these descs have externalPortId and internalLayerId as mapping.
    According to different processings on I/O, there are 5 kinds of descs below.

    slice_input_descs: attrs inside are IE::SliceInputPortMapAttr.
    It has axis, begin, end, stride, part_size, to help slice the op input along the axis
    from begin to end into part_size as body input while stride being +1 or -1 decides the direction.
    For example, IE.SliceInputPortMap<
    external_port_id = 2 : i64,
    internal_layer_id = 0 : i64,
    axis = 1 : i64,
    start = 0 : i64,
    stride = 1 : i64,
    part_size = 1 : i64,
    end = 2 : i64>
    Will take op input[2] and slice it along axis 1 from index 0 until index 2 into (end - start)/part_size = 2 pieces.
    And the first piece will be body input[0] in the first iteration.
    The second one will be body input[0] in the second iteration.

    concat_output_descs: attrs inside are IE::ConcatOutputPortMapAttr.
    It has axis, begin, end, stride, part_size, to help concat the body output along the axis
    into op output while stride being +1 or -1 decides the direction.

    invariant_input_descs: attrs inside are IE::InvariantInputPortMapAttr.
    It will map op input to body input in every iteration.

    invariant_output_descs: attrs inside are IE::InvariantOutputPortMapAttr.
    It will map body output to op output in the last iteration.

    feedback_input_descs: attrs inside are IE::MergedInputPortMapAttr.
    It describes how body output will be reused as body input after the first iteration.
    It has the body_input_index to mapping the body output[body_input_index] to body input[internalLayerId]
    after the first iteration.
    }];

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,
        IntAttr:$num_iterations,
        IntAttr:$current_iter_index,
        IntAttr:$exec_cond_index,
        OptionalAttr<ArrayAttr>:$slice_input_descs,
        OptionalAttr<ArrayAttr>:$invariant_input_descs,
        OptionalAttr<ArrayAttr>:$feedback_input_descs,
        OptionalAttr<ArrayAttr>:$concat_output_descs,
        OptionalAttr<ArrayAttr>:$invariant_output_descs
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$output
    );

    let regions = (region
        SizedRegion<1>:$body_module
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results) `\n`
        `(` `num_iterations` `:` $num_iterations `current_iter_index` `:` $current_iter_index `exec_cond_index` `:` $exec_cond_index  `)` `\n`
        `slice_input_descs` `:` $slice_input_descs `\n`
        `invariant_input_descs` `:` $invariant_input_descs `\n`
        `feedback_input_descs` `:` $feedback_input_descs `\n`
        `concat_output_descs` `:` $concat_output_descs `\n`
        `invariant_output_descs` `:` $invariant_output_descs `\n`
        `body_module` `:` $body_module
    }];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//
// LoopSelectOp
//

def IE_LoopSelectOp :
        IE_LayerOp<
            "LoopSelect"
        > {
    let summary = "Select a slice of the input according to collected execution conditions, helping implement Loop Op.";

    let arguments = (ins
        1DTensorOf<[Bool8, SI8]>:$initExecCond,
        1DTensorOf<[Bool8, SI8]>:$execConds,
        AnyRankedTensor:$input,

        BoolAttr:$do_concat,
        IntAttr:$axis,
        IntAttr:$stride
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
}

//
// DFTOp
//

def IE_DFTOp :
        IE_LayerOp<
            "DFT",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine DFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,
        Optional<RankedTensorOf<[AnyInteger]>>:$signal_size,

        OptionalAttr<I64ArrayAttr>:$axes_attr,
        OptionalAttr<I64ArrayAttr>:$signal_size_attr
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
    let hasCanonicalizer = 1;
}

//
// RDFTOp
//

def IE_RDFTOp :
        IE_LayerOp<
            "RDFT",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine RDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,
        Optional<RankedTensorOf<[AnyInteger]>>:$signal_size,

        OptionalAttr<I64ArrayAttr>:$axes_attr,
        OptionalAttr<I64ArrayAttr>:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
    let hasCanonicalizer = 1;
}

//
// IDFTOp
//

def IE_IDFTOp :
        IE_LayerOp<
            "IDFT",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine IDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,
        Optional<RankedTensorOf<[AnyInteger]>>:$signal_size,

        OptionalAttr<I64ArrayAttr>:$axes_attr,
        OptionalAttr<I64ArrayAttr>:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
    let hasCanonicalizer = 1;
}

//
// IRDFTOp
//

def IE_IRDFTOp :
        IE_LayerOp<
            "IRDFT",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine IRDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,
        Optional<RankedTensorOf<[AnyInteger]>>:$signal_size,

        OptionalAttr<I64ArrayAttr>:$axes_attr,
        OptionalAttr<I64ArrayAttr>:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
    let hasCanonicalizer = 1;
}

//
// AccumulateOp
//

def IE_AccumulateOp :
        IE_LayerOp<
            "Accumulate",
            [
                AttrSizedOperandSegments,
                IE_EltwiseOp
            ]
        > {
    let summary = "Accumulates intermediate results of IE.MatMul";

    let description = [{
        This operation maintains the semantics of `IE.Add` except that it does not support any
        hardware-specific features like permutations, post-ops, broadcasting.
        ExpandActivationChannels, AdjustInputShapeForEltwise and other DPU specific passes do not
        apply to `IE.Accumulate`.
        The point of using `IE.Accumulate` instead of `IE.Add` is to be able to execute it as a
        software operation in parallel to DPU tasks that perform `IE.MatMul` in group quantization case:
        ```
                     Input [N, G * H] -> Split
                                        /     \
                                  [N, H]  ...  [N, H]
                                      |           |
            IE.FQ [1, H, W] -> IE.MatMul ...  IE.MatMul <- IE.FQ
                                       \          /
                                      IE.Accumulate ([N, W], [N, W])
        ```
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64, quant_QuantizedType]>:$lhs,
        RankedTensorOf<[F16, F32, SI32, SI64, quant_QuantizedType]>:$rhs,
        Optional<RankedTensorOf<[F16, F32]>>:$lhsScale,
        Optional<RankedTensorOf<[F16, F32]>>:$rhsScale
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI64, quant_QuantizedType]>:$output
    );

    let hasVerifier = 1;
}

//
// ShapeOfOp
//

def IE_ShapeOfOp :
        IE_LayerOp<
            "ShapeOf"
        > {
    let summary = "InferenceEngine ShapeOf layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        1DTensorOf<[SI32, SI64]>:$output
    );

    let hasFolder = 1;
}

//
// RangeOp
//

def IE_RangeOp :
        IE_LayerOp<
            "Range"
        > {
    let summary = "InferenceEngine Range layer";

    let arguments = (ins
        AnyRankedTensor:$start,
        AnyRankedTensor:$stop,
        AnyRankedTensor:$step,
        TypeAttr:$dstElemType
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// NonZeroOp
//

def IE_NonZeroOp :
        IE_LayerOp<
            "NonZero"
        > {
    let summary = "InferenceEngine NonZero layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI64, Bool8]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        2DTensorOf<[SI32, SI64]>:$output
    );
}

//
// DynamicReshapeOp
//

def IE_DynamicReshapeOp :
        IE_LayerOp<
            "DynamicReshape"
        > {
    let summary = "InferenceEngine Reshape layer for dynamic shapes";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$shape,

        I64ArrayAttr:$output_shape,
        I64ArrayAttr:$output_bounds,

        UnitAttr:$only_set_shape
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let elemComparisonModes = [IE_TypeComparisonMode_STRICT_EQUAL];
    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;

    let hasVerifier = 1;
}

//
// DynamicBroadcastOp
//

def IE_DynamicBroadcastOp :
        IE_LayerOp<
            "DynamicBroadcast"
        > {
    let summary = "InferenceEngine Broadcast layer for dynamic shapes";

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$target_shape,
        Optional<1DTensorOf<[AnyInteger]>>:$axes_mapping,

        OptionalAttr<IE_BroadcastTypeAttr>:$mode,

        I64ArrayAttr:$output_shape,
        I64ArrayAttr:$output_bounds
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}


//
// DynamicFakeQuantizeOp
//

def IE_DynamicFakeQuantizeOp :
        IE_LayerOp<
            "DynamicFakeQuantize",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine DynamicFakeQuantize layer";

    let description = [{
        The operation works in two modes:
        * integral quantization: specified by the 'levels' attribute
        * floating-point quantization: specified by the 'low_fp_type' attribute, [f8E4M3FN | f8E5M2]

        Only one of these attributes should be provided.

        Scale and zp attributes provided for weights quantization
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        // Used for dyn quant on weights path
        RankedTensorOf<[F16, F32]>:$scale,
        RankedTensorOf<[I8, I4]>:$zp,

        OptionalAttr<IntAttr>:$levels,
        OptionalAttr<TypeAttr>:$low_fp_type,
        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasVerifier = 1;
}

//
// DynamicDequantizeOp
//

def IE_DynamicDequantizeOp :
        IE_LayerOp<"DynamicDequantize",
            [
                IE_EltwiseOp
            ]
        > {
    let summary = "InferenceEngine Dynamic Dequantize layer";

    let arguments = (ins
        RankedTensorOf<[quant_QuantizedType]>:$input,
        RankedTensorOf<[F16, F32]>:$scale,
        Optional<RankedTensorOf<[I8, I4]>>:$zp,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );

    let hasVerifier = 1;
}

//
// RMSOp
//

def IE_RMSOp :
        IE_LayerOp<
            "RMS"
        > {
    let summary = "InferenceEngine RMS layer";


    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$gamma,
        F64Attr:$epsilon
     );
   let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// DeformableConvolutionOp
//

def IE_DeformableConvolutionOp :
        IE_LayerOp<
            "DeformableConvolution"
        > {
    let summary = "InferenceEngine DeformableConvolution layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$offset,
        RankedTensorOf<[F16, F32]>:$kernel,
        Optional<RankedTensorOf<[F16, F32]>>:$mask,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,

        IntAttr:$group,
        IntAttr:$deformable_group,
        UnitAttr:$biliniar_interpolate_pad

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// InverseOp
//

def IE_InverseOp :
        IE_LayerOp<
            "Inverse"
        > {
    let summary = "InferenceEngine Inverse layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        UnitAttr:$adjoint
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// DynamicExpandOp
//

def IE_DynamicExpandOp :
        IE_LayerOp<
            "DynamicExpand"
        > {
    let summary = "DynamicExpand operation";

    let description = [{
        DynamicExpand operation isn't related to any InferenceEngine layer and is designed to take an input
        tensor with dynamic shapes and perform zero-padding to match the dimensions of an upper bound.
        This operation is specifically intended for converting a subgraph of a dynamic network into a static one
        Example:
        Input tensor<1x1x?x?xf16, {bounds = [1, 1, 5, 5]>
        Input dynamic shape: 1x1x3x3
        1, 1, 1
        1, 1, 1
        1, 1, 1
        In memory: 1, 1, 1, 1, 1, 1, 1, 1, 1
        Output tensor<1x1x5x5xf16>
        1, 1, 1, 0, 0
        1, 1, 1, 0, 0
        1, 1, 1, 0, 0
        0, 0, 0, 0, 0
        0, 0, 0, 0, 0
        In memory: 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    }];

    let arguments = (ins
        AnyRankedTensor:$input
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

//
// VariadicSplit
//

def IE_VariadicSplitOp :
        IE_LayerOp<
            "VariadicSplit"
        > {
    let summary = "VariadicSplit operation";

    let description = [{
        This operation is analogous to the OpenVINO VariadicSplit operation. It splits an input tensor along a given axis into multiple tensors according to split lengths.

        'input' is the input tensor to split. It can have an arbitrary fixed shape. Its element type must not be quantized.

        'axis' is the target axis to split along. It can be in the range [-rank('input'); rank('input')-1]. Negative values are counted from the back, with -1 referring to the last axis, -2 to the second last, and so on.

        'split_lengths' specifies the sizes of the output tensor along the target axis. If all entries are positive, they must sum to the size of the target axis in 'input'.
        At most one entry can be -1. If that's the case then the size of the target axis of the corresponding output tensor is deduced from the sum of remaining 'split_lengths' values.

        To demonstrate the role of 'axis' and 'split_lengths', the following ops are all equal:
        ```MLIR
            %variadic_split_0 = IE.VariadicSplit(%input) {axis = 3, split_lengths = [1, 2, 3]} : tensor<3x3x3x6xf32> -> tensor<3x3x3x1xf32>, tensor<3x3x3x2xf32>, tensor<3x3x3x3xf32>
            %variadic_split_1 = IE.VariadicSplit(%input) {axis = 3, split_lengths = [1, 2, -1]} : tensor<3x3x3x6xf32> -> tensor<3x3x3x1xf32>, tensor<3x3x3x2xf32>, tensor<3x3x3x3xf32>
            %variadic_split_2 = IE.VariadicSplit(%input) {axis = -1, split_lengths = [1, 2, -1]} : tensor<3x3x3x6xf32> -> tensor<3x3x3x1xf32>, tensor<3x3x3x2xf32>, tensor<3x3x3x3xf32>
        ```
    }];

    let arguments = (ins
        RankedTensorOf<[AnyInteger, AnyFloat]>:$input,
        IntAttr:$axis,
        I64ArrayAttr:$split_lengths
    );

    let results = (outs
        Variadic<RankedTensorOf<[AnyInteger, AnyFloat]>>:$output
    );

    let extraClassDeclaration = [{
        // returns a copy of 'split_lengths' where the -1 value is inferred
        mlir::SmallVector<int64_t> getInferredSplitLengths();

        // returns the actual axis index which counts from the back if it's negative
        int64_t getInferredAxis();
    }];

    let hasVerifier = 1;
}

//
// RoPEOp
//

def IE_RoPEOp :
        IE_LayerOp<
            "RoPE"
        > {
    let summary = "InferenceEngine RoPE layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$input_cos,
        RankedTensorOf<[F16, F32]>:$input_sin
     );
   let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

#endif
