//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS
#define VPUX_COMPILER_DIALECT_VPU_OPS

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/core/types.td"
include "vpux/compiler/dialect/core/constraints.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/dialect.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//
// Base classes
//

class VPU_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPU_Dialect,
            mnemonic,
            traits
        >;

class VPU_LayerOp<string mnemonic, list<Trait> traits = []> :
        VPU_Op<
            mnemonic,
            [
                Pure,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
                DeclareOpInterfaceMethods<VPU_LayerOpInterface>
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
// DPU.Workload
//

def VPU_DPUWorkloadOp :
        VPU_Op<
            "DPU.Workload",
            [
                ParentOneOf<[
                    "vpux::VPU::NCEConvolutionOp",
                    "vpux::VPU::NCEMatMulOp",
                    "vpux::VPU::NCEDepthConvolutionOp",
                    "vpux::VPU::NCEMaxPoolOp",
                    "vpux::VPU::NCEAveragePoolOp",
                    "vpux::VPU::NCEEltwiseOp",
                    "vpux::VPU::NCECompressConvolutionOp",
                    "vpux::VPU::NCEInterpolateOp",
                    "vpux::VPU::NCEPermuteOp",
                    "vpux::VPU::NCEReduceOp"
                ]>
            ]
        > {
    let summary = "Workload for a single DPU tile";

    let arguments = (ins
        I64ArrayAttr:$outOffsets,
        I64ArrayAttr:$outSizes,

        OptionalAttr<I64ArrayAttr>:$inOffsets,
        OptionalAttr<I64ArrayAttr>:$inSizes,

        VPU_PaddingAttr:$pad,
        VPU_MPEModeAttr:$mpe_mode,

        OptionalAttr<IntAttr>:$cluster_id
    );

    let builders = [
        OpBuilder<(ins
            "mlir::ArrayAttr":$outOffsets,
            "mlir::ArrayAttr":$outSizes,
            "vpux::VPU::PaddingAttr":$kernelFunction,
            "vpux::VPU::MPEMode":$mpe_mode
        )>,

        OpBuilder<(ins
            "mlir::ArrayAttr":$outOffsets,
            "mlir::ArrayAttr":$outSizes,
            "vpux::VPU::PaddingAttr":$kernelFunction,
            "vpux::VPU::MPEModeAttr":$mpe_mode,
            "mlir::IntegerAttr":$cluster_id
        )>,

        OpBuilder<(ins
            "mlir::ArrayAttr":$outOffsets,
            "mlir::ArrayAttr":$outSizes,
            "vpux::VPU::PaddingAttr":$kernelFunction,
            "vpux::VPU::MPEMode":$mpe_mode,
            "mlir::IntegerAttr":$cluster_id
        )>
    ];

    let assemblyFormat = [{
        ( `inOffsets` $inOffsets^ )? ( `inSizes` $inSizes^ )? `outOffsets` $outOffsets `outSizes` $outSizes $pad $mpe_mode attr-dict-with-keyword
    }];
}

//
// NCE.Convolution
//

def VPU_NCEConvolutionOp :
        VPU_LayerOp<
            "NCE.Convolution",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "isOperationSplitOverBatchCompatible",
                                                                     "doesLayerFitIntoCMX",
                                                                     "doesLayerChangeOutputAlignmentFitIntoCMX",
                                                                     "getDistributedTypeForOpOperand"]>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "NCE version of Convolution layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$filter,
        AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>:$weightsTable,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        VPU_PPEAttr:$ppe,
        OptionalAttr<VPU_MPEEngineAttr>:$mpe_engine,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<4>]>:$rawFilterShape,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<IntAttr>:$output_channels
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, F32, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(`
            $input `,`
            $filter `,`
            $weightsTable
        `)`
        attr-dict
        custom<OptionalTypes>(type($input), type($filter), type($weightsTable)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::ConvolutionOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);

        vpux::Shape inferAlignedFilterShape(vpux::NDTypeInterface input, vpux::NDTypeInterface output, vpux::NDTypeInterface filter);

        DimArr restrictedFusionAxes();

        static mlir::LogicalResult verifyKernel(IE::ConvolutionOp origOp, Logger log = Logger::global());
        static mlir::LogicalResult verifyKernel(IE::TransposedConvolutionOp origOp, Logger log = Logger::global());

        static mlir::LogicalResult verifyConvCMX(mlir::Location loc, mlir::ModuleOp module, vpux::NDTypeInterface inputType,
                                             vpux::NDTypeInterface filterType, vpux::NDTypeInterface outputType,
                                             mlir::ArrayAttr kernelStrides, Logger log = Logger::global());
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// NCE.MatMul
//

def VPU_NCEMatMulOp :
        VPU_LayerOp<
            "NCE.MatMul",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                SameVariadicOperandSize,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "doesLayerFitIntoCMX",
                                                                     "doesLayerChangeOutputAlignmentFitIntoCMX",
                                                                     "getDistributedTypeForOpOperand"]>
            ]
        > {
    let summary = "NCE version of MatMul executed on DPU";

    let description = [{
        NCE version of MatMul that will be executed on DPU.

        This operation supports 5D operands so that we may encode batch unrolling as part of the shape
        and defer this unrolling until later in the compilation pipeline when it's more beneficial.

        Instead of unrolling batch into a chain of discrete slice -> convolution -> concat operations in IE dialect,
        we instead encode this as part of the shape and preserve it until later on in multiclustering passes where we
        can use split-over-group (in the case we have more groups than clusters or else use another approach).

        We use this split-over-group strategy to solve the problem of having too many small workloads generated
        which introduce a lot of DMA overhead.

        With 5D operands, we have the following layouts (where G = Group):
        * NCHW -> GNCHW
        * NHWC -> GNHWC
        * OIYX -> GOIYX

        Before we would convert IE.MatMul like this with repeated chains for the number of batches we need to unroll:

                                        |
                                    ---/ \---------------------- ... (repeated for number of batches)
                                /             \            \
                                |              |            |
                            IE.Slice       IE.Slice        ...
                                |              |            |
                            IE.Reshape     IE.Reshape       ...
                                |              |            |
            IE.MatMul  =>  IE.Convolution IE.Convolution    ...
                                |              |            |
                            IE.Reshape     IE.Reshape       ...
                                |              |            |
                                \          /              /
                                    ---\ /---------------------- ... (repeated for number of batches)
                                        |
                                    Concat

        Now we keep a single chain of operations (which we later unroll in multiclustering):

                           VPU.AffineReshape
                                   |
                            VPU.PermuteCast
                                   |
            IE.MatMul  =>    VPU.NCE.MatMul
                                   |
                            VPU.PermuteCast
                                   |
                           VPU.AffineReshape

        See E#125047 for more information.
    }];

    let arguments = (ins
        AnyTypeOf<[5DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[5DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$weights,

        AnyTypeOf<[5DTensorOf<[SI32]>, VPU_DistributedTensor]>:$weightsTable,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        VPU_PPEAttr:$ppe,
        OptionalAttr<VPU_MPEEngineAttr>:$mpe_engine,
        ConfinedAttr<I64ArrayAttr, [ArrayCount<5>]>:$rawFilterShape,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[5DTensorOf<[F16, BF16, F32, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(`
            $input `,`
            $weights `,`
            $weightsTable
        `)`
        attr-dict
        custom<OptionalTypes>(type($input), type($weights), type($weightsTable)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        static bool isSupported(vpux::IE::MatMulOp origOp, vpux::LogCb logCb,
                                bool checkLayout = false, bool checkChannelAlignment = false);
        static bool isSupported(vpux::VPU::NCEMatMulOp origOp, vpux::LogCb logCb,
                                bool checkLayout = false, bool checkChannelAlignment = false);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output, Byte reservedMem);
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);

        static mlir::LogicalResult verifyKernel(IE::MatMulOp origOp);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [
        IE_TypeComparisonMode_ALLOW_MIXED_PRECISION,
        IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
        IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT,
        IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT
    ];
}

//
// NCE.DepthConvolution
//

def VPU_NCEDepthConvolutionOp :
        VPU_LayerOp<
            "NCE.DepthConvolution",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "doesLayerFitIntoCMX",
                                                                     "doesLayerChangeOutputAlignmentFitIntoCMX",
                                                                     "getDistributedTypeForOpOperand"]>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "NCE version of Depthwise Convolution layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$filter,
        AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>:$weightsTable,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        VPU_PPEAttr:$ppe,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<4>]>:$rawFilterShape,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<IntAttr>:$output_channels
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, F32, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(`
            $input `,`
            $filter `,`
            $weightsTable
        `)`
        attr-dict
        custom<OptionalTypes>(type($input), type($filter), type($weightsTable)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::GroupConvolutionOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);

        vpux::Shape inferAlignedFilterShape(vpux::NDTypeInterface output, vpux::NDTypeInterface filter);

        static mlir::LogicalResult verifyKernel(IE::GroupConvolutionOp origOp, Logger log = Logger::global());

        static mlir::LogicalResult verifyGroupConvCMX(mlir::Location loc, mlir::ModuleOp module,
                                                  vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
                                                  vpux::NDTypeInterface outputType, mlir::ArrayAttr kernelStrides,
                                                  Logger log = Logger::global());
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// NCE.CompressConvolution
//

def VPU_NCECompressConvolutionOp :
        VPU_LayerOp<
            "NCE.CompressConvolution",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "isOperationSplitOverBatchCompatible",
                                                                     "doesLayerFitIntoCMX",
                                                                     "doesLayerChangeOutputAlignmentFitIntoCMX",
                                                                     "getDistributedTypeForOpOperand"]>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>
            ]
        > {
    let summary = "NCE version of Compressed Convolution layer";

    let description = [{
        This operation must have 4 or less input channels,
        instead of the usual multiple of 16 as for a normal Convolution op.
    }];

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$filter,
        AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>:$weightsTable,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        VPU_PPEAttr:$ppe,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<4>]>:$rawFilterShape,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        IntAttr:$cm_sp_pattern,
        OptionalAttr<IntAttr>:$output_channels
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, F32, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input `,` $filter `,` $weightsTable `)`
        attr-dict
        custom<OptionalTypes>(type($input), type($filter), type($weightsTable)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::ConvolutionOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// NCE.MaxPool
//

def VPU_NCEMaxPoolOp :
        VPU_LayerOp<
            "NCE.MaxPool",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "isOperationSplitOverBatchCompatible",
                                                                     "doesLayerFitIntoCMX",
                                                                     "doesLayerChangeOutputAlignmentFitIntoCMX",
                                                                     "getDistributedTypeForOpOperand"]>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "NCE version of MaxPool layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weightsTable,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$kernel_size,
        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        VPU_PPEAttr:$ppe,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<IntAttr>:$output_channels
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, F32, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input
            (`,` $weightsTable^ custom<OptionalTypes>(type($weightsTable)) ``)?
        `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::MaxPoolOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);

        static mlir::LogicalResult verifyKernel(IE::MaxPoolOp origOp, Logger log = Logger::global());
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// NCE.AveragePool
//

def VPU_NCEAveragePoolOp :
        VPU_LayerOp<
            "NCE.AveragePool",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "isOperationSplitOverBatchCompatible",
                                                                     "doesLayerChangeOutputAlignmentFitIntoCMX",
                                                                     "getDistributedTypeForOpOperand"]>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "NCE version of AveragePool layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$kernel_size,
        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        VPU_PPEAttr:$ppe,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<IntAttr>:$output_channels
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, F32, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output);

        static bool isSupported(vpux::IE::AvgPoolOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);

        static mlir::LogicalResult verifyKernel(IE::AvgPoolOp origOp, Logger log = Logger::global());
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// NCE.Eltwise
//

def VPU_NCEEltwiseOp :
        VPU_LayerOp<
            "NCE.Eltwise",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "doesLayerFitIntoCMX",
                                                                     "getDistributedTypeForOpOperand"]>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "NCE version of Eltwise layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input2,

        VPU_EltwiseTypeAttr:$op_type,

        VPU_PPEAttr:$ppe,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<BoolAttr>:$is_inplace,
        OptionalAttr<IntAttr>:$output_channels
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, F32, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input1 `,` $input2 `)`
        attr-dict
        custom<OptionalTypes>(type($input1), type($input2)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2, vpux::NDTypeInterface output);

        bool fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2, Byte reservedMem);

        static bool isSupported(mlir::Operation* op, bool allowDifferentScales, bool allowDifferentZp,
                                vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);

        static mlir::LogicalResult verifyKernel(IE::AddOp origOp, Logger log = Logger::global());
        static mlir::LogicalResult verifyKernel(IE::MultiplyOp origOp, Logger log = Logger::global());
        static mlir::LogicalResult verifyKernel(IE::SubtractOp origOp, Logger log = Logger::global());

        static mlir::LogicalResult verifyEltwiseCMX(mlir::Location loc, mlir::ModuleOp module, bool isInplace,
                                                vpux::NDTypeInterface firstInputType,
                                                vpux::NDTypeInterface secondInputType, vpux::NDTypeInterface outputType,
                                                Logger log = Logger::global());
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// NCE.Reduce
//

def VPU_NCEReduceOp :
        VPU_LayerOp<
            "NCE.Reduce",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock
            ]
        > {
    let summary = "NCE version of Reduce layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>]>:$input,
        I64ArrayAttr:$axes,

        VPU_PPEAttr:$ppe,
        VPU_ReduceTypeAttr:$op_type
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, F32, quant_QuantizedType]>]>:$output
    );

    let extraClassDeclaration = [{
        static bool isSupported(mlir::Operation* op,
                                vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);
        static mlir::LogicalResult verifyKernel(mlir::Operation* origOp, Logger log = Logger::global());

    }] # baseExtraClassDeclaration;

    let assemblyFormat = [{
       `(` $input`)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let regions = (region
        AnyRegion:$workloads
    );

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION];
}

//
// NCE.Permute
//

def VPU_NCEPermuteOp :
        VPU_LayerOp<
            "NCE.Permute",
        [
            NoRegionArguments,
            NoTerminator,
            SingleBlock,
            DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
            DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                 "isOperationSplitOverWidthCompatible",
                                                                 "isOperationSplitOverKernelCompatible",
                                                                 "getDistributedTypeForOpOperand"]>,
            DeclareOpInterfaceMethods<VPU_SparseOpInterface>
        ]
        > {
    let summary = "More abstract version of combined NCE Permute and Quantization layers";

    let description = [{
        Used to perform a datatype conversion, relayout of data and shape expansion,
        all using a single NCE HW op.

        * expandedChannels - target size of output channels after expansion, usual values are 4 and 16
        * dstElemType - output tensor datatype
        * dstOrder - output tensor layout, NCHW input to NHWC output relayout is supported
    }];

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,

        IntAttr:$expandedChannels,
        TypeAttr:$dstElemType,
        AffineMapAttr:$dstOrder,
        VPU_PPEAttr:$ppe,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem);
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output);
        static bool isSupported(vpux::IE::PermuteQuantizeOp origOp, vpux::LogCb logCb, bool checkLayout = true, bool checkAlignment = true);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// NCE.ClusterTiling
//

def VPU_NCEClusterTilingOp :
        VPU_Op<
            "NCE.ClusterTiling",
            [
                Pure,
                IsolatedFromAbove,
                DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getEntrySuccessorOperands",
                                                                    "areTypesCompatible"]>,
                SingleBlockImplicitTerminator<"YieldOp">
            ]
        > {
    let summary = "Operation that encapsulates details of tiling operation between clusters";

    let arguments = (ins
        Variadic<AnyTypeOf<[4DTensorOf<[AnyType]>, 5DTensorOf<[AnyType]>, VPU_DistributedTensor, VPU_SparseTensor]>>:$operands
    );

    let results = (outs
        Variadic<AnyTypeOf<[4DTensorOf<[AnyType]>, 5DTensorOf<[AnyType]>, VPU_DistributedTensor, VPU_SparseTensor]>>:$results
    );

    let regions = (region SizedRegion<1>:$body);

    let hasVerifier = 1;

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::ValueRange":$operands,
            "llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>":$bodyBuilder)>,
    ];

    let extraClassDeclaration = [{
        using BodyBuilderFn =
            llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>;

            mlir::Operation* getInnerTaskOp();
            template <typename T>
            T getInnerTaskOpOfType();

        void print(::mlir::OpAsmPrinter& p);
        static ::mlir::ParseResult parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result);
    }];

    let hasCanonicalizer = 1;
}

//
// NCE.Interpolate
//

def VPU_NCEInterpolateOp :
        VPU_LayerOp<
            "NCE.Interpolate",
            [
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                SameVariadicOperandSize,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "doesLayerFitIntoCMX",
                                                                     "doesLayerChangeOutputAlignmentFitIntoCMX",
                                                                     "getDistributedTypeForOpOperand"]>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>
            ]
        > {
    let summary = "NCE version of Interpolate layer";

    let arguments = (ins
        AnyTypeOf<[VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>>:$weights,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weightsTable,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,

        VPU_PPEAttr:$ppe,
        ConfinedAttr<I64ArrayAttr, [ArrayCount<4>]>:$rawFilterShape,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<VPU_NCEInterpolateModeAttr>:$mode
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input
            (`,` $weights^ `` custom<OptionalTypes>(type($weights)))?
            (`,` $weightsTable^ `` custom<OptionalTypes>(type($weightsTable)))?
        `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        static bool isSupported(vpux::IE::InterpolateOp origOp, vpux::LogCb logCb,
                                bool checkLayout = false, bool checkChannelAlignment = false, bool checkBatch = false);
        static bool isSupported(vpux::VPU::InterpolateOp origOp, vpux::LogCb logCb,
                                bool checkLayout = false, bool checkChannelAlignment = false, bool checkBatch = false);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output, Byte reservedMem);
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// VerticalFusion
//

def VPU_VerticalFusionOp :
        VPU_Op<
            "VerticalFusion",
            [
                Pure,
                IsolatedFromAbove,
                DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getEntrySuccessorOperands",
                                                                    "areTypesCompatible"]>,
                SingleBlockImplicitTerminator<"YieldOp">
            ]
        > {
    let summary = "Operation that encapsulates details of VF subgraph";

    let arguments = (ins
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor]>>:$operands,
        I64ArrayAttr:$tilingStrategy,
        OptionalAttr<VPU_VFScenarioAttr>:$scenario
    );

    let results = (outs
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor]>>:$results
    );

    let regions = (region SizedRegion<1>:$ops);

    let hasVerifier = 1;

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::ValueRange":$operands,
            "llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>":$bodyBuilder,
            "mlir::ArrayAttr":$tilingInfo)>,
    ];

    let extraClassDeclaration = [{
        using BodyBuilderFn =
            llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>;

        mlir::Operation* getFirstInnerTaskOp();

        void print(::mlir::OpAsmPrinter& p);
        static ::mlir::ParseResult parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result);
    }];
}

//
// YieldOp
//

def VPU_YieldOp :
        VPU_Op<
            "Yield",
            [
                HasParent<"NCEClusterTilingOp, VerticalFusionOp">,
                DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
                Pure,
                Terminator
            ]
        > {
    let summary = "Terminator for wrapping operation";

    let arguments = (ins
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>>:$operands
    );

    let assemblyFormat = [{
        $operands
        custom<OptionalTypes>(type($operands)) ``
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// DistributedCastOp
//

def VPU_DistributedCastOp :
        VPU_Op<
            "DistributedCast",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Operation that casts one DistributedTensor type to another.";

    let description = [{
        Used to cast one DistributedTensor type to another and help with NNCMX retention
        of data.

        Currently following distribution mode pairs are compatible:

        DUPLICATED|SEGMENTED -> DUPLICATED ## needed for K cluster tiling
    }];

    let arguments = (ins
        AnyTypeOf<[VPU_DistributedTensor, VPU_SparseTensor]>:$input
    );

    let results = (outs
        AnyTypeOf<[VPU_DistributedTensor, VPU_SparseTensor]>:$output
    );

    let assemblyFormat = [{
        `(` $input `:` qualified(type($input)) `)`
        attr-dict
        `->` qualified(type($output))
    }];

    let hasFolder = 1;

    let hasVerifier = 1;
}

//
// GroupSparseTensor
//

def VPU_GroupSparseTensorOp :
        VPU_Op<
            "GroupSparseTensor",
            [
                Pure,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
                AttrSizedOperandSegments,
                VPU_GroupedViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_TilingViewLikeOpInterface>
            ]
        > {
    let summary = "Groups sparse data and metadata into a single value";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>:$data,
        Optional<AnyTypeOf<[I1Tensor, VPU_DistributedTensor]>>:$sparsityMap,
        Optional<AnyTypeOf<[I32Tensor, VPU_DistributedTensor]>>:$storageElementTable,

        UnitAttr:$is_weights,
        OptionalAttr<VPU_SparsityCompressionAttr>:$sparsity_compression,

        OptionalAttr<VPU_SEAttr>:$seAttr
    );

    let results = (outs
        AnyTypeOf<[VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$data,
                CArg<"bool", "{}">:$is_weights, CArg<"VPU::SparsityCompressionAttr", "{}">:$sparsity_compression)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap,
                CArg<"bool", "{}">:$is_weights, CArg<"VPU::SparsityCompressionAttr", "{}">:$sparsity_compression)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap, "mlir::Value":$storageElementTable,
                CArg<"bool", "{}">:$is_weights, CArg<"VPU::SparsityCompressionAttr", "{}">:$sparsity_compression)
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

    let hasCanonicalizer = 1;
}


//
// UngroupSparseTensor
//

def VPU_UngroupSparseTensorOp :
        VPU_Op<
            "UngroupSparseTensor",
            [
                Pure,
                AttrSizedResultSegments,
                DeclareOpInterfaceMethods<MultiViewOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
            ]
        > {
    let summary = "Ungroups sparse data and metadata into multiple values";

    let arguments = (ins
        VPU_SparseTensor:$input
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$data,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$sparsityMap,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$storageElementTable
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        `` custom<OptionalTypes>(type($input))
        `->` type(results)
    }];
}

//
// SliceOp
//

def VPU_SliceOp :
        VPU_LayerOp<
            "Slice",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Extract single slice from ranked tensor or distributed tensor";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor]>:$source,
        I64ArrayAttr:$static_offsets,
        I64ArrayAttr:$static_sizes
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor]>:$result
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
    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// ConditionalCopyOp
//

def VPU_ConditionalCopyOp :
        VPU_LayerOp<
            "ConditionalCopyOp"
        > {
    let summary = "Conditional copy VPU layer";

    let arguments = (ins
        1DTensorOf<[Bool8, SI8]>:$cond,
        AnyRankedTensor:$input1,
        AnyRankedTensor:$input2
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
}

//
// LoopSelectOp
//

def VPU_LoopSelectOp :
        VPU_LayerOp<
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
// ConcatOp
//

def VPU_ConcatOp :
        VPU_LayerOp<
            "Concat",
            [
                VPU_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "doesLayerFitIntoCMX"]>
            ]
        > {
    let summary = "VPU Concat layer";

    let arguments = (ins
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor, VPU_DistributedTensor]>>:$inputs,

        OptionalAttr<IE_ConcatAttr>:$per_axis,
        OptionalAttr<I64ArrayOfArraysAttr>:$static_offsets,
        OptionalAttr<I64ArrayOfArraysAttr>:$strides,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "vpux::IE::ConcatAttr":$per_axis)
        >,
        OpBuilder<
            (ins "mlir::ValueRange":$inputs, "vpux::IE::ConcatAttr":$per_axis, "mlir::ArrayAttr":$static_offsets)
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
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::IE::ConcatAttr":$per_axis, "mlir::ArrayAttr":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::ArrayRef<vpux::Shape>":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "vpux::ArrayRef<vpux::ShapeRef>":$static_offsets)
        >,
        OpBuilder<
            (ins "mlir::Type":$outType, "mlir::ValueRange":$inputs, "mlir::ArrayAttr":$static_offsets, "mlir::ArrayAttr":$strides)
        >
    ];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        ::mlir::LogicalResult customVerify();
        bool fitIntoCMX(vpux::NDTypeInterface output, Byte reservedMem);
        bool fitIntoCMX(vpux::NDTypeInterface output);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// RollOp
//

def VPU_RollOp :
        VPU_LayerOp<
            "Roll"
        > {
    let summary = "Roll VPU layer";

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
// Tanh
//

def VPU_TanhOp :
        VPU_LayerOp<
            "Tanh",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "Tanh VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Sin
//

def VPU_SinOp :
        VPU_LayerOp<
            "Sin",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Sin VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;
}

//
// Cos
//

def VPU_CosOp :
        VPU_LayerOp<
            "Cos",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Cos VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;
}

//
// Tan
//

def VPU_TanOp :
        VPU_LayerOp<
            "Tan",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Tan VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Sqrt
//

def VPU_SqrtOp :
        VPU_LayerOp<
            "Sqrt",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Sqrt VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Sinh
//

def VPU_SinhOp :
        VPU_LayerOp<
            "Sinh",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Sinh VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Cosh
//

def VPU_CoshOp :
        VPU_LayerOp<
            "Cosh",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Cosh VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Asinh
//

def VPU_AsinhOp :
        VPU_LayerOp<
            "Asinh",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Asinh VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Acosh
//

def VPU_AcoshOp :
        VPU_LayerOp<
            "Acosh",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Acosh VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Abs
//

def VPU_AbsOp :
        VPU_LayerOp<
            "Abs",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Abs VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Atan
//

def VPU_AtanOp :
        VPU_LayerOp<
            "Atan",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Atan VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Asin
//

def VPU_AsinOp :
        VPU_LayerOp<
            "Asin",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Asin VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Acos
//

def VPU_AcosOp :
        VPU_LayerOp<
            "Acos",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Acos VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Atanh
//

def VPU_AtanhOp :
        VPU_LayerOp<
            "Atanh",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Atanh VPU layer";

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

def VPU_HSigmoidOp :
        VPU_LayerOp<
            "HSigmoid",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "HSigmoid VPU layer";

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

def VPU_GridSampleOp :
        VPU_LayerOp<
            "GridSample",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "GridSample VPU layer";

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
// Log
//

def VPU_LogOp :
        VPU_LayerOp<
            "Log",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Log VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Gelu
//

def VPU_GeluOp :
        VPU_LayerOp<
            "Gelu",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Gelu VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Exp
//

def VPU_ExpOp :
        VPU_LayerOp<
            "Exp",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Exp VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;
}

//
// HSwish
//

def VPU_HSwishOp :
        VPU_LayerOp<
            "HSwish",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "HSwish VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Floor
//

def VPU_FloorOp :
        VPU_LayerOp<
            "Floor",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "Floor VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Round
//

def VPU_RoundOp :
        VPU_LayerOp<
            "Round",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "Round VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        IE_RoundModeAttr:$mode,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "vpux::IE::RoundModeAttr":$mode
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Mish
//

def VPU_MishOp :
        VPU_LayerOp<
            "Mish",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "Mish VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input
        )>
    ];
}

//
// Erf
//

def VPU_ErfOp :
        VPU_LayerOp<
            "Erf",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Erf VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Clamp
//

def VPU_ClampOp :
        VPU_LayerOp<
            "Clamp",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Clamp VPU layer";

    let arguments = (ins
       AnyTypeOf<[RankedTensorOf<[F16, F32, quant_QuantizedType]>, VPU_DistributedTensor]>:$input,

        F64Attr:$min,
        F64Attr:$max,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, quant_QuantizedType]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::FloatAttr":$min,
            "::mlir::FloatAttr":$max
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Elu
//

def VPU_EluOp :
        VPU_LayerOp<
            "Elu",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Elu VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$x
    );


    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Sigmoid
//

def VPU_SigmoidOp :
        VPU_LayerOp<
            "Sigmoid",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "Sigmoid VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// HardSigmoidOp
//

def VPU_HardSigmoidOp :
        VPU_LayerOp<
            "HardSigmoid",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "HardSigmoid VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        F64Attr:$alpha_value,
        F64Attr:$beta_value,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::FloatAttr":$alpha,
            "::mlir::FloatAttr":$beta
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// EmbeddingBagOffsetsSumOp
//

def VPU_EmbeddingBagOffsetsSumOp :
        VPU_LayerOp<
            "EmbeddingBagOffsetsSum",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "InferenceEngine EmbeddingBagOffsetsSum layer";

    let arguments = (ins
        AnyRankedTensor:$emb_table,
        Optional<1DTensorOf<[SI32, SI64]>>:$indices,
        Optional<1DTensorOf<[SI32, SI64]>>:$offsets,
        Optional<1DTensorOf<[AnyInteger, AnyFloat]>>:$per_sample_weights,

        OptionalAttr<I64ArrayAttr>:$indices_value,
        OptionalAttr<I64ArrayAttr>:$offsets_value,
        IntAttr:$default_index_value,
        OptionalAttr<F64ArrayAttr>:$per_sample_weights_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// EmbeddingSegmentsSumOp
//

def VPU_EmbeddingSegmentsSumOp :
        VPU_LayerOp<
            "EmbeddingSegmentsSum",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "EmbeddingSegmentsSum VPU layer";

    let arguments = (ins
        AnyRankedTensor:$emb_table,
        Optional<1DTensorOf<[SI64, SI32]>>:$indices,
        Optional<1DTensorOf<[SI64, SI32]>>:$segment_ids,
        Optional<1DTensorOf<[AnyInteger, AnyFloat]>>:$per_sample_weights,

        OptionalAttr<I64ArrayAttr>:$indices_value,
        OptionalAttr<I64ArrayAttr>:$segment_ids_value,
        IntAttr:$num_segments_value,
        IntAttr:$default_index_value,
        OptionalAttr<F64ArrayAttr>:$per_sample_weights_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// EmbeddingBagPackedSumOp
//

def VPU_EmbeddingBagPackedSumOp :
        VPU_LayerOp<
            "EmbeddingBagPackedSum"
        > {
    let summary = "EmbeddingBagPackedSum VPU layer";

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
// SeluOp
//

def VPU_SeluOp :
        VPU_LayerOp<
            "Selu",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Selu VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        F64Attr:$alpha_value,
        F64Attr:$lambda_value
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// NormalizeL2
//

def VPU_NormalizeL2Op :
        VPU_LayerOp<
            "NormalizeL2",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "NormalizeL2 VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$data,
        ArrayAttr:$axes_value,

        F64Attr:$eps,
        IE_EpsModeAttr:$eps_mode,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let hasVerifier = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// NormalizeIE
//

def VPU_NormalizeIEOp :
        VPU_LayerOp<
            "NormalizeIE"
        > {
    let summary = "NormalizeIE VPU layer";

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
// CumSum
//

def VPU_CumSumOp :
        VPU_LayerOp<
            "CumSum"
        > {
    let summary = "CumSum VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        OptionalAttr<IntAttr>:$axis_value,
        UnitAttr:$exclusive,
        UnitAttr:$reverse
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// Eye
//

def VPU_EyeOp :
        VPU_LayerOp<
            "Eye"
        > {
    let summary = "Eye VPU layer";

    let arguments = (ins
        1DTensorOf<[SI32, SI64]>:$diagonal_index,

        IntAttr:$num_rows_value,
        IntAttr:$num_columns_value,
        I64ArrayAttr:$batch_shape_value,

        TypeAttr:$outputType
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// Ceiling
//

def VPU_CeilingOp :
        VPU_LayerOp<
            "Ceiling",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "Ceiling VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// SoftPlus
//

def VPU_SoftPlusOp :
        VPU_LayerOp<
            "SoftPlus",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "SoftPlus VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Convert
//

def VPU_ConvertOp :
        VPU_LayerOp<
            "Convert",
            [
                DeclareOpInterfaceMethods<CastOpInterface>,
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "Convert VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        TypeAttr:$dstElemType,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::TypeAttr":$dstElemType
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// SoftMax
//

def VPU_SoftMaxOp :
        VPU_LayerOp<
            "SoftMax",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "SoftMax VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        IntAttr:$axisInd,
        OptionalAttr<IntAttr>:$padSize,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);

        bool isVFSupported();
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::IntegerAttr":$axisInd,
            "::mlir::IntegerAttr":$padSize
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// LogSoftmax
//

def VPU_LogSoftmaxOp :
        VPU_LayerOp<
            "LogSoftmax",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "LogSoftmax VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        IntAttr:$axisInd,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::IntegerAttr":$axisInd
        )>
    ];
}

//
// PerAxisTile
//

def VPU_PerAxisTileOp :
        VPU_LayerOp<
            "PerAxisTile"
        > {
    let summary = "Per axis Tile VPU layer";

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
// ReLU
//

def VPU_ReLUOp :
        VPU_LayerOp<
            "ReLU",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "ReLU VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// LogicalNot
//

def VPU_LogicalNotOp :
        VPU_LayerOp<
            "LogicalNot",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Logical Not VPU layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, SI32]>:$input1
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, SI32]>:$output
    );
}

//
// Convolution
//

def VPU_ConvolutionOp :
        VPU_LayerOp<
            "Convolution",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "Convolution VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$filter,
        Optional<RankedTensorOf<[F16, F32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,

        OptionalAttr<IE_PostOpAttr>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Gather
//

def VPU_GatherOp :
        VPU_LayerOp<
            "Gather",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "Gather VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[AnyInteger]>, VPU_DistributedTensor]>:$indices,
        Optional<AnyRankedTensor>:$axis,
        OptionalAttr<IntAttr>:$axis_value,
        IntAttr:$batch_dims,

        OptionalAttr<IntAttr>:$indices_rank,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$indices,
            "::mlir::Value":$axis,
            "::mlir::IntegerAttr":$axis_value,
            "::mlir::IntegerAttr":$batch_dims,
            "::mlir::IntegerAttr":$indices_rank
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// GatherDMA
//

def VPU_GatherDMAOp :
        VPU_LayerOp<
            "GatherDMA",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "GatherDMA VPU layer which will be lowered to DMA, used for GatherOps which can be lowered to DMA";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        Optional<AnyRankedTensor>:$axis,
        OptionalAttr<IntAttr>:$axis_value,
        IntAttr:$batch_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// GatherNDOp
//

def VPU_GatherNDOp :
        VPU_LayerOp<
            "GatherND",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "GatherND VPU layer";

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
// GatherElements
//

def VPU_GatherElementsOp :
        VPU_LayerOp<
            "GatherElements",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "GatherElements VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[AnyInteger]>, VPU_DistributedTensor]>:$indices,
        IntAttr:$axis,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$indices,
            "::mlir::IntegerAttr":$axis
        )>
    ];
}

//
// GatherTree
//

def VPU_GatherTreeOp :
        VPU_LayerOp<
              "GatherTree"
        > {
    let summary = "GatherTree VPU layer";

    let arguments = (ins
        AnyRankedTensor:$stepIds,
        AnyRankedTensor:$parentIds,
        AnyRankedTensor:$maxSeqLen,
        AnyRankedTensor:$endToken
    );

    let results = (outs
        AnyRankedTensor:$finalIds
    );
}

//
// ScatterNDUpdate
//

def VPU_ScatterNDUpdateOp :
        VPU_LayerOp<
            "ScatterNDUpdate"
        > {
    let summary = "ScatterNDUpdate VPU layer";

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
// ScatterUpdate
//

def VPU_ScatterUpdateOp :
        VPU_LayerOp<
            "ScatterUpdate"
        > {
    let summary = "ScatterUpdate VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        AnyRankedTensor:$updates,
        OptionalAttr<IntAttr>:$axis_value

    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ScatterElementsUpdate
//

def VPU_ScatterElementsUpdateOp :
        VPU_LayerOp<
            "ScatterElementsUpdate"
        > {
    let summary = "ScatterElementsUpdate VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[AnyInteger]>:$indices,
        AnyRankedTensor:$updates,
        IntAttr:$axis,
        IE_ScatterElementsUpdateReductionTypeAttr:$reduction,
        BoolAttr:$use_init_val
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// Broadcast
//

def VPU_BroadcastOp :
        VPU_LayerOp<
            "Broadcast"
        > {
    let summary = "Broadcast VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        1DTensorOf<[AnyInteger]>:$target_shape,
        Optional<1DTensorOf<[AnyInteger]>>:$axes_mapping,

        OptionalAttr<IE_BroadcastTypeAttr>:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// BucketizeOp
//

def VPU_BucketizeOp :
        VPU_LayerOp<
            "Bucketize"
        > {
    let summary = "Bucketize VPU layer";

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
// FakeQuantize
//

def VPU_FakeQuantizeOp :
        VPU_LayerOp<
            "FakeQuantize",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "FakeQuantize VPU layer";

    let description = [{
        The operation works in two modes:
         * integral quantization: specified by the 'levels' attribute
         * floating-point quantization: specified by the 'low_fp_type' attribute, [f8E4M3FN | f8E5M2]

        Only one of these attributes should be provided.
    }];

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input_low,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input_high,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output_low,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output_high,

        OptionalAttr<IntAttr>:$levels,
        OptionalAttr<TypeAttr>:$low_fp_type,
        IE_AutoBroadcastTypeAttr:$auto_broadcast,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$input_low,
            "::mlir::Value":$input_high,
            "::mlir::Value":$output_low,
            "::mlir::Value":$output_high,
            "::mlir::IntegerAttr":$levels,
            "::mlir::TypeAttr":$low_fp_type,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let hasVerifier = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Proposal
//

def VPU_ProposalOp :
        VPU_LayerOp<
            "Proposal"
        > {
    let summary = "Proposal VPU layer";

    let description = [{
        Proposal operation filters bounding boxes and outputs only those with the highest prediction confidence.
        The auxiliary buffer has the role of storing the intermediate results obtained inside the operation,
        then sorting them. Depending on some criteria, it recalculates the results and extracts the output from them.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$class_probs,
        RankedTensorOf<[F16, F32]>:$bbox_deltas,
        RankedTensorOf<[F16, F32]>:$image_shape,
        Optional<1DTensorOf<[UI8]>>:$auxiliary,

        IE_ProposalAttr:$proposal_attrs
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output,
        RankedTensorOf<[F16, F32]>:$probs
    );

}

//
// Interpolate
//

def VPU_InterpolateOp :
        VPU_LayerOp<
            "Interpolate",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "Interpolate VPU layer";

    let description = [{
        The `coordinates` contain byte offsets for the current `input` tensor.
        The `lambdas` contain two interleaved values for each coordinate.
    }];

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[UI8, F16, F32, quant_QuantizedType]>, VPU_DistributedTensor]>:$input,
        Optional<AnyTypeOf<[RankedTensorOf<[AnyInteger]>, VPU_DistributedTensor]>>:$sizes,
        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>>:$scales,
        Optional<AnyTypeOf<[RankedTensorOf<[AnyInteger]>, VPU_DistributedTensor]>>:$axes,
        Optional<AnyTypeOf<[RankedTensorOf<[SI32]>, VPU_DistributedTensor]>>:$coordinates,
        Optional<AnyTypeOf<[RankedTensorOf<[F16]>, VPU_DistributedTensor]>>:$lambdas,

        OptionalAttr<I64ArrayAttr>:$sizes_attr,
        OptionalAttr<F64ArrayAttr>:$scales_attr,
        OptionalAttr<I64ArrayAttr>:$axes_attr,
        OptionalAttr<F64ArrayAttr>:$tile_offset_attr,
        OptionalAttr<I64ArrayAttr>:$initial_input_dims_attr,
        OptionalAttr<I64ArrayAttr>:$initial_output_dims_attr,
        OptionalAttr<I64ArrayAttr>:$initial_input_offset_attr,
        OptionalAttr<I64ArrayAttr>:$initial_output_offset_attr,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,

        IE_InterpolateAttr:$attr,
        OptionalAttr<IntAttr>:$output_channels
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[UI8, F16, F32, quant_QuantizedType]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$sizes,
            "::mlir::Value":$scales,
            "::mlir::Value":$axes,
            "::mlir::Value":$coordinates,
            "::mlir::Value":$lambdas,
            "::mlir::ArrayAttr":$sizes_attr,
            "::mlir::ArrayAttr":$scales_attr,
            "::mlir::ArrayAttr":$axes_attr,
            "::mlir::ArrayAttr":$tile_offset_attr,
            "::mlir::ArrayAttr":$initial_input_dims_attr,
            "::mlir::ArrayAttr":$initial_output_dims_attr,
            "vpux::IE::InterpolateAttr":$attr,
            "::mlir::IntegerAttr":$output_channels
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// TopK
//

def VPU_TopKOp :
        VPU_LayerOp<
            "TopK",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "TopK VPU layer";

    let description = [{
        * lineBuffer - it is an auxiliary buffer, which has the role of storing
        some intermediate results in the software kernel of the operation.
    }];

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$k,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$lineBuffer,
        OptionalAttr<IntAttr>:$k_value,

        IntAttr:$axis,
        IE_TopKModeAttr:$mode,
        IE_TopKSortTypeAttr:$sort,
        TypeAttr:$element_type,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output_values,
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$target_shape
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
        OutputTiling getOutputTiling(const vpux::TileInfo& outputTile, vpux::Logger log);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// AdaptiveAvgPoolOp
//

def VPU_AdaptiveAvgPoolOp :
        VPU_LayerOp<
            "AdaptiveAvgPool"
        > {
    let summary = "AdaptiveAvgPool VPU layer";

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

def VPU_AdaptiveMaxPoolOp :
        VPU_LayerOp<
            "AdaptiveMaxPool"
        > {
    let summary = "AdaptiveMaxPool VPU layer";

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
// RegionYolo
//

def VPU_RegionYoloOp :
        VPU_LayerOp<
            "RegionYolo"
        > {
    let summary = "RegionYolo VPU layer";

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
// ReorgYolo
//

def VPU_ReorgYoloOp :
        VPU_LayerOp<
            "ReorgYolo"
        > {
    let summary = "ReorgYolo VPU layer";

    let arguments = (ins
        4DTensorOf<[AnyInteger, AnyFloat]>:$input,

        IntAttr:$stride
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// DetectionOutput
//

def VPU_DetectionOutputOp :
        VPU_LayerOp<
            "DetectionOutput",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "DetectionOutput VPU layer";

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
// DetectionOutputNormalize
//

def VPU_DetectionOutputNormalizeOp:
        VPU_LayerOp<
            "DetectionOutputNormalize"
        > {
    let summary = "DetectionOutputNormalize VPU layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$prior_boxes,

        IntAttr:$input_width,
        IntAttr:$input_height
    );

    let results = (outs
        4DTensorOf<[AnyFloat]>:$out_prior_boxes
    );

    let hasVerifier = 1;
}

//
// DetectionOutputDecodeBoxes
//

def VPU_DetectionOutputDecodeBoxesOp:
        VPU_LayerOp<
            "DetectionOutputDecodeBoxes",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "DetectionOutputDecodeBoxes VPU layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$box_logits,
        4DTensorOf<[AnyFloat]>:$prior_boxes,

        IE_DetectionOutputCodeTypeAttr:$code_type,
        BoolAttr:$clip_before_nms
    );

    let results = (outs
        4DTensorOf<[AnyFloat]>:$out_decoded_boxes
    );
}

//
// DetectionOutputSort
//

def VPU_DetectionOutputSortOp:
        VPU_LayerOp<
            "DetectionOutputSort",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible"]>,
            ]
        > {
    let summary = "DetectionOutputSort VPU layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[AnyFloat]>, VPU_DistributedTensor]>:$confidence,
        AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>:$indicesBuffer,
        AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>:$sortingBuffer,

        F64Attr:$confidence_threshold,
        IntAttr:$top_k,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[AnyFloat]>, VPU_DistributedTensor]>:$out_confidence,
        AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>:$out_indices,
        AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>:$out_sizes
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$confidence,
            "::mlir::FloatAttr":$confidence_threshold,
            "::mlir::IntegerAttr":$top_k
        )>
    ];

    let extraClassDeclaration = [{
        OutputTiling getOutputTiling(const vpux::TileInfo& outputTile, vpux::Logger log);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// DetectionOutputNmsCaffe
//

def VPU_DetectionOutputNmsCaffeOp:
        VPU_LayerOp<
            "DetectionOutputNmsCaffe",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "DetectionOutputNmsCaffe VPU layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$confidence,
        4DTensorOf<[AnyFloat]>:$boxes,
        4DTensorOf<[SI32]>:$indices,
        4DTensorOf<[SI32]>:$sizes,

        IntAttr:$top_k,
        F64Attr:$nms_threshold,
        IntAttr:$background_id
    );

    let results = (outs
        4DTensorOf<[AnyFloat]>:$out_confidence,
        4DTensorOf<[AnyFloat]>:$out_boxes,
        4DTensorOf<[SI32]>:$out_sizes
    );

    let extraClassDeclaration = [{
        OutputTiling getOutputTiling(const vpux::TileInfo& outputTile, vpux::Logger log);
    }] # baseExtraClassDeclaration;
}

//
// DetectionOutputCollectResults
//

def VPU_DetectionOutputCollectResultsOp:
        VPU_LayerOp<
            "DetectionOutputCollectResults"
        > {
    let summary = "DetectionOutputCollectResults VPU layer";

    let arguments = (ins
        4DTensorOf<[AnyFloat]>:$confidence,
        4DTensorOf<[AnyFloat]>:$boxes,
        4DTensorOf<[SI32]>:$sizes,

        IntAttr:$keep_top_k,
        BoolAttr:$clip_after_nms
    );

    let results = (outs
        4DTensorOf<[AnyFloat]>:$out_detections
    );
}

//
// MVN
//

def VPU_MVNOp :
        VPU_LayerOp<
            "MVN",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                VPU_EltwiseOp
            ]

        > {
    let summary = "MVN1 VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        BoolAttr:$across_channels,
        BoolAttr:$normalize_variance,
        F64Attr:$eps,
        OptionalAttr<I64ArrayAttr>:$internal_reshape,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);

        DimArr getNonNormDims();
    }] # baseExtraClassDeclaration;


    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::BoolAttr":$across_channels,
            "::mlir::BoolAttr":$normalize_variance,
            "::mlir::FloatAttr":$eps
        )>,
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::BoolAttr":$across_channels,
            "::mlir::BoolAttr":$normalize_variance,
            "::mlir::FloatAttr":$eps,
            "::mlir::ArrayAttr":$internal_reshape
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// MVN1Sum
//

def VPU_MVN1SumOp :
        VPU_LayerOp<
            "MVN1SumOp",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "MVN1SumOp VPU layer (step 1/3 in MVN1 decomposition)";

    let description = [{
        Overview:

        Large MVN1 tensors that cannot be tiled, are decomposed into 3 tileable sub-ops:
        1. **MVN1SumOp** : computes partial sums on input tiles
        2. **MVN1MeanVarOp** : sum-reduces concatenated partial sums from previous step and computes _mean_, _variance_
        3. **MVN1Normalize**: applies normalization on input tiles

        Details:
        - **input** - tile of original _MVN1Op_ input tensor
        - **sum** - output tensor of shape **[N, C, H, W]**, with (0,1,2,3)->(0,2,3,1) layout (irrespective of input layout)
            - N = input N
            - C = input C if _across_channels_ = false, or 1 if _across_channels_ = true
            - H = number of clusters (output_height)
            - W = 2 if _normalize_variance_ = true (compute _sum_ and _sumOfSquares_ terms), else 1 (compute just _sum_ term)
    }];

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,

        BoolAttr:$across_channels,
        BoolAttr:$normalize_variance,
        IntAttr:$output_height,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
        static bool buffsFitIntoCMX(mlir::ModuleOp module, vpux::NDTypeInterface in, vpux::NDTypeInterface out);
    }] # baseExtraClassDeclaration;

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F32]>, VPU_DistributedTensor]>:$sum
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "bool":$across_channels,
            "bool":$normalize_variance,
            "int64_t":$output_height
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// MVN1MeanVar
//

def VPU_MVN1MeanVarOp :
        VPU_LayerOp<
            "MVN1MeanVar",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "MVN1MeanVar VPU layer (step 2/3 in MVN1 decomposition)";

    let description = [{
        Background: see _MVN1SumOp_ description

        Details:
        Accumulates partial _sum_ (and optionally _sumOfSquares_) of concatenated input and computes _mean_ (and optionally _1/variance_ ) required for normalization.

        - **sum** - input is a concatenation (over W) of _MVN1SumOp_ outputs, shape = [N,C,W x num_parts]
            - W = 2 if _normalize_variance_ = true (input _sum_ and _sumOfSquares_), else 1 (input _sum_)
        - **meanVar** - output shape = [N,C,W], where
            - W = 2 if _normalize_variance_ = true (output _mean_ and _1/variance_ terms), else 1 (output just _mean_)
    }];

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F32]>, VPU_DistributedTensor]>:$sum,

        I64ArrayAttr:$orig_shape,
        BoolAttr:$across_channels,
        BoolAttr:$normalize_variance,
        F64Attr:$eps,
        TypeAttr:$output_type,
        OptionalAttr<I64ArrayAttr>:$internal_reshape,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$meanVar
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$sum,
            "::mlir::ArrayAttr":$orig_shape,
            "bool":$across_channels,
            "bool":$normalize_variance,
            "::mlir::APFloat":$eps,
            "::mlir::Type":$output_type
        )>,
        OpBuilder<(ins
            "::mlir::Value":$sum,
            "::mlir::ArrayAttr":$orig_shape,
            "bool":$across_channels,
            "bool":$normalize_variance,
            "::mlir::APFloat":$eps,
            "::mlir::Type":$output_type,
            "::mlir::ArrayAttr":$internal_reshape
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// MVN1Normalize
//

def VPU_MVN1NormalizeOp :
        VPU_LayerOp<
            "MVN1Normalize",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "MVN1Normalize VPU layer (step 3/3 in MVN1 decomposition)";

    let description = [{
        Background: see _MVN1SumOp_ description

        Applies normalization on a tile of input tensor.

        Details:
        - **input** - input tile of original _MVN1Op_ input tensor
        - **meanVar** - this input is the output of _MVN1MeanVarOp_
        - **output** - output tile of final result
    }];

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$meanVar,

        BoolAttr:$across_channels,
        BoolAttr:$normalize_variance,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$meanVar,
            "::mlir::BoolAttr":$across_channels,
            "::mlir::BoolAttr":$normalize_variance
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// MVN6
//

def VPU_MVN6Op :
        VPU_LayerOp<
            "MVN6",
            [
                AttrSizedOperandSegments,
                VPU_EltwiseOp,
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "MVN6 VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$scale,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$bias,
        I64ArrayAttr:$axes,
        BoolAttr:$normalize_variance,
        F64Attr:$eps,
        IE_MvnEpsModeAttr:$eps_mode,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
        DimArr getNonNormDims();
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes,
            "::mlir::BoolAttr":$normalize_variance,
            "::mlir::FloatAttr":$eps,
            "vpux::IE::MvnEpsModeAttr":$eps_mode
        )>,
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$scale,
            "::mlir::Value":$bias,
            "::mlir::ArrayAttr":$axes,
            "::mlir::BoolAttr":$normalize_variance,
            "::mlir::FloatAttr":$eps,
            "vpux::IE::MvnEpsModeAttr":$eps_mode
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// ROIPooling
//

def VPU_ROIPoolingOp :
        VPU_LayerOp<
            "ROIPooling"
        > {
    let summary = "ROIPooling VPU layer";

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
// PSROIPooling
//

def VPU_PSROIPoolingOp :
        VPU_LayerOp<
            "PSROIPooling"
        > {
    let summary = "PSROIPooling VPU layer";

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
// ROIAlign
//

def VPU_ROIAlignOp :
        VPU_LayerOp<
            "ROIAlign",
            [
               ResultsAreFloatLike
            ]
        > {
    let summary = "ROIAlign VPU layer";

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
// ExperimentalDetectronROIFeatureExtractor
//

def VPU_ExperimentalDetectronROIFeatureExtractorOp :
        VPU_LayerOp<
            "ExperimentalDetectronROIFeatureExtractor", [
               AttrSizedOperandSegments
            ]
        > {
    let summary = "ExperimentalDetectronROIFeatureExtractor VPU layer";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,
        Optional<1DTensorOf<[F16, F32]>>:$reorderedRois,
        Optional<1DTensorOf<[UI32]>>:$originalRoiMap,
        Optional<1DTensorOf<[F16, F32]>>:$outputRoisFeaturesTemp,
        Optional<1DTensorOf<[UI32]>>:$levels,

        IE_ExperimentalDetectronROIFeatureExtractorAttr:$attr

   );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output,
        RankedTensorOf<[F16, F32]>:$outputROIs
    );
}

//
// StridedSlice
//

def VPU_StridedSliceOp :
        VPU_LayerOp<
            "StridedSlice",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "StridedSlice VPU layer";

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

    let extraClassDeclaration = [{
        bool isSimplified();
    }];
}

//
// PRelu
//

def VPU_PReluOp :
        VPU_LayerOp<
            "PRelu",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "PRelu VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$negative_slope,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2
        )>
    ];

    let hasVerifier = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// LeakyRelu
//

def VPU_LeakyReluOp :
        VPU_LayerOp<
            "LeakyRelu",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "LeakyRelu VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$negative_slope
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Swish
//

def VPU_SwishOp :
        VPU_LayerOp<
            "Swish",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "Swish VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        Optional<AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>>:$beta,

        OptionalAttr<F64Attr>:$beta_value,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$beta,
            "::mlir::FloatAttr":$beta_value
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// ScaleShift
//

def VPU_ScaleShiftOp :
        VPU_LayerOp<
            "ScaleShift",
            [
                VPU_TilingBuilderOpInterface,
                AttrSizedOperandSegments,
                VPU_EltwiseOp
            ]
        > {
    let summary = "ScaleShift VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$weights,
        Optional<RankedTensorOf<[F16, F32]>>:$biases
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Upsampling
//

def VPU_UpsamplingOp :
        VPU_LayerOp<
            "Upsampling"
        > {
    let summary = "Upsampling VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input,
        I64ArrayAttr:$upsampling_factor,
        OptionalAttr<IE_UpsamplingPadAttr>:$pad
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );
}

//
// GRN
//

def VPU_GRNOp :
        VPU_LayerOp<
            "GRN"
        > {
    let summary = "GRN VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        F64Attr:$bias
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// Negative
//

def VPU_NegativeOp :
        VPU_LayerOp<
            "Negative",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Negative VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
    );
}

//
// Sign
//

def VPU_SignOp :
        VPU_LayerOp<
            "Sign",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Sign VPU layer";

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

def VPU_FullyConnectedOp:
        VPU_LayerOp<
            "FullyConnected"
        > {
    let summary = "FullyConnected VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$weights,
        Optional<RankedTensorOf<[F16, F32]>>:$bias
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// MatMul
//

def VPU_MatMulOp:
        VPU_LayerOp<
            "MatMul",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "MatMul VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        UnitAttr:$transpose_a,
        UnitAttr:$transpose_b,
        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);

        static bool isSupported(vpux::IE::MatMulOp matmulOp);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "::mlir::UnitAttr":$transpose_a,
            "::mlir::UnitAttr":$transpose_b,
            "vpux::IE::PostOpAttr":$post_op
        )>
    ];

    let hasVerifier = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// CTCGreedyDecoder
//

def VPU_CTCGreedyDecoderOp :
        VPU_LayerOp<
            "CTCGreedyDecoder"
        > {
    let summary = "CTCGreedyDecoder VPU layer";

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
// Reverse
//

def VPU_ReverseOp :
        VPU_LayerOp<
            "Reverse"
        > {
    let summary = "Reverse VPU operation";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayAttr:$axis_value,
        IE_ReverseModeAttr:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// CTCGreedyDecoderSeqLen
//

def VPU_CTCGreedyDecoderSeqLenOp :
        VPU_LayerOp<
            "CTCGreedyDecoderSeqLen"
        > {
    let summary = "CTCGreedyDecoderSeqLen VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[SI32]>:$sequenceLength,
        Optional<RankedTensorOf<[SI32]>>:$blankIndex,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        RankedTensorOf<[SI32]>:$output,
        RankedTensorOf<[SI32]>:$outputLength
    );
}

//
// Pad
//

def VPU_PadOp :
        VPU_LayerOp<
            "Pad",
            [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "Pad VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$pads_begin,
        Optional<RankedTensorOf<[AnyInteger]>>:$pads_end,
        Optional<RankedTensorOf<[AnyInteger, AnyFloat]>>:$pad_value,

        OptionalAttr<I64ArrayAttr>:$pads_begin_attr,
        OptionalAttr<I64ArrayAttr>:$pads_end_attr,
        OptionalAttr<F64Attr>:$pad_value_attr,

        IE_PadModeAttr:$mode,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<IntAttr>:$output_channels
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)` (`[` $pads_begin^ `,` $pads_end (`,` $pad_value^)? `]`)? attr-dict `:` type(operands) `->` type(results)
    }];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$pads_begin,
            "::mlir::Value":$pads_end,
            "::mlir::Value":$pad_value,
            "::mlir::ArrayAttr":$pads_begin_attr,
            "::mlir::ArrayAttr":$pads_end_attr,
            "::mlir::FloatAttr":$pad_value_attr,
            "vpux::IE::PadModeAttr":$mode,
            "::mlir::IntegerAttr":$output_channels
        )>,
        OpBuilder<(ins
            "vpux::NDTypeInterface&":$input_type,
            "::mlir::Value":$input,
            "::mlir::Value":$pads_begin,
            "::mlir::Value":$pads_end,
            "::mlir::Value":$pad_value,
            "::mlir::ArrayAttr":$pads_begin_attr,
            "::mlir::ArrayAttr":$pads_end_attr,
            "::mlir::FloatAttr":$pad_value_attr,
            "vpux::IE::PadMode":$mode,
            "::mlir::IntegerAttr":$output_channels
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// LSTMCell
//

def VPU_LSTMCellOp :
        VPU_LayerOp<
            "LSTMCell"
        > {
    let summary = "LSTMCell VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$inputData,
        RankedTensorOf<[F16, F32]>:$initialHiddenState,
        RankedTensorOf<[F16, F32]>:$initialCellState,
        RankedTensorOf<[F16, F32]>:$weights,
        RankedTensorOf<[F16, F32]>:$recurrenceWeights,
        RankedTensorOf<[F16, F32]>:$biases,

        IntAttr:$hiddenSize
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$outputHiddenState,
        RankedTensorOf<[F16, F32]>:$outputCellState
    );

    let extraClassDeclaration = [{
        static bool isSupported(vpux::IE::LSTMCellOp op);
    }] # baseExtraClassDeclaration;


}

//
// LSTMGatesOp
//

def VPU_LSTMGatesOp :
        VPU_LayerOp<
            "LSTMGates",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
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
        AnyTypeOf<[2DTensorOf<[F16, F32]>, 4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$gatesInput,
        AnyTypeOf<[2DTensorOf<[F16, F32]>, 4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$initialCellState,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[2DTensorOf<[F16, F32]>, 4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$outputHiddenState,
        AnyTypeOf<[2DTensorOf<[F16, F32]>, 4DTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$outputCellState
    );

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$gatesInput,
            "::mlir::Value":$initialCellState
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// LSTMSequence
//

def VPU_LSTMSequenceOp :
        VPU_LayerOp<
            "LSTMSequence",
        [
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverKernelCompatible",
                                                                     "isOperationSplitOverBatchCompatible"]>
        ]
        > {
    let summary = "LSTMSequence VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$inputData,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$initialHiddenState,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$initialCellState,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$reccurenceWeights,
        AnyTypeOf<[RankedTensorOf<[SI32]>, VPU_DistributedTensor]>:$syncBuffer,

        OptionalAttr<IntAttr>:$sequenceLength,
        IE_RNNSequenceDirectionAttr:$direction,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$outputHiddenValues,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$outputHiddenState,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$outputCellState
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$inputData,
            "::mlir::Value":$initialHiddenState,
            "::mlir::Value":$initialCellState,
            "::mlir::Value":$reccurenceWeights,
            "::mlir::IntegerAttr":$sequenceLength,
            "vpux::IE::RNNSequenceDirectionAttr":$direction,
            "vpux::VPU::MultiClusterStrategyAttr":$multiClusterStrategy
        )>
    ];

    let extraClassDeclaration = [{
        static bool isSupported(vpux::IE::LSTMSequenceOp origOp);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Select
//

def VPU_SelectOp :
        VPU_LayerOp<
            "Select",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
            ]
        > {
    let summary = "Select VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[Bool8, SI32, F16]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[SI32, F16]>, VPU_DistributedTensor]>:$input2,
        AnyTypeOf<[RankedTensorOf<[SI32, F16]>, VPU_DistributedTensor]>:$input3,
        IE_AutoBroadcastTypeAttr:$auto_broadcast,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[SI32, F16]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "::mlir::Value":$input3,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// SpaceToDepth
//

def VPU_SpaceToDepthOp :
        VPU_LayerOp<
            "SpaceToDepthOp",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "SpaceToDepthOp VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        DefaultValuedAttr<IntAttr, "1">:$block_size,
        IE_SpaceToDepthModeAttr:$mode
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// SpaceToBatch
//

def VPU_SpaceToBatch :
        VPU_LayerOp<
            "SpaceToBatch"
        > {
    let summary = "SpaceToBatch VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        OptionalAttr<I64ArrayAttr>:$block_shape_value,
        OptionalAttr<I64ArrayAttr>:$pads_begin_value,
        OptionalAttr<I64ArrayAttr>:$pads_end_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// BatchToSpace
//

def VPU_BatchToSpace :
        VPU_LayerOp<
            "BatchToSpace"
        > {
    let summary = "BatchToSpace VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        OptionalAttr<I64ArrayAttr>:$block_shape_value,
        OptionalAttr<I64ArrayAttr>:$crops_begin_value,
        OptionalAttr<I64ArrayAttr>:$crops_end_value
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// ReverseSequence
//

def VPU_ReverseSequenceOp :
        VPU_LayerOp<
            "ReverseSequence"
        > {
    let summary = "Reverse variable length sequence  VPU operation";

    let arguments = (ins
        AnyRankedTensor:$data,
        1DTensorOf<[AnyInteger]>:$seq_length,

        IntAttr:$seq_axis,
        IntAttr:$batch_axis
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// DepthToSpace
//

def VPU_DepthToSpaceOp :
        VPU_LayerOp<
            "DepthToSpace",
            [
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "DepthToSpace VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,
        IntAttr:$block_size,
        IE_DepthToSpaceModeAttr:$mode,
        OptionalAttr<IE_ChannelPaddingAttr>:$padded_channels,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);

        bool isVFSupported();
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::IntegerAttr":$block_size,
            "vpux::IE::DepthToSpaceModeAttr":$mode,
            "vpux::IE::ChannelPaddingAttr":$padded_channels
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// ExtractImagePatches
//

def VPU_ExtractImagePatchesOp :
        VPU_LayerOp<
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

//
// YuvToRgb
//  Conversions:
//   NV12toRGB, NV12toBGR,
//   I420toRGB, I420toBGR
//

def VPU_YuvToRgbOp :
        VPU_LayerOp<
            "YuvToRgb",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
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
}

//
// RandomUniform
//

def VPU_RandomUniformOp :
        VPU_LayerOp<
            "RandomUniform"
        > {
    let summary = "RandomUniform VPU layer";

    let arguments = (ins
        1DTensorOf<[F16, F32, SI32]>:$min,
        1DTensorOf<[F16, F32, SI32]>:$max,

        I64ArrayAttr:$output_shape,
        TypeAttr:$outputType,
        IntAttr:$global_seed,
        IntAttr:$op_seed
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
    );
}

//
// OneHot
//

def VPU_OneHotOp :
        VPU_LayerOp<
            "OneHot"
        > {
    let summary = "InferenceEngine OneHot layer";

    let arguments = (ins
        RankedTensorOf<[SI32, SI64]> :$input,

        IntAttr:$depth,
        F64Attr:$on_value,
        F64Attr:$off_value,
        IntAttr:$axis,

        TypeAttr:$outputType
    );

    let results = (outs
        AnyRankedTensor:$output
    );
}

//
// M2I.ColorConvert
//

def VPU_M2IColorConvertOp :
        VPU_LayerOp<
            "M2I.ColorConvert"
        > {
    let summary = "M2I version for color-convert operations";

    let arguments = (ins
        4DTensorOf<[UI8, F16]>:$input,
        IE_ColorFmtAttr:$inFmt,
        IE_ColorFmtAttr:$outFmt
    );

    let results = (outs
         4DTensorOf<[UI8, F16]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem);
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output);
        static bool isSupported(vpux::IE::YuvToRgbOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);
    }] # baseExtraClassDeclaration;
}

//
// M2I.Resize
//

def VPU_M2IResizeOp :
        VPU_LayerOp<
            "M2I.Resize"
        > {
    let summary = "M2I version for resize operations";

    let arguments = (ins
        4DTensorOf<[UI8, F16]>:$input,

        I64ArrayAttr:$sizes,
        I64ArrayAttr:$axes,
        VPU_M2iInterpAttr:$interp
    );

    let results = (outs
         4DTensorOf<[UI8, F16]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem);
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output);
        static bool isSupported(vpux::IE::InterpolateOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);
    }] # baseExtraClassDeclaration;
}

//
// M2I.Norm
//

def VPU_M2INormOp :
        VPU_LayerOp<
            "M2I.Norm"
        > {
    let summary = "M2I version for BatchNormInference";

    let arguments = (ins
        4DTensorOf<[F16]>:$input,

        F64ArrayAttr:$gamma_value,
        F64ArrayAttr:$beta_value,
        F64ArrayAttr:$mean_value,
        F64ArrayAttr:$variance_value,

        F64Attr:$eps
    );

    let results = (outs
         4DTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem);
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output);
        static bool isSupported(vpux::IE::BatchNormInferenceOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);
    }] # baseExtraClassDeclaration;
}

//
// M2I.Task
//

def VPU_M2ITaskOp :
        VPU_LayerOp<
            "M2I.Task"
        > {
    let summary = "M2I full task op";

    let arguments = (ins
        4DTensorOf<[UI8, F16]>:$input,

        BoolAttr:$do_csc,
        BoolAttr:$do_norm,
        VPU_M2iColorFmtAttr:$inFmt,
        VPU_M2iColorFmtAttr:$outFmt,
        UnitAttr:$chroma_in_reverse_channels,
        UnitAttr:$chroma_out_reverse_channels,
        UnitAttr:$luma_in_reverse_channels,
        UnitAttr:$luma_out_reverse_channels,
        OptionalAttr<I64ArrayAttr>:$sizes,
        OptionalAttr<I64ArrayAttr>:$axes,
        OptionalAttr<F64ArrayAttr>:$norm,
        DefaultValuedAttr<VPU_M2iInterpAttr, "vpux::VPU::M2iInterp::NEAREST">:$interp
    );

    let results = (outs
         4DTensorOf<[UI8, F16]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
    }];
}

//
// Tile
//

def VPU_TileOp :
        VPU_LayerOp<
            "Tile",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "Tile VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        I64ArrayAttr:$repeats_values
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

//
// DynamicTile
//

def VPU_DynamicTileOp :
        VPU_LayerOp<
            "DynamicTile",
            [
                DeclareOpInterfaceMethods<VPU_SWOpInterface>
            ]
        > {
    let summary = "DynamicTile VPU layer";

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

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;
}

//
// Split
//

def VPU_SplitOp :
        VPU_LayerOp<
            "Split",
            [
            VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Split VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,
        Optional<AnyRankedTensor>:$axis,

        IntAttr:$num_splits,
        OptionalAttr<IntAttr>:$axis_value
    );

    let results = (outs
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>>:$outputs
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
    let hasVerifier = 1;
}

//
// Power
//

def VPU_PowerOp :
        VPU_LayerOp<
            "Power",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Power VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Add
//

def VPU_AddOp :
        VPU_LayerOp<
            "Add",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Add VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast,
            "vpux::IE::PostOpAttr":$post_op
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Divide
//

def VPU_DivideOp :
        VPU_LayerOp<
            "Divide",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Divide VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// SquaredDiff
//

def VPU_SquaredDifferenceOp :
        VPU_LayerOp<
            "SquaredDiff",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "SquaredDiff VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
    );
}

//
// FloorMod
//

def VPU_FloorModOp :
        VPU_LayerOp<
            "FloorMod",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "FloorMod VPU layer";

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
// Mod
//

def VPU_ModOp :
        VPU_LayerOp<
            "Mod",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Mod VPU layer";

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
// Less
//

def VPU_LessOp :
        VPU_LayerOp<
            "Less",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Less VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[Bool8]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// LessEqual
//

def VPU_LessEqualOp :
        VPU_LayerOp<
            "LessEqual",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "LessEqual VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[Bool8]>:$output
    );
}

//
// Greater
//

def VPU_GreaterOp :
        VPU_LayerOp<
            "Greater",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Greater VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[Bool8]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// GreaterEqual
//

def VPU_GreaterEqualOp :
        VPU_LayerOp<
            "GreaterEqual",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "GreaterEqual VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32]>:$output
    );
}

//
// LogicalOr
//

def VPU_LogicalOrOp :
        VPU_LayerOp<
            "LogicalOr",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "LogicalOr VPU layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, SI32]>:$input1,
        RankedTensorOf<[I8, F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, SI32]>:$output
    );
}

//
// LogicalXor
//

def VPU_LogicalXorOp :
        VPU_LayerOp<
            "LogicalXor",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "LogicalXor VPU layer";

    let arguments = (ins
        RankedTensorOf<[I8, F16, F32, SI32]>:$input1,
        RankedTensorOf<[I8, F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[I8, F16, F32, SI32]>:$output
    );
}

//
// Multiply
//

def VPU_MultiplyOp :
        VPU_LayerOp<
            "Multiply",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "Multiply VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast,
            "vpux::IE::PostOpAttr":$post_op
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// And
//

def VPU_AndOp :
        VPU_LayerOp<
            "And",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "And VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[I8, F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[I8, F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[I8, F16, F32, SI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// BitwiseAnd
//

def VPU_BitwiseAndOp :
        VPU_LayerOp<
            "BitwiseAnd",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "BitwiseAnd VPU layer";

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
// BitwiseOr
//

def VPU_BitwiseOrOp :
        VPU_LayerOp<
            "BitwiseOr",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "BitwiseOr VPU layer";

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
// BitwiseXor
//

def VPU_BitwiseXorOp :
        VPU_LayerOp<
            "BitwiseXor",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "BitwiseXor VPU layer";

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
// BitwiseNot
//

def VPU_BitwiseNotOp :
        VPU_LayerOp<
            "BitwiseNot",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "BitwiseNot VPU layer";

    let arguments = (ins
        RankedTensorOf<[AnyInteger]>:$input1
    );

    let results = (outs
        RankedTensorOf<[AnyInteger]>:$output
    );
}

//
// GroupConvolution
//

def VPU_GroupConvolutionOp :
        VPU_LayerOp<
            "GroupConvolution",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "GroupConvolution VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$input,
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$filter,
        Optional<RankedTensorOf<[F16, F32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        OptionalAttr<IntAttr>:$groups,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<IntAttr>:$output_channels
    );

    let results = (outs
        RankedTensorOf<[F16, F32, quant_QuantizedType]>:$output
    );

    list<string> elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output, Byte reservedMem);

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);
    }] # baseExtraClassDeclaration;
}

//
// GroupNormalizationOp
//

def VPU_GroupNormalizationOp :
        VPU_LayerOp<
            "GroupNormalization",
            [
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
            ]
        > {
    let summary = "GroupNormalization VPU layer";

    let arguments = (ins
       RankedTensorOf<[F16, F32]>:$input,
       RankedTensorOf<[F16, F32]>:$scale,
       RankedTensorOf<[F16, F32]>:$bias,

       I32Attr:$num_groups,
       F32Attr: $epsilon
    );

    let results = (outs
       RankedTensorOf<[F16, F32]>:$output
    );
}

//
// AvgPool
//

def VPU_AvgPoolOp :
        VPU_LayerOp<
            "AvgPool",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "AvgPool VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,
        UnitAttr:$exclude_pads
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$output
    );
}

//
// MaxPool
//

def VPU_MaxPoolOp :
        VPU_LayerOp<
            "MaxPool",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "MaxPool VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$input,

        I64ArrayAttr:$kernel_size,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        IE_RoundingTypeAttr:$rounding_type,

        OptionalAttr<IE_PostOpAttr>:$post_op
    );

    let results = (outs
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$output
    );
}

//
// MaxPool8
//

def VPU_MaxPool8Op :
        VPU_LayerOp<
            "MaxPool8"
        > {
    let summary = "MaxPool8 VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$input,

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
        RankedTensorOf<[F16, F32, SI32, SI8, UI8]>:$output,
        RankedTensorOf<[SI32, SI64]>:$output_index
    );
}

//
// Reshape
//

def VPU_ReshapeOp :
        VPU_LayerOp<
            "Reshape",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Reshape VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$shape,

        UnitAttr:$special_zero,
        OptionalAttr<I64ArrayAttr>:$shape_value
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;

    let hasCanonicalizer = 1;
}

//
// Squeeze
//

def VPU_SqueezeOp :
        VPU_LayerOp<
            "Squeeze",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Squeeze VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// Unsqueeze
//

def VPU_UnsqueezeOp :
        VPU_LayerOp<
            "Unsqueeze",
            [
                VPU_ViewLikeOpInterface
            ]
        > {

    let summary = "Unsqueeze VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$input,
        Optional<RankedTensorOf<[AnyInteger]>>:$axes,

        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// LRN
//

def VPU_LRNOp :
        VPU_LayerOp<
            "LRN"
        > {
    let summary = "LRN VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        I64ArrayAttr:$axes,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$size
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// LRN_IE
//

def VPU_LRN_IEOp :
        VPU_LayerOp<
            "LRN_IE"
        > {
    let summary = "LRN_IE VPU layer";

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
// ReduceMax
//

def VPU_ReduceMaxOp :
        VPU_LayerOp<
            "ReduceMax",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceMax VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// ReduceMean
//

def VPU_ReduceMeanOp :
        VPU_LayerOp<
            "ReduceMean",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceMean VPU Layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// ReduceSum
//

def VPU_ReduceSumOp :
        VPU_LayerOp<
            "ReduceSum",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceSum VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// ReduceL1
//

def VPU_ReduceL1Op :
        VPU_LayerOp<
            "ReduceL1",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceL1 VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// ReduceLogicalOr
//

def VPU_ReduceLogicalOrOp :
        VPU_LayerOp<
            "ReduceLogicalOr",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceLogicalOr VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// ReduceLogicalAnd
//

def VPU_ReduceLogicalAndOp :
        VPU_LayerOp<
            "ReduceLogicalAnd",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceLogicalAnd VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// ReduceProd
//

def VPU_ReduceProdOp :
        VPU_LayerOp<
            "ReduceProd",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceProd VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// ReduceMin
//

def VPU_ReduceMinOp :
        VPU_LayerOp<
            "ReduceMin",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
            ]
        > {
    let summary = "ReduceMin VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasFolder = 1;
}

//
// ReduceL2
//

def VPU_ReduceL2Op :
        VPU_LayerOp<
            "ReduceL2",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "ReduceL2 VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::ArrayAttr":$axes_value,
            "::mlir::UnitAttr":$keep_dims
        )>
    ];

    let hasFolder = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Minimum
//

def VPU_MinimumOp :
        VPU_LayerOp<
            "Minimum",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "InferenceEngine Minimum layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Maximum
//

def VPU_MaximumOp :
        VPU_LayerOp<
            "Maximum",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "Maximum VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input2,
        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Sparsify
//

def VPU_SparsifyOp :
        VPU_LayerOp<"Sparsify",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Sparsify VPU layer";

    let arguments = (ins
        4DTensorOf<[quant_QuantizedType, F16, BF16]>:$input
    );

    let results = (outs
        VPU_SparseTensor:$output
    );
}

//
// Desparsify
//

def VPU_DesparsifyOp :
        VPU_LayerOp<"Desparsify",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Desparsify VPU layer";

    let arguments = (ins
        VPU_SparseTensor:$input
    );

    let results = (outs
        4DTensorOf<[quant_QuantizedType, F16, BF16]>:$output
    );
}

//
// Quantize
//

def VPU_QuantizeOp :
        VPU_LayerOp<"Quantize",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Quantize VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        RankedTensorOf<[quant_QuantizedType]>:$output
    );
}

//
// Dequantize
//

def VPU_DequantizeOp :
        VPU_LayerOp<"Dequantize",
            [
                VPU_TilingBuilderOpInterface,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, [
                    "isOperationSplitOverKernelCompatible",]>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "Dequantize VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[quant_QuantizedType]>, VPU_DistributedTensor]>:$input,
        TypeAttr:$dstElemType,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32,]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::TypeAttr":$dstElemType
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
        bool isVFSupported();
    }] # baseExtraClassDeclaration;
}

//
// DynamicQuantize
//

def VPU_DynamicQuantizeOp :
        VPU_LayerOp<"DynamicQuantize"
        > {
    let summary = "Dynamic-Quantize VPU layer";

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
// QuantizeCast
//

def VPU_QuantizeCastOp :
        VPU_LayerOp<
            "QuantizeCast",
            [
                VPU_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_TilingViewLikeOpInterface>,
                DeclareOpInterfaceMethods<VPU_DistributedCastOpInterface>
            ]
        > {
    let summary = "Quantize Cast VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$input,

        TypeAttr:$dstElemType
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let hasVerifier = 1;
    let hasFolder = 1;
}

//
// TransposedConvolution
//

def VPU_TransposedConvolutionOp:
        VPU_LayerOp<
            "TransposedConvolution",
            [
                AttrSizedOperandSegments
            ]
        > {
    let summary = "TransposedConvolution VPU layer";

    let description = [{
        Represents a transposed convolution, which consumes an input and filter tensor
        and generates an output larger than the input.

        Operands:
        - `input`: the input tensor of the operation; 4D layout [N, C_IN, Y, X]
        - `filter`: the convolutional kernel tensor; expected layout [C_OUT, C_IN, KY, KX]
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
        Optional<RankedTensorOf<[F16, F32]>>:$bias,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$output_padding,

        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<DictionaryAttr>:$clamp,
        OptionalAttr<IntAttr>:$output_channels
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT];
}

//
// Expand
//

def VPU_ExpandOp :
        VPU_LayerOp<
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

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "std::optional<vpux::ShapeRef>":$pads_begin, "std::optional<vpux::ShapeRef>":$pads_end)
        >
    ];

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// Subtract
//

def VPU_SubtractOp :
        VPU_LayerOp<
            "Subtract",
            [
                VPU_TilingBuilderOpInterface,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Subtract VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<IE_PostOpAttr>:$post_op,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32, UI8, UI16, UI32]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast,
            "vpux::IE::PostOpAttr":$post_op
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// MemPermute
//

def VPU_MemPermuteOp :
        VPU_LayerOp<
            "MemPermute",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "MemPermute VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 0;
    let hasCanonicalizer = 1;
}

//
// PermuteCast
//

def VPU_PermuteCastOp :
        VPU_LayerOp<
            "PermuteCast",
            [
                VPU_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_DistributedCastOpInterface>
            ]
        > {
    let summary = "PermuteCast VPU layer";

    let description = [{
        The op changes layout information in the following way:
            * dst_order: layout attribute of result is set to value of this arg
            * mem_perm: describes the permutation applied on the input value's memory shape
                        to obtain the memory shape of the output value.
    }];

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// Equal
//

def VPU_EqualOp :
        VPU_LayerOp<
            "Equal",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
                VPU_EltwiseOp
            ]
        > {
    let summary = "Equal VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[RankedTensorOf<[F16, F32, SI32]>, VPU_DistributedTensor]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[Bool8]>, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input1,
            "::mlir::Value":$input2,
            "vpux::IE::AutoBroadcastTypeAttr":$auto_broadcast
        )>
    ];

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// AffineReshape
//

def VPU_AffineReshapeOp :
        VPU_LayerOp<
            "AffineReshape",
            [
                VPU_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_DistributedCastOpInterface>
            ]
        > {
    let summary = "AffineReshape VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$input,

        I64ArrayOfArraysAttr:$dim_mapping,
        I64ArrayAttr:$shape_value
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;
}

//
// NotEqual
//

def VPU_NotEqualOp :
        VPU_LayerOp<
            "NotEqual",
            [
                VPU_TilingBuilderOpInterface,
                Commutative,
                VPU_EltwiseOp
            ]
        > {
    let summary = "NotEqual VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32, SI32]>:$input1,
        RankedTensorOf<[F16, F32, SI32]>:$input2,

        IE_AutoBroadcastTypeAttr:$auto_broadcast
    );

    let results = (outs
        RankedTensorOf<[Bool8]>:$output
    );
}

//
// Copy
//

def VPU_CopyOp :
        VPU_LayerOp<
            "Copy"
        > {
    let summary = "Copy VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$input,

        OptionalAttr<IndexedSymbolAttr>:$out_mem_space
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let checkInferredDimsOrder = 1;
    let checkInferredMemSpace = 1;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// ExpandDilated
//

def VPU_ExpandDilatedOp :
        VPU_LayerOp<
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
}

//
// StorageElementTable
//

def VPU_StorageElementTableOp :
        VPU_Op<
            "StorageElementTable",
            [
                Pure,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
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
        - seSize: the size of a Storage Element
        - seDepth: the number of Storage Elements per depth
        - seAttr: information on how the input data is transformed
        - basePtrs: base pointers associated with each Storage Element pointer
    }];

    let arguments = (ins
        I64ArrayAttr:$dataShape,
        TypeAttr:$dataElemType,
        IntAttr:$seSize,
        IntAttr:$seDepth,
        OptionalAttr<VPU_SEAttr>:$seAttr,
        OptionalAttr<I64ArrayAttr>:$dataStrides,
        OptionalAttr<I32ElementsAttr>:$basePtrs
    );

    let results = (outs
        RankedTensorOf<[I32]>:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
         attr-dict `->` type(results)
    }];

    let builders = [
        OpBuilder<(ins
            CArg<"llvm::ArrayRef<int64_t>">:$dataShape,
            CArg<"mlir::Type">:$dataElemType,
            CArg<"int64_t">:$seSize,
            CArg<"int64_t">:$seDepth,
            CArg<"VPU::SEAttr">:$seAttr
        )>
    ];

    let hasCanonicalizer = 1;
}

//
// NonMaxSuppression
//

def VPU_NonMaxSuppressionOp :
        VPU_LayerOp<
            "NonMaxSuppression"
        > {
    let summary = "NonMaxSuppression VPU layer";

    let arguments = (ins
        3DTensorOf<[F16, F32]>:$in_box_coords,
        3DTensorOf<[F16, F32]>:$in_box_scores,

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
}


//
// StubOp
//

def VPU_StubOp :
        VPU_Op<
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
// GRUSequence
//

def VPU_GRUSequenceOp :
        VPU_LayerOp<
            "GRUSequence",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "GRUSequence VPU layer";

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

    let extraClassDeclaration = [{
        OutputTiling getOutputTiling(const vpux::TileInfo& outputTile, vpux::Logger log);
    }] # baseExtraClassDeclaration;
}

//
// GRUSequenceFirstPart
//

def VPU_GRUSequenceFirstPartOp :
        VPU_LayerOp<
            "GRUSequenceFirstPart"
        > {
    let summary = "GRUSequenceFirstPart VPU layer";

    let arguments = (ins
        3DTensorOf<[F16, F32]>:$input_data,
        3DTensorOf<[F16, F32]>:$weights,

        IntAttr:$hidden_size,
        IntAttr:$seq_length,
        F64Attr:$clip
    );

    let results = (outs
        4DTensorOf<[F16, F32]>:$output
    );
}

//
// GRUSequenceLastPart
//

def VPU_GRUSequenceLastPartOp :
        VPU_LayerOp<
            "GRUSequenceLastPart",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "GRUSequenceLastPart VPU layer";

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

    let extraClassDeclaration = [{
        OutputTiling getOutputTiling(const vpux::TileInfo& outputTile, vpux::Logger log);
    }] # baseExtraClassDeclaration;
}

//
// DeformablePSROIPoolingOp
//

def VPU_DeformablePSROIPoolingOp :
        VPU_LayerOp<
            "DeformablePSROIPooling"
        > {
    let summary = "DeformablePSROIPooling VPU layer";

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
// PermuteQuantize
//

def VPU_PermuteQuantizeOp :
        VPU_LayerOp<
            "PermuteQuantize",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "PermuteQuantize VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm,
        TypeAttr:$dstElemType,
        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end
    );

    let results = (outs
        RankedTensorOf<[quant_QuantizedType]>:$output
    );
}

//
// DFTOp
//

def VPU_DFTOp :
        VPU_LayerOp<
            "DFT",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "InferenceEngine DFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$twiddle_factors,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// RDFTOp
//

def VPU_RDFTOp :
        VPU_LayerOp<
            "RDFT"
        > {
    let summary = "InferenceEngine RDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$twiddle_factors,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// IDFTOp
//

def VPU_IDFTOp :
        VPU_LayerOp<
            "IDFT",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "InferenceEngine IDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$twiddle_factors,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// IRDFTOp
//

def VPU_IRDFTOp :
        VPU_LayerOp<
            "IRDFT"
        > {
    let summary = "InferenceEngine IRDFT layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$twiddle_factors,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// RDFTUncutOp
//

def VPU_RDFTUncutOp :
        VPU_LayerOp<
            "RDFTUncut",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "RDFTUncut VPU layer";

    let description = [{
        Operation apply RDFT transformation but not cut symmetric part on last axis width value from axes_attr.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$twiddle_factors,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// IRDFTLastAxisOp
//

def VPU_IRDFTLastAxisOp :
        VPU_LayerOp<
            "IRDFTLastAxis",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>
            ]
        > {
    let summary = "IRDFTLastAxis VPU layer";

    let description = [{
        Operation apply IRDFT transformation but just on last axis from standard IRDFT operation.
        Used to produce full IRDFT capability (in combination with IDFT) without computation
        and the data movement unnecessary for the direction of the last transform axis.
    }];

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        Optional<RankedTensorOf<[F16, F32]>>:$twiddle_factors,
        I64ArrayAttr:$axes_attr,
        I64ArrayAttr:$signal_size_attr

    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}

//
// ShapeCastOp
//

def VPU_ShapeCastOp :
        VPU_LayerOp<
            "ShapeCast",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "ShapeCast VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$source,
        I64ArrayAttr:$shape
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>:$result
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $source `:` type($source) `)`
        `->` type(results)
    }];

    let hasFolder = 1;

    let hasCanonicalizer = 1;
}

//
// LayoutCastOp
//

def VPU_LayoutCastOp :
        VPU_LayerOp<
            "LayoutCast",
            [
                VPU_ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_DistributedCastOpInterface>
            ]
        > {
    let summary = "This layer overrides layout of a given tensor.";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        AffineMapAttr:$dst_order
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let hasVerifier = 1;
}

//
// UnrolledTypeOp
//

def VPU_UnrolledTypeOp :
        VPU_Op<
            "UnrolledType",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "This layer mediate between unrolled distributed tensor type and usual type";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$input
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let assemblyFormat = [{
        `(` $input `:` qualified(type($input)) `)`
        attr-dict
        `->` qualified(type($output))
    }];

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

//
// WorkloadCastOp
//

def VPU_WorkloadCastOp :
        VPU_Op<
            "WorkloadCast",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Operation that casts one DistributedTensor type to another.";

    let description = [{
        This operation is required in order to support cluster tiling for VPU.NCE.PermuteQuantize.
        PermuteQuantize operates on workloads split over width, while input DMA must tile over height.
        For example consider the following chain of operations:
        ```
            Input 1x3x16x32 -> Reshape 1x32x3x16 -> VPU.NCE.PermuteQuantize -> Reshape 1x3x16x32
        ```
        VPU.NCE.PermuteQuantize operates with 1x32x3x16 workload while original input has 1x3x16x32 shape.
        Original input must be split over height, PermuteQuantize workload must be split over width:
        ```
            Tile 1: Copy 1x3x8x32 -> Reshape 1x32x3x8 -> VPU.NCE.PermuteQuantize -> Reshape 1x3x8x32
            Tile 2: Copy 1x3x8x32 -> Reshape 1x32x3x8 -> VPU.NCE.PermuteQuantize -> Reshape 1x3x8x32
        ```
        However, Reshape prohibits such tiling because split axis differs in input and output.
        VPU.WorkloadCast solves this problem because it doesn't have strict checks:
        ```
            Copy 1x3x16x32, SOH -> WorkloadCast 1x32x3x8, SOW -> PermuteQuantize -> WorkloadCast 1x32x3x8, SOH
            SOH = [1, 1, 2, 1]
            SOW = [1, 1, 1, 2]
        ```
    }];

    let arguments = (ins
        AnyTypeOf<[VPU_DistributedTensor, VPU_SparseTensor]>:$input
    );

    let results = (outs
        AnyTypeOf<[VPU_DistributedTensor, VPU_SparseTensor]>:$output
    );

    let assemblyFormat = [{
        `(` $input `:` qualified(type($input)) `)`
        attr-dict
        `->` qualified(type($output))
    }];
}

//
// Accumulate
//

def VPU_AccumulateOp :
        VPU_LayerOp<
            "Accumulate",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "Accumulate VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$lhs,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$rhs,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$lhsScale,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$rhsScale,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// ShapeOf
//

def VPU_ShapeOfOp :
        VPU_LayerOp<
            "ShapeOf",
            [
                DeclareOpInterfaceMethods<VPU_SWOpInterface>
            ]
        > {
    let summary = "ShapeOf VPU layer";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor]>:$input
    );

    let results = (outs
        1DTensorOf<[SI32]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;
}

//
// Range
//

def VPU_RangeOp :
        VPU_LayerOp<
            "Range"
        > {
    let summary = "Range VPU layer";

    let arguments = (ins
        1DTensorOf<[AnyInteger, AnyFloat]>:$start,
        1DTensorOf<[AnyInteger, AnyFloat]>:$stop,
        1DTensorOf<[AnyInteger, AnyFloat]>:$step,
        TypeAttr:$dstElemType
    );

    let results = (outs
        1DTensorOf<[AnyInteger, AnyFloat]>:$output
    );

    let hasVerifier = 1;
}

//
// NonZero
//

def VPU_NonZeroOp :
        VPU_LayerOp<
            "NonZero"
        > {
    let summary = "NonZero VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, SI32, Bool8]>:$input
    );

    let results = (outs
        2DTensorOf<[SI32]>:$output
    );

    let hasVerifier = 1;
}

//
// DynamicReshape
//

def VPU_DynamicReshapeOp :
        VPU_LayerOp<
            "DynamicReshape",
            [
                DeclareOpInterfaceMethods<VPU_SWOpInterface>
            ]
        > {
    let summary = "DynamicReshape VPU layer";

    let arguments = (ins
        AnyRankedTensor:$input,
        RankedTensorOf<[SI32]>:$shape,

        I64ArrayAttr:$output_shape,
        I64ArrayAttr:$output_bounds,

        UnitAttr:$only_set_shape
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;
}

//
// DynamicDequantizeOp
//

def VPU_DynamicDequantizeOp :
        VPU_LayerOp<"DynamicDequantize",
            [
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]
        > {
    let summary = "InferenceEngine Dynamic Dequantize layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[quant_QuantizedType]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$scale,
        Optional<AnyTypeOf<[RankedTensorOf<[I8, I4]>, VPU_DistributedTensor]>>:$zp,

        TypeAttr:$dstElemType,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$scale,
            "::mlir::Value":$zp,
            "::mlir::TypeAttr":$dstElemType
        )>
    ];

    let hasVerifier = 1;
}

//
// DeformableConvolution
//

def VPU_DeformableConvolutionOp :
        VPU_LayerOp<
            "DeformableConvolution"
        > {
    let summary = "DeformableConvolution VPU layer";

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
// PopulateWeightTable
//

def VPU_PopulateWeightTableOp :
        VPU_LayerOp<
            "PopulateWeightTable",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>
            ]

        > {
    let summary = "Populate weight table VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>,VPU_DistributedTensor]>:$scale,

        IntAttr:$base,
        IntAttr:$step,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[SI32]>,VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;


    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$scale,
            CArg<"int64_t">:$base,
            CArg<"int64_t">:$step
        )>
    ];
    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// RMS
//

def VPU_RMSOp :
        VPU_LayerOp<
            "RMS",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
            ]
        > {
    let summary = "RMS VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$gamma,

        F64Attr:$epsilon,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$gamma,
            "::mlir::FloatAttr":$epsilon
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

//
// Inverse
//

def VPU_InverseOp :
        VPU_LayerOp<
            "Inverse"
        > {
    let summary = "Inverse VPU layer";

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

def VPU_DynamicExpandOp :
        VPU_LayerOp<
            "DynamicExpand"
        > {
    let summary = "DynamicExpand operation";

    let description = [{
        DynamicExpand operation is designed to take an input tensor with dynamic shapes and perform zero-padding to
        match the dimensions of an upper bound.
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
}

//
// GenericSwLayerOp
//

def VPU_GenericSwLayerOp :
        VPU_Op<
            "GenericSwLayer",
            [
                Pure,
                DeclareOpInterfaceMethods<CallOpInterface>,
                DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_LayerOpInterface>
            ]
        > {
    let summary = "VPU Op that describes a generic SW Layer";

    let arguments = (ins
        SymbolRefAttr:$callee, // Generated sw layer function

        Variadic<AnyType>:$inputs
    );

    let results = (outs
        Variadic<AnyShaped>:$outputs
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }];

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

//
// RoPE
//

def VPU_RoPEOp :
        VPU_LayerOp<
            "RoPE",
            [
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface>,
            ]
        > {
    let summary = "RoPE VPU layer";

    let arguments = (ins
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input_cos,
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$input_sin,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[RankedTensorOf<[F16, F32]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }] # baseExtraClassDeclaration;

    let builders = [
        OpBuilder<(ins
            "::mlir::Value":$input,
            "::mlir::Value":$input_cos,
            "::mlir::Value":$input_sin
        )>
    ];

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}

#endif
