//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_DPU
#define VPUX_COMPILER_DIALECT_VPU_OPS_DPU

include "vpux/compiler/dialect/core/constraints.td"
include "vpux/compiler/dialect/core/types.td"
include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"


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


def VPU_NCEConvolutionOp :
        VPU_LayerOp<
            "NCE.Convolution",
            [
                AttrSizedOperandSegments,
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
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "NCE version of Convolution layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$filter,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weightsTable,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weight_table_data_ptr,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weight_table_sp_ptr,
        Optional<AnyTypeOf<[4DTensorOf<[F32, F8E5M2, F8E4M3FN]>, VPU_DistributedTensor]>>:$weight_table_scale,
        Optional<AnyTypeOf<[4DTensorOf<[F32]>, VPU_DistributedTensor]>>:$weight_table_bias,
        Optional<AnyTypeOf<[4DTensorOf<[I8, I4]>, VPU_DistributedTensor]>>:$weight_zero_points,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        VPU_PPEAttr:$ppe,
        OptionalAttr<VPU_MPEEngineAttr>:$mpe_engine,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<4>]>:$rawFilterShape,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
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
            $filter
            (`,` $weightsTable^)?
            (`,` $weight_table_data_ptr^)?
            (`,` $weight_table_sp_ptr^)?
            (`,` $weight_table_scale^)?
            (`,` $weight_table_bias^)?
            (`,` $weight_zero_points^)?
        `)`
        attr-dict `:` type(operands) `->` type(results)
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


def VPU_NCEMatMulOp :
        VPU_LayerOp<
            "NCE.MatMul",
            [
                AttrSizedOperandSegments,
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
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

        Optional<AnyTypeOf<[5DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weightsTable,
        Optional<AnyTypeOf<[5DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weight_table_data_ptr,
        Optional<AnyTypeOf<[5DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weight_table_sp_ptr,
        Optional<AnyTypeOf<[5DTensorOf<[F32, F8E5M2, F8E4M3FN]>, VPU_DistributedTensor]>>:$weight_table_scale,
        Optional<AnyTypeOf<[5DTensorOf<[F32]>, VPU_DistributedTensor]>>:$weight_table_bias,
        Optional<AnyTypeOf<[5DTensorOf<[I8, I4]>, VPU_DistributedTensor]>>:$weight_zero_points,

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
            $weights
            (`,` $weightsTable^ `` custom<OptionalTypes>(type($weightsTable)))?
            (`,` $weight_table_data_ptr^ `` custom<OptionalTypes>(type($weight_table_data_ptr)))?
            (`,` $weight_table_sp_ptr^ `` custom<OptionalTypes>(type($weight_table_sp_ptr)))?
            (`,` $weight_table_scale^ `` custom<OptionalTypes>(type($weight_table_scale)))?
            (`,` $weight_table_bias^ `` custom<OptionalTypes>(type($weight_table_bias)))?
            (`,` $weight_zero_points^ `` custom<OptionalTypes>(type($weight_zero_points)))?
        `)`
        attr-dict
        custom<OptionalTypes>(type($input), type($weights)) ``
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


def VPU_NCEDepthConvolutionOp :
        VPU_LayerOp<
            "NCE.DepthConvolution",
            [
                AttrSizedOperandSegments,
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
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
            ]
        > {
    let summary = "NCE version of Depthwise Convolution layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$filter,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weightsTable,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weight_table_data_ptr,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weight_table_sp_ptr,
        Optional<AnyTypeOf<[4DTensorOf<[F32, F8E5M2, F8E4M3FN]>, VPU_DistributedTensor]>>:$weight_table_scale,
        Optional<AnyTypeOf<[4DTensorOf<[F32]>, VPU_DistributedTensor]>>:$weight_table_bias,
        Optional<AnyTypeOf<[4DTensorOf<[I8, I4]>, VPU_DistributedTensor]>>:$weight_zero_points,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        VPU_PPEAttr:$ppe,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<4>]>:$rawFilterShape,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
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
            $filter
            (`,` $weightsTable^ `` custom<OptionalTypes>(type($weightsTable)))?
            (`,` $weight_table_data_ptr^ `` custom<OptionalTypes>(type($weight_table_data_ptr)))?
            (`,` $weight_table_sp_ptr^ `` custom<OptionalTypes>(type($weight_table_sp_ptr)))?
            (`,` $weight_table_scale^ `` custom<OptionalTypes>(type($weight_table_scale)))?
            (`,` $weight_table_bias^ `` custom<OptionalTypes>(type($weight_table_bias)))?
            (`,` $weight_zero_points^ `` custom<OptionalTypes>(type($weight_zero_points)))?
        `)`
        attr-dict
        custom<OptionalTypes>(type($input), type($filter)) ``
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
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
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

        static bool isDeprecated(mlir::Operation* op);

        static bool isSupported(vpux::IE::ConvolutionOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


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
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
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
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
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
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
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
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
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
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
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


def VPU_NCEReduceOp :
        VPU_LayerOp<
            "NCE.Reduce",
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
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
            ]
        > {
    let summary = "NCE version of Reduce layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>:$input,
        I64ArrayAttr:$axes,

        VPU_PPEAttr:$ppe,
        VPU_ReduceTypeAttr:$op_type,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, F32, quant_QuantizedType]>, VPU_DistributedTensor]>:$output
    );

    let extraClassDeclaration = [{
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem);
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output);

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

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];

    let hasVerifier = 1;
}


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
            DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
            DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
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

        static mlir::LogicalResult verifyKernel(IE::InterpolateOp origOp, Logger log = Logger::global());

        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output, Byte reservedMem);
        bool fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter, vpux::NDTypeInterface output);
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


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
            CArg<"llvm::ArrayRef<int64_t>">:$seSize,
            CArg<"int64_t">:$seDepth,
            CArg<"VPU::SEAttr">:$seAttr
        )>
    ];

    let hasCanonicalizer = 1;
}


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

#endif
