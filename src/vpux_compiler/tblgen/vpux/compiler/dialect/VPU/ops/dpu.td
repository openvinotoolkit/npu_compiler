//
// Copyright (C) 2025-2026 Intel Corporation
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
include "mlir/Interfaces/ViewLikeInterface.td"

def VPU_DPUWorkloadOp :
        VPU_Op<
            "DPU.Workload",
            [
                AttrSizedOperandSegments,
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
                    "vpux::VPU::NCEReduceOp",
                    "mlir::scf::ForOp",
                    "mlir::scf::IfOp"
                ]>
            ]
        > {
    let summary = "Workload for a single DPU tile";

    let arguments = (ins
        Variadic<Index>:$outOffsets,
        Variadic<Index>:$outSizes,
        DenseI64ArrayAttr:$static_out_offsets,
        DenseI64ArrayAttr:$static_out_sizes,

        Variadic<Index>:$inOffsets,
        Variadic<Index>:$inSizes,

        OptionalAttr<DenseI64ArrayAttr>:$static_in_offsets,
        OptionalAttr<DenseI64ArrayAttr>:$static_in_sizes,

        VPU_MPEModeAttr:$mpe_mode,

        Variadic<Index>:$pad,
        DenseI64ArrayAttr:$static_pad,
        OptionalAttr<IntAttr>:$cluster_id
    );

    let builders = [
        OpBuilder<(ins
            "mlir::ArrayRef<mlir::OpFoldResult>":$outOffsets,
            "mlir::ArrayRef<mlir::OpFoldResult>":$outSizes,
            "mlir::ArrayRef<mlir::OpFoldResult>":$pad,
            "vpux::VPU::MPEMode":$mpe_mode
        )>,

        OpBuilder<(ins
            "mlir::ArrayRef<mlir::OpFoldResult>":$outOffsets,
            "mlir::ArrayRef<mlir::OpFoldResult>":$outSizes,
            "mlir::ArrayRef<mlir::OpFoldResult>":$pad,
            "vpux::VPU::MPEModeAttr":$mpe_mode,
            "mlir::IntegerAttr":$cluster_id
        )>,

        OpBuilder<(ins
            "mlir::DenseI64ArrayAttr":$outOffsets,
            "mlir::DenseI64ArrayAttr":$outSizes,
            "VPU::PaddingAttr":$pad,
            "vpux::VPU::MPEModeAttr":$mpe_mode,
            "mlir::IntegerAttr":$cluster_id
        )>,

        OpBuilder<(ins
            "mlir::DenseI64ArrayAttr":$outOffsets,
            "mlir::DenseI64ArrayAttr":$outSizes,
            "mlir::DenseI64ArrayAttr":$inOffsets,
            "mlir::DenseI64ArrayAttr":$inSizes,
            "VPU::PaddingAttr":$pad,
            "vpux::VPU::MPEModeAttr":$mpe_mode,
            "mlir::IntegerAttr":$cluster_id
        )>,

        OpBuilder<(ins
            "mlir::DenseI64ArrayAttr":$outOffsets,
            "mlir::DenseI64ArrayAttr":$outSizes,
            "VPU::PaddingAttr":$pad,
            "vpux::VPU::MPEMode":$mpe_mode,
            "mlir::IntegerAttr":$cluster_id
        )>,

        OpBuilder<(ins
            "mlir::ArrayRef<mlir::OpFoldResult>":$outOffsets,
            "mlir::ArrayRef<mlir::OpFoldResult>":$outSizes,
            "mlir::ArrayRef<mlir::OpFoldResult>":$inOffsets,
            "mlir::ArrayRef<mlir::OpFoldResult>":$inSizes,
            "mlir::ArrayRef<mlir::OpFoldResult>":$pad,
            "vpux::VPU::MPEModeAttr":$mpe_mode,
            "mlir::IntegerAttr":$cluster_id
        )>
    ];

    let assemblyFormat = [{
        ( `inOffsets` custom<DynamicIndexList>($inOffsets, $static_in_offsets)^ )? ( `inSizes` custom<DynamicIndexList>($inSizes, $static_in_sizes)^ )? `outOffsets` custom<DynamicIndexList>($outOffsets, $static_out_offsets) `outSizes` custom<DynamicIndexList>($outSizes, $static_out_sizes) `pad` custom<DynamicIndexList>($pad, $static_pad) $mpe_mode attr-dict-with-keyword
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        VPU::PaddingAttr getPadAttribute();

        SmallVector<mlir::OpFoldResult> getMixedOutputOffsets();
        SmallVector<mlir::OpFoldResult> getMixedOutputSizes();
        std::optional<SmallVector<mlir::OpFoldResult>> getMixedInputOffsets();
        std::optional<SmallVector<mlir::OpFoldResult>> getMixedInputSizes();

        SmallVector<int64_t> getConstOutputOffsets();
        SmallVector<int64_t> getConstOutputSizes();
        std::optional<SmallVector<int64_t>> getConstInputOffsets();
        std::optional<SmallVector<int64_t>> getConstInputSizes();
    }];
}


def VPU_NCEConvolutionOp :
        VPU_LayerOp<
            "NCE.Convolution",
            [
                AttrSizedOperandSegments,
                AttrSizedResultSegments,
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getOutputTiling",
                                                                         "getMainOutputTile"]>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "isOperationSplitOverBatchCompatible",
                                                                     "doesLayerFitIntoCMX",
                                                                     "doesLayerChangeOutputAlignmentFitIntoCMX",
                                                                     "getDistributedTypeForOpOperand"]>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_ShapeInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "NCE version of Convolution layer";

    let description = [{
        NCE-mapped convolution executed on DPU.

        In addition to the primary `output` result, the op optionally produces up to three extra
        reduce results driven by the ODU per-cluster reduction logic:

        * `reduce_xy_max` — element-wise maximum over the channel axis (`axes_value = [1]` in
          NCHW/NHWC), shape `[N, 1, H, W]`.
        * `reduce_xy_min` — element-wise minimum over the same axis, same shape.
        * `reduce_tensor_min_max` — per-tensor scalar minimum/maximum (all axes), shape `[1,1,1,1]`.
          Not yet supported with tiling (E#207252).

        Post-tile reconstruction contract:
        * Tiling along H, W, or N — tile reduce outputs have distinct offsets;
          `applyTileStrategy` combines them with `VPU.Concat`.

        Multi-cluster strategies that split the channel axis - SOK - are incompatible with
        the per-cluster ODU reduce output and are rejected for now by `checkStrategyCompatibility` when any
        reduce result is present.
    }];

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[4DTensorOf<[AnyInteger, F16, BF16, vpux_QuantileType, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$filter,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weightsTable,
        Optional<AnyTypeOf<[4DTensorOf<[F32, F8E8M0FNU, F8E5M2, F8E4M3FN]>, VPU_DistributedTensor]>>:$weight_table_scale,
        Optional<AnyTypeOf<[4DTensorOf<[F32]>, VPU_DistributedTensor]>>:$weight_table_bias,
        Optional<AnyTypeOf<[4DTensorOf<[I8, I4]>, VPU_DistributedTensor]>>:$weight_zero_points,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        VPU_PPEAttrInterface:$ppe,
        OptionalAttr<VPU_MPEEngineAttrInterface>:$mpe_engine,

        Variadic<Index>:$rawFilterShape,
        DenseI64ArrayAttr:$static_raw_filter_shape,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding,
        OptionalAttr<I64ArrayAttr>:$axes_value

    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, F32, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output,
        Optional<AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_xy_max,
        Optional<AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_xy_min,
        Optional<AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_tensor_min_max
    );

    let regions = (region
        AnyRegion:$workloads
    );

   let assemblyFormat = [{
        `(`
            $input `,`
            $filter
            (`,` $weightsTable^)?
            (`,` $weight_table_scale^)?
            (`,` $weight_table_bias^)?
            (`,` $weight_zero_points^)?
        `)`
        `rawFilterShape` custom<DynamicIndexList>($rawFilterShape, $static_raw_filter_shape)
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
        bool isVFSupported();

        SmallVector<mlir::OpFoldResult> getMixedRawFilterShape();
        SmallVector<int64_t> getConstRawFilterShape();

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
                AttrSizedResultSegments,
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getOutputTiling",
                                                                         "getMainOutputTile"]>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "doesLayerFitIntoCMX",
                                                                     "doesLayerChangeOutputAlignmentFitIntoCMX",
                                                                     "getDistributedTypeForOpOperand"]>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>
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

        In addition to the primary `output` result, the op optionally produces up to three extra
        reduce results driven by the ODU per-cluster reduction logic:

        * `reduce_xy_max` — element-wise maximum over the channel axis (`axes_value = [2]` in
          GNHWC, corresponding to the C dimension in DimsGroups5D), shape `[G, N, 1, H, W]`.
        * `reduce_xy_min` — element-wise minimum over the same axis, same shape.
        * `reduce_tensor_min_max` — per-tensor scalar minimum/maximum; not yet supported with
          tiling (E#207252).

        Post-tile reconstruction contract:
        * Tiling along any axis other than C — `applyTileStrategy` combines them with `VPU.Concat`.
    }];

    let arguments = (ins
        AnyTypeOf<[5DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[5DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$weights,

        Optional<AnyTypeOf<[5DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weightsTable,
        Optional<AnyTypeOf<[5DTensorOf<[F32, F8E8M0FNU, F8E5M2, F8E4M3FN]>, VPU_DistributedTensor]>>:$weight_table_scale,
        Optional<AnyTypeOf<[5DTensorOf<[F32]>, VPU_DistributedTensor]>>:$weight_table_bias,
        Optional<AnyTypeOf<[5DTensorOf<[I8, I4]>, VPU_DistributedTensor]>>:$weight_zero_points,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        VPU_PPEAttrInterface:$ppe,
        OptionalAttr<VPU_MPEEngineAttrInterface>:$mpe_engine,
        Variadic<Index>:$rawFilterShape,
        DenseI64ArrayAttr:$static_raw_filter_shape,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<I64ArrayAttr>:$axes_value

    );

    let results = (outs
        AnyTypeOf<[5DTensorOf<[F16, BF16, F32, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output,
        Optional<AnyTypeOf<[5DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_xy_max,
        Optional<AnyTypeOf<[5DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_xy_min,
        Optional<AnyTypeOf<[5DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_tensor_min_max
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(`
            $input `,`
            $weights
            (`,` $weightsTable^ `` custom<OptionalTypes>(type($weightsTable)))?
            (`,` $weight_table_scale^ `` custom<OptionalTypes>(type($weight_table_scale)))?
            (`,` $weight_table_bias^ `` custom<OptionalTypes>(type($weight_table_bias)))?
            (`,` $weight_zero_points^ `` custom<OptionalTypes>(type($weight_zero_points)))?
        `)`
        `rawFilterShape` custom<DynamicIndexList>($rawFilterShape, $static_raw_filter_shape)
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

        DimArr restrictedFusionAxes();
        bool isVFSupported();

        SmallVector<mlir::OpFoldResult> getMixedRawFilterShape();
        SmallVector<int64_t> getConstRawFilterShape();

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
                AttrSizedResultSegments,
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getOutputTiling",
                                                                         "getMainOutputTile"]>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "doesLayerFitIntoCMX",
                                                                     "doesLayerChangeOutputAlignmentFitIntoCMX",
                                                                     "getDistributedTypeForOpOperand"]>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_ShapeInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "NCE version of Depthwise Convolution layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$filter,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weightsTable,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weight_table_data_ptr,
        Optional<AnyTypeOf<[4DTensorOf<[F32, F8E8M0FNU, F8E5M2, F8E4M3FN]>, VPU_DistributedTensor]>>:$weight_table_scale,
        Optional<AnyTypeOf<[4DTensorOf<[F32]>, VPU_DistributedTensor]>>:$weight_table_bias,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        VPU_PPEAttrInterface:$ppe,
        OptionalAttr<VPU_MPEEngineAttrInterface>:$mpe_engine,

        Variadic<Index>:$rawFilterShape,
        DenseI64ArrayAttr:$static_raw_filter_shape,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding,
        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, F32, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output,
        Optional<AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_xy_max,
        Optional<AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_xy_min,
        Optional<AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_tensor_min_max
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
            (`,` $weight_table_scale^ `` custom<OptionalTypes>(type($weight_table_scale)))?
            (`,` $weight_table_bias^ `` custom<OptionalTypes>(type($weight_table_bias)))?
        `)`
        `rawFilterShape` custom<DynamicIndexList>($rawFilterShape, $static_raw_filter_shape)
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

        DimArr restrictedFusionAxes();
        bool isVFSupported();

        SmallVector<mlir::OpFoldResult> getMixedRawFilterShape();
        SmallVector<int64_t> getConstRawFilterShape();
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
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_ShapeInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
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

        VPU_PPEAttrInterface:$ppe,
        OptionalAttr<VPU_MPEEngineAttrInterface>:$mpe_engine,

        Variadic<Index>:$rawFilterShape,
        DenseI64ArrayAttr:$static_raw_filter_shape,

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
        `rawFilterShape` custom<DynamicIndexList>($rawFilterShape, $static_raw_filter_shape)
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

        SmallVector<mlir::OpFoldResult> getMixedRawFilterShape();
        SmallVector<int64_t> getConstRawFilterShape();
    }] # baseExtraClassDeclaration;

    let elemComparisonModes = [IE_TypeComparisonMode_ALLOW_MIXED_PRECISION, IE_TypeComparisonMode_ALLOW_DIFFERENT_QUANT,
                               IE_TypeComparisonMode_ALLOW_GROUPED_OUTPUT, IE_TypeComparisonMode_ALLOW_DISTRIBUTED_OUTPUT];
}


def VPU_NCEMaxPoolOp :
        VPU_LayerOp<
            "NCE.MaxPool",
            [
                AttrSizedOperandSegments,
                AttrSizedResultSegments,
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getOutputTiling", "getMainOutputTile"]>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "isOperationSplitOverBatchCompatible",
                                                                     "doesLayerFitIntoCMX",
                                                                     "doesLayerChangeOutputAlignmentFitIntoCMX",
                                                                     "getDistributedTypeForOpOperand"]>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_ShapeInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "NCE version of MaxPool layer";

    let description = [{
        NCE-mapped max-pooling executed on DPU.

        In addition to the primary `output` result, the op optionally produces up to three extra
        reduce results driven by the ODU per-cluster reduction logic:

        * `reduce_xy_max` — element-wise maximum over the channel axis (`axes_value = [1]` in
          NCHW/NHWC), shape `[N, 1, H, W]`.
        * `reduce_xy_min` — element-wise minimum over the same axis, same shape.
        * `reduce_tensor_min_max` — per-tensor scalar minimum/maximum (all axes), shape `[1,1,1,1]`.
          Not yet supported with tiling (E#207252).

        Post-tile reconstruction contract:
        * Tiling along H, W, or N — tile reduce outputs have distinct offsets;
          `applyTileStrategy` combines them with `VPU.Concat`.

        Multi-cluster strategies that split the channel axis (SOK) are incompatible with
        the per-cluster ODU reduce output and are rejected by `checkStrategyCompatibility` when any
        reduce result is present.
    }];

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weightsTable,
        Optional<AnyTypeOf<[4DTensorOf<[F32, F8E8M0FNU, F8E5M2, F8E4M3FN]>, VPU_DistributedTensor]>>:$weight_table_scale,
        Optional<AnyTypeOf<[4DTensorOf<[F32]>, VPU_DistributedTensor]>>:$weight_table_bias,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$kernel_size,
        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        VPU_PPEAttrInterface:$ppe,
        OptionalAttr<VPU_MPEEngineAttrInterface>:$mpe_engine,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding,
        OptionalAttr<VPU_S2DD2SConfigAttr>:$s2dd2s_config,
        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, F32, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output,
        Optional<AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_xy_max,
        Optional<AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_xy_min,
        Optional<AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_tensor_min_max
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(` $input
            (`,` $weightsTable^ custom<OptionalTypes>(type($weightsTable)) ``)?
            (`,` $weight_table_scale^ `` custom<OptionalTypes>(type($weight_table_scale)))?
            (`,` $weight_table_bias^ `` custom<OptionalTypes>(type($weight_table_bias)))?
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

        DimArr restrictedFusionAxes();
        bool isVFSupported();

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
                DeclareOpInterfaceMethods<VPU_ShapeInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "NCE version of AveragePool layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        Optional<AnyTypeOf<[4DTensorOf<[F32, F8E8M0FNU, F8E5M2, F8E4M3FN]>, VPU_DistributedTensor]>>:$weight_table_scale,
        Optional<AnyTypeOf<[4DTensorOf<[F32]>, VPU_DistributedTensor]>>:$weight_table_bias,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$kernel_size,
        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,
        VPU_PaddingAttr:$pad,

        VPU_PPEAttrInterface:$ppe,
        OptionalAttr<VPU_MPEEngineAttrInterface>:$mpe_engine,

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
            $input
            (`,` $weight_table_scale^ `` custom<OptionalTypes>(type($weight_table_scale)))?
            (`,` $weight_table_bias^ `` custom<OptionalTypes>(type($weight_table_bias)))?
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
                AttrSizedOperandSegments,
                AttrSizedResultSegments,
                NoRegionArguments,
                NoTerminator,
                SingleBlock,
                VPU_EltwiseOp,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface, ["getOutputTiling",
                                                                         "getMainOutputTile"]>,
                DeclareOpInterfaceMethods<VPU_ClusteredOpInterface, ["isOperationSplitOverHeightCompatible",
                                                                     "isOperationSplitOverWidthCompatible",
                                                                     "isOperationSplitOverKernelCompatible",
                                                                     "doesLayerFitIntoCMX",
                                                                     "getDistributedTypeForOpOperand"]>,
                DeclareOpInterfaceMethods<VPU_SparseOpInterface>,
                DeclareOpInterfaceMethods<VPU_ShapeInfoOpInterface>,
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "NCE version of Eltwise layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input1,
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input2,
        Optional<AnyTypeOf<[4DTensorOf<[F32, F8E8M0FNU, F8E5M2, F8E4M3FN]>, VPU_DistributedTensor]>>:$weight_table_scale,
        Optional<AnyTypeOf<[4DTensorOf<[F32]>, VPU_DistributedTensor]>>:$weight_table_bias,

        VPU_EltwiseTypeAttr:$op_type,

        VPU_PPEAttrInterface:$ppe,
        OptionalAttr<VPU_MPEEngineAttrInterface>:$mpe_engine,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<BoolAttr>:$is_inplace,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding,
        OptionalAttr<I64ArrayAttr>:$axes_value
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, F32, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output,
        Optional<AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_xy_max,
        Optional<AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_xy_min,
        Optional<AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>>:$reduce_tensor_min_max
    );

    let regions = (region
        AnyRegion:$workloads
    );

    let assemblyFormat = [{
        `(`
            $input1
            `,` $input2
            (`,` $weight_table_scale^ `` custom<OptionalTypes>(type($weight_table_scale)))?
            (`,` $weight_table_bias^ `` custom<OptionalTypes>(type($weight_table_bias)))?
        `)`
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

        DimArr restrictedFusionAxes();
        bool isVFSupported();
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
                DeclareOpInterfaceMethods<VPU_VerticalFusionOpInterface>,
                DeclareOpInterfaceMethods<VPU_ShapeInfoOpInterface>,
                DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
            ]
        > {
    let summary = "NCE version of Reduce layer";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>:$input,
        I64ArrayAttr:$axes,

        VPU_PPEAttrInterface:$ppe,
        OptionalAttr<VPU_MPEEngineAttrInterface>:$mpe_engine,

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
            DeclareOpInterfaceMethods<VPU_ShapeInfoOpInterface>,
            DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>
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
        VPU_PPEAttrInterface:$ppe,
        OptionalAttr<VPU_MPEEngineAttrInterface>:$mpe_engine,

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
                DeclareOpInterfaceMethods<VPU_ShapeInfoOpInterface>
            ]
        > {
    let summary = "NCE version of Interpolate layer";

    let arguments = (ins
        AnyTypeOf<[VPU_SparseTensor, VPU_DistributedTensor]>:$input,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>>:$weights,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weightsTable,
        Optional<AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>>:$weight_table_data_ptr,
        Optional<AnyTypeOf<[4DTensorOf<[F32, F8E8M0FNU, F8E5M2, F8E4M3FN]>, VPU_DistributedTensor]>>:$weight_table_scale,
        Optional<AnyTypeOf<[4DTensorOf<[F32]>, VPU_DistributedTensor]>>:$weight_table_bias,

        ConfinedAttr<I64ArrayAttr, [ArrayCount<2>]>:$strides,

        VPU_PPEAttrInterface:$ppe,
        OptionalAttr<VPU_MPEEngineAttrInterface>:$mpe_engine,

        Variadic<Index>:$rawFilterShape,
        DenseI64ArrayAttr:$static_raw_filter_shape,
        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy,
        OptionalAttr<I64ArrayAttr>:$output_padding,
        OptionalAttr<I64ArrayAttr>:$input_padding,
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
            (`,` $weight_table_data_ptr^ `` custom<OptionalTypes>(type($weight_table_data_ptr)))?
            (`,` $weight_table_scale^ `` custom<OptionalTypes>(type($weight_table_scale)))?
            (`,` $weight_table_bias^ `` custom<OptionalTypes>(type($weight_table_bias)))?
        `)`
        `rawFilterShape` custom<DynamicIndexList>($rawFilterShape, $static_raw_filter_shape)
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
        custom<OptionalRegion>($workloads)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        SmallVector<mlir::OpFoldResult> getMixedRawFilterShape();
        SmallVector<int64_t> getConstRawFilterShape();

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
        OptionalAttr<VPU_SEAttrInterface>:$seAttr,
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

def VPU_ZeroPointTableOp :
        VPU_Op<
            "ZeroPointTable",
            [Pure]
        > {
    let summary = "Operation which helps to create zero-point table.";

    let description = [{
        This operation creates a zero-point table that will be converted to a constant in VPUIP dialect.
        It holds the zero-point values and metadata needed to generate
        the actual zero-point table data based on workload information.

        The operation is populated by CreateNewWeightTablesData pass.

        Arguments:
        - zero_points: Tensor of per-channel zero-point values used for quantization.
        - workloadSizes: Optional array of channel counts extracted from each workload in the
          associated NCE operation. Used to properly size and align the zero-point table entries.
        - zeroPointTableData: Optional array containing the generated zero-point table data.
    }];

    let arguments = (ins
        RankedTensorOf<[AnyInteger, F8E4M3FN, F8E5M2, F4E2M1FN]>:$zero_points,
        OptionalAttr<I64ArrayAttr>:$workloadSizes,
        OptionalAttr<I64ArrayAttr>:$zeroPointTableData
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[I8]>, 5DTensorOf<[I8]>, VPU_DistributedTensor]>:$output
    );

    let assemblyFormat = [{
        `(`$zero_points`)`
        attr-dict `:` type(operands)
        `->` type(results)
    }];

    let hasVerifier = 1;
}

def VPU_DataPointerTableOp :
        VPU_Op<
            "DataPointerTable",
            [Pure]
        > {
    let summary = "Operation which helps to create data-pointer table.";

    let description = [{
        This operation creates a data-pointer table that will be converted to a constant in VPUIP dialect.
        It holds the data-pointer values and metadata needed to generate
        the actual data-pointer table data based on workload information.

        The operation is populated by CreateNewWeightTablesData pass.

        Arguments:
        - zero_points: Optional tensor of per-channel zero-point values used for quantization.
        - workloadSizes: Optional array of channel counts extracted from each workload in the
          associated NCE operation. Used to properly size and align the data-pointer table entries.
        - dataPointerTableData: Optional array containing the generated data-pointer table data.
    }];

    let arguments = (ins
        Optional<RankedTensorOf<[AnyInteger, F8E4M3FN, F8E5M2, F4E2M1FN]>>:$zero_points,
        OptionalAttr<I64ArrayAttr>:$workloadSizes,
        OptionalAttr<I64ArrayAttr>:$dataPointerTableData
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[SI32]>, VPU_DistributedTensor]>:$output
    );

    let assemblyFormat = [{
        (`(`$zero_points^`)` `:` type($zero_points))?
        attr-dict
        `->` type(results)
    }];

    let hasVerifier = 1;
}

#endif
