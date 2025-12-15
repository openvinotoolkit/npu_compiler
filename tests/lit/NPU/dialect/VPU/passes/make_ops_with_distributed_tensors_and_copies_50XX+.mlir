//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --make-ops-with-distributed-tensor="enable-explicit-distributed-attr=true" --make-distributed-copies %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 2 of @NCE at 1.700000e+03 MHz

// Different memory offsets and shapes are generated for the output of the Convolution and the input of the Interpolate,
// which would force a spill to be preserved between them

// CHECK:       func.func @OverlappedConvToOverlappedSEPOp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x32x60x60xf16, {order = #NHWC}>
func.func @OverlappedConvToOverlappedSEPOp(%input: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x32x60x60xf16, {order = #NHWC}> {
    %conv_weights = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %conv = VPU.NCE.Convolution(%input, %conv_weights) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 16, 3, 3],
        strides = [1, 1]}
      : tensor<1x16x30x30xf16, {order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}> -> tensor<1x32x30x30xf16, {order = #NHWC}>

    %input_sparsity_map = const.Declare tensor<1x32x62x62xi1> = dense<1> : tensor<1x32x62x62xi1>
    %input_storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [32], dataShape = [1, 32, 30, 30],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 62, 62]>
    } -> tensor<1x1x62x62xi32, {order = #NHWC}>
    %input_sparse = VPU.GroupSparseTensor(%conv, %input_sparsity_map, %input_storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 62, 62]>
    } -> !VPU.SparseTensor<data=tensor<1x32x30x30xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x32x62x62xi1>,
                           storage_element_table=tensor<1x1x62x62xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 62, 62]>>

    %weights_interp = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table_interp = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %interpolate = VPU.NCE.Interpolate(%input_sparse, %weights_interp, %weights_table_interp) {
        rawFilterShape = [32, 32, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>,
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        scales_attr = [1.0, 1.0, 2.0, 2.0]
    } -> tensor<1x32x60x60xf16, {order = #NHWC}>

    return %interpolate : tensor<1x32x60x60xf16, {order = #NHWC}>

    // CHECK-DAG:    [[CONV_WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:        [[CONV_INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x30x30xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 16, 15, 30], [1, 16, 15, 30]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 16, 16, 30], [1, 16, 16, 30]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]}

    // CHECK:        [[CONV_WEIGHTS_CMX:%.+]] = VPU.Copy([[CONV_WEIGHTS]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x16x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[32, 16, 3, 3], [32, 16, 3, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[32, 16, 3, 3], [32, 16, 3, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}

    // CHECK:        [[CONV_CMX:%.+]] = VPU.NCE.Convolution(
    // CHECK-SAME:          [[CONV_INPUT_CMX]],
    // CHECK-SAME:          [[CONV_WEIGHTS_CMX]])
    // CHECK-SAME:        -> !VPU.DistributedTensor<1x32x30x30xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 15, 30], [1, 32, 15, 30]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0]],
    // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 16, 30], [1, 32, 16, 30]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]}

    // CHECK:        [[CONV_DDR:%.+]] = VPU.Copy([[CONV_CMX]]
    // CHECK-SAME:      -> tensor<1x32x30x30xf16, {order = #NHWC}>

    // CHECK-DAG:    [[INTERP_INPUT_SM:%.+]] = const.Declare tensor<1x32x62x62xi1> = dense<true> : tensor<1x32x62x62xi1>
    // CHECK:        [[INTERP_INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 30, 30],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 62, 62]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [32]}
    // CHECK-SAME:       -> tensor<1x1x62x62xi32, {order = #NHWC}>
    // CHECK:        [[INTERP_INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[CONV_DDR]], [[INTERP_INPUT_SM]], [[INTERP_INPUT_SE]])

    // CHECK-DAG:    [[INTERP_WEIGHTS:%.+]] = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[INTERP_WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    // CHECK:        [[INTER_INPUT_CMX:%.+]] = VPU.Copy([[INTERP_INPUT_SPARSE]]
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x32x30x30xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 32, 15, 30], [1, 32, 15, 30]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 32, 16, 30], [1, 32, 16, 30]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]}
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x32x62x62xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 32, 31, 62], [1, 32, 31, 62]], compute_offsets = [[0, 0, 0, 0], [0, 0, 31, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 32, 32, 62], [1, 32, 32, 62]], memory_offsets = [[0, 0, 0, 0], [0, 0, 30, 0]]}
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x62x62xi32, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 31, 62], [1, 1, 31, 62]], compute_offsets = [[0, 0, 0, 0], [0, 0, 31, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 32, 62], [1, 1, 32, 62]], memory_offsets = [[0, 0, 0, 0], [0, 0, 30, 0]]}
    // CHECK-SAME:               #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                     scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 62, 62]>>

    // CHECK:        [[INTERP_WEIGHTS_CMX:%.+]] = VPU.Copy([[INTERP_WEIGHTS]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x32x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 32, 3, 3], [32, 32, 3, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 32, 3, 3], [32, 32, 3, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}

    // CHECK:        [[INTERP_WEIGHTS_TABLE_CMX:%.+]] = VPU.Copy([[INTERP_WEIGHTS_TABLE]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}

    // CHECK:        [[INTERP_CMX:%.+]] = VPU.NCE.Interpolate(
    // CHECK-SAME:             [[INTER_INPUT_CMX]],
    // CHECK-SAME:             [[INTERP_WEIGHTS_CMX]],
    // CHECK-SAME:             [[INTERP_WEIGHTS_TABLE_CMX]]
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x32x60x60xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 30, 60], [1, 32, 30, 60]], compute_offsets = [[0, 0, 0, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 30, 60], [1, 32, 30, 60]], memory_offsets = [[0, 0, 0, 0], [0, 0, 30, 0]]}

    // CHECK:        [[INTERP_DDR:%.+]] = VPU.Copy([[INTERP_CMX]]

    // CHECK:        return [[INTERP_DDR]] : tensor<1x32x60x60xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 10.351000019148284:128>
!qElemType1 = !quant.uniform<u8:f16, 33.033453967524508:128>
!qElemType2 = !quant.uniform<u8:f16, 37.162151501225487:128>
!qElemType3 = !quant.uniform<u8:f16, 37.503749234068628:128>

module @executors {
config.Resources 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseAddMulticlusterSOHOverlappedConvolution
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x64x64x64x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[ARG1:%.+]]: tensor<1x64x64x64x!qElemType1, {order = #NHWC}>)
func.func @EltwiseAddMulticlusterSOHOverlappedConvolution(%arg0: tensor<1x64x64x64x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x64x64x!qElemType1, {order = #NHWC}>) -> tensor<1x64x130x130x!qElemType2, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
        quant_mult = [27959], quant_shift = [29], quant_post_shift = 0 : i64, in1_quant_mult = [5299], in2_quant_mult = [16913], fp_prelu_alpha = 1.000000e+00 : f64>}
        -> tensor<1x64x64x64x!qElemType3, {order = #NHWC}>

    %1 = VPU.StorageElementTable {dataElemType = !qElemType3, dataShape = [1, 64, 64, 64],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>, seDepth = 1 : i64, seSize = [64]}
        -> tensor<1x1x130x130xi32, {order = #NHWC}>
    %cst_220 = const.Declare tensor<1x64x130x130xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1> : tensor<1x64x130x130xi8>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.CastElemType<i1>]
    %2 = VPU.GroupSparseTensor(%0, %cst_220, %1) {
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>}
        -> !VPU.SparseTensor<data=tensor<1x64x64x64x!qElemType3, {order = #NHWC}>,
        sparsity_map=tensor<1x64x130x130xi1, {order = #NHWC}>,
        storage_element_table=tensor<1x1x130x130xi32, {order = #NHWC}>,
        #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>>


    %cst = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<0.200000e+00> : tensor<64x64x1x1xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [64, 64, 1, 1], strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x64x64x64x!qElemType3, {order = #NHWC}>,
        sparsity_map=tensor<1x64x130x130xi1, {order = #NHWC}>,
        storage_element_table=tensor<1x1x130x130xi32, {order = #NHWC}>,
        #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>>, tensor<64x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x130x130x!qElemType2, {order = #NHWC}>
    return %3: tensor<1x64x130x130x!qElemType2, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.+]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x64x64x64x!qElemType, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x64x64x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>
// CHECK:               [[IN_CP1:%.+]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x64x64x64x!qElemType1, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x64x64x!qElemType1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>
// CHECK:               [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[IN_CP0]], [[IN_CP1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x64x64x!qElemType3, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 11, 64], [1, 64, 12, 64], [1, 64, 12, 64], [1, 64, 12, 64], [1, 64, 11, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 21, 0], [0, 0, 32, 0], [0, 0, 43, 0], [0, 0, 54, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.Copy([[ELTWISE]]) : !VPU.DistributedTensor<1x64x64x64x!qElemType3, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 11, 64], [1, 64, 10, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 11, 64], [1, 64, 12, 64], [1, 64, 12, 64], [1, 64, 12, 64], [1, 64, 11, 64], [1, 64, 10, 64]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 21, 0], [0, 0, 32, 0], [0, 0, 43, 0], [0, 0, 54, 0]]}>
// CHECK-SAME{LITERAL}:                                    -> tensor<1x64x64x64x!qElemType3, {order = #NHWC}>
// CHECK:               [[GROUP_ST:%.+]] = VPU.GroupSparseTensor([[OUT_CP]]
// CHECK:               [[IN_CONV:%.+]] = VPU.Copy([[GROUP_ST]]
// CHECK:               [[CONV:%.+]] = VPU.NCE.Convolution([[IN_CONV]]
// CHECK:               [[OUT_CONV:%.+]] = VPU.Copy([[CONV]]
// CHECK:               return [[OUT_CONV]] : tensor<1x64x130x130x!qElemType2, {order = #NHWC}>

}

}
