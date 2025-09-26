//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --tiling-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @executors {
    config.Resources 3 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    // CHECK-LABEL:   @MultiplyPipeliningTiling
    // CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x1024x1x3072xf16>,
    // CHECK-SAME:     [[INPUT1:%.+]]: tensor<1x1x1x3072xf16>)
    func.func @MultiplyPipeliningTiling(%arg0: tensor<1x1024x1x3072xf16>, %arg1: tensor<1x1x1x3072xf16>) -> tensor<1x1024x1x3072xf16> {
        %multiply = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} :
            tensor<1x1024x1x3072xf16>, tensor<1x1x1x3072xf16> -> tensor<1x1024x1x3072xf16>
        return %multiply : tensor<1x1024x1x3072xf16>

    // pipelining tiling is disabled for Multiply operation
    // CHECK:      VPU.Multiply
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 5]
    // CHECK-NOT:       tilingStrategy = [1, 1, 1, 6]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0043085547638874429:24>

// CHECK-LABEL: @DontTileD2SDMA
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x64x128x128x!qElemType, {order = #NHWC}>
func.func @DontTileD2SDMA(%arg0: tensor<1x64x128x128x!qElemType, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]}
            -> tensor<1x64x128x128x!qElemType, {order = #NHWC}>
    %d2s = VPU.DepthToSpace(%avgpool) {
        block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 2, 1]} : tensor<1x64x128x128x!qElemType, {order = #NHWC}>
            -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    %eltwise = VPU.NCE.Eltwise(%d2s, %d2s) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>,
        tilingStrategy = [1, 1, 2, 1]}
            -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    return %eltwise : tensor<1x16x256x256x!qElemType, {order = #NHWC}>


    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]])
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[AVGPOOL]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 1]
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[D2S]], [[D2S]])
    // CHECK:       return [[ELTWISE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0043085547638874429:24>

// CHECK-LABEL: @DontTileD2SDMAWithSlice
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x64x128x129x!qElemType, {order = #NHWC}>
func.func @DontTileD2SDMAWithSlice(%arg0: tensor<1x64x128x129x!qElemType, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]}
            -> tensor<1x64x128x129x!qElemType, {order = #NHWC}>
    %slice = VPU.Slice %avgpool [0, 0, 0, 0] [1, 64, 128, 128] : tensor<1x64x128x129x!qElemType, {order = #NHWC}> to tensor<1x64x128x128x!qElemType, {order = #NHWC}>
    %d2s = VPU.DepthToSpace(%slice) {
        block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 2, 1]} : tensor<1x64x128x128x!qElemType, {order = #NHWC}>
            -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    %eltwise = VPU.NCE.Eltwise(%d2s, %d2s) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>,
        tilingStrategy = [1, 1, 2, 1]}
            -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    return %eltwise : tensor<1x16x256x256x!qElemType, {order = #NHWC}>


    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]])
    // CHECK:       [[SLICE:%.+]] = VPU.Slice [[AVGPOOL]]
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[SLICE]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 1]
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[D2S]], [[D2S]])
    // CHECK:       return [[ELTWISE]]
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @NCEMatMulSOGAndGTile
func.func @NCEMatMulSOGAndGTile(%arg0: tensor<64x8x64x32xf16>, %arg1: tensor<64x8x64x32xf16>) -> tensor<512x1x64x64x1xf16, {order = #GNHWC}> {
  %cst_0 = const.Declare tensor<512x64x1x1x4xsi32> = dense<10> : tensor<512x64x1x1x4xsi32>
  %0 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg0 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %1 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg1 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<512x64x32x1x1xf16> -> tensor<512x1x32x64x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<512x64x32x1x1xf16> -> tensor<512x64x32x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [512, 1, 32, 16, 4]} : tensor<512x1x32x64x1xf16, {order = #GNHWC}> -> tensor<512x1x32x16x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst_0) {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [512, 64, 32, 1, 1], strides = [1, 1], tilingStrategy = [2, 1, 1, 1, 1]} -> tensor<512x1x64x16x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [512, 1, 64, 64, 1]} : tensor<512x1x64x16x4xf16, {order = #GNHWC}> -> tensor<512x1x64x64x1xf16, {order = #GNHWC}>
  return %8 : tensor<512x1x64x64x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [2, 1, 1, 1, 1]
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @NCEMatMulSOGAndHTile
func.func @NCEMatMulSOGAndHTile(%arg0: tensor<6x1x512x512xf16>, %arg1: tensor<6x1x512x512xf16>) -> tensor<6x1x512x512x1xf16, {order = #GNHWC}> {
  %cst = const.Declare tensor<6x512x1x1x4xsi32> = dense<10> : tensor<6x512x1x1x4xsi32>
  %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 6, 512, 512]} : tensor<6x1x512x512xf16> -> tensor<1x6x512x512xf16>
  %1 = VPU.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 6, 512, 512]} : tensor<6x1x512x512xf16> -> tensor<1x6x512x512xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [6, 512, 512, 1, 1]} : tensor<1x6x512x512xf16> -> tensor<6x512x512x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<6x512x512x1x1xf16> -> tensor<6x1x512x512x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [6, 512, 512, 1, 1]} : tensor<1x6x512x512xf16> -> tensor<6x512x512x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<6x512x512x1x1xf16> -> tensor<6x512x512x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [6, 1, 512, 128, 4]} : tensor<6x1x512x512x1xf16, {order = #GNHWC}> -> tensor<6x1x512x128x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [6, 512, 512, 1, 1], strides = [1, 1]} -> tensor<6x1x512x128x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [6, 1, 512, 512, 1]} : tensor<6x1x512x128x4xf16, {order = #GNHWC}> -> tensor<6x1x512x512x1xf16, {order = #GNHWC}>
  return %8 : tensor<6x1x512x512x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [1, 1, 1, 3, 1]
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @NCEMatMulSOGAndGHTile
func.func @NCEMatMulSOGAndGHTile(%arg0: tensor<12x1x512x512xf16>, %arg1: tensor<12x1x512x512xf16>) -> tensor<12x1x512x512x1xf16, {order = #GNHWC}> {
  %cst = const.Declare tensor<12x512x1x1x4xsi32> = dense<10> : tensor<12x512x1x1x4xsi32>
  %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 12, 512, 512]} : tensor<12x1x512x512xf16> -> tensor<1x12x512x512xf16>
  %1 = VPU.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 12, 512, 512]} : tensor<12x1x512x512xf16> -> tensor<1x12x512x512xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [12, 512, 512, 1, 1]} : tensor<1x12x512x512xf16> -> tensor<12x512x512x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<12x512x512x1x1xf16> -> tensor<12x1x512x512x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [12, 512, 512, 1, 1]} : tensor<1x12x512x512xf16> -> tensor<12x512x512x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<12x512x512x1x1xf16> -> tensor<12x512x512x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [12, 1, 512, 128, 4]} : tensor<12x1x512x512x1xf16, {order = #GNHWC}> -> tensor<12x1x512x128x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [12, 512, 512, 1, 1], strides = [1, 1]} -> tensor<12x1x512x128x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [12, 1, 512, 512, 1]} : tensor<12x1x512x128x4xf16, {order = #GNHWC}> -> tensor<12x1x512x512x1xf16, {order = #GNHWC}>
  return %8 : tensor<12x1x512x512x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [1, 1, 1, 16, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
  config.Resources 4 of @NCE at 1.850000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }

  // CHECK-LABEL: @pipeliningTilingForBigFilter
  func.func @pipeliningTilingForBigFilter(%arg0: tensor<1x1536x1x1xf16, {order = #NHWC}>, %arg1: tensor<8160x1536x1x1x!quant.uniform<i8:f16, 1.000000e+00>, {order = #NHWC}>) -> tensor<1x8160x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<8160x1x1x4xsi32> = dense<10> : tensor<8160x1x1x4xsi32>
    %310 = VPU.NCE.Convolution(%arg0, %arg1, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [8160, 1536, 1, 1], strides = [1, 1]} : tensor<1x1536x1x1xf16, {order = #NHWC}>, tensor<8160x1536x1x1x!quant.uniform<i8:f16, 1.000000e+00>, {order = #NHWC}>, tensor<8160x1x1x4xsi32> -> tensor<1x8160x1x1xf16, {order = #NHWC}>
    return %310 : tensor<1x8160x1x1xf16, {order = #NHWC}>

    // CHECK:         VPU.NCE.Convolution
    // CHECK-SAME:    tilingStrategy = [1, 6, 1, 1]
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL:   @SplitNCEConvOverOH
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x48xf16, {order = #NHWC}>
    func.func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x48xf16, {order = #NHWC}>) -> tensor<1x256x64x48xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [256, 32, 3, 3],
            strides = [1, 1]
        } : tensor<1x32x64x48xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x64x48xf16, {order = #NHWC}>

        return %0 : tensor<1x256x64x48xf16, {order = #NHWC}>

        // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

        // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1>
        // CHECK-SAME:      : tensor<256x1x1x4xsi32>

        // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:          {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        // CHECK-SAME:          ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
        // CHECK-SAME:          -> tensor<1x256x64x48xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x256x64x48xf16, {order = #NHWC}>
    }

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.3385416666666667>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL:   @SplitI4QuantNCEConvOverOC
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x128x256x4xf16, {order = #NHWC}>
    func.func @SplitI4QuantNCEConvOverOC(%arg0: tensor<1x128x256x4xf16, {order = #NHWC}>) -> tensor<1x6320x256x4xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<6320x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<6320x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<6320x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<6320x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [6320, 128, 1, 1], strides = [1, 1]
        } : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<6320x128x1x1x!qElemType, {order = #NHWC}>, tensor<6320x1x1x4xsi32, {order = #NCHW}> -> tensor<1x6320x256x4xf16, {order = #NHWC}>

        return %0 : tensor<1x6320x256x4xf16, {order = #NHWC}>

        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<6320x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-SAME:      : tensor<6320x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

        // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<6320x1x1x4xsi32, {order = #NCHW}> = dense<10>
        // CHECK-SAME:      : tensor<6320x1x1x4xsi32>

        // CHECK:           [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        // CHECK-SAME:          rawFilterShape = [6320, 128, 1, 1],
        // CHECK-SAME:          strides = [1, 1],
        // CHECK-SAME:          tilingStrategy = [1, 3, 1, 1]}
        // CHECK-SAME:          -> tensor<1x6320x256x4xf16, {order = #NHWC}>

        // CHECK:           return [[CONV]] : tensor<1x6320x256x4xf16, {order = #NHWC}>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @TileOverCWithBigC
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x1024x4x4xf16, {order = #NHWC}>
    func.func @TileOverCWithBigC(
            %arg0: tensor<1x1024x4x4xf16, {order = #NHWC}>)
                -> tensor<1x8016x4x4xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<8016x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
            : tensor<8016x1024x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<8016x1x1x4xsi32> = dense<1>
            : tensor<8016x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [8016, 1024, 1, 1],
            strides = [1, 1]
        } : tensor<1x1024x4x4xf16, {order = #NHWC}>, tensor<8016x1024x1x1xf16, {order = #NHWC}>, tensor<8016x1x1x4xsi32> -> tensor<1x8016x4x4xf16, {order = #NHWC}>

        return %0 : tensor<1x8016x4x4xf16, {order = #NHWC}>

    // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<8016x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<8016x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<8016x1x1x4xsi32> = dense<1>
    // CHECK-SAME:      : tensor<8016x1x1x4xsi32>

    // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>, rawFilterShape = [8016, 1024, 1, 1], strides = [1, 1], tilingStrategy = [1, 28, 1, 1]}
    // CHECK-SAME:          -> tensor<1x8016x4x4xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x8016x4x4xf16, {order = #NHWC}>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitNCEPoolOverH
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x340x256xf16, {order = #NHWC}>)
    func.func @SplitNCEPoolOverH(%arg0: tensor<1x16x340x256xf16, {order = #NHWC}>) -> tensor<1x16x340x256xf16, {order = #NHWC}> {
        %0 = VPU.NCE.MaxPool(%arg0) {
            kernel_size = [3, 3],
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } -> tensor<1x16x340x256xf16, {order = #NHWC}>

        return %0 : tensor<1x16x340x256xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {
        // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
        // CHECK-SAME:      tilingStrategy = [1, 1, 7, 1]
        // CHECK-SAME:      } -> tensor<1x16x340x256xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x16x340x256xf16, {order = #NHWC}>
    }

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @TileWithSOK
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x30x30xf16, {order = #NHWC}>
    func.func @TileWithSOK(
            %arg0: tensor<1x32x30x30xf16, {order = #NHWC}>)
                -> tensor<1x768x30x30xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
            : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<768x1x1x4xsi32> = dense<1>
            : tensor<768x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [768, 32, 7, 7],
            strides = [1, 1]
        } : tensor<1x32x30x30xf16, {order = #NHWC}>, tensor<768x32x7x7xf16, {order = #NHWC}>, tensor<768x1x1x4xsi32> -> tensor<1x768x30x30xf16, {order = #NHWC}>

        return %0 : tensor<1x768x30x30xf16, {order = #NHWC}>

        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-SAME:          : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]
        // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<768x1x1x4xsi32> = dense<1>
        // CHECK-SAME:          : tensor<768x1x1x4xsi32>

        // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        // CHECK-SAME:          pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>
        // CHECK-SAME:          rawFilterShape = [768, 32, 7, 7],
        // CHECK-SAME:          strides = [1, 1],
        // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]}
        // CHECK-SAME:        -> tensor<1x768x30x30xf16, {order = #NHWC}>

        // CHECK:       return [[CONV1]] : tensor<1x768x30x30xf16, {order = #NHWC}>
    }

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL:   @SplitSparseNCEConvOverOH
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x60xf16, {order = #NHWC}>
    func.func @SplitSparseNCEConvOverOH(%arg0: tensor<1x32x80x60xf16, {order = #NHWC}>) -> tensor<1x160x80x60xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
        %weights_sm = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
        %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
            -> !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>
        %weights_table = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<160x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [160, 32, 3, 3],
            strides = [1, 1]
        } : tensor<1x32x80x60xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>, tensor<160x1x1x4xsi32, {order = #NCHW}> -> tensor<1x160x80x60xf16, {order = #NHWC}>

        return %0 : tensor<1x160x80x60xf16, {order = #NHWC}>

        // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

        // CHECK-DAG:        [[WEIGHTS_SM:%.+]] = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>
        // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

        // CHECK:        [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor
        // CHECK-SAME:       data=tensor<160x32x3x3xf16, {order = #NHWC}>,
        // CHECK-SAME:       sparsity_map=tensor<160x1x1x384xi1>, is_weights

        // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10>
        // CHECK-SAME:      : tensor<160x1x1x4xsi32>

        // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SPARSE]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
        // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3]
        // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
        // CHECK-SAME:          -> tensor<1x160x80x60xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x160x80x60xf16, {order = #NHWC}>
    }

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL:   @SplitSparseNCEConvOverOH
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x60xf16, {order = #NHWC}>
    func.func @SplitSparseNCEConvOverOH(%arg0: tensor<1x32x80x60xf16, {order = #NHWC}>) -> tensor<1x160x80x60xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
        %weights_sm = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
        %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
            -> !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>
        %weights_table = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<160x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [160, 32, 3, 3],
            strides = [1, 1]
        } : tensor<1x32x80x60xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>, tensor<160x1x1x4xsi32, {order = #NCHW}> -> tensor<1x160x80x60xf16, {order = #NHWC}>

        return %0 : tensor<1x160x80x60xf16, {order = #NHWC}>

        // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

        // CHECK-DAG:        [[WEIGHTS_SM:%.+]] = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>
        // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

        // CHECK:        [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor
        // CHECK-SAME:       data=tensor<160x32x3x3xf16, {order = #NHWC}>,
        // CHECK-SAME:       sparsity_map=tensor<160x1x1x384xi1>, is_weights

        // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10>
        // CHECK-SAME:      : tensor<160x1x1x4xsi32>

        // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SPARSE]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
        // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3]
        // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
        // CHECK-SAME:          -> tensor<1x160x80x60xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x160x80x60xf16, {order = #NHWC}>
    }

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitNCEAveragePoolOverW
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x8640xf16, {order = #NHWC}>
    func.func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x8640xf16, {order = #NHWC}>) -> tensor<1x16x1x8640xf16, {order = #NHWC}> {
        %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x1x8640xf16, {order = #NHWC}>
        return %0 : tensor<1x16x1x8640xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [7, 1]
        // CHECK-SAME:      tilingStrategy = [1, 1, 1, 4]
        // CHECK-SAME:      -> tensor<1x16x1x8640xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x16x1x8640xf16, {order = #NHWC}>
    }

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL:   @SplitNCECompressConv
    func.func @SplitNCECompressConv(
            %arg0: tensor<1x4x512x512xf16, {order = #NHWC}>,
            %arg1: tensor<64x1x1x160xf16, {order = #NHWC}>,
            %arg2: tensor<64x1x1x4xsi32>)
            -> tensor<1x64x256x256xf16, {order = #NHWC}> {
        %0 = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
            cm_sp_pattern = 15 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
            pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [64, 4, 7, 7], strides = [2, 2]
        } -> tensor<1x64x256x256xf16, {order = #NHWC}>

        return %0 : tensor<1x64x256x256xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
        // CHECK-SAME:      cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        // CHECK-SAME:      pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
        // CHECK-SAME:      rawFilterShape = [64, 4, 7, 7], strides = [2, 2], tilingStrategy = [1, 1, 2, 1]}
        // CHECK-SAME:      -> tensor<1x64x256x256xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x64x256x256xf16, {order = #NHWC}>
    }

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.0>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @PrefetchTilingWithSOHParentConsidered
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x512x14x14x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<512x512x3x3x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<2048x512x1x1x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS_TABLE1:%arg[0-9]]]: tensor<512x1x1x4xsi32, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS_TABLE2:%arg[0-9]]]: tensor<2048x1x1x4xsi32, {order = #NHWC}>
    func.func @PrefetchTilingWithSOHParentConsidered(
            %input: tensor<1x512x14x14x!qElemType, {order = #NHWC}>,
            %weights1: tensor<512x512x3x3x!qElemType, {order = #NHWC}>,
            %weights2: tensor<2048x512x1x1x!qElemType, {order = #NHWC}>,
            %weights_table1: tensor<512x1x1x4xsi32, {order = #NHWC}>,
            %weights_table2: tensor<2048x1x1x4xsi32, {order = #NHWC}>)
                -> tensor<1x2048x7x7x!qElemType, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights1, %weights_table1) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [512, 512, 3, 3], strides = [2, 2]}
                : tensor<1x512x14x14x!qElemType, {order = #NHWC}>, tensor<512x512x3x3x!qElemType, {order = #NHWC}>, tensor<512x1x1x4xsi32, {order = #NHWC}> -> tensor<1x512x7x7x!qElemType, {order = #NHWC}>
        %1 = VPU.NCE.Convolution(%0, %weights2, %weights_table2) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [2048, 512, 1, 1], strides = [1, 1]}
                : tensor<1x512x7x7x!qElemType, {order = #NHWC}>, tensor<2048x512x1x1x!qElemType, {order = #NHWC}>, tensor<2048x1x1x4xsi32, {order = #NHWC}> -> tensor<1x2048x7x7x!qElemType, {order = #NHWC}>
        return %1 : tensor<1x2048x7x7x!qElemType, {order = #NHWC}>

        // Prefetching mode is triggered for the child conv
        // with tiled parent memory considered
        // CHECK:       [[PARENT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
        // CHECK-SAME:  tilingStrategy = [1, 4, 1, 1]
        // CHECK:       [[CHILD:%.+]] = VPU.NCE.Convolution([[PARENT]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
        // CHECK-SAME:  tilingStrategy = [1, 2, 1, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  @DepthToSpacePrefetchNotPossible
func.func @DepthToSpacePrefetchNotPossible(%arg0: tensor<1x8x27x40xf16>) -> tensor<1x256x54x80xf16> {
  %cst = const.Declare tensor<1024x1x1x4xsi32> = dense<0> : tensor<1024x1x1x4xsi32>
  %cst_0 = const.Declare tensor<1024x16x3x3xf16, {order = #NHWC}> = dense<0.0> : tensor<1024x16x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
  %cst_1 = const.Declare tensor<1024x1x1x256xi1> = dense<0.0> : tensor<1024x16x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
  %0 = VPU.GroupSparseTensor(%cst_0, %cst_1) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<1024xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<1024x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<1024x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<1024xi64>, alignment = 16 : i64>>
  %1 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 8]} : tensor<1x8x27x40xf16> -> tensor<1x8x27x48xf16>
  %2 = VPU.NCE.Permute(%1) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x27x48xf16, {order = #NHWC}>
  %3 = VPU.Slice %2 [0, 0, 0, 0] [1, 16, 27, 40] : tensor<1x16x27x48xf16, {order = #NHWC}> to tensor<1x16x27x40xf16, {order = #NHWC}>
  %4 = VPU.NCE.Convolution(%3, %0, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [1024, 16, 3, 3], strides = [1, 1]} : tensor<1x16x27x40xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<1024x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<1024x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<1024xi64>, alignment = 16 : i64>>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x27x40xf16, {order = #NHWC}>
  %5 = VPU.DepthToSpace(%4) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x27x40xf16, {order = #NHWC}> -> tensor<1x256x54x80xf16, {order = #NHWC}>
  %6 = VPU.NCE.MaxPool(%5) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x256x54x80xf16>
  return %6 : tensor<1x256x54x80xf16>

  // CHECK:       VPU.DepthToSpace
  // CHECK-SAME:  tilingStrategy = [1, 1, 1, 1]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:  @DepthToSpacefillDividedTilesCorrectly
func.func @DepthToSpacefillDividedTilesCorrectly(%arg0: tensor<1x8x184x40xf16>) -> tensor<1x200x368x80xf16> {
  %cst = const.Declare tensor<3200x1x1x4xsi32> = dense<0> : tensor<3200x1x1x4xsi32>
  %cst_0 = const.Declare tensor<3200x32x3x3xf16, {order = #NHWC}> = dense<0.0> : tensor<3200x32x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
  %cst_1 = const.Declare tensor<3200x1x1x384xi1> = dense<0.0> : tensor<3200x32x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
  %0 = VPU.GroupSparseTensor(%cst_0, %cst_1) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<3200xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<3200x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<3200x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<3200xi64>, alignment = 16 : i64>>
  %1 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 8]} : tensor<1x8x184x40xf16> -> tensor<1x8x184x48xf16>
  %2 = VPU.NCE.Permute(%1) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 8 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x8x184x48xf16, {order = #NHWC}>
  %3 = VPU.Slice %2 [0, 0, 0, 0] [1, 8, 184, 40] : tensor<1x8x184x48xf16, {order = #NHWC}> to tensor<1x8x184x40xf16, {order = #NHWC}>
  %4 = VPU.ShapeCast {shape = [1, 32, 184, 10]} inputs(%3 : tensor<1x8x184x40xf16, {order = #NHWC}>) -> tensor<1x32x184x10xf16, {order = #NHWC}>
  %5 = VPU.NCE.Convolution(%4, %0, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [3200, 32, 3, 3], strides = [1, 1]} : tensor<1x32x184x10xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<3200x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<3200x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<3200xi64>, alignment = 16 : i64>>, tensor<3200x1x1x4xsi32> -> tensor<1x3200x184x10xf16, {order = #NHWC}>
  %6 = VPU.ShapeCast {shape = [1, 800, 184, 40]} inputs(%5 : tensor<1x3200x184x10xf16, {order = #NHWC}>) -> tensor<1x800x184x40xf16, {order = #NHWC}>
  %7 = VPU.DepthToSpace(%6) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x800x184x40xf16, {order = #NHWC}> -> tensor<1x200x368x80xf16, {order = #NHWC}>
  %8 = VPU.ShapeCast {shape = [1, 16, 368, 1000]} inputs(%7 : tensor<1x200x368x80xf16, {order = #NHWC}>) -> tensor<1x16x368x1000xf16, {order = #NHWC}>
  %9 = VPU.NCE.MaxPool(%8) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x16x368x1000xf16, {order = #NWCH}>
  %10 = VPU.LayoutCast(%9) {dst_order = #NHWC} : tensor<1x16x368x1000xf16, {order = #NWCH}> -> tensor<1x16x368x1000xf16, {order = #NHWC}>
  %11 = VPU.ShapeCast {shape = [1, 16, 80, 4600]} inputs(%10 : tensor<1x16x368x1000xf16, {order = #NHWC}>) -> tensor<1x16x80x4600xf16, {order = #NHWC}>
  %12 = VPU.NCE.MaxPool(%11) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x16x80x4600xf16, {order = #NWCH}>
  %13 = VPU.PermuteCast(%12) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x80x4600xf16, {order = #NWCH}> -> tensor<1x4600x16x80xf16>
  %14 = VPU.ShapeCast {shape = [1, 200, 368, 80]} inputs(%13 : tensor<1x4600x16x80xf16>) -> tensor<1x200x368x80xf16>
  return %14 : tensor<1x200x368x80xf16>

  // CHECK:       VPU.DepthToSpace
  // CHECK-SAME:  tilingStrategy = [1, 1, 3, 1]
}
