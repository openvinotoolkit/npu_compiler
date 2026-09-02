//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --vertical-fusion-tiling="workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.0097752798880849558>

// CHECK-LABEL: @VfTilingWithMultiTilingDims
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x320x64x64x!qElemType, {order = #NHWC}>
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x320x64x64xf16, {order = #NHWC}>
func.func @VfTilingWithMultiTilingDims(%arg0: tensor<1x320x64x64x!qElemType, {order = #NHWC}>, %arg1: tensor<1x320x64x64xf16, {order = #NHWC}>) -> tensor<1x320x64x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<320x320x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<320x320x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x320x64x64x!qElemType, {order = #NHWC}>, %cst as %arg3: tensor<320x320x1x1x!qElemType, {order = #NHWC}>, %arg1 as %arg5: tensor<1x320x64x64xf16, {order = #NHWC}>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 2, 4], vf_loop_index = 0} -> tensor<1x320x64x64xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg2, %arg3) rawFilterShape [320, 320, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x320x64x64x!qElemType, {order = #NHWC}>, tensor<320x320x1x1x!qElemType, {order = #NHWC}> -> tensor<1x320x64x64xf16, {order = #NHWC}>
      %2 = VPU.NCE.Eltwise(%1, %arg5) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x320x64x64xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    return %0 : tensor<1x320x64x64xf16, {order = #NHWC}>

    // CHECK:       [[INPUT0_SLICE00:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 320, 32, 16]
    // CHECK:       [[CONV_00:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE00]]
    // CHECK:       [[INPUT1_SLICE00:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_00:%.+]] = VPU.NCE.Eltwise([[CONV_00]], [[INPUT1_SLICE00]])

    // CHECK:       [[INPUT0_SLICE01:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 16] [1, 320, 32, 16]
    // CHECK:       [[CONV_01:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE01]]
    // CHECK:       [[INPUT1_SLICE01:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 16] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_01:%.+]] = VPU.NCE.Eltwise([[CONV_01]], [[INPUT1_SLICE01]])

    // CHECK:       [[INPUT0_SLICE02:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 32] [1, 320, 32, 16]
    // CHECK:       [[CONV_02:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE02]]
    // CHECK:       [[INPUT1_SLICE02:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 32] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_02:%.+]] = VPU.NCE.Eltwise([[CONV_02]], [[INPUT1_SLICE02]])

    // CHECK:       [[INPUT0_SLICE03:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 48] [1, 320, 32, 16]
    // CHECK:       [[CONV_03:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE03]]
    // CHECK:       [[INPUT1_SLICE03:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 48] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_03:%.+]] = VPU.NCE.Eltwise([[CONV_03]], [[INPUT1_SLICE03]])

    // CHECK:       [[INPUT0_SLICE10:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 32, 0] [1, 320, 32, 16]
    // CHECK:       [[CONV_10:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE10]]
    // CHECK:       [[INPUT1_SLICE10:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 32, 0] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_10:%.+]] = VPU.NCE.Eltwise([[CONV_10]], [[INPUT1_SLICE10]])

    // CHECK:       [[INPUT0_SLICE11:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 32, 16] [1, 320, 32, 16]
    // CHECK:       [[CONV_11:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE11]]
    // CHECK:       [[INPUT1_SLICE11:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 32, 16] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_11:%.+]] = VPU.NCE.Eltwise([[CONV_11]], [[INPUT1_SLICE11]])

    // CHECK:       [[INPUT0_SLICE12:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 32, 32] [1, 320, 32, 16]
    // CHECK:       [[CONV_12:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE12]]
    // CHECK:       [[INPUT1_SLICE12:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 32, 32] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_12:%.+]] = VPU.NCE.Eltwise([[CONV_12]], [[INPUT1_SLICE12]])

    // CHECK:       [[INPUT0_SLICE13:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 32, 48] [1, 320, 32, 16]
    // CHECK:       [[CONV_13:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE13]]
    // CHECK:       [[INPUT1_SLICE13:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 32, 48] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_13:%.+]] = VPU.NCE.Eltwise([[CONV_13]], [[INPUT1_SLICE13]])

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_00]], [[ELTWISE_01]], [[ELTWISE_02]], [[ELTWISE_03]], [[ELTWISE_10]], [[ELTWISE_11]], [[ELTWISE_12]], [[ELTWISE_13]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 16], [0, 0, 0, 32], [0, 0, 0, 48], [0, 0, 32, 0], [0, 0, 32, 16], [0, 0, 32, 32], [0, 0, 32, 48]]}
    // CHECK:       return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Unroll2DTilingIncludingC
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x320x64x64xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x320x64x64xf16, {order = #NHWC}>)
func.func @Unroll2DTilingIncludingC(%arg0: tensor<1x320x64x64xf16, {order = #NHWC}>, %arg1: tensor<1x320x64x64xf16, {order = #NHWC}>) -> tensor<1x320x64x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<320x320x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<320x320x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x320x64x64xf16, {order = #NHWC}>, %cst as %arg3: tensor<320x320x1x1xf16, {order = #NHWC}>, %arg1 as %arg5: tensor<1x320x64x64xf16, {order = #NHWC}>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 2, 2, 1], vf_loop_index = 0} -> tensor<1x320x64x64xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Eltwise(%arg2, %arg5) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x320x64x64xf16, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg3) rawFilterShape [320, 320, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x320x64x64xf16, {order = #NHWC}>, tensor<320x320x1x1xf16, {order = #NHWC}> -> tensor<1x320x64x64xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    return %0 : tensor<1x320x64x64xf16, {order = #NHWC}>

    //CHECK:  [[WEIGHTS:%.+]] = const.Declare tensor<320x320x1x1xf16, {order = #NHWC}>

    // Tile0
    //CHECK:  [[INPUT0_SLICE0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 320, 32, 64]
    //CHECK:  [[INPUT1_SLICE0:%.+]] = VPU.Slice [[ARG1]] [0, 0, 0, 0] [1, 320, 32, 64]
    //CHECK:  [[ELTWISE_TILE0:%.+]] = VPU.NCE.Eltwise([[INPUT0_SLICE0]], [[INPUT1_SLICE0]])
    //CHECK:  [[WEIGHTS_SLICE0:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [160, 320, 1, 1]
    //CHECK:  [[CONV_TILE0:%.+]] = VPU.NCE.Convolution([[ELTWISE_TILE0]], [[WEIGHTS_SLICE0]])

    // Tile1
    //CHECK:  [[WEIGHTS_SLICE1:%.+]] = VPU.Slice [[WEIGHTS]] [160, 0, 0, 0] [160, 320, 1, 1]
    //CHECK:  [[CONV_TILE1:%.+]] = VPU.NCE.Convolution([[ELTWISE_TILE0]], [[WEIGHTS_SLICE1]])

    // Tile2
    //CHECK:  [[INPUT0_SLICE2:%.+]] = VPU.Slice [[ARG0]] [0, 0, 32, 0] [1, 320, 32, 64]
    //CHECK:  [[INPUT1_SLICE2:%.+]] = VPU.Slice [[ARG1]] [0, 0, 32, 0] [1, 320, 32, 64]
    //CHECK:  [[ELTWISE_TILE2:%.+]] = VPU.NCE.Eltwise([[INPUT0_SLICE2]], [[INPUT1_SLICE2]])
    //CHECK:  [[WEIGHTS_TILE2:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [160, 320, 1, 1]
    //CHECK:  [[CONV_TILE2:%.+]] = VPU.NCE.Convolution([[ELTWISE_TILE2]], [[WEIGHTS_TILE2]])

    // Tile3
    //CHECK:  [[WEIGHTS_TILE3:%.+]] = VPU.Slice [[WEIGHTS]] [160, 0, 0, 0] [160, 320, 1, 1]
    //CHECK:  [[CONV_TILE3:%.+]] = VPU.NCE.Convolution([[ELTWISE_TILE2]], [[WEIGHTS_TILE3]])

    //CHECK:  [[CONCAT:%.+]] = VPU.Concat([[CONV_TILE0]], [[CONV_TILE1]], [[CONV_TILE2]], [[CONV_TILE3]])
    //CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 160, 0, 0], [0, 0, 32, 0], [0, 160, 32, 0]]}
    //CHECK:  return [[CONCAT]] : tensor<1x320x64x64xf16, {order = #NHWC}>

}

// -----

// Verify that a merged VF containing NCE.MaxPool with a reduce output
// (axes_value=[1]) followed by Subtract is correctly tiled along H: each
// tile produces a Slice -> MaxPool (main + reduce) -> Subtract sequence,
// and the results are concatenated.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolReduceOutToSubtractTiling
// CHECK-SAME:  (%[[INPUT:.*]]: tensor<1x256x128x4xf16, {order = #NHWC}>)
func.func @MaxPoolReduceOutToSubtractTiling(
        %arg0: tensor<1x256x128x4xf16, {order = #NHWC}>)
        -> tensor<1x256x128x4xf16, {order = #NHWC}> {
  %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x256x128x4xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>,
                    tilingStrategy = [1, 1, 4, 1]}
        -> tensor<1x256x128x4xf16, {order = #NHWC}> {
    %output, %reduce_xy_max = VPU.NCE.MaxPool(%arg1) {
      axes_value = [1],
      kernel_size = [1, 1],
      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64,
                         top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 1, 0, 0>,
      strides = [1, 1]
    } : tensor<1x256x128x4xf16, {order = #NHWC}>
      -> tensor<1x256x128x4xf16, {order = #NHWC}>,
         tensor<1x1x128x4xf16, {order = #NHWC}>
    %1 = VPU.Subtract(%output, %reduce_xy_max) {
      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x256x128x4xf16, {order = #NHWC}>,
        tensor<1x1x128x4xf16, {order = #NHWC}>
      -> tensor<1x256x128x4xf16, {order = #NHWC}>
    VPU.Yield %1
  }
  return %0 : tensor<1x256x128x4xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:      [[SLICE_0:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 0, 0] [1, 256, 32, 4]
    // CHECK:      [[MAXPOOL_0:%.+]], [[REDUCE_0:%.+]] = VPU.NCE.MaxPool([[SLICE_0]])
    // CHECK-SAME:   resultSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:   -> tensor<1x256x32x4xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x32x4xf16, {order = #NHWC}>
    // CHECK:      [[SUB_0:%.+]] = VPU.Subtract([[MAXPOOL_0]], [[REDUCE_0]])
    // CHECK-SAME:   -> tensor<1x256x32x4xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:      [[SLICE_1:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 32, 0] [1, 256, 32, 4]
    // CHECK:      [[MAXPOOL_1:%.+]], [[REDUCE_1:%.+]] = VPU.NCE.MaxPool([[SLICE_1]])
    // CHECK-SAME:   -> tensor<1x256x32x4xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x32x4xf16, {order = #NHWC}>
    // CHECK:      [[SUB_1:%.+]] = VPU.Subtract([[MAXPOOL_1]], [[REDUCE_1]])

    // Tile 2
    // CHECK:      [[SLICE_2:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 64, 0] [1, 256, 32, 4]
    // CHECK:      [[MAXPOOL_2:%.+]], [[REDUCE_2:%.+]] = VPU.NCE.MaxPool([[SLICE_2]])
    // CHECK-SAME:   -> tensor<1x256x32x4xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x32x4xf16, {order = #NHWC}>
    // CHECK:      [[SUB_2:%.+]] = VPU.Subtract([[MAXPOOL_2]], [[REDUCE_2]])

    // Tile 3
    // CHECK:      [[SLICE_3:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 96, 0] [1, 256, 32, 4]
    // CHECK:      [[MAXPOOL_3:%.+]], [[REDUCE_3:%.+]] = VPU.NCE.MaxPool([[SLICE_3]])
    // CHECK-SAME:   -> tensor<1x256x32x4xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x32x4xf16, {order = #NHWC}>
    // CHECK:      [[SUB_3:%.+]] = VPU.Subtract([[MAXPOOL_3]], [[REDUCE_3]])

    // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[SUB_0]], [[SUB_1]], [[SUB_2]], [[SUB_3]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}
    // CHECK-SAME:  -> tensor<1x256x128x4xf16, {order = #NHWC}>
    // CHECK:      return [[CONCAT]]
}

// -----

// Verify that a merged VF containing NCE.Convolution with a reduce output
// (axes_value=[1]) followed by SoftMax is correctly tiled along H: each
// tile produces a Slice -> Convolution (main + reduce) -> SoftMax sequence,
// and the results are concatenated.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvReduceOutToSoftMaxWithMaxTiling
// CHECK-SAME:  (%[[INPUT:.*]]: tensor<1x32x160x160xf16, {order = #NHWC}>,
// CHECK-SAME:   %[[WEIGHTS:.*]]: tensor<32x32x1x1xf16, {order = #NHWC}>)
func.func @ConvReduceOutToSoftMaxWithMaxTiling(
        %arg0: tensor<1x32x160x160xf16, {order = #NHWC}>,
        %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
        -> tensor<1x32x160x160xf16, {order = #NHWC}> {
  %0 = VPU.VerticalFusion (
        %arg0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
        %arg1 as %arg3: tensor<32x32x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>,
                    tilingStrategy = [1, 1, 4, 1]}
        -> tensor<1x32x160x160xf16, {order = #NHWC}> {
    %output, %reduce_xy_max =
      VPU.NCE.Convolution(%arg2, %arg3) rawFilterShape [32, 32, 1, 1] {
        axes_value = [1],
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64,
                           top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        resultSegmentSizes = array<i32: 1, 1, 0, 0>,
        strides = [1, 1]
      } : tensor<1x32x160x160xf16, {order = #NHWC}>,
          tensor<32x32x1x1xf16, {order = #NHWC}>
        -> tensor<1x32x160x160xf16, {order = #NHWC}>,
           tensor<1x1x160x160xf16, {order = #NHWC}>
    %1 = VPU.SoftMax(%output, %reduce_xy_max) {
      axisInd = 1 : i64,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x32x160x160xf16, {order = #NHWC}>,
        tensor<1x1x160x160xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>
    VPU.Yield %1
  }
  return %0 : tensor<1x32x160x160xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:      [[SLICE_0:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 0, 0] [1, 32, 40, 160]
    // CHECK:      [[CONV_0:%.+]], [[REDUCE_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], %[[WEIGHTS]])
    // CHECK-SAME:   resultSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x40x160xf16, {order = #NHWC}>
    // CHECK:      [[SOFTMAX_0:%.+]] = VPU.SoftMax([[CONV_0]], [[REDUCE_0]])
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:      [[SLICE_1:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 40, 0] [1, 32, 40, 160]
    // CHECK:      [[CONV_1:%.+]], [[REDUCE_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], %[[WEIGHTS]])
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x40x160xf16, {order = #NHWC}>
    // CHECK:      [[SOFTMAX_1:%.+]] = VPU.SoftMax([[CONV_1]], [[REDUCE_1]])

    // Tile 2
    // CHECK:      [[SLICE_2:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 80, 0] [1, 32, 40, 160]
    // CHECK:      [[CONV_2:%.+]], [[REDUCE_2:%.+]] = VPU.NCE.Convolution([[SLICE_2]], %[[WEIGHTS]])
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x40x160xf16, {order = #NHWC}>
    // CHECK:      [[SOFTMAX_2:%.+]] = VPU.SoftMax([[CONV_2]], [[REDUCE_2]])

    // Tile 3
    // CHECK:      [[SLICE_3:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 120, 0] [1, 32, 40, 160]
    // CHECK:      [[CONV_3:%.+]], [[REDUCE_3:%.+]] = VPU.NCE.Convolution([[SLICE_3]], %[[WEIGHTS]])
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x40x160xf16, {order = #NHWC}>
    // CHECK:      [[SOFTMAX_3:%.+]] = VPU.SoftMax([[CONV_3]], [[REDUCE_3]])

    // CHECK:      [[CONCAT:%.+]] = VPU.Concat(
    // CHECK-SAME:   [[SOFTMAX_0]], [[SOFTMAX_1]], [[SOFTMAX_2]], [[SOFTMAX_3]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 40, 0], [0, 0, 80, 0], [0, 0, 120, 0]]}
    // CHECK-SAME:  -> tensor<1x32x160x160xf16, {order = #NHWC}>
    // CHECK:      return [[CONCAT]]
}

// -----

// Verify that a merged VF containing NCE.Convolution with a reduce-min output
// (axes_value=[1], resultSegmentSizes=[1,0,1,0]) followed by two PermuteCasts
// (NHWC->NCHW) and Subtract is correctly tiled along H: each tile produces a
// Slice -> Convolution (main + reduce-min) -> PermuteCast x2 -> Subtract
// sequence, and the results are concatenated along the first spatial dimension.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvReduceOutPermuteCastToSubtractTiling
// CHECK-SAME:  (%[[INPUT:.*]]: tensor<1x32x160x160xf16, {order = #NHWC}>,
// CHECK-SAME:   %[[WEIGHTS:.*]]: tensor<32x32x1x1xf16, {order = #NHWC}>)
func.func @ConvReduceOutPermuteCastToSubtractTiling(
        %arg0: tensor<1x32x160x160xf16, {order = #NHWC}>,
        %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
        -> tensor<1x160x160x32xf16> {
  %0 = VPU.VerticalFusion (
        %arg0 as %arg2: tensor<1x32x160x160xf16, {order = #NHWC}>,
        %arg1 as %arg3: tensor<32x32x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>,
                    tilingStrategy = [1, 4, 1, 1]}
        -> tensor<1x160x160x32xf16> {
    %output, %reduce_xy_min =
      VPU.NCE.Convolution(%arg2, %arg3) rawFilterShape [32, 32, 1, 1] {
        axes_value = [1],
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64,
                           top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        resultSegmentSizes = array<i32: 1, 0, 1, 0>,
        strides = [1, 1]
      } : tensor<1x32x160x160xf16, {order = #NHWC}>,
          tensor<32x32x1x1xf16, {order = #NHWC}>
        -> tensor<1x32x160x160xf16, {order = #NHWC}>,
           tensor<1x1x160x160xf16, {order = #NHWC}>
    %1 = VPU.PermuteCast(%output) {dst_order = #NCHW, mem_perm = #NCHW}
        : tensor<1x32x160x160xf16, {order = #NHWC}> -> tensor<1x160x160x32xf16>
    %2 = VPU.PermuteCast(%reduce_xy_min) {dst_order = #NCHW, mem_perm = #NCHW}
        : tensor<1x1x160x160xf16, {order = #NHWC}> -> tensor<1x160x160x1xf16>
    %3 = VPU.Subtract(%1, %2) {
      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    } : tensor<1x160x160x32xf16>, tensor<1x160x160x1xf16>
      -> tensor<1x160x160x32xf16>
    VPU.Yield %3
  }
  return %0 : tensor<1x160x160x32xf16>

    // Tile 0
    // CHECK:      [[SLICE_0:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 0, 0] [1, 32, 40, 160]
    // CHECK:      [[CONV_0:%.+]], [[REDUCE_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], %[[WEIGHTS]])
    // CHECK-SAME:   resultSegmentSizes = array<i32: 1, 0, 1, 0>
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x40x160xf16, {order = #NHWC}>
    // CHECK:      [[PERM_ACT_0:%.+]] = VPU.PermuteCast([[CONV_0]])
    // CHECK-SAME:   -> tensor<1x40x160x32xf16>
    // CHECK:      [[PERM_RED_0:%.+]] = VPU.PermuteCast([[REDUCE_0]])
    // CHECK-SAME:   -> tensor<1x40x160x1xf16>
    // CHECK:      [[SUB_0:%.+]] = VPU.Subtract([[PERM_ACT_0]], [[PERM_RED_0]])
    // CHECK-SAME:   -> tensor<1x40x160x32xf16>

    // Tile 1
    // CHECK:      [[SLICE_1:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 40, 0] [1, 32, 40, 160]
    // CHECK:      [[CONV_1:%.+]], [[REDUCE_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], %[[WEIGHTS]])
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x40x160xf16, {order = #NHWC}>
    // CHECK:      [[PERM_ACT_1:%.+]] = VPU.PermuteCast([[CONV_1]])
    // CHECK:      [[PERM_RED_1:%.+]] = VPU.PermuteCast([[REDUCE_1]])
    // CHECK:      [[SUB_1:%.+]] = VPU.Subtract([[PERM_ACT_1]], [[PERM_RED_1]])

    // Tile 2
    // CHECK:      [[SLICE_2:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 80, 0] [1, 32, 40, 160]
    // CHECK:      [[CONV_2:%.+]], [[REDUCE_2:%.+]] = VPU.NCE.Convolution([[SLICE_2]], %[[WEIGHTS]])
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x40x160xf16, {order = #NHWC}>
    // CHECK:      [[PERM_ACT_2:%.+]] = VPU.PermuteCast([[CONV_2]])
    // CHECK:      [[PERM_RED_2:%.+]] = VPU.PermuteCast([[REDUCE_2]])
    // CHECK:      [[SUB_2:%.+]] = VPU.Subtract([[PERM_ACT_2]], [[PERM_RED_2]])

    // Tile 3
    // CHECK:      [[SLICE_3:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 120, 0] [1, 32, 40, 160]
    // CHECK:      [[CONV_3:%.+]], [[REDUCE_3:%.+]] = VPU.NCE.Convolution([[SLICE_3]], %[[WEIGHTS]])
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x40x160xf16, {order = #NHWC}>
    // CHECK:      [[PERM_ACT_3:%.+]] = VPU.PermuteCast([[CONV_3]])
    // CHECK:      [[PERM_RED_3:%.+]] = VPU.PermuteCast([[REDUCE_3]])
    // CHECK:      [[SUB_3:%.+]] = VPU.Subtract([[PERM_ACT_3]], [[PERM_RED_3]])

    // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[SUB_0]], [[SUB_1]], [[SUB_2]], [[SUB_3]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 40, 0, 0], [0, 80, 0, 0], [0, 120, 0, 0]]}
    // CHECK-SAME:  -> tensor<1x160x160x32xf16>
    // CHECK:      return [[CONCAT]]
}

// -----

// Verify that a merged VF containing Convert (f32->f16) followed by
// NCE.Convolution with a reduce-max output (axes_value=[1]) is correctly tiled
// along H with tilingStrategy=[1,1,4,1]: each tile produces a
// Slice -> Convert -> Convolution (main + reduce) sequence, and the two VF
// results are each concatenated independently — one Concat for the main
// convolution output and one for the reduce-max output.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertToConvReduceOutTiling
// CHECK-SAME:  (%[[INPUT:.*]]: tensor<1x32x160x160xf32, {order = #NHWC}>,
// CHECK-SAME:   %[[WEIGHTS:.*]]: tensor<32x32x1x1xf16, {order = #NHWC}>)
func.func @ConvertToConvReduceOutTiling(
        %arg0: tensor<1x32x160x160xf32, {order = #NHWC}>,
        %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
        -> (tensor<1x32x160x160xf16, {order = #NHWC}>,
            tensor<1x1x160x160xf16, {order = #NHWC}>) {
  %0:2 = VPU.VerticalFusion (
        %arg0 as %arg2: tensor<1x32x160x160xf32, {order = #NHWC}>,
        %arg1 as %arg3: tensor<32x32x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>,
                    tilingStrategy = [1, 1, 4, 1]}
        -> (tensor<1x32x160x160xf16, {order = #NHWC}>,
            tensor<1x1x160x160xf16, {order = #NHWC}>) {
    %cvt = VPU.Convert(%arg2) {
        dstElemType = f16,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x32x160x160xf32, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>
    %out, %rmax = VPU.NCE.Convolution(%cvt, %arg3) rawFilterShape [32, 32, 1, 1] {
      axes_value = [1],
      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 1, 0, 0>,
      strides = [1, 1]
    } : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>
    VPU.Yield %out, %rmax
  }
  return %0#0, %0#1
      : tensor<1x32x160x160xf16, {order = #NHWC}>, tensor<1x1x160x160xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:      [[SLICE_0:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 0, 0] [1, 32, 40, 160]
    // CHECK-SAME:   tensor<1x32x160x160xf32, {{.+}}> to tensor<1x32x40x160xf32, {{.+}}>
    // CHECK:      [[CVT_0:%.+]] = VPU.Convert([[SLICE_0]])
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>
    // CHECK:      [[CONV_0:%.+]], [[REDUCE_0:%.+]] = VPU.NCE.Convolution([[CVT_0]], %[[WEIGHTS]])
    // CHECK-SAME:   resultSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x40x160xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:      [[SLICE_1:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 40, 0] [1, 32, 40, 160]
    // CHECK:      [[CVT_1:%.+]] = VPU.Convert([[SLICE_1]])
    // CHECK:      [[CONV_1:%.+]], [[REDUCE_1:%.+]] = VPU.NCE.Convolution([[CVT_1]], %[[WEIGHTS]])
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x40x160xf16, {order = #NHWC}>

    // Tile 2
    // CHECK:      [[SLICE_2:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 80, 0] [1, 32, 40, 160]
    // CHECK:      [[CVT_2:%.+]] = VPU.Convert([[SLICE_2]])
    // CHECK:      [[CONV_2:%.+]], [[REDUCE_2:%.+]] = VPU.NCE.Convolution([[CVT_2]], %[[WEIGHTS]])
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x40x160xf16, {order = #NHWC}>

    // Tile 3
    // CHECK:      [[SLICE_3:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 120, 0] [1, 32, 40, 160]
    // CHECK:      [[CVT_3:%.+]] = VPU.Convert([[SLICE_3]])
    // CHECK:      [[CONV_3:%.+]], [[REDUCE_3:%.+]] = VPU.NCE.Convolution([[CVT_3]], %[[WEIGHTS]])
    // CHECK-SAME:   -> tensor<1x32x40x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x40x160xf16, {order = #NHWC}>

    // CHECK:      [[CONCAT_MAIN:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]], [[CONV_2]], [[CONV_3]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 40, 0], [0, 0, 80, 0], [0, 0, 120, 0]]}
    // CHECK-SAME:  -> tensor<1x32x160x160xf16, {order = #NHWC}>
    // CHECK:      [[CONCAT_RED:%.+]] = VPU.Concat([[REDUCE_0]], [[REDUCE_1]], [[REDUCE_2]], [[REDUCE_3]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 40, 0], [0, 0, 80, 0], [0, 0, 120, 0]]}
    // CHECK-SAME:  -> tensor<1x1x160x160xf16, {order = #NHWC}>
    // CHECK:      return [[CONCAT_MAIN]], [[CONCAT_RED]]
}

// -----

// Verify that a merged VF containing NCE.DepthConvolution with a reduce-max
// output (axes_value=[1]) followed by Subtract is correctly tiled along H:
// each tile produces Slice -> DepthConvolution (main + reduce) -> Subtract,
// and the results are concatenated.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvReduceOutToSubtractTiling
// CHECK-SAME:  (%[[INPUT:.*]]: tensor<1x32x128x4xf16, {order = #NHWC}>)
func.func @DepthConvReduceOutToSubtractTiling(
        %arg0: tensor<1x32x128x4xf16, {order = #NHWC}>)
        -> tensor<1x32x128x4xf16, {order = #NHWC}> {
  %filter = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x128x4xf16, {order = #NHWC}>,
                           %filter as %arg2: tensor<32x16x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>,
                    tilingStrategy = [1, 1, 4, 1]}
        -> tensor<1x32x128x4xf16, {order = #NHWC}> {
    %output, %reduce_xy_max = VPU.NCE.DepthConvolution(%arg1, %arg2) rawFilterShape [32, 1, 1, 1] {
      axes_value = [1],
      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64,
                         top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 1, 0, 0>,
      strides = [1, 1]
    } : tensor<1x32x128x4xf16, {order = #NHWC}>,
        tensor<32x16x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x128x4xf16, {order = #NHWC}>,
         tensor<1x1x128x4xf16, {order = #NHWC}>
    %1 = VPU.Subtract(%output, %reduce_xy_max) {
      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x32x128x4xf16, {order = #NHWC}>,
        tensor<1x1x128x4xf16, {order = #NHWC}>
      -> tensor<1x32x128x4xf16, {order = #NHWC}>
    VPU.Yield %1
  }
  return %0 : tensor<1x32x128x4xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:      [[SLICE_0:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 0, 0] [1, 32, 32, 4]
    // CHECK:      [[DW_0:%.+]], [[REDUCE_0:%.+]] = VPU.NCE.DepthConvolution([[SLICE_0]]
    // CHECK-SAME:   resultSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:   -> tensor<1x32x32x4xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x32x4xf16, {order = #NHWC}>
    // CHECK:      [[SUB_0:%.+]] = VPU.Subtract([[DW_0]], [[REDUCE_0]])
    // CHECK-SAME:   -> tensor<1x32x32x4xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:      [[SLICE_1:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 32, 0] [1, 32, 32, 4]
    // CHECK:      [[DW_1:%.+]], [[REDUCE_1:%.+]] = VPU.NCE.DepthConvolution([[SLICE_1]]
    // CHECK-SAME:   -> tensor<1x32x32x4xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x32x4xf16, {order = #NHWC}>
    // CHECK:      [[SUB_1:%.+]] = VPU.Subtract([[DW_1]], [[REDUCE_1]])

    // Tile 2
    // CHECK:      [[SLICE_2:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 64, 0] [1, 32, 32, 4]
    // CHECK:      [[DW_2:%.+]], [[REDUCE_2:%.+]] = VPU.NCE.DepthConvolution([[SLICE_2]]
    // CHECK-SAME:   -> tensor<1x32x32x4xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x32x4xf16, {order = #NHWC}>
    // CHECK:      [[SUB_2:%.+]] = VPU.Subtract([[DW_2]], [[REDUCE_2]])

    // Tile 3
    // CHECK:      [[SLICE_3:%.+]] = VPU.Slice %[[INPUT]] [0, 0, 96, 0] [1, 32, 32, 4]
    // CHECK:      [[DW_3:%.+]], [[REDUCE_3:%.+]] = VPU.NCE.DepthConvolution([[SLICE_3]]
    // CHECK-SAME:   -> tensor<1x32x32x4xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x32x4xf16, {order = #NHWC}>
    // CHECK:      [[SUB_3:%.+]] = VPU.Subtract([[DW_3]], [[REDUCE_3]])

    // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[SUB_0]], [[SUB_1]], [[SUB_2]], [[SUB_3]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}
    // CHECK-SAME:  -> tensor<1x32x128x4xf16, {order = #NHWC}>
    // CHECK:      return [[CONCAT]]
}

// -----

// Verify that a merged VF containing NCE.Eltwise with a reduce-max output
// (axes_value=[1]) followed by Subtract is correctly tiled along H:
// each tile produces Slice x2 -> Eltwise (main + reduce) -> Subtract,
// and the results are concatenated.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseReduceOutToSubtractTiling
// CHECK-SAME:  (%[[INPUT0:.*]]: tensor<1x32x128x4xf16, {order = #NHWC}>,
// CHECK-SAME:   %[[INPUT1:.*]]: tensor<1x32x128x4xf16, {order = #NHWC}>)
func.func @EltwiseReduceOutToSubtractTiling(
        %arg0: tensor<1x32x128x4xf16, {order = #NHWC}>,
        %arg1: tensor<1x32x128x4xf16, {order = #NHWC}>)
        -> tensor<1x32x128x4xf16, {order = #NHWC}> {
  %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x128x4xf16, {order = #NHWC}>,
                           %arg1 as %arg3: tensor<1x32x128x4xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>,
                    tilingStrategy = [1, 1, 4, 1]}
        -> tensor<1x32x128x4xf16, {order = #NHWC}> {
    %output, %reduce_xy_max = VPU.NCE.Eltwise(%arg2, %arg3) {
      axes_value = [1],
      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      op_type = #VPU.eltwise_type<ADD>,
      ppe = #VPU.PPEStub<>,
      resultSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<1x32x128x4xf16, {order = #NHWC}>,
        tensor<1x32x128x4xf16, {order = #NHWC}>
      -> tensor<1x32x128x4xf16, {order = #NHWC}>,
         tensor<1x1x128x4xf16, {order = #NHWC}>
    %1 = VPU.Subtract(%output, %reduce_xy_max) {
      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x32x128x4xf16, {order = #NHWC}>,
        tensor<1x1x128x4xf16, {order = #NHWC}>
      -> tensor<1x32x128x4xf16, {order = #NHWC}>
    VPU.Yield %1
  }
  return %0 : tensor<1x32x128x4xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:      [[SLICE0_0:%.+]] = VPU.Slice %[[INPUT0]] [0, 0, 0, 0] [1, 32, 32, 4]
    // CHECK:      [[SLICE1_0:%.+]] = VPU.Slice %[[INPUT1]] [0, 0, 0, 0] [1, 32, 32, 4]
    // CHECK:      [[ELT_0:%.+]], [[REDUCE_0:%.+]] = VPU.NCE.Eltwise([[SLICE0_0]], [[SLICE1_0]])
    // CHECK-SAME:   resultSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:   -> tensor<1x32x32x4xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x32x4xf16, {order = #NHWC}>
    // CHECK:      [[SUB_0:%.+]] = VPU.Subtract([[ELT_0]], [[REDUCE_0]])
    // CHECK-SAME:   -> tensor<1x32x32x4xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:      [[SLICE0_1:%.+]] = VPU.Slice %[[INPUT0]] [0, 0, 32, 0] [1, 32, 32, 4]
    // CHECK:      [[SLICE1_1:%.+]] = VPU.Slice %[[INPUT1]] [0, 0, 32, 0] [1, 32, 32, 4]
    // CHECK:      [[ELT_1:%.+]], [[REDUCE_1:%.+]] = VPU.NCE.Eltwise([[SLICE0_1]], [[SLICE1_1]])
    // CHECK-SAME:   -> tensor<1x32x32x4xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x32x4xf16, {order = #NHWC}>
    // CHECK:      [[SUB_1:%.+]] = VPU.Subtract([[ELT_1]], [[REDUCE_1]])

    // Tile 2
    // CHECK:      [[SLICE0_2:%.+]] = VPU.Slice %[[INPUT0]] [0, 0, 64, 0] [1, 32, 32, 4]
    // CHECK:      [[SLICE1_2:%.+]] = VPU.Slice %[[INPUT1]] [0, 0, 64, 0] [1, 32, 32, 4]
    // CHECK:      [[ELT_2:%.+]], [[REDUCE_2:%.+]] = VPU.NCE.Eltwise([[SLICE0_2]], [[SLICE1_2]])
    // CHECK-SAME:   -> tensor<1x32x32x4xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x32x4xf16, {order = #NHWC}>
    // CHECK:      [[SUB_2:%.+]] = VPU.Subtract([[ELT_2]], [[REDUCE_2]])

    // Tile 3
    // CHECK:      [[SLICE0_3:%.+]] = VPU.Slice %[[INPUT0]] [0, 0, 96, 0] [1, 32, 32, 4]
    // CHECK:      [[SLICE1_3:%.+]] = VPU.Slice %[[INPUT1]] [0, 0, 96, 0] [1, 32, 32, 4]
    // CHECK:      [[ELT_3:%.+]], [[REDUCE_3:%.+]] = VPU.NCE.Eltwise([[SLICE0_3]], [[SLICE1_3]])
    // CHECK-SAME:   -> tensor<1x32x32x4xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x32x4xf16, {order = #NHWC}>
    // CHECK:      [[SUB_3:%.+]] = VPU.Subtract([[ELT_3]], [[REDUCE_3]])

    // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[SUB_0]], [[SUB_1]], [[SUB_2]], [[SUB_3]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}
    // CHECK-SAME:  -> tensor<1x32x128x4xf16, {order = #NHWC}>
    // CHECK:      return [[CONCAT]]
}
