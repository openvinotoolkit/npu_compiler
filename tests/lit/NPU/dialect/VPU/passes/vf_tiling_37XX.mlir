//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --vertical-fusion-tiling %s | FileCheck %s
// REQUIRES: platform-NPU3720


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @VfTilingDepthConvAndConv
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x576x65x65xf16, {order = #NHWC}>)
func.func @VfTilingDepthConvAndConv(%arg0: tensor<1x576x65x65xf16, {order = #NHWC}>) -> tensor<1x160x65x65xf16, {order = #NHWC}>   {
    %cst_0 = const.Declare tensor<576x32x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<576x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<576x1x1x4xsi32> = dense<1> : tensor<576x1x1x4xsi32>
    %cst_3 = const.Declare tensor<160x576x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<160x576x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<160x1x1x4xsi32> = dense<0> : tensor<160x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x576x65x65xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<576x32x1x1xf16, {order = #NHWC}>, %cst_1 as %arg3: tensor<576x1x1x4xsi32>, %cst_3 as %arg4: tensor<160x576x1x1xf16, {order = #NHWC}>, %cst_4 as %arg5: tensor<160x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 6, 1], vf_loop_index = 0} -> tensor<1x160x65x65xf16, {order = #NHWC}> {
      %1 = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3) rawFilterShape [576, 1, 5, 5] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
         pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
         ppe = #VPU.PPEStub<>,  strides = [1, 1]}
         -> tensor<1x576x65x65xf16, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4, %arg5) rawFilterShape [160, 576, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEStub<>,
          strides = [1, 1]} : tensor<1x576x65x65xf16, {order = #NHWC}>, tensor<160x576x1x1xf16, {order = #NHWC}>, tensor<160x1x1x4xsi32> -> tensor<1x160x65x65xf16, {order = #NHWC}>
      VPU.Yield %2
    }

    return %0 : tensor<1x160x65x65xf16, {order = #NHWC}>

    // CHECK-DAG:  [[W_DWCONV:%.+]] = const.Declare tensor<576x32x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:  [[WT_DWCONV:%.+]] = const.Declare tensor<576x1x1x4xsi32>
    // CHECK-DAG:  [[W_CONV:%.+]] = const.Declare tensor<160x576x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:  [[WT_CONV:%.+]] = const.Declare tensor<160x1x1x4xsi32>

    // CHECK: [[SLICEARG0TILE0:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 0, 0] [1, 576, 13, 65]
    // CHECK: [[DEPTHCONVTILE0:%.+]] = VPU.NCE.DepthConvolution([[SLICEARG0TILE0]], [[W_DWCONV]], [[WT_DWCONV]]) rawFilterShape [576, 1, 5, 5]
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], vf_loop_index = 0 : i64, vf_loop_tile_index = 0 : i64} -> tensor<1x576x11x65xf16, {order = #NHWC}>
    // CHECK: [[CONVTILE0:%.+]] = VPU.NCE.Convolution([[DEPTHCONVTILE0]], [[W_CONV]], [[WT_CONV]]) rawFilterShape [160, 576, 1, 1]
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], vf_loop_index = 0 : i64, vf_loop_tile_index = 0 : i64}
    // CHECK-SAME: -> tensor<1x160x11x65xf16, {order = #NHWC}>
    // CHECK: [[SLICEARG0TILE1:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 9, 0] [1, 576, 15, 65]
    // CHECK: [[DEPTHCONVTILE1:%.+]] = VPU.NCE.DepthConvolution([[SLICEARG0TILE1]], [[W_DWCONV]], [[WT_DWCONV]]) rawFilterShape [576, 1, 5, 5]
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], vf_loop_index = 0 : i64, vf_loop_tile_index = 1 : i64} -> tensor<1x576x11x65xf16, {order = #NHWC}>
    // CHECK: [[CONVTILE1:%.+]] = VPU.NCE.Convolution([[DEPTHCONVTILE1]], [[W_CONV]], [[WT_CONV]]) rawFilterShape [160, 576, 1, 1]
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], vf_loop_index = 0 : i64, vf_loop_tile_index = 1 : i64}
    // CHECK-SAME: -> tensor<1x160x11x65xf16, {order = #NHWC}>
    // CHECK: [[SLICEARG0TILE2:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 20, 0] [1, 576, 15, 65]
    // CHECK: [[DEPTHCONVTILE2:%.+]] = VPU.NCE.DepthConvolution([[SLICEARG0TILE2]], [[W_DWCONV]], [[WT_DWCONV]]) rawFilterShape [576, 1, 5, 5]
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], vf_loop_index = 0 : i64, vf_loop_tile_index = 2 : i64} -> tensor<1x576x11x65xf16, {order = #NHWC}>
    // CHECK: [[CONVTILE2:%.+]] = VPU.NCE.Convolution([[DEPTHCONVTILE2]], [[W_CONV]], [[WT_CONV]]) rawFilterShape [160, 576, 1, 1]
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], vf_loop_index = 0 : i64, vf_loop_tile_index = 2 : i64}
    // CHECK-SAME: -> tensor<1x160x11x65xf16, {order = #NHWC}>
    // CHECK: [[SLICEARG0TILE3:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 31, 0] [1, 576, 15, 65]
    // CHECK: [[DEPTHCONVTILE3:%.+]] = VPU.NCE.DepthConvolution([[SLICEARG0TILE3]], [[W_DWCONV]], [[WT_DWCONV]]) rawFilterShape [576, 1, 5, 5]
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], vf_loop_index = 0 : i64, vf_loop_tile_index = 3 : i64} -> tensor<1x576x11x65xf16, {order = #NHWC}>
    // CHECK: [[CONVTILE3:%.+]] = VPU.NCE.Convolution([[DEPTHCONVTILE3]], [[W_CONV]], [[WT_CONV]]) rawFilterShape [160, 576, 1, 1]
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], vf_loop_index = 0 : i64, vf_loop_tile_index = 3 : i64}
    // CHECK-SAME: -> tensor<1x160x11x65xf16, {order = #NHWC}>
    // CHECK: [[SLICEARG0TILE4:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 42, 0] [1, 576, 15, 65]
    // CHECK: [[DEPTHCONVTILE4:%.+]] = VPU.NCE.DepthConvolution([[SLICEARG0TILE4]], [[W_DWCONV]], [[WT_DWCONV]]) rawFilterShape [576, 1, 5, 5]
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], vf_loop_index = 0 : i64, vf_loop_tile_index = 4 : i64} -> tensor<1x576x11x65xf16, {order = #NHWC}>
    // CHECK: [[CONVTILE4:%.+]] = VPU.NCE.Convolution([[DEPTHCONVTILE4]], [[W_CONV]], [[WT_CONV]]) rawFilterShape [160, 576, 1, 1]
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], vf_loop_index = 0 : i64, vf_loop_tile_index = 4 : i64}
    // CHECK-SAME: -> tensor<1x160x11x65xf16, {order = #NHWC}>
    // CHECK: [[SLICEARG0TILE5:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 53, 0] [1, 576, 12, 65]
    // CHECK: [[DEPTHCONVTILE5:%.+]] = VPU.NCE.DepthConvolution([[SLICEARG0TILE5]], [[W_DWCONV]], [[WT_DWCONV]]) rawFilterShape [576, 1, 5, 5]
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], vf_loop_index = 0 : i64, vf_loop_tile_index = 5 : i64} -> tensor<1x576x10x65xf16, {order = #NHWC}>
    // CHECK: [[CONVTILE5:%.+]] = VPU.NCE.Convolution([[DEPTHCONVTILE5]], [[W_CONV]], [[WT_CONV]]) rawFilterShape [160, 576, 1, 1]
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], vf_loop_index = 0 : i64, vf_loop_tile_index = 5 : i64}
    // CHECK-SAME: -> tensor<1x160x10x65xf16, {order = #NHWC}>
    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[CONVTILE0]], [[CONVTILE1]], [[CONVTILE2]], [[CONVTILE3]], [[CONVTILE4]], [[CONVTILE5]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]]} : tensor<1x160x11x65xf16, {order = #NHWC}>, tensor<1x160x11x65xf16, {order = #NHWC}>, tensor<1x160x11x65xf16, {order = #NHWC}>, tensor<1x160x11x65xf16, {order = #NHWC}>, tensor<1x160x11x65xf16, {order = #NHWC}>, tensor<1x160x10x65xf16, {order = #NHWC}> -> tensor<1x160x65x65xf16, {order = #NHWC}>
    // CHECK: return [[CONCAT]] : tensor<1x160x65x65xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @VfTilingBetweenEltwise
// CHECK-SAME: [[ARG_3:%[^:]+]]: tensor<1x256x256x256xf16, {order = #NHWC}>
func.func @VfTilingBetweenEltwise(%input: tensor<1x64x256x256xf16, {order = #NHWC}>, %cst_1: tensor<256x64x1x1xf16, {order = #NHWC}>, %cst_2: tensor<256x1x1x4xsi32>, %cst_3: tensor<1x256x256x256xf16, {order = #NHWC}>, %cst_4: tensor<64x256x1x1xf16, {order = #NHWC}>, %cst_5: tensor<64x1x1x4xsi32>, %cst_6: tensor<64x64x3x3xf16, {order = #NHWC}>, %cst_7: tensor<64x1x1x4xsi32>, %cst_8: tensor<256x64x1x1xf16, {order = #NHWC}>, %cst_9: tensor<256x1x1x4xsi32>) -> tensor<1x256x256x256xf16>  {

  %0 = VPU.VerticalFusion (%input as %arg1: tensor<1x64x256x256xf16, {order = #NHWC}>, %cst_1 as %arg2: tensor<256x64x1x1xf16, {order = #NHWC}>, %cst_2 as %arg3: tensor<256x1x1x4xsi32>, %cst_3 as %arg4: tensor<1x256x256x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 32, 1], vf_loop_index = 0} -> tensor<1x256x256x256xf16, {order = #NHWC}> {
    %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) rawFilterShape [256, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
         strides = [1, 1]} : tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<256x64x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x256x256xf16, {order = #NHWC}>

    %3 = VPU.NCE.Eltwise(%2, %arg4) {
        is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>} -> tensor<1x256x256x256xf16, {order = #NHWC}>
    VPU.Yield %3
  }

  %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x256x256x256xf16, {order = #NHWC}>, %cst_4 as %arg2: tensor<64x256x1x1xf16, {order = #NHWC}>, %cst_5 as %arg3: tensor<64x1x1x4xsi32>, %cst_6 as %arg4: tensor<64x64x3x3xf16, {order = #NHWC}>, %cst_7 as %arg5: tensor<64x1x1x4xsi32>, %cst_8 as %arg6: tensor<256x64x1x1xf16, {order = #NHWC}>, %cst_9 as %arg7: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 32, 1], vf_loop_index = 0} -> tensor<1x256x256x256xf16> {
    %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) rawFilterShape [64, 256, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
                strides = [1, 1]} : tensor<1x256x256x256xf16, {order = #NHWC}>, tensor<64x256x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x256x256xf16, {order = #NHWC}>

    %3 = VPU.NCE.Convolution(%2, %arg4, %arg5) rawFilterShape [64, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
                strides = [1, 1]} : tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<64x64x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x256x256xf16, {order = #NHWC}>

    %4 = VPU.NCE.Convolution(%3, %arg6, %arg7) rawFilterShape [256, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
                strides = [1, 1]} : tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<256x64x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x256x256xf16, {order = #NHWC}>
    %5 = VPU.NCE.Eltwise(%4, %arg1) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>} -> tensor<1x256x256x256xf16>
    VPU.Yield %5
  }
  return %1 : tensor<1x256x256x256xf16>

    // CHECK: [[VF1_ELT_SLICE:%.+]] = VPU.Slice [[ARG_3]] [0, 0, 0, 0] [1, 256, 8, 256] :
    // CHECK:                  tensor<1x256x256x256xf16, {order = #NHWC}> to tensor<1x256x8x256xf16, {order = #NHWC}>
    // CHECK: [[VF1_ELT:%.+]] = VPU.NCE.Eltwise
    // CHECK:                   -> tensor<1x256x8x256xf16, {order = #NHWC}>

    // CHECK: [[VF1_ELT_SLICE:%.+]] = VPU.Slice [[ARG_3]] [0, 0, 8, 0] [1, 256, 8, 256] :
    // CHECK:                   tensor<1x256x256x256xf16, {order = #NHWC}> to tensor<1x256x8x256xf16, {order = #NHWC}>
    // CHECK: [[VF1_ELT:%.+]] = VPU.NCE.Eltwise
    // CHECK:                   -> tensor<1x256x8x256xf16, {order = #NHWC}>
    // ...
    // CHECK: [[VF1_ELT_SLICE:%.+]] = VPU.Slice [[ARG_3]] [0, 0, 240, 0] [1, 256, 8, 256] :
    // CHECK:                  tensor<1x256x256x256xf16, {order = #NHWC}> to tensor<1x256x8x256xf16, {order = #NHWC}>
    // CHECK: [[VF1_ELT:%.+]] = VPU.NCE.Eltwise
    // CHECK:                   -> tensor<1x256x8x256xf16, {order = #NHWC}>

    // CHECK: [[VF1_ELT_SLICE:%.+]] = VPU.Slice [[ARG_3]] [0, 0, 248, 0] [1, 256, 8, 256] :
    // CHECK:                   tensor<1x256x256x256xf16, {order = #NHWC}> to tensor<1x256x8x256xf16, {order = #NHWC}>
    // CHECK: [[VF1_ELT:%.+]] = VPU.NCE.Eltwise
    // CHECK:                   -> tensor<1x256x8x256xf16, {order = #NHWC}>

    // CHECK: [[VF1_CONCAT:%.+]] = VPU.Concat

    // CHECK: [[VF2_CONV_SLICE:%.+]] = VPU.Slice [[VF1_CONCAT]] [0, 0, 0, 0] [1, 256, 9, 256]
    // CHECK: [[VF2_ELT_SLICE:%.+]] = VPU.Slice [[VF1_CONCAT]] [0, 0, 0, 0] [1, 256, 8, 256]
    // CHECK: [[VF2_ELT:%.+]] = VPU.NCE.Eltwise
    // CHECK:                    -> tensor<1x256x8x256xf16>

    // CHECK: [[VF2_CONV_SLICE:%.+]] = VPU.Slice [[VF1_CONCAT]] [0, 0, 7, 0] [1, 256, 10, 256]
    // CHECK: [[VF2_ELT_SLICE:%.+]] = VPU.Slice [[VF1_CONCAT]] [0, 0, 8, 0] [1, 256, 8, 256]
    // CHECK: [[VF2_ELT:%.+]] = VPU.NCE.Eltwise
    // CHECK:                     -> tensor<1x256x8x256xf16>
    // ...
    // CHECK: [[VF2_CONV_SLICE:%.+]] = VPU.Slice [[VF1_CONCAT]] [0, 0, 239, 0] [1, 256, 10, 256]
    // CHECK: [[VF2_ELT_SLICE:%.+]] = VPU.Slice [[VF1_CONCAT]] [0, 0, 240, 0] [1, 256, 8, 256]
    // CHECK: [[VF2_ELT:%.+]] = VPU.NCE.Eltwise
    // CHECK:                    -> tensor<1x256x8x256xf16>

    // CHECK: [[VF2_CONV_SLICE:%.+]] = VPU.Slice [[VF1_CONCAT]] [0, 0, 247, 0] [1, 256, 9, 256]
    // CHECK: [[VF2_ELT_SLICE:%.+]] = VPU.Slice [[VF1_CONCAT]] [0, 0, 248, 0] [1, 256, 8, 256]
    // CHECK: [[VF2_ELT:%.+]] = VPU.NCE.Eltwise
    // CHECK:                     -> tensor<1x256x8x256xf16>

    // CHECK: [[VF2_CONCAT:%.+]] = VPU.Concat
    // CHECK: return [[VF2_CONCAT]] : tensor<1x256x256x256xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @VfTilingWithEltwises(%arg0: tensor<1x32x360x640xf16, {order = #NHWC}>, %arg1: tensor<1x32x360x640xf16, {order = #NHWC}>, %cst_1: tensor<32x32x1x1xf16, {order = #NHWC}>, %cst_13: tensor<32x1x1x4xsi32>, %cst_2: tensor<32x32x3x3xf16, {order = #NHWC}>, %cst_12: tensor<32x1x1x4xsi32>, %cst_3: tensor<32x32x3x3xf16, {order = #NHWC}>, %cst_11: tensor<32x1x1x4xsi32>, %cst_4: tensor<32x32x3x3xf16, {order = #NHWC}>, %cst_10: tensor<32x1x1x4xsi32>) -> tensor<1x32x360x640xf16, {order = #NHWC}>   {
   %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x360x640xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x32x360x640xf16, {order = #NHWC}>, %cst_1 as %arg4: tensor<32x32x1x1xf16, {order = #NHWC}>, %cst_13 as %arg5: tensor<32x1x1x4xsi32>, %cst_2 as %arg6: tensor<32x32x3x3xf16, {order = #NHWC}>, %cst_12 as %arg7: tensor<32x1x1x4xsi32>, %cst_3 as %arg8: tensor<32x32x3x3xf16, {order = #NHWC}>, %cst_11 as %arg9: tensor<32x1x1x4xsi32>, %cst_4 as %arg10: tensor<32x32x3x3xf16, {order = #NHWC}>, %cst_10 as %arg11: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 15, 1], vf_loop_index = 0} -> tensor<1x32x360x640xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Eltwise(%arg2, %arg3) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x360x640xf16, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4, %arg5) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x360x640xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x360x640xf16, {order = #NHWC}>
      %3 = VPU.NCE.Convolution(%2, %arg6, %arg7) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x360x640xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x360x640xf16, {order = #NHWC}>
      %4 = VPU.NCE.Eltwise(%3, %1) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x360x640xf16, {order = #NHWC}>
      %5 = VPU.NCE.Convolution(%4, %arg8, %arg9) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x360x640xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x360x640xf16, {order = #NHWC}>
      %6 = VPU.NCE.Eltwise(%5, %4) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x360x640xf16, {order = #NHWC}>
      %7 = VPU.NCE.Convolution(%6, %arg10, %arg11) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x360x640xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x360x640xf16, {order = #NHWC}>
      VPU.Yield %7
    }

   return %0 : tensor<1x32x360x640xf16, {order = #NHWC}>


    // CHECK: [[SLICE_ARG0_TILE0:%.+]] = VPU.Slice {{[^:]+}} [0, 0, 0, 0] [1, 32, 27, 640]
    // CHECK: [[SLICE_ARG1_TILE0:%.+]] = VPU.Slice {{[^:]+}} [0, 0, 0, 0] [1, 32, 27, 640]
    // CHECK: [[ELT0_TILE0:%.+]] = VPU.NCE.Eltwise([[SLICE_ARG0_TILE0]], [[SLICE_ARG1_TILE0]])
    // CHECK: [[CONV0_TILE0:%.+]] = VPU.NCE.Convolution([[ELT0_TILE0]], {{[^:]+}}, {{[^:]+}})
    // CHECK: [[CONV1_TILE0:%.+]] = VPU.NCE.Convolution([[CONV0_TILE0]], {{[^:]+}}, {{[^:]+}})
    // CHECK: [[SLICE_ELT0_TILE0:%.+]] = VPU.Slice [[ELT0_TILE0]] [0, 0, 0, 0] [1, 32, 26, 640]
    // CHECK: [[ELT1_TILE0:%.+]] = VPU.NCE.Eltwise([[CONV1_TILE0]], [[SLICE_ELT0_TILE0]])
    // CHECK: [[CONV2_TILE0:%.+]] = VPU.NCE.Convolution([[ELT1_TILE0]], {{[^:]+}}, {{[^:]+}})
    // CHECK: [[SLICE_ELT1_TILE0:%.+]] = VPU.Slice [[ELT1_TILE0]] [0, 0, 0, 0] [1, 32, 25, 640]
    // CHECK: [[ELT2_TILE0:%.+]] = VPU.NCE.Eltwise([[CONV2_TILE0]], [[SLICE_ELT1_TILE0]])
    // CHECK: [[CONV3_TILE0:%.+]] = VPU.NCE.Convolution([[ELT2_TILE0]], {{[^:]+}}, {{[^:]+}})
    // CHECK: [[SLICE_ARG0_TILE1:%.+]] = VPU.Slice {{[^:]+}} [0, 0, 21, 0] [1, 32, 30, 640]
    // CHECK: [[SLICE_ARG1_TILE1:%.+]] = VPU.Slice {{[^:]+}} [0, 0, 21, 0] [1, 32, 30, 640]
    // CHECK: [[ELT0_TILE1:%.+]] = VPU.NCE.Eltwise([[SLICE_ARG0_TILE1]], [[SLICE_ARG1_TILE1]])
    // CHECK: [[CONV0_TILE1:%.+]] = VPU.NCE.Convolution([[ELT0_TILE1]], {{[^:]+}}, {{[^:]+}})
    // CHECK: [[CONV1_TILE1:%.+]] = VPU.NCE.Convolution([[CONV0_TILE1]], {{[^:]+}}, {{[^:]+}})
    // CHECK: [[SLICE_ELT0_TILE1:%.+]] = VPU.Slice [[ELT0_TILE1]] [0, 0, 1, 0] [1, 32, 28, 640]
    // CHECK: [[ELT1_TILE1:%.+]] = VPU.NCE.Eltwise([[CONV1_TILE1]], [[SLICE_ELT0_TILE1]])
    // CHECK: [[CONV2_TILE1:%.+]] = VPU.NCE.Convolution([[ELT1_TILE1]], {{[^:]+}}, {{[^:]+}})
    // CHECK: [[SLICE_ELT1_TILE1:%.+]] = VPU.Slice [[ELT1_TILE1]] [0, 0, 1, 0] [1, 32, 26, 640]
    // CHECK: [[ELT2_TILE1:%.+]] = VPU.NCE.Eltwise([[CONV2_TILE1]], [[SLICE_ELT1_TILE1]])
    // CHECK: [[CONV3_TILE1:%.+]] = VPU.NCE.Convolution([[ELT2_TILE1]], {{[^:]+}}, {{[^:]+}})
}
