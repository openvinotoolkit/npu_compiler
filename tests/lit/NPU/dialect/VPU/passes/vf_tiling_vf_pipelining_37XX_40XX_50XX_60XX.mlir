//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --vertical-fusion-tiling="enable-vertical-fusion-pipelining=true" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8<0:254>:f16, 0.003937007874015748>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType2 = !quant.uniform<u8:f16, 5.000000e-01>

// CHECK-LABEL: @VfTilingWithSwish
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x176x176x!qElemType, {order = #NHWC}>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<96x16x1x1x!qElemType1, {order = #NHWC}>,
// CHECK-SAME: [[ARG_2:%[^:]+]]: tensor<96x16x1x1xf16, {order = #NHWC}>)
func.func @VfTilingWithSwish(%arg0: tensor<1x16x176x176x!quant.uniform<u8:f16, 0.14376571505677466:128>, {order = #NHWC}>, %cst_1: tensor<96x16x1x1x!qElemType1, {order = #NHWC}>, %cst_3: tensor<96x16x1x1xf16, {order = #NHWC}>) -> tensor<1x96x176x176x!qElemType2, {order = #NHWC}>  {
   %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x176x176x!quant.uniform<u8:f16, 0.14376571505677466:128>, {order = #NHWC}>, %cst_1 as %arg3: tensor<96x16x1x1x!qElemType1, {order = #NHWC}>, %cst_3 as %arg5: tensor<96x16x1x1xf16, {order = #NHWC}>) attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1], vf_loop_index = 0} -> tensor<1x96x176x176x!qElemType2, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg3) rawFilterShape [96, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEStub<>,
          strides = [1, 1]} : tensor<1x16x176x176x!quant.uniform<u8:f16, 0.14376571505677466:128>, {order = #NHWC}>, tensor<96x16x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x96x176x176xf16, {order = #NHWC}>

      %2 = VPU.Swish(%1)
         {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x96x176x176xf16, {order = #NHWC}> -> tensor<1x96x176x176xf16, {order = #NHWC}>

      %3 = VPU.NCE.DepthConvolution(%2, %arg5) rawFilterShape [96, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEStub<>,
          strides = [1, 1]} -> tensor<1x96x176x176x!qElemType2, {order = #NHWC}>

      VPU.Yield %3
   }

   return %0 : tensor<1x96x176x176x!qElemType2, {order = #NHWC}>

   // CHECK: [[SLICE0:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 0, 0] [1, 16, 44, 176] : tensor<1x16x176x176x!qElemType, {order = #NHWC}> to tensor<1x16x44x176x!qElemType, {order = #NHWC}>
   // CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[ARG_1]]) rawFilterShape [96, 16, 1, 1] {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
   // CHECK-SAME: strides = [1, 1], vf_loop_index = 1 : i64, vf_loop_tile_index = 0 : i64}
   // CHECK-SAME: -> tensor<1x96x44x176xf16, {order = #NHWC}>

   // CHECK: [[SWISH0:%.+]] = VPU.Swish([[CONV0]]) {beta_value = 1.000000e+00 : f64, loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, vf_loop_index = 1 : i64, vf_loop_tile_index = 0 : i64} : tensor<1x96x44x176xf16, {order = #NHWC}> -> tensor<1x96x44x176xf16, {order = #NHWC}>
   // CHECK: [[SLICE1:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 44, 0] [1, 16, 44, 176] : tensor<1x16x176x176x!qElemType, {order = #NHWC}> to tensor<1x16x44x176x!qElemType, {order = #NHWC}>
   // CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[ARG_1]]) rawFilterShape [96, 16, 1, 1] {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 1>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
   // CHECK-SAME: strides = [1, 1], vf_loop_index = 1 : i64, vf_loop_tile_index = 1 : i64}
   // CHECK-SAME: -> tensor<1x96x44x176xf16, {order = #NHWC}>
   // CHECK: [[DEPTHCONV0:%.+]] = VPU.NCE.DepthConvolution([[SWISH0]], [[ARG_2]]) rawFilterShape [96, 1, 1, 1] {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
   // CHECK-SAME: resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], vf_loop_index = 1 : i64, vf_loop_tile_index = 0 : i64} -> tensor<1x96x44x176x!qElemType2, {order = #NHWC}>

   // CHECK: [[SWISH1:%.+]] = VPU.Swish([[CONV1]]) {beta_value = 1.000000e+00 : f64, loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 1>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, vf_loop_index = 1 : i64, vf_loop_tile_index = 1 : i64} : tensor<1x96x44x176xf16, {order = #NHWC}> -> tensor<1x96x44x176xf16, {order = #NHWC}>
   // CHECK: [[SLICE2:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 88, 0] [1, 16, 44, 176] : tensor<1x16x176x176x!qElemType, {order = #NHWC}> to tensor<1x16x44x176x!qElemType, {order = #NHWC}>
   // CHECK: [[CONV2:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[ARG_1]]) rawFilterShape [96, 16, 1, 1] {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 2>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
   // CHECK-SAME: strides = [1, 1], vf_loop_index = 1 : i64, vf_loop_tile_index = 2 : i64}
   // CHECK-SAME: -> tensor<1x96x44x176xf16, {order = #NHWC}>
   // CHECK: [[DEPTHCONV1:%.+]] = VPU.NCE.DepthConvolution([[SWISH1]], [[ARG_2]]) rawFilterShape [96, 1, 1, 1] {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 1>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
   // CHECK-SAME: resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], vf_loop_index = 1 : i64, vf_loop_tile_index = 1 : i64} -> tensor<1x96x44x176x!qElemType2, {order = #NHWC}>

   // CHECK: [[SWISH2:%.+]] = VPU.Swish([[CONV2]]) {beta_value = 1.000000e+00 : f64, loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 2>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, vf_loop_index = 1 : i64, vf_loop_tile_index = 2 : i64} : tensor<1x96x44x176xf16, {order = #NHWC}> -> tensor<1x96x44x176xf16, {order = #NHWC}>
   // CHECK: [[SLICE3:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 132, 0] [1, 16, 44, 176] : tensor<1x16x176x176x!qElemType, {order = #NHWC}> to tensor<1x16x44x176x!qElemType, {order = #NHWC}>
   // CHECK: [[CONV3:%.+]] = VPU.NCE.Convolution([[SLICE3]], [[ARG_1]]) rawFilterShape [96, 16, 1, 1] {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 3>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
   // CHECK-SAME: strides = [1, 1], vf_loop_index = 1 : i64, vf_loop_tile_index = 3 : i64}
   // CHECK-SAME: -> tensor<1x96x44x176xf16, {order = #NHWC}>
   // CHECK: [[DEPTHCONV2:%.+]] = VPU.NCE.DepthConvolution([[SWISH2]], [[ARG_2]]) rawFilterShape [96, 1, 1, 1] {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 2>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
   // CHECK-SAME: resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], vf_loop_index = 1 : i64, vf_loop_tile_index = 2 : i64} -> tensor<1x96x44x176x!qElemType2, {order = #NHWC}>

   // CHECK: [[SWISH3:%.+]] = VPU.Swish([[CONV3]]) {beta_value = 1.000000e+00 : f64, loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 3>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, vf_loop_index = 1 : i64, vf_loop_tile_index = 3 : i64} : tensor<1x96x44x176xf16, {order = #NHWC}> -> tensor<1x96x44x176xf16, {order = #NHWC}>

   // CHECK: [[DEPTHCONV3:%.+]] = VPU.NCE.DepthConvolution([[SWISH3]], [[ARG_2]]) rawFilterShape [96, 1, 1, 1] {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 3>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
   // CHECK-SAME: resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], vf_loop_index = 1 : i64, vf_loop_tile_index = 3 : i64} -> tensor<1x96x44x176x!qElemType2, {order = #NHWC}>

   // CHECK: [[CONCAT:%.+]] = VPU.Concat([[DEPTHCONV0]], [[DEPTHCONV1]], [[DEPTHCONV2]], [[DEPTHCONV3]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 44, 0], [0, 0, 88, 0], [0, 0, 132, 0]]} : tensor<1x96x44x176x!qElemType2, {order = #NHWC}>, tensor<1x96x44x176x!qElemType2, {order = #NHWC}>, tensor<1x96x44x176x!qElemType2, {order = #NHWC}>, tensor<1x96x44x176x!qElemType2, {order = #NHWC}> -> tensor<1x96x176x176x!qElemType2, {order = #NHWC}>
   // CHECK: return [[CONCAT]] : tensor<1x96x176x176x!qElemType2, {order = #NHWC}>
}
