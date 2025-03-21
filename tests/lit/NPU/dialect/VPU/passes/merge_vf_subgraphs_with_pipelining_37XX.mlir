//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --merge-vertical-fusion-subgraphs="enable-vertical-fusion-pipelining=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MergeNonTiledRegion(
                %arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
                %arg1: tensor<80x48x1x1xf16, {order = #NHWC}>,
                %arg2: tensor<48x80x1x1xf16, {order = #NHWC}>)
                    -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x1024x4xf16, {order = #NHWC}>,
        %arg1 as %arg4: tensor<80x48x1x1xf16, {order = #NHWC}>,
        %cst_0 as %arg5: tensor<80x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x80x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      rawFilterShape = [80, 48, 1, 1], strides = [1, 1]} -> tensor<1x80x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
   }

    %1 = VPU.VerticalFusion (%0 as %arg3: tensor<1x80x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x80x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.SoftMax(%arg3) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x80x1024x4xf16, {order = #NHWC}> -> tensor<1x80x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
   }

   %2 = VPU.VerticalFusion (%1 as %arg3: tensor<1x80x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg4: tensor<48x80x1x1xf16, {order = #NHWC}>,
        %cst as %arg5: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      rawFilterShape = [48, 80, 1, 1], strides = [1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
   }

   return %2: tensor<1x48x1024x4xf16, {order = #NHWC}>

   // Merge non-tiled operations when the pipelining is enabled
   //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
   //CHECK-SAME:    tilingStrategy = [1, 1, 2, 1]
   //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
   //CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV0]])
   //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], %arg6, %arg7)
   //CHECK: VPU.Yield [[CONV1]]

   //CHECK: return [[VERTICAL_FUSION]] : tensor<1x48x1024x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16, 0.00565029593075023:128>

func.func @VFIncreaseTileStrategy(
                %arg0: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>,
                %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>,
                %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
    %cst_2 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>,
        %arg1 as %arg4: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>,
        %cst_0 as %arg5: tensor<4096x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
   }

    %1 = VPU.VerticalFusion (%0 as %arg3: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.SoftMax(%arg3) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
   }

   %2 = VPU.VerticalFusion (%1 as %arg3: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg4: tensor<48x4096x1x1xf16, {order = #NHWC}>,
        %cst_2 as %arg5: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
   }

   return %2: tensor<1x48x1024x4xf16, {order = #NHWC}>

   // CHECK: VPU.VerticalFusion
   // CHECK-SAME: scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 57, 1]

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.013263059129901961:21>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MergeVFWithoutVFPipelining(
      %arg0: tensor<1x640x64x64xf16, {order = #NHWC}>,
      %arg1: tensor<640x16x1x1xf16, {order = #NHWC}>) -> tensor<1x640x64x64x!qElemType, {order = #NHWC}> {
      %cst = const.Declare tensor<640x1x1x4xsi32> = dense<1> : tensor<640x1x1x4xsi32>
   %0 = VPU.VerticalFusion (
      %arg0 as %arg2: tensor<1x640x64x64xf16, {order = #NHWC}>,
      %arg1 as %arg3: tensor<640x16x1x1xf16, {order = #NHWC}>,
      %cst as %arg4: tensor<640x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
            -> tensor<1x640x64x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
      %3 = VPU.NCE.DepthConvolution(%arg2, %arg3, %arg4) {
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [640, 1, 1, 1], strides = [1, 1]} -> tensor<1x640x64x64xf16, {order = #NHWC}>
      %4 = VPU.Swish(%3) {
         beta_value = 1.000000e+00 : f64,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x640x64x64xf16, {order = #NHWC}>
            -> tensor<1x640x64x64xf16, {order = #NHWC}>
      VPU.Yield %4
   }

   %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x640x64x64xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]}
            -> tensor<1x640x64x64x!qElemType, {order = #NHWC}> {
      %5 = VPU.NCE.AveragePool(%arg2) {
         kernel_size = [1, 1],
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEStub<>,
         strides = [1, 1]} -> tensor<1x640x64x64x!qElemType, {order = #NHWC}>
      VPU.Yield %5
   }

   return %1 : tensor<1x640x64x64x!qElemType, {order = #NHWC}>

   //CHECK: [[VF:%.+]] = VPU.VerticalFusion
   //CHECK-SAME: scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 7, 1]
   //CHECK:    [[DWCONV:%.+]] = VPU.NCE.DepthConvolution
   //CHECK:    [[SWISH:%.+]] = VPU.Swish
   //CHECK:    [[AVGPOOL:%.+]] = VPU.NCE.AveragePool

   //CHECK: return [[VF]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.00565029593075023:128>
!qElemType1 = !quant.uniform<u8:f16, 0.013744638480392157:128>

func.func @VFTestCorrectTilingAxisWhenGetPrefetchingCost(
                %arg0: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>,
                %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>,
                %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
    %cst_1 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>,
        %arg1 as %arg4: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>,
        %cst_0 as %arg5: tensor<4096x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
   }

    %1 = VPU.VerticalFusion (%0 as %arg3: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.SoftMax(%arg3) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
   }

   %2 = VPU.VerticalFusion (%1 as %arg3: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg4: tensor<48x4096x1x1xf16, {order = #NHWC}>,
        %cst_1 as %arg5: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
   }

   return %2: tensor<1x48x1024x4xf16, {order = #NHWC}>

   // CHECK: VPU.VerticalFusion
   // CHECK-SAME: scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 57, 1]

}
