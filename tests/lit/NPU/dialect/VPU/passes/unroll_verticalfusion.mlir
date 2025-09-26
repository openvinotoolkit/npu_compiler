//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --unroll-unused-vertical-fusion %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UnrollVFRegion(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>, %weights: tensor<32x32x3x3xf16, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x256x256xf16, {order = #NHWC}>, %weights as %arg2: tensor<32x32x3x3xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
      VPU.Yield %1
    }
    return %0 : tensor<1x32x256x256xf16, {order = #NHWC}>

    //CHECK-NOT: VPU.VerticalFusion
    //CHECK:     VPU.NCE.Convolution
    //CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    //CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:  rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
    //CHECK-SAME:  -> tensor<1x32x256x256xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UnrollVFTwoOpsRegion(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>, %weights: tensor<32x32x3x3xf16, {order = #NHWC}>) -> tensor<1x16x256x256xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x256x256xf16, {order = #NHWC}>, %weights as %arg2: tensor<32x32x3x3xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x256x256xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
      %2 = VPU.Slice %1 [0, 0, 0, 0] [1, 16, 256, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x16x256x256xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    return %0 : tensor<1x16x256x256xf16, {order = #NHWC}>

    //CHECK-NOT: VPU.VerticalFusion
    //CHECK:     VPU.NCE.Convolution
    //CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    //CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME: rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
    //CHECK-SAME: -> tensor<1x32x256x256xf16, {order = #NHWC}>
    //CHECK:     VPU.Slice
    //CHECK-SAME:  [0, 0, 0, 0] [1, 16, 256, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x16x256x256xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DontUnrollVFTwoOpsRegion(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>, %weights1: tensor<32x32x1x1xf16, {order = #NHWC}>, %weights2: tensor<32x32x3x3xf16, {order = #NHWC}>) -> tensor<1x16x256x256xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x256x256xf16, {order = #NHWC}>, %weights1 as %arg2: tensor<32x32x1x1xf16, {order = #NHWC}>, %weights2 as %arg3: tensor<32x32x3x3xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x256x256xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 32, 1, 1], strides = [1, 1]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    return %0 : tensor<1x16x256x256xf16, {order = #NHWC}>

    //CHECK: [[VF:%.+]] = VPU.VerticalFusion ({{[^ ]+}} as [[INPUT:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>, {{[^ ]+}} as [[WEIGHTS_0:%.+]]: tensor<32x32x1x1xf16, {order = #NHWC}>, {{[^ ]+}} as [[WEIGHTS_1:%.+]]: tensor<32x32x3x3xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x256x256xf16, {order = #NHWC}> {
    //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_0]])
    // CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:              rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
    // CHECK-SAME:              -> tensor<1x32x256x256xf16, {order = #NHWC}>
    //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV0]], [[WEIGHTS_1]])
    // CHECK-SAME:             multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:             pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:             rawFilterShape = [32, 32, 3, 3], strides = [1, 1]}
    // CHECK-SAME:             -> tensor<1x32x256x256xf16, {order = #NHWC}>
    //CHECK:   VPU.Yield [[CONV1]]
    //CHECK: return [[VF]] : tensor<1x16x256x256xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @UnrollNonTiledRegion(%arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>, %arg1: tensor<80x48x1x1xf16, {order = #NHWC}>, %arg2: tensor<48x80x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x1024x4xf16, {order = #NHWC}>, %arg1 as %arg4: tensor<80x48x1x1xf16, {order = #NHWC}>, %arg2 as %arg6: tensor<48x80x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg3, %arg4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<80x48x1x1xf16, {order = #NHWC}> -> tensor<1x80x1024x4xf16, {order = #NHWC}>
        %2 = VPU.SoftMax(%1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x80x1024x4xf16, {order = #NHWC}> -> tensor<1x80x1024x4xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %arg6) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 80, 1, 1], strides = [1, 1]} : tensor<1x80x1024x4xf16, {order = #NHWC}>, tensor<48x80x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
    }
    return %0 : tensor<1x48x1024x4xf16, {order = #NHWC}>

   //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution(%arg0, %arg1)
   //CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV0]])
   //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], %arg2)

   //CHECK: return [[CONV1]] : tensor<1x48x1024x4xf16, {order = #NHWC}>
}
