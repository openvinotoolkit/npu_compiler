//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW weights-table-reuse-mode=VF_ENABLED" --vertical-fusion-outlining="vf-outlining-tile-threshold=1 vf-outlining-instance-threshold=1" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @MultipleOpsInVF {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x48x1024x4xf16>
        DataInfo "input2" : tensor<4096x48x1x1xf16>
        DataInfo "input3" : tensor<48x4096x1x1xf16>
    }
    outputsInfo : {
        DataInfo "output" : tensor<1x48x1024x4xf16>
    }

  func.func @main(%arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>, %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>, %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x1024x4xf16, {order = #NHWC}>, %arg1 as %arg4: tensor<4096x48x1x1xf16, {order = #NHWC}>, %arg2 as %arg6: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 19, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg3, %arg4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      %2 = VPU.SoftMax(%1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      %3 = VPU.NCE.Convolution(%2, %arg6) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    return %0 : tensor<1x48x1024x4xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @MultipleOpsInVF

// CHECK: DataInfo "input1" : tensor<1x48x1024x4xf16>
// CHECK: DataInfo "input2" : tensor<4096x48x1x1xf16>
// CHECK: DataInfo "input3" : tensor<48x4096x1x1xf16>
// CHECK: DataInfo "output" : tensor<1x48x1024x4xf16>

// CHECK: func.func private @main_vf1([[ARG0:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>, [[ARG2:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG0]] as {{[^:]+}}: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1]] as {{[^:]+}}: tensor<4096x48x1x1xf16, {order = #NHWC}>, [[ARG2]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 19, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:    [[OP1:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]}
// CHECK-SAME:    -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:    [[OP2:%.+]] = VPU.SoftMax([[OP1]]) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:    [[OP3:%.+]] = VPU.NCE.Convolution([[OP2]], {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:    -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                  VPU.Yield [[OP3]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK: func.func @main([[INPUT1:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>, [[INPUT3:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:  [[CALL:%.+]] = call @main_vf1([[INPUT1]], [[INPUT2]], [[INPUT3]]) : (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK:  return [[CALL]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @ParallelConcatInput {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x32x256x256xf16>
    }
    outputsInfo : {
        DataInfo "output0" : tensor<1x64x256x256xf16>
        DataInfo "output1" : tensor<1x64x256x256xf16>
    }

  func.func @main(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>) -> (tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>) {
      %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16, {order = #NHWC}>
      %cst_2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16, {order = #NHWC}>
      %cst_3 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

      %0 = VPU.NCE.Convolution(%arg0, %cst_0)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [32, 32, 3, 3], strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]}
          : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>

      %1 = VPU.NCE.Convolution(%arg0, %cst_0)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [32, 32, 3, 3], strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]}
          : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>

      %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>

      %3 = VPU.VerticalFusion (%2 as %arg1: tensor<1x64x256x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x256x256xf16, {order = #NHWC}> {
        %6 = VPU.SoftMax(%arg1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>
        VPU.Yield %6
      }

      %4 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>

      %5 = VPU.VerticalFusion (%4 as %arg1: tensor<1x64x256x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x256x256xf16, {order = #NHWC}> {
        %6 = VPU.SoftMax(%arg1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>
        VPU.Yield %6
      }

      return %3, %5 : tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @ParallelConcatInput

// CHECK: DataInfo "input" : tensor<1x32x256x256xf16>
// CHECK: DataInfo "output0" : tensor<1x64x256x256xf16>
// CHECK: DataInfo "output1" : tensor<1x64x256x256xf16>

// CHECK: func.func private @main_vf1([[ARG:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>) -> (tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>) {
// CHECK:  [[CST0:%.+]] = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16, {order = #NHWC}>
// CHECK:  [[OP0:%.+]] = VPU.NCE.Convolution([[ARG]], [[CST0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
// CHECK-SAME:  -> tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:  [[OP1:%.+]] = VPU.NCE.Convolution([[ARG]], [[CST0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
// CHECK-SAME:  -> tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:  [[OP2:%.+]] = VPU.Concat([[OP0]], [[OP1]])
// CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]}
// CHECK-SAME:               : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[OP2]] as {{[^:]+}}: tensor<1x64x256x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x256x256xf16, {order = #NHWC}> {
// CHECK:                [[VFOP:%.+]] = VPU.SoftMax({{[^:]+}}) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[VFOP]]
// CHECK:               }
// CHECK:  [[OP4:%.+]] = VPU.Concat([[OP0]], [[OP1]])
// CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]}
// CHECK-SAME:               : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>
// CHECK:  return [[VF]], [[OP4]] : tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf2({{[^:]+}}: tensor<1x64x256x256xf16, {order = #NHWC}>) -> tensor<1x64x256x256xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ({{[^:]+}} as {{[^:]+}}: tensor<1x64x256x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x256x256xf16, {order = #NHWC}> {
// CHECK:                [[VFOP:%.+]] = VPU.SoftMax({{[^:]+}}) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[VFOP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x64x256x256xf16, {order = #NHWC}>

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>) -> (tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>) {
// CHECK:  [[CALL0:%.+]]:2 = call @main_vf1([[INPUT]]) : (tensor<1x32x256x256xf16, {order = #NHWC}>) -> (tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>)
// CHECK:  [[CALL1:%.+]] = call @main_vf2([[CALL0]]#1) : (tensor<1x64x256x256xf16, {order = #NHWC}>) -> tensor<1x64x256x256xf16, {order = #NHWC}>

// CHECK:  return [[CALL0]]#0, [[CALL1]] : tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @ParallelConcatInputWithMultiTilingDim {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x32x256x256xf16>
    }
    outputsInfo : {
        DataInfo "output0" : tensor<1x64x256x256xf16>
        DataInfo "output1" : tensor<1x64x256x256xf16>
    }

  func.func @main(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>) -> (tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>) {
      %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16, {order = #NHWC}>
      %cst_2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16, {order = #NHWC}>
      %cst_3 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

      %0 = VPU.NCE.Convolution(%arg0, %cst_0)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [32, 32, 3, 3], strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]}
          : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>

      %1 = VPU.NCE.Convolution(%arg0, %cst_0)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [32, 32, 3, 3], strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]}
          : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>

      %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>

      %3 = VPU.VerticalFusion (%2 as %arg1: tensor<1x64x256x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 4]} -> tensor<1x64x256x256xf16, {order = #NHWC}> {
        %6 = VPU.SoftMax(%arg1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>
        VPU.Yield %6
      }

      %4 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>

      %5 = VPU.VerticalFusion (%4 as %arg1: tensor<1x64x256x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 4]} -> tensor<1x64x256x256xf16, {order = #NHWC}> {
        %6 = VPU.SoftMax(%arg1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>
        VPU.Yield %6
      }

      return %3, %5 : tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @ParallelConcatInputWithMultiTilingDim

// CHECK: DataInfo "input" : tensor<1x32x256x256xf16>
// CHECK: DataInfo "output0" : tensor<1x64x256x256xf16>
// CHECK: DataInfo "output1" : tensor<1x64x256x256xf16>

// CHECK: func.func private @main_vf1([[ARG:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>) -> (tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>) {
// CHECK:  [[CST0:%.+]] = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16, {order = #NHWC}>
// CHECK:  [[OP0:%.+]] = VPU.NCE.Convolution([[ARG]], [[CST0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
// CHECK-SAME:  -> tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:  [[OP1:%.+]] = VPU.NCE.Convolution([[ARG]], [[CST0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
// CHECK-SAME:  -> tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:  [[OP2:%.+]] = VPU.Concat([[OP0]], [[OP1]])
// CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]}
// CHECK-SAME:               : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>
// CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[OP2]] as {{[^:]+}}: tensor<1x64x256x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 4]} -> tensor<1x64x256x256xf16, {order = #NHWC}> {
// CHECK:                [[VFOP0:%.+]] = VPU.SoftMax({{[^:]+}}) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[VFOP0]]
// CHECK:               }
// CHECK:  [[OP4:%.+]] = VPU.Concat([[OP0]], [[OP1]])
// CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]}
// CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[OP4]] as {{[^:]+}}: tensor<1x64x256x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 4]} -> tensor<1x64x256x256xf16, {order = #NHWC}> {
// CHECK:                [[VFOP1:%.+]] = VPU.SoftMax({{[^:]+}}) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x256x256xf16, {order = #NHWC}> -> tensor<1x64x256x256xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[VFOP1]]
// CHECK:               }
// CHECK:  return [[VF0]], [[VF1]] : tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>) -> (tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>) {
// CHECK:  [[CALL0:%.+]]:2 = call @main_vf1([[INPUT]]) : (tensor<1x32x256x256xf16, {order = #NHWC}>) -> (tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>)
// CHECK:  return [[CALL0]]#0, [[CALL0]]#1 : tensor<1x64x256x256xf16, {order = #NHWC}>, tensor<1x64x256x256xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @ConcatSliceUsers {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x32x256x256xf16>
    }
    outputsInfo : {
        DataInfo "output" : tensor<1x48x256x256xf16>
    }

  func.func @main(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>) -> tensor<1x48x256x256xf16, {order = #NHWC}> {
      %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x256x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}> {
        %9 = VPU.SoftMax(%arg1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
        VPU.Yield %9
      }
      %1 = VPU.Slice %0 [0, 16, 0, 0] [1, 16, 256, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x16x256x256xf16, {order = #NHWC}>

      %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x16x256x256xf16, {order = #NHWC}> -> tensor<1x48x256x256xf16, {order = #NHWC}>

      %3 = VPU.Slice %2 [0, 0, 0, 0] [1, 32, 256, 256] : tensor<1x48x256x256xf16, {order = #NHWC}> to tensor<1x32x256x256xf16, {order = #NHWC}>
      %4 = VPU.Slice %2 [0, 32, 0, 0] [1, 16, 256, 256] : tensor<1x48x256x256xf16, {order = #NHWC}> to tensor<1x16x256x256xf16, {order = #NHWC}>

      %5 = VPU.SoftMax(%3) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 4, 1]} : tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
      %6 = VPU.SoftMax(%4) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 4, 1]} : tensor<1x16x256x256xf16, {order = #NHWC}> -> tensor<1x16x256x256xf16, {order = #NHWC}>

      %7 = VPU.Concat(%5, %6) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x16x256x256xf16, {order = #NHWC}> -> tensor<1x48x256x256xf16, {order = #NHWC}>
      return %7 : tensor<1x48x256x256xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @ConcatSliceUsers

// CHECK: DataInfo "input" : tensor<1x32x256x256xf16>
// CHECK: DataInfo "output" : tensor<1x48x256x256xf16>

// CHECK: func.func private @main_vf1([[ARG:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>) -> (tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x16x256x256xf16, {order = #NHWC}>) {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG]] as {{[^:]+}}: tensor<1x32x256x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}> {
// CHECK:                [[VFOP:%.+]] = VPU.SoftMax({{[^:]+}}) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[VFOP]]
// CHECK:               }
// CHECK:  [[SLICE:%.+]] = VPU.Slice [[VF]] [0, 16, 0, 0] [1, 16, 256, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x16x256x256xf16, {order = #NHWC}>
// CHECK:  [[CONCAT:%.+]] = VPU.Concat([[VF]], [[SLICE]])
// CHECK:  [[SLICE1:%.+]] = VPU.Slice [[CONCAT]] [0, 0, 0, 0] [1, 32, 256, 256] : tensor<1x48x256x256xf16, {order = #NHWC}> to tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:  [[SLICE2:%.+]] = VPU.Slice [[CONCAT]] [0, 32, 0, 0] [1, 16, 256, 256] : tensor<1x48x256x256xf16, {order = #NHWC}> to tensor<1x16x256x256xf16, {order = #NHWC}>
// CHECK:  return [[SLICE1]], [[SLICE2]] : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x16x256x256xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf2([[ARG0:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x16x256x256xf16, {order = #NHWC}>) -> tensor<1x48x256x256xf16, {order = #NHWC}> {
// CHECK:  [[SM0:%.+]] = VPU.SoftMax([[ARG0]]) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 4, 1]} : tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:  [[SM1:%.+]] = VPU.SoftMax([[ARG1]]) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 4, 1]} : tensor<1x16x256x256xf16, {order = #NHWC}> -> tensor<1x16x256x256xf16, {order = #NHWC}>
// CHECK:  [[CONCAT:%.+]] = VPU.Concat([[SM0]], [[SM1]])
// CHECK:  return [[CONCAT]] : tensor<1x48x256x256xf16, {order = #NHWC}>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>) -> tensor<1x48x256x256xf16, {order = #NHWC}> {
// CHECK:  [[CALL0:%.+]]:2 = call @main_vf1([[ARG0]]) : (tensor<1x32x256x256xf16, {order = #NHWC}>) -> (tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x16x256x256xf16, {order = #NHWC}>)
// CHECK:  [[CALL1:%.+]] = call @main_vf2([[CALL0]]#0, [[CALL0]]#1) : (tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x16x256x256xf16, {order = #NHWC}>) -> tensor<1x48x256x256xf16, {order = #NHWC}>
// CHECK:  return [[CALL1]] : tensor<1x48x256x256xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @ConcatMultiUsers {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x32x256x256xf16>
    }
    outputsInfo : {
        DataInfo "output" : tensor<1x80x256x256xf16>
    }

  func.func @main(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>) -> tensor<1x80x256x256xf16, {order = #NHWC}> {
      %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x256x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}> {
        %9 = VPU.SoftMax(%arg1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
        VPU.Yield %9
      }
      %1 = VPU.Slice %0 [0, 16, 0, 0] [1, 16, 256, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x16x256x256xf16, {order = #NHWC}>

      %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x16x256x256xf16, {order = #NHWC}> -> tensor<1x48x256x256xf16, {order = #NHWC}>

      %3 = VPU.Slice %2 [0, 0, 0, 0] [1, 32, 256, 256] : tensor<1x48x256x256xf16, {order = #NHWC}> to tensor<1x32x256x256xf16, {order = #NHWC}>

      %4 = VPU.SoftMax(%3) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 4, 1]} : tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
      %5 = VPU.SoftMax(%2) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 4, 1]} : tensor<1x48x256x256xf16, {order = #NHWC}> -> tensor<1x48x256x256xf16, {order = #NHWC}>

      %6 = VPU.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<1x48x256x256xf16, {order = #NHWC}> -> tensor<1x80x256x256xf16, {order = #NHWC}>
      return %6 : tensor<1x80x256x256xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @ConcatMultiUsers

// CHECK: DataInfo "input" : tensor<1x32x256x256xf16>
// CHECK: DataInfo "output" : tensor<1x80x256x256xf16>

// CHECK: func.func private @main_vf1([[ARG:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>) -> tensor<1x48x256x256xf16, {order = #NHWC}> {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG]] as {{[^:]+}}: tensor<1x32x256x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}> {
// CHECK:                [[VFOP:%.+]] = VPU.SoftMax({{[^:]+}}) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[VFOP]]
// CHECK:               }
// CHECK:  [[SLICE:%.+]] = VPU.Slice [[VF]] [0, 16, 0, 0] [1, 16, 256, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x16x256x256xf16, {order = #NHWC}>
// CHECK:  [[CONCAT:%.+]] = VPU.Concat([[VF]], [[SLICE]])
// CHECK:  return [[CONCAT]] : tensor<1x48x256x256xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf2([[ARG:%.+]]: tensor<1x48x256x256xf16, {order = #NHWC}>) -> tensor<1x80x256x256xf16, {order = #NHWC}> {
// CHECK:  [[SLICE:%.+]] = VPU.Slice [[ARG]] [0, 0, 0, 0] [1, 32, 256, 256] : tensor<1x48x256x256xf16, {order = #NHWC}> to tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:  [[SM0:%.+]] = VPU.SoftMax([[SLICE]]) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 4, 1]} : tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:  [[SM1:%.+]] = VPU.SoftMax([[ARG]]) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 4, 1]} : tensor<1x48x256x256xf16, {order = #NHWC}> -> tensor<1x48x256x256xf16, {order = #NHWC}>
// CHECK:  [[CONCAT:%.+]] = VPU.Concat([[SM0]], [[SM1]])
// CHECK:  return [[CONCAT]] : tensor<1x80x256x256xf16, {order = #NHWC}>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>) -> tensor<1x80x256x256xf16, {order = #NHWC}> {
// CHECK:  [[CALL0:%.+]] = call @main_vf1([[ARG0]]) : (tensor<1x32x256x256xf16, {order = #NHWC}>) -> tensor<1x48x256x256xf16, {order = #NHWC}>
// CHECK:  [[CALL1:%.+]] = call @main_vf2([[CALL0]]) : (tensor<1x48x256x256xf16, {order = #NHWC}>) -> tensor<1x80x256x256xf16, {order = #NHWC}>
// CHECK:  return [[CALL1]] : tensor<1x80x256x256xf16, {order = #NHWC}>

// -----

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16, 0.013744638480392158:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @OutliningWithQuantizeCast {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x48x1024x4xui8>
        DataInfo "input2" : tensor<4096x48x1x1xui8>
        DataInfo "input3" : tensor<48x4096x1x1xui8>
    }
    outputsInfo : {
        DataInfo "output" : tensor<1x48x1024x4xui8>
    }

  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType1, {order = #NHWC}>, %arg1: tensor<4096x48x1x1x!qElemType, {order = #NHWC}>, %arg2: tensor<48x4096x1x1x!qElemType, {order = #NHWC}>) -> tensor<1x48x1024x4x!qElemType, {order = #NHWC}> {
    %0 = VPU.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<1x48x1024x4x!qElemType1, {order = #NHWC}> -> tensor<1x48x1024x4x!qElemType, {order = #NHWC}>

    %1 = VPU.VerticalFusion (%0 as %arg3: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1 as %arg4: tensor<4096x48x1x1x!qElemType, {order = #NHWC}>, %arg2 as %arg6: tensor<48x4096x1x1x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 19, 1]} -> tensor<1x48x1024x4x!qElemType, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg3, %arg4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType, {order = #NHWC}> -> tensor<1x4096x1024x4x!qElemType, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg6) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4x!qElemType, {order = #NHWC}>, tensor<48x4096x1x1x!qElemType, {order = #NHWC}> -> tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
      VPU.Yield %2
    }
    return %1 : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
  }
}

// CHECK-LABEL: @OutliningWithQuantizeCast

// CHECK: func.func private @main_vf1([[ARG0:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, [[ARG1:%.+]]: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, [[ARG2:%.+]]: tensor<48x4096x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x48x1024x4x!qElemType1, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[CAST:%.+]] = VPU.QuantizeCast([[ARG0]]) {dstElemType = !qElemType1} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}> -> tensor<1x48x1024x4x!qElemType1, {order = #NHWC}>
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[CAST]] as {{[^:]+}}: tensor<1x48x1024x4x!qElemType1, {order = #NHWC}>, [[ARG1]] as {{[^:]+}}: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, [[ARG2]] as {{[^:]+}}: tensor<48x4096x1x1x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 19, 1]} -> tensor<1x48x1024x4x!qElemType1, {order = #NHWC}> {
// CHECK:    [[OP1:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]}
// CHECK-SAME:    -> tensor<1x4096x1024x4x!qElemType1, {order = #NHWC}>
// CHECK:    [[OP2:%.+]] = VPU.NCE.Convolution([[OP1]], {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:    -> tensor<1x48x1024x4x!qElemType1, {order = #NHWC}>
// CHECK:                  VPU.Yield [[OP2]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x48x1024x4x!qElemType1, {order = #NHWC}>

// CHECK: func.func @main([[INPUT1:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, [[INPUT3:%.+]]: tensor<48x4096x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x48x1024x4x!qElemType1, {order = #NHWC}> {
// CHECK:  [[CALL:%.+]] = call @main_vf1([[INPUT1]], [[INPUT2]], [[INPUT3]]) : (tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, tensor<48x4096x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x48x1024x4x!qElemType1, {order = #NHWC}>

// CHECK:  return [[CALL]] : tensor<1x48x1024x4x!qElemType1, {order = #NHWC}>

// -----

!qElemType = !quant.uniform<i8:f16, 0.006920521633297789>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!SparseTensorType = !VPU.SparseTensor<
  data=tensor<16x16x11x11x!qElemType, {order = #NHWC}>,
  sparsity_map=tensor<16x1x1x2048xi1>,
  is_weights,
  #VPU.SparsityCompression<
    axis = 0 : i64,
    numElems = dense<[121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>,
    alignment = 16 : i64>>

module @OutliningWithGroupSparseTensor {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x16x1034x2058xf16>
        DataInfo "input2" : tensor<1x16x1024x2048xf16>
        DataInfo "input3" : tensor<1x16x1024x2048xf16>
        DataInfo "input4" : tensor<1x16x1034x2058xf16>
    }
    outputsInfo : {
        DataInfo "output" : tensor<1x16x1024x2048xf16>
    }

  func.func @main(%arg0: tensor<1x16x1034x2058xf16, {order = #NHWC}>, %arg1: tensor<1x16x1024x2048xf16, {order = #NHWC}>, %arg2: tensor<1x16x1024x2048xf16, {order = #NHWC}>, %arg3: tensor<1x16x1034x2058xf16, {order = #NHWC}>) -> tensor<1x16x1024x2048xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x1x1x2048xi1> = dense<1> : tensor<16x16x11x11xsi8, {order = #NHWC}>, [#const.CastElemType<!qElemType>, #const.GetSparsityMap]
    %cst_1 = const.Declare tensor<16x16x11x11x!qElemType, {order = #NHWC}> = dense<1> : tensor<16x16x11x11xsi8, {order = #NHWC}>, [#const.CastElemType<!qElemType>, #const.Sparsify<false>]
    %0 = VPU.GroupSparseTensor(%cst_1, %cst_0) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>} -> !SparseTensorType
    %1 = VPU.GroupSparseTensor(%cst_1, %cst_0) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>} -> !SparseTensorType

    %2 = VPU.NCE.Convolution(%arg0, %1) {
      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 0.0069205216132104397 : f64>,
      rawFilterShape = [16, 16, 11, 11], strides = [1, 1], tilingStrategy = [1, 1, 47, 1]
      } : tensor<1x16x1034x2058xf16, {order = #NHWC}>, !SparseTensorType -> tensor<1x16x1024x2048xf16, {order = #NHWC}>

    %3 = VPU.VerticalFusion (%2 as %arg4: tensor<1x16x1024x2048xf16, {order = #NHWC}>, %arg1 as %arg5: tensor<1x16x1024x2048xf16, {order = #NHWC}>, %arg2 as %arg6: tensor<1x16x1024x2048xf16, {order = #NHWC}>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 1, 98]} -> tensor<1x16x1024x2048xf16, {order = #NHWC}> {
      %4 = VPU.NCE.Eltwise(%arg4, %arg5) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x1024x2048xf16, {order = #NHWC}>
      %5 = VPU.NCE.Eltwise(%4, %arg6) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x1024x2048xf16, {order = #NHWC}>
      VPU.Yield %5
    }

    %6 = VPU.NCE.Convolution(%arg3, %0) {
      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 0.0069205216132104397 : f64>,
      rawFilterShape = [16, 16, 11, 11], strides = [1, 1], tilingStrategy = [1, 1, 47, 1]
      } : tensor<1x16x1034x2058xf16, {order = #NHWC}>, !SparseTensorType -> tensor<1x16x1024x2048xf16, {order = #NHWC}>

    %7 = VPU.NCE.Eltwise(%6, %3) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x1024x2048xf16, {order = #NHWC}>
    return %7 : tensor<1x16x1024x2048xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @OutliningWithGroupSparseTensor

// CHECK: func.func private @main_vf1([[ARG0:%.+]]: tensor<1x16x1034x2058xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x16x1024x2048xf16, {order = #NHWC}>, [[ARG2:%.+]]: tensor<1x16x1024x2048xf16, {order = #NHWC}>, [[ARG3:%.+]]: tensor<1x16x1034x2058xf16, {order = #NHWC}>) -> tensor<1x16x1024x2048xf16, {order = #NHWC}> {
// CHECK:  [[CST0:%.+]] = const.Declare tensor<16x1x1x2048xi1>
// CHECK:  [[CST1:%.+]] = const.Declare tensor<16x16x11x11x!qElemType, {order = #NHWC}>
// CHECK:  [[GROUP_SPARSE1:%.+]] = VPU.GroupSparseTensor([[CST1]], [[CST0]])
// CHECK:  [[GROUP_SPARSE2:%.+]] = VPU.GroupSparseTensor([[CST1]], [[CST0]])
// CHECK:  [[CONV1:%.+]] = VPU.NCE.Convolution([[ARG0]], [[GROUP_SPARSE2]])
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[CONV1]] as {{[^:]+}}: tensor<1x16x1024x2048xf16, {order = #NHWC}>, [[ARG1]] as {{[^:]+}}: tensor<1x16x1024x2048xf16, {order = #NHWC}>, [[ARG2]] as {{[^:]+}}: tensor<1x16x1024x2048xf16, {order = #NHWC}>)
// CHECK:    [[ELTWISE1:%.+]] = VPU.NCE.Eltwise({{[^:]+}}, {{[^:]+}})
// CHECK:    [[ELTWISE2:%.+]] = VPU.NCE.Eltwise([[ELTWISE1]], {{[^:]+}})
// CHECK:    VPU.Yield [[ELTWISE2]]
// CHECK:  [[CONV2:%.+]] = VPU.NCE.Convolution([[ARG3]], [[GROUP_SPARSE1]])
// CHECK:  [[ELTWISE3:%.+]] = VPU.NCE.Eltwise([[CONV2]], [[VF]])
// CHECK:  return [[ELTWISE3]]

// CHECK-NOT:  func.func private @main_vf2

// CHECK: func.func @main([[INPUT1:%.+]]: tensor<1x16x1034x2058xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x16x1024x2048xf16, {order = #NHWC}>, [[INPUT3:%.+]]: tensor<1x16x1024x2048xf16, {order = #NHWC}>, [[INPUT4:%.+]]: tensor<1x16x1034x2058xf16, {order = #NHWC}>) -> tensor<1x16x1024x2048xf16, {order = #NHWC}> {
// CHECK:  [[CALL:%.+]] = call @main_vf1([[INPUT1]], [[INPUT2]], [[INPUT3]], [[INPUT4]])

// CHECK:  return [[CALL]]
