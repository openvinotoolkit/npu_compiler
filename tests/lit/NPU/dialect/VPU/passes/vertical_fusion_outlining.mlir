//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW weights-table-reuse-mode=VF_ENABLED" --vertical-fusion-outlining="vf-outlining-tile-threshold=10 vf-outlining-instance-threshold=4" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @VerticalFusionSimpleOutlining {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x48x1024x4xf16>
        DataInfo "input2" : tensor<4096x48x1x1xf16>
        DataInfo "input3" : tensor<48x4096x1x1xf16>
    }
    outputsInfo : {
        DataInfo "output1" : tensor<1x48x1024x4xf16>
        DataInfo "output2" : tensor<1x48x1024x4xf16>
    }

  func.func @main(
      %arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
      %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>,
      %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>) {

      %0 = VPU.VerticalFusion (
        %arg0 as %arg4: tensor<1x48x1024x4xf16, {order = #NHWC}>,
        %arg1 as %arg5: tensor<4096x48x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %1 = VPU.VerticalFusion (
        %0 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.SoftMax(%arg4) {
            axisInd = 1 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %2 = VPU.VerticalFusion (
        %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %3 = VPU.VerticalFusion (
        %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      return %2, %3 : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @VerticalFusionSimpleOutlining

// CHECK: DataInfo "input1" : tensor<1x48x1024x4xf16>
// CHECK: DataInfo "input2" : tensor<4096x48x1x1xf16>
// CHECK: DataInfo "input3" : tensor<48x4096x1x1xf16>
// CHECK: DataInfo "output1" : tensor<1x48x1024x4xf16>
// CHECK: DataInfo "output2" : tensor<1x48x1024x4xf16>

// CHECK: func.func private @main_vf1([[ARG0:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG0]] as {{[^:]+}}: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1]] as {{[^:]+}}: tensor<4096x48x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf2([[ARG0:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region}
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG0]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.SoftMax({{[^:]+}}) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf3([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf4([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK: func.func @main([[INPUT1:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>, [[INPUT3:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>) {
// CHECK:  [[CALL1:%.+]] = call @main_vf1([[INPUT1]], [[INPUT2]]) : (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL2:%.+]] = call @main_vf2([[CALL1]]) : (tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL3:%.+]] = call @main_vf3([[INPUT3]], [[CALL2]]) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL4:%.+]] = call @main_vf4([[INPUT3]], [[CALL2]]) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK:  return [[CALL3]], [[CALL4]] : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @VerticalFusionWithViewOpOutlining {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x48x1024x4xf16>
        DataInfo "input2" : tensor<4096x48x1x1xf16>
        DataInfo "input3" : tensor<48x4096x1x1xf16>
    }
    outputsInfo : {
        DataInfo "output1" : tensor<1x48x1024x4xf16>
        DataInfo "output2" : tensor<1x48x1024x4xf16>
    }

  func.func @main(
      %arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
      %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>,
      %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>) {

      %0 = VPU.VerticalFusion (
        %arg0 as %arg4: tensor<1x48x1024x4xf16, {order = #NHWC}>,
        %arg1 as %arg5: tensor<4096x48x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %1 = VPU.VerticalFusion (
        %0 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.SoftMax(%arg4) {
            axisInd = 1 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %2 = VPU.VerticalFusion (
        %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %3 = VPU.ShapeCast { shape = [1, 4096, 1024, 4] } inputs(%1 : tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

      %4 = VPU.VerticalFusion (
        %3 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %5 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %5
      }

      return %2, %4 : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @VerticalFusionWithViewOpOutlining

// CHECK: DataInfo "input1" : tensor<1x48x1024x4xf16>
// CHECK: DataInfo "input2" : tensor<4096x48x1x1xf16>
// CHECK: DataInfo "input3" : tensor<48x4096x1x1xf16>
// CHECK: DataInfo "output1" : tensor<1x48x1024x4xf16>
// CHECK: DataInfo "output2" : tensor<1x48x1024x4xf16>

// CHECK: func.func private @main_vf1([[ARG0:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG0]] as {{[^:]+}}: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1]] as {{[^:]+}}: tensor<4096x48x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf2([[ARG0:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region}
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG0]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.SoftMax({{[^:]+}}) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf3([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf4([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VIEW:%.+]] = VPU.ShapeCast {shape = [1, 4096, 1024, 4]} inputs([[ARG1]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[VIEW]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK: func.func @main([[INPUT1:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>, [[INPUT3:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>) {
// CHECK:  [[CALL1:%.+]] = call @main_vf1([[INPUT1]], [[INPUT2]]) : (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL2:%.+]] = call @main_vf2([[CALL1]]) : (tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL3:%.+]] = call @main_vf3([[INPUT3]], [[CALL2]]) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL4:%.+]] = call @main_vf4([[INPUT3]], [[CALL2]]) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK:  return [[CALL3]], [[CALL4]] : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @VerticalFusionWithSliceOpOutlining {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x48x1024x4xf16>
        DataInfo "input2" : tensor<4096x48x1x1xf16>
        DataInfo "input3" : tensor<48x4096x1x1xf16>
    }
    outputsInfo : {
        DataInfo "output1" : tensor<1x48x1024x4xf16>
        DataInfo "output2" : tensor<1x40x1024x4xf16>
    }

  func.func @main(
      %arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
      %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>,
      %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x40x1024x4xf16, {order = #NHWC}>) {
      %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
      %cst_1 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
      %cst_2 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

      %0 = VPU.VerticalFusion (
        %arg0 as %arg4: tensor<1x48x1024x4xf16, {order = #NHWC}>,
        %arg1 as %arg5: tensor<4096x48x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %1 = VPU.VerticalFusion (
        %0 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.SoftMax(%arg4) {
            axisInd = 1 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %2 = VPU.VerticalFusion (
        %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %3 = VPU.VerticalFusion (
        %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %4 = VPU.Slice %3 [0, 8, 0, 0] [1, 40, 1024, 4] : tensor<1x48x1024x4xf16, {order = #NHWC}> to tensor<1x40x1024x4xf16, {order = #NHWC}>

      return %2, %4 : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x40x1024x4xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @VerticalFusionWithSliceOpOutlining

// CHECK: DataInfo "input1" : tensor<1x48x1024x4xf16>
// CHECK: DataInfo "input2" : tensor<4096x48x1x1xf16>
// CHECK: DataInfo "input3" : tensor<48x4096x1x1xf16>
// CHECK: DataInfo "output1" : tensor<1x48x1024x4xf16>
// CHECK: DataInfo "output2" : tensor<1x40x1024x4xf16>

// CHECK: func.func private @main_vf1([[ARG0:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG0]] as {{[^:]+}}: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1]] as {{[^:]+}}: tensor<4096x48x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf2([[ARG0:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region}
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG0]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.SoftMax({{[^:]+}}) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf3([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x48x1024x4xf16, {order = #NHWC}>


// CHECK: func.func private @main_vf4([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x40x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  [[SLICE:%.+]] = VPU.Slice [[VF]] [0, 8, 0, 0] [1, 40, 1024, 4] : tensor<1x48x1024x4xf16, {order = #NHWC}> to tensor<1x40x1024x4xf16, {order = #NHWC}>
// CHECK:  return [[SLICE]] : tensor<1x40x1024x4xf16, {order = #NHWC}>

// CHECK: func.func @main([[INPUT1:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>, [[INPUT3:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x40x1024x4xf16, {order = #NHWC}>) {
// CHECK:  [[CALL1:%.+]] = call @main_vf1([[INPUT1]], [[INPUT2]]) : (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL2:%.+]] = call @main_vf2([[CALL1]]) : (tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL3:%.+]] = call @main_vf3([[INPUT3]], [[CALL2]]) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL4:%.+]] = call @main_vf4([[INPUT3]], [[CALL2]]) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x40x1024x4xf16, {order = #NHWC}>

// CHECK:  return [[CALL3]], [[CALL4]] : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x40x1024x4xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @VerticalFusionWithSlicedConstantOutlining {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x48x1024x4xf16>
        DataInfo "input2" : tensor<4096x48x1x1xf16>
        DataInfo "input3" : tensor<48x4096x1x1xf16>
    }
    outputsInfo : {
        DataInfo "output1" : tensor<1x48x1024x4xf16>
        DataInfo "output2" : tensor<1x48x1024x4xf16>
    }

  func.func @main(
      %arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
      %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>,
      %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>) {

      %0 = VPU.VerticalFusion (
        %arg0 as %arg4: tensor<1x48x1024x4xf16, {order = #NHWC}>,
        %arg1 as %arg5: tensor<4096x48x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %1 = VPU.VerticalFusion (
        %0 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.SoftMax(%arg4) {
            axisInd = 1 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %2 = VPU.VerticalFusion (
        %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %3 = VPU.VerticalFusion (
        %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      return %2, %3 : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @VerticalFusionWithSlicedConstantOutlining

// CHECK: DataInfo "input1" : tensor<1x48x1024x4xf16>
// CHECK: DataInfo "input2" : tensor<4096x48x1x1xf16>
// CHECK: DataInfo "input3" : tensor<48x4096x1x1xf16>
// CHECK: DataInfo "output1" : tensor<1x48x1024x4xf16>
// CHECK: DataInfo "output2" : tensor<1x48x1024x4xf16>

// CHECK: func.func private @main_vf1([[ARG0:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG0]] as {{[^:]+}}: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1]] as {{[^:]+}}: tensor<4096x48x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf2([[ARG0:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region}
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG0]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.SoftMax({{[^:]+}}) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf3([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x48x1024x4xf16, {order = #NHWC}>


// CHECK: func.func private @main_vf4([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK: func.func @main([[INPUT1:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>, [[INPUT3:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>) {
// CHECK:  [[CALL1:%.+]] = call @main_vf1([[INPUT1]], [[INPUT2]]) : (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL2:%.+]] = call @main_vf2([[CALL1]]) : (tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL3:%.+]] = call @main_vf3([[INPUT3]], [[CALL2]]) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL4:%.+]] = call @main_vf4([[INPUT3]], [[CALL2]]) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK:  return [[CALL3]], [[CALL4]] : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @VerticalFusionWithConcatOpOutlining {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x48x1024x4xf16>
        DataInfo "input2" : tensor<4096x48x1x1xf16>
        DataInfo "input3" : tensor<48x4096x1x1xf16>
    }
    outputsInfo : {
        DataInfo "output" : tensor<1x96x1024x4xf16>
    }

  func.func @main(
      %arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
      %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>,
      %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x96x1024x4xf16, {order = #NHWC}> {

      %0 = VPU.VerticalFusion (
        %arg0 as %arg4: tensor<1x48x1024x4xf16, {order = #NHWC}>,
        %arg1 as %arg5: tensor<4096x48x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %1 = VPU.VerticalFusion (
        %0 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.SoftMax(%arg4) {
            axisInd = 1 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %2 = VPU.VerticalFusion (
        %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %3 = VPU.VerticalFusion (
        %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %4 = VPU.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}> -> tensor<1x96x1024x4xf16, {order = #NHWC}>

      return %4 : tensor<1x96x1024x4xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @VerticalFusionWithConcatOpOutlining

// CHECK: DataInfo "input1" : tensor<1x48x1024x4xf16>
// CHECK: DataInfo "input2" : tensor<4096x48x1x1xf16>
// CHECK: DataInfo "input3" : tensor<48x4096x1x1xf16>
// CHECK: DataInfo "output" : tensor<1x96x1024x4xf16>

// CHECK: func.func private @main_vf1([[ARG0:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG0]] as {{[^:]+}}: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1]] as {{[^:]+}}: tensor<4096x48x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf2([[ARG0:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region}
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG0]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.SoftMax({{[^:]+}}) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf3([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf4([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG2:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>) -> tensor<1x96x1024x4xf16, {order = #NHWC}> {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  [[CONCAT:%.+]] = VPU.Concat([[ARG2]], [[VF]])
// CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]}
// CHECK-SAME:               : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}> -> tensor<1x96x1024x4xf16, {order = #NHWC}>
// CHECK:  return [[CONCAT]] : tensor<1x96x1024x4xf16, {order = #NHWC}>

// CHECK: func.func @main([[INPUT1:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>, [[INPUT3:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x96x1024x4xf16, {order = #NHWC}> {
// CHECK:  [[CALL1:%.+]] = call @main_vf1([[INPUT1]], [[INPUT2]]) : (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL2:%.+]] = call @main_vf2([[CALL1]]) : (tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL3:%.+]] = call @main_vf3([[INPUT3]], [[CALL2]]) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL4:%.+]] = call @main_vf4([[INPUT3]], [[CALL2]], [[CALL3]]) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>) -> tensor<1x96x1024x4xf16, {order = #NHWC}>

// CHECK:  return [[CALL4]] : tensor<1x96x1024x4xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @NotEnoughInstancesForOutlining {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x48x1024x4xf16>
        DataInfo "input2" : tensor<4096x48x1x1xf16>
        DataInfo "input3" : tensor<48x4096x1x1xf16>
    }
    outputsInfo : {
        DataInfo "output" : tensor<1x48x1024x4xf16>
    }

  func.func @main(
      %arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
      %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>,
      %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {

      %0 = VPU.VerticalFusion (
        %arg0 as %arg4: tensor<1x48x1024x4xf16, {order = #NHWC}>,
        %arg1 as %arg5: tensor<4096x48x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %1 = VPU.VerticalFusion (
        %0 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.SoftMax(%arg4) {
            axisInd = 1 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %2 = VPU.VerticalFusion (
        %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      return %2 : tensor<1x48x1024x4xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @NotEnoughInstancesForOutlining

// CHECK-NOT: func.func private

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @OutliningMixedCase {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x48x1024x4xf16>
        DataInfo "input2" : tensor<4096x48x1x1xf16>
        DataInfo "input3" : tensor<48x4096x1x1xf16>
    }
    outputsInfo : {
        DataInfo "output" : tensor<1x190x1024x4xf16>
    }

  func.func @main(
      %arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
      %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>,
      %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x190x1024x4xf16, {order = #NHWC}> {

      %0 = VPU.Slice %arg0 [0, 8, 0, 0] [1, 40, 1024, 4] : tensor<1x48x1024x4xf16, {order = #NHWC}> to tensor<1x40x1024x4xf16, {order = #NHWC}>
      %1 = VPU.Expand (%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x40x1024x4xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>

      %2 = VPU.VerticalFusion (
        %1 as %arg4: tensor<1x48x1024x4xf16, {order = #NHWC}>,
        %arg1 as %arg5: tensor<4096x48x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 9, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %13 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %13
      }

      %3 = VPU.VerticalFusion (
        %2 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 9, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %13 = VPU.SoftMax(%arg4) {
            axisInd = 1 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %13
      }

      %4 = VPU.ShapeCast { shape = [1, 4096, 1024, 4] } inputs(%3 : tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      %5 = VPU.ShapeCast { shape = [1, 4096, 1024, 4] } inputs(%4 : tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

      %6 = VPU.VerticalFusion (
        %5 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %13 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %13
      }

      %7 = VPU.VerticalFusion (
        %3 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %13 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %13
      }

      %8 = VPU.VerticalFusion (
        %3 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %13 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %13
      }

      %9 = VPU.VerticalFusion (
        %5 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %13 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %13
      }

      %10 = VPU.Concat(%6, %7, %8, %9) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0]]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}> -> tensor<1x192x1024x4xf16, {order = #NHWC}>
      %11 = VPU.Slice %10 [0, 2, 0, 0] [1, 190, 1024, 4] : tensor<1x192x1024x4xf16, {order = #NHWC}> to tensor<1x190x1024x4xf16, {order = #NHWC}>

      return %11 : tensor<1x190x1024x4xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @OutliningMixedCase

// CHECK: DataInfo "input1" : tensor<1x48x1024x4xf16>
// CHECK: DataInfo "input2" : tensor<4096x48x1x1xf16>
// CHECK: DataInfo "input3" : tensor<48x4096x1x1xf16>
// CHECK: DataInfo "output" : tensor<1x190x1024x4xf16>

// CHECK: func.func private @main_vf1([[ARG0:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>) -> (tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>) {
// CHECK:  [[SLICE:%.+]] = VPU.Slice [[ARG0]] [0, 8, 0, 0] [1, 40, 1024, 4] : tensor<1x48x1024x4xf16, {order = #NHWC}> to tensor<1x40x1024x4xf16, {order = #NHWC}>
// CHECK:  [[EXPAND:%.+]] = VPU.Expand([[SLICE]])
// CHECK-SAME{LITERAL}:      {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]}
// CHECK-SAME:               : tensor<1x40x1024x4xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[EXPAND]] as {{[^:]+}}: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG1]] as {{[^:]+}}: tensor<4096x48x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 9, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
// CHECK:                 [[OP:%.+]] =  VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]}
// CHECK-SAME:                 -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:                 VPU.Yield [[OP]]
// CHECK:                }
// CHECK:  [[VF2:%.+]] = VPU.VerticalFusion ([[VF1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 9, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
// CHECK:                 [[OP:%.+]] =  VPU.SoftMax({{[^:]+}}) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:                 VPU.Yield [[OP]]
// CHECK:                }
// CHECK:  [[SC1:%.+]] = VPU.ShapeCast {shape = [1, 4096, 1024, 4]} inputs([[VF2]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:  [[SC2:%.+]] = VPU.ShapeCast {shape = [1, 4096, 1024, 4]} inputs([[SC1]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK:  return [[VF2]], [[SC2]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf2([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf3([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf4([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  return [[VF]] : tensor<1x48x1024x4xf16, {order = #NHWC}>

// CHECK: func.func private @main_vf5([[ARG0:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG2:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG3:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[ARG4:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>) -> tensor<1x190x1024x4xf16, {order = #NHWC}> {
// CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[ARG1]] as {{[^:]+}}: tensor<1x4096x1024x4xf16, {order = #NHWC}>, [[ARG0]] as {{[^:]+}}: tensor<48x4096x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
// CHECK:                [[OP:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]}
// CHECK-SAME:                -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:                VPU.Yield [[OP]]
// CHECK:               }
// CHECK:  [[CONCAT:%.+]] = VPU.Concat([[ARG2]], [[ARG3]], [[ARG4]], [[VF]])
// CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0]]}
// CHECK-SAME:               : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}> -> tensor<1x192x1024x4xf16, {order = #NHWC}>
// CHECK:  [[SLICE:%.+]] = VPU.Slice [[CONCAT]] [0, 2, 0, 0] [1, 190, 1024, 4] : tensor<1x192x1024x4xf16, {order = #NHWC}> to tensor<1x190x1024x4xf16, {order = #NHWC}>
// CHECK:  return [[SLICE]] : tensor<1x190x1024x4xf16, {order = #NHWC}>

// CHECK: func.func @main([[INPUT1:%.+]]: tensor<1x48x1024x4xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<4096x48x1x1xf16, {order = #NHWC}>, [[INPUT3:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x190x1024x4xf16, {order = #NHWC}> {
// CHECK:  [[CALL1:%.+]]:2 = call @main_vf1([[INPUT1]], [[INPUT2]]) : (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>) -> (tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>)
// CHECK:  [[CALL2:%.+]] = call @main_vf2([[INPUT3]], [[CALL1]]#1) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL3:%.+]] = call @main_vf3([[INPUT3]], [[CALL1]]#0) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL4:%.+]] = call @main_vf4([[INPUT3]], [[CALL1]]#0) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}>
// CHECK:  [[CALL5:%.+]] = call @main_vf5([[INPUT3]], [[CALL1]]#1, [[CALL2]], [[CALL3]], [[CALL4]]) : (tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>) -> tensor<1x190x1024x4xf16, {order = #NHWC}>

// CHECK:  return [[CALL5]] : tensor<1x190x1024x4xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @SkipVerticalFusionOutliningForExistingCalls {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x48x1024x4xf16>
        DataInfo "input2" : tensor<4096x48x1x1xf16>
        DataInfo "input3" : tensor<48x4096x1x1xf16>
    }
    outputsInfo : {
        DataInfo "output1" : tensor<1x48x1024x4xf16>
        DataInfo "output2" : tensor<1x48x1024x4xf16>
    }

  func.func private @function(%arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
      %0 = VPU.SoftMax(%arg0) { axisInd = 1 : i64} : tensor<1x48x1024x4xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
      return %0 : tensor<1x48x1024x4xf16, {order = #NHWC}>
  }

  func.func @main(
      %arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
      %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>,
      %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> (tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>) {

      %call = func.call @function(%arg0) : (tensor<1x48x1024x4xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}>

      %0 = VPU.VerticalFusion (
        %call as %arg4: tensor<1x48x1024x4xf16, {order = #NHWC}>,
        %arg1 as %arg5: tensor<4096x48x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %1 = VPU.VerticalFusion (
        %0 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]}
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.SoftMax(%arg4) {
            axisInd = 1 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>
              -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %2 = VPU.VerticalFusion (
        %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      %3 = VPU.VerticalFusion (
        %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 15, 1]}
              -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
      }

      return %2, %3 : tensor<1x48x1024x4xf16, {order = #NHWC}>, tensor<1x48x1024x4xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @SkipVerticalFusionOutliningForExistingCalls

// CHECK:     func.func @main
// CHECK:       call @function
// CHECK-NOT:   call
