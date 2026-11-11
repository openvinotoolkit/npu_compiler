//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --create-new-weight-tables-data %s | FileCheck %s
// REQUIRES: platform-NPU6010

!qElemType = !quant.uniform<i8:f16:0, {1.000000e+00:1,1.000000e+00:2,1.000000e+00:3,1.000000e+00:4,1.000000e+00:5,1.000000e+00:6,1.000000e+00:7,1.000000e+00:8,1.000000e+00:9,1.000000e+00:10,1.000000e+00:11,1.000000e+00:12,1.000000e+00:13,1.000000e+00:14,1.000000e+00:15,1.000000e+00:16}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedInput = !VPU.DistributedTensor<
    1x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights = !VPU.DistributedTensor<
    16x16x2x2x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedBiasAndScaleTable = !VPU.DistributedTensor<
    16x1x1x1xf32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedZeroPointTable = !VPU.DistributedTensor<
    16x1x1x1xi8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedOutput = !VPU.DistributedTensor<
    1x16x3x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

func.func @Conv2x2_ZeroPointTable_Duplicated_I8(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x16x3x3xf16> {
  %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
  %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
  %weights = const.Declare tensor<16x16x2x2x!qElemType, {order = #NHWC}> = dense<1> : tensor<16x16x2x2xsi8>, [#const.Reorder<#NHWC>]
  %zero_points = const.Declare tensor<16x1x1x1xsi8> = dense<[[[[1]]], [[[2]]], [[[3]]], [[[4]]], [[[5]]], [[[6]]], [[[7]]], [[[8]]], [[[9]]], [[[10]]], [[[11]]], [[[12]]], [[[13]]], [[[14]]], [[[15]]], [[[16]]]]> : tensor<16x1x1x1xsi8>
  %0 = VPU.ZeroPointTable(%zero_points) : tensor<16x1x1x1xsi8> -> tensor<16x1x1x1xi8>
  %1 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %2 = VPU.Copy(%weights) {out_mem_space = @CMX_NN} : tensor<16x16x2x2x!qElemType, {order = #NHWC}> -> !DistributedWeights
  %3 = VPU.Copy(%bias_table) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %4 = VPU.Copy(%scale_table) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %5 = VPU.Copy(%0) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xi8> -> !DistributedZeroPointTable
  %6 = VPU.NCE.Convolution(%1, %2, %3, %4, %5) rawFilterShape [16, 16, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } : !DistributedInput, !DistributedWeights, !DistributedBiasAndScaleTable, !DistributedBiasAndScaleTable, !DistributedZeroPointTable -> !DistributedOutput {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 3 : i64}
          }
  %7 = VPU.Copy(%6) : !DistributedOutput -> tensor<1x16x3x3xf16>
  return %7 : tensor<1x16x3x3xf16>

  // CHECK:       [[CST_BIAS:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x2x2x!qElemType, {order = #NHWC}>
  // CHECK:       [[CST_ZP:%.+]] = const.Declare tensor<16x1x1x1xsi8>
  // CHECK:       [[ZP_TABLE:%.+]] = VPU.ZeroPointTable([[CST_ZP]]) {workloadSizes = [16], zeroPointTableData = [1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]} : tensor<16x1x1x1xsi8> -> tensor<32x1x1x1xi8>
  // CHECK:       [[ZP_TABLE_CMX:%.+]] = VPU.Copy([[ZP_TABLE]]) {out_mem_space = @CMX_NN} : tensor<32x1x1x1xi8> -> !VPU.DistributedTensor<32x1x1x1xi8
  // CHECK-SAME:  mode = "DUPLICATED", num_clusters = 4 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 0 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 1 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 2 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 3 : i64}
}

// -----

!qElemType = !quant.uniform<i8:f16:0, {1.000000e+00:1,1.000000e+00:2,1.000000e+00:3,1.000000e+00:4,1.000000e+00:5,1.000000e+00:6,1.000000e+00:7,1.000000e+00:8,1.000000e+00:9,1.000000e+00:10,1.000000e+00:11,1.000000e+00:12,1.000000e+00:13,1.000000e+00:14,1.000000e+00:15,1.000000e+00:16,1.000000e+00:17,1.000000e+00:18,1.000000e+00:19,1.000000e+00:20,1.000000e+00:21,1.000000e+00:22,1.000000e+00:23,1.000000e+00:24,1.000000e+00:25,1.000000e+00:26,1.000000e+00:27,1.000000e+00:28,1.000000e+00:29,1.000000e+00:30,1.000000e+00:31,1.000000e+00:32,1.000000e+00:33,1.000000e+00:34,1.000000e+00:35,1.000000e+00:36,1.000000e+00:37,1.000000e+00:38,1.000000e+00:39,1.000000e+00:40,1.000000e+00:41,1.000000e+00:42,1.000000e+00:43,1.000000e+00:44,1.000000e+00:45,1.000000e+00:46,1.000000e+00:47,1.000000e+00:48}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedInput = !VPU.DistributedTensor<
    1x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights = !VPU.DistributedTensor<
    48x16x2x2x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2]],
    compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]],
    memory_shapes = [[16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2]],
    memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]]
}>

!DistributedBiasAndScaleTable = !VPU.DistributedTensor<
    48x1x1x1xf32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]]
}>

!DistributedZeroPointTable = !VPU.DistributedTensor<
    48x1x1x1xi8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]]
}>

!DistributedOutput = !VPU.DistributedTensor<
    1x48x3x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]
}>

func.func @Conv2x2_ZeroPointTable_Segmented_EvenWorkloadSizes_I8(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x48x3x3xf16> {
  %bias_table = const.Declare tensor<48x1x1x1xf32> = dense<0.000000e+00> : tensor<48x1x1x1xf32>
  %scale_table = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>
  %weights = const.Declare tensor<48x16x2x2x!qElemType, {order = #NHWC}> = dense<1> : tensor<48x16x2x2xsi8>, [#const.Reorder<#NHWC>]
  %zero_points = const.Declare tensor<48x1x1x1xsi8> = dense<[[[[1]]], [[[2]]], [[[3]]], [[[4]]], [[[5]]], [[[6]]], [[[7]]], [[[8]]], [[[9]]], [[[10]]], [[[11]]], [[[12]]], [[[13]]], [[[14]]], [[[15]]], [[[16]]], [[[17]]], [[[18]]], [[[19]]], [[[20]]], [[[21]]], [[[22]]], [[[23]]], [[[24]]], [[[25]]], [[[26]]], [[[27]]], [[[28]]], [[[29]]], [[[30]]], [[[31]]], [[[32]]], [[[33]]], [[[34]]], [[[35]]], [[[36]]], [[[37]]], [[[38]]], [[[39]]], [[[40]]], [[[41]]], [[[42]]], [[[43]]], [[[44]]], [[[45]]], [[[46]]], [[[47]]], [[[48]]]]> : tensor<48x1x1x1xsi8>
  %0 = VPU.ZeroPointTable(%zero_points) : tensor<48x1x1x1xsi8> -> tensor<48x1x1x1xi8>
  %1 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %2 = VPU.Copy(%weights) {out_mem_space = @CMX_NN} : tensor<48x16x2x2x!qElemType, {order = #NHWC}> -> !DistributedWeights
  %3 = VPU.Copy(%bias_table) {out_mem_space = @CMX_NN} : tensor<48x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %4 = VPU.Copy(%scale_table) {out_mem_space = @CMX_NN} : tensor<48x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %5 = VPU.Copy(%0) {out_mem_space = @CMX_NN} : tensor<48x1x1x1xi8> -> !DistributedZeroPointTable
  %6 = VPU.NCE.Convolution(%1, %2, %3, %4, %5) rawFilterShape [48, 16, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } : !DistributedInput, !DistributedWeights, !DistributedBiasAndScaleTable, !DistributedBiasAndScaleTable, !DistributedZeroPointTable -> !DistributedOutput {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 16, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
          }
  %7 = VPU.Copy(%6) : !DistributedOutput -> tensor<1x48x3x3xf16>
  return %7 : tensor<1x48x3x3xf16>

  // CHECK:       [[CST_BIAS:%.+]] = const.Declare tensor<48x1x1x1xf32>
  // CHECK:       [[CST_SCALE:%.+]] = const.Declare tensor<48x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<48x16x2x2x!qElemType, {order = #NHWC}>
  // Each cluster gets 16 channels, padded to 32, total 96 with interleaved padding
  // CHECK:       [[CST_ZP:%.+]] = const.Declare tensor<48x1x1x1xsi8>
  // CHECK:       [[ZP_TABLE:%.+]] = VPU.ZeroPointTable([[CST_ZP]]) {workloadSizes = [16, 16, 16]
  // CHECK-SAME:  zeroPointTableData = [
  // CHECK-SAME:    1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:    17, 19, 18, 20, 21, 23, 22, 24, 25, 27, 26, 28, 29, 31, 30, 32, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:    33, 35, 34, 36, 37, 39, 38, 40, 41, 43, 42, 44, 45, 47, 46, 48, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  // CHECK-SAME:  ]} : tensor<48x1x1x1xsi8> -> tensor<96x1x1x1xi8>
  // CHECK:       [[ZP_TABLE_CMX:%.+]] = VPU.Copy([[ZP_TABLE]]) {out_mem_space = @CMX_NN} : tensor<96x1x1x1xi8> -> !VPU.DistributedTensor<96x1x1x1xi8
  // CHECK-SAME:  mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0]{{\]}}
  // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 0 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 16, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 1 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 2 : i64}
}

// -----

!qElemType = !quant.uniform<i8:f16:0, {1.000000e+00:1,1.000000e+00:2,1.000000e+00:3,1.000000e+00:4,1.000000e+00:5,1.000000e+00:6,1.000000e+00:7,1.000000e+00:8,1.000000e+00:9,1.000000e+00:10,1.000000e+00:11,1.000000e+00:12,1.000000e+00:13,1.000000e+00:14,1.000000e+00:15,1.000000e+00:16,1.000000e+00:17,1.000000e+00:18,1.000000e+00:19,1.000000e+00:20,1.000000e+00:21,1.000000e+00:22,1.000000e+00:23,1.000000e+00:24,1.000000e+00:25,1.000000e+00:26,1.000000e+00:27,1.000000e+00:28,1.000000e+00:29,1.000000e+00:30,1.000000e+00:31,1.000000e+00:32,1.000000e+00:33,1.000000e+00:34,1.000000e+00:35,1.000000e+00:36,1.000000e+00:37,1.000000e+00:38,1.000000e+00:39,1.000000e+00:40,1.000000e+00:41,1.000000e+00:42,1.000000e+00:43,1.000000e+00:44,1.000000e+00:45,1.000000e+00:46,1.000000e+00:47,1.000000e+00:48,1.000000e+00:49,1.000000e+00:50,1.000000e+00:51,1.000000e+00:52,1.000000e+00:53,1.000000e+00:54,1.000000e+00:55,1.000000e+00:56,1.000000e+00:57,1.000000e+00:58,1.000000e+00:59,1.000000e+00:60,1.000000e+00:61,1.000000e+00:62,1.000000e+00:63,1.000000e+00:64,1.000000e+00:65,1.000000e+00:66,1.000000e+00:67,1.000000e+00:68,1.000000e+00:69,1.000000e+00:70,1.000000e+00:71,1.000000e+00:72,1.000000e+00:73,1.000000e+00:74,1.000000e+00:75,1.000000e+00:76,1.000000e+00:77,1.000000e+00:78,1.000000e+00:79,1.000000e+00:80,1.000000e+00:81,1.000000e+00:82,1.000000e+00:83,1.000000e+00:84,1.000000e+00:85,1.000000e+00:86,1.000000e+00:87,1.000000e+00:88,1.000000e+00:89,1.000000e+00:90,1.000000e+00:91,1.000000e+00:92,1.000000e+00:93,1.000000e+00:94,1.000000e+00:95,1.000000e+00:96,1.000000e+00:97,1.000000e+00:98,1.000000e+00:99,1.000000e+00:100,1.000000e+00:101,1.000000e+00:102,1.000000e+00:103,1.000000e+00:104,1.000000e+00:105,1.000000e+00:106,1.000000e+00:107,1.000000e+00:108,1.000000e+00:109,1.000000e+00:110,1.000000e+00:111,1.000000e+00:112}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedInput = !VPU.DistributedTensor<
    1x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights = !VPU.DistributedTensor<
    112x16x2x2x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    compute_shapes = [[48, 16, 2, 2], [32, 16, 2, 2], [32, 16, 2, 2]],
    compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]],
    memory_shapes = [[48, 16, 2, 2], [32, 16, 2, 2], [32, 16, 2, 2]],
    memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]]
}>

!DistributedBiasAndScaleTable = !VPU.DistributedTensor<
    112x1x1x1xf32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    compute_shapes = [[48, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]],
    memory_shapes = [[48, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]]
}>

!DistributedZeroPointTable = !VPU.DistributedTensor<
    112x1x1x1xi8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    compute_shapes = [[48, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]],
    memory_shapes = [[48, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]]
}>

!DistributedOutput = !VPU.DistributedTensor<
    1x112x3x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3 : i64,
    compute_shapes = [[1, 48, 3, 3], [1, 32, 3, 3], [1, 32, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 80, 0, 0]],
    memory_shapes = [[1, 48, 3, 3], [1, 32, 3, 3], [1, 32, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 80, 0, 0]]
}>

func.func @Conv2x2_ZeroPointTable_Segmented_UnevenWorkloadSizes_I8(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x112x3x3xf16> {
  %bias_table = const.Declare tensor<112x1x1x1xf32> = dense<0.000000e+00> : tensor<112x1x1x1xf32>
  %scale_table = const.Declare tensor<112x1x1x1xf32> = dense<1.000000e+00> : tensor<112x1x1x1xf32>
  %weights = const.Declare tensor<112x16x2x2x!qElemType, {order = #NHWC}> = dense<1> : tensor<112x16x2x2xsi8>, [#const.Reorder<#NHWC>]
  %zero_points = const.Declare tensor<112x1x1x1xsi8> = dense<[[[[1]]], [[[2]]], [[[3]]], [[[4]]], [[[5]]], [[[6]]], [[[7]]], [[[8]]], [[[9]]], [[[10]]], [[[11]]], [[[12]]], [[[13]]], [[[14]]], [[[15]]], [[[16]]], [[[17]]], [[[18]]], [[[19]]], [[[20]]], [[[21]]], [[[22]]], [[[23]]], [[[24]]], [[[25]]], [[[26]]], [[[27]]], [[[28]]], [[[29]]], [[[30]]], [[[31]]], [[[32]]], [[[33]]], [[[34]]], [[[35]]], [[[36]]], [[[37]]], [[[38]]], [[[39]]], [[[40]]], [[[41]]], [[[42]]], [[[43]]], [[[44]]], [[[45]]], [[[46]]], [[[47]]], [[[48]]], [[[49]]], [[[50]]], [[[51]]], [[[52]]], [[[53]]], [[[54]]], [[[55]]], [[[56]]], [[[57]]], [[[58]]], [[[59]]], [[[60]]], [[[61]]], [[[62]]], [[[63]]], [[[64]]], [[[65]]], [[[66]]], [[[67]]], [[[68]]], [[[69]]], [[[70]]], [[[71]]], [[[72]]], [[[73]]], [[[74]]], [[[75]]], [[[76]]], [[[77]]], [[[78]]], [[[79]]], [[[80]]], [[[81]]], [[[82]]], [[[83]]], [[[84]]], [[[85]]], [[[86]]], [[[87]]], [[[88]]], [[[89]]], [[[90]]], [[[91]]], [[[92]]], [[[93]]], [[[94]]], [[[95]]], [[[96]]], [[[97]]], [[[98]]], [[[99]]], [[[100]]], [[[101]]], [[[102]]], [[[103]]], [[[104]]], [[[105]]], [[[106]]], [[[107]]], [[[108]]], [[[109]]], [[[110]]], [[[111]]], [[[112]]]]> : tensor<112x1x1x1xsi8>
  %0 = VPU.ZeroPointTable(%zero_points) : tensor<112x1x1x1xsi8> -> tensor<112x1x1x1xi8>
  %1 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %2 = VPU.Copy(%weights) {out_mem_space = @CMX_NN} : tensor<112x16x2x2x!qElemType, {order = #NHWC}> -> !DistributedWeights
  %3 = VPU.Copy(%bias_table) {out_mem_space = @CMX_NN} : tensor<112x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %4 = VPU.Copy(%scale_table) {out_mem_space = @CMX_NN} : tensor<112x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %5 = VPU.Copy(%0) {out_mem_space = @CMX_NN} : tensor<112x1x1x1xi8> -> !DistributedZeroPointTable
  %6 = VPU.NCE.Convolution(%1, %2, %3, %4, %5) rawFilterShape [112, 16, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } : !DistributedInput, !DistributedWeights, !DistributedBiasAndScaleTable, !DistributedBiasAndScaleTable, !DistributedZeroPointTable -> !DistributedOutput {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 48, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 48, 0, 0] outSizes [1, 32, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 80, 0, 0] outSizes [1, 32, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
          }
  %7 = VPU.Copy(%6) : !DistributedOutput -> tensor<1x112x3x3xf16>
  return %7 : tensor<1x112x3x3xf16>

  // CHECK:       [[CST_BIAS:%.+]] = const.Declare tensor<112x1x1x1xf32>
  // CHECK:       [[CST_SCALE:%.+]] = const.Declare tensor<112x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<112x16x2x2x!qElemType, {order = #NHWC}>
  // Workloads [48, 32, 32]: cluster 0 gets 48 channels (aligned to 64), clusters 1 and 2 get 32 each, total 128
  // CHECK:       [[CST_ZP:%.+]] = const.Declare tensor<112x1x1x1xsi8>
  // CHECK:       [[ZP_TABLE:%.+]] = VPU.ZeroPointTable([[CST_ZP]]) {workloadSizes = [48, 32, 32]
  // CHECK-SAME:  zeroPointTableData = [
  // CHECK-SAME:    1, 3, 17, 19, 33, 35, 2, 4, 18, 20, 34, 36, 5, 7, 21, 23, 37, 39, 6, 8, 22, 24, 38, 40, 9, 11, 25, 27, 41, 43, 10, 12, 26, 28, 42, 44, 13, 15, 29, 31, 45, 47, 14, 16, 30, 32, 46, 48, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:    49, 51, 65, 67, 50, 52, 66, 68, 53, 55, 69, 71, 54, 56, 70, 72, 57, 59, 73, 75, 58, 60, 74, 76, 61, 63, 77, 79, 62, 64, 78, 80,
  // CHECK-SAME:    81, 83, 97, 99, 82, 84, 98, 100, 85, 87, 101, 103, 86, 88, 102, 104, 89, 91, 105, 107, 90, 92, 106, 108, 93, 95, 109, 111, 94, 96, 110, 112
  // CHECK-SAME:  ]} : tensor<112x1x1x1xsi8> -> tensor<128x1x1x1xi8>
  // CHECK:       [[ZP_TABLE_CMX:%.+]] = VPU.Copy([[ZP_TABLE]]) {out_mem_space = @CMX_NN} : tensor<128x1x1x1xi8> -> !VPU.DistributedTensor<128x1x1x1xi8
  // CHECK-SAME:  mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[64, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[64, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]{{\]}}
  // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 48, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 0 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 48, 0, 0] outSizes [1, 32, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 1 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 80, 0, 0] outSizes [1, 32, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 2 : i64}
}

// -----

!qElemType = !quant.uniform<i8:f16:0, {1.000000e+00:1,1.000000e+00:2,1.000000e+00:3,1.000000e+00:4,1.000000e+00:5,1.000000e+00:6,1.000000e+00:7,1.000000e+00:8,1.000000e+00:9,1.000000e+00:10,1.000000e+00:11,1.000000e+00:12,1.000000e+00:13,1.000000e+00:14,1.000000e+00:15,1.000000e+00:16}>
!qElemType1 = !quant.uniform<i8:f16:0, {1.000000e+00:17,1.000000e+00:18,1.000000e+00:19,1.000000e+00:20,1.000000e+00:21,1.000000e+00:22,1.000000e+00:23,1.000000e+00:24,1.000000e+00:25,1.000000e+00:26,1.000000e+00:27,1.000000e+00:28,1.000000e+00:29,1.000000e+00:30,1.000000e+00:31,1.000000e+00:32}>
!qElemType2 = !quant.uniform<i8:f16:0, {1.000000e+00:33,1.000000e+00:34,1.000000e+00:35,1.000000e+00:36,1.000000e+00:37,1.000000e+00:38,1.000000e+00:39,1.000000e+00:40,1.000000e+00:41,1.000000e+00:42,1.000000e+00:43,1.000000e+00:44,1.000000e+00:45,1.000000e+00:46,1.000000e+00:47,1.000000e+00:48}>
!qElemTypeFull = !quant.uniform<i8:f16:0, {1.000000e+00:1,1.000000e+00:2,1.000000e+00:3,1.000000e+00:4,1.000000e+00:5,1.000000e+00:6,1.000000e+00:7,1.000000e+00:8,1.000000e+00:9,1.000000e+00:10,1.000000e+00:11,1.000000e+00:12,1.000000e+00:13,1.000000e+00:14,1.000000e+00:15,1.000000e+00:16,1.000000e+00:17,1.000000e+00:18,1.000000e+00:19,1.000000e+00:20,1.000000e+00:21,1.000000e+00:22,1.000000e+00:23,1.000000e+00:24,1.000000e+00:25,1.000000e+00:26,1.000000e+00:27,1.000000e+00:28,1.000000e+00:29,1.000000e+00:30,1.000000e+00:31,1.000000e+00:32,1.000000e+00:33,1.000000e+00:34,1.000000e+00:35,1.000000e+00:36,1.000000e+00:37,1.000000e+00:38,1.000000e+00:39,1.000000e+00:40,1.000000e+00:41,1.000000e+00:42,1.000000e+00:43,1.000000e+00:44,1.000000e+00:45,1.000000e+00:46,1.000000e+00:47,1.000000e+00:48}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedInput = !VPU.DistributedTensor<
    1x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights = !VPU.DistributedTensor<
    16x16x2x2x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights1 = !VPU.DistributedTensor<
    16x16x2x2x!qElemType1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights2 = !VPU.DistributedTensor<
    16x16x2x2x!qElemType2, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2], [16, 16, 2, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedBiasAndScaleTable = !VPU.DistributedTensor<
    16x1x1x1xf32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedZeroPointTable = !VPU.DistributedTensor<
    16x1x1x1xi8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedOutput = !VPU.DistributedTensor<
    1x16x3x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

func.func @Conv2x2_ZeroPointTable_Sliced_I8(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x48x3x3xf16> {
  %bias_table0 = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
  %scale_table0 = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
  %weights0 = const.Declare tensor<16x16x2x2x!qElemType, {order = #NHWC}> = dense<1> : tensor<16x16x2x2xsi8>, [#const.Reorder<#NHWC>]

  %bias_table1 = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
  %scale_table1 = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
  %weights1 = const.Declare tensor<16x16x2x2x!qElemType1, {order = #NHWC}> = dense<1> : tensor<16x16x2x2xsi8>, [#const.Reorder<#NHWC>]

  %bias_table2 = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
  %scale_table2 = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
  %weights2 = const.Declare tensor<16x16x2x2x!qElemType2, {order = #NHWC}> = dense<1> : tensor<16x16x2x2xsi8>, [#const.Reorder<#NHWC>]

  %zero_points = const.Declare tensor<48x1x1x1xsi8> = dense<[[[[1]]], [[[2]]], [[[3]]], [[[4]]], [[[5]]], [[[6]]], [[[7]]], [[[8]]], [[[9]]], [[[10]]], [[[11]]], [[[12]]], [[[13]]], [[[14]]], [[[15]]], [[[16]]], [[[17]]], [[[18]]], [[[19]]], [[[20]]], [[[21]]], [[[22]]], [[[23]]], [[[24]]], [[[25]]], [[[26]]], [[[27]]], [[[28]]], [[[29]]], [[[30]]], [[[31]]], [[[32]]], [[[33]]], [[[34]]], [[[35]]], [[[36]]], [[[37]]], [[[38]]], [[[39]]], [[[40]]], [[[41]]], [[[42]]], [[[43]]], [[[44]]], [[[45]]], [[[46]]], [[[47]]], [[[48]]]]> : tensor<48x1x1x1xsi8>
  %zp_table_dummy = VPU.ZeroPointTable(%zero_points) : tensor<48x1x1x1xsi8> -> tensor<48x1x1x1xi8>

  %zp_slice0 = VPU.Slice %zp_table_dummy [0, 0, 0, 0] [16, 1, 1, 1] : tensor<48x1x1x1xi8> to tensor<16x1x1x1xi8>
  %zp_slice1 = VPU.Slice %zp_table_dummy [16, 0, 0, 0] [16, 1, 1, 1] : tensor<48x1x1x1xi8> to tensor<16x1x1x1xi8>
  %zp_slice2 = VPU.Slice %zp_table_dummy [32, 0, 0, 0] [16, 1, 1, 1] : tensor<48x1x1x1xi8> to tensor<16x1x1x1xi8>

  // First convolution with its slice
  %input0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %weights_cmx0 = VPU.Copy(%weights0) {out_mem_space = @CMX_NN} : tensor<16x16x2x2x!qElemType, {order = #NHWC}> -> !DistributedWeights
  %bias_cmx0 = VPU.Copy(%bias_table0) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %scale_cmx0 = VPU.Copy(%scale_table0) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %zp_cmx0 = VPU.Copy(%zp_slice0) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xi8> -> !DistributedZeroPointTable
  %conv0 = VPU.NCE.Convolution(%input0, %weights_cmx0, %bias_cmx0, %scale_cmx0, %zp_cmx0) rawFilterShape [16, 16, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } : !DistributedInput,
          !DistributedWeights,
          !DistributedBiasAndScaleTable,
          !DistributedBiasAndScaleTable,
          !DistributedZeroPointTable
       -> !DistributedOutput  {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 3 : i64}
          }
  %out0 = VPU.Copy(%conv0) : !DistributedOutput -> tensor<1x16x3x3xf16>

  // Second convolution with its slice
  %input1 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %weights_cmx1 = VPU.Copy(%weights1) {out_mem_space = @CMX_NN} : tensor<16x16x2x2x!qElemType1, {order = #NHWC}> -> !DistributedWeights1
  %bias_cmx1 = VPU.Copy(%bias_table1) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %scale_cmx1 = VPU.Copy(%scale_table1) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %zp_cmx1 = VPU.Copy(%zp_slice1) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xi8> -> !DistributedZeroPointTable
  %conv1 = VPU.NCE.Convolution(%input1, %weights_cmx1, %bias_cmx1, %scale_cmx1, %zp_cmx1) rawFilterShape [16, 16, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,

          strides = [1, 1]
      } : !DistributedInput,
          !DistributedWeights1,
          !DistributedBiasAndScaleTable,
          !DistributedBiasAndScaleTable,
          !DistributedZeroPointTable
       -> !DistributedOutput  {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 3 : i64}
          }
  %out1 = VPU.Copy(%conv1) : !DistributedOutput -> tensor<1x16x3x3xf16>

  // Third convolution with its slice
  %input2 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %weights_cmx2 = VPU.Copy(%weights2) {out_mem_space = @CMX_NN} : tensor<16x16x2x2x!qElemType2, {order = #NHWC}> -> !DistributedWeights2
  %bias_cmx2 = VPU.Copy(%bias_table2) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %scale_cmx2 = VPU.Copy(%scale_table2) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %zp_cmx2 = VPU.Copy(%zp_slice2) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xi8> -> !DistributedZeroPointTable
  %conv2 = VPU.NCE.Convolution(%input2, %weights_cmx2, %bias_cmx2, %scale_cmx2, %zp_cmx2) rawFilterShape [16, 16, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } : !DistributedInput,
          !DistributedWeights2,
          !DistributedBiasAndScaleTable,
          !DistributedBiasAndScaleTable,
          !DistributedZeroPointTable
       -> !DistributedOutput  {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 3 : i64}
          }
  %out2 = VPU.Copy(%conv2) : !DistributedOutput -> tensor<1x16x3x3xf16>

  %concat = VPU.Concat(%out0, %out1, %out2) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]} : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x48x3x3xf16>
  return %concat : tensor<1x48x3x3xf16>

  // CHECK:       [[CST_BIAS0:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_SCALE0:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS0:%.+]] = const.Declare tensor<16x16x2x2x!qElemType, {order = #NHWC}>
  // CHECK:       [[CST_BIAS1:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_SCALE1:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS1:%.+]] = const.Declare tensor<16x16x2x2x!qElemType1, {order = #NHWC}>
  // CHECK:       [[CST_BIAS2:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_SCALE2:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS2:%.+]] = const.Declare tensor<16x16x2x2x!qElemType2, {order = #NHWC}>

  // CHECK-NOT:   VPU.ZeroPointTable({{.*}}) {{.*}} : tensor<48x1x1x1xsi8> -> tensor<48x1x1x1xi8>
  // CHECK-NOT:   VPU.Slice

  // CHECK:       [[CST_ZP0:%.+]] = const.Declare tensor<16x1x1x1xsi8>
  // CHECK:       [[ZP_TABLE0:%.+]] = VPU.ZeroPointTable([[CST_ZP0]]) {workloadSizes = [16]
  // CHECK-SAME:  zeroPointTableData = [1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  // CHECK-SAME:  } : tensor<16x1x1x1xsi8> -> tensor<32x1x1x1xi8>
  // CHECK:       [[CST_ZP1:%.+]] = const.Declare tensor<16x1x1x1xsi8>
  // CHECK:       [[ZP_TABLE1:%.+]] = VPU.ZeroPointTable([[CST_ZP1]]) {workloadSizes = [16]
  // CHECK-SAME:  zeroPointTableData = [17, 19, 18, 20, 21, 23, 22, 24, 25, 27, 26, 28, 29, 31, 30, 32, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  // CHECK-SAME:  } : tensor<16x1x1x1xsi8> -> tensor<32x1x1x1xi8>
  // CHECK:       [[CST_ZP2:%.+]] = const.Declare tensor<16x1x1x1xsi8>
  // CHECK:       [[ZP_TABLE2:%.+]] = VPU.ZeroPointTable([[CST_ZP2]]) {workloadSizes = [16]
  // CHECK-SAME:  zeroPointTableData = [33, 35, 34, 36, 37, 39, 38, 40, 41, 43, 42, 44, 45, 47, 46, 48, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  // CHECK-SAME:  } : tensor<16x1x1x1xsi8> -> tensor<32x1x1x1xi8>

  // CHECK:       [[ZP_TABLE_CMX0:%.+]] = VPU.Copy([[ZP_TABLE0]]) {out_mem_space = @CMX_NN} : tensor<32x1x1x1xi8> -> !VPU.DistributedTensor<32x1x1x1xi8
  // CHECK-SAME:  mode = "DUPLICATED", num_clusters = 4 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution
  // CHECK:       [[OUT0:%.+]] = VPU.Copy([[CONV0]])

  // CHECK:       [[ZP_TABLE_CMX1:%.+]] = VPU.Copy([[ZP_TABLE1]]) {out_mem_space = @CMX_NN} : tensor<32x1x1x1xi8> -> !VPU.DistributedTensor<32x1x1x1xi8
  // CHECK-SAME:  mode = "DUPLICATED", num_clusters = 4 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution
  // CHECK:       [[OUT1:%.+]] = VPU.Copy([[CONV1]])

  // CHECK:       [[ZP_TABLE_CMX2:%.+]] = VPU.Copy([[ZP_TABLE2]]) {out_mem_space = @CMX_NN} : tensor<32x1x1x1xi8> -> !VPU.DistributedTensor<32x1x1x1xi8
  // CHECK-SAME:  mode = "DUPLICATED", num_clusters = 4 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution
  // CHECK:       [[OUT2:%.+]] = VPU.Copy([[CONV2]])

  // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUT0]], [[OUT1]], [[OUT2]])
  // CHECK:       return [[CONCAT]]
}

// -----

!qElemType = !quant.uniform<i4:f16:0, {1.000000e+00:-8,1.000000e+00:-8,1.000000e+00:-8,1.000000e+00:-8,1.000000e+00:-8,1.000000e+00:-7,1.000000e+00:-7,1.000000e+00:-7,1.000000e+00:-7,1.000000e+00:-7,1.000000e+00:-6,1.000000e+00:-6,1.000000e+00:-6,1.000000e+00:-6,1.000000e+00:-6,1.000000e+00:-5,1.000000e+00:-5,1.000000e+00:-5,1.000000e+00:-5,1.000000e+00:-5,1.000000e+00:-4,1.000000e+00:-4,1.000000e+00:-4,1.000000e+00:-4,1.000000e+00:-4,1.000000e+00:-3,1.000000e+00:-3,1.000000e+00:-3,1.000000e+00:-3,1.000000e+00:-3,1.000000e+00:-2,1.000000e+00:-2,1.000000e+00:-2,1.000000e+00:-2,1.000000e+00:-2,1.000000e+00:-1,1.000000e+00:-1,1.000000e+00:-1,1.000000e+00:-1,1.000000e+00:-1,1.000000e+00:0,1.000000e+00:0,1.000000e+00:0,1.000000e+00:0,1.000000e+00:0,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:3,1.000000e+00:3,1.000000e+00:3,1.000000e+00:3,1.000000e+00:3,1.000000e+00:4,1.000000e+00:4,1.000000e+00:4,1.000000e+00:4,1.000000e+00:4,1.000000e+00:5,1.000000e+00:5,1.000000e+00:5,1.000000e+00:5,1.000000e+00:5,1.000000e+00:6,1.000000e+00:6,1.000000e+00:6,1.000000e+00:6,1.000000e+00:6,1.000000e+00:7,1.000000e+00:7,1.000000e+00:7,1.000000e+00:7,1.000000e+00:7}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedInput = !VPU.DistributedTensor<
    1x80x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 80, 4, 4], [1, 80, 4, 4], [1, 80, 4, 4], [1, 80, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 80, 4, 4], [1, 80, 4, 4], [1, 80, 4, 4], [1, 80, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights = !VPU.DistributedTensor<
    80x80x2x2x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[80, 80, 2, 2], [80, 80, 2, 2], [80, 80, 2, 2], [80, 80, 2, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[80, 80, 2, 2], [80, 80, 2, 2], [80, 80, 2, 2], [80, 80, 2, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedBiasAndScaleTable = !VPU.DistributedTensor<
    80x1x1x1xf32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedZeroPointTable = !VPU.DistributedTensor<
    80x1x1x1xi8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1], [80, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedOutput = !VPU.DistributedTensor<
    1x80x3x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 80, 3, 3], [1, 80, 3, 3], [1, 80, 3, 3], [1, 80, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 80, 3, 3], [1, 80, 3, 3], [1, 80, 3, 3], [1, 80, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

func.func @Conv2x2_ZeroPointTable_Duplicated_I4(%arg0: tensor<1x80x4x4xf16, {order = #NHWC}>) -> tensor<1x80x3x3xf16> {
  %bias_table = const.Declare tensor<80x1x1x1xf32> = dense<0.000000e+00> : tensor<80x1x1x1xf32>
  %scale_table = const.Declare tensor<80x1x1x1xf32> = dense<1.000000e+00> : tensor<80x1x1x1xf32>
  %weights = const.Declare tensor<80x80x2x2x!qElemType, {order = #NHWC}> = dense<1> : tensor<80x80x2x2xsi8>, [#const.Reorder<#NHWC>]
  %zero_points = const.Declare tensor<80x1x1x1xsi4> = dense<[[[[-8]]], [[[-8]]], [[[-8]]], [[[-8]]], [[[-8]]], [[[-7]]], [[[-7]]], [[[-7]]], [[[-7]]], [[[-7]]], [[[-6]]], [[[-6]]], [[[-6]]], [[[-6]]], [[[-6]]], [[[-5]]], [[[-5]]], [[[-5]]], [[[-5]]], [[[-5]]], [[[-4]]], [[[-4]]], [[[-4]]], [[[-4]]], [[[-4]]], [[[-3]]], [[[-3]]], [[[-3]]], [[[-3]]], [[[-3]]], [[[-2]]], [[[-2]]], [[[-2]]], [[[-2]]], [[[-2]]], [[[-1]]], [[[-1]]], [[[-1]]], [[[-1]]], [[[-1]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[2]]], [[[2]]], [[[2]]], [[[2]]], [[[2]]], [[[3]]], [[[3]]], [[[3]]], [[[3]]], [[[3]]], [[[4]]], [[[4]]], [[[4]]], [[[4]]], [[[4]]], [[[5]]], [[[5]]], [[[5]]], [[[5]]], [[[5]]], [[[6]]], [[[6]]], [[[6]]], [[[6]]], [[[6]]], [[[7]]], [[[7]]], [[[7]]], [[[7]]], [[[7]]]]> : tensor<80x1x1x1xsi4>
  %0 = VPU.ZeroPointTable(%zero_points) : tensor<80x1x1x1xsi4> -> tensor<80x1x1x1xi8>
  %1 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x80x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %2 = VPU.Copy(%weights) {out_mem_space = @CMX_NN} : tensor<80x80x2x2x!qElemType, {order = #NHWC}> -> !DistributedWeights
  %3 = VPU.Copy(%bias_table) {out_mem_space = @CMX_NN} : tensor<80x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %4 = VPU.Copy(%scale_table) {out_mem_space = @CMX_NN} : tensor<80x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %5 = VPU.Copy(%0) {out_mem_space = @CMX_NN} : tensor<80x1x1x1xi8> -> !DistributedZeroPointTable
  %6 = VPU.NCE.Convolution(%1, %2, %3, %4, %5) rawFilterShape [80, 80, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } : !DistributedInput, !DistributedWeights, !DistributedBiasAndScaleTable, !DistributedBiasAndScaleTable, !DistributedZeroPointTable -> !DistributedOutput {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 80, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 80, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 80, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 80, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 3 : i64}
          }
  %7 = VPU.Copy(%6) : !DistributedOutput -> tensor<1x80x3x3xf16>
  return %7 : tensor<1x80x3x3xf16>

  // CHECK:       [[CST_BIAS:%.+]] = const.Declare tensor<80x1x1x1xf32>
  // CHECK:       [[CST_SCALE:%.+]] = const.Declare tensor<80x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<80x80x2x2x!qElemType, {order = #NHWC}>
  // CHECK:       [[CST_ZP:%.+]] = const.Declare tensor<80x1x1x1xsi4>
  // CHECK:       [[ZP_TABLE:%.+]] = VPU.ZeroPointTable([[CST_ZP]]) {workloadSizes = [80], zeroPointTableData = [-120, -69, -18, 33, 84, -120, -69, -2, 33, 85, -104, -52, -1, 34, 101, -103, -52, -1, 50, 101, -87, -36, 0, 51, 102, -87, -35, 0, 51, 118, -86, -19, 16, 68, 119, -70, -19, 17, 68, 119, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]} : tensor<80x1x1x1xsi4> -> tensor<64x1x1x1xi8>
  // CHECK:       [[ZP_TABLE_CMX:%.+]] = VPU.Copy([[ZP_TABLE]]) {out_mem_space = @CMX_NN} : tensor<64x1x1x1xi8> -> !VPU.DistributedTensor<64x1x1x1xi8
  // CHECK-SAME:  mode = "DUPLICATED", num_clusters = 4 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 80, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 0 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 80, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 1 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 80, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 2 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 80, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 3 : i64}
}

// -----

!qElemType = !quant.uniform<i4:f16:0, {1.000000e+00:-8,1.000000e+00:-8,1.000000e+00:-8,1.000000e+00:-8,1.000000e+00:-8,1.000000e+00:-8,1.000000e+00:-8,1.000000e+00:-7,1.000000e+00:-7,1.000000e+00:-7,1.000000e+00:-7,1.000000e+00:-7,1.000000e+00:-7,1.000000e+00:-7,1.000000e+00:-6,1.000000e+00:-6,1.000000e+00:-6,1.000000e+00:-6,1.000000e+00:-6,1.000000e+00:-6,1.000000e+00:-6,1.000000e+00:-5,1.000000e+00:-5,1.000000e+00:-5,1.000000e+00:-5,1.000000e+00:-5,1.000000e+00:-5,1.000000e+00:-5,1.000000e+00:-4,1.000000e+00:-4,1.000000e+00:-4,1.000000e+00:-4,1.000000e+00:-4,1.000000e+00:-4,1.000000e+00:-4,1.000000e+00:-3,1.000000e+00:-3,1.000000e+00:-3,1.000000e+00:-3,1.000000e+00:-3,1.000000e+00:-3,1.000000e+00:-3,1.000000e+00:-2,1.000000e+00:-2,1.000000e+00:-2,1.000000e+00:-2,1.000000e+00:-2,1.000000e+00:-2,1.000000e+00:-2,1.000000e+00:-1,1.000000e+00:-1,1.000000e+00:-1,1.000000e+00:-1,1.000000e+00:-1,1.000000e+00:-1,1.000000e+00:-1,1.000000e+00:0,1.000000e+00:0,1.000000e+00:0,1.000000e+00:0,1.000000e+00:0,1.000000e+00:0,1.000000e+00:0,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:1,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:2,1.000000e+00:3,1.000000e+00:3,1.000000e+00:3,1.000000e+00:3,1.000000e+00:3,1.000000e+00:3,1.000000e+00:3,1.000000e+00:4,1.000000e+00:4,1.000000e+00:4,1.000000e+00:4,1.000000e+00:4,1.000000e+00:4,1.000000e+00:4,1.000000e+00:5,1.000000e+00:5,1.000000e+00:5,1.000000e+00:5,1.000000e+00:5,1.000000e+00:5,1.000000e+00:5,1.000000e+00:6,1.000000e+00:6,1.000000e+00:6,1.000000e+00:6,1.000000e+00:6,1.000000e+00:6,1.000000e+00:6,1.000000e+00:7,1.000000e+00:7,1.000000e+00:7,1.000000e+00:7,1.000000e+00:7,1.000000e+00:7,1.000000e+00:7}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedInput = !VPU.DistributedTensor<
    1x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights = !VPU.DistributedTensor<
    112x16x2x2x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    compute_shapes = [[48, 16, 2, 2], [32, 16, 2, 2], [32, 16, 2, 2]],
    compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]],
    memory_shapes = [[48, 16, 2, 2], [32, 16, 2, 2], [32, 16, 2, 2]],
    memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]]
}>

!DistributedBiasAndScaleTable = !VPU.DistributedTensor<
    112x1x1x1xf32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    compute_shapes = [[48, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]],
    memory_shapes = [[48, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]]
}>

!DistributedZeroPointTable = !VPU.DistributedTensor<
    112x1x1x1xi8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    compute_shapes = [[48, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]],
    memory_shapes = [[48, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]]
}>

!DistributedOutput = !VPU.DistributedTensor<
    1x112x3x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3 : i64,
    compute_shapes = [[1, 48, 3, 3], [1, 32, 3, 3], [1, 32, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 80, 0, 0]],
    memory_shapes = [[1, 48, 3, 3], [1, 32, 3, 3], [1, 32, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 80, 0, 0]]
}>

func.func @Conv2x2_ZeroPointTable_Segmented_UnevenWorkloadSizes_I4(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x112x3x3xf16> {
  %bias_table = const.Declare tensor<112x1x1x1xf32> = dense<0.000000e+00> : tensor<112x1x1x1xf32>
  %scale_table = const.Declare tensor<112x1x1x1xf32> = dense<1.000000e+00> : tensor<112x1x1x1xf32>
  %weights = const.Declare tensor<112x16x2x2x!qElemType, {order = #NHWC}> = dense<1> : tensor<112x16x2x2xsi8>, [#const.Reorder<#NHWC>]
  %zero_points = const.Declare tensor<112x1x1x1xsi4> = dense<[[[[-8]]], [[[-8]]], [[[-8]]], [[[-8]]], [[[-8]]], [[[-8]]], [[[-8]]], [[[-7]]], [[[-7]]], [[[-7]]], [[[-7]]], [[[-7]]], [[[-7]]], [[[-7]]], [[[-6]]], [[[-6]]], [[[-6]]], [[[-6]]], [[[-6]]], [[[-6]]], [[[-6]]], [[[-5]]], [[[-5]]], [[[-5]]], [[[-5]]], [[[-5]]], [[[-5]]], [[[-5]]], [[[-4]]], [[[-4]]], [[[-4]]], [[[-4]]], [[[-4]]], [[[-4]]], [[[-4]]], [[[-3]]], [[[-3]]], [[[-3]]], [[[-3]]], [[[-3]]], [[[-3]]], [[[-3]]], [[[-2]]], [[[-2]]], [[[-2]]], [[[-2]]], [[[-2]]], [[[-2]]], [[[-2]]], [[[-1]]], [[[-1]]], [[[-1]]], [[[-1]]], [[[-1]]], [[[-1]]], [[[-1]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[2]]], [[[2]]], [[[2]]], [[[2]]], [[[2]]], [[[2]]], [[[2]]], [[[3]]], [[[3]]], [[[3]]], [[[3]]], [[[3]]], [[[3]]], [[[3]]], [[[4]]], [[[4]]], [[[4]]], [[[4]]], [[[4]]], [[[4]]], [[[4]]], [[[5]]], [[[5]]], [[[5]]], [[[5]]], [[[5]]], [[[5]]], [[[5]]], [[[6]]], [[[6]]], [[[6]]], [[[6]]], [[[6]]], [[[6]]], [[[6]]], [[[7]]], [[[7]]], [[[7]]], [[[7]]], [[[7]]], [[[7]]], [[[7]]]]> : tensor<112x1x1x1xsi4>
  %0 = VPU.ZeroPointTable(%zero_points) : tensor<112x1x1x1xsi4> -> tensor<112x1x1x1xi8>
  %1 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %2 = VPU.Copy(%weights) {out_mem_space = @CMX_NN} : tensor<112x16x2x2x!qElemType, {order = #NHWC}> -> !DistributedWeights
  %3 = VPU.Copy(%bias_table) {out_mem_space = @CMX_NN} : tensor<112x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %4 = VPU.Copy(%scale_table) {out_mem_space = @CMX_NN} : tensor<112x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %5 = VPU.Copy(%0) {out_mem_space = @CMX_NN} : tensor<112x1x1x1xi8> -> !DistributedZeroPointTable
  %6 = VPU.NCE.Convolution(%1, %2, %3, %4, %5) rawFilterShape [112, 16, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } : !DistributedInput, !DistributedWeights, !DistributedBiasAndScaleTable, !DistributedBiasAndScaleTable, !DistributedZeroPointTable -> !DistributedOutput {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 48, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 48, 0, 0] outSizes [1, 32, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 80, 0, 0] outSizes [1, 32, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
          }
  %7 = VPU.Copy(%6) : !DistributedOutput -> tensor<1x112x3x3xf16>
  return %7 : tensor<1x112x3x3xf16>

  // CHECK:       [[CST_BIAS:%.+]] = const.Declare tensor<112x1x1x1xf32>
  // CHECK:       [[CST_SCALE:%.+]] = const.Declare tensor<112x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<112x16x2x2x!qElemType, {order = #NHWC}>
  // Workloads [48, 32, 32]: cluster 0 gets 48 channels (aligned to 64), clusters 1 and 2 get 32 each, total 128
  // CHECK:       [[CST_ZP:%.+]] = const.Declare tensor<112x1x1x1xsi4>
  // CHECK:       [[ZP_TABLE:%.+]] = VPU.ZeroPointTable([[CST_ZP]]) {workloadSizes = [48, 32, 32]
  // CHECK-SAME:  zeroPointTableData = [
  // CHECK-SAME:    -120, -86, -52, -120, -86, -36, -120, -70, -35, -104, -69, -35, -103, -69, -19, -103, -69, -19, -87, -52, -18, -87, -52, -18, -1, -1, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:    -2, 17, -1, 17, -1, 33, -1, 33, 0, 34, 0, 34, 0, 50, 16, 51, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:    51, 101, 51, 101, 68, 102, 68, 102, 68, 118, 84, 119, 85, 119, 85, 119, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  // CHECK-SAME:  ]} : tensor<112x1x1x1xsi4> -> tensor<96x1x1x1xi8>
  // CHECK:       [[ZP_TABLE_CMX:%.+]] = VPU.Copy([[ZP_TABLE]]) {out_mem_space = @CMX_NN} : tensor<96x1x1x1xi8> -> !VPU.DistributedTensor<96x1x1x1xi8
  // CHECK-SAME:  mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0]{{\]}}
  // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 48, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 0 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 48, 0, 0] outSizes [1, 32, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 1 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 80, 0, 0] outSizes [1, 32, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 2 : i64}
}

// -----

!qElemType = !quant.uniform<i8:f16:1, {1.000000e+00:1,1.000000e+00:2,1.000000e+00:3,1.000000e+00:4,1.000000e+00:5,1.000000e+00:6,1.000000e+00:7,1.000000e+00:8,1.000000e+00:9,1.000000e+00:10,1.000000e+00:11,1.000000e+00:12,1.000000e+00:13,1.000000e+00:14,1.000000e+00:15,1.000000e+00:16,1.000000e+00:17,1.000000e+00:18,1.000000e+00:19,1.000000e+00:20,1.000000e+00:21,1.000000e+00:22,1.000000e+00:23,1.000000e+00:24,1.000000e+00:25,1.000000e+00:26,1.000000e+00:27,1.000000e+00:28,1.000000e+00:29,1.000000e+00:30,1.000000e+00:31,1.000000e+00:32}>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!DistributedInput = !VPU.DistributedTensor<
    8x1x64x4x4xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[3, 1, 64, 4, 4], [3, 1, 64, 4, 4], [2, 1, 64, 4, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    memory_shapes = [[3, 1, 64, 4, 4], [3, 1, 64, 4, 4], [2, 1, 64, 4, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]]
}>

!DistributedWeights = !VPU.DistributedTensor<
    8x32x64x1x1x!qElemType, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[3, 32, 64, 1, 1], [3, 32, 64, 1, 1], [2, 32, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    memory_shapes = [[3, 32, 64, 1, 1], [3, 32, 64, 1, 1], [2, 32, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]]
}>

!DistributedBiasAndScaleTable = !VPU.DistributedTensor<
    1x32x1x1x1xf32, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 1, 1, 1], [1, 32, 1, 1, 1], [1, 32, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    memory_shapes = [[1, 32, 1, 1, 1], [1, 32, 1, 1, 1], [1, 32, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
}>

!DistributedZeroPointTable = !VPU.DistributedTensor<
    1x32x1x1x1xi8, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 1, 1, 1], [1, 32, 1, 1, 1], [1, 32, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    memory_shapes = [[1, 32, 1, 1, 1], [1, 32, 1, 1, 1], [1, 32, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
}>

!DistributedOutput = !VPU.DistributedTensor<
    8x1x32x4x4xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[3, 1, 32, 4, 4], [3, 1, 32, 4, 4], [2, 1, 32, 4, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    memory_shapes = [[3, 1, 32, 4, 4], [3, 1, 32, 4, 4], [2, 1, 32, 4, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]]
}>

func.func @MatMul5D_ZeroPointTable_SegmentedOverBatch(%arg0: tensor<8x1x64x4x4xf16, {order = #GNHWC}>) -> tensor<8x1x32x4x4xf16, {order = #GNHWC}> {
  %bias_table = const.Declare tensor<1x32x1x1x1xf32> = dense<0.000000e+00> : tensor<1x32x1x1x1xf32>
  %scale_table = const.Declare tensor<1x32x1x1x1xf32> = dense<1.000000e+00> : tensor<1x32x1x1x1xf32>
  %weights = const.Declare tensor<8x32x64x1x1x!qElemType, {order = #GNHWC}> = dense<1> : tensor<8x32x64x1x1xsi8>, [#const.CastElemType<!qElemType>, #const.Reorder<#GNHWC>]
  %zero_points = const.Declare tensor<1x32x1x1x1xsi8> = dense<[[[[[1]]], [[[2]]], [[[3]]], [[[4]]], [[[5]]], [[[6]]], [[[7]]], [[[8]]], [[[9]]], [[[10]]], [[[11]]], [[[12]]], [[[13]]], [[[14]]], [[[15]]], [[[16]]], [[[17]]], [[[18]]], [[[19]]], [[[20]]], [[[21]]], [[[22]]], [[[23]]], [[[24]]], [[[25]]], [[[26]]], [[[27]]], [[[28]]], [[[29]]], [[[30]]], [[[31]]], [[[32]]]]]> : tensor<1x32x1x1x1xsi8>
  %0 = VPU.ZeroPointTable(%zero_points) : tensor<1x32x1x1x1xsi8> -> tensor<1x32x1x1x1xi8>
  %1 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<8x1x64x4x4xf16, {order = #GNHWC}> -> !DistributedInput
  %2 = VPU.Copy(%weights) {out_mem_space = @CMX_NN} : tensor<8x32x64x1x1x!qElemType, {order = #GNHWC}> -> !DistributedWeights
  %3 = VPU.Copy(%bias_table) {out_mem_space = @CMX_NN} : tensor<1x32x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %4 = VPU.Copy(%scale_table) {out_mem_space = @CMX_NN} : tensor<1x32x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %5 = VPU.Copy(%0) {out_mem_space = @CMX_NN} : tensor<1x32x1x1x1xi8> -> !DistributedZeroPointTable
  %6 = VPU.NCE.MatMul(%1, %2, %3, %4, %5) rawFilterShape [8, 32, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
          strides = [1, 1]
      } -> !DistributedOutput {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0, 0] outSizes [3, 1, 32, 4, 4] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [3, 0, 0, 0, 0] outSizes [3, 1, 32, 4, 4] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [6, 0, 0, 0, 0] outSizes [2, 1, 32, 4, 4] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 2 : i64}
          }
  %7 = VPU.Copy(%6) : !DistributedOutput -> tensor<8x1x32x4x4xf16, {order = #GNHWC}>
  return %7 : tensor<8x1x32x4x4xf16, {order = #GNHWC}>

  // CHECK:       [[CST_BIAS:%.+]] = const.Declare tensor<1x32x1x1x1xf32>
  // CHECK:       [[CST_SCALE:%.+]] = const.Declare tensor<1x32x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<8x32x64x1x1x!qElemType, {order = #GNHWC}>
  // CHECK:       [[CST_ZP:%.+]] = const.Declare tensor<1x32x1x1x1xsi8>
  // CHECK:       [[ZP_TABLE:%.+]] = VPU.ZeroPointTable([[CST_ZP]]) {workloadSizes = [32]
  // CHECK-SAME:  zeroPointTableData = [1, 3, 17, 19, 2, 4, 18, 20, 5, 7, 21, 23, 6, 8, 22, 24, 9, 11, 25, 27, 10, 12, 26, 28, 13, 15, 29, 31, 14, 16, 30, 32]
  // CHECK-SAME:  } : tensor<1x32x1x1x1xsi8> -> tensor<1x32x1x1x1xi8>
  // CHECK:       [[ZP_TABLE_CMX:%.+]] = VPU.Copy([[ZP_TABLE]]) {out_mem_space = @CMX_NN} : tensor<1x32x1x1x1xi8> -> !VPU.DistributedTensor<1x32x1x1x1xi8
  // CHECK-SAME:  mode = "DUPLICATED", num_clusters = 3 : i64
  // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0, 0] outSizes [3, 1, 32, 4, 4]
  // CHECK-SAME:  <CUBOID_16x16> attributes {cluster_id = 0 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [3, 0, 0, 0, 0] outSizes [3, 1, 32, 4, 4]
  // CHECK-SAME:  <CUBOID_16x16> attributes {cluster_id = 1 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [6, 0, 0, 0, 0] outSizes [2, 1, 32, 4, 4]
  // CHECK-SAME:  <CUBOID_16x16> attributes {cluster_id = 2 : i64}
}

// -----

!qElemType = !quant.uniform<i8:f16:1, {1.000000e+00:1,1.000000e+00:2,1.000000e+00:3,1.000000e+00:4,1.000000e+00:5,1.000000e+00:6,1.000000e+00:7,1.000000e+00:8,1.000000e+00:9,1.000000e+00:10,1.000000e+00:11,1.000000e+00:12,1.000000e+00:13,1.000000e+00:14,1.000000e+00:15,1.000000e+00:16,1.000000e+00:17,1.000000e+00:18,1.000000e+00:19,1.000000e+00:20,1.000000e+00:21,1.000000e+00:22,1.000000e+00:23,1.000000e+00:24,1.000000e+00:25,1.000000e+00:26,1.000000e+00:27,1.000000e+00:28,1.000000e+00:29,1.000000e+00:30,1.000000e+00:31,1.000000e+00:32,1.000000e+00:33,1.000000e+00:34,1.000000e+00:35,1.000000e+00:36,1.000000e+00:37,1.000000e+00:38,1.000000e+00:39,1.000000e+00:40,1.000000e+00:41,1.000000e+00:42,1.000000e+00:43,1.000000e+00:44,1.000000e+00:45,1.000000e+00:46,1.000000e+00:47,1.000000e+00:48}>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!DistributedInput = !VPU.DistributedTensor<
    8x1x64x4x4xf16, #GNHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[8, 1, 64, 4, 4], [8, 1, 64, 4, 4], [8, 1, 64, 4, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    memory_shapes = [[8, 1, 64, 4, 4], [8, 1, 64, 4, 4], [8, 1, 64, 4, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
}>

!DistributedWeights = !VPU.DistributedTensor<
    8x48x64x1x1x!qElemType, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[8, 16, 64, 1, 1], [8, 16, 64, 1, 1], [8, 16, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [0, 16, 0, 0, 0], [0, 32, 0, 0, 0]],
    memory_shapes = [[8, 16, 64, 1, 1], [8, 16, 64, 1, 1], [8, 16, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [0, 16, 0, 0, 0], [0, 32, 0, 0, 0]]
}>

!DistributedBiasAndScaleTable = !VPU.DistributedTensor<
    1x48x1x1x1xf32, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 1, 1, 1], [1, 16, 1, 1, 1], [1, 16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [0, 16, 0, 0, 0], [0, 32, 0, 0, 0]],
    memory_shapes = [[1, 16, 1, 1, 1], [1, 16, 1, 1, 1], [1, 16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [0, 16, 0, 0, 0], [0, 32, 0, 0, 0]]
}>

!DistributedZeroPointTable = !VPU.DistributedTensor<
    1x48x1x1x1xi8, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 1, 1, 1], [1, 16, 1, 1, 1], [1, 16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [0, 16, 0, 0, 0], [0, 32, 0, 0, 0]],
    memory_shapes = [[1, 16, 1, 1, 1], [1, 16, 1, 1, 1], [1, 16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [0, 16, 0, 0, 0], [0, 32, 0, 0, 0]]
}>

!DistributedOutput = !VPU.DistributedTensor<
    8x1x48x4x4xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 3, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[8, 1, 16, 4, 4], [8, 1, 16, 4, 4], [8, 1, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 16, 0, 0], [0, 0, 32, 0, 0]],
    memory_shapes = [[8, 1, 16, 4, 4], [8, 1, 16, 4, 4], [8, 1, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 16, 0, 0], [0, 0, 32, 0, 0]]
}>

func.func @MatMul5D_ZeroPointTable_SegmentedOverOC(%arg0: tensor<8x1x64x4x4xf16, {order = #GNHWC}>) -> tensor<8x1x48x4x4xf16, {order = #GNHWC}> {
  %bias_table = const.Declare tensor<1x48x1x1x1xf32> = dense<0.000000e+00> : tensor<1x48x1x1x1xf32>
  %scale_table = const.Declare tensor<1x48x1x1x1xf32> = dense<1.000000e+00> : tensor<1x48x1x1x1xf32>
  %weights = const.Declare tensor<8x48x64x1x1x!qElemType, {order = #GNHWC}> = dense<1> : tensor<8x48x64x1x1xsi8>, [#const.CastElemType<!qElemType>, #const.Reorder<#GNHWC>]
  %zero_points = const.Declare tensor<1x48x1x1x1xsi8> = dense<[[[[[1]]], [[[2]]], [[[3]]], [[[4]]], [[[5]]], [[[6]]], [[[7]]], [[[8]]], [[[9]]], [[[10]]], [[[11]]], [[[12]]], [[[13]]], [[[14]]], [[[15]]], [[[16]]], [[[17]]], [[[18]]], [[[19]]], [[[20]]], [[[21]]], [[[22]]], [[[23]]], [[[24]]], [[[25]]], [[[26]]], [[[27]]], [[[28]]], [[[29]]], [[[30]]], [[[31]]], [[[32]]], [[[33]]], [[[34]]], [[[35]]], [[[36]]], [[[37]]], [[[38]]], [[[39]]], [[[40]]], [[[41]]], [[[42]]], [[[43]]], [[[44]]], [[[45]]], [[[46]]], [[[47]]], [[[48]]]]]> : tensor<1x48x1x1x1xsi8>
  %0 = VPU.ZeroPointTable(%zero_points) : tensor<1x48x1x1x1xsi8> -> tensor<1x48x1x1x1xi8>
  %1 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<8x1x64x4x4xf16, {order = #GNHWC}> -> !DistributedInput
  %2 = VPU.Copy(%weights) {out_mem_space = @CMX_NN} : tensor<8x48x64x1x1x!qElemType, {order = #GNHWC}> -> !DistributedWeights
  %3 = VPU.Copy(%bias_table) {out_mem_space = @CMX_NN} : tensor<1x48x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %4 = VPU.Copy(%scale_table) {out_mem_space = @CMX_NN} : tensor<1x48x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %5 = VPU.Copy(%0) {out_mem_space = @CMX_NN} : tensor<1x48x1x1x1xi8> -> !DistributedZeroPointTable
  %6 = VPU.NCE.MatMul(%1, %2, %3, %4, %5) rawFilterShape [8, 48, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
          strides = [1, 1]
      } -> !DistributedOutput {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0, 0] outSizes [8, 1, 16, 4, 4] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 16, 0, 0] outSizes [8, 1, 16, 4, 4] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 32, 0, 0] outSizes [8, 1, 16, 4, 4] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 2 : i64}
          }
  %7 = VPU.Copy(%6) : !DistributedOutput -> tensor<8x1x48x4x4xf16, {order = #GNHWC}>
  return %7 : tensor<8x1x48x4x4xf16, {order = #GNHWC}>

  // CHECK:       [[CST_BIAS:%.+]] = const.Declare tensor<1x48x1x1x1xf32>
  // CHECK:       [[CST_SCALE:%.+]] = const.Declare tensor<1x48x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<8x48x64x1x1x!qElemType, {order = #GNHWC}>
  // Each cluster gets 16 channels, padded to 32, total 96 with interleaved padding
  // CHECK:       [[CST_ZP:%.+]] = const.Declare tensor<1x48x1x1x1xsi8>
  // CHECK:       [[ZP_TABLE:%.+]] = VPU.ZeroPointTable([[CST_ZP]]) {workloadSizes = [16, 16, 16]
  // CHECK-SAME:  zeroPointTableData = [
  // CHECK-SAME:    1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:    17, 19, 18, 20, 21, 23, 22, 24, 25, 27, 26, 28, 29, 31, 30, 32, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:    33, 35, 34, 36, 37, 39, 38, 40, 41, 43, 42, 44, 45, 47, 46, 48, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  // CHECK-SAME:  ]} : tensor<1x48x1x1x1xsi8> -> tensor<1x96x1x1x1xi8>
  // CHECK:       [[ZP_TABLE_CMX:%.+]] = VPU.Copy([[ZP_TABLE]]) {out_mem_space = @CMX_NN} : tensor<1x96x1x1x1xi8> -> !VPU.DistributedTensor<1x96x1x1x1xi8
  // CHECK-SAME:  mode = "SEGMENTED", num_tiles = [1, 3, 1, 1, 1], num_clusters = 3 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[1, 32, 1, 1, 1], [1, 32, 1, 1, 1], [1, 32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0, 0], [0, 32, 0, 0, 0], [0, 64, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[1, 32, 1, 1, 1], [1, 32, 1, 1, 1], [1, 32, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0, 0], [0, 32, 0, 0, 0], [0, 64, 0, 0, 0]{{\]}}
  // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0, 0] outSizes [8, 1, 16, 4, 4]
  // CHECK-SAME:  <CUBOID_16x16> attributes {cluster_id = 0 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 16, 0, 0] outSizes [8, 1, 16, 4, 4]
  // CHECK-SAME:  <CUBOID_16x16> attributes {cluster_id = 1 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 32, 0, 0] outSizes [8, 1, 16, 4, 4]
  // CHECK-SAME:  <CUBOID_16x16> attributes {cluster_id = 2 : i64}
}

// -----

!qElemType = !quant.uniform<i8:f16:0, {1.000000e+00:1,1.000000e+00:2,1.000000e+00:3,1.000000e+00:4,1.000000e+00:5,1.000000e+00:6,1.000000e+00:7,1.000000e+00:8,1.000000e+00:9,1.000000e+00:10,1.000000e+00:11,1.000000e+00:12,1.000000e+00:13,1.000000e+00:14,1.000000e+00:15,1.000000e+00:16}>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedInput = !VPU.DistributedTensor<
    1x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights = !VPU.DistributedTensor<
    16x16x1x1x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedDataPointerTable = !VPU.DistributedTensor<
    16x1x1x1xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedBiasAndScaleTable = !VPU.DistributedTensor<
    16x1x1x1xf32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedOutput = !VPU.DistributedTensor<
    1x16x3x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// Zero-points can be stored on 8 most significant bits of each data-pointer.
func.func @DepthConv2x2_DataPointerTable_Duplicated_With_ZeroPoints(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x16x3x3xf16> {
  %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
  %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
  %weights = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<16x16x1x1xsi8>, [#const.Reorder<#NHWC>]
  %zero_points = const.Declare tensor<16x1x1x1xsi8> = dense<[[[[1]]], [[[2]]], [[[3]]], [[[4]]], [[[5]]], [[[6]]], [[[7]]], [[[8]]], [[[9]]], [[[10]]], [[[11]]], [[[12]]], [[[13]]], [[[14]]], [[[15]]], [[[16]]]]> : tensor<16x1x1x1xsi8>
  %0 = VPU.DataPointerTable(%zero_points) : tensor<16x1x1x1xsi8> -> tensor<16x1x1x1xsi32>
  %1 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %2 = VPU.Copy(%weights) {out_mem_space = @CMX_NN} : tensor<16x16x1x1x!qElemType, {order = #NHWC}> -> !DistributedWeights
  %3 = VPU.Copy(%0) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xsi32> -> !DistributedDataPointerTable
  %4 = VPU.Copy(%bias_table) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %5 = VPU.Copy(%scale_table) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %6 = VPU.NCE.DepthConvolution(%1, %2, %3, %4, %5) rawFilterShape [16, 1, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } -> !DistributedOutput {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 3 : i64}
          }
  %7 = VPU.Copy(%6) : !DistributedOutput -> tensor<1x16x3x3xf16>
  return %7 : tensor<1x16x3x3xf16>

  // CHECK:       [[CST_BIAS:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}>
  // CHECK:       [[CST_ZP:%.+]] = const.Declare tensor<16x1x1x1xsi8>
  // CHECK:       [[DATA_PTR_TABLE:%.+]] = VPU.DataPointerTable([[CST_ZP]]) : tensor<16x1x1x1xsi8> {
  // CHECK-SAME:  dataPointerTableData = [16777216, 50331680, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          33554448, 67108912, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          83886144, 117440608, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          100663376, 134217840, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          150995072, 184549536, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          167772304, 201326768, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          218104000, 251658464, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          234881232, 268435696, -1, -1, -1, -1, -1, -1], workloadSizes = [16]} -> tensor<64x1x1x1xsi32>
  // CHECK:       [[DATA_PTR_TABLE_CMX:%.+]] = VPU.Copy([[DATA_PTR_TABLE]]) {out_mem_space = @CMX_NN} : tensor<64x1x1x1xsi32> -> !VPU.DistributedTensor<64x1x1x1xsi32
  // CHECK-SAME:  mode = "DUPLICATED", num_clusters = 4 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK:       [[DWCONV:%.+]] = VPU.NCE.DepthConvolution({{%.+}}, {{%.+}}, [[DATA_PTR_TABLE_CMX]], {{%.+}}, {{%.+}})
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 0 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 1 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 2 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 3 : i64}
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00:1>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedInput = !VPU.DistributedTensor<
    1x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights = !VPU.DistributedTensor<
    48x16x1x1x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]],
    memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]]
}>

!DistributedDataPointerTable = !VPU.DistributedTensor<
    48x1x1x1xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]]
}>

!DistributedBiasAndScaleTable = !VPU.DistributedTensor<
    48x1x1x1xf32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]]
}>

!DistributedOutput = !VPU.DistributedTensor<
    1x48x3x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]
}>

func.func @DepthConv2x2_DataPointerTable_Segmented_EvenWorkloadSizes(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x48x3x3xf16> {
  %bias_table = const.Declare tensor<48x1x1x1xf32> = dense<0.000000e+00> : tensor<48x1x1x1xf32>
  %scale_table = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>
  %weights = const.Declare tensor<48x16x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<48x16x1x1xsi8>, [#const.Reorder<#NHWC>]
  %0 = VPU.DataPointerTable -> tensor<48x1x1x1xsi32>
  %1 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %2 = VPU.Copy(%weights) {out_mem_space = @CMX_NN} : tensor<48x16x1x1x!qElemType, {order = #NHWC}> -> !DistributedWeights
  %3 = VPU.Copy(%0) {out_mem_space = @CMX_NN} : tensor<48x1x1x1xsi32> -> !DistributedDataPointerTable
  %4 = VPU.Copy(%bias_table) {out_mem_space = @CMX_NN} : tensor<48x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %5 = VPU.Copy(%scale_table) {out_mem_space = @CMX_NN} : tensor<48x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %6 = VPU.NCE.DepthConvolution(%1, %2, %3, %4, %5) rawFilterShape [48, 1, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } -> !DistributedOutput {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 16, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
          }
  %7 = VPU.Copy(%6) : !DistributedOutput -> tensor<1x48x3x3xf16>
  return %7 : tensor<1x48x3x3xf16>

  // CHECK:       [[CST_BIAS:%.+]] = const.Declare tensor<48x1x1x1xf32>
  // CHECK:       [[CST_SCALE:%.+]] = const.Declare tensor<48x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<48x16x1x1x!qElemType, {order = #NHWC}>
  // Each cluster gets 16 channels, padded to 64, total 192 with interleaved padding
  // CHECK:       [[DATA_PTR_TABLE:%.+]] = VPU.DataPointerTable {
  // CHECK-SAME:  dataPointerTableData = [0, 32, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          16, 48, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          64, 96, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          80, 112, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          128, 160, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          144, 176, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          192, 224, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          208, 240, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          0, 32, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          16, 48, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          64, 96, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          80, 112, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          128, 160, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          144, 176, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          192, 224, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          208, 240, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          0, 32, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          16, 48, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          64, 96, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          80, 112, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          128, 160, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          144, 176, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          192, 224, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          208, 240, -1, -1, -1, -1, -1, -1], workloadSizes = [16, 16, 16]} -> tensor<192x1x1x1xsi32>
  // CHECK:       [[DATA_PTR_TABLE_CMX:%.+]] = VPU.Copy([[DATA_PTR_TABLE]]) {out_mem_space = @CMX_NN} : tensor<192x1x1x1xsi32> -> !VPU.DistributedTensor<192x1x1x1xsi32
  // CHECK-SAME:  mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0]{{\]}}
  // CHECK:       [[DWCONV:%.+]] = VPU.NCE.DepthConvolution({{%.+}}, {{%.+}}, [[DATA_PTR_TABLE_CMX]], {{%.+}}, {{%.+}})
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 0 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 16, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 1 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 16, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 2 : i64}
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00:1>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedInput = !VPU.DistributedTensor<
    1x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights = !VPU.DistributedTensor<
    112x16x1x1x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    compute_shapes = [[48, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]],
    memory_shapes = [[48, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]]
}>

!DistributedDataPointerTable = !VPU.DistributedTensor<
    112x1x1x1xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    compute_shapes = [[48, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]],
    memory_shapes = [[48, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]]
}>

!DistributedBiasAndScaleTable = !VPU.DistributedTensor<
    112x1x1x1xf32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    compute_shapes = [[48, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]],
    memory_shapes = [[48, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [80, 0, 0, 0]]
}>

!DistributedOutput = !VPU.DistributedTensor<
    1x112x3x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3 : i64,
    compute_shapes = [[1, 48, 3, 3], [1, 32, 3, 3], [1, 32, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 80, 0, 0]],
    memory_shapes = [[1, 48, 3, 3], [1, 32, 3, 3], [1, 32, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 80, 0, 0]]
}>

func.func @DepthConv2x2_DataPointerTable_Segmented_UnevenWorkloadSizes(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x112x3x3xf16> {
  %bias_table = const.Declare tensor<112x1x1x1xf32> = dense<0.000000e+00> : tensor<112x1x1x1xf32>
  %scale_table = const.Declare tensor<112x1x1x1xf32> = dense<1.000000e+00> : tensor<112x1x1x1xf32>
  %weights = const.Declare tensor<112x16x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<112x16x1x1xsi8>, [#const.Reorder<#NHWC>]
  %0 = VPU.DataPointerTable -> tensor<112x1x1x1xsi32>
  %1 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %2 = VPU.Copy(%weights) {out_mem_space = @CMX_NN} : tensor<112x16x1x1x!qElemType, {order = #NHWC}> -> !DistributedWeights
  %3 = VPU.Copy(%0) {out_mem_space = @CMX_NN} : tensor<112x1x1x1xsi32> -> !DistributedDataPointerTable
  %4 = VPU.Copy(%bias_table) {out_mem_space = @CMX_NN} : tensor<112x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %5 = VPU.Copy(%scale_table) {out_mem_space = @CMX_NN} : tensor<112x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %6 = VPU.NCE.DepthConvolution(%1, %2, %3, %4, %5) rawFilterShape [112, 1, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } -> !DistributedOutput {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 48, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 48, 0, 0] outSizes [1, 32, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 80, 0, 0] outSizes [1, 32, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
          }
  %7 = VPU.Copy(%6) : !DistributedOutput -> tensor<1x112x3x3xf16>
  return %7 : tensor<1x112x3x3xf16>

  // CHECK:       [[CST_BIAS:%.+]] = const.Declare tensor<112x1x1x1xf32>
  // CHECK:       [[CST_SCALE:%.+]] = const.Declare tensor<112x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<112x16x1x1x!qElemType, {order = #NHWC}>
  // Workloads [48, 32, 32]: cluster 0 gets 48 channels (aligned to 64), clusters 1 gets 32 channels (aligned to 64) and cluster 2 gets 32 channels (aligned to 64), total 192
  // CHECK:       [[DATA_PTR_TABLE:%.+]] = VPU.DataPointerTable {
  // CHECK-SAME:  dataPointerTableData = [0, 32, 256, 288, 512, 544, -1, -1,
  // CHECK-SAME:                          16, 48, 272, 304, 528, 560, -1, -1,
  // CHECK-SAME:                          64, 96, 320, 352, 576, 608, -1, -1,
  // CHECK-SAME:                          80, 112, 336, 368, 592, 624, -1, -1,
  // CHECK-SAME:                          128, 160, 384, 416, 640, 672, -1, -1,
  // CHECK-SAME:                          144, 176, 400, 432, 656, 688, -1, -1,
  // CHECK-SAME:                          192, 224, 448, 480, 704, 736, -1, -1,
  // CHECK-SAME:                          208, 240, 464, 496, 720, 752, -1, -1,
  // CHECK-SAME:                          0, 32, 256, 288, -1, -1, -1, -1,
  // CHECK-SAME:                          16, 48, 272, 304, -1, -1, -1, -1,
  // CHECK-SAME:                          64, 96, 320, 352, -1, -1, -1, -1,
  // CHECK-SAME:                          80, 112, 336, 368, -1, -1, -1, -1,
  // CHECK-SAME:                          128, 160, 384, 416, -1, -1, -1, -1,
  // CHECK-SAME:                          144, 176, 400, 432, -1, -1, -1, -1,
  // CHECK-SAME:                          192, 224, 448, 480, -1, -1, -1, -1,
  // CHECK-SAME:                          208, 240, 464, 496, -1, -1, -1, -1,
  // CHECK-SAME:                          0, 32, 256, 288, -1, -1, -1, -1,
  // CHECK-SAME:                          16, 48, 272, 304, -1, -1, -1, -1,
  // CHECK-SAME:                          64, 96, 320, 352, -1, -1, -1, -1,
  // CHECK-SAME:                          80, 112, 336, 368, -1, -1, -1, -1,
  // CHECK-SAME:                          128, 160, 384, 416, -1, -1, -1, -1,
  // CHECK-SAME:                          144, 176, 400, 432, -1, -1, -1, -1,
  // CHECK-SAME:                          192, 224, 448, 480, -1, -1, -1, -1,
  // CHECK-SAME:                          208, 240, 464, 496, -1, -1, -1, -1], workloadSizes = [48, 32, 32]} -> tensor<192x1x1x1xsi32>
  // CHECK:       [[DATA_PTR_TABLE_CMX:%.+]] = VPU.Copy([[DATA_PTR_TABLE]]) {out_mem_space = @CMX_NN} : tensor<192x1x1x1xsi32> -> !VPU.DistributedTensor<192x1x1x1xsi32,
  // CHECK-SAME:  mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0]{{\]}}
  // CHECK:       [[CONV:%.+]] = VPU.NCE.DepthConvolution({{%.+}}, {{%.+}}, [[DATA_PTR_TABLE_CMX]], {{%.+}}, {{%.+}})
  // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 48, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 0 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 48, 0, 0] outSizes [1, 32, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 1 : i64}
  // CHECK:       VPU.DPU.Workload outOffsets [0, 80, 0, 0] outSizes [1, 32, 3, 3]
  // CHECK-SAME:  <CUBOID_8x16> attributes {cluster_id = 2 : i64}
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00:1>
!qElemType1 = !quant.uniform<i8:f16, 1.000000e+00:2>
!qElemType2 = !quant.uniform<i8:f16, 1.000000e+00:3>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedInput = !VPU.DistributedTensor<
    1x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4], [1, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights = !VPU.DistributedTensor<
    16x16x1x1x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights1 = !VPU.DistributedTensor<
    16x16x1x1x!qElemType1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedWeights2 = !VPU.DistributedTensor<
    16x16x1x1x!qElemType2, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedDataPointerTable = !VPU.DistributedTensor<
    16x1x1x1xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedBiasAndScaleTable = !VPU.DistributedTensor<
    16x1x1x1xf32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1], [16, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedOutput = !VPU.DistributedTensor<
    1x16x3x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3], [1, 16, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

func.func @DepthConv2x2_DataPointerTable_Sliced(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x48x3x3xf16> {
  %bias_table0 = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
  %scale_table0 = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
  %weights0 = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<16x16x1x1xsi8>, [#const.Reorder<#NHWC>]

  %bias_table1 = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
  %scale_table1 = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
  %weights1 = const.Declare tensor<16x16x1x1x!qElemType1, {order = #NHWC}> = dense<1> : tensor<16x16x1x1xsi8>, [#const.Reorder<#NHWC>]

  %bias_table2 = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
  %scale_table2 = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
  %weights2 = const.Declare tensor<16x16x1x1x!qElemType2, {order = #NHWC}> = dense<1> : tensor<16x16x1x1xsi8>, [#const.Reorder<#NHWC>]

  %data_pointer_table_dummy = VPU.DataPointerTable -> tensor<48x1x1x1xsi32>

  %dpt_slice0 = VPU.Slice %data_pointer_table_dummy [0, 0, 0, 0] [16, 1, 1, 1] : tensor<48x1x1x1xsi32> to tensor<16x1x1x1xsi32>
  %dpt_slice1 = VPU.Slice %data_pointer_table_dummy [16, 0, 0, 0] [16, 1, 1, 1] : tensor<48x1x1x1xsi32> to tensor<16x1x1x1xsi32>
  %dpt_slice2 = VPU.Slice %data_pointer_table_dummy [32, 0, 0, 0] [16, 1, 1, 1] : tensor<48x1x1x1xsi32> to tensor<16x1x1x1xsi32>

  // First convolution
  %input0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %weights_cmx0 = VPU.Copy(%weights0) {out_mem_space = @CMX_NN} : tensor<16x16x1x1x!qElemType, {order = #NHWC}> -> !DistributedWeights
  %dpt_cmx0 = VPU.Copy(%dpt_slice0) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xsi32> -> !DistributedDataPointerTable
  %bias_cmx0 = VPU.Copy(%bias_table0) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %scale_cmx0 = VPU.Copy(%scale_table0) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %conv0 = VPU.NCE.DepthConvolution(%input0, %weights_cmx0, %dpt_cmx0, %bias_cmx0, %scale_cmx0) rawFilterShape [16, 1, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } -> !DistributedOutput  {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 3 : i64}
          }
  %out0 = VPU.Copy(%conv0) : !DistributedOutput -> tensor<1x16x3x3xf16>

  // Second convolution
  %input1 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %weights_cmx1 = VPU.Copy(%weights1) {out_mem_space = @CMX_NN} : tensor<16x16x1x1x!qElemType1, {order = #NHWC}> -> !DistributedWeights1
  %dpt_cmx1 = VPU.Copy(%dpt_slice1) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xsi32> -> !DistributedDataPointerTable
  %bias_cmx1 = VPU.Copy(%bias_table1) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %scale_cmx1 = VPU.Copy(%scale_table1) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %conv1 = VPU.NCE.DepthConvolution(%input1, %weights_cmx1, %dpt_cmx1, %bias_cmx1, %scale_cmx1) rawFilterShape [16, 1, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,

          strides = [1, 1]
      } -> !DistributedOutput  {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 3 : i64}
          }
  %out1 = VPU.Copy(%conv1) : !DistributedOutput -> tensor<1x16x3x3xf16>

  // Third convolution
  %input2 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x4x4xf16, {order = #NHWC}> -> !DistributedInput
  %weights_cmx2 = VPU.Copy(%weights2) {out_mem_space = @CMX_NN} : tensor<16x16x1x1x!qElemType2, {order = #NHWC}> -> !DistributedWeights2
  %dpt_cmx2 = VPU.Copy(%dpt_slice2) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xsi32> -> !DistributedDataPointerTable
  %bias_cmx2 = VPU.Copy(%bias_table2) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %scale_cmx2 = VPU.Copy(%scale_table2) {out_mem_space = @CMX_NN} : tensor<16x1x1x1xf32> -> !DistributedBiasAndScaleTable
  %conv2 = VPU.NCE.DepthConvolution(%input2, %weights_cmx2, %dpt_cmx2, %bias_cmx2, %scale_cmx2) rawFilterShape [16, 1, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 1, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,

          strides = [1, 1]
      } -> !DistributedOutput  {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 2 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 3, 3] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 3 : i64}
          }
  %out2 = VPU.Copy(%conv2) : !DistributedOutput -> tensor<1x16x3x3xf16>

  %concat = VPU.Concat(%out0, %out1, %out2) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]} : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x48x3x3xf16>
  return %concat : tensor<1x48x3x3xf16>

  // CHECK:       [[CST_BIAS0:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_SCALE0:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS0:%.+]] = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}>
  // CHECK:       [[CST_BIAS1:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_SCALE1:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS1:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:       [[CST_BIAS2:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_SCALE2:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK:       [[CST_WEIGHTS2:%.+]] = const.Declare tensor<16x16x1x1x!qElemType2, {order = #NHWC}>

  // CHECK-NOT:   VPU.DataPointerTable -> tensor<48x1x1x1xsi32>
  // CHECK-NOT:   VPU.Slice

  // All DW Convs can use the same table, as it has the same data.
  // CHECK:       [[DATA_PTR_TABLE:%.+]] = VPU.DataPointerTable {
  // CHECK-SAME:  dataPointerTableData = [0, 32, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          16, 48, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          64, 96, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          80, 112, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          128, 160, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          144, 176, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          192, 224, -1, -1, -1, -1, -1, -1,
  // CHECK-SAME:                          208, 240, -1, -1, -1, -1, -1, -1], workloadSizes = [16]} -> tensor<64x1x1x1xsi32>

  // CHECK:       [[DATA_PTR_TABLE_CMX0:%.+]] = VPU.Copy([[DATA_PTR_TABLE]]) {out_mem_space = @CMX_NN} : tensor<64x1x1x1xsi32> -> !VPU.DistributedTensor<64x1x1x1xsi32,
  // CHECK-SAME:  mode = "DUPLICATED", num_clusters = 4 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK:       [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution({{%.+}}, {{%.+}}, [[DATA_PTR_TABLE_CMX0]], {{%.+}}, {{%.+}})
  // CHECK:       [[OUT0:%.+]] = VPU.Copy([[DWCONV0]])

  // CHECK:       [[DATA_PTR_TABLE_CMX1:%.+]] = VPU.Copy([[DATA_PTR_TABLE]]) {out_mem_space = @CMX_NN} : tensor<64x1x1x1xsi32> -> !VPU.DistributedTensor<64x1x1x1xsi32,
  // CHECK-SAME:  mode = "DUPLICATED", num_clusters = 4 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK:       [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution({{%.+}}, {{%.+}}, [[DATA_PTR_TABLE_CMX1]], {{%.+}}, {{%.+}})
  // CHECK:       [[OUT1:%.+]] = VPU.Copy([[DWCONV1]])

  // CHECK:       [[DATA_PTR_TABLE_CMX2:%.+]] = VPU.Copy([[DATA_PTR_TABLE]]) {out_mem_space = @CMX_NN} : tensor<64x1x1x1xsi32> -> !VPU.DistributedTensor<64x1x1x1xsi32,
  // CHECK-SAME:  mode = "DUPLICATED", num_clusters = 4 : i64
  // CHECK-SAME:  compute_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  compute_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK-SAME:  memory_shapes = {{\[}}[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]{{\]}}
  // CHECK-SAME:  memory_offsets = {{\[}}[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]{{\]}}
  // CHECK:       [[DWCONV2:%.+]] = VPU.NCE.DepthConvolution({{%.+}}, {{%.+}}, [[DATA_PTR_TABLE_CMX2]], {{%.+}}, {{%.+}})
  // CHECK:       [[OUT2:%.+]] = VPU.Copy([[DWCONV2]])

  // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUT0]], [[OUT1]], [[OUT2]])
  // CHECK:       return [[CONCAT]]
}
