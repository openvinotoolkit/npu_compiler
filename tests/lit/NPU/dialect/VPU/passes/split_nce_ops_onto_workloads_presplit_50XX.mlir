//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true enable-vpunn-pre-split=true" --split-NCE-ops-onto-workloads %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @SplitNCEPermute
func.func @SplitNCEPermute(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
    } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       VPU.NCE.Permute(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64,
    // CHECK-SAME:      minimumHardwareExecutionCost = {{[1-9][0-9]+}} : i64,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
    // CHECK-SAME:      } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {

    // CHECK:       DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 4, 224, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputqElemType = !VPU.DistributedTensor<1x1024x21x18xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
        compute_shapes = [[1, 1024, 7, 18], [1, 1024, 7, 18], [1, 1024, 7, 18]], compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0]],
        memory_shapes = [[1, 1024, 9, 18], [1, 1024, 8, 18], [1, 1024, 8, 18]], memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 13, 0]]}>
!weightsqElemType = !VPU.DistributedTensor<16x1024x3x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
        compute_shapes = [[16, 1024, 3, 3], [16, 1024, 3, 3], [16, 1024, 3, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[16, 1024, 3, 3], [16, 1024, 3, 3], [16, 1024, 3, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!wtqElemType = !VPU.DistributedTensor<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
        compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!outqElemType = !VPU.DistributedTensor<1x16x19x18xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
        compute_shapes = [[1, 16, 7, 18], [1, 16, 6, 18], [1, 16, 6, 18]], compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 13, 0]],
        memory_shapes = [[1, 16, 7, 18], [1, 16, 6, 18], [1, 16, 6, 18]], memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 13, 0]]}>

// CHECK-LABEL: @SplitMultiClusterNCEConv
func.func @SplitMultiClusterNCEConv(%input: !inputqElemType, %weights: !weightsqElemType, %weights_table: !wtqElemType) -> !outqElemType {
  %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>,
        clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        rawFilterShape = [16, 1024, 3, 3], strides = [1, 1]} : !inputqElemType, !weightsqElemType, !wtqElemType -> !outqElemType
  return %conv : !outqElemType
    // CHECK:    [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 7, 18] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:    VPU.DPU.Workload outOffsets [0, 0, 7, 0] outSizes [1, 16, 6, 18] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:    VPU.DPU.Workload outOffsets [0, 0, 13, 0] outSizes [1, 16, 6, 18] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 2 : i64}
    // CHECK:    return [[CONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputqElemType = !VPU.DistributedTensor<1x256x32x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
        compute_shapes = [[1, 256, 32, 4], [1, 256, 32, 4], [1, 256, 32, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[1, 256, 32, 4], [1, 256, 32, 4], [1, 256, 32, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
!weightsqElemType = !VPU.DistributedTensor<256x16x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
        compute_shapes = [[96, 16, 1, 1], [80, 16, 1, 1], [80, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [176, 0, 0, 0]],
        memory_shapes = [[96, 16, 1, 1], [80, 16, 1, 1], [80, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [176, 0, 0, 0]]}>
!wtqElemType = !VPU.DistributedTensor<256x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
        compute_shapes = [[96, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [176, 0, 0, 0]],
        memory_shapes = [[96, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [176, 0, 0, 0]]}>
!outqElemType = !VPU.DistributedTensor<1x256x32x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
        compute_shapes = [[1, 96, 32, 4], [1, 80, 32, 4], [1, 80, 32, 4]], compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 176, 0, 0]],
        memory_shapes = [[1, 96, 32, 4], [1, 80, 32, 4], [1, 80, 32, 4]], memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 176, 0, 0]]}>

// CHECK-LABEL: @SplitMultiClusterNCEDWConv
func.func @SplitMultiClusterNCEDWConv(%input: !inputqElemType, %weights: !weightsqElemType, %weights_table: !wtqElemType) -> !outqElemType {
    %dwconv = VPU.NCE.DepthConvolution(%input, %weights, %weights_table) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            rawFilterShape = [256, 1, 3, 3], strides = [1, 1]} -> !outqElemType
    return %dwconv : !outqElemType

    // Track E#159358. Fix the channel split to avoid too many workloads
    // CHECK:    [[DWCONV:%.+]] = VPU.NCE.DepthConvolution
    // CHECK:      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 16, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 48, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 80, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 16, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 48, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 2 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 16, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 2 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 2 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 48, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 2 : i64}
    // CHECK:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 16, 32, 4] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 2 : i64}
    // CHECK:    return [[DWCONV]]
}
