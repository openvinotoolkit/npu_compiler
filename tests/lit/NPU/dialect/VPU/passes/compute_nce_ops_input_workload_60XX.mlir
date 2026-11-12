//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --compute-nce-input-workloads %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvSegOverlappedInputWorkloads
module @DepthConvSegOverlappedInputWorkloads {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : false
    config.Option @config.EnableODULocalRegion : true
  }

  func.func @main(
      %arg0: !VPU.DistributedTensor<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>,
      %arg1: !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>,
      %arg2: !VPU.DistributedTensor<32x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 4], uniform_distributed_segments}>
    ) -> !VPU.DistributedTensor<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
      compute_shapes = [[1, 16, 16, 16], [1, 16, 16, 16]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
      memory_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]], memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
      memory_num_tiles = [1, 1, 2, 1]}> {

    %out = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,

        strides = [1, 1]
      } -> !VPU.DistributedTensor<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
        compute_shapes = [[1, 16, 16, 16], [1, 16, 16, 16]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
        memory_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]], memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
        memory_num_tiles = [1, 1, 2, 1]}> {
      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
      VPU.DPU.Workload outOffsets [0, 16, 0, 0] outSizes [1, 16, 16, 16] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
    }

    return %out : !VPU.DistributedTensor<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
      compute_shapes = [[1, 16, 16, 16], [1, 16, 16, 16]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
      memory_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]], memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
      memory_num_tiles = [1, 1, 2, 1]}>
  }

  // CHECK:            VPU.NCE.DepthConvolution
  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 16, 16]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 16, 16]

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 16, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 16, 16]
  // CHECK-SAME:          outOffsets [0, 16, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 16, 16]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvSegmentedInputWorkloads
module @DepthConvSegmentedInputWorkloads {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : false
    config.Option @config.EnableODULocalRegion : true
  }

  func.func @main(
      %arg0: !VPU.DistributedTensor<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>,
      %arg1: !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>,
      %arg2: !VPU.DistributedTensor<32x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 4], uniform_distributed_segments}>
    ) -> !VPU.DistributedTensor<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
      compute_shapes = [[1, 16, 16, 16], [1, 16, 16, 16]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
      memory_shapes = [[1, 16, 16, 16], [1, 16, 16, 16]], memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]}> {

    %out = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,

        strides = [1, 1]
      } -> !VPU.DistributedTensor<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
        compute_shapes = [[1, 16, 16, 16], [1, 16, 16, 16]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
        memory_shapes = [[1, 16, 16, 16], [1, 16, 16, 16]], memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]}> {
      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 0 : i64}
      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] pad [0, 0, 0, 0] <CUBOID_8x16> attributes {cluster_id = 1 : i64}
    }

    return %out : !VPU.DistributedTensor<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
      compute_shapes = [[1, 16, 16, 16], [1, 16, 16, 16]], compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
      memory_shapes = [[1, 16, 16, 16], [1, 16, 16, 16]], memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]}>
  }

  // CHECK:            VPU.NCE.DepthConvolution
  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 16, 16]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 16, 16]

  // CHECK:           VPU.DPU.Workload
  // CHECK-SAME:          inOffsets [0, 0, 0, 0]
  // CHECK-SAME:          inSizes [1, 16, 16, 16]
  // CHECK-SAME:          outOffsets [0, 0, 0, 0]
  // CHECK-SAME:          outSizes [1, 16, 16, 16]
}
