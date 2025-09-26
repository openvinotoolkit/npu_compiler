//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --apply-tiling-mvn1sum %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @executors {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
  }
  // CHECK-LABEL: func.func @SOHTilingMVN1SumNotCorrectHForMS
  // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<3x1x1000001x1xf16, {order = #NHWC}>
  func.func @SOHTilingMVN1SumNotCorrectHForMS(%arg0: tensor<3x1x1000001x1xf16, {order = #NHWC}>) -> (tensor<3x1x1x2xf16, {order = #NHWC}>) {
        %0 = VPU.MVN1SumOp(%arg0) {across_channels = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, output_height = 2 : i64} : tensor<3x1x1000001x1xf16, {order = #NHWC}> -> tensor<3x1x2x2xf32, {order = #NHWC}>
        %1 = VPU.MVN1MeanVar(%0) {across_channels = true, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [3, 1, 1000001, 1], output_type = f16} : tensor<3x1x2x2xf32, {order = #NHWC}> -> tensor<3x1x1x2xf16, {order = #NHWC}>
        return %1 : tensor<3x1x1x2xf16, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [3, 1, 250001, 1] : tensor<3x1x1000001x1xf16, {order = #NHWC}> to tensor<3x1x250001x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_1]])
      // CHECK-SAME:           :  tensor<3x1x250001x1xf16, {order = #NHWC}> -> tensor<3x1x2x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 250001, 0] [3, 1, 250000, 1] : tensor<3x1x1000001x1xf16, {order = #NHWC}> to tensor<3x1x250000x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_2]])
      // CHECK-SAME:           :  tensor<3x1x250000x1xf16, {order = #NHWC}> -> tensor<3x1x2x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 500001, 0] [3, 1, 250000, 1] : tensor<3x1x1000001x1xf16, {order = #NHWC}> to tensor<3x1x250000x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_3:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_3]])
      // CHECK-SAME:           :  tensor<3x1x250000x1xf16, {order = #NHWC}> -> tensor<3x1x2x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 750001, 0] [3, 1, 250000, 1] : tensor<3x1x1000001x1xf16, {order = #NHWC}> to tensor<3x1x250000x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_4:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_4]])
      // CHECK-SAME:           :  tensor<3x1x250000x1xf16, {order = #NHWC}> -> tensor<3x1x2x2xf32, {order = #NHWC}>

      // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]], [[TILE_3]], [[TILE_4]])
      // CHECK-SAME:           -> tensor<3x1x2x8xf32, {order = #NHWC}>

      // CHECK:            [[MVN1MEANVAR:%.+]] = VPU.MVN1MeanVar([[CONCAT]])
      // CHECK-SAME:           : tensor<3x1x2x8xf32, {order = #NHWC}> -> tensor<3x1x1x2xf16, {order = #NHWC}>

      // CHECK:            return [[MVN1MEANVAR]]
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @executors {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
  }
  // CHECK-LABEL: func.func @SOHTilingMVN1SumCorrectHForMS
  // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x1000001x1xf16, {order = #NHWC}>
  func.func @SOHTilingMVN1SumCorrectHForMS(%arg0: tensor<1x3x1000001x1xf16, {order = #NHWC}>) -> (tensor<1x3x1x2xf16, {order = #NHWC}>) {
        %0 = VPU.MVN1SumOp(%arg0) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, output_height = 2 : i64} : tensor<1x3x1000001x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>
        %1 = VPU.MVN1MeanVar(%0) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 1000001, 1], output_type = f16} : tensor<1x3x2x2xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>
        return %1 : tensor<1x3x1x2xf16, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 250001, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x250001x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_1]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, output_height = 4 : i64}
      // CHECK-SAME:           : tensor<1x3x250001x1xf16, {order = #NHWC}> -> tensor<1x3x4x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 250001, 0] [1, 3, 250000, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x250000x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_2]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, output_height = 4 : i64}
      // CHECK-SAME:           : tensor<1x3x250000x1xf16, {order = #NHWC}> -> tensor<1x3x4x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 500001, 0] [1, 3, 250000, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x250000x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_3:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_3]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, output_height = 4 : i64}
      // CHECK-SAME:           : tensor<1x3x250000x1xf16, {order = #NHWC}> -> tensor<1x3x4x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 750001, 0] [1, 3, 250000, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x250000x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_4:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_4]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, output_height = 4 : i64}
      // CHECK-SAME:           : tensor<1x3x250000x1xf16, {order = #NHWC}> -> tensor<1x3x4x2xf32, {order = #NHWC}>

      // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]], [[TILE_3]], [[TILE_4]]) {per_axis = #IE.Concat<axis = 3 : i64>}
      // CHECK-SAME:           -> tensor<1x3x4x8xf32, {order = #NHWC}>

      // CHECK:            [[MVN1MEANVAR:%.+]] = VPU.MVN1MeanVar([[CONCAT]]) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 1000001, 1], output_type = f16}
      // CHECK-SAME:           : tensor<1x3x4x8xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>

      // CHECK:            return [[MVN1MEANVAR]] : tensor<1x3x1x2xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @executors {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
  }
  // CHECK-LABEL: func.func @ClusteringTilingMVN1SumCorrectHForMS
  // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x1000001x1xf16, {order = #NHWC}>
  func.func @ClusteringTilingMVN1SumCorrectHForMS(%arg0: tensor<1x3x1000001x1xf16, {order = #NHWC}>) -> (tensor<1x3x1x2xf16, {order = #NHWC}>) {
        %0 = VPU.MVN1SumOp(%arg0) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 4 : i64} : tensor<1x3x1000001x1xf16, {order = #NHWC}> -> tensor<1x3x4x2xf32, {order = #NHWC}>
        %1 = VPU.MVN1MeanVar(%0) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 1000001, 1], output_type = f16} : tensor<1x3x4x2xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>
        return %1 : tensor<1x3x1x2xf16, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 142858, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x142858x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_1]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
      // CHECK-SAME:           : tensor<1x3x142858x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 142858, 0] [1, 3, 142858, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x142858x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_2]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
      // CHECK-SAME:           : tensor<1x3x142858x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 285716, 0] [1, 3, 142857, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x142857x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_3:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_3]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
      // CHECK-SAME:           : tensor<1x3x142857x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 428573, 0] [1, 3, 142857, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x142857x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_4:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_4]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
      // CHECK-SAME:           : tensor<1x3x142857x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 571430, 0] [1, 3, 142857, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x142857x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_5:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_5]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
      // CHECK-SAME:           : tensor<1x3x142857x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_6:%.+]] = VPU.Slice [[INPUT]] [0, 0, 714287, 0] [1, 3, 142857, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x142857x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_6:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_6]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
      // CHECK-SAME:           : tensor<1x3x142857x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_7:%.+]] = VPU.Slice [[INPUT]] [0, 0, 857144, 0] [1, 3, 142857, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x142857x1xf16, {order = #NHWC}>
      // CHECK:            [[TILE_7:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_7]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
      // CHECK-SAME:           : tensor<1x3x142857x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

      // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]], [[TILE_3]], [[TILE_4]], [[TILE_5]], [[TILE_6]], [[TILE_7]]) {per_axis = #IE.Concat<axis = 3 : i64>}
      // CHECK-SAME:           -> tensor<1x3x2x14xf32, {order = #NHWC}>

      // CHECK:            [[MVN1MEANVAR:%.+]] = VPU.MVN1MeanVar([[CONCAT]]) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 1000001, 1], output_type = f16}
      // CHECK-SAME:           : tensor<1x3x2x14xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>

      // CHECK:            return [[MVN1MEANVAR]] : tensor<1x3x1x2xf16, {order = #NHWC}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @executors {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
      config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
  }
  // CHECK-LABEL: func.func @SOKTilingMVN1Sum
  // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x1000001x1xf16>
  func.func @SOKTilingMVN1Sum(%arg0: tensor<1x3x1000001x1xf16>) -> (tensor<1x3x1x2xf16, {order = #NHWC}>) {
        %0 = VPU.MVN1SumOp(%arg0) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, output_height = 4 : i64} : tensor<1x3x1000001x1xf16> -> tensor<1x3x4x2xf32, {order = #NHWC}>
        %1 = VPU.MVN1MeanVar(%0) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 20001, 1], output_type = f16} : tensor<1x3x4x2xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>
        return %1 : tensor<1x3x1x2xf16, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 200001, 1] : tensor<1x3x1000001x1xf16> to tensor<1x3x200001x1xf16>
      // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_1]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, output_height = 1 : i64}
      // CHECK-SAME:           : tensor<1x3x200001x1xf16> -> tensor<1x3x1x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 200001, 0] [1, 3, 200000, 1] : tensor<1x3x1000001x1xf16> to tensor<1x3x200000x1xf16>
      // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_2]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, output_height = 1 : i64}
      // CHECK-SAME:           : tensor<1x3x200000x1xf16> -> tensor<1x3x1x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 400001, 0] [1, 3, 200000, 1] : tensor<1x3x1000001x1xf16> to tensor<1x3x200000x1xf16>
      // CHECK:            [[TILE_3:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_3]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, output_height = 1 : i64}
      // CHECK-SAME:           : tensor<1x3x200000x1xf16> -> tensor<1x3x1x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 600001, 0] [1, 3, 200000, 1] : tensor<1x3x1000001x1xf16> to tensor<1x3x200000x1xf16>
      // CHECK:            [[TILE_4:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_4]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, output_height = 1 : i64}
      // CHECK-SAME:           : tensor<1x3x200000x1xf16> -> tensor<1x3x1x2xf32, {order = #NHWC}>

      // CHECK:            [[INPUT_TILE_5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 800001, 0] [1, 3, 200000, 1] : tensor<1x3x1000001x1xf16> to tensor<1x3x200000x1xf16>
      // CHECK:            [[TILE_5:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_5]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, output_height = 1 : i64}
      // CHECK-SAME:           : tensor<1x3x200000x1xf16> -> tensor<1x3x1x2xf32, {order = #NHWC}>

      // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]], [[TILE_3]], [[TILE_4]], [[TILE_5]]) {per_axis = #IE.Concat<axis = 3 : i64>}
      // CHECK-SAME:           -> tensor<1x3x1x10xf32, {order = #NHWC}>

      // CHECK:            [[MVN1MEANVAR:%.+]] = VPU.MVN1MeanVar([[CONCAT]]) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 20001, 1], output_type = f16}
      // CHECK-SAME:           : tensor<1x3x1x10xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>

      // CHECK:            return [[MVN1MEANVAR]]
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {

config.Resources 2 of @NCE at 1.850000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: func.func @SOKTilingMVN1SumEqualChannelsToClusters
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<3x2x200000x1xf16, {order = #NHWC}>
func.func @SOKTilingMVN1SumEqualChannelsToClusters(%arg0: tensor<3x2x200000x1xf16, {order = #NHWC}>) -> (tensor<3x1x1x2xf16, {order = #NHWC}>) {
    %0 = VPU.MVN1SumOp(%arg0) {across_channels = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, output_height = 2 : i64} : tensor<3x2x200000x1xf16, {order = #NHWC}> -> tensor<3x1x2x2xf32, {order = #NHWC}>
    %1 = VPU.MVN1MeanVar(%0) {across_channels = true, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [3, 2, 200000, 1], output_type = f16} : tensor<3x1x2x2xf32, {order = #NHWC}> -> tensor<3x1x1x2xf16, {order = #NHWC}>
    return %1 : tensor<3x1x1x2xf16, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [3, 2, 100000, 1] : tensor<3x2x200000x1xf16, {order = #NHWC}> to tensor<3x2x100000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_1]])

    // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 100000, 0] [3, 2, 100000, 1] : tensor<3x2x200000x1xf16, {order = #NHWC}> to tensor<3x2x100000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_2]])

    // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]])
    // CHECK-SAME:           -> tensor<3x1x1x4xf32, {order = #NHWC}>

    // CHECK:            [[VAL3:%.+]] = VPU.MVN1MeanVar([[CONCAT]])
    // CHECK-SAME:           -> tensor<3x1x1x2xf16, {order = #NHWC}>

    // CHECK:            return [[VAL3]]
}

}
