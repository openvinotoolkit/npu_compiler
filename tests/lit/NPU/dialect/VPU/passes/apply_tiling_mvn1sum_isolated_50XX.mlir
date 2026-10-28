//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --apply-tiling-mvn1sum="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @TilingMVN1SumModelNativeSliceInput
// Reproduces the Kokoro CMX overflow: a model-native VPU.Slice feeds MVN1SumOp.
// The two SliceOp early-exits in MVN1SumCorrectHeight and MVN1SumTiling incorrectly
// blocked both height correction and tiling, leaving a 2MB/cluster SEGMENTED demand
// (8001 rows × 128ch × 2B = 2,048,256 > 1,473,536 CMX) that failed the scheduler.
// After the fix: height is corrected 3→6, then the input is tiled into two sub-ops
// whose per-cluster demand (4001×128×2 = 1,024,256 B < available CMX) fits.
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x128x24004x1xf16, {order = #NHWC}>
func.func @TilingMVN1SumModelNativeSliceInput(%arg0: tensor<1x128x24004x1xf16, {order = #NHWC}>) -> tensor<1x128x1x2xf16, {order = #NHWC}> {
    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 128, 24001, 1]
            : tensor<1x128x24004x1xf16, {order = #NHWC}> to tensor<1x128x24001x1xf16, {order = #NHWC}>
    %1 = VPU.MVN1SumOp(%0) {
            across_channels = false,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            normalize_variance = true,
            output_height = 3 : i64
         } : tensor<1x128x24001x1xf16, {order = #NHWC}> -> tensor<1x128x3x2xf32, {order = #NHWC}>
    %2 = VPU.MVN1MeanVar(%1) {
            across_channels = false,
            eps = 9.9999997473787516E-6 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            normalize_variance = true,
            orig_shape = [1, 128, 24001, 1],
            output_type = f16
         } : tensor<1x128x3x2xf32, {order = #NHWC}> -> tensor<1x128x1x2xf16, {order = #NHWC}>
    return %2 : tensor<1x128x1x2xf16, {order = #NHWC}>

    // MVN1SumCorrectHeight raises output_height 3→6 (SOH + H=24001 >= numActShave=6).
    // MVN1SumTiling tiles the 6MB input into 2 sub-ops; with SOH+3 clusters each
    // per-cluster demand is 4001×128×2 = 1,024,256 B < available CMX.
    // The model-native VPU.Slice is preserved; tiling slices are cut from its result.
    // CHECK:        [[ORIG_SLICE:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 128, 24001, 1]
    // CHECK-SAME:       tensor<1x128x24004x1xf16, {order = #NHWC}> to tensor<1x128x24001x1xf16, {order = #NHWC}>
    // CHECK:        [[SLICE0:%.+]] = VPU.Slice [[ORIG_SLICE]] [0, 0, 0, 0] [1, 128, 12001, 1]
    // CHECK-SAME:       tensor<1x128x24001x1xf16, {order = #NHWC}> to tensor<1x128x12001x1xf16, {order = #NHWC}>
    // CHECK:        [[TILE0:%.+]] = VPU.MVN1SumOp([[SLICE0]])
    // CHECK-SAME:       across_channels = false
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:       output_height = 6 : i64
    // CHECK-SAME:       tensor<1x128x12001x1xf16, {order = #NHWC}> -> tensor<1x128x6x2xf32, {order = #NHWC}>
    // CHECK:        [[SLICE1:%.+]] = VPU.Slice [[ORIG_SLICE]] [0, 0, 12001, 0] [1, 128, 12000, 1]
    // CHECK-SAME:       tensor<1x128x24001x1xf16, {order = #NHWC}> to tensor<1x128x12000x1xf16, {order = #NHWC}>
    // CHECK:        [[TILE1:%.+]] = VPU.MVN1SumOp([[SLICE1]])
    // CHECK-SAME:       across_channels = false
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:       output_height = 6 : i64
    // CHECK-SAME:       tensor<1x128x12000x1xf16, {order = #NHWC}> -> tensor<1x128x6x2xf32, {order = #NHWC}>
    // CHECK:        [[CONCAT:%.+]] = VPU.Concat([[TILE0]], [[TILE1]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:       -> tensor<1x128x6x4xf32, {order = #NHWC}>
    // CHECK:        VPU.MVN1MeanVar([[CONCAT]])
    // CHECK-SAME:       tensor<1x128x6x4xf32, {order = #NHWC}> -> tensor<1x128x1x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @SOHTilingMVN1SumNotCorrectHForMS
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<3x1x1420001x1xf16, {order = #NHWC}>
func.func @SOHTilingMVN1SumNotCorrectHForMS(%arg0: tensor<3x1x1420001x1xf16, {order = #NHWC}>) -> (tensor<3x1x1x2xf16, {order = #NHWC}>) {
      %0 = VPU.MVN1SumOp(%arg0) {across_channels = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, output_height = 4 : i64} : tensor<3x1x1420001x1xf16, {order = #NHWC}> -> tensor<3x1x4x2xf32, {order = #NHWC}>
      %1 = VPU.MVN1MeanVar(%0) {across_channels = true, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [3, 1, 1420001, 1], output_type = f16} : tensor<3x1x4x2xf32, {order = #NHWC}> -> tensor<3x1x1x2xf16, {order = #NHWC}>
      return %1 : tensor<3x1x1x2xf16, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [3, 1, 710001, 1] : tensor<3x1x1420001x1xf16, {order = #NHWC}> to tensor<3x1x710001x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_1]])
    // CHECK-SAME:           :  tensor<3x1x710001x1xf16, {order = #NHWC}> -> tensor<3x1x3x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 710001, 0] [3, 1, 710000, 1] : tensor<3x1x1420001x1xf16, {order = #NHWC}> to tensor<3x1x710000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_2]])
    // CHECK-SAME:           :  tensor<3x1x710000x1xf16, {order = #NHWC}> -> tensor<3x1x3x2xf32, {order = #NHWC}>

    // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]])
    // CHECK-SAME:           -> tensor<3x1x3x4xf32, {order = #NHWC}>

    // CHECK:            [[VAL3:%.+]] = VPU.MVN1MeanVar([[CONCAT]])
    // CHECK-SAME:           : tensor<3x1x3x4xf32, {order = #NHWC}> -> tensor<3x1x1x2xf16, {order = #NHWC}>

    // CHECK:            return [[VAL3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @SOHTilingMVN1SumCorrectHForMS
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x1420001x1xf16, {order = #NHWC}>
func.func @SOHTilingMVN1SumCorrectHForMS(%arg0: tensor<1x3x1420001x1xf16, {order = #NHWC}>) -> (tensor<1x3x1x2xf16, {order = #NHWC}>) {
      %0 = VPU.MVN1SumOp(%arg0) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, output_height = 4 : i64} : tensor<1x3x1420001x1xf16, {order = #NHWC}> -> tensor<1x3x4x2xf32, {order = #NHWC}>
      %1 = VPU.MVN1MeanVar(%0) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 1420001, 1], output_type = f16} : tensor<1x3x4x2xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>
      return %1 : tensor<1x3x1x2xf16, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 710001, 1] : tensor<1x3x1420001x1xf16, {order = #NHWC}> to tensor<1x3x710001x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_1]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, output_height = 6 : i64}
    // CHECK-SAME:           : tensor<1x3x710001x1xf16, {order = #NHWC}> -> tensor<1x3x6x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 710001, 0] [1, 3, 710000, 1] : tensor<1x3x1420001x1xf16, {order = #NHWC}> to tensor<1x3x710000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_2]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, output_height = 6 : i64}
    // CHECK-SAME:           : tensor<1x3x710000x1xf16, {order = #NHWC}> -> tensor<1x3x6x2xf32, {order = #NHWC}>

    // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:           -> tensor<1x3x6x4xf32, {order = #NHWC}>

    // CHECK:            [[VAL3:%.+]] = VPU.MVN1MeanVar([[CONCAT]]) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 1420001, 1], output_type = f16}
    // CHECK-SAME:           : tensor<1x3x6x4xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>

    // CHECK:            return [[VAL3]] : tensor<1x3x1x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @ClusteringTilingMVN1SumCorrectHForMS
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x1000001x1xf16, {order = #NHWC}>
func.func @ClusteringTilingMVN1SumCorrectHForMS(%arg0: tensor<1x3x1000001x1xf16, {order = #NHWC}>) -> (tensor<1x3x1x2xf16, {order = #NHWC}>) {
      %0 = VPU.MVN1SumOp(%arg0) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64} : tensor<1x3x1000001x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>
      %1 = VPU.MVN1MeanVar(%0) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 1000001, 1], output_type = f16} : tensor<1x3x2x2xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>
      return %1 : tensor<1x3x1x2xf16, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 200001, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x200001x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_1]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
    // CHECK-SAME:           : tensor<1x3x200001x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 200001, 0] [1, 3, 200000, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x200000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_2]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
    // CHECK-SAME:           : tensor<1x3x200000x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 400001, 0] [1, 3, 200000, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x200000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_3:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_3]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
    // CHECK-SAME:           : tensor<1x3x200000x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 600001, 0] [1, 3, 200000, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x200000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_4:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_4]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
    // CHECK-SAME:           : tensor<1x3x200000x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 800001, 0] [1, 3, 200000, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x200000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_5:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_5]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
    // CHECK-SAME:           : tensor<1x3x200000x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

    // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]], [[TILE_3]], [[TILE_4]], [[TILE_5]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:           -> tensor<1x3x2x10xf32, {order = #NHWC}>

    // CHECK:            [[VAL3:%.+]] = VPU.MVN1MeanVar([[CONCAT]]) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 1000001, 1], output_type = f16}
    // CHECK-SAME:           : tensor<1x3x2x10xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>

    // CHECK:            return [[VAL3]] : tensor<1x3x1x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @SOKTilingMVN1Sum
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x1420001x1xf16>
func.func @SOKTilingMVN1Sum(%arg0: tensor<1x3x1420001x1xf16>) -> (tensor<1x3x1x2xf16, {order = #NHWC}>) {
      %0 = VPU.MVN1SumOp(%arg0) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, output_height = 4 : i64} : tensor<1x3x1420001x1xf16> -> tensor<1x3x4x2xf32, {order = #NHWC}>
      %1 = VPU.MVN1MeanVar(%0) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 20001, 1], output_type = f16} : tensor<1x3x4x2xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>
      return %1 : tensor<1x3x1x2xf16, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 710001, 1] : tensor<1x3x1420001x1xf16> to tensor<1x3x710001x1xf16>
    // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_1]])
    // CHECK-SAME:           :  tensor<1x3x710001x1xf16> -> tensor<1x3x1x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 710001, 0] [1, 3, 710000, 1] : tensor<1x3x1420001x1xf16> to tensor<1x3x710000x1xf16>
    // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_2]])
    // CHECK-SAME:           :  tensor<1x3x710000x1xf16> -> tensor<1x3x1x2xf32, {order = #NHWC}>

    // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]])
    // CHECK-SAME:           -> tensor<1x3x1x4xf32, {order = #NHWC}>

    // CHECK:            [[VAL3:%.+]] = VPU.MVN1MeanVar([[CONCAT]])
    // CHECK-SAME:           : tensor<1x3x1x4xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>

    // CHECK:            return [[VAL3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @TilingMVN1SumCorrectFor1C
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x512x4096x1xf16, {order = #NHWC}>
func.func @TilingMVN1SumCorrectFor1C(%arg0: tensor<1x512x4096x1xf16, {order = #NHWC}>) -> tensor<1x512x1x2xf16, {order = #NHWC}> {
  %0 = VPU.MVN1SumOp(%arg0) {across_channels = false, normalize_variance = true, output_height = 1 : i64} : tensor<1x512x4096x1xf16, {order = #NHWC}> -> tensor<1x512x1x2xf32, {order = #NHWC}>
  %1 = VPU.MVN1MeanVar(%0) {across_channels = false, eps = 9.9999999747524271E-7 : f64, internal_reshape = [1, 32, 65536, 1], normalize_variance = true, orig_shape = [1, 512, 64, 64], output_type = f16} : tensor<1x512x1x2xf32, {order = #NHWC}> -> tensor<1x512x1x2xf16, {order = #NHWC}>
  return %1 : tensor<1x512x1x2xf16, {order = #NHWC}>

  // CHECK:            [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 1366, 1] : tensor<1x512x4096x1xf16, {order = #NHWC}> to tensor<1x512x1366x1xf16, {order = #NHWC}>
  // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_1]]) {across_channels = false, normalize_variance = true, output_height = 2 : i64} : tensor<1x512x1366x1xf16, {order = #NHWC}> -> tensor<1x512x2x2xf32, {order = #NHWC}>

  // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 1366, 0] [1, 512, 1365, 1] : tensor<1x512x4096x1xf16, {order = #NHWC}> to tensor<1x512x1365x1xf16, {order = #NHWC}>
  // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_2]]) {across_channels = false, normalize_variance = true, output_height = 2 : i64} : tensor<1x512x1365x1xf16, {order = #NHWC}> -> tensor<1x512x2x2xf32, {order = #NHWC}>

  // CHECK:            [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 2731, 0] [1, 512, 1365, 1] : tensor<1x512x4096x1xf16, {order = #NHWC}> to tensor<1x512x1365x1xf16, {order = #NHWC}>
  // CHECK:            [[TILE_3:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_3]]) {across_channels = false, normalize_variance = true, output_height = 2 : i64} : tensor<1x512x1365x1xf16, {order = #NHWC}> -> tensor<1x512x2x2xf32, {order = #NHWC}>

  // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]], [[TILE_3]])
  // CHECK-SAME:           -> tensor<1x512x2x6xf32, {order = #NHWC}>

  // CHECK:            [[RES:%.+]] = VPU.MVN1MeanVar([[CONCAT]])
  // CHECK-SAME:           : tensor<1x512x2x6xf32, {order = #NHWC}> -> tensor<1x512x1x2xf16, {order = #NHWC}>

  // CHECK:            return [[RES]]
}
