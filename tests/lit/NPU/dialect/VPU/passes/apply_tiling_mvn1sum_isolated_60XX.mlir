//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --apply-tiling-mvn1sum="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @TilingMVN1SumModelNativeSliceInput
// Reproduces the Kokoro CMX overflow on NPU6010: a model-native VPU.Slice feeds MVN1SumOp.
// The two SliceOp early-exits incorrectly blocked height correction and tiling.
// After fix: height is corrected 4→8 (SOH + H=24001 >= numActShave=8), then tiled into
// 2 sub-ops (per-cluster demand: divUp(24001,8)=3001 rows × 128ch × 2B = 768,256 B < CMX).
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x128x24004x1xf16, {order = #NHWC}>
func.func @TilingMVN1SumModelNativeSliceInput(%arg0: tensor<1x128x24004x1xf16, {order = #NHWC}>) -> tensor<1x128x1x2xf16, {order = #NHWC}> {
    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 128, 24001, 1]
            : tensor<1x128x24004x1xf16, {order = #NHWC}> to tensor<1x128x24001x1xf16, {order = #NHWC}>
    %1 = VPU.MVN1SumOp(%0) {
            across_channels = false,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            normalize_variance = true,
            output_height = 4 : i64
         } : tensor<1x128x24001x1xf16, {order = #NHWC}> -> tensor<1x128x4x2xf32, {order = #NHWC}>
    %2 = VPU.MVN1MeanVar(%1) {
            across_channels = false,
            eps = 9.9999997473787516E-6 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            normalize_variance = true,
            orig_shape = [1, 128, 24001, 1],
            output_type = f16
         } : tensor<1x128x4x2xf32, {order = #NHWC}> -> tensor<1x128x1x2xf16, {order = #NHWC}>
    return %2 : tensor<1x128x1x2xf16, {order = #NHWC}>

    // CHECK:        [[ORIG_SLICE:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 128, 24001, 1]
    // CHECK-SAME:       tensor<1x128x24004x1xf16, {order = #NHWC}> to tensor<1x128x24001x1xf16, {order = #NHWC}>
    // CHECK:        [[SLICE0:%.+]] = VPU.Slice [[ORIG_SLICE]] [0, 0, 0, 0] [1, 128, 12001, 1]
    // CHECK-SAME:       tensor<1x128x24001x1xf16, {order = #NHWC}> to tensor<1x128x12001x1xf16, {order = #NHWC}>
    // CHECK:        [[TILE0:%.+]] = VPU.MVN1SumOp([[SLICE0]])
    // CHECK-SAME:       across_channels = false
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:       output_height = 8 : i64
    // CHECK-SAME:       tensor<1x128x12001x1xf16, {order = #NHWC}> -> tensor<1x128x8x2xf32, {order = #NHWC}>
    // CHECK:        [[SLICE1:%.+]] = VPU.Slice [[ORIG_SLICE]] [0, 0, 12001, 0] [1, 128, 12000, 1]
    // CHECK-SAME:       tensor<1x128x24001x1xf16, {order = #NHWC}> to tensor<1x128x12000x1xf16, {order = #NHWC}>
    // CHECK:        [[TILE1:%.+]] = VPU.MVN1SumOp([[SLICE1]])
    // CHECK-SAME:       across_channels = false
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:       output_height = 8 : i64
    // CHECK-SAME:       tensor<1x128x12000x1xf16, {order = #NHWC}> -> tensor<1x128x8x2xf32, {order = #NHWC}>
    // CHECK:        [[CONCAT:%.+]] = VPU.Concat([[TILE0]], [[TILE1]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:       -> tensor<1x128x8x4xf32, {order = #NHWC}>
    // CHECK:        VPU.MVN1MeanVar([[CONCAT]])
    // CHECK-SAME:       tensor<1x128x8x4xf32, {order = #NHWC}> -> tensor<1x128x1x2xf16, {order = #NHWC}>
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
    // CHECK-SAME:           :  tensor<3x1x710001x1xf16, {order = #NHWC}> -> tensor<3x1x4x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 710001, 0] [3, 1, 710000, 1] : tensor<3x1x1420001x1xf16, {order = #NHWC}> to tensor<3x1x710000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_2]])
    // CHECK-SAME:           :  tensor<3x1x710000x1xf16, {order = #NHWC}> -> tensor<3x1x4x2xf32, {order = #NHWC}>

    // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]])
    // CHECK-SAME:           -> tensor<3x1x4x4xf32, {order = #NHWC}>

    // CHECK:            [[MVN1MEANVAR:%.+]] = VPU.MVN1MeanVar([[CONCAT]])
    // CHECK-SAME:           : tensor<3x1x4x4xf32, {order = #NHWC}> -> tensor<3x1x1x2xf16, {order = #NHWC}>

    // CHECK:            return [[MVN1MEANVAR]]
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
    // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_1]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, output_height = 8 : i64}
    // CHECK-SAME:           : tensor<1x3x710001x1xf16, {order = #NHWC}> -> tensor<1x3x8x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 710001, 0] [1, 3, 710000, 1] : tensor<1x3x1420001x1xf16, {order = #NHWC}> to tensor<1x3x710000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_2]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, output_height = 8 : i64}
    // CHECK-SAME:           : tensor<1x3x710000x1xf16, {order = #NHWC}> -> tensor<1x3x8x2xf32, {order = #NHWC}>

    // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:           -> tensor<1x3x8x4xf32, {order = #NHWC}>

    // CHECK:            [[MVN1MEANVAR:%.+]] = VPU.MVN1MeanVar([[CONCAT]]) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 1420001, 1], output_type = f16}
    // CHECK-SAME:           : tensor<1x3x8x4xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>

    // CHECK:            return [[MVN1MEANVAR]] : tensor<1x3x1x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @ClusteringTilingMVN1SumCorrectHForMS
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x1000001x1xf16, {order = #NHWC}>
func.func @ClusteringTilingMVN1SumCorrectHForMS(%arg0: tensor<1x3x1000001x1xf16, {order = #NHWC}>) -> (tensor<1x3x1x2xf16, {order = #NHWC}>) {
      %0 = VPU.MVN1SumOp(%arg0) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64} : tensor<1x3x1000001x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>
      %1 = VPU.MVN1MeanVar(%0) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 1000001, 1], output_type = f16} : tensor<1x3x2x2xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>
      return %1 : tensor<1x3x1x2xf16, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 250001, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x250001x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_1]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
    // CHECK-SAME:           : tensor<1x3x250001x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 250001, 0] [1, 3, 250000, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x250000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_2]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
    // CHECK-SAME:           : tensor<1x3x250000x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 500001, 0] [1, 3, 250000, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x250000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_3:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_3]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
    // CHECK-SAME:           : tensor<1x3x250000x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 750001, 0] [1, 3, 250000, 1] : tensor<1x3x1000001x1xf16, {order = #NHWC}> to tensor<1x3x250000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_4:%.+]] = VPU.MVN1SumOp([[INPUT_TILE_4]]) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 2 : i64}
    // CHECK-SAME:           : tensor<1x3x250000x1xf16, {order = #NHWC}> -> tensor<1x3x2x2xf32, {order = #NHWC}>

    // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]], [[TILE_3]], [[TILE_4]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK-SAME:           -> tensor<1x3x2x8xf32, {order = #NHWC}>

    // CHECK:            [[MVN1MEANVAR:%.+]] = VPU.MVN1MeanVar([[CONCAT]]) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 1000001, 1], output_type = f16}
    // CHECK-SAME:           : tensor<1x3x2x8xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>

    // CHECK:            return [[MVN1MEANVAR]] : tensor<1x3x1x2xf16, {order = #NHWC}>
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

    // CHECK:            [[MVN1MEANVAR:%.+]] = VPU.MVN1MeanVar([[CONCAT]])
    // CHECK-SAME:           : tensor<1x3x1x4xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>

    // CHECK:            return [[MVN1MEANVAR]]
}
