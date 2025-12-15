//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --apply-tiling-mvn1sum="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @ClusteringNoTilingMVN1Sum
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<3x1x20001x1xf16, {order = #NHWC}>
func.func @ClusteringNoTilingMVN1Sum(%arg0: tensor<3x1x20001x1xf16, {order = #NHWC}>) -> (tensor<3x1x1x2xf16, {order = #NHWC}>) {
      %0 = VPU.MVN1SumOp(%arg0) {across_channels = true, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, output_height = 4 : i64} : tensor<3x1x20001x1xf16, {order = #NHWC}> -> tensor<3x1x4x2xf32, {order = #NHWC}>
      %1 = VPU.MVN1MeanVar(%0) {across_channels = true, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [3, 1, 20001, 1], output_type = f16} : tensor<3x1x4x2xf32, {order = #NHWC}> -> tensor<3x1x1x2xf16, {order = #NHWC}>
      return %1 : tensor<3x1x1x2xf16, {order = #NHWC}>

    // CHECK:            [[SUM:%.+]] = VPU.MVN1SumOp([[INPUT]])
    // CHECK-SAME:           :  tensor<3x1x20001x1xf16, {order = #NHWC}> -> tensor<3x1x1x2xf32, {order = #NHWC}>

    // CHECK:            [[VAR:%.+]] = VPU.MVN1MeanVar([[SUM]])
    // CHECK-SAME:           : tensor<3x1x1x2xf32, {order = #NHWC}> -> tensor<3x1x1x2xf16, {order = #NHWC}>

    // CHECK:            return [[VAR]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @NoTilingMVN1Sum
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<3x1x20001x1xf16, {order = #NHWC}>
func.func @NoTilingMVN1Sum(%arg0: tensor<3x1x20001x1xf16, {order = #NHWC}>) -> (tensor<3x1x1x2xf16, {order = #NHWC}>) {
      %0 = VPU.MVN1SumOp(%arg0) {across_channels = true, normalize_variance = true, output_height = 4 : i64} : tensor<3x1x20001x1xf16, {order = #NHWC}> -> tensor<3x1x4x2xf32, {order = #NHWC}>
      %1 = VPU.MVN1MeanVar(%0) {across_channels = true, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [3, 1, 20001, 1], output_type = f16} : tensor<3x1x4x2xf32, {order = #NHWC}> -> tensor<3x1x1x2xf16, {order = #NHWC}>
      return %1 : tensor<3x1x1x2xf16, {order = #NHWC}>

    // CHECK:            [[SUM:%.+]] = VPU.MVN1SumOp([[INPUT]])
    // CHECK-SAME:           :  tensor<3x1x20001x1xf16, {order = #NHWC}> -> tensor<3x1x1x2xf32, {order = #NHWC}>

    // CHECK:            [[VAR:%.+]] = VPU.MVN1MeanVar([[SUM]])
    // CHECK-SAME:           : tensor<3x1x1x2xf32, {order = #NHWC}> -> tensor<3x1x1x2xf16, {order = #NHWC}>

    // CHECK:            return [[VAR]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @SOKNoTilingMVN1Sum
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x20001x1xf16>
func.func @SOKNoTilingMVN1Sum(%arg0: tensor<1x3x20001x1xf16>) -> (tensor<1x3x1x2xf16, {order = #NHWC}>) {
      %0 = VPU.MVN1SumOp(%arg0) {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, output_height = 4 : i64} : tensor<1x3x20001x1xf16> -> tensor<1x3x4x2xf32, {order = #NHWC}>
      %1 = VPU.MVN1MeanVar(%0) {across_channels = false, eps = 1.000000e-09 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true, orig_shape = [1, 3, 20001, 1], output_type = f16} : tensor<1x3x4x2xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>
      return %1 : tensor<1x3x1x2xf16, {order = #NHWC}>

    // CHECK:            [[SUM:%.+]] = VPU.MVN1SumOp([[INPUT]])
    // CHECK-SAME:           :  tensor<1x3x20001x1xf16> -> tensor<1x3x1x2xf32, {order = #NHWC}>

    // CHECK:            [[VAR:%.+]] = VPU.MVN1MeanVar([[SUM]])
    // CHECK-SAME:           : tensor<1x3x1x2xf32, {order = #NHWC}> -> tensor<1x3x1x2xf16, {order = #NHWC}>

    // CHECK:            return [[VAR]]
}
