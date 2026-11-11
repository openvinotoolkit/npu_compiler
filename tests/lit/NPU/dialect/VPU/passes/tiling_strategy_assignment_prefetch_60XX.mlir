//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --tiling-strategy-assignment %s | FileCheck %s
// REQUIRES: platform-NPU6010

// CHECK-LABEL: func.func @RMSSOK_TileOverC
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x192x192x512xf16>)
func.func @RMSSOK_TileOverC(%arg0: tensor<1x192x192x512xf16>) -> tensor<1x192x192x512xf16> {
    %cst = const.Declare tensor<1x1x1x512xf16> = dense<1.0> : tensor<1x1x1x512xf16>
    %0 = VPU.RMS(%arg0, %cst) {
        eps = 9.9999999747524271E-7 : f64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
        : tensor<1x192x192x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x192x192x512xf16>
    return %0 : tensor<1x192x192x512xf16>

    // CHECK:      [[CST:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+00> : tensor<1x1x1x512xf16>
    // CHECK:      [[OUTPUT:%.+]] = VPU.RMS([[INPUT]], [[CST]]) {
    // CHECK-SAME:    eps = 9.9999999747524271E-7 : f64
    // CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:    tilingStrategy = [1, 24, 1, 1]

    // CHECK:      return [[OUTPUT]] : tensor<1x192x192x512xf16>
}
