//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @LegalizeEps
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x100x512x1xf32>)
func.func @LegalizeEps(%arg0 : tensor<1x100x512x1xf32>) -> tensor<1x100x512x1xf32> {
    %0 = IE.MVN(%arg0) {across_channels = false, eps = 9.999999960041972E-13 : f64, normalize_variance = true} : tensor<1x100x512x1xf32> -> tensor<1x100x512x1xf32>
    return %0 : tensor<1x100x512x1xf32>

    // CHECK:    [[MVN:%.+]] = IE.MVN([[INPUT]]) {across_channels = false, eps = 1.1920928955078125E-7 : f64, normalize_variance = true} : tensor<1x100x512x1xf32> -> tensor<1x100x512x1xf32>
    // CHECK:    return [[MVN]]
}

// CHECK-LABEL: @ReshapeBatched
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<32x16x64x64xf32>)
func.func @ReshapeBatched(%arg0 : tensor<32x16x64x64xf32>) -> tensor<32x16x64x64xf32> {
    %0 = IE.MVN(%arg0) {across_channels = true, eps = 9.999999960041972E-13 : f64, normalize_variance = true} : tensor<32x16x64x64xf32> -> tensor<32x16x64x64xf32>
    return %0 : tensor<32x16x64x64xf32>

    // CHECK:    [[RESHAPE_1:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 32, 16, 4096]} : tensor<32x16x64x64xf32> -> tensor<1x32x16x4096xf32>
    // CHECK:    [[MVN:%.+]] = IE.MVN([[RESHAPE_1]])
    // CHECK-SAME{LITERAL}:   {across_channels = false, eps = 1.1920928955078125E-7 : f64, normalize_variance = true} : tensor<1x32x16x4096xf32> -> tensor<1x32x16x4096xf32>
    // CHECK:    [[RESHAPE_2:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [32, 16, 64, 64]} : tensor<1x32x16x4096xf32> -> tensor<32x16x64x64xf32>
    // CHECK:    return [[RESHAPE_2]]
}
