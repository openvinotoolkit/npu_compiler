//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @UseLeakyRelu
func.func @UseLeakyRelu(%arg0: tensor<1x16x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %0 = const.Declare tensor<1x16xf32> = dense<1.0> : tensor<1x16xf32>
    %1 = IE.PRelu(%arg0, %0) :
        tensor<1x16x300x300xf32>, tensor<1x16xf32> -> tensor<1x16x300x300xf32>
    return %1 : tensor<1x16x300x300xf32>

    // CHECK:       %[[VAL0:.*]] = IE.LeakyRelu(%arg0)
    // CHECK-SAME:      negative_slope = 1.000000e+00
    // CHECK-NOT:   IE.PRelu
    // CHECK:       return %[[VAL0]]
}

// CHECK-LABEL: @LegalizeSlopeCase1
func.func @LegalizeSlopeCase1(%arg0: tensor<1x64x64x64xf32>, %arg1: tensor<64x1x1xf32>) -> tensor<1x64x64x64xf32> {
    %0 = IE.PRelu(%arg0, %arg1) :
        tensor<1x64x64x64xf32>, tensor<64x1x1xf32> -> tensor<1x64x64x64xf32>
    return %0 : tensor<1x64x64x64xf32>

    // CHECK:       %[[VAL0:.*]] = IE.AffineReshape(%arg1) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 64, 1, 1]}
    // CHECK-SAME:              : tensor<64x1x1xf32> -> tensor<1x64x1x1xf32>
    // CHECK:       %[[VAL1:.*]] = IE.PRelu(%arg0, %[[VAL0]])
    // CHECK-SAME:              : tensor<1x64x64x64xf32>, tensor<1x64x1x1xf32> -> tensor<1x64x64x64xf32>
    // CHECK:       return %[[VAL1]]
}

// CHECK-LABEL: @LegalizeSlopeCase2
func.func @LegalizeSlopeCase2(%arg0: tensor<1x32x96x96xf32>, %arg1: tensor<1x32x1x1xf32>) -> tensor<1x32x96x96xf32> {
    %0 = IE.PRelu(%arg0, %arg1) :
        tensor<1x32x96x96xf32>, tensor<1x32x1x1xf32> -> tensor<1x32x96x96xf32>
    return %0 : tensor<1x32x96x96xf32>

    // CHECK:       %[[VAL0:.*]] = IE.PRelu(%arg0, %arg1)
    // CHECK-SAME:              : tensor<1x32x96x96xf32>, tensor<1x32x1x1xf32> -> tensor<1x32x96x96xf32>
    // CHECK:                   return %[[VAL0]]
}

// CHECK-LABEL: @LegalizeSlopeCase3
func.func @LegalizeSlopeCase3(%arg0: tensor<1x32x96x96xf32>, %arg1: tensor<32xf32>) -> tensor<1x32x96x96xf32> {
    %0 = IE.PRelu(%arg0, %arg1) :
        tensor<1x32x96x96xf32>, tensor<32xf32> -> tensor<1x32x96x96xf32>
    return %0 : tensor<1x32x96x96xf32>

    // CHECK:       %[[VAL0:.*]] = IE.AffineReshape(%arg1) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 32, 1, 1]}
    // CHECK-SAME:              : tensor<32xf32> -> tensor<1x32x1x1xf32>
    // CHECK:       %[[VAL1:.*]] = IE.PRelu(%arg0, %[[VAL0]])
    // CHECK-SAME:              : tensor<1x32x96x96xf32>, tensor<1x32x1x1xf32> -> tensor<1x32x96x96xf32>
    // CHECK:                   return %[[VAL1]]
}
