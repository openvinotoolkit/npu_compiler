//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @FoldNegative
// CHECK: [[INPUT:%.+]]: tensor<1x2x3x4xf32>
func.func @FoldNegative(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
    %0 = IE.Negative(%arg0) : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32>
    %1 = IE.Negative(%0) : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32>
    return %1 : tensor<1x2x3x4xf32>

    // CHECK-NOT:   IE.Negative
    // CHECK:       return [[INPUT]] : tensor<1x2x3x4xf32>
}

// -----

// CHECK-LABEL: @FoldNegativeMultiUsersAdd
// CHECK: [[INPUT:%.+]]: tensor<1x2x3x4xf32>
func.func @FoldNegativeMultiUsersAdd(%arg0: tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) {
    %0 = IE.Negative(%arg0) : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32>
    %1 = IE.Negative(%0) : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32>
    %2 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32>
    return %1, %2 : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>

    // CHECK: [[NEG:%.+]] = IE.Negative([[INPUT]]) : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[NEG]], [[NEG]])
    // CHECK:   -> tensor<1x2x3x4xf32>
    // CHECK: return [[INPUT]], [[ADD]] : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>
}

// -----

// CHECK-LABEL: @FoldNegativeMultiUsersNegative
// CHECK: [[INPUT:%.+]]: tensor<1x2x3x4xf32>
func.func @FoldNegativeMultiUsersNegative(%arg0: tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) {
    %0 = IE.Negative(%arg0) : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32>
    %1 = IE.Negative(%0) : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32>
    %2 = IE.Negative(%0) : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32>
    return %1, %2 : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>

    // CHECK: return [[INPUT]], [[INPUT]] : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>
}
