//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-batch %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK-LABEL:  @UnrollSubtract
// CHECK-SAME:   [[IN1:%.+]]: tensor<2x3x576x576xf16>
// CHECK-SAME:   [[IN2:%.+]]: tensor<2x1x576x576xf16>
func.func @UnrollSubtract(%input1 : tensor<2x3x576x576xf16>, %input2 : tensor<2x1x576x576xf16>) -> tensor<2x3x576x576xf16> {
    %0 = IE.Subtract(%input1, %input2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }
                     : tensor<2x3x576x576xf16>, tensor<2x1x576x576xf16> -> tensor<2x3x576x576xf16>
    return %0 : tensor<2x3x576x576xf16>

    // CHECK: [[SLICE0_ARG0:%.+]] = IE.Slice [[IN1]] [0, 0, 0, 0] [1, 3, 576, 576] :
    // CHECK-SAME:      tensor<2x3x576x576xf16> to tensor<1x3x576x576xf16>

    // CHECK: [[SLICE0_ARG1:%.+]] = IE.Slice [[IN2]] [0, 0, 0, 0] [1, 1, 576, 576] :
    // CHECK-SAME:      tensor<2x1x576x576xf16> to tensor<1x1x576x576xf16>

    // CHECK: [[SUB1:%.+]] = IE.Subtract([[SLICE0_ARG0]], [[SLICE0_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME: tensor<1x3x576x576xf16>, tensor<1x1x576x576xf16> -> tensor<1x3x576x576xf16>

    // CHECK: [[SLICE1_ARG0:%.+]] = IE.Slice [[IN1]] [1, 0, 0, 0] [1, 3, 576, 576] :
    // CHECK-SAME: tensor<2x3x576x576xf16> to tensor<1x3x576x576xf16>

    // CHECK: [[SLICE1_ARG1:%.+]] = IE.Slice [[IN2]] [1, 0, 0, 0] [1, 1, 576, 576] :
    // CHECK-SAME: tensor<2x1x576x576xf16> to tensor<1x1x576x576xf16>

    // CHECK: [[SUB2:%.+]] = IE.Subtract([[SLICE1_ARG0]], [[SLICE1_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME: tensor<1x3x576x576xf16>, tensor<1x1x576x576xf16> -> tensor<1x3x576x576xf16>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[SUB1]], [[SUB2]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1x3x576x576xf16>, tensor<1x3x576x576xf16> -> tensor<2x3x576x576xf16>

    // CHECK:   return [[CONCAT]] : tensor<2x3x576x576xf16>
}
