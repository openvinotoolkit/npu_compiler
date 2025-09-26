//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @ConvertToAdd
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<512x1024xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<512x1024xf16>
func.func @ConvertToAdd(%arg0 : tensor<512x1024xf16>, %arg1: tensor<512x1024xf16>) -> tensor<512x1024xf16> {
    %0 = IE.Accumulate(%arg0, %arg1) {operandSegmentSizes = array<i32: 1, 1, 0, 0>} : tensor<512x1024xf16>, tensor<512x1024xf16> -> tensor<512x1024xf16>
    return %0 : tensor<512x1024xf16>

    // CHECK:    [[ADD:%.+]] = IE.Add([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<512x1024xf16>, tensor<512x1024xf16> -> tensor<512x1024xf16>

    // CHECK:     return [[ADD]] : tensor<512x1024xf16>
}

// -----

// CHECK-LABEL: @NotConvertAccumulate
// CHECK-SAME:   ([[LHS:%.+]]: tensor<16x96xf16>, [[RHS:%.+]]: tensor<16x96xf16>
// CHECK-SAME:    [[LHS_SCALE:%.+]]: tensor<96xf16>, [[RHS_SCALE:%.+]]: tensor<96xf16>)
func.func @NotConvertAccumulate(%LHS: tensor<16x96xf16>,
                                %RHS: tensor<16x96xf16>,
                                %LHS_SCALE: tensor<96xf16>,
                                %RHS_SCALE: tensor<96xf16>) -> tensor<16x96xf16> {

    %0 = IE.Accumulate(%LHS, %RHS, %LHS_SCALE, %RHS_SCALE) {
        operandSegmentSizes = array<i32: 1, 1, 1, 1>
    } : tensor<16x96xf16>, tensor<16x96xf16>, tensor<96xf16>, tensor<96xf16> -> tensor<16x96xf16>

    return %0 : tensor<16x96xf16>

    // CHECK: [[ACCUMULATE:%.+]] = IE.Accumulate([[LHS]], [[RHS]], [[LHS_SCALE]], [[RHS_SCALE]]) {
    // CHECK:   operandSegmentSizes = array<i32: 1, 1, 1, 1>
    // CHECK: } : tensor<16x96xf16>, tensor<16x96xf16>, tensor<96xf16>, tensor<96xf16> -> tensor<16x96xf16>

    // CHECK: return [[ACCUMULATE]] : tensor<16x96xf16>
}
