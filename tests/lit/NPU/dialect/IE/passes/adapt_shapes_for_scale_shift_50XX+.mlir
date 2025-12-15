//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adapt-shapes-for-scale-shift --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @DoNotConvert4dMulWithTwoActivationsSameShape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x1x80xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x1x80xf16>
func.func @DoNotConvert4dMulWithTwoActivationsSameShape(%arg0: tensor<1x1x1x80xf16>, %arg1: tensor<1x1x1x80xf16>) -> tensor<1x1x1x80xf16> {
    %MUL = IE.Multiply(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x1x1x80xf16>, tensor<1x1x1x80xf16> -> tensor<1x1x1x80xf16>

    return %MUL : tensor<1x1x1x80xf16>


    // CHECK:   [[MUL:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x1x80xf16>, tensor<1x1x1x80xf16> -> tensor<1x1x1x80xf16>


    // CHECK:   return [[MUL]] : tensor<1x1x1x80xf16>
}

// -----

// CHECK-LABEL: @Convert4dMulWithTwoActivations
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x19x80x1xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x80x1xf16>
func.func @Convert4dMulWithTwoActivations(%arg0: tensor<1x19x80x1xf16>, %arg1: tensor<1x1x80x1xf16>) -> tensor<1x19x80x1xf16> {
    %MUL = IE.Multiply(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x19x80x1xf16>, tensor<1x1x80x1xf16> -> tensor<1x19x80x1xf16>

    return %MUL : tensor<1x19x80x1xf16>

    // CHECK:   [[TRANSPOSE_INPUT1:%.+]] = IE.Transpose([[INPUT_0]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[RESHAPE_INPUT1:%.+]] = IE.AffineReshape([[INPUT_1]]) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x1x80x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.+]] = IE.Multiply([[TRANSPOSE_INPUT1]], [[RESHAPE_INPUT1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x19x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[MUL]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x80x19x1xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   return [[TRANSPOSE_OUT]] : tensor<1x19x80x1xf16>
}
