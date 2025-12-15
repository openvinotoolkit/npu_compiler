//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --reshape-max-pool %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// -----

// CHECK-LABEL: @MaxPoolReshaped
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x42840x14x1xf16>)
func.func @MaxPoolReshaped(%arg0: tensor<1x42840x14x1xf16>) -> tensor<1x42840x2x1xf16> {
    %1 =IE.MaxPool(%arg0) {kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [7, 1]} : tensor<1x42840x14x1xf16> -> tensor<1x42840x2x1xf16>

    return %1 : tensor<1x42840x2x1xf16>

    // CHECK:       [[VAL0:%.+]] = IE.Transpose([[ARG0]]) {order_value = #NCWH} : tensor<1x42840x14x1xf16> -> tensor<1x42840x1x14xf16>
    // CHECK:       [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {shape_value = [1, 6, 7140, 14]} : tensor<1x42840x1x14xf16> -> tensor<1x6x7140x14xf16>
    // CHECK:       [[VAL2:%.+]] = IE.MaxPool([[VAL1]]) {kernel_size = [1, 7], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 7]} : tensor<1x6x7140x14xf16> -> tensor<1x6x7140x2xf16>
    // CHECK:       [[VAL3:%.+]] = IE.Reshape([[VAL2]]) {shape_value = [1, 42840, 1, 2]} : tensor<1x6x7140x2xf16> -> tensor<1x42840x1x2xf16>
    // CHECK:       [[VAL4:%.+]] = IE.Transpose([[VAL3]]) {order_value = #NCWH} : tensor<1x42840x1x2xf16> -> tensor<1x42840x2x1xf16>
    // CHECK:       return [[VAL4]]
}

// -----

// CHECK-LABEL: @MaxPoolReshapedChannelAligned
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x46560x14x1xf16>)
func.func @MaxPoolReshapedChannelAligned(%arg0: tensor<1x46560x14x1xf16>) -> tensor<1x46560x2x1xf16> {
    %1 =IE.MaxPool(%arg0) {kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [7, 1]} : tensor<1x46560x14x1xf16> -> tensor<1x46560x2x1xf16>

    return %1 : tensor<1x46560x2x1xf16>

    // CHECK:       [[VAL0:%.+]] = IE.Transpose([[ARG0]]) {order_value = #NCWH} : tensor<1x46560x14x1xf16> -> tensor<1x46560x1x14xf16>
    // CHECK:       [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {shape_value = [1, 16, 2910, 14]} : tensor<1x46560x1x14xf16> -> tensor<1x16x2910x14xf16>
    // CHECK:       [[VAL2:%.+]] = IE.MaxPool([[VAL1]]) {kernel_size = [1, 7], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 7]} : tensor<1x16x2910x14xf16> -> tensor<1x16x2910x2xf16>
    // CHECK:       [[VAL3:%.+]] = IE.Reshape([[VAL2]]) {shape_value = [1, 46560, 1, 2]} : tensor<1x16x2910x2xf16> -> tensor<1x46560x1x2xf16>
    // CHECK:       [[VAL4:%.+]] = IE.Transpose([[VAL3]]) {order_value = #NCWH} : tensor<1x46560x1x2xf16> -> tensor<1x46560x2x1xf16>
    // CHECK:       return [[VAL4]]
}

// -----

// CHECK-LABEL: @MaxPoolChannelDimLowerThanVPUDimensionLimit
// CHECK-SAME: ([[ARG0:%.+]]: tensor<3x4x64x1xf16>)
func.func @MaxPoolChannelDimLowerThanVPUDimensionLimit(%arg0: tensor<3x4x64x1xf16>) -> tensor<3x4x64x1xf16> {
    %1 =IE.MaxPool(%arg0) {kernel_size = [3, 1], pads_begin = [1, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<3x4x64x1xf16> -> tensor<3x4x64x1xf16>

    return %1 : tensor<3x4x64x1xf16>

    // CHECK:       [[VAL0:%.+]] = IE.MaxPool([[ARG0]]) {kernel_size = [3, 1], pads_begin = [1, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<3x4x64x1xf16> -> tensor<3x4x64x1xf16>
    // CHECK:       return [[VAL0]]
}
