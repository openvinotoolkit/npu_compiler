//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --reshape-max-pool %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

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
    // CHECK:       [[VAL1:%.+]] = IE.Expand([[VAL0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 32, 0, 0]} : tensor<1x46560x1x14xf16> -> tensor<1x46592x1x14xf16>
    // CHECK:       [[VAL2:%.+]] = IE.Reshape([[VAL1]]) {shape_value = [1, 16, 2912, 14]} : tensor<1x46592x1x14xf16> -> tensor<1x16x2912x14xf16>
    // CHECK:       [[VAL3:%.+]] = IE.MaxPool([[VAL2]]) {kernel_size = [1, 7], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 7]} : tensor<1x16x2912x14xf16> -> tensor<1x16x2912x2xf16>
    // CHECK:       [[VAL4:%.+]] = IE.Reshape([[VAL3]]) {shape_value = [1, 46592, 1, 2]} : tensor<1x16x2912x2xf16> -> tensor<1x46592x1x2xf16>
    // CHECK:       [[VAL5:%.+]] = IE.Slice [[VAL4]] [0, 0, 0, 0] [1, 46560, 1, 2] : tensor<1x46592x1x2xf16> to tensor<1x46560x1x2xf16>
    // CHECK:       [[VAL6:%.+]] = IE.Transpose([[VAL5]]) {order_value = #NCWH} : tensor<1x46560x1x2xf16> -> tensor<1x46560x2x1xf16>
    // CHECK:       return [[VAL6]]
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

// -----

// CHECK-LABEL: @MaxPoolReshapedAlignedNoPadding
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x8448x80x1xf16>)
func.func @MaxPoolReshapedAlignedNoPadding(%arg0: tensor<1x8448x80x1xf16>) -> tensor<1x8448x1x1xf16> {
    %1 =IE.MaxPool(%arg0) {kernel_size = [80, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x8448x80x1xf16> -> tensor<1x8448x1x1xf16>

    return %1 : tensor<1x8448x1x1xf16>

    // CHECK:       [[VAL0:%.+]] = IE.Transpose([[ARG0]]) {order_value = #NCWH} : tensor<1x8448x80x1xf16> -> tensor<1x8448x1x80xf16>
    // CHECK:       [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {shape_value = [1, 16, 528, 80]} : tensor<1x8448x1x80xf16> -> tensor<1x16x528x80xf16>
    // CHECK:       [[VAL2:%.+]] = IE.MaxPool([[VAL1]]) {kernel_size = [1, 80], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x528x80xf16> -> tensor<1x16x528x1xf16>
    // CHECK:       [[VAL3:%.+]] = IE.Reshape([[VAL2]]) {shape_value = [1, 8448, 1, 1]} : tensor<1x16x528x1xf16> -> tensor<1x8448x1x1xf16>
    // CHECK:       [[VAL4:%.+]] = IE.Transpose([[VAL3]]) {order_value = #NCWH} : tensor<1x8448x1x1xf16> -> tensor<1x8448x1x1xf16>
    // CHECK:       return [[VAL4]]
}
