//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --adjust-maxpool-input-shape %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @ReshapeInputForMaxPool
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x512x512x1xf16>
func.func @ReshapeInputForMaxPool(%arg0 : tensor<1x512x512x1xf16>) -> (tensor<1x512x1x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [512, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x512x512x1xf16> -> tensor<1x512x1x1xf16>

    return %max_pool : tensor<1x512x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 512, 32, 16]} : tensor<1x512x512x1xf16> -> tensor<1x512x32x16xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME{LITERAL}:  {kernel_size = [32, 16], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:  tensor<1x512x32x16xf16> -> tensor<1x512x1x1xf16>
}

// -----

// CHECK-LABEL: @DoNotReshapeInputForMaxPool
func.func @DoNotReshapeInputForMaxPool(%arg0 : tensor<1x512x512x1xf16>) -> (tensor<1x512x1x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [514, 1],
        pads_begin = [1, 0],
        pads_end = [1, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x512x512x1xf16> -> tensor<1x512x1x1xf16>

    return %max_pool : tensor<1x512x1x1xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME{LITERAL}:  {kernel_size = [514, 1], pads_begin = [1, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:  tensor<1x512x512x1xf16> -> tensor<1x512x1x1xf16>
}

// -----

// CHECK-LABEL: @DoNotReshapeInputForMaxPool1x1Kernel
func.func @DoNotReshapeInputForMaxPool1x1Kernel(%arg0 : tensor<1x512x1x1xf16>) -> (tensor<1x512x1x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x512x1x1xf16> -> tensor<1x512x1x1xf16>

    return %max_pool : tensor<1x512x1x1xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME{LITERAL}:  {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:  tensor<1x512x1x1xf16> -> tensor<1x512x1x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeInputForMaxPoolWithSameKnernelAndStrideOnW
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x128x1x1000xf16>
func.func @ReshapeInputForMaxPoolWithSameKnernelAndStrideOnW(%arg0 : tensor<1x128x1x1000xf16>) -> (tensor<1x128x1x200xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 5],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 5]
    } : tensor<1x128x1x1000xf16> -> tensor<1x128x1x200xf16>

    return %max_pool : tensor<1x128x1x200xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 128, 4, 250]} : tensor<1x128x1x1000xf16> -> tensor<1x128x4x250xf16>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[RESHAPE0]])
    // CHECK-SAME{LITERAL}:  {kernel_size = [1, 5], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 5]} : tensor<1x128x4x250xf16> -> tensor<1x128x4x50xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[MAXPOOL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 1, 200]} : tensor<1x128x4x50xf16> -> tensor<1x128x1x200xf16>

    // CHECK: return [[RESHAPE1]] : tensor<1x128x1x200xf16>
}

// -----

// CHECK-LABEL: @DoNotReshapeInputForMaxPoolWithDifferentKnernelAndStrideOnW
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x128x1x1000xf16>
func.func @DoNotReshapeInputForMaxPoolWithDifferentKnernelAndStrideOnW(%arg0 : tensor<1x128x1x1000xf16>) -> (tensor<1x128x1x199xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 10],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 5]
    } : tensor<1x128x1x1000xf16> -> tensor<1x128x1x199xf16>

    return %max_pool : tensor<1x128x1x199xf16>

    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[ARG_0]])
    // CHECK-SAME{LITERAL}:  {kernel_size = [1, 10], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 5]} : tensor<1x128x1x1000xf16> -> tensor<1x128x1x199xf16>

    // CHECK: return [[MAXPOOL]] : tensor<1x128x1x199xf16>
}

// -----

// CHECK-LABEL: @ReshapeInputForMaxPoolWithSameKnernelAndStrideOnH
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x128x1000x1xf16>
func.func @ReshapeInputForMaxPoolWithSameKnernelAndStrideOnH(%arg0 : tensor<1x128x1000x1xf16>) -> (tensor<1x128x200x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [5, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [5, 1]
    } : tensor<1x128x1000x1xf16> -> tensor<1x128x200x1xf16>

    return %max_pool : tensor<1x128x200x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 250, 4]} : tensor<1x128x1000x1xf16> -> tensor<1x128x250x4xf16>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[RESHAPE0]])
    // CHECK-SAME{LITERAL}:  {kernel_size = [5, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [5, 1]} : tensor<1x128x250x4xf16> -> tensor<1x128x50x4xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[MAXPOOL]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 128, 200, 1]} : tensor<1x128x50x4xf16> -> tensor<1x128x200x1xf16>

    // CHECK: return [[RESHAPE1]] : tensor<1x128x200x1xf16>
}

// -----

// CHECK-LABEL: @DoNotReshapeInputForMaxPoolWithDifferentKnernelAndStrideOnH
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x128x1000x1xf16>
func.func @DoNotReshapeInputForMaxPoolWithDifferentKnernelAndStrideOnH(%arg0 : tensor<1x128x1000x1xf16>) -> (tensor<1x128x199x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [10, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [5, 1]
    } : tensor<1x128x1000x1xf16> -> tensor<1x128x199x1xf16>

    return %max_pool : tensor<1x128x199x1xf16>

    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[ARG_0]])
    // CHECK-SAME{LITERAL}:  {kernel_size = [10, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [5, 1]} : tensor<1x128x1000x1xf16> -> tensor<1x128x199x1xf16>

    // CHECK: return [[MAXPOOL]] : tensor<1x128x199x1xf16>
}
