//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --handle-large-kernels %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @HandleLargeKernelsAvgPoolWithSameKernelSize
func.func @HandleLargeKernelsAvgPoolWithSameKernelSize(%arg0 : tensor<1x128x16x16xf16>) -> (tensor<1x128x1x1xf16>) {
    %avg_pool = IE.AvgPool(%arg0) {
        kernel_size = [16, 16],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [16, 16]
    } : tensor<1x128x16x16xf16> -> tensor<1x128x1x1xf16>

    return %avg_pool : tensor<1x128x1x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [8, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x128x16x16xf16> -> tensor<1x128x2x2xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [2, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x128x2x2xf16> -> tensor<1x128x1x1xf16>
}

// CHECK-LABEL: @HandleLargeKernelsAvgPoolWithPadNeedAndMultiSplit
func.func @HandleLargeKernelsAvgPoolWithPadNeedAndMultiSplit(%arg0 : tensor<1x2048x69x69xf16>) -> (tensor<1x2048x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [69, 69],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [69, 69]
    } : tensor<1x2048x69x69xf16> -> tensor<1x2048x1x1xf16>

    return %ave_pool : tensor<1x2048x1x1xf16>
    // CHECK-DAG:       const.Declare
    // CHECK-SAME:      tensor<2048x1x8x8xf16> = dense<1.701350e-02> : tensor<2048x1x8x8xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [3, 3]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [3, 3]
    // CHECK-SAME:      : tensor<1x2048x69x69xf16> -> tensor<1x2048x23x23xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 2048 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x2048x23x23xf16>, tensor<2048x1x8x8xf16> -> tensor<1x2048x3x3xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [3, 3]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x2048x3x3xf16> -> tensor<1x2048x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsAvgPoolWithPadNeedAndMultiSplitAsymmetric
func.func @HandleLargeKernelsAvgPoolWithPadNeedAndMultiSplitAsymmetric(%arg0 : tensor<1x16x258x257xf16>) -> (tensor<1x16x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [258, 257],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x258x257xf16> -> tensor<1x16x1x1xf16>

    return %ave_pool : tensor<1x16x1x1xf16>
    // CHECK-DAG:       const.Declare tensor<16x1x6x6xf16> = dense<2.789310e-02> : tensor<16x1x6x6xf16>
    // CHECK-DAG:       const.Declare tensor<16x1x4x4xf16> = dense<6.542960e-02> : tensor<16x1x4x4xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 16 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [6, 6]
    // CHECK-SAME:      : tensor<1x16x258x257xf16>, tensor<16x1x6x6xf16> -> tensor<1x16x43x43xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 16 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [4, 4]
    // CHECK-SAME:      : tensor<1x16x43x43xf16>, tensor<16x1x4x4xf16> -> tensor<1x16x11x11xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [11, 11]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x11x11xf16> -> tensor<1x16x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsXMaxPool
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x64x10x17xf16>
func.func @HandleLargeKernelsXMaxPool(%arg0 : tensor<1x64x10x17xf16>) -> (tensor<1x64x10x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 17],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 17]
    } : tensor<1x64x10x17xf16> -> tensor<1x64x10x1xf16>

    return %max_pool : tensor<1x64x10x1xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 64, 10, 1] : tensor<1x64x10x17xf16> to tensor<1x64x10x1xf16>
    // CHECK:       IE.Concat([[ARG_0]], [[SLICE]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 3 : i64>
    // CHECK-SAME:      : tensor<1x64x10x17xf16>, tensor<1x64x10x1xf16> -> tensor<1x64x10x18xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [1, 6]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 6]
    // CHECK-SAME:      : tensor<1x64x10x18xf16> -> tensor<1x64x10x3xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [1, 3]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x64x10x3xf16> -> tensor<1x64x10x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsYMaxPool
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x64x17x10xf16>
func.func @HandleLargeKernelsYMaxPool(%arg0 : tensor<1x64x17x10xf16>) -> (tensor<1x64x1x10xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [17, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [17, 1]
    } : tensor<1x64x17x10xf16> -> tensor<1x64x1x10xf16>

    return %max_pool : tensor<1x64x1x10xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 64, 1, 10] : tensor<1x64x17x10xf16> to tensor<1x64x1x10xf16>
    // CHECK:       IE.Concat([[ARG_0]], [[SLICE]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 2 : i64>
    // CHECK-SAME:      : tensor<1x64x17x10xf16>, tensor<1x64x1x10xf16> -> tensor<1x64x18x10xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [6, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [6, 1]
    // CHECK-SAME:      : tensor<1x64x18x10xf16> -> tensor<1x64x3x10xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [3, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x64x3x10xf16> -> tensor<1x64x1x10xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsMaxPoolWithSameKernelSize
func.func @HandleLargeKernelsMaxPoolWithSameKernelSize(%arg0 : tensor<1x16x32x32xf16>) -> (tensor<1x16x1x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [32, 32],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x32x32xf16> -> tensor<1x16x1x1xf16>

    return %max_pool : tensor<1x16x1x1xf16>

    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [8, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x16x32x32xf16> -> tensor<1x16x4x4xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [4, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x4x4xf16> -> tensor<1x16x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsMaxPoolWithDiffKernelSize
func.func @HandleLargeKernelsMaxPoolWithDiffKernelSize(%arg0 : tensor<1x128x9x16xf16>) -> (tensor<1x128x1x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [9, 16],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [9, 16]
    } : tensor<1x128x9x16xf16> -> tensor<1x128x1x1xf16>

    return %max_pool : tensor<1x128x1x1xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [9, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 8]
    // CHECK-SAME:      : tensor<1x128x9x16xf16> -> tensor<1x128x1x2xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x128x1x2xf16> -> tensor<1x128x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsMaxPoolWithPadNeed
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x1x71x2xf16>
func.func @HandleLargeKernelsMaxPoolWithPadNeed(%arg0 : tensor<1x1x71x2xf16>) -> (tensor<1x1x1x2xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [71, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x1x71x2xf16> -> tensor<1x1x1x2xf16>

    return %max_pool : tensor<1x1x1x2xf16>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 1, 1, 2] : tensor<1x1x71x2xf16> to tensor<1x1x1x2xf16>
    // CHECK:       IE.Concat([[ARG_0]], [[SLICE]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 2 : i64>
    // CHECK-SAME:      : tensor<1x1x71x2xf16>, tensor<1x1x1x2xf16> -> tensor<1x1x72x2xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [8, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [8, 1]
    // CHECK-SAME:      : tensor<1x1x72x2xf16> -> tensor<1x1x9x2xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [9, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x1x9x2xf16> -> tensor<1x1x1x2xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsMaxPoolWithPadNeedAndMultiSplit
func.func @HandleLargeKernelsMaxPoolWithPadNeedAndMultiSplit(%arg0 : tensor<1x2048x69x69xf16>) -> (tensor<1x2048x1x1xf16>) {
    %ave_pool = IE.MaxPool(%arg0) {
        kernel_size = [69, 69],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [69, 69]
    } : tensor<1x2048x69x69xf16> -> tensor<1x2048x1x1xf16>

    return %ave_pool : tensor<1x2048x1x1xf16>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool
    // CHECK-SAME:      kernel_size = [3, 3]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [3, 3]
    // CHECK-SAME:      : tensor<1x2048x69x69xf16> -> tensor<1x2048x23x23xf16>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[MAXPOOL]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 2048, 23, 1] : tensor<1x2048x23x23xf16> to tensor<1x2048x23x1xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[MAXPOOL]], [[SLICE0]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 3 : i64>
    // CHECK-SAME:      : tensor<1x2048x23x23xf16>, tensor<1x2048x23x1xf16> -> tensor<1x2048x23x24xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[CONCAT]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 2048, 1, 23] : tensor<1x2048x23x24xf16> to tensor<1x2048x1x23xf16>
    // CHECK:       IE.Concat([[CONCAT]], [[SLICE1]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 2 : i64>
    // CHECK-SAME:      : tensor<1x2048x23x24xf16>, tensor<1x2048x1x23xf16> -> tensor<1x2048x24x24xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [8, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x2048x24x24xf16> -> tensor<1x2048x3x3xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [3, 3]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x2048x3x3xf16> -> tensor<1x2048x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsMaxPoolWithPadNeedAndMultiSplitAsymmetric
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x258x257xf16>
func.func @HandleLargeKernelsMaxPoolWithPadNeedAndMultiSplitAsymmetric(%arg0 : tensor<1x16x258x257xf16>) -> (tensor<1x16x1x1xf16>) {
    %ave_pool = IE.MaxPool(%arg0) {
        kernel_size = [258, 257],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x258x257xf16> -> tensor<1x16x1x1xf16>

    return %ave_pool : tensor<1x16x1x1xf16>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[ARG_0]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 16, 258, 1] : tensor<1x16x258x257xf16> to tensor<1x16x258x1xf16>
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[ARG_0]], [[SLICE0]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 3 : i64>
    // CHECK-SAME:      : tensor<1x16x258x257xf16>, tensor<1x16x258x1xf16> -> tensor<1x16x258x258xf16>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool
    // CHECK-SAME:      kernel_size = [6, 6]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [6, 6]
    // CHECK-SAME:      : tensor<1x16x258x258xf16> -> tensor<1x16x43x43xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[MAXPOOL]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 16, 43, 1] : tensor<1x16x43x43xf16> to tensor<1x16x43x1xf16>
    // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[MAXPOOL]], [[SLICE1]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 3 : i64>
    // CHECK-SAME:      : tensor<1x16x43x43xf16>, tensor<1x16x43x1xf16> -> tensor<1x16x43x44xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[CONCAT1]]
    // CHECK-SAME:       [0, 0, 0, 0] [1, 16, 1, 43] : tensor<1x16x43x44xf16> to tensor<1x16x1x43xf16>
    // CHECK:       [[CONCAT2:%.+]] = IE.Concat([[CONCAT1]], [[SLICE2]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 2 : i64>
    // CHECK-SAME:      : tensor<1x16x43x44xf16>, tensor<1x16x1x43xf16> -> tensor<1x16x44x44xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [4, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [4, 4]
    // CHECK-SAME:      : tensor<1x16x44x44xf16> -> tensor<1x16x11x11xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [11, 11]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x11x11xf16> -> tensor<1x16x1x1xf16>
}

// -----

// CHECK-LABEL: @CanNotHandleLargeKernelsMaxPoolWithPadNeed
func.func @CanNotHandleLargeKernelsMaxPoolWithPadNeed(%arg0 : tensor<1x2048x46x46xf16>) -> (tensor<1x2048x2x2xf16>) {
    %ave_pool = IE.MaxPool(%arg0) {
        kernel_size = [23, 23],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [23, 23]
    } : tensor<1x2048x46x46xf16> -> tensor<1x2048x2x2xf16>

    return %ave_pool : tensor<1x2048x2x2xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [23, 23]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [23, 23]
    // CHECK-SAME:      : tensor<1x2048x46x46xf16> -> tensor<1x2048x2x2xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsMaxPoolWithPadNeedIsGlobalPool
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x2048x48x23xf16>
func.func @HandleLargeKernelsMaxPoolWithPadNeedIsGlobalPool(%arg0 : tensor<1x2048x48x23xf16>) -> (tensor<1x2048x2x1xf16>) {
    %ave_pool = IE.MaxPool(%arg0) {
        kernel_size = [24, 23],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [24, 23]
    } : tensor<1x2048x48x23xf16> -> tensor<1x2048x2x1xf16>

    return %ave_pool : tensor<1x2048x2x1xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 2048, 48, 1] : tensor<1x2048x48x23xf16> to tensor<1x2048x48x1xf16>
    // CHECK:       IE.Concat([[ARG_0]], [[SLICE]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 3 : i64>
    // CHECK-SAME:      : tensor<1x2048x48x23xf16>, tensor<1x2048x48x1xf16> -> tensor<1x2048x48x24xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [8, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x2048x48x24xf16> -> tensor<1x2048x6x3xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [3, 3]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [3, 1]
    // CHECK-SAME:      : tensor<1x2048x6x3xf16> -> tensor<1x2048x2x1xf16>
}

// -----


// CHECK-LABEL: @HandleLargeKernelsOverlappedMaxPool
func.func @HandleLargeKernelsOverlappedMaxPool(%arg0 : tensor<1x512x23x23xf16>) -> (tensor<1x512x23x23xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [17, 17],
        pads_begin = [8, 8],
        pads_end = [8, 8],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x512x23x23xf16> -> tensor<1x512x23x23xf16>

    return %max_pool : tensor<1x512x23x23xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [9, 9]
    // CHECK-SAME:      pads_begin = [4, 4]
    // CHECK-SAME:      pads_end = [4, 4]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x512x23x23xf16> -> tensor<1x512x23x23xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [9, 9]
    // CHECK-SAME:      pads_begin = [4, 4]
    // CHECK-SAME:      pads_end = [4, 4]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x512x23x23xf16> -> tensor<1x512x23x23xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsOvSerlappedMaxPoolWithOneAxis
func.func @HandleLargeKernelsOvSerlappedMaxPoolWithOneAxis(%arg0 : tensor<1x512x23x23xf16>) -> (tensor<1x512x23x23xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [17, 1],
        pads_begin = [8, 0],
        pads_end = [8, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x512x23x23xf16> -> tensor<1x512x23x23xf16>

    return %max_pool : tensor<1x512x23x23xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [9, 1]
    // CHECK-SAME:      pads_begin = [4, 0]
    // CHECK-SAME:      pads_end = [4, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x512x23x23xf16> -> tensor<1x512x23x23xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [9, 1]
    // CHECK-SAME:      pads_begin = [4, 0]
    // CHECK-SAME:      pads_end = [4, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x512x23x23xf16> -> tensor<1x512x23x23xf16>
}

// -----
// CHECK-LABEL: @NotConvertDilatedConv
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x256x1x512xf16>
func.func @NotConvertDilatedConv(%arg0: tensor<1x256x1x512xf16>) -> tensor<1x256x1x512xf16> {
    %cst = const.Declare tensor<256x256x1x13xf16> = dense<1.000000e+00> : tensor<256x256x13xf32>, [#const.CastElemType<f16>, #const.Reshape<[256, 256, 1, 13]>]
    %conv = IE.Convolution(%arg0, %cst) {
        dilations = [1, 2],
        pads_begin = [0, 12],
        pads_end = [0, 12],
        strides = [1, 1]
    } : tensor<1x256x1x512xf16>, tensor<256x256x1x13xf16> -> tensor<1x256x1x512xf16>

    return %conv : tensor<1x256x1x512xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<256x256x1x13xf16> = dense<1.000000e+00> : tensor<256x256x13xf32>, [#const.CastElemType<f16>, #const.Reshape<[256, 256, 1, 13]>]
    // CHECK:           [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[CST]]) {
    // CHECK-SAME:          dilations = [1, 2],
    // CHECK-SAME:          pads_begin = [0, 12],
    // CHECK-SAME:          pads_end = [0, 12],
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x256x1x512xf16>, tensor<256x256x1x13xf16> -> tensor<1x256x1x512xf16>

    // CHECK:           return [[CONV]] : tensor<1x256x1x512xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConvWithOneDimOnW
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x1x1x160000xf16>
func.func @HandleLargeKernelsConvWithOneDimOnW(%arg0 : tensor<1x1x1x160000xf16>) -> tensor<1x257x1x1247xf16> {
    %cst = const.Declare tensor<257x1x1x512xf16> = dense<1.000000e+00> : tensor<257x1x1x512xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 128]} : tensor<1x1x1x160000xf16>, tensor<257x1x1x512xf16> -> tensor<1x257x1x1247xf16>
    return %conv : tensor<1x257x1x1247xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<257x64x1x8xf16> = dense<1.000000e+00> : tensor<257x1x1x512xf16>, [#const.Reshape<[257, 8, 1, 64]>, #const.Transpose<#NWHC>]
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[ARG_0]]) {shape_value = [1, 2500, 1, 64]} : tensor<1x1x1x160000xf16> -> tensor<1x2500x1x64xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]]) {order_value = #NWHC} : tensor<1x2500x1x64xf16> -> tensor<1x64x1x2500xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]} : tensor<1x64x1x2500xf16>, tensor<257x64x1x8xf16> -> tensor<1x257x1x1247xf16>
    // CHECK: return [[CONV]] : tensor<1x257x1x1247xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConvWithOneDimOnH
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x1x2176x1xf16>
func.func @HandleLargeKernelsConvWithOneDimOnH(%arg0 : tensor<1x1x2176x1xf16>) -> tensor<1x258x16x1xf16> {
    %cst = const.Declare tensor<258x1x512x1xf16> = dense<1.000000e+00> : tensor<258x1x512x1xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [128, 0], pads_end = [128, 0], strides = [128, 128]} : tensor<1x1x2176x1xf16>, tensor<258x1x512x1xf16> -> tensor<1x258x16x1xf16>
    return %conv : tensor<1x258x16x1xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<258x64x8x1xf16> = dense<1.000000e+00> : tensor<258x1x512x1xf16>, [#const.Reshape<[258, 8, 64, 1]>, #const.Transpose<#NHCW>]
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[ARG_0]]) {shape_value = [1, 34, 64, 1]} : tensor<1x1x2176x1xf16> -> tensor<1x34x64x1xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]]) {order_value = #NHCW} : tensor<1x34x64x1xf16> -> tensor<1x64x34x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE]], [[CST]]) {dilations = [1, 1], pads_begin = [2, 0], pads_end = [2, 0], strides = [2, 1]} : tensor<1x64x34x1xf16>, tensor<258x64x8x1xf16> -> tensor<1x258x16x1xf16>
    // CHECK: return [[CONV]] : tensor<1x258x16x1xf16>
}

// -----

// CHECK-LABEL: @AvgPoolWithPadding
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x48x64xf16>
func.func @AvgPoolWithPadding(%arg0 : tensor<1x16x48x64xf16>) -> (tensor<1x16x3x3xf16>) {
    %avg_pool = IE.AvgPool(%arg0) {
        kernel_size = [16, 22],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [16, 22]
    } : tensor<1x16x48x64xf16> -> tensor<1x16x3x3xf16>

    return %avg_pool : tensor<1x16x3x3xf16>
    // CHECK:           [[AVGPOOL1:%.+]] = IE.AvgPool([[ARG_0]]) {kernel_size = [8, 2], pads_begin = [0, 1], pads_end = [0, 1],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 2]} : tensor<1x16x48x64xf16> -> tensor<1x16x6x33xf16>
    // CHECK:           [[AVGPOOL2:%.+]] = IE.AvgPool([[AVGPOOL1]]) {kernel_size = [2, 11], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 11]} : tensor<1x16x6x33xf16> -> tensor<1x16x3x3xf16>

    // CHECK:           return [[AVGPOOL2]] : tensor<1x16x3x3xf16>
}

// -----

// CHECK-LABEL: @MaxPoolWithPadding
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x48x64xf16>
func.func @MaxPoolWithPadding(%arg0 : tensor<1x16x48x64xf16>) -> (tensor<1x16x3x3xf16>) {
    %avg_pool = IE.MaxPool(%arg0) {
        kernel_size = [16, 22],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [16, 22]
    } : tensor<1x16x48x64xf16> -> tensor<1x16x3x3xf16>

    return %avg_pool : tensor<1x16x3x3xf16>
    // CHECK:           [[MAXPOOL1:%.+]] = IE.MaxPool([[ARG_0]]) {kernel_size = [8, 2], pads_begin = [0, 1], pads_end = [0, 1],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 2]} : tensor<1x16x48x64xf16> -> tensor<1x16x6x33xf16>
    // CHECK:           [[MAXPOOL2:%.+]] = IE.MaxPool([[MAXPOOL1]]) {kernel_size = [2, 11], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 11]} : tensor<1x16x6x33xf16> -> tensor<1x16x3x3xf16>

    // CHECK:           return [[MAXPOOL2]] : tensor<1x16x3x3xf16>
}


// -----

// CHECK-LABEL: @AvgPoolWithExcludePadding
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x48x64xf16>
func.func @AvgPoolWithExcludePadding(%arg0 : tensor<1x16x48x64xf16>) -> (tensor<1x16x3x3xf16>) {
    %avg_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [16, 22],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [16, 22]
    } : tensor<1x16x48x64xf16> -> tensor<1x16x3x3xf16>

    return %avg_pool : tensor<1x16x3x3xf16>
    // CHECK:           [[AVGPOOL1:%.+]] = IE.AvgPool([[ARG_0]]) {exclude_pads, kernel_size = [8, 2], pads_begin = [0, 1], pads_end = [0, 1],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 2]} : tensor<1x16x48x64xf16> -> tensor<1x16x6x33xf16>
    // CHECK:           [[AVGPOOL2:%.+]] = IE.AvgPool([[AVGPOOL1]]) {exclude_pads, kernel_size = [2, 11], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 11]} : tensor<1x16x6x33xf16> -> tensor<1x16x3x3xf16>

    // CHECK:           return [[AVGPOOL2]] : tensor<1x16x3x3xf16>
}


// -----

// CHECK-LABEL: @AvgPoolWithPaddingAndKernelPadding
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x16x21xf16>
func.func @AvgPoolWithPaddingAndKernelPadding(%arg0 : tensor<1x16x16x21xf16>) -> (tensor<1x16x1x1xf16>) {
    %avg_pool = IE.AvgPool(%arg0) {
        kernel_size = [16, 23],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [16, 23]
    } : tensor<1x16x16x21xf16> -> tensor<1x16x1x1xf16>

    return %avg_pool : tensor<1x16x1x1xf16>
    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<16x1x8x8xf16> = dense<1.631160e-02> : tensor<16x1x8x8xf16>
    // CHECK:           [[GCONV:%.+]] = IE.GroupConvolution([[ARG_0]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 1], pads_end = [0, 2],
    // CHECK-SAME:              strides = [8, 8]} : tensor<1x16x16x21xf16>, tensor<16x1x8x8xf16> -> tensor<1x16x2x3xf16>
    // CHECK:           [[AVGPOOL:%.+]] = IE.AvgPool([[GCONV]]) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x2x3xf16> -> tensor<1x16x1x1xf16>

    // CHECK:           return [[AVGPOOL]] : tensor<1x16x1x1xf16>
}


// -----

// CHECK-LABEL: @AvgPoolWithExcludePaddingAndKernelPadding
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x16x21xf16>
func.func @AvgPoolWithExcludePaddingAndKernelPadding(%arg0 : tensor<1x16x16x21xf16>) -> (tensor<1x16x1x1xf16>) {
    %avg_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [16, 23],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [16, 23]
    } : tensor<1x16x16x21xf16> -> tensor<1x16x1x1xf16>

    return %avg_pool : tensor<1x16x1x1xf16>
    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<16x1x8x8xf16> = dense<1.785280e-02> : tensor<16x1x8x8xf16>
    // CHECK:           [[GCONV:%.+]] = IE.GroupConvolution([[ARG_0]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 1], pads_end = [0, 2],
    // CHECK-SAME:              strides = [8, 8]} : tensor<1x16x16x21xf16>, tensor<16x1x8x8xf16> -> tensor<1x16x2x3xf16>
    // CHECK:           [[AVGPOOL:%.+]] = IE.AvgPool([[GCONV]]) {exclude_pads, kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x2x3xf16> -> tensor<1x16x1x1xf16>

    // CHECK:           return [[AVGPOOL]] : tensor<1x16x1x1xf16>
}


// -----

// CHECK-LABEL: @MaxPoolWithPaddingAndKernelPadding
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x16x21xf16>
func.func @MaxPoolWithPaddingAndKernelPadding(%arg0 : tensor<1x16x16x21xf16>) -> (tensor<1x16x1x1xf16>) {
    %avg_pool = IE.MaxPool(%arg0) {
        kernel_size = [16, 23],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [16, 23]
    } : tensor<1x16x16x21xf16> -> tensor<1x16x1x1xf16>

    return %avg_pool : tensor<1x16x1x1xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 16, 16, 1] : tensor<1x16x16x21xf16> to tensor<1x16x16x1xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[ARG_0]], [[SLICE]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x16x16x21xf16>, tensor<1x16x16x1xf16> -> tensor<1x16x16x22xf16>
    // CHECK:       [[MAXPOOL1:%.+]] = IE.MaxPool([[CONCAT]]) {kernel_size = [8, 8], pads_begin = [0, 1], pads_end = [0, 1],
    // CHECK-SAME:           rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 8]} : tensor<1x16x16x22xf16> -> tensor<1x16x2x3xf16>
    // CHECK:       [[MAXPOOL2:%.+]] = IE.MaxPool([[MAXPOOL1]]) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x2x3xf16> -> tensor<1x16x1x1xf16>

    // CHECK:           return [[MAXPOOL2]] : tensor<1x16x1x1xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @HandleLargeKernelsConvWithOneDimOnHAndStrideOne
// CHECK-SAME:  [[INPUT:%.+]]: tensor<2x1x38623x1xf16>
func.func @HandleLargeKernelsConvWithOneDimOnHAndStrideOne(%arg0 : tensor<2x1x38623x1xf16>) -> tensor<2x2x38571x1xf16> {
    %cst = const.Declare tensor<2x1x53x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<2x1x38623x1xf16>, tensor<2x1x53x1xf16> -> tensor<2x2x38571x1xf16>
    return %conv : tensor<2x2x38571x1xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2x1x60x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>, [#const.PadWithZero<[0, 0, 5, 0], [0, 0, 2, 0]>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<2x1x60x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>, [#const.PadWithZero<[0, 0, 4, 0], [0, 0, 3, 0]>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<2x1x60x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>, [#const.PadWithZero<[0, 0, 3, 0], [0, 0, 4, 0]>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<2x1x60x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>, [#const.PadWithZero<[0, 0, 2, 0], [0, 0, 5, 0]>]
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<2x1x60x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>, [#const.PadWithZero<[0, 0, 1, 0], [0, 0, 6, 0]>]
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<2x1x60x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 7, 0]>]

    // CHECK: [[EXPAND_0:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 11, 0]} : tensor<2x1x38623x1xf16> -> tensor<2x1x38634x1xf16>
    // CHECK: [[RESHAPE_0:%.+]] = IE.Reshape([[EXPAND_0]]) {shape_value = [2, 6439, 6, 1]} : tensor<2x1x38634x1xf16> -> tensor<2x6439x6x1xf16>
    // CHECK: [[TRANSPOSE_0:%.+]] = IE.Transpose([[RESHAPE_0]]) {order_value = #NHCW} : tensor<2x6439x6x1xf16> -> tensor<2x6x6439x1xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST_4]], [[CST_3]], [[CST_2]], [[CST_1]], [[CST_0]], [[CST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<2x1x60x1xf16>, tensor<2x1x60x1xf16>, tensor<2x1x60x1xf16>, tensor<2x1x60x1xf16>, tensor<2x1x60x1xf16>, tensor<2x1x60x1xf16> -> tensor<12x1x60x1xf16>
    // CHECK: [[RESHAPE_1:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [12, 10, 6, 1]} : tensor<12x1x60x1xf16> -> tensor<12x10x6x1xf16>
    // CHECK: [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_1]]) {order_value = #NHCW} : tensor<12x10x6x1xf16> -> tensor<12x6x10x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_0]], [[TRANSPOSE_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<2x6x6439x1xf16>, tensor<12x6x10x1xf16> -> tensor<2x12x6430x1xf16>
    // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[CONV]]) {shape_value = [2, 6, 2, 6430]} : tensor<2x12x6430x1xf16> -> tensor<2x6x2x6430xf16>
    // CHECK: [[TRANSPOSE_2:%.+]] = IE.Transpose([[RESHAPE_2]]) {order_value = #NHWC} : tensor<2x6x2x6430xf16> -> tensor<2x2x6430x6xf16>
    // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[TRANSPOSE_2]]) {shape_value = [2, 2, 38580, 1]} : tensor<2x2x6430x6xf16> -> tensor<2x2x38580x1xf16>
    // CHECK: [[SLICE_0:%.+]] = IE.Slice [[RESHAPE_3]] [0, 0, 0, 0] [2, 2, 38571, 1] : tensor<2x2x38580x1xf16> to tensor<2x2x38571x1xf16>

    // CHECK: return [[SLICE_0]] : tensor<2x2x38571x1xf16>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @HandleLargeKernelsConvWithOneDimOnWAndStrideOne
// CHECK-SAME:  [[INPUT:%.+]]: tensor<2x1x1x38623xf16>
func.func @HandleLargeKernelsConvWithOneDimOnWAndStrideOne(%arg0 : tensor<2x1x1x38623xf16>) -> tensor<2x2x1x38571xf16> {
    %cst = const.Declare tensor<2x1x1x53xf16> = dense<1.000000e+00> : tensor<2x1x1x53xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<2x1x1x38623xf16>, tensor<2x1x1x53xf16> -> tensor<2x2x1x38571xf16>
    return %conv : tensor<2x2x1x38571xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2x1x1x60xf16> = dense<1.000000e+00> : tensor<2x1x1x53xf16>, [#const.PadWithZero<[0, 0, 0, 5], [0, 0, 0, 2]>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<2x1x1x60xf16> = dense<1.000000e+00> : tensor<2x1x1x53xf16>, [#const.PadWithZero<[0, 0, 0, 4], [0, 0, 0, 3]>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<2x1x1x60xf16> = dense<1.000000e+00> : tensor<2x1x1x53xf16>, [#const.PadWithZero<[0, 0, 0, 3], [0, 0, 0, 4]>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<2x1x1x60xf16> = dense<1.000000e+00> : tensor<2x1x1x53xf16>, [#const.PadWithZero<[0, 0, 0, 2], [0, 0, 0, 5]>]
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<2x1x1x60xf16> = dense<1.000000e+00> : tensor<2x1x1x53xf16>, [#const.PadWithZero<[0, 0, 0, 1], [0, 0, 0, 6]>]
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<2x1x1x60xf16> = dense<1.000000e+00> : tensor<2x1x1x53xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 7]>]

    // CHECK: [[EXPAND_0:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 11]} : tensor<2x1x1x38623xf16> -> tensor<2x1x1x38634xf16>
    // CHECK: [[RESHAPE_0:%.+]] = IE.Reshape([[EXPAND_0]]) {shape_value = [2, 6439, 1, 6]} : tensor<2x1x1x38634xf16> -> tensor<2x6439x1x6xf16>
    // CHECK: [[TRANSPOSE_0:%.+]] = IE.Transpose([[RESHAPE_0]]) {order_value = #NWHC} : tensor<2x6439x1x6xf16> -> tensor<2x6x1x6439xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST_4]], [[CST_3]], [[CST_2]], [[CST_1]], [[CST_0]], [[CST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<2x1x1x60xf16>, tensor<2x1x1x60xf16>, tensor<2x1x1x60xf16>, tensor<2x1x1x60xf16>, tensor<2x1x1x60xf16>, tensor<2x1x1x60xf16> -> tensor<12x1x1x60xf16>
    // CHECK: [[RESHAPE_1:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [12, 10, 1, 6]} : tensor<12x1x1x60xf16> -> tensor<12x10x1x6xf16>
    // CHECK: [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_1]]) {order_value = #NWHC} : tensor<12x10x1x6xf16> -> tensor<12x6x1x10xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_0]], [[TRANSPOSE_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<2x6x1x6439xf16>, tensor<12x6x1x10xf16> -> tensor<2x12x1x6430xf16>
    // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[CONV]]) {shape_value = [2, 6, 2, 6430]} : tensor<2x12x1x6430xf16> -> tensor<2x6x2x6430xf16>
    // CHECK: [[TRANSPOSE_2:%.+]] = IE.Transpose([[RESHAPE_2]]) {order_value = #NHWC} : tensor<2x6x2x6430xf16> -> tensor<2x2x6430x6xf16>
    // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[TRANSPOSE_2]]) {shape_value = [2, 2, 1, 38580]} : tensor<2x2x6430x6xf16> -> tensor<2x2x1x38580xf16>
    // CHECK: [[SLICE_0:%.+]] = IE.Slice [[RESHAPE_3]] [0, 0, 0, 0] [2, 2, 1, 38571] : tensor<2x2x1x38580xf16> to tensor<2x2x1x38571xf16>

    // CHECK: return [[SLICE_0]] : tensor<2x2x1x38571xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @HandleLargeKernelsConvHasBiasWithOneDimOnHAndStrideOne
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x38623x1xf16>
func.func @HandleLargeKernelsConvHasBiasWithOneDimOnHAndStrideOne(%arg0 : tensor<1x1x38623x1xf16>) -> tensor<1x2x38571x1xf16> {
    %cst = const.Declare tensor<2x1x53x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>
    %cst_0 = const.Declare tensor<1x2x1x1xf16> = dense<1.000000e+00> : tensor<1x2x1x1xf16>

    %conv = IE.Convolution(%arg0, %cst, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x38623x1xf16>, tensor<2x1x53x1xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x38571x1xf16>
    return %conv : tensor<1x2x38571x1xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2x1x60x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>, [#const.PadWithZero<[0, 0, 5, 0], [0, 0, 2, 0]>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<2x1x60x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>, [#const.PadWithZero<[0, 0, 4, 0], [0, 0, 3, 0]>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<2x1x60x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>, [#const.PadWithZero<[0, 0, 3, 0], [0, 0, 4, 0]>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<2x1x60x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>, [#const.PadWithZero<[0, 0, 2, 0], [0, 0, 5, 0]>]
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<2x1x60x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>, [#const.PadWithZero<[0, 0, 1, 0], [0, 0, 6, 0]>]
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<2x1x60x1xf16> = dense<1.000000e+00> : tensor<2x1x53x1xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 7, 0]>]
    // CHECK-DAG: [[CST_5:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<1.000000e+00> : tensor<1x2x1x1xf16>

    // CHECK: [[EXPAND_0:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 11, 0]} : tensor<1x1x38623x1xf16> -> tensor<1x1x38634x1xf16>
    // CHECK: [[RESHAPE_0:%.+]] = IE.Reshape([[EXPAND_0]]) {shape_value = [1, 6439, 6, 1]} : tensor<1x1x38634x1xf16> -> tensor<1x6439x6x1xf16>
    // CHECK: [[TRANSPOSE_0:%.+]] = IE.Transpose([[RESHAPE_0]]) {order_value = #NHCW} : tensor<1x6439x6x1xf16> -> tensor<1x6x6439x1xf16>
    // CHECK: [[CONCAT_0:%.+]] = IE.Concat([[CST_4]], [[CST_3]], [[CST_2]], [[CST_1]], [[CST_0]], [[CST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<2x1x60x1xf16>, tensor<2x1x60x1xf16>, tensor<2x1x60x1xf16>, tensor<2x1x60x1xf16>, tensor<2x1x60x1xf16>, tensor<2x1x60x1xf16> -> tensor<12x1x60x1xf16>
    // CHECK: [[RESHAPE_1:%.+]] = IE.Reshape([[CONCAT_0]]) {shape_value = [12, 10, 6, 1]} : tensor<12x1x60x1xf16> -> tensor<12x10x6x1xf16>
    // CHECK: [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_1]]) {order_value = #NHCW} : tensor<12x10x6x1xf16> -> tensor<12x6x10x1xf16>
    // CHECK: [[CONCAT_1:%.+]] = IE.Concat([[CST_5]], [[CST_5]], [[CST_5]], [[CST_5]], [[CST_5]], [[CST_5]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16>, tensor<1x2x1x1xf16> -> tensor<1x12x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_0]], [[TRANSPOSE_1]], [[CONCAT_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x6x6439x1xf16>, tensor<12x6x10x1xf16>, tensor<1x12x1x1xf16> -> tensor<1x12x6430x1xf16>
    // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[CONV]]) {shape_value = [1, 6, 2, 6430]} : tensor<1x12x6430x1xf16> -> tensor<1x6x2x6430xf16>
    // CHECK: [[TRANSPOSE_2:%.+]] = IE.Transpose([[RESHAPE_2]]) {order_value = #NHWC} : tensor<1x6x2x6430xf16> -> tensor<1x2x6430x6xf16>
    // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[TRANSPOSE_2]]) {shape_value = [1, 2, 38580, 1]} : tensor<1x2x6430x6xf16> -> tensor<1x2x38580x1xf16>
    // CHECK: [[SLICE_0:%.+]] = IE.Slice [[RESHAPE_3]] [0, 0, 0, 0] [1, 2, 38571, 1] : tensor<1x2x38580x1xf16> to tensor<1x2x38571x1xf16>

    // CHECK: return [[SLICE_0]] : tensor<1x2x38571x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsXAvgPool
func.func @HandleLargeKernelsXAvgPool(%arg0 : tensor<1x64x10x17xf16>) -> (tensor<1x64x10x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 17],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 17]
    } : tensor<1x64x10x17xf16> -> tensor<1x64x10x1xf16>

    return %ave_pool : tensor<1x64x10x1xf16>

    // CHECK:       const.Declare
    // CHECK-SAME:      tensor<64x1x1x6xf16> = dense<1.765140e-01>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 64 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 6]
    // CHECK-SAME:      : tensor<1x64x10x17xf16>, tensor<64x1x1x6xf16> -> tensor<1x64x10x3xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 3]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x64x10x3xf16> -> tensor<1x64x10x1xf16>

}

// -----

// CHECK-LABEL: @HandleLargeKernelsYAvgPool
func.func @HandleLargeKernelsYAvgPool(%arg0 : tensor<1x64x17x10xf16>) -> (tensor<1x64x1x10xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [17, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [17, 1]
    } : tensor<1x64x17x10xf16> -> tensor<1x64x1x10xf16>

    return %ave_pool : tensor<1x64x1x10xf16>

    // CHECK:       const.Declare
    // CHECK-SAME:      tensor<64x1x6x1xf16> = dense<1.765140e-01>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 64 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 0]
    // CHECK-SAME:      strides = [6, 1]
    // CHECK-SAME:      : tensor<1x64x17x10xf16>, tensor<64x1x6x1xf16> -> tensor<1x64x3x10xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [3, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x64x3x10xf16> -> tensor<1x64x1x10xf16>

}

// -----

// CHECK-LABEL: @HandleLargerKernelsAvgPoolPaddingNeededOneDim
func.func @HandleLargerKernelsAvgPoolPaddingNeededOneDim(%arg0 : tensor<1x128x1x75076xf16>) -> (tensor<1x128x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 75076],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x128x1x75076xf16> -> tensor<1x128x1x1xf16>

    return %ave_pool : tensor<1x128x1x1xf16>

    // CHECK-DAG:   const.Declare tensor<128x1x1x5xf16> = dense<1.999510e-01> : tensor<128x1x1x5xf16>
    // CHECK-DAG:   const.Declare tensor<128x1x1x6xf16> = dense<1.667480e-01> : tensor<128x1x1x6xf16>
    // CHECK-DAG:   const.Declare tensor<128x1x1x2xf16> = dense<5.014650e-01> : tensor<128x1x1x2xf16>
    // CHECK-DAG:   const.Declare tensor<128x1x1x2xf16> = dense<5.034180e-01> : tensor<128x1x1x2xf16>
    // CHECK-DAG:   const.Declare tensor<128x1x1x8xf16> = dense<1.265870e-01> : tensor<128x1x1x8xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 4]
    // CHECK-SAME:      : tensor<1x128x1x75076xf16> -> tensor<1x128x1x18769xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 5]
    // CHECK-SAME:      : tensor<1x128x1x18769xf16>, tensor<128x1x1x5xf16> -> tensor<1x128x1x3754xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 2]
    // CHECK-SAME:      : tensor<1x128x1x3754xf16> -> tensor<1x128x1x1877xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 6]
    // CHECK-SAME:      : tensor<1x128x1x1877xf16>, tensor<128x1x1x6xf16> -> tensor<1x128x1x313xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 2]
    // CHECK-SAME:      : tensor<1x128x1x313xf16>, tensor<128x1x1x2xf16> -> tensor<1x128x1x157xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 2]
    // CHECK-SAME:      : tensor<1x128x1x157xf16>, tensor<128x1x1x2xf16> -> tensor<1x128x1x79xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 8]
    // CHECK-SAME:      : tensor<1x128x1x79xf16>, tensor<128x1x1x8xf16> -> tensor<1x128x1x10xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 10]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x128x1x10xf16> -> tensor<1x128x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargerKernelsAvgPoolWithStaticScalePaddingNeededOneDim
func.func @HandleLargerKernelsAvgPoolWithStaticScalePaddingNeededOneDim(%arg0 : tensor<1x128x1x75076xf16>) -> (tensor<1x128x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 75076],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        static_scale = 0.135327876 : f32,
        strides = [1, 1]
    } : tensor<1x128x1x75076xf16> -> tensor<1x128x1x1xf16>

    return %ave_pool : tensor<1x128x1x1xf16>

    // CHECK-DAG:   const.Declare tensor<128x1x1x5xf16> = dense<1.999510e-01> : tensor<128x1x1x5xf16>
    // CHECK-DAG:   const.Declare tensor<128x1x1x6xf16> = dense<1.667480e-01> : tensor<128x1x1x6xf16>
    // CHECK-DAG:   const.Declare tensor<128x1x1x2xf16> = dense<5.014650e-01> : tensor<128x1x1x2xf16>
    // CHECK-DAG:   const.Declare tensor<128x1x1x2xf16> = dense<5.034180e-01> : tensor<128x1x1x2xf16>
    // CHECK-DAG:   const.Declare tensor<128x1x1x8xf16> = dense<1.265870e-01> : tensor<128x1x1x8xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-NOT:       static_scale
    // CHECK-SAME:      strides = [1, 4]
    // CHECK-SAME:      : tensor<1x128x1x75076xf16> -> tensor<1x128x1x18769xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 5]
    // CHECK-SAME:      : tensor<1x128x1x18769xf16>, tensor<128x1x1x5xf16> -> tensor<1x128x1x3754xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-NOT:       static_scale
    // CHECK-SAME:      strides = [1, 2]
    // CHECK-SAME:      : tensor<1x128x1x3754xf16> -> tensor<1x128x1x1877xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 6]
    // CHECK-SAME:      : tensor<1x128x1x1877xf16>, tensor<128x1x1x6xf16> -> tensor<1x128x1x313xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 2]
    // CHECK-SAME:      : tensor<1x128x1x313xf16>, tensor<128x1x1x2xf16> -> tensor<1x128x1x157xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 2]
    // CHECK-SAME:      : tensor<1x128x1x157xf16>, tensor<128x1x1x2xf16> -> tensor<1x128x1x79xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 8]
    // CHECK-SAME:      : tensor<1x128x1x79xf16>, tensor<128x1x1x8xf16> -> tensor<1x128x1x10xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 10]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      static_scale = 0.135327876 : f32,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x128x1x10xf16> -> tensor<1x128x1x1xf16>
}

// -----

// Kernel size equals stride size and there is no padding, so every output position reads a
// disjoint 16x16 patch: the convolution folds into SpaceToDepth plus a 1x1 convolution.

// CHECK-LABEL: @SpaceToDepthLargeConv
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x3x224x224xf16>
// CHECK-SAME:  [[FILTER:%.+]]: tensor<768x3x16x16xf16>
// CHECK-SAME:  [[BIAS:%.+]]: tensor<1x768x1x1xf16>
func.func @SpaceToDepthLargeConv(%arg0: tensor<1x3x224x224xf16>, %arg1: tensor<768x3x16x16xf16>, %arg2: tensor<1x768x1x1xf16>) -> tensor<1x768x14x14xf16> {
    %conv = IE.Convolution(%arg0, %arg1, %arg2) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [16, 16]
    } : tensor<1x3x224x224xf16>, tensor<768x3x16x16xf16>, tensor<1x768x1x1xf16> -> tensor<1x768x14x14xf16>

    return %conv : tensor<1x768x14x14xf16>

    // CHECK-NOT:   IE.Slice
    // CHECK-NOT:   IE.Add
    // CHECK:   [[SPACE_TO_DEPTH:%.+]] = IE.SpaceToDepthOp([[INPUT]]) {block_size = 16 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x3x224x224xf16> -> tensor<1x768x14x14xf16>
    // CHECK:   [[RESHAPE_FILTER:%.+]] = IE.Reshape([[FILTER]]) {shape_value = [768, 768, 1, 1]} : tensor<768x3x16x16xf16> -> tensor<768x768x1x1xf16>
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[SPACE_TO_DEPTH]], [[RESHAPE_FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x768x14x14xf16>, tensor<768x768x1x1xf16>, tensor<1x768x1x1xf16> -> tensor<1x768x14x14xf16>
    // CHECK:   return [[CONV]] : tensor<1x768x14x14xf16>
}

// -----

// IC*KY*KX (3*17*17 = 867) is not aligned to the channel alignment requirement, so a later pass
// (ExpandActivationChannels) will need to pad it up. The fold still applies here: profiling showed
// this SpaceToDepth-based form outperforms the Slice/Add fallback even with that extra padding cost.

// CHECK-LABEL: @SpaceToDepthLargeConvUnalignedChannels
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x3x272x272xf16>
// CHECK-SAME:  [[FILTER:%.+]]: tensor<1024x3x17x17xf16>
func.func @SpaceToDepthLargeConvUnalignedChannels(%arg0: tensor<1x3x272x272xf16>, %arg1: tensor<1024x3x17x17xf16>) -> tensor<1x1024x16x16xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [17, 17]
    } : tensor<1x3x272x272xf16>, tensor<1024x3x17x17xf16> -> tensor<1x1024x16x16xf16>

    return %conv : tensor<1x1024x16x16xf16>

    // CHECK-NOT:   IE.Slice
    // CHECK-NOT:   IE.Add
    // CHECK:   [[SPACE_TO_DEPTH:%.+]] = IE.SpaceToDepthOp([[INPUT]]) {block_size = 17 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x3x272x272xf16> -> tensor<1x867x16x16xf16>
    // CHECK:   [[RESHAPE_FILTER:%.+]] = IE.Reshape([[FILTER]]) {shape_value = [1024, 867, 1, 1]} : tensor<1024x3x17x17xf16> -> tensor<1024x867x1x1xf16>
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[SPACE_TO_DEPTH]], [[RESHAPE_FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x867x16x16xf16>, tensor<1024x867x1x1xf16> -> tensor<1x1024x16x16xf16>
    // CHECK:   return [[CONV]] : tensor<1x1024x16x16xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607:128>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:0, {0.001:0, 0.002:0, 0.003:0, 0.004:0}>

// The filter reshape keeps OC as the outermost dimension, so a per-output-channel quantized
// filter keeps its scales aligned.

// CHECK-LABEL: @SpaceToDepthLargeConvPerAxisQuantFilter
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x3x224x224x!qElemType>
// CHECK-SAME:  [[FILTER:%.+]]: tensor<4x3x16x16x!qElemType1>
func.func @SpaceToDepthLargeConvPerAxisQuantFilter(%arg0: tensor<1x3x224x224x!qElemType>, %arg1: tensor<4x3x16x16x!qElemType1>) -> tensor<1x4x14x14x!qElemType> {
    %conv = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [16, 16]
    } : tensor<1x3x224x224x!qElemType>, tensor<4x3x16x16x!qElemType1> -> tensor<1x4x14x14x!qElemType>

    return %conv : tensor<1x4x14x14x!qElemType>

    // CHECK:   [[SPACE_TO_DEPTH:%.+]] = IE.SpaceToDepthOp([[INPUT]]) {block_size = 16 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x3x224x224x!qElemType> -> tensor<1x768x14x14x!qElemType>
    // CHECK:   [[RESHAPE_FILTER:%.+]] = IE.Reshape([[FILTER]]) {shape_value = [4, 768, 1, 1]} : tensor<4x3x16x16x!qElemType1> -> tensor<4x768x1x1x!qElemType1>
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[SPACE_TO_DEPTH]], [[RESHAPE_FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x768x14x14x!qElemType>, tensor<4x768x1x1x!qElemType1> -> tensor<1x4x14x14x!qElemType>
    // CHECK:   return [[CONV]] : tensor<1x4x14x14x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607:128>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:1, {0.001:0, 0.002:0, 0.003:0}>

// Mixed precision: quantized input with FP16 output. The new 1x1 convolution must keep the FP16 output type.

// CHECK-LABEL: @SpaceToDepthLargeConvMixedPrecision
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x3x224x224x!qElemType>
// CHECK-SAME:  [[FILTER:%.+]]: tensor<4x3x16x16xf16>
func.func @SpaceToDepthLargeConvMixedPrecision(%arg0: tensor<1x3x224x224x!qElemType>, %arg1: tensor<4x3x16x16xf16>) -> tensor<1x4x14x14xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [16, 16]
    } : tensor<1x3x224x224x!qElemType>, tensor<4x3x16x16xf16> -> tensor<1x4x14x14xf16>

    return %conv : tensor<1x4x14x14xf16>

    // CHECK:   [[SPACE_TO_DEPTH:%.+]] = IE.SpaceToDepthOp([[INPUT]]) {block_size = 16 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x3x224x224x!qElemType> -> tensor<1x768x14x14x!qElemType>
    // CHECK:   [[RESHAPE_FILTER:%.+]] = IE.Reshape([[FILTER]]) {shape_value = [4, 768, 1, 1]} : tensor<4x3x16x16xf16> -> tensor<4x768x1x1xf16>
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[SPACE_TO_DEPTH]], [[RESHAPE_FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x768x14x14x!qElemType>, tensor<4x768x1x1xf16> -> tensor<1x4x14x14xf16>
    // CHECK:   return [[CONV]] : tensor<1x4x14x14xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607:128>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:1, {0.001:0, 0.002:0, 0.003:0}>

// Quantization over IC cannot be preserved when the filter is reshaped to merge IC and KY/KX.

// CHECK-LABEL: @SpaceToDepthLargeConvNonOutputAxisQuantFilter
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x3x224x224x!qElemType>
// CHECK-SAME:  [[FILTER:%.+]]: tensor<4x3x16x16x!qElemType1>
func.func @SpaceToDepthLargeConvNonOutputAxisQuantFilter(%arg0: tensor<1x3x224x224x!qElemType>, %arg1: tensor<4x3x16x16x!qElemType1>) -> tensor<1x4x14x14x!qElemType> {
    %conv = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [16, 16]
    } : tensor<1x3x224x224x!qElemType>, tensor<4x3x16x16x!qElemType1> -> tensor<1x4x14x14x!qElemType>

    return %conv : tensor<1x4x14x14x!qElemType>

    // CHECK-NOT:   IE.SpaceToDepthOp
}

// -----

#OYXI = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// DEPTH_FIRST consumes flattened filter data in OIYX order. A filter with another physical order
// needs a reorder before flattening, so this rewrite must not apply.

// CHECK-LABEL: @SpaceToDepthLargeConvNonOIYXFilter
func.func @SpaceToDepthLargeConvNonOIYXFilter(%arg0: tensor<1x3x224x224xf16>, %arg1: tensor<16x3x16x16xf16, {order = #OYXI}>) -> tensor<1x16x14x14xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [16, 16]
    } : tensor<1x3x224x224xf16>, tensor<16x3x16x16xf16, {order = #OYXI}> -> tensor<1x16x14x14xf16>

    return %conv : tensor<1x16x14x14xf16>

    // CHECK-NOT: IE.SpaceToDepthOp
}

// -----

// Stride is smaller than the kernel, so the patches overlap and the fold does not apply:
// the kernel is still split into slices recombined by Adds.

// CHECK-LABEL: @SpaceToDepthLargeConvStrideNotEqualKernel
func.func @SpaceToDepthLargeConvStrideNotEqualKernel(%arg0: tensor<1x3x40x40xf16>, %arg1: tensor<16x3x16x16xf16>) -> tensor<1x16x4x4xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [8, 8]
    } : tensor<1x3x40x40xf16>, tensor<16x3x16x16xf16> -> tensor<1x16x4x4xf16>

    return %conv : tensor<1x16x4x4xf16>

    // CHECK-NOT:   IE.SpaceToDepthOp
    // CHECK:       IE.Add
}

// -----

// The input is not an exact multiple of the kernel: the convolution ignores the partial
// trailing patch but SpaceToDepth would keep it, so the fold does not apply.

// CHECK-LABEL: @SpaceToDepthLargeConvInputNotDivisible
func.func @SpaceToDepthLargeConvInputNotDivisible(%arg0: tensor<1x3x230x230xf16>, %arg1: tensor<16x3x16x16xf16>) -> tensor<1x16x14x14xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [16, 16]
    } : tensor<1x3x230x230xf16>, tensor<16x3x16x16xf16> -> tensor<1x16x14x14xf16>

    return %conv : tensor<1x16x14x14xf16>

    // CHECK-NOT:   IE.SpaceToDepthOp
    // CHECK:       IE.Add
}

// -----

// SpaceToDepth uses a single block size for both spatial dimensions, so a non-square
// kernel cannot be folded even though it equals the stride.

// CHECK-LABEL: @SpaceToDepthLargeConvNonSquareKernel
func.func @SpaceToDepthLargeConvNonSquareKernel(%arg0: tensor<1x3x32x64xf16>, %arg1: tensor<16x3x16x32xf16>) -> tensor<1x16x2x2xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [16, 32]
    } : tensor<1x3x32x64xf16>, tensor<16x3x16x32xf16> -> tensor<1x16x2x2xf16>

    return %conv : tensor<1x16x2x2xf16>

    // CHECK-NOT:   IE.SpaceToDepthOp
}

// -----

// Padding makes the patches overlap the input border, so the fold does not apply.

// CHECK-LABEL: @SpaceToDepthLargeConvWithPadding
func.func @SpaceToDepthLargeConvWithPadding(%arg0: tensor<1x3x224x224xf16>, %arg1: tensor<16x3x16x16xf16>) -> tensor<1x16x15x15xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [8, 8],
        pads_end = [8, 8],
        strides = [16, 16]
    } : tensor<1x3x224x224xf16>, tensor<16x3x16x16xf16> -> tensor<1x16x15x15xf16>

    return %conv : tensor<1x16x15x15xf16>

    // CHECK-NOT:   IE.SpaceToDepthOp
}

// -----

// The kernel fits the DPU limit. Activation-channel expansion handles the low IC without introducing
// an activation-sized SpaceToDepth DMA.

// CHECK-LABEL: @SpaceToDepthSmallKernelUnalignedChannels
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x3x32x32xf16>
// CHECK-SAME:  [[FILTER:%.+]]: tensor<16x3x8x8xf16>
func.func @SpaceToDepthSmallKernelUnalignedChannels(%arg0: tensor<1x3x32x32xf16>, %arg1: tensor<16x3x8x8xf16>) -> tensor<1x16x4x4xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [8, 8]
    } : tensor<1x3x32x32xf16>, tensor<16x3x8x8xf16> -> tensor<1x16x4x4xf16>

    return %conv : tensor<1x16x4x4xf16>

    // CHECK-NOT:   IE.SpaceToDepthOp
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:      strides = [8, 8]
    // CHECK:   return [[CONV]] : tensor<1x16x4x4xf16>
}

// -----

// The kernel fits the DPU limit and IC is already aligned, so the convolution runs at full channel
// utilization and the gather would only add cost.

// CHECK-LABEL: @SpaceToDepthSmallKernelAlignedChannels
func.func @SpaceToDepthSmallKernelAlignedChannels(%arg0: tensor<1x32x32x32xf16>, %arg1: tensor<16x32x8x8xf16>) -> tensor<1x16x4x4xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [8, 8]
    } : tensor<1x32x32x32xf16>, tensor<16x32x8x8xf16> -> tensor<1x16x4x4xf16>

    return %conv : tensor<1x16x4x4xf16>

    // CHECK-NOT:   IE.SpaceToDepthOp
    // CHECK:       [[CONV:%.+]] = IE.Convolution
    // CHECK-SAME:      strides = [8, 8]
    // CHECK:       return [[CONV]] : tensor<1x16x4x4xf16>
}

// -----

// A 2x2 patch over IC=3 only reaches 12 channels, still below the alignment, so the gather would
// not buy back any DPU throughput.

// CHECK-LABEL: @SpaceToDepthSmallKernelStillUnaligned
func.func @SpaceToDepthSmallKernelStillUnaligned(%arg0: tensor<1x3x32x32xf16>, %arg1: tensor<16x3x2x2xf16>) -> tensor<1x16x16x16xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x3x32x32xf16>, tensor<16x3x2x2xf16> -> tensor<1x16x16x16xf16>

    return %conv : tensor<1x16x16x16xf16>

    // CHECK-NOT:   IE.SpaceToDepthOp
    // CHECK:       [[CONV:%.+]] = IE.Convolution
    // CHECK-SAME:      strides = [2, 2]
    // CHECK:       return [[CONV]] : tensor<1x16x16x16xf16>
}
