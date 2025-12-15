//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --run-adjust-for-vpu-rewriters="rewriter=fuse-pad-ops-set"  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @fusePadIntoConv
func.func @fusePadIntoConv(%arg0: tensor<1x8x13x29xf16>) -> tensor<1x16x12x28xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]}
                        : tensor<1x8x13x29xf16> -> tensor<1x8x16x32xf16>
    %1 = const.Declare tensor<16x8x5x5xf16> = dense<1.0> : tensor<16x8x5x5xf16>
    %2 = IE.Convolution(%0, %1)
    {
        strides = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        dilations = [1, 1]
    } :
    tensor<1x8x16x32xf16>, tensor<16x8x5x5xf16> -> tensor<1x16x12x28xf16>
    return %2 : tensor<1x16x12x28xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK-DAG:      [[CST0:%.*]] = const.Declare tensor<16x8x5x5xf16> = dense<1.000000e+00> : tensor<16x8x5x5xf16>
    // CHECK:       [[VAR0:%.*]] = IE.Convolution(%arg0, [[CST0]])
    // CHECK-SAME:        {dilations = [1, 1], pads_begin = [1, 2], pads_end = [2, 1], strides = [1, 1]}
    // CHECK:       tensor<1x8x13x29xf16>, tensor<16x8x5x5xf16> -> tensor<1x16x12x28xf16>
    // CHECK:       return [[VAR0]] : tensor<1x16x12x28xf16>
}

// -----

// CHECK-LABEL: @fusePadIntoGroupConv
func.func @fusePadIntoGroupConv(%arg0: tensor<1x8x13x29xf16>) -> tensor<1x8x12x28xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]}
                        : tensor<1x8x13x29xf16> -> tensor<1x8x16x32xf16>
    %WEIGHT = const.Declare tensor<8x1x5x5xf16> = dense<1.0> : tensor<8x1x5x5xf16>
    %BIAS = const.Declare tensor<1x8x1x1xf16> = dense<0.0> : tensor<1x8x1x1xf16>
    %3 = IE.GroupConvolution(%0, %WEIGHT, %BIAS)
    {
        dilations = [1, 1],
        groups = 8 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } :
    tensor<1x8x16x32xf16>, tensor<8x1x5x5xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x12x28xf16>
    return %3 : tensor<1x8x12x28xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK-DAG:      [[CST0:%.*]] = const.Declare tensor<1x8x1x1xf16> = dense<0.000000e+00> : tensor<1x8x1x1xf16>
    // CHECK-DAG:      [[CST1:%.*]] = const.Declare tensor<8x1x5x5xf16> = dense<1.000000e+00> : tensor<8x1x5x5xf16>
    // CHECK:       [[VAR0:%.*]] = IE.GroupConvolution(%arg0, [[CST1]], [[CST0]])
    // CHECK-SAME:        {dilations = [1, 1], groups = 8 : i64, pads_begin = [1, 2], pads_end = [2, 1], strides = [1, 1]}
    // CHECK:       tensor<1x8x13x29xf16>, tensor<8x1x5x5xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x12x28xf16>
    // CHECK:       return [[VAR0]] : tensor<1x8x12x28xf16>
}

// -----

// CHECK-LABEL: @fusePadIntoMaxPool
func.func @fusePadIntoMaxPool(%arg0: tensor<1x8x13x29xf16>) -> tensor<1x8x6x14xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]}
                        : tensor<1x8x13x29xf16> -> tensor<1x8x16x32xf16>
    %1 = IE.MaxPool(%0) { kernel_size = [5, 5], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2] } : tensor<1x8x16x32xf16> -> tensor<1x8x6x14xf16>
    return %1 : tensor<1x8x6x14xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[VAR0:%.*]] = IE.MaxPool(%arg0)
    // CHECK-SAME:        {kernel_size = [5, 5], pads_begin = [1, 2], pads_end = [2, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]}
    // CHECK:       tensor<1x8x13x29xf16> -> tensor<1x8x6x14xf16>
    // CHECK:       return [[VAR0]] : tensor<1x8x6x14xf16>
}
