//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --fuse-activation-ops %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --run-adjust-for-vpu-rewriters="rewriter=fuse-activation-ops-set" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @TransposedConv2dWithLeakyReluTest
func.func @TransposedConv2dWithLeakyReluTest(%arg0: tensor<1x32x64x100xf16>) -> tensor<1x16x128x101xf16> {
    %filters = const.Declare tensor<16x32x3x2xf16> = dense<1.0> : tensor<16x32x3x2xf16>
    %0 = IE.TransposedConvolution(%arg0, %filters)
        {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0>,
            spatial_output_padding = [1, 0],
            pads_begin = [1, 0],
            pads_end = [1, 0],
            strides = [2, 1]
        } :
        tensor<1x32x64x100xf16>, tensor<16x32x3x2xf16> -> tensor<1x16x128x101xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.500000e-01 : f64} : tensor<1x16x128x101xf16> -> tensor<1x16x128x101xf16>

    return %1 : tensor<1x16x128x101xf16>

    // CHECK:       IE.TransposedConvolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     operandSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:     pads_begin = [1, 0]
    // CHECK-SAME:     pads_end = [1, 0]
    // CHECK-SAME:     post_op = #IE.LeakyRelu<negative_slope = 1.500000e-01 : f64>
    // CHECK-SAME:     spatial_output_padding = [1, 0]
    // CHECK-SAME:     strides = [2, 1]
    // CHECK-NOT:   IE.LeakyRelu
    // CHECK: return
}

// -----
// CHECK-LABEL: @TransposedConv2dWithLeakyReluNotFuseTest
func.func @TransposedConv2dWithLeakyReluNotFuseTest(%arg0: tensor<1x32x64x100xf16>, %arg1: tensor<16x32x3x2xf16>) -> tensor<1x16x128x101xf16> {
    %0 = IE.TransposedConvolution(%arg0, %arg1)
        {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0>,
            spatial_output_padding = [1, 0],
            pads_begin = [1, 0],
            pads_end = [1, 0],
            strides = [2, 1]
        } :
        tensor<1x32x64x100xf16>, tensor<16x32x3x2xf16> -> tensor<1x16x128x101xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.500000e-01 : f64} : tensor<1x16x128x101xf16> -> tensor<1x16x128x101xf16>

    return %1 : tensor<1x16x128x101xf16>

    // CHECK:       IE.TransposedConvolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     operandSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:     pads_begin = [1, 0]
    // CHECK-SAME:     pads_end = [1, 0]
    // CHECK-SAME:     post_op = #IE.LeakyRelu<negative_slope = 1.500000e-01 : f64>
    // CHECK-SAME:     spatial_output_padding = [1, 0]
    // CHECK-SAME:     strides = [2, 1]
    // CHECK-NOT:   IE.LeakyRelu
    // CHECK: return
}
