//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// An unbounded channel stays dynamic when divided by the number of groups.

// CHECK-LABEL: @UnboundedChannels
func.func @UnboundedChannels(%arg0: tensor<1x?x32x32xf16>) -> tensor<1x16x30x30xf16> {
    %filter = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.GroupConvolution(%arg0, %filter) {
        dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x?x32x32xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x30x30xf16>
    return %0 : tensor<1x16x30x30xf16>

    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      -> tensor<1x16x30x30xf16>
}

// -----

// CHECK-LABEL: @UnboundedSpatial
func.func @UnboundedSpatial(%arg0: tensor<1x16x?x32xf16>) -> tensor<1x16x?x30xf16> {
    %filter = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.GroupConvolution(%arg0, %filter) {
        dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x16x?x32xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x?x30xf16>
    return %0 : tensor<1x16x?x30xf16>

    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      -> tensor<1x16x?x30xf16>
}
