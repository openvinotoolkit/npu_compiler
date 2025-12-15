//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=512 --init-compiler="vpu-arch=%arch% enable-qdq-optimization-aggressive=true" --handle-u16-fake-quantize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ReplaceSingleFQU16WithReLU
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @ReplaceSingleFQU16WithReLU(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x19xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.FakeQuantize(%0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %conv = IE.Convolution(%1, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    return %conv : tensor<1x512x25x19xf32>

    // CHECK:   [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:   [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>
    // CHECK:   [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[FQ]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    // CHECK:   return [[CONV]] : tensor<1x512x25x19xf32>
}

// -----

// CHECK-LABEL: @ReplaceFQU16WithFQU8ForConv
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @ReplaceFQU16WithFQU8ForConv(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x19xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.Convolution(%0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    return %1 : tensor<1x512x25x19xf32>

    // CHECK:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:    [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>
    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[FQ]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    // CHECK:    return [[CONV]] : tensor<1x512x25x19xf32>
}

// -----

// CHECK-LABEL: @DoNotLowerBothFQForConv
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x4xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @DoNotLowerBothFQForConv(%arg0: tensor<1x512x51x4xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x1xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>
    %low_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %high_2 = const.Declare tensor<1x1x1x1xf32> = dense<13.7559032> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x4xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x4xf32>
    %1 = IE.FakeQuantize(%arg1, %low_2, %high_2, %low_2, %high_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<512x512x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<512x512x3x3xf32>
    %2 = IE.Convolution(%0, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x4xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x1xf32>
    return %2 : tensor<1x512x25x1xf32>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x4xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x1xf32>
    // CHECK: return [[CONV]] : tensor<1x512x25x1xf32>
}

// -----

// CHECK-LABEL: @DoNotLowerBothFQForMatMul
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x4xf32>, [[ARG1:%.+]]: tensor<512x512x4x3xf32>)
func.func @DoNotLowerBothFQForMatMul(%arg0: tensor<1x512x51x4xf32>, %arg1: tensor<512x512x4x3xf32>) -> tensor<512x512x51x3xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x4xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x4xf32>
    %1 = IE.FakeQuantize(%arg1, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<512x512x4x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<512x512x4x3xf32>
    %2 = IE.MatMul(%0, %1) : tensor<1x512x51x4xf32>, tensor<512x512x4x3xf32> -> tensor<512x512x51x3xf32>
    return %2 : tensor<512x512x51x3xf32>

    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) : tensor<1x512x51x4xf32>, tensor<512x512x4x3xf32> -> tensor<512x512x51x3xf32>
    // CHECK: return [[MATMUL]] : tensor<512x512x51x3xf32>
}

// -----

// CHECK-LABEL: @NoLoweringWhenFailedToSplat
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x4xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @NoLoweringWhenFailedToSplat(%arg0: tensor<1x512x51x4xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x1xf32> {
    %low = const.Declare tensor<4xf32> = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<4xf32>
    %high = const.Declare tensor<4xf32> = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<4xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32> -> tensor<1x512x51x4xf32>
    %1 = IE.Convolution(%0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x4xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x1xf32>
    return %1 : tensor<1x512x25x1xf32>

    // CHECK:    [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x4xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x1xf32>
    // CHECK:    return [[CONV]] : tensor<1x512x25x1xf32>
}

// -----

// CHECK-LABEL: @ReplaceFQU16WithFQU8ForGroupConv
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @ReplaceFQU16WithFQU8ForGroupConv(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x51x39xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.GroupConvolution(%0, %arg1) {dilations = [1, 1], groups = 1 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x51x39xf32>
    return %1 : tensor<1x512x51x39xf32>

    // CHECK:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:    [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>
    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[GROUPCONV:%.+]] = IE.GroupConvolution([[FQ]], [[ARG1]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    return [[GROUPCONV]] : tensor<1x512x51x39xf32>
}

// -----

// CHECK-LABEL: @ReplaceFQU16WithFQU8ForMatmul
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>)
func.func @ReplaceFQU16WithFQU8ForMatmul(%arg0: tensor<1x512x51x39xf32>) -> tensor<51x512x51x5xf32> {
    %cst = const.Declare tensor<51x512x39x5xf32> = dense<1.000000e+00> : tensor<51x512x39x5xf32>
    %low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.MatMul(%0, %cst) : tensor<1x512x51x39xf32>, tensor<51x512x39x5xf32> -> tensor<51x512x51x5xf32>
    return %1 : tensor<51x512x51x5xf32>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<51x512x39x5xf32> = dense<1.000000e+00> : tensor<51x512x39x5xf32>
    // CHECK:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:    [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[MATMUL:%.+]] = IE.MatMul([[FQ]], [[CST]]) : tensor<1x512x51x39xf32>, tensor<51x512x39x5xf32> -> tensor<51x512x51x5xf32>
    // CHECK:    return [[MATMUL]] : tensor<51x512x51x5xf32>
}

// -----

// CHECK-LABEL: @ModifyZeroPointForLow
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @ModifyZeroPointForLow(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x19xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<-2.578910e-01> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<-1.578910e-01> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.Convolution(%0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    return %1 : tensor<1x512x25x19xf32>

    // CHECK:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.257549942> : tensor<1x1x1x1xf32>
    // CHECK:    [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.578910e-01> : tensor<1x1x1x1xf32>
    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[FQ]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    // CHECK:    return [[CONV]] : tensor<1x512x25x19xf32>
}

// -----

// CHECK-LABEL: @ModifyZeroPointForHigh
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @ModifyZeroPointForHigh(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x19xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<-2.578910e-01> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<14.1136> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.Convolution(%0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    return %1 : tensor<1x512x25x19xf32>

    // CHECK:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.578910e-01> : tensor<1x1x1x1xf32>
    // CHECK:    [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<16.1826611> : tensor<1x1x1x1xf32>
    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[FQ]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    // CHECK:    return [[CONV]] : tensor<1x512x25x19xf32>
}

// -----

// CHECK-LABEL: @AvoidLoweringWhenMatMulHasTwoInputsFQ
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<1x512x39x51xf32>)
func.func @AvoidLoweringWhenMatMulHasTwoInputsFQ(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<1x512x39x51xf32>) -> tensor<1x512x51x51xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.FakeQuantize(%arg1, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x39x51xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x39x51xf32>
    %2 = IE.MatMul(%0, %1) : tensor<1x512x51x39xf32>, tensor<1x512x39x51xf32> -> tensor<1x512x51x51xf32>
    return %2 : tensor<1x512x51x51xf32>

    // CHECK:    [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[ARG1]]) : tensor<1x512x51x39xf32>, tensor<1x512x39x51xf32> -> tensor<1x512x51x51xf32>
    // CHECK:    return [[MATMUL]] : tensor<1x512x51x51xf32>
}

// -----

// CHECK-LABEL: @AvoidLoweringWhenLowAndHighEqual
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @AvoidLoweringWhenLowAndHighEqual(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x19xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.Convolution(%0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    return %1 : tensor<1x512x25x19xf32>

    // CHECK:    [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    // CHECK:    return [[CONV]] : tensor<1x512x25x19xf32>
}

// -----

// CHECK-LABEL: @FullyConnectedWithOneFQThroughAffineReshape
// CHECK:   [[ARG0:%.+]]: tensor<4x16x32xf32>, [[ARG1:%.+]]: tensor<32xf32>
func.func @FullyConnectedWithOneFQThroughAffineReshape(%arg0: tensor<4x16x32xf32>, %arg1: tensor<32xf32>) -> tensor<64x1xf32> {
    %low = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %high = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<4x16x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<4x16x32xf32>

    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 32]} : tensor<32xf32> -> tensor<1x32xf32>
    %2 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [64, 32]} : tensor<4x16x32xf32> -> tensor<64x32xf32>

    %3 = IE.FullyConnected(%2, %1) : tensor<64x32xf32>, tensor<1x32xf32> -> tensor<64x1xf32>

    return %3 : tensor<64x1xf32>

    // CHECK: [[LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    // CHECK: [[HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<4x16x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<4x16x32xf32>

    // CHECK:   [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[ARG1]]) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [2, 3]]
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf32> -> tensor<1x32xf32>

    // CHECK:   [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[FQ]]) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [2, 3]]
    // CHECK-SAME:      shape_value = [64, 32]
    // CHECK-SAME:  } : tensor<4x16x32xf32> -> tensor<64x32xf32>

    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[AFFINE_RESHAPE_2]], [[AFFINE_RESHAPE_1]]) :
    // CHECK-SAME:  tensor<64x32xf32>, tensor<1x32xf32> -> tensor<64x1xf32>

    // CHECK:   return [[FC]] : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @FullyConnectedWithTwoFQInputThroughAffineReshape
// CHECK:   [[ARG0:%.+]]: tensor<4x16x32xf32>, [[ARG1:%.+]]: tensor<32xf32>
func.func @FullyConnectedWithTwoFQInputThroughAffineReshape(%arg0: tensor<4x16x32xf32>, %arg1: tensor<32xf32>) -> tensor<64x1xf32> {
    %low = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %high = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>

    %low_2 = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    %high_2 = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<4x16x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<4x16x32xf32>
    %1 = IE.FakeQuantize(%arg1, %low_2, %high_2, %low_2, %high_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<32xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<32xf32>

    %2 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [64, 32]} : tensor<4x16x32xf32> -> tensor<64x32xf32>
    %3 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 32]} : tensor<32xf32> -> tensor<1x32xf32>

    %4 = IE.FullyConnected(%2, %3) : tensor<64x32xf32>, tensor<1x32xf32> -> tensor<64x1xf32>

    return %4 : tensor<64x1xf32>

    // CHECK:   [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [2, 3]]
    // CHECK-SAME:      shape_value = [64, 32]
    // CHECK-SAME:  } : tensor<4x16x32xf32> -> tensor<64x32xf32>

    // CHECK:   [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[ARG1]]) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [2, 3]]
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf32> -> tensor<1x32xf32>

    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[AFFINE_RESHAPE_1]], [[AFFINE_RESHAPE_2]]) :
    // CHECK-SAME:  tensor<64x32xf32>, tensor<1x32xf32> -> tensor<64x1xf32>

    // CHECK:   return [[FC]] : tensor<64x1xf32>
}

// -----

// CHECK-LABEL: @MatMulShareFQInputThroughAffineReshape
// CHECK:   [[ARG0:%.+]]: tensor<4x16x32xf32>, [[ARG1:%.+]]: tensor<32x64x1x1xf32>
func.func @MatMulShareFQInputThroughAffineReshape(%arg0: tensor<4x16x32xf32>, %arg1: tensor<32x64x1x1xf32>) -> tensor<64x64xf32> {
    %low = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %high = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<4x16x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<4x16x32xf32>

    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [32, 64]} : tensor<32x64x1x1xf32> -> tensor<32x64xf32>
    %2 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [64, 32]} : tensor<4x16x32xf32> -> tensor<64x32xf32>

    %3 = IE.MatMul(%2, %1) : tensor<64x32xf32>, tensor<32x64xf32> -> tensor<64x64xf32>

    return %3 : tensor<64x64xf32>

    // CHECK: [[LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    // CHECK: [[HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<4x16x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<4x16x32xf32>

    // CHECK:   [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[ARG1]]) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [1]]
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<32x64x1x1xf32> -> tensor<32x64xf32>

    // CHECK:   [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[FQ]]) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [2, 3]]
    // CHECK-SAME:      shape_value = [64, 32]
    // CHECK-SAME:  } : tensor<4x16x32xf32> -> tensor<64x32xf32>

    // CHECK:   [[MATMUL:%.+]] = IE.MatMul([[AFFINE_RESHAPE_2]], [[AFFINE_RESHAPE_1]]) :
    // CHECK-SAME:  tensor<64x32xf32>, tensor<32x64xf32> -> tensor<64x64xf32>

    // CHECK:   return [[MATMUL]] : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: @MatMulWithTwoFQInputThroughAffineReshape
// CHECK:   [[ARG0:%.+]]: tensor<4x16x32xf32>, [[ARG1:%.+]]: tensor<32x64x1x1xf32>
func.func @MatMulWithTwoFQInputThroughAffineReshape(%arg0: tensor<4x16x32xf32>, %arg1: tensor<32x64x1x1xf32>) -> tensor<64x64xf32> {
    %low = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %high = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<4x16x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<4x16x32xf32>
    %1 = IE.FakeQuantize(%arg1, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<32x64x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<32x64x1x1xf32>

    %2 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [64, 32]} : tensor<4x16x32xf32> -> tensor<64x32xf32>
    %3 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [32, 64]} : tensor<32x64x1x1xf32> -> tensor<32x64xf32>

    %4 = IE.MatMul(%2, %3) : tensor<64x32xf32>, tensor<32x64xf32> -> tensor<64x64xf32>

    return %4 : tensor<64x64xf32>

    // CHECK:   [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [2, 3]]
    // CHECK-SAME:      shape_value = [64, 32]
    // CHECK-SAME:  } : tensor<4x16x32xf32> -> tensor<64x32xf32>

    // CHECK:   [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[ARG1]]) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [1]]
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<32x64x1x1xf32> -> tensor<32x64xf32>

    // CHECK:   [[MATMUL:%.+]] = IE.MatMul([[AFFINE_RESHAPE_1]], [[AFFINE_RESHAPE_2]]) :
    // CHECK-SAME:  tensor<64x32xf32>, tensor<32x64xf32> -> tensor<64x64xf32>

    // CHECK:   return [[MATMUL]] : tensor<64x64xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @MatMulShareFQInputThroughTranspose
// CHECK:   [[ARG0:%.+]]: tensor<4x4xf32>
func.func @MatMulShareFQInputThroughTranspose(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %low = const.Declare tensor<1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
    %high = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<4x4xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<4x4xf32>

    %1 = IE.Transpose(%0) {order_value = #CN} : tensor<4x4xf32> -> tensor<4x4xf32>

    %2 = IE.MatMul(%0, %1) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>

    return %2 : tensor<4x4xf32>

    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[ARG0]]) {order_value = #CN} : tensor<4x4xf32> -> tensor<4x4xf32>
    // CHECK:   [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[TRANSPOSE]]) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    // CHECK:   return [[MATMUL]] : tensor<4x4xf32>

}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @MatMulShareFQInputThroughReshape
// CHECK:   [[ARG0:%.+]]: tensor<4x4xf32>
func.func @MatMulShareFQInputThroughReshape(%arg0: tensor<4x4xf32>) -> tensor<4x8xf32> {
    %low = const.Declare tensor<1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
    %high = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<4x4xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<4x4xf32>

    %1 = IE.Reshape(%0) {shape_value = [4, 8]} : tensor<4x4xf32> -> tensor<4x8xf32>

    %2 = IE.MatMul(%0, %1) : tensor<4x4xf32>, tensor<4x8xf32> -> tensor<4x8xf32>

    return %2 : tensor<4x8xf32>

    // CHECK:   [[RESHAPE:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [4, 8]} : tensor<4x4xf32> -> tensor<4x8xf32>
    // CHECK:   [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[RESHAPE]]) : tensor<4x4xf32>, tensor<4x8xf32> -> tensor<4x8xf32>
    // CHECK:   return [[MATMUL]] : tensor<4x8xf32>

}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @MatMulWithBothNonFQInputs
// CHECK:   [[ARG0:%.+]]: tensor<4x4xf32>, [[ARG1:%.+]]: tensor<4x4xf32>
func.func @MatMulWithBothNonFQInputs(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = IE.Reshape(%arg0) {shape_value = [4, 8]} : tensor<4x4xf32> -> tensor<4x8xf32>
    %1 = IE.Reshape(%arg1) {shape_value = [8, 4]} : tensor<4x4xf32> -> tensor<8x4xf32>

    %2 = IE.MatMul(%0, %1) : tensor<4x8xf32>, tensor<8x4xf32> -> tensor<4x4xf32>

    return %2 : tensor<4x4xf32>

    // CHECK:   [[RESHAPE_FIRST:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [4, 8]} : tensor<4x4xf32> -> tensor<4x8xf32>
    // CHECK:   [[RESHAPE_LAST:%.+]] = IE.Reshape([[ARG1]]) {shape_value = [8, 4]} : tensor<4x4xf32> -> tensor<8x4xf32>
    // CHECK:   [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_FIRST]], [[RESHAPE_LAST]]) : tensor<4x8xf32>, tensor<8x4xf32> -> tensor<4x4xf32>
    // CHECK:   return [[MATMUL]] : tensor<4x4xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @MatMulWithFQOnRightInput
// CHECK:   [[ARG0:%.+]]: tensor<4x4xf32>, [[ARG1:%.+]]: tensor<4x4xf32>
func.func @MatMulWithFQOnRightInput(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %low = const.Declare tensor<1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
    %high = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<4x4xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<4x4xf32>

    %1 = IE.Reshape(%arg0) {shape_value = [4, 8]} : tensor<4x4xf32> -> tensor<4x8xf32>
    %2 = IE.Reshape(%0) {shape_value = [8, 4]} : tensor<4x4xf32> -> tensor<8x4xf32>

    %3 = IE.MatMul(%1, %2) : tensor<4x8xf32>, tensor<8x4xf32> -> tensor<4x4xf32>

    return %3 : tensor<4x4xf32>

    // CHECK: [[LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
    // CHECK: [[HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<4x4xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<4x4xf32>
    // CHECK: [[RESHAPE_FIRST:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [4, 8]} : tensor<4x4xf32> -> tensor<4x8xf32>
    // CHECK: [[RESHAPE_SECOND:%.+]] = IE.Reshape([[FQ]]) {shape_value = [8, 4]} : tensor<4x4xf32> -> tensor<8x4xf32>
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_FIRST]], [[RESHAPE_SECOND]]) : tensor<4x8xf32>, tensor<8x4xf32> -> tensor<4x4xf32>
    // CHECK: return [[MATMUL]] : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @CheckFQWithMatMulThroughSomeReshapes
// CHECK:   [[ARG0:%.+]]: tensor<1x25x512x19xf32>
func.func @CheckFQWithMatMulThroughSomeReshapes(%arg0: tensor<1x25x512x19xf32>) -> tensor<1x25x512xf32> {
    %subtract_first_input = const.Declare tensor<9728x512xf32> = dense<0.000000e+00> : tensor<9728x512xf32>
    %fq_last_out_high = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32> isSplat
    %fq_last_out_low = const.Declare tensor<1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1xf32> isSplat
    %fq_last_in_high = const.Declare tensor<1x1x1xf32> = dense<3.000000e+00> : tensor<1x1x1xf32> isSplat
    %fq_last_in_low = const.Declare tensor<1x1x1xf32> = dense<4.000000e+00> : tensor<1x1x1xf32> isSplat
    %multiply_input = const.Declare tensor<1x1xf32> = dense<5.000000e+00> : tensor<1x1xf32> isSplat
    %subtract_second_input = const.Declare tensor<1x1xf32> = dense<6.000000e+00> : tensor<1x1xf32> isSplat
    %fq_second_out_high = const.Declare tensor<1x1x1xf32> = dense<7.000000e+00> : tensor<1x1x1xf32> isSplat
    %fq_second_out_low = const.Declare tensor<1x1x1xf32> = dense<8.000000e+00> : tensor<1x1x1xf32> isSplat
    %fq_second_in_high = const.Declare tensor<1x1x1xf32> = dense<9.000000e+00> : tensor<1x1x1xf32> isSplat
    %fq_second_in_low = const.Declare tensor<1x1x1xf32> = dense<10.000000e+00> : tensor<1x1x1xf32> isSplat
    %fq_first_out_high = const.Declare tensor<1x1x1x1xf32> = dense<11.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq_first_out_low = const.Declare tensor<1x1x1x1xf32> = dense<12.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq_first_in_high = const.Declare tensor<1x1x1x1xf32> = dense<13.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq_first_in_low = const.Declare tensor<1x1x1x1xf32> = dense<14.000000e+00> : tensor<1x1x1x1xf32> isSplat

    %fq_first = IE.FakeQuantize(%arg0, %fq_first_in_low, %fq_first_in_high, %fq_first_out_low, %fq_first_out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x25x512x19xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x25x512x19xf32>
    %affine_reshape = IE.AffineReshape(%fq_first) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 25, 9728]} : tensor<1x25x512x19xf32> -> tensor<1x25x9728xf32>
    %fq_second = IE.FakeQuantize(%affine_reshape, %fq_second_in_low, %fq_second_in_high, %fq_second_out_low, %fq_second_out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x25x9728xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x25x9728xf32>
    %subtract = IE.Subtract(%subtract_first_input, %subtract_second_input) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<9728x512xf32>, tensor<1x1xf32> -> tensor<9728x512xf32>
    %mul = IE.Multiply(%subtract, %multiply_input) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<9728x512xf32>, tensor<1x1xf32> -> tensor<9728x512xf32>
    %transpose = IE.Transpose(%mul) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<9728x512xf32> -> tensor<512x9728xf32>
    %matmul = IE.MatMul(%fq_second, %transpose) {transpose_b} : tensor<1x25x9728xf32>, tensor<512x9728xf32> -> tensor<1x25x512xf32>
    %fq_last = IE.FakeQuantize(%matmul, %fq_last_in_low, %fq_last_in_high, %fq_last_out_low, %fq_last_out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x25x512xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x25x512xf32>

    return %fq_last : tensor<1x25x512xf32>

    // CHECK: [[ADD_LAST_INPUT:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-2.000000e+00> : tensor<1x1x1xf32>
    // CHECK: [[INPUT_OUT_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<7.00000048> : tensor<1x1x1xf32>
    // CHECK: [[INPUT_OUT_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<8.000000e+00> : tensor<1x1x1xf32>
    // CHECK: [[INPUT_IN_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<9.000000e+00> : tensor<1x1x1xf32>
    // CHECK: [[INPUT_IN_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<9.99956512> : tensor<1x1x1xf32>
    // CHECK: [[SUBTRACT_FIRST_INPUT:%.+]] = const.Declare tensor<9728x512xf32> = dense<0.000000e+00> : tensor<9728x512xf32>
    // CHECK: [[MUL_INPUT:%.+]] = const.Declare tensor<1x1xf32> = dense<5.000000e+00> : tensor<1x1xf32>
    // CHECK: [[SUBTRACT_SECOND_INPUT:%.+]] = const.Declare tensor<1x1xf32> = dense<6.000000e+00> : tensor<1x1xf32>
    // CHECK: [[ADD_FIRST_INPUT:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK: [[ADD_FIRST:%.+]] = IE.Add([[ARG0]], [[ADD_FIRST_INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x25x512x19xf32>, tensor<1x1x1x1xf32> -> tensor<1x25x512x19xf32>
    // CHECK: [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[ADD_FIRST]])
    // CHECK{LITERAL}: {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 25, 9728]} : tensor<1x25x512x19xf32> -> tensor<1x25x9728xf32>
    // CHECK: [[FQ_INPUT:%.+]] = IE.FakeQuantize([[AFFINE_RESHAPE]], [[INPUT_IN_LOW]], [[INPUT_IN_HIGH]], [[INPUT_OUT_LOW]], [[INPUT_OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x25x9728xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x25x9728xf32>
    // CHECK: [[SUBTRACT:%.+]] = IE.Subtract([[SUBTRACT_FIRST_INPUT]], [[SUBTRACT_SECOND_INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<9728x512xf32>, tensor<1x1xf32> -> tensor<9728x512xf32>
    // CHECK: [[MUL:%.+]] = IE.Multiply([[SUBTRACT]], [[MUL_INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<9728x512xf32>, tensor<1x1xf32> -> tensor<9728x512xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[MUL]]) {order_value = #CN} : tensor<9728x512xf32> -> tensor<512x9728xf32>
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[FQ_INPUT]], [[TRANSPOSE]]) {transpose_b} : tensor<1x25x9728xf32>, tensor<512x9728xf32> -> tensor<1x25x512xf32>
    // CHECK: [[ADD_LAST:%.+]] = IE.Add([[MATMUL]], [[ADD_LAST_INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x25x512xf32>, tensor<1x1x1xf32> -> tensor<1x25x512xf32>
    // CHECK: return [[ADD_LAST]] : tensor<1x25x512xf32>
}

// -----

// CHECK-LABEL: @CheckFQWithFullyConnectedThroughSomeReshapes
// CHECK:   [[ARG0:%.+]]: tensor<1x25x9728xf32>
func.func @CheckFQWithFullyConnectedThroughSomeReshapes(%arg0: tensor<1x25x9728xf32>) -> tensor<1x25x512xf32> {
    %subtract_first_input = const.Declare tensor<9728x512xf32> = dense<0.000000e+00> : tensor<9728x512xf32>
    %fq_last_out_high = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32> isSplat
    %fq_last_out_low = const.Declare tensor<1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1xf32> isSplat
    %fq_last_in_high = const.Declare tensor<1x1x1xf32> = dense<3.000000e+00> : tensor<1x1x1xf32> isSplat
    %fq_last_in_low = const.Declare tensor<1x1x1xf32> = dense<4.000000e+00> : tensor<1x1x1xf32> isSplat
    %multiply_input = const.Declare tensor<1x1xf32> = dense<5.000000e+00> : tensor<1x1xf32> isSplat
    %subtract_second_input = const.Declare tensor<1x1xf32> = dense<6.000000e+00> : tensor<1x1xf32> isSplat
    %fq_first_out_high = const.Declare tensor<1x1x1xf32> = dense<7.000000e+00> : tensor<1x1x1xf32> isSplat
    %fq_first_out_low = const.Declare tensor<1x1x1xf32> = dense<8.000000e+00> : tensor<1x1x1xf32> isSplat
    %fq_first_in_high = const.Declare tensor<1x1x1xf32> = dense<9.000000e+00> : tensor<1x1x1xf32> isSplat
    %fq_first_in_low = const.Declare tensor<1x1x1xf32> = dense<10.000000e+00> : tensor<1x1x1xf32> isSplat
    %add_last = const.Declare tensor<1x1x512xf32> = dense<15.000000e+00> : tensor<1x1x512xf32>

    %fq_first = IE.FakeQuantize(%arg0, %fq_first_in_low, %fq_first_in_high, %fq_first_out_low, %fq_first_out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x25x9728xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x25x9728xf32>
    %subtract = IE.Subtract(%subtract_first_input, %subtract_second_input) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<9728x512xf32>, tensor<1x1xf32> -> tensor<9728x512xf32>
    %multiply = IE.Multiply(%subtract, %multiply_input) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<9728x512xf32>, tensor<1x1xf32> -> tensor<9728x512xf32>
    %transpose = IE.Transpose(%multiply) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<9728x512xf32> -> tensor<512x9728xf32>
    %reshape_first = IE.Reshape(%fq_first) {shape_value = [25, 9728]} : tensor<1x25x9728xf32> -> tensor<25x9728xf32>
    %fc = IE.FullyConnected(%reshape_first, %transpose) : tensor<25x9728xf32>, tensor<512x9728xf32> -> tensor<25x512xf32>
    %reshape_last = IE.Reshape(%fc) {shape_value = [1, 25, 512]} : tensor<25x512xf32> -> tensor<1x25x512xf32>
    %fq_last = IE.FakeQuantize(%reshape_last, %fq_last_in_low, %fq_last_in_high, %fq_last_out_low, %fq_last_out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x25x512xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x25x512xf32>
    %add = IE.Add(%fq_last, %add_last) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x25x512xf32>, tensor<1x1x512xf32> -> tensor<1x25x512xf32>

    return %add : tensor<1x25x512xf32>

    // CHECK: [[ADD_FIRST_INPUT:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-2.000000e+00> : tensor<1x1x1xf32>
    // CHECK: [[SUBTRACT_FIRST_INPUT:%.+]] = const.Declare tensor<9728x512xf32> = dense<0.000000e+00> : tensor<9728x512xf32>
    // CHECK: [[MUL_INPUT:%.+]] = const.Declare tensor<1x1xf32> = dense<5.000000e+00> : tensor<1x1xf32>
    // CHECK: [[SUBTRACT_SECOND_INPUT:%.+]] = const.Declare tensor<1x1xf32> = dense<6.000000e+00> : tensor<1x1xf32>
    // CHECK: [[ADD_LAST_INPUT:%.+]] = const.Declare tensor<1x1x512xf32> = dense<1.500000e+01> : tensor<1x1x512xf32>
    // CHECK: [[INPUT_IN_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<9.99956512> : tensor<1x1x1xf32>
    // CHECK: [[INPUT_IN_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<9.000000e+00> : tensor<1x1x1xf32>
    // CHECK: [[INPUT_OUT_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<8.000000e+00> : tensor<1x1x1xf32>
    // CHECK: [[INPUT_OUT_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<7.00000048> : tensor<1x1x1xf32>
    // CHECK: [[FQ_INPUT:%.+]] = IE.FakeQuantize([[ARG0]], [[INPUT_IN_LOW]], [[INPUT_IN_HIGH]], [[INPUT_OUT_LOW]], [[INPUT_OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x25x9728xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x25x9728xf32>
    // CHECK: [[SUBTRACT:%.+]] = IE.Subtract([[SUBTRACT_FIRST_INPUT]], [[SUBTRACT_SECOND_INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<9728x512xf32>, tensor<1x1xf32> -> tensor<9728x512xf32>
    // CHECK: [[MUL:%.+]] = IE.Multiply([[SUBTRACT]], [[MUL_INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<9728x512xf32>, tensor<1x1xf32> -> tensor<9728x512xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[MUL]]) {order_value = #CN} : tensor<9728x512xf32> -> tensor<512x9728xf32>
    // CHECK: [[RESHAPE_FIRST:%.+]] = IE.Reshape([[FQ_INPUT]]) {shape_value = [25, 9728]} : tensor<1x25x9728xf32> -> tensor<25x9728xf32>
    // CHECK: [[FC:%.+]] = IE.FullyConnected([[RESHAPE_FIRST]], [[TRANSPOSE]]) : tensor<25x9728xf32>, tensor<512x9728xf32> -> tensor<25x512xf32>
    // CHECK: [[RESHAPE_LAST:%.+]] = IE.Reshape([[FC]]) {shape_value = [1, 25, 512]} : tensor<25x512xf32> -> tensor<1x25x512xf32>
    // CHECK: [[ADD_FIRST:%.+]] = IE.Add([[RESHAPE_LAST]], [[ADD_FIRST_INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x25x512xf32>, tensor<1x1x1xf32> -> tensor<1x25x512xf32>
    // CHECK: [[ADD_LAST:%.+]] = IE.Add([[ADD_FIRST]], [[ADD_LAST_INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x25x512xf32>, tensor<1x1x512xf32> -> tensor<1x25x512xf32>
    // CHECK: return [[ADD_LAST:%.+]] : tensor<1x25x512xf32>
}
