//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=512 --init-compiler="vpu-arch=%arch% enable-qdq-optimization-aggressive=true" --handle-u16-fake-quantize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @ReplaceSingleFQU16WithReLU
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @ReplaceSingleFQU16WithReLU(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x19xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.FakeQuantize(%0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %conv = IE.Convolution(%1, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    return %conv : tensor<1x512x25x19xf32>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>
    // CHECK:    [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:    [[RELU:%.+]] = IE.ReLU(%arg0) : tensor<1x512x51x39xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[RELU]], [[CST_0]], [[CST]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[FQ]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    // CHECK:    return [[CONV]] : tensor<1x512x25x19xf32>
}

// -----

// CHECK-LABEL: @ReplaceFQU16WithFQU8ForConv
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @ReplaceFQU16WithFQU8ForConv(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x19xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.Convolution(%0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    return %1 : tensor<1x512x25x19xf32>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:    [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>
    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[FQ]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    // CHECK:    return [[CONV]] : tensor<1x512x25x19xf32>
}

// -----

// CHECK-LABEL: @ReplaceFQU16WithFQU8ForGroupConv
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @ReplaceFQU16WithFQU8ForGroupConv(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x51x39xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.GroupConvolution(%0, %arg1) {dilations = [1, 1], groups = 1 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x51x39xf32>
    return %1 : tensor<1x512x51x39xf32>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:    [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>
    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[GROUPCONV:%.+]] = IE.GroupConvolution([[FQ]], [[ARG1]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    return [[GROUPCONV]] : tensor<1x512x51x39xf32>
}

// -----

// CHECK-LABEL: @ReplaceFQU16WithFQU8ForMatmul
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>)
func.func @ReplaceFQU16WithFQU8ForMatmul(%arg0: tensor<1x512x51x39xf32>) -> tensor<51x512x51x5xf32> {
    %cst = const.Declare tensor<51x512x39x5xf32> = dense<1.000000e+00> : tensor<51x512x39x5xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.MatMul(%0, %cst) : tensor<1x512x51x39xf32>, tensor<51x512x39x5xf32> -> tensor<51x512x51x5xf32>
    return %1 : tensor<51x512x51x5xf32>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<51x512x39x5xf32> = dense<1.000000e+00> : tensor<51x512x39x5xf32>
    // CHECK:    [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:    [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[MATMUL:%.+]] = IE.MatMul([[FQ]], [[CST]]) : tensor<1x512x51x39xf32>, tensor<51x512x39x5xf32> -> tensor<51x512x51x5xf32>
    // CHECK:    return [[MATMUL]] : tensor<51x512x51x5xf32>
}
