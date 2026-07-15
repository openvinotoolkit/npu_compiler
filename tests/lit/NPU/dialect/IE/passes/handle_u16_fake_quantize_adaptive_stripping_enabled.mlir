//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=512 --init-compiler="platform=%platform% enable-adaptive-stripping=true" --handle-u16-fake-quantize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010


// CHECK-LABEL: func.func @RemoveFQU16
// CHECK-SAME:        [[INPUT:%[^:]+]]: tensor<1x4x640x640xf16>
func.func @RemoveFQU16(%arg0: tensor<1x4x640x640xf16>) -> tensor<1x4x640x640xf16> {
    %low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %high = const.Declare tensor<1x1x1x1xf16> = dense<57.1374702> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x4x640x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x640x640xf16>
    %1 = IE.Sigmoid(%0) : tensor<1x4x640x640xf16> -> tensor<1x4x640x640xf16>
    return %1 : tensor<1x4x640x640xf16>

    // CHECK: [[SIGMOID:%.+]] = IE.Sigmoid([[INPUT]]) : tensor<1x4x640x640xf16> -> tensor<1x4x640x640xf16>

    // CHECK: return [[SIGMOID]]
}

// -----

// CHECK-LABEL: @ReplaceFQU16WithScaleShift
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x64x64xf32>)
func.func @ReplaceFQU16WithScaleShift(%arg0: tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32> {
    %cst = const.Declare tensor<1x512x1x1xf32> = dense<-12.2241688> : tensor<1x512x1x1xf32>
    %cst_0 = const.Declare tensor<1x512x1x1xf32> = dense<12.7559032> : tensor<1x512x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-28.7695446> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<24.0061626> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32>, tensor<1x512x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x64x64xf32>
    %1 = IE.Sigmoid(%0) : tensor<1x512x64x64xf32> -> tensor<1x512x64x64xf32>
    return %1 : tensor<1x512x64x64xf32>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<1x512x1x1xf32> = dense<2.11271238> : tensor<1x512x1x1xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x512x1x1xf32> = dense<-2.94339228> : tensor<1x512x1x1xf32>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[ARG0]], [[WEIGHTS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32> -> tensor<1x512x64x64xf32>
    // CHECK:       [[ADD:%.+]] = IE.Add([[MULTIPLY]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x64x64xf32>, tensor<1x512x1x1xf32> -> tensor<1x512x64x64xf32>
    // CHECK:       [[SIGMOID:%.+]] = IE.Sigmoid([[ADD]]) : tensor<1x512x64x64xf32> -> tensor<1x512x64x64xf32>

    // CHECK: return [[SIGMOID]]
}

// -----

// CHECK-LABEL: @ApplyU16FQToConst
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x512x128x128xf32>)
func.func @ApplyU16FQToConst(%arg0: tensor<1x512x128x128xf32>) -> tensor<1x512x129x129xf32> {
    %pad_value = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    %val_low = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    %val_high = const.Declare tensor<1xf32> = dense<9.875000e-05> : tensor<1xf32>
    %pads_begin = const.Declare tensor<4xsi64> = dense<0> : tensor<4xsi64>
    %pads_end = const.Declare tensor<4xsi64> = dense<[0, 0, 1, 1]> : tensor<4xsi64>

    %pad_value_fq = IE.FakeQuantize(%pad_value, %val_low, %val_high, %val_low, %val_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %pad = IE.Pad(%arg0) [%pads_begin, %pads_end, %pad_value_fq] {mode = #IE.pad_mode<CONSTANT>} : tensor<1x512x128x128xf32>, tensor<4xsi64>, tensor<4xsi64>, tensor<1xf32> -> tensor<1x512x129x129xf32>
    return %pad : tensor<1x512x129x129xf32>

    // CHECK:     [[PADS_BEGIN:%.+]] = const.Declare tensor<4xsi64> = dense<0> : tensor<4xsi64>
    // CHECK:     [[PADS_END:%.+]] = const.Declare tensor<4xsi64> = dense<[0, 0, 1, 1]> : tensor<4xsi64>
    // CHECK:     [[PAD_VALUE:%.+]] = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    // CHECK:     [[PAD_OP:%.+]] = IE.Pad([[INPUT]]) [[[PADS_BEGIN]], [[PADS_END]], [[PAD_VALUE]]] {mode = #IE.pad_mode<CONSTANT>}
    // CHECK-SAME:      tensor<1x512x128x128xf32>, tensor<4xsi64>, tensor<4xsi64>, tensor<1xf32> -> tensor<1x512x129x129xf32>
    // CHECK:     return [[PAD_OP]] : tensor<1x512x129x129xf32>
}

// -----

// CHECK-LABEL: @ApplyOneU16FQToConstMultipleUses
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x512x128x128xf32>)
func.func @ApplyOneU16FQToConstMultipleUses(%arg0: tensor<1x512x128x128xf32>) -> (tensor<1x512x129x129xf32>, tensor<1xf32>) {
    %pad_value = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>
    %val_low = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    %val_high_1 = const.Declare tensor<1xf32> = dense<9.875000e-05> : tensor<1xf32>
    %pads_begin = const.Declare tensor<4xsi64> = dense<0> : tensor<4xsi64>
    %pads_end = const.Declare tensor<4xsi64> = dense<[0, 0, 1, 1]> : tensor<4xsi64>
    %val_high_2 = const.Declare tensor<1xf32> = dense<4.000000e-05> : tensor<1xf32>

    %pad_value_fq = IE.FakeQuantize(%pad_value, %val_low, %val_high_1, %val_low, %val_high_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %pad = IE.Pad(%arg0) [%pads_begin, %pads_end, %pad_value_fq] {mode = #IE.pad_mode<CONSTANT>} : tensor<1x512x128x128xf32>, tensor<4xsi64>, tensor<4xsi64>, tensor<1xf32> -> tensor<1x512x129x129xf32>

    %fq2 = IE.FakeQuantize(%pad_value, %val_low, %val_high_2, %val_low, %val_high_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %relu = IE.ReLU(%fq2) : tensor<1xf32> -> tensor<1xf32>

    %relu_2 = IE.ReLU(%pad_value) : tensor<1xf32> -> tensor<1xf32>

    %add = IE.Add(%relu, %relu_2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    return %pad, %add : tensor<1x512x129x129xf32>, tensor<1xf32>

    //CHECK:    [[FQ_INPUT:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>
    //CHECK:    [[VAL_LOW:%.+]] = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    //CHECK:    [[PADS_BEGIN:%.+]] = const.Declare tensor<4xsi64> = dense<0> : tensor<4xsi64>
    //CHECK:    [[PADS_END:%.+]] = const.Declare tensor<4xsi64> = dense<[0, 0, 1, 1]> : tensor<4xsi64>
    //CHECK:    [[VAL_HIGH:%.+]] = const.Declare tensor<1xf32> = dense<4.000000e-05> : tensor<1xf32>
    //CHECK:    [[PAD_VALUE:%.+]] = const.Declare tensor<1xf32> = dense<9.875000e-05> : tensor<1xf32>
    //CHECK:    [[PAD:%.+]] = IE.Pad([[INPUT]]) [[[PADS_BEGIN]], [[PADS_END]], [[PAD_VALUE]]] {mode = #IE.pad_mode<CONSTANT>} : tensor<1x512x128x128xf32>, tensor<4xsi64>, tensor<4xsi64>, tensor<1xf32> -> tensor<1x512x129x129xf32>
    //CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[FQ_INPUT]], [[VAL_LOW]], [[VAL_HIGH]], [[VAL_LOW]], [[VAL_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    //CHECK:    [[RELU_0:%.+]] = IE.ReLU([[FQ]]) : tensor<1xf32> -> tensor<1xf32>
    //CHECK:    [[RELU_1:%.+]] = IE.ReLU([[FQ_INPUT]]) : tensor<1xf32> -> tensor<1xf32>
    //CHECK:    [[ADD:%.+]] = IE.Add([[RELU_0]], [[RELU_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    //CHECK:    return [[PAD]], [[ADD]] : tensor<1x512x129x129xf32>, tensor<1xf32>
}

// -----

// CHECK-LABEL: @ApplyTwoU16FQToConstMultipleUses
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x512x128x128xf32>)
func.func @ApplyTwoU16FQToConstMultipleUses(%arg0: tensor<1x512x128x128xf32>) -> (tensor<1x512x129x129xf32>, tensor<1xf32>) {
    %pad_value = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>
    %val_low = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    %val_high_1 = const.Declare tensor<1xf32> = dense<9.875000e-05> : tensor<1xf32>
    %pads_begin = const.Declare tensor<4xsi64> = dense<0> : tensor<4xsi64>
    %pads_end = const.Declare tensor<4xsi64> = dense<[0, 0, 1, 1]> : tensor<4xsi64>
    %val_high_2 = const.Declare tensor<1xf32> = dense<4.000000e-05> : tensor<1xf32>

    %pad_value_fq = IE.FakeQuantize(%pad_value, %val_low, %val_high_1, %val_low, %val_high_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %pad = IE.Pad(%arg0) [%pads_begin, %pads_end, %pad_value_fq] {mode = #IE.pad_mode<CONSTANT>} : tensor<1x512x128x128xf32>, tensor<4xsi64>, tensor<4xsi64>, tensor<1xf32> -> tensor<1x512x129x129xf32>

    %fq2 = IE.FakeQuantize(%pad_value, %val_low, %val_high_2, %val_low, %val_high_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %relu = IE.ReLU(%fq2) : tensor<1xf32> -> tensor<1xf32>

    return %pad, %relu : tensor<1x512x129x129xf32>, tensor<1xf32>

    //CHECK:    [[PADS_BEGIN:%.+]] = const.Declare tensor<4xsi64> = dense<0> : tensor<4xsi64>
    //CHECK:    [[PADS_END:%.+]] = const.Declare tensor<4xsi64> = dense<[0, 0, 1, 1]> : tensor<4xsi64>
    //CHECK:    [[PAD_VALUE:%.+]] = const.Declare tensor<1xf32> = dense<9.875000e-05> : tensor<1xf32>
    //CHECK:    [[PAD:%.+]] = IE.Pad([[INPUT]]) [[[PADS_BEGIN]], [[PADS_END]], [[PAD_VALUE]]] {mode = #IE.pad_mode<CONSTANT>} : tensor<1x512x128x128xf32>, tensor<4xsi64>, tensor<4xsi64>, tensor<1xf32> -> tensor<1x512x129x129xf32>
    //CHECK:    [[FQ_INPUT:%.+]] = const.Declare tensor<1xf32> = dense<4.000000e-05> : tensor<1xf32>
    //CHECK:    [[RELU_0:%.+]] = IE.ReLU([[FQ_INPUT]]) : tensor<1xf32> -> tensor<1xf32>
    //CHECK:    return [[PAD]], [[RELU_0]] : tensor<1x512x129x129xf32>, tensor<1xf32>
}

// -----

// CHECK-LABEL: @ReplaceDoubleFQU16WithReLU
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @ReplaceDoubleFQU16WithReLU(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x19xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<5.000000> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.FakeQuantize(%0, %cst, %cst_1, %cst, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %conv = IE.Convolution(%1, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    return %conv : tensor<1x512x25x19xf32>

    // CHECK:    [[RELU:%.+]] = IE.ReLU([[ARG0]]) : tensor<1x512x51x39xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[RELU]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    // CHECK:    return [[CONV]] : tensor<1x512x25x19xf32>
}

// -----

// CHECK-LABEL: @ReplaceDoubleFQU16WithReLUDifferentValues
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @ReplaceDoubleFQU16WithReLUDifferentValues(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x19xf32> {
    %fq1_low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %fq1_high = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>
    %fq2_low = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %fq2_high = const.Declare tensor<1x1x1x1xf32> = dense<14.7559032> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %fq1_low, %fq1_high, %fq1_low, %fq1_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.FakeQuantize(%0, %fq2_low, %fq2_high, %fq2_low, %fq2_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %conv = IE.Convolution(%1, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    return %conv : tensor<1x512x25x19xf32>

    // CHECK:    [[RELU:%.+]] = IE.ReLU([[ARG0]]) : tensor<1x512x51x39xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[RELU]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    // CHECK:    return [[CONV]] : tensor<1x512x25x19xf32>
}

// -----

// CHECK-LABEL: @ReplaceDoubleFQU16WithReLUDifferentValuesSecondFQ
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @ReplaceDoubleFQU16WithReLUDifferentValuesSecondFQ(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x19xf32> {
    %fq2_low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %fq1_high = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>
    %fq1_low = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %fq2_high = const.Declare tensor<1x1x1x1xf32> = dense<14.7559032> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %fq1_low, %fq1_high, %fq1_low, %fq1_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.FakeQuantize(%0, %fq2_low, %fq2_high, %fq2_low, %fq2_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %conv = IE.Convolution(%1, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    return %conv : tensor<1x512x25x19xf32>

    // CHECK:    [[RELU:%.+]] = IE.ReLU([[ARG0]]) : tensor<1x512x51x39xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[RELU]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    // CHECK:    return [[CONV]] : tensor<1x512x25x19xf32>
}

// -----

// CHECK-LABEL: @AsymmetricFakeQuantizeOpToScaleShiftOp
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>
func.func @AsymmetricFakeQuantizeOpToScaleShiftOp(%arg0: tensor<1x128x32x64xf32>) -> tensor<1x128x32x64xf32> {
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32>
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <1.0> : tensor<1x1x1x1xf32>
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32>
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <6.5535e+04> : tensor<1x1x1x1xf32>

    %1 = IE.FakeQuantize(%arg0, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    return %1 : tensor<1x128x32x64xf32>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK: return [[MULTIPLY]] : tensor<1x128x32x64xf32>
}

// -----

// CHECK-LABEL: @RemoveFQU16WithOutputConv
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @RemoveFQU16WithOutputConv(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x19xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.Convolution(%0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    return %1 : tensor<1x512x25x19xf32>

    // CHECK:    [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    // CHECK:    return [[CONV]] : tensor<1x512x25x19xf32>
}

// -----

// CHECK-LABEL: @RemoveFQU16WithOutputGroupConv
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @RemoveFQU16WithOutputGroupConv(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x51x39xf32> {
    %low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.GroupConvolution(%0, %arg1) {dilations = [1, 1], groups = 1 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x51x39xf32>
    return %1 : tensor<1x512x51x39xf32>

    // CHECK:    [[GROUPCONV:%.+]] = IE.GroupConvolution([[ARG0]], [[ARG1]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    return [[GROUPCONV]] : tensor<1x512x51x39xf32>
}

// -----

// CHECK-LABEL: @RemoveFQU16WithOutputMatmul
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>)
func.func @RemoveFQU16WithOutputMatmul(%arg0: tensor<1x512x51x39xf32>) -> tensor<51x512x51x5xf32> {
    %cst = const.Declare tensor<51x512x39x5xf32> = dense<1.000000e+00> : tensor<51x512x39x5xf32>
    %low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %high = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.MatMul(%0, %cst) : tensor<1x512x51x39xf32>, tensor<51x512x39x5xf32> -> tensor<51x512x51x5xf32>
    return %1 : tensor<51x512x51x5xf32>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<51x512x39x5xf32> = dense<1.000000e+00> : tensor<51x512x39x5xf32>
    // CHECK:    [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[CST]]) : tensor<1x512x51x39xf32>, tensor<51x512x39x5xf32> -> tensor<51x512x51x5xf32>
    // CHECK:    return [[MATMUL]] : tensor<51x512x51x5xf32>
}

// -----

// CHECK-LABEL: @U16FQConsolidationWithReshape
// CHECK:   [[INPUT:%.+]]: tensor<1x1x32x32xf32>
func.func @U16FQConsolidationWithReshape(%input: tensor<1x1x32x32xf32>) -> tensor<1x32x1x32xf32> {
  %fq_in_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<-5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_in_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<50000.000000e+00> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e-04> : tensor<1x1x1x1xf32>
  %zp = const.Declare tensor<1x1x1x1xf32> = dense<-5.000000e+00> : tensor<1x1x1x1xf32>
  %fq = IE.FakeQuantize(%input, %fq_in_low_cst, %fq_in_high_cst, %fq_out_low_cst, %fq_out_high_cst) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                        levels = 65536 : i64} : tensor<1x1x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x32x32xf32>
  %convert_int = IE.Convert(%fq) {dstElemType = ui16} : tensor<1x1x32x32xf32> -> tensor<1x1x32x32xui16>
  %reshape = IE.Reshape(%convert_int) {
                        shape_value = [1, 32, 1, 32]} : tensor<1x1x32x32xui16> -> tensor<1x32x1x32xui16>
  %convert_float = IE.Convert(%reshape) {dstElemType = f32} : tensor<1x32x1x32xui16> -> tensor<1x32x1x32xf32>
  %mul = IE.Multiply(%convert_float, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1x32xf32>
  %add = IE.Add(%mul, %zp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1x32xf32>
  return %add : tensor<1x32x1x32xf32>

  // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[INPUT]])
  // CHECK: return [[RESHAPE]] : tensor<1x32x1x32xf32>
}

// -----

// CHECK-LABEL: @U16FQConsolidationWithAffineReshape
// CHECK:   [[INPUT:%.+]]: tensor<1x1x32x32xf32>
func.func @U16FQConsolidationWithAffineReshape(%input: tensor<1x1x32x32xf32>) -> tensor<1x32x1x32xf32> {
  %fq_in_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<-5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_in_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<50000.000000e+00> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e-04> : tensor<1x1x1x1xf32>
  %zp = const.Declare tensor<1x1x1x1xf32> = dense<-5.000000e+00> : tensor<1x1x1x1xf32>
  %fq = IE.FakeQuantize(%input, %fq_in_low_cst, %fq_in_high_cst, %fq_out_low_cst, %fq_out_high_cst) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                        levels = 65536 : i64} : tensor<1x1x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x32x32xf32>
  %convert_int = IE.Convert(%fq) {dstElemType = ui16} : tensor<1x1x32x32xf32> -> tensor<1x1x32x32xui16>
  %reshape = IE.AffineReshape(%convert_int) {
                              dim_mapping = [[0], [0], [1, 2], [3]],
                              shape_value = [1, 32, 1, 32]} : tensor<1x1x32x32xui16> -> tensor<1x32x1x32xui16>
  %convert_float = IE.Convert(%reshape) {dstElemType = f32} : tensor<1x32x1x32xui16> -> tensor<1x32x1x32xf32>
  %mul = IE.Multiply(%convert_float, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1x32xf32>
  %add = IE.Add(%mul, %zp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1x32xf32>
  return %add : tensor<1x32x1x32xf32>

  // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
  // CHECK: return [[RESHAPE]] : tensor<1x32x1x32xf32>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @U16FQConsolidationWithTranspose
// CHECK:   [[INPUT:%.+]]: tensor<1x1x32x64xf32>
func.func @U16FQConsolidationWithTranspose(%input: tensor<1x1x32x64xf32>) -> tensor<1x1x64x32xf32> {
  %fq_in_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<-5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_in_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<50000.000000e+00> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e-04> : tensor<1x1x1x1xf32>
  %zp = const.Declare tensor<1x1x1x1xf32> = dense<-5.000000e+00> : tensor<1x1x1x1xf32>
  %fq = IE.FakeQuantize(%input, %fq_in_low_cst, %fq_in_high_cst, %fq_out_low_cst, %fq_out_high_cst) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                        levels = 65536 : i64} : tensor<1x1x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x32x64xf32>
  %convert_int = IE.Convert(%fq) {dstElemType = ui16} : tensor<1x1x32x64xf32> -> tensor<1x1x32x64xui16>
  %transpose = IE.Transpose(%convert_int) {
                          order_value = #NCWH} : tensor<1x1x32x64xui16> -> tensor<1x1x64x32xui16>
  %convert_float = IE.Convert(%transpose) {dstElemType = f32} : tensor<1x1x64x32xui16> -> tensor<1x1x64x32xf32>
  %mul = IE.Multiply(%convert_float, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x32xf32>
  %add = IE.Add(%mul, %zp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x32xf32>
  return %add : tensor<1x1x64x32xf32>
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT]])
  // CHECK: return [[TRANSPOSE]] : tensor<1x1x64x32xf32>
}

// -----

// CHECK-LABEL: @U16FQConsolidationNoConvert
// CHECK:   [[INPUT:%.+]]: tensor<1x1x32x32xf32>
func.func @U16FQConsolidationNoConvert(%input: tensor<1x1x32x32xf32>) -> tensor<1x32x1x32xf32> {
  %fq_in_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<-5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_in_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<50000.000000e+00> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e-04> : tensor<1x1x1x1xf32>
  %zp = const.Declare tensor<1x1x1x1xf32> = dense<-5.000000e+00> : tensor<1x1x1x1xf32>
  %fq = IE.FakeQuantize(%input, %fq_in_low_cst, %fq_in_high_cst, %fq_out_low_cst, %fq_out_high_cst) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                        levels = 65536 : i64} : tensor<1x1x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x32x32xf32>
  %reshape = IE.AffineReshape(%fq) {
                              dim_mapping = [[0], [0], [1, 2], [3]],
                              shape_value = [1, 32, 1, 32]} : tensor<1x1x32x32xf32> -> tensor<1x32x1x32xf32>
  %mul = IE.Multiply(%reshape, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1x32xf32>
  %add = IE.Add(%mul, %zp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1x32xf32>
  return %add : tensor<1x32x1x32xf32>

  // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
  // CHECK: return [[RESHAPE]] : tensor<1x32x1x32xf32>
}

// -----

// No Reshape/Transpose between converts (nonComputeOp is optional).
// Integer grid: outLow=0, outHigh=65535 (levels-1) → identity FQ → stripped.

// CHECK-LABEL: @U16FQConsolidationWithoutNonComputeOp
// CHECK:   [[INPUT:%.+]]: tensor<1x1x32x32xf32>
func.func @U16FQConsolidationWithoutNonComputeOp(%input: tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32> {
  %fq_in_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<-5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_in_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<65535.000000e+00> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<1.525900e-04> : tensor<1x1x1x1xf32>
  %zp = const.Declare tensor<1x1x1x1xf32> = dense<-5.000000e+00> : tensor<1x1x1x1xf32>
  %fq = IE.FakeQuantize(%input, %fq_in_low_cst, %fq_in_high_cst, %fq_out_low_cst, %fq_out_high_cst) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                        levels = 65536 : i64} : tensor<1x1x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x32x32xf32>
  %convert_int = IE.Convert(%fq) {dstElemType = ui16} : tensor<1x1x32x32xf32> -> tensor<1x1x32x32xui16>
  %convert_float = IE.Convert(%convert_int) {dstElemType = f32} : tensor<1x1x32x32xui16> -> tensor<1x1x32x32xf32>
  %mul = IE.Multiply(%convert_float, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x32x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x32x32xf32>
  %add = IE.Add(%mul, %zp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x32x32xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x32x32xf32>
  return %add : tensor<1x1x32x32xf32>

  // CHECK: return [[INPUT]] : tensor<1x1x32x32xf32>
}

// -----

// f16 destination Convert (f32→ui16→f16 Q/DQ pattern).
// Integer grid + type mismatch → identity FQ → Convert(f32→f16) inserted → FQ stripped.

// CHECK-LABEL: @U16FQConsolidationWithF16Converts
// CHECK:   [[INPUT:%.+]]: tensor<1x1x32x32xf32>
func.func @U16FQConsolidationWithF16Converts(%input: tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf16> {
  %fq_in_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<-5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_in_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<65535.000000e+00> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf16> = dense<1.525900e-04> : tensor<1x1x1x1xf16>
  %zp = const.Declare tensor<1x1x1x1xf16> = dense<-5.000000e+00> : tensor<1x1x1x1xf16>
  %fq = IE.FakeQuantize(%input, %fq_in_low_cst, %fq_in_high_cst, %fq_out_low_cst, %fq_out_high_cst) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                        levels = 65536 : i64} : tensor<1x1x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x32x32xf32>
  %convert_int = IE.Convert(%fq) {dstElemType = ui16} : tensor<1x1x32x32xf32> -> tensor<1x1x32x32xui16>
  %convert_float = IE.Convert(%convert_int) {dstElemType = f16} : tensor<1x1x32x32xui16> -> tensor<1x1x32x32xf16>
  %mul = IE.Multiply(%convert_float, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x32x32xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x32x32xf16>
  %add = IE.Add(%mul, %zp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x32x32xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x32x32xf16>
  return %add : tensor<1x1x32x32xf16>

  // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x1x32x32xf32> -> tensor<1x1x32x32xf16>
  // CHECK: return [[CONVERT]] : tensor<1x1x32x32xf16>
}

// -----

// Single Convert (only convertToFloat, no convertToInt).
// Integer grid + type mismatch → identity FQ → Convert(f32→f16) inserted → FQ stripped.

// CHECK-LABEL: @U16FQConsolidationSingleConvert
// CHECK:   [[INPUT:%.+]]: tensor<1x1x32x32xf32>
func.func @U16FQConsolidationSingleConvert(%input: tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf16> {
  %fq_in_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<-5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_in_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %fq_out_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<65535.000000e+00> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf16> = dense<1.525900e-04> : tensor<1x1x1x1xf16>
  %zp = const.Declare tensor<1x1x1x1xf16> = dense<-5.000000e+00> : tensor<1x1x1x1xf16>
  %fq = IE.FakeQuantize(%input, %fq_in_low_cst, %fq_in_high_cst, %fq_out_low_cst, %fq_out_high_cst) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                        levels = 65536 : i64} : tensor<1x1x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x32x32xf32>
  %convert_float = IE.Convert(%fq) {dstElemType = f16} : tensor<1x1x32x32xf32> -> tensor<1x1x32x32xf16>
  %mul = IE.Multiply(%convert_float, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x32x32xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x32x32xf16>
  %add = IE.Add(%mul, %zp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x32x32xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x32x32xf16>
  return %add : tensor<1x1x32x32xf16>

  // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x1x32x32xf32> -> tensor<1x1x32x32xf16>
  // CHECK: return [[CONVERT]] : tensor<1x1x32x32xf16>
}

// -----

// CHECK: !qElemType = !quant.uniform<u16:f32, 7.6295109483482109E-5:39321>

// CHECK-LABEL: @ConvertU16FQConvertToU16ToQuantize
// CHECK:   [[ARG0:%.+]]: tensor<1x1x64x3072xf32>
func.func @ConvertU16FQConvertToU16ToQuantize(%arg0: tensor<1x1x64x3072xf32>) -> (tensor<1x1x64x3072xf32>, tensor<1x1x64x3072xui16>) {
    %fq_in_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<-3.000000e+00> : tensor<1x1x1x1xf32>
    %fq_in_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    %fq_out_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %fq_out_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<65535.000000e+00> : tensor<1x1x1x1xf32>
    %mul_cst = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
    %fq = IE.FakeQuantize(%arg0, %fq_in_low_cst, %fq_in_high_cst, %fq_out_low_cst, %fq_out_high_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>
    %cvt = IE.Convert(%fq) {dstElemType = ui16} : tensor<1x1x64x3072xf32> -> tensor<1x1x64x3072xui16>
    %mul = IE.Multiply(%fq, %mul_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>

    return %mul, %cvt : tensor<1x1x64x3072xf32>, tensor<1x1x64x3072xui16>

    // CHECK: [[MUL1_CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK: [[MUL0_CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.310700e+04> : tensor<1x1x1x1xf32>
    // CHECK: [[ADD_CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.932100e+04> : tensor<1x1x1x1xf32>
    // CHECK: [[MUL0:%.+]] = IE.Multiply([[ARG0]], [[MUL0_CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[MUL0]], [[ADD_CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>
    // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x1x64x3072xf32> -> tensor<1x1x64x3072x!qElemType>
    // CHECK: [[QCAST:%.+]] = IE.QuantizeCast([[QUANT]]) {dstElemType = ui16} : tensor<1x1x64x3072x!qElemType> -> tensor<1x1x64x3072xui16>
    // CHECK: [[MUL1:%.+]] = IE.Multiply([[ADD]], [[MUL1_CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>
    // CHECK: return [[MUL1]], [[QCAST]] : tensor<1x1x64x3072xf32>, tensor<1x1x64x3072xui16>
}

// -----

// CHECK-LABEL: @DoNotConvertU8FQConvertToU16ToQuantize
// CHECK:   [[ARG0:%.+]]: tensor<1x1x64x3072xf32>
func.func @DoNotConvertU8FQConvertToU16ToQuantize(%arg0: tensor<1x1x64x3072xf32>) -> (tensor<1x1x64x3072xf32>, tensor<1x1x64x3072xui16>) {
    %fq_in_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<-3.000000e+00> : tensor<1x1x1x1xf32>
    %fq_in_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    %fq_out_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %fq_out_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<256.000000e+00> : tensor<1x1x1x1xf32>
    %mul_cst = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
    %fq = IE.FakeQuantize(%arg0, %fq_in_low_cst, %fq_in_high_cst, %fq_out_low_cst, %fq_out_high_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>
    %cvt = IE.Convert(%fq) {dstElemType = ui16} : tensor<1x1x64x3072xf32> -> tensor<1x1x64x3072xui16>
    %mul = IE.Multiply(%fq, %mul_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>

    return %mul, %cvt : tensor<1x1x64x3072xf32>, tensor<1x1x64x3072xui16>

    // CHECK: [[FQ_IN_LOW_CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-3.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK: [[FQ_IN_HIGH_CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK: [[FQ_OUT_LOW_CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK: [[FQ_OUT_HIGH_CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.560000e+02> : tensor<1x1x1x1xf32>
    // CHECK: [[MUL1_CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[FQ_IN_LOW_CST]], [[FQ_IN_HIGH_CST]], [[FQ_OUT_LOW_CST]], [[FQ_OUT_HIGH_CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>
    // CHECK: [[CVT:%.+]] = IE.Convert([[FQ]]) {dstElemType = ui16} : tensor<1x1x64x3072xf32> -> tensor<1x1x64x3072xui16>
    // CHECK: [[MUL1:%.+]] = IE.Multiply([[FQ]], [[MUL1_CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>
    // CHECK: return [[MUL1]], [[CVT]] : tensor<1x1x64x3072xf32>, tensor<1x1x64x3072xui16>
}

// -----

// CHECK-LABEL: @DoNotConvertU16FQConvertToU8ToQuantize
// CHECK:   [[ARG0:%.+]]: tensor<1x1x64x3072xf32>
func.func @DoNotConvertU16FQConvertToU8ToQuantize(%arg0: tensor<1x1x64x3072xf32>) -> (tensor<1x1x64x3072xf32>, tensor<1x1x64x3072xui8>) {
    %fq_in_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<-3.000000e+00> : tensor<1x1x1x1xf32>
    %fq_in_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    %fq_out_low_cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %fq_out_high_cst = const.Declare tensor<1x1x1x1xf32> = dense<65535.000000e+00> : tensor<1x1x1x1xf32>
    %mul_cst = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
    %fq = IE.FakeQuantize(%arg0, %fq_in_low_cst, %fq_in_high_cst, %fq_out_low_cst, %fq_out_high_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>
    %cvt = IE.Convert(%fq) {dstElemType = ui8} : tensor<1x1x64x3072xf32> -> tensor<1x1x64x3072xui8>
    %mul = IE.Multiply(%fq, %mul_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>

    return %mul, %cvt : tensor<1x1x64x3072xf32>, tensor<1x1x64x3072xui8>

    // CHECK: [[MUL1_CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK: [[MUL0_CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.310700e+04> : tensor<1x1x1x1xf32>
    // CHECK: [[ADD_CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.932100e+04> : tensor<1x1x1x1xf32>
    // CHECK: [[MUL0:%.+]] = IE.Multiply([[ARG0]], [[MUL0_CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[MUL0]], [[ADD_CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>
    // CHECK: [[CVT:%.+]] = IE.Convert([[ADD]]) {dstElemType = ui8} : tensor<1x1x64x3072xf32> -> tensor<1x1x64x3072xui8>
    // CHECK: [[MUL1:%.+]] = IE.Multiply([[ADD]], [[MUL1_CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x64x3072xf32>
    // CHECK: return [[MUL1]], [[CVT]] : tensor<1x1x64x3072xf32>, tensor<1x1x64x3072xui8>
}

// -----

// Parent U16 FQ has multiple uses; only the child matches the il=ol=0 ReLU pattern.
// The child must be rewritten as ReLU consuming the parent's input.

// CHECK-LABEL: @ReplaceChildFQU16WithReLUParentMultipleUses
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @ReplaceChildFQU16WithReLUParentMultipleUses(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>)
        -> (tensor<1x512x25x19xf32>, tensor<1x512x51x39xf32>) {
    %fq1_low  = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %fq1_high = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>
    %fq2_low  = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %fq2_high = const.Declare tensor<1x1x1x1xf32> = dense<14.7559032> : tensor<1x1x1x1xf32>

    %parent = IE.FakeQuantize(%arg0, %fq1_low, %fq1_high, %fq1_low, %fq1_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %child  = IE.FakeQuantize(%parent, %fq2_low, %fq2_high, %fq2_low, %fq2_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %sibling = IE.Sigmoid(%parent) : tensor<1x512x51x39xf32> -> tensor<1x512x51x39xf32>
    %conv = IE.Convolution(%child, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x512x51x39xf32>, tensor<512x512x3x3xf32> -> tensor<1x512x25x19xf32>
    return %conv, %sibling : tensor<1x512x25x19xf32>, tensor<1x512x51x39xf32>

    // The child FQ becomes a ReLU consuming the parent FQ's input directly. The parent
    // FQ itself is left in the IR for its remaining (sibling) use and is independently
    // simplified by RemoveU16FakeQuantizeRewriter (here: il=ol so it is dropped to its input,
    // making the Sigmoid consume %arg0 directly).
    // CHECK:    [[RELU:%.+]] = IE.ReLU([[ARG0]]) : tensor<1x512x51x39xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[SIGMOID:%.+]] = IE.Sigmoid([[ARG0]]) : tensor<1x512x51x39xf32> -> tensor<1x512x51x39xf32>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[RELU]], [[ARG1]])
    // CHECK:    return [[CONV]], [[SIGMOID]]
}

// -----

// The child U16 FQ has per-channel input/output low/high constants (shape 1x4x1x1),
// so the per-tensor-FQ ReLU pattern does not apply and it must not be rewritten as a ReLU.

// CHECK-LABEL: @DoNotReplaceChildFQU16PerChannel
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x4x51x39xf32>, [[ARG1:%.+]]: tensor<4x4x3x3xf32>)
func.func @DoNotReplaceChildFQU16PerChannel(%arg0: tensor<1x4x51x39xf32>, %arg1: tensor<4x4x3x3xf32>) -> tensor<1x4x25x19xf32> {
    %fq1_low  = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %fq1_high = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>
    %fq2_low  = const.Declare tensor<1x4x1x1xf32> = dense<0.000000e+00> : tensor<1x4x1x1xf32>
    %fq2_high = const.Declare tensor<1x4x1x1xf32> = dense<[[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]]> : tensor<1x4x1x1xf32>

    %parent = IE.FakeQuantize(%arg0, %fq1_low, %fq1_high, %fq1_low, %fq1_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x4x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x51x39xf32>
    %child  = IE.FakeQuantize(%parent, %fq2_low, %fq2_high, %fq2_low, %fq2_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x4x51x39xf32>, tensor<1x4x1x1xf32>, tensor<1x4x1x1xf32>, tensor<1x4x1x1xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x51x39xf32>
    %conv = IE.Convolution(%child, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x4x51x39xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x25x19xf32>
    return %conv : tensor<1x4x25x19xf32>

    // The chain must NOT be collapsed into a ReLU because the child FQ is per-channel.
    // CHECK-NOT: IE.ReLU
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x4x51x39xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x25x19xf32>
    // CHECK:    return [[CONV]] : tensor<1x4x25x19xf32>
}

// -----

// CHECK-LABEL: @ReplaceFQU16AfterLayerWithPostOpInterfaceWithReLU
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x64x56x56xf32>, [[ARG1:%.+]]: tensor<64x64x1x1xf32>)
func.func @ReplaceFQU16AfterLayerWithPostOpInterfaceWithReLU(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<64x64x1x1xf32>) -> (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) {
    %bias     = const.Declare tensor<1x64x1x1xf32> = dense<1.0>          : tensor<1x64x1x1xf32>
    %bias2    = const.Declare tensor<1x64x1x1xf32> = dense<2.0>          : tensor<1x64x1x1xf32>
    %low      = const.Declare tensor<1x1x1x1xf32>  = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %in_high  = const.Declare tensor<1x1x1x1xf32>  = dense<12.7559032>   : tensor<1x1x1x1xf32>
    %out_high = const.Declare tensor<1x1x1x1xf32>  = dense<6.3779516>    : tensor<1x1x1x1xf32>
    %fq2_high = const.Declare tensor<1x1x1x1xf32>  = dense<14.7559032>   : tensor<1x1x1x1xf32>

    %add = IE.Add(%arg0, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32> -> tensor<1x64x56x56xf32>
    %fq  = IE.FakeQuantize(%add, %low, %in_high, %low, %out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x64x56x56xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x56x56xf32>

    %add2 = IE.Add(%fq, %bias2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32> -> tensor<1x64x56x56xf32>

    %fq2  = IE.FakeQuantize(%fq, %low, %fq2_high, %low, %fq2_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x64x56x56xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x56x56xf32>
    %conv = IE.Convolution(%fq2, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32> -> tensor<1x64x56x56xf32>

    return %add2, %conv : tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>

    // CHECK:    [[ADD_OP:%.+]] = IE.Add([[ARG0]], {{%.+}})
    // CHECK:    [[RELU_FQ:%.+]] = IE.ReLU([[ADD_OP]]) : tensor<1x64x56x56xf32> -> tensor<1x64x56x56xf32>
    // CHECK:    [[ADD2:%.+]] = IE.Add([[RELU_FQ]], {{%.+}})
    // CHECK:    [[RELU_FQ2:%.+]] = IE.ReLU([[ADD_OP]]) : tensor<1x64x56x56xf32> -> tensor<1x64x56x56xf32>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[RELU_FQ2]], [[ARG1]])
    // CHECK:    return [[ADD2]], [[CONV]]
}

// -----

// CHECK-LABEL: @ReplaceFQU16AfterAddWithNonU16FQAndAddUsers
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x64x56x56xf32>, [[ARG1:%.+]]: tensor<64x64x1x1xf32>)
func.func @ReplaceFQU16AfterAddWithNonU16FQAndAddUsers(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<64x64x1x1xf32>) -> (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) {
    %bias     = const.Declare tensor<1x64x1x1xf32> = dense<1.0>          : tensor<1x64x1x1xf32>
    %bias2    = const.Declare tensor<1x64x1x1xf32> = dense<2.0>          : tensor<1x64x1x1xf32>
    %low      = const.Declare tensor<1x1x1x1xf32>  = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %in_high  = const.Declare tensor<1x1x1x1xf32>  = dense<12.7559032>   : tensor<1x1x1x1xf32>
    %fq8_high = const.Declare tensor<1x1x1x1xf32>  = dense<6.0>          : tensor<1x1x1x1xf32>

    %add = IE.Add(%arg0, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32> -> tensor<1x64x56x56xf32>
    %fq  = IE.FakeQuantize(%add, %low, %in_high, %low, %in_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x64x56x56xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x56x56xf32>

    // Two users: a non-FQ Add and a non-U16 FQ.
    %add2 = IE.Add(%fq, %bias2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32> -> tensor<1x64x56x56xf32>
    %fq8  = IE.FakeQuantize(%fq, %low, %fq8_high, %low, %fq8_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x64x56x56xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x56x56xf32>
    %conv = IE.Convolution(%fq8, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32> -> tensor<1x64x56x56xf32>

    return %add2, %conv : tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>

    // CHECK:    [[ADD_OP:%.+]] = IE.Add([[ARG0]], {{%.+}})
    // CHECK:    [[RELU:%.+]] = IE.ReLU([[ADD_OP]]) : tensor<1x64x56x56xf32> -> tensor<1x64x56x56xf32>
    // CHECK:    [[ADD2:%.+]] = IE.Add([[RELU]], {{%.+}})
    // CHECK:    [[FQ8:%.+]] = IE.FakeQuantize([[RELU]], {{%.+}}) {{.*}} levels = 256
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[FQ8]], [[ARG1]])
    // CHECK:    return [[ADD2]], [[CONV]]
}
