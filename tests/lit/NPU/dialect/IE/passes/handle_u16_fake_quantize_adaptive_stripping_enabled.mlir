//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=512 --init-compiler="vpu-arch=%arch% enable-adaptive-stripping=true" --handle-u16-fake-quantize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


// CHECK-LABEL: func.func @RemoveFQU16
// CHECK-SAME:        [[INPUT:%arg0]]: tensor<1x4x640x640xf16>
func.func @RemoveFQU16(%arg0: tensor<1x4x640x640xf16>) -> tensor<1x4x640x640xf16> {
    %low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %high = const.Declare tensor<1x1x1x1xf16> = dense<57.1374702> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x4x640x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x640x640xf16>
    %1 = IE.Sigmoid(%0) : tensor<1x4x640x640xf16> -> tensor<1x4x640x640xf16>
    return %1 : tensor<1x4x640x640xf16>

    // CHECK: [[SIGMOID:%.*]] = IE.Sigmoid([[INPUT]]) : tensor<1x4x640x640xf16> -> tensor<1x4x640x640xf16>

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

    //CHECK:    [[FQ_INPUT:%.+]] = const.Declare tensor<1xf32> = dense<4.000000e-05> : tensor<1xf32>
    //CHECK:    [[PADS_BEGIN:%.+]] = const.Declare tensor<4xsi64> = dense<0> : tensor<4xsi64>
    //CHECK:    [[PADS_END:%.+]] = const.Declare tensor<4xsi64> = dense<[0, 0, 1, 1]> : tensor<4xsi64>
    //CHECK:    [[PAD_VALUE:%.+]] = const.Declare tensor<1xf32> = dense<9.875000e-05> : tensor<1xf32>
    //CHECK:    [[PAD:%.+]] = IE.Pad([[INPUT]]) [[[PADS_BEGIN]], [[PADS_END]], [[PAD_VALUE]]] {mode = #IE.pad_mode<CONSTANT>} : tensor<1x512x128x128xf32>, tensor<4xsi64>, tensor<4xsi64>, tensor<1xf32> -> tensor<1x512x129x129xf32>
    //CHECK:    [[RELU_0:%.+]] = IE.ReLU([[FQ_INPUT]]) : tensor<1xf32> -> tensor<1xf32>
    //CHECK:    return [[PAD]], [[RELU_0]] : tensor<1x512x129x129xf32>, tensor<1xf32>
}

// -----

// CHECK-LABEL: @ReplaceDoubleFQU16WithReLU
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf32>, [[ARG1:%.+]]: tensor<512x512x3x3xf32>)
func.func @ReplaceDoubleFQU16WithReLU(%arg0: tensor<1x512x51x39xf32>, %arg1: tensor<512x512x3x3xf32>) -> tensor<1x512x25x19xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<12.7559032> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
    %1 = IE.FakeQuantize(%0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x51x39xf32>
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
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <1.0> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <6.5535e+04> : tensor<1x1x1x1xf32> isSplat

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
