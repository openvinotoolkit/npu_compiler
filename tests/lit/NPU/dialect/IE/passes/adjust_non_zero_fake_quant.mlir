//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-non-zero-fake-quant %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


// CHECK-LABEL: @AdjustFakeQuantLow
func.func @AdjustFakeQuantLow(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.01> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<5.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.01> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<5.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    %1 = IE.MaxPool(%0) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1] } : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    return %1 : tensor<1x3x30x30xf16>


    // CHECK-DAG:       [[LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:       [[HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[CLAMP:%.*]] = IE.Clamp([[FQ]]) {max = 5.000000e+00 : f64, min = 0.01000213623046875 : f64} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[MAXPOOL:%.*]] = IE.MaxPool([[CLAMP]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       return [[MAXPOOL]]
}

// -----

// CHECK-LABEL: @AdjustFakeQuantHigh
func.func @AdjustFakeQuantHigh(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<-5.0> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<-0.01> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<-5.0> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<-0.01> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    %1 = IE.MaxPool(%0) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1] } : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    return %1 : tensor<1x3x30x30xf16>


    // CHECK-DAG:       [[HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:       [[LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-5.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[CLAMP:%.*]] = IE.Clamp(%0) {max = -0.01000213623046875 : f64, min = -5.000000e+00 : f64} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[MAXPOOL:%.*]] = IE.MaxPool([[CLAMP]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       return [[MAXPOOL]]
}

// -----

// CHECK-LABEL: @NotAdjustFakeQuantWithBigRange
func.func @NotAdjustFakeQuantWithBigRange(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<4.5> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<5.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<4.5> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<5.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    %1 = IE.MaxPool(%0) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1] } : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    return %1 : tensor<1x3x30x30xf16>


    // CHECK-DAG:       [[LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<4.500000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:           [[FQ:%.*]] = IE.FakeQuantize
    // CHECK:           [[MAXPOOL:%.*]] = IE.MaxPool
    // CHECK:           return [[MAXPOOL]]
}

// -----

// CHECK-LABEL: @NotAdjustWithZeroInRange
func.func @NotAdjustWithZeroInRange(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<-0.1> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<5.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<-0.1> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<5.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    %1 = IE.MaxPool(%0) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1] } : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    return %1 : tensor<1x3x30x30xf16>


    // CHECK-DAG:       [[LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-9.997550e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:           [[FQ:%.*]] = IE.FakeQuantize
    // CHECK:           [[MAXPOOL:%.*]] = IE.MaxPool
    // CHECK:           return [[MAXPOOL]]
}

// -----

// CHECK-LABEL: @NotAdjustWithDifferentInOutRange
func.func @NotAdjustWithDifferentInOutRange(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.1> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<5.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.1> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<6.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    %1 = IE.MaxPool(%0) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1] } : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    return %1 : tensor<1x3x30x30xf16>


    // CHECK-DAG:       [[LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<9.997550e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:           [[OUT_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<6.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:           [[FQ:%.*]] = IE.FakeQuantize
    // CHECK:           [[MAXPOOL:%.*]] = IE.MaxPool
    // CHECK:           return [[MAXPOOL]]
}

// -----

// CHECK-LABEL: @NotAdjustWithPerChannelQuantize
func.func @NotAdjustWithPerChannelQuantize(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %input_low = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.0]],[[0.0]], [[0.0]]]]>  : tensor<1x3x1x1xf16>
    %input_high = const.Declare tensor<1x3x1x1xf16> = dense<[[[[5.0]],[[6.0]], [[5.0]]]]>  : tensor<1x3x1x1xf16>
    %output_low = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.0]],[[0.0]], [[0.0]]]]> : tensor<1x3x1x1xf16>
    %output_high = const.Declare tensor<1x3x1x1xf16> = dense<[[[[5.0]],[[6.0]], [[5.0]]]]> : tensor<1x3x1x1xf16>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x30x30xf16>
    %1 = IE.MaxPool(%0) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1] } : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    return %1 : tensor<1x3x30x30xf16>


    // CHECK-DAG:       [[LOW:%.*]] = const.Declare tensor<1x3x1x1xf16> = dense<0.000000e+00> : tensor<1x3x1x1xf16>
    // CHECK-DAG:       [[HIGH:%.*]] = const.Declare tensor<1x3x1x1xf16> =
    // CHECK-SAME{LITERAL}:          dense<[[[[5.000000e+00]], [[6.000000e+00]], [[5.000000e+00]]]]> : tensor<1x3x1x1xf16>
    // CHECK:           [[FQ:%.*]] = IE.FakeQuantize
    // CHECK:           [[MAXPOOL:%.*]] = IE.MaxPool
    // CHECK:           return [[MAXPOOL]]
}

// -----

// CHECK-LABEL: @NotAdjustFQWithNonStaticLowHighValues
func.func @NotAdjustFQWithNonStaticLowHighValues(%arg0: tensor<1x256x2048x1xf16>, %arg1: tensor<1x1x1xf16>, %arg2: tensor<1x1x1xf16>, %arg3: tensor<1x1x1xf16>, %arg4: tensor<1x1x1xf16>) -> tensor<1x256x2048x1xf16> {
    %input_low = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>
    %input_high = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>
    %output_low = IE.AffineReshape(%arg3) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>
    %output_high = IE.AffineReshape(%arg4) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    } : tensor<1x256x2048x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x256x2048x1xf16>

    return %0 : tensor<1x256x2048x1xf16>

    // CHECK:           [[IN_LOW:%.+]] = IE.AffineReshape
    // CHECK:           [[IN_HIGH:%.+]] = IE.AffineReshape
    // CHECK:           [[OUT_LOW:%.+]] = IE.AffineReshape
    // CHECK:           [[OUT_HIGH:%.+]] = IE.AffineReshape
    // CHECK:           [[FQ:%.+]] = IE.FakeQuantize

    // CHECK:           return [[FQ]]
}
