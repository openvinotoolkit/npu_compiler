//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --split-conv-with-multiple-fq %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @SplitConvWithOnlyFakeQuantConsumers
func.func @SplitConvWithOnlyFakeQuantConsumers(%input: tensor<1x3x62x62xf32>) -> (tensor<1x4x60x60xf32>, tensor<1x4x60x60xf32>, tensor<1x4x60x60xf32>) {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x62x62xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x62x62xf32>

    %weights_low = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
    %weights_high = const.Declare tensor<1xf32> = dense<10.0> : tensor<1xf32>
    %weights = const.Declare tensor<4x3x3x3xf32> = dense<128> : tensor<4x3x3x3xui8>, [#const.CastElemType<f32>]
    %1 = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 } :
        tensor<4x3x3x3xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<4x3x3x3xf32>

    %2 = IE.Convolution(%0, %1)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>

    %bias = const.Declare tensor<1x4x1x1xf32> = dense<1.0> : tensor<1x4x1x1xf32>
    %3 = IE.Add(%2, %bias)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x4x60x60xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x60x60xf32>

    %fq3_low = const.Declare tensor<f32> = dense<-5.0> : tensor<f32>
    %fq3_high = const.Declare tensor<f32> = dense<5.0> : tensor<f32>
    %4 = IE.FakeQuantize(%3, %fq3_low, %fq3_high, %fq3_low, %fq3_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    %5 = IE.ReLU(%3) :
        tensor<1x4x60x60xf32> -> tensor<1x4x60x60xf32>

    %fq2_low = const.Declare tensor<f32> = dense<-4.0> : tensor<f32>
    %fq2_high = const.Declare tensor<f32> = dense<4.0> : tensor<f32>
    %6 = IE.FakeQuantize(%5, %fq2_low, %fq2_high, %fq2_low, %fq2_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    %fq1_low = const.Declare tensor<f32> = dense<-3.0> : tensor<f32>
    %fq1_high = const.Declare tensor<f32> = dense<3.0> : tensor<f32>
    %7 = IE.FakeQuantize(%5, %fq1_low, %fq1_high, %fq1_low, %fq1_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    return %4, %6, %7 : tensor<1x4x60x60xf32>, tensor<1x4x60x60xf32>, tensor<1x4x60x60xf32>

    // CHECK-DAG: [[MIN_IN:%.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MAX_IN:%.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-DAG: [[MIN_WEIGHTS:%.*]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
    // CHECK-DAG: [[MAX_WEIGHTS:%.*]] = const.Declare tensor<1xf32> = dense<1.000000e+01> : tensor<1xf32>

    // CHECK-DAG: [[FILTERS:%.*]] = const.Declare tensor<4x3x3x3xf32> = dense<128> : tensor<4x3x3x3xui8>, [#const.CastElemType<f32>]
    // CHECK-DAG: [[BIAS:%.*]] = const.Declare tensor<1x4x1x1xf32> = dense<1.000000e+00> : tensor<1x4x1x1xf32>

    // CHECK-DAG: [[MIN3:%.*]] = const.Declare tensor<f32> = dense<-5.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MAX3:%.*]] = const.Declare tensor<f32> = dense<5.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MIN2:%.*]] = const.Declare tensor<f32> = dense<-4.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MAX2:%.*]] = const.Declare tensor<f32> = dense<4.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MIN1:%.*]] = const.Declare tensor<f32> = dense<-3.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MAX1:%.*]] = const.Declare tensor<f32> = dense<3.000000e+00> : tensor<f32>

    // CHECK: [[VAL0:%.*]] = IE.FakeQuantize(%arg0, [[MIN_IN]], [[MAX_IN]], [[MIN_IN]], [[MAX_IN]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x62x62xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x62x62xf32>
    // CHECK: [[VAL1:%.*]] = IE.FakeQuantize([[FILTERS]], [[MIN_WEIGHTS]], [[MAX_WEIGHTS]], [[MIN_WEIGHTS]], [[MAX_WEIGHTS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<4x3x3x3xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<4x3x3x3xf32>

    // CHECK: [[VAL2:%.*]] = IE.Convolution([[VAL0]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL3:%.*]] = IE.Add([[VAL2]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x60x60xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL4:%.*]] = IE.FakeQuantize([[VAL3]], [[MIN3]], [[MAX3]], [[MIN3]], [[MAX3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    // CHECK: [[VAL5:%.*]] = IE.Convolution([[VAL0]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL6:%.*]] = IE.Add([[VAL5]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x60x60xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL7:%.*]] = IE.ReLU([[VAL6]]) : tensor<1x4x60x60xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL8:%.*]] = IE.FakeQuantize([[VAL7]], [[MIN2]], [[MAX2]], [[MIN2]], [[MAX2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    // CHECK: [[VAL9:%.*]] = IE.Convolution([[VAL0]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL10:%.*]] = IE.Add([[VAL9]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x60x60xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL11:%.*]] = IE.ReLU([[VAL10]]) : tensor<1x4x60x60xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL12:%.*]] = IE.FakeQuantize([[VAL11]], [[MIN1]], [[MAX1]], [[MIN1]], [[MAX1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    // CHECK: return [[VAL4]], [[VAL8]], [[VAL12]]
}

// CHECK-LABEL: @SplitConvWithReLUAndFakeQuant
func.func @SplitConvWithReLUAndFakeQuant(%input: tensor<1x3x62x62xf32>) -> (tensor<1x4x60x60xf32>, tensor<1x4x60x60xf32>) {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x62x62xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x62x62xf32>

    %weights = const.Declare tensor<4x3x3x3xf32> = dense<128> : tensor<4x3x3x3xui8>, [#const.CastElemType<f32>]
    %1 = IE.FakeQuantize(%weights, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 } :
        tensor<4x3x3x3xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<4x3x3x3xf32>

    %2 = IE.Convolution(%0, %1)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>

    %bias = const.Declare tensor<1x4x1x1xf32> = dense<1.0> : tensor<1x4x1x1xf32>
    %3 = IE.Add(%2, %bias)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x4x60x60xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x60x60xf32>

    %4 = IE.ReLU(%3) :
        tensor<1x4x60x60xf32> -> tensor<1x4x60x60xf32>

    %5 = IE.FakeQuantize(%4, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    return %3, %5 : tensor<1x4x60x60xf32>, tensor<1x4x60x60xf32>

    // CHECK-DAG: [[MIN:%.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MAX:%.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-DAG: [[FILTERS:%.*]] = const.Declare tensor<4x3x3x3xf32> = dense<128> : tensor<4x3x3x3xui8>, [#const.CastElemType<f32>]
    // CHECK-DAG: [[BIAS:%.*]] = const.Declare tensor<1x4x1x1xf32> = dense<1.000000e+00> : tensor<1x4x1x1xf32>

    // CHECK: [[VAL0:%.*]] = IE.FakeQuantize(%arg0, [[MIN]], [[MAX]], [[MIN]], [[MAX]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x62x62xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x62x62xf32>
    // CHECK: [[VAL1:%.*]] = IE.FakeQuantize([[FILTERS]], [[MIN]], [[MAX]], [[MIN]], [[MAX]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<4x3x3x3xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<4x3x3x3xf32>

    // CHECK: [[VAL2:%.*]] = IE.Convolution([[VAL0]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL3:%.*]] = IE.Add([[VAL2]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x60x60xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x60x60xf32>

    // CHECK: [[VAL4:%.*]] = IE.Convolution([[VAL0]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL5:%.*]] = IE.Add([[VAL4]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x60x60xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL6:%.*]] = IE.ReLU([[VAL5]]) : tensor<1x4x60x60xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL7:%.*]] = IE.FakeQuantize([[VAL6]], [[MIN]], [[MAX]], [[MIN]], [[MAX]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    // CHECK: return [[VAL3]], [[VAL7]]
}

// CHECK-LABEL: @SplitConvWithFakeQuant
func.func @SplitConvWithFakeQuant(%input: tensor<1x3x62x62xf32>) -> (tensor<1x4x60x60xf32>, tensor<1x4x60x60xf32>, tensor<1x4x60x60xf32>) {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x62x62xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x62x62xf32>

    %weights_low = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
    %weights_high = const.Declare tensor<1xf32> = dense<10.0> : tensor<1xf32>
    %weights = const.Declare tensor<4x3x3x3xf32> = dense<128> : tensor<4x3x3x3xui8>, [#const.CastElemType<f32>]
    %1 = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 } :
        tensor<4x3x3x3xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<4x3x3x3xf32>

    %2 = IE.Convolution(%0, %1)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>

    %bias = const.Declare tensor<1x4x1x1xf32> = dense<1.0> : tensor<1x4x1x1xf32>
    %3 = IE.Add(%2, %bias)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x4x60x60xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x60x60xf32>

    %fq3_low = const.Declare tensor<f32> = dense<-5.0> : tensor<f32>
    %fq3_high = const.Declare tensor<f32> = dense<5.0> : tensor<f32>
    %4 = IE.FakeQuantize(%3, %fq3_low, %fq3_high, %fq3_low, %fq3_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    %fq2_low = const.Declare tensor<f32> = dense<-4.0> : tensor<f32>
    %fq2_high = const.Declare tensor<f32> = dense<4.0> : tensor<f32>
    %5 = IE.FakeQuantize(%3, %fq2_low, %fq2_high, %fq2_low, %fq2_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    %fq1_low = const.Declare tensor<f32> = dense<-3.0> : tensor<f32>
    %fq1_high = const.Declare tensor<f32> = dense<3.0> : tensor<f32>
    %6 = IE.FakeQuantize(%3, %fq1_low, %fq1_high, %fq1_low, %fq1_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    return %4, %5, %6 : tensor<1x4x60x60xf32>, tensor<1x4x60x60xf32>, tensor<1x4x60x60xf32>

    // CHECK-DAG: [[MIN_IN:%.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MAX_IN:%.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-DAG: [[MIN_WEIGHTS:%.*]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
    // CHECK-DAG: [[MAX_WEIGHTS:%.*]] = const.Declare tensor<1xf32> = dense<1.000000e+01> : tensor<1xf32>

    // CHECK-DAG: [[FILTERS:%.*]] = const.Declare tensor<4x3x3x3xf32> = dense<128> : tensor<4x3x3x3xui8>, [#const.CastElemType<f32>]
    // CHECK-DAG: [[BIAS:%.*]] = const.Declare tensor<1x4x1x1xf32> = dense<1.000000e+00> : tensor<1x4x1x1xf32>

    // CHECK-DAG: [[MIN3:%.*]] = const.Declare tensor<f32> = dense<-5.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MAX3:%.*]] = const.Declare tensor<f32> = dense<5.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MIN2:%.*]] = const.Declare tensor<f32> = dense<-4.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MAX2:%.*]] = const.Declare tensor<f32> = dense<4.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MIN1:%.*]] = const.Declare tensor<f32> = dense<-3.000000e+00> : tensor<f32>
    // CHECK-DAG: [[MAX1:%.*]] = const.Declare tensor<f32> = dense<3.000000e+00> : tensor<f32>

    // CHECK: [[VAL0:%.*]] = IE.FakeQuantize(%arg0, [[MIN_IN]], [[MAX_IN]], [[MIN_IN]], [[MAX_IN]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x62x62xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x62x62xf32>
    // CHECK: [[VAL1:%.*]] = IE.FakeQuantize([[FILTERS]], [[MIN_WEIGHTS]], [[MAX_WEIGHTS]], [[MIN_WEIGHTS]], [[MAX_WEIGHTS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<4x3x3x3xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<4x3x3x3xf32>

    // CHECK: [[VAL2:%.*]] = IE.Convolution([[VAL0]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL3:%.*]] = IE.Add([[VAL2]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x60x60xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x60x60xf32>

    // CHECK: [[VAL4:%.*]] = IE.Convolution([[VAL0]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL5:%.*]] = IE.Add([[VAL4]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x60x60xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x60x60xf32>

    // CHECK: [[VAL6:%.*]] = IE.Convolution([[VAL0]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL7:%.*]] = IE.Add([[VAL6]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x60x60xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x60x60xf32>

    // CHECK: [[VAL8:%.*]] = IE.FakeQuantize([[VAL7]], [[MIN3]], [[MAX3]], [[MIN3]], [[MAX3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL9:%.*]] = IE.FakeQuantize([[VAL5]], [[MIN2]], [[MAX2]], [[MIN2]], [[MAX2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>
    // CHECK: [[VAL10:%.*]] = IE.FakeQuantize([[VAL3]], [[MIN1]], [[MAX1]], [[MIN1]], [[MAX1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    // CHECK: return [[VAL8]], [[VAL9]], [[VAL10]]
}
