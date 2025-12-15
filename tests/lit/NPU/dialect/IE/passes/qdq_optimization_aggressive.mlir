//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% enable-qdq-optimization-aggressive=false" --qdq-optimization-aggressive %s | FileCheck %s --check-prefix=CHECK-AGG-OFF
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% enable-qdq-optimization-aggressive=true" --qdq-optimization-aggressive %s | FileCheck %s --check-prefix=CHECK-AGG-ON
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-AGG-OFF-LABEL: @QDQOptimizationAggressiveForConv
// CHECK-AGG-OFF-SAME:      ([[INPUT:%.+]]: tensor<1x1x60x60xui8>) -> tensor<1x1x60x60xf32>

// CHECK-AGG-ON-LABEL: @QDQOptimizationAggressiveForConv
// CHECK-AGG-ON-SAME:      ([[INPUT:%.+]]: tensor<1x1x60x60xui8>) -> tensor<1x1x60x60xf32>
func.func @QDQOptimizationAggressiveForConv(%input: tensor<1x1x60x60xui8>) -> tensor<1x1x60x60xf32> {
    %convert = IE.Convert(%input) {dstElemType = f32} : tensor<1x1x60x60xui8> -> tensor<1x1x60x60xf32>

    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>

    %input_fq = IE.FakeQuantize(%convert, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 } :
        tensor<1x1x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x60x60xf32>

    %weights = const.Declare tensor<1x1x1x1xf32> = dense<128> : tensor<1x1x1x1xui8>, [#const.CastElemType<f32>]

    %weights_in_low = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    %weights_in_high = const.Declare tensor<1x1x1x1xf32> = dense<3.000000e+00> : tensor<1x1x1x1xf32>

    %weights_out_low = const.Declare tensor<1x1x1x1xf32> = dense<4.000000e+00> : tensor<1x1x1x1xf32>
    %weights_out_high = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>

    %weights_fq = IE.FakeQuantize(%weights, %weights_in_low, %weights_in_high, %weights_out_low, %weights_out_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 } :
        tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>

    %conv = IE.Convolution(%input_fq, %weights_fq)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x1x60x60xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x60x60xf32>

    %last_fq = IE.FakeQuantize(%conv, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 } :
        tensor<1x1x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x60x60xf32>

    return %last_fq : tensor<1x1x60x60xf32>

    /// Expected IR when enable-qdq-optimization-aggressive=false
    // CHECK-AGG-OFF: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x1x60x60xui8> -> tensor<1x1x60x60xf32>
    // CHECK-AGG-OFF: [[INPUT_LOW:%.+]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK-AGG-OFF: [[INPUT_HIGH:%.+]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-AGG-OFF: [[INPUT_FQ:%.+]] = IE.FakeQuantize([[CONVERT]], [[INPUT_LOW]], [[INPUT_HIGH]], [[INPUT_LOW]], [[INPUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x1x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x60x60xf32>
    // CHECK-AGG-OFF: [[WEIGHTS:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<128> : tensor<1x1x1x1xui8>, [#const.CastElemType<f32>]
    // CHECK-AGG-OFF: [[WEIGHTS_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-AGG-OFF: [[WEIGHTS_IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-AGG-OFF: [[WEIGHTS_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<4.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-AGG-OFF: [[WEIGHTS_OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-AGG-OFF: [[WEIGHTS_FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[WEIGHTS_IN_LOW]], [[WEIGHTS_IN_HIGH]], [[WEIGHTS_OUT_LOW]], [[WEIGHTS_OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
    // CHECK-AGG-OFF: [[CONV:%.+]] = IE.Convolution([[INPUT_FQ]], [[WEIGHTS_FQ]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x60x60xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x60x60xf32>
    // CHECK-AGG-OFF: [[LAST_FQ:%.+]] = IE.FakeQuantize([[CONV]], [[INPUT_LOW]], [[INPUT_HIGH]], [[INPUT_LOW]], [[INPUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x1x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x60x60xf32>
    // CHECK-AGG-OFF: return [[LAST_FQ]] : tensor<1x1x60x60xf32>

    /// Expected IR when enable-qdq-optimization-aggressive=true
    // CHECK-AGG-ON: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-AGG-ON: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x1x60x60xui8> -> tensor<1x1x60x60xf32>
    // CHECK-AGG-ON: [[CONV:%.+]] = IE.Convolution([[CONVERT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x60x60xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x60x60xf32>
    // CHECK-AGG-ON: return [[CONV]] : tensor<1x1x60x60xf32>
}
