//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --low-precision %s | FileCheck %s
// REQUIRES: platform-NPU3720

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK:   !qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @QuantizedInterpolate
module @QuantizedInterpolate {

config.PipelineOptions @Options {
        config.Option @config.EnableSEPtrsOperations : true
}

// CHECK:   func.func @main
// CHECK:      ([[INPUT:%.+]]: tensor<1x16x10x10xui8>) -> tensor<1x16x20x20xf32>
func.func @main(%input: tensor<1x16x10x10xui8>) -> tensor<1x16x20x20xf32> {
    %0 = IE.Convert(%input) {dstElemType = f32} : tensor<1x16x10x10xui8> -> tensor<1x16x10x10xf32>

    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>

    %input_fq = IE.FakeQuantize(%0, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 } :
        tensor<1x16x10x10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x10x10xf32>

    %interp = IE.Interpolate(%input_fq)
        {
            attr = #IE.Interpolate<mode = <NEAREST>,
                                   shape_calc_mode = <SCALES>,
                                   coord_mode = <ASYMMETRIC>,
                                   nearest_mode = <FLOOR>,
                                   antialias = false,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
                                   axes_attr = [2, 3],
                                   operandSegmentSizes = array<i32: 1, 0, 0, 0>,
                                   scales_attr = [2.000000e+00, 2.000000e+00],
                                   sizes_attr = [20, 20]
        } :
        tensor<1x16x10x10xf32> -> tensor<1x16x20x20xf32>

    %last_fq = IE.FakeQuantize(%interp, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 } :
        tensor<1x16x20x20xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x20x20xf32>

    return %last_fq : tensor<1x16x20x20xf32>

    // CHECK:     [[INPUT_QUANT:%.+]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !qElemType} :
    // CHECK-SAME:     tensor<1x16x10x10xui8> -> tensor<1x16x10x10x!qElemType>

    // CHECK:     [[INTERP:%.+]] = IE.Interpolate([[INPUT_QUANT:%.+]])
    // CHECK-SAME:     tensor<1x16x10x10x!qElemType> -> tensor<1x16x20x20x!qElemType>

    // CHECK:     [[OUT_DEQ:%.+]] = IE.AvgPool([[INTERP]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} :
    // CHECK-SAME:     tensor<1x16x20x20x!qElemType> -> tensor<1x16x20x20xf32>

    // CHECK:     return [[OUT_DEQ]] : tensor<1x16x20x20xf32>
}

}
