//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% enable-weights-dynamic-dequantization=true" --run-initial-low-precision-transformations-rewriters="rewriter=consolidate-weights-dequantization" --mlir-print-elementsattrs-with-hex-if-larger -1 %s | FileCheck %s
// REQUIRES: platform-NPU5010


// CHECK-LABEL: @StaticScaleDequantizationF8E4M3
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xf8E4M3FN>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @StaticScaleDequantizationF8E4M3(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xf8E4M3FN>) -> tensor<1x4x28x28xf32> {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

    %convert = IE.Convert(%weights) {dstElemType = f32} : tensor<4x4x3x3xf8E4M3FN> -> tensor<4x4x3x3xf32>
    %multiply = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
    %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

    return %conv : tensor<1x4x28x28xf32>


    // CHECK-DAG:  [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK:  [[DEQUANT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]]) {dstElemType = f32, vpux.synthetic_dyn_dequant} : tensor<4x4x3x3xf8E4M3FN>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

    // CHECK: return [[CONV]]
}

// -----


// CHECK-LABEL: @StaticScaleShiftDequantizationF8E5M2
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xf8E5M2>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @StaticScaleShiftDequantizationF8E5M2(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xf8E5M2>) -> tensor<1x4x28x28xf16> {
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<0.5> : tensor<1x1x1x1xf16>
    %shift = const.Declare tensor<1x1x1x1xf16> = dense<10.0> : tensor<1x1x1x1xf16>

    %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xf8E5M2> -> tensor<4x4x3x3xf16>
    %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
    %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
    %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

    return %conv : tensor<1x4x28x28xf16>

    // CHECK:  [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e-01> : tensor<1x1x1x1xf16>
    // CHECK: [[ZP:%.+]] = const.Declare tensor<1x1x1x1xf8E5M2> = dense<1.000000e+01> : tensor<1x1x1x1xf8E5M2>
    // CHECK:  [[DEQUANT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f16, vpux.synthetic_dyn_dequant} : tensor<4x4x3x3xf8E5M2>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf8E5M2> -> tensor<4x4x3x3xf16>
    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

    // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @ConstScaleTableGatheredWeightsAlsoGatheredF8E4M3
// CHECK-SAME:      [[INPUT:%.+]]:          tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHT_INDICES:%.+]]: tensor<1xsi32>,
// CHECK-SAME:      [[SCALE_INDICES:%.+]]:  tensor<1xsi32>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @ConstScaleTableGatheredWeightsAlsoGatheredF8E4M3(%input: tensor<1x4x28x28xf32>, %weight_indices: tensor<1xsi32>, %scale_indices: tensor<1xsi32>) -> tensor<1x4x28x28xf32> {
    %weight_table = const.Declare tensor<2x4x4x3x3xf8E4M3FN> = dense<1.0> : tensor<2x4x4x3x3xf32>, [#const.CastElemType<f8E4M3FN>]
    %weights = IE.Gather(%weight_table, %weight_indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<2x4x4x3x3xf8E4M3FN>, tensor<1xsi32> -> tensor<4x4x3x3xf8E4M3FN>

    %scale_table = const.Declare tensor<2x1x1x1xf32> = dense<0.5> : tensor<2x1x1x1xf32>
    %scale = IE.Gather(%scale_table, %scale_indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<2x1x1x1xf32>, tensor<1xsi32> -> tensor<1x1x1x1xf32>

    %convert = IE.Convert(%weights) {dstElemType = f32} : tensor<4x4x3x3xf8E4M3FN> -> tensor<4x4x3x3xf32>
    %multiply = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
    %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

    return %conv : tensor<1x4x28x28xf32>

    // CHECK-DAG:  [[SCALE_TABLE:%.+]] = const.Declare tensor<2x1x1x1xf32> = dense<5.000000e-01> : tensor<2x1x1x1xf32>
    // CHECK-DAG:  [[WEIGHT_TABLE:%.+]] = const.Declare tensor<2x4x4x3x3xf8E4M3FN> = dense<1.000000e+00> : tensor<2x4x4x3x3xf32>, [#const.CastElemType<f8E4M3FN>]
    // CHECK:  [[WEIGHTS:%.+]] = IE.Gather([[WEIGHT_TABLE]], [[WEIGHT_INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 0 : i64} : tensor<2x4x4x3x3xf8E4M3FN>, tensor<1xsi32> -> tensor<4x4x3x3xf8E4M3FN>
    // CHECK:  [[SCALE:%.+]] = IE.Gather([[SCALE_TABLE]], [[SCALE_INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<2x1x1x1xf32>, tensor<1xsi32> -> tensor<1x1x1x1xf32>
    // CHECK:  [[DEQUANT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]]) {dstElemType = f32} : tensor<4x4x3x3xf8E4M3FN>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

    // CHECK: return [[CONV]]
}
