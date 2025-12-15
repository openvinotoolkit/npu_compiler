//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% enable-weights-dynamic-dequantization=true" --consolidate-weights-dequantize --mlir-print-elementsattrs-with-hex-if-larger -1 %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK:  !quant.uniform<f8E4M3FN:f32, 5.000000e-01>

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

    // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xf8E4M3FN> -> tensor<4x4x3x3x!qElemType>
    // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f32} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf32>
    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

    // CHECK: return [[CONV]]
}

// -----

// CHECK:  !qElemType = !quant.uniform<f8E5M2:f16, 5.000000e-01:100>

// CHECK-LABEL: @StaticScaleShiftDequantizationF8E5M2
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xf8E5M2>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @StaticScaleShiftDequantizationF8E5M2(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xf8E5M2>) -> tensor<1x4x28x28xf16> {
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<0.5> : tensor<1x1x1x1xf16>
    %shift = const.Declare tensor<1x1x1x1xf16> = dense<100.0> : tensor<1x1x1x1xf16>

    %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xf8E5M2> -> tensor<4x4x3x3xf16>
    %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
    %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
    %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

    return %conv : tensor<1x4x28x28xf16>

    // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<4x4x3x3xf8E5M2> -> tensor<4x4x3x3x!qElemType>
    // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f16} : tensor<4x4x3x3x!qElemType> -> tensor<4x4x3x3xf16>
    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

    // CHECK: return [[CONV]]
}
