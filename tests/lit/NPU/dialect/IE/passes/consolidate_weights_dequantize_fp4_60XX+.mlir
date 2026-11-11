//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% enable-weights-dynamic-dequantization=true" --run-initial-low-precision-transformations-rewriters="rewriter=consolidate-weights-dequantization" --mlir-print-elementsattrs-with-hex-if-larger -1 %s | FileCheck %s
// REQUIRES: platform-NPU6010
// COM: FP4 is only supported on NPU60+ (via palletization), no need to run these tests on all platforms.


// CHECK-LABEL: @StaticScaleDequantizationF4E2M1FN
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xf4E2M1FN>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @StaticScaleDequantizationF4E2M1FN(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xf4E2M1FN>) -> tensor<1x4x28x28xf32> {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

    %convert = IE.Convert(%weights) {dstElemType = f32} : tensor<4x4x3x3xf4E2M1FN> -> tensor<4x4x3x3xf32>
    %multiply = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
    %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

    return %conv : tensor<1x4x28x28xf32>

    // CHECK:  [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK:  [[DEQUANT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]]) {dstElemType = f32, vpux.synthetic_dyn_dequant} : tensor<4x4x3x3xf4E2M1FN>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

    // CHECK: return [[CONV]]
}

// -----

// FP4 weights with a non-zero shift must NOT be consolidated: this storage type does not
// support zero-point shifts, so the Convert + Subtract + Multiply pattern must pass through
// unchanged.

// CHECK-LABEL: @StaticScaleShiftDequantizationF4E2M1FN_NoMatch
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xf4E2M1FN>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @StaticScaleShiftDequantizationF4E2M1FN_NoMatch(%input: tensor<1x4x28x28xf16>, %weights: tensor<4x4x3x3xf4E2M1FN>) -> tensor<1x4x28x28xf16> {
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<0.5> : tensor<1x1x1x1xf16>
    %shift = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>

    %convert = IE.Convert(%weights) {dstElemType = f16} : tensor<4x4x3x3xf4E2M1FN> -> tensor<4x4x3x3xf16>
    %subtract = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
    %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
    %conv = IE.Convolution(%input, %multiply) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

    return %conv : tensor<1x4x28x28xf16>

    // CHECK-DAG:  [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e-01>
    // CHECK-DAG:  [[SHIFT:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00>
    // CHECK:      [[CONVERT:%.+]] = IE.Convert([[WEIGHTS]])
    // CHECK-NOT:  IE.DynamicDequantize
    // CHECK:      [[SUBTRACT:%.+]] = IE.Subtract([[CONVERT]], [[SHIFT]])
    // CHECK:      [[MULTIPLY:%.+]] = IE.Multiply([[SUBTRACT]], [[SCALE]])
    // CHECK:      [[CONV:%.+]] = IE.Convolution([[INPUT]], [[MULTIPLY]])
    // CHECK:      return [[CONV]]
}
