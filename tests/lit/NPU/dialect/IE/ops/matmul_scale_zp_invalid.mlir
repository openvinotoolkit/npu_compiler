//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

func.func @MatMulZeroPointsTooWide(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<1x16x16x16x!qElemType>, %arg2: tensor<16x1x1x1xf32>, %arg3: tensor<16x1x1x1xsi32> ) -> tensor<1x16x16x16xf16> {
    // expected-error@+1 {{ZeroPoints element type must be narrower than 32 bits; si32/u32 and wider are not supported.}}
    %0 = IE.MatMul(%arg0, %arg1, , %arg2, %arg3) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xsi32> -> tensor<1x16x16x16xf16>
    return %0 : tensor<1x16x16x16xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

func.func @MatMulZeroPointsSignless(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<1x16x16x16x!qElemType>, %arg2: tensor<16x1x1x1xf32>, %arg3: tensor<16x1x1x1xi8>) -> tensor<1x16x16x16xf16> {
    // expected-error@+1 {{Type element must not be a signless integer; use signed (si) or unsigned (u).}}
    %0 = IE.MatMul(%arg0, %arg1, , %arg2, %arg3) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xi8> -> tensor<1x16x16x16xf16>
    return %0 : tensor<1x16x16x16xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

func.func @MatMulScaleZeroPointsShapeMismatch(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<1x16x16x16x!qElemType>, %arg2: tensor<16x1x1x1xf32>, %arg3: tensor<1x1x1x1xsi8>) -> tensor<1x16x16x16xf16> {
    // expected-error@+1 {{Scale and ZeroPoints must have the same shape, got [16, 1, 1, 1] and [1, 1, 1, 1]}}
    %0 = IE.MatMul(%arg0, %arg1, , %arg2, %arg3) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<16x1x1x1xf32>, tensor<1x1x1x1xsi8> -> tensor<1x16x16x16xf16>
    return %0 : tensor<1x16x16x16xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

func.func @MatMulScaleNotPerOutputChannelForm(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<1x16x16x16x!qElemType>, %arg2: tensor<1x16x16x1xf32>) -> tensor<1x16x16x16xf16> {
    // expected-error@+1 {{Scale and ZeroPoints must have the form [OC, 1, ..., 1] or [1, ..., 1], got [1, 16, 16, 1]}}
    %0 = IE.MatMul(%arg0, %arg1, , %arg2) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<1x16x16x1xf32> -> tensor<1x16x16x16xf16>
    return %0 : tensor<1x16x16x16xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

func.func @MatMulScaleWrongChannelCount(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<1x16x16x16x!qElemType>, %arg2: tensor<8x1x1x1xf32>) -> tensor<1x16x16x16xf16> {
    // expected-error@+1 {{Scale and ZeroPoints must hold a single value or one value per output channel (16), got 8}}
    %0 = IE.MatMul(%arg0, %arg1, , %arg2) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<8x1x1x1xf32> -> tensor<1x16x16x16xf16>
    return %0 : tensor<1x16x16x16xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// expected-error@+1 {{Expected 4D input, got rank 1}}
func.func @MatMulBiasNot4D(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<1x16x16x16x!qElemType>, %arg2: tensor<16xf16>) -> tensor<1x16x16x16xf16> {
    %0 = IE.MatMul(%arg0, %arg1, %arg2) {transpose_b} : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>, tensor<16xf16> -> tensor<1x16x16x16xf16>
    return %0 : tensor<1x16x16x16xf16>
}
