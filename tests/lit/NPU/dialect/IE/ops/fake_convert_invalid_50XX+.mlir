//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics
// REQUIRES: arch-NPU50XX

// CHECK-LABEL: @UnsupportedInputElemType
func.func @UnsupportedInputElemType(%arg0: tensor<1x3x30x30xf64>) -> tensor<1x3x30x30xf32> {
    %scale = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.CastElemType<f16>]

    // expected-error@+1 {{'IE.FakeConvert' op operand #0 must be ranked tensor of 16-bit float or 32-bit float values, but got 'tensor<1x3x30x30xf64>'}}
    %0 = IE.FakeConvert(%arg0, %scale)
        { dst_type = f8E4M3FN } : tensor<1x3x30x30xf64>, tensor<1xf16> -> tensor<1x3x30x30xf64>

    return %0 : tensor<1x3x30x30xf64>
}

// -----

// CHECK-LABEL: @MissingScale
func.func @MissingScale(%arg0: tensor<1x3x30x30xf64>) -> tensor<1x3x30x30xf32> {
    // expected-error@+1 {{'IE.FakeConvert' op expected 2 or more operands, but found 1}}
    %0 = IE.FakeConvert(%arg0)
        { dst_type = f8E4M3FN } : tensor<1x3x30x30xf64> -> tensor<1x3x30x30xf64>

    return %0 : tensor<1x3x30x30xf64>
}

// -----

// CHECK-LABEL: @UnsupportedLowFpType
func.func @UnsupportedLowFpType(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %scale = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<f32>, [#const.Reshape<[1]>, #const.CastElemType<f16>]

    // expected-error@+1 {{Unsupported FakeConvert destination type f16}}
    %0 = IE.FakeConvert(%arg0, %scale)
        { dst_type = f16 } : tensor<1x3x30x30xf32>, tensor<1xf16> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>
}
