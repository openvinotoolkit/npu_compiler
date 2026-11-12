//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

func.func @ReduceMeanInvalidPadding(%arg0: tensor<1x16x4x2xf16>) -> tensor<1x1x4x2xf16> {
    // expected-error@+2 {{'IE.ReduceMean' op inferred type(s) 'tensor<1x16x4x2xf16>' are incompatible with return type(s) of operation 'tensor<1x1x4x2xf16>'}}
    // expected-error@+1 {{'IE.ReduceMean' op failed to infer returned types}}
    %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims, output_padding = [0, 15, 0, 0]} : tensor<1x16x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>
}

// -----

func.func @ReduceMeanInvalidPaddingRank(%arg0: tensor<1x16x4x2xf16>) -> tensor<1x1x4x2xf16> {
    // expected-error@+1 {{Output padding [0, 15] incompatible with output type tensor<1x1x4x2xf16>}}
    %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims, output_padding = [0, 15]} : tensor<1x16x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>
}

// -----

func.func @ReduceSumInvalidPadding(%arg0: tensor<1x16x4x2xf16>) -> tensor<1x1x4x2xf16> {
    // expected-error@+2 {{'IE.ReduceSum' op inferred type(s) 'tensor<1x16x4x2xf16>' are incompatible with return type(s) of operation 'tensor<1x1x4x2xf16>'}}
    // expected-error@+1 {{'IE.ReduceSum' op failed to infer returned types}}
    %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims, output_padding = [0, 15, 0, 0]} : tensor<1x16x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>
}

// -----

func.func @ReduceSumInvalidPaddingRank(%arg0: tensor<1x16x4x2xf16>) -> tensor<1x1x4x2xf16> {
    // expected-error@+1 {{Output padding [0, 15] incompatible with output type tensor<1x1x4x2xf16>}}
    %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims, output_padding = [0, 15]} : tensor<1x16x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>
}
