//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @AxisNotSingleElement
func.func @AxisNotSingleElement(%arg0: tensor<2x3x4xf16>, %arg1: tensor<1x3x1xf16>) -> tensor<2x3x4xf16> {
    %cst = const.Declare tensor<1x3x1xsi32> = dense<[[[1], [0], [1]]]> : tensor<1x3x1xsi32>
    %cst_0 = const.Declare tensor<2xsi32> = dense<[0, 1]> : tensor<2xsi32>
    // expected-error@+1 {{Axis should have only 1 element, while it has 2}}
    %0 = IE.ScatterElementsUpdate(%arg0, %cst, %arg1, %cst_0) {reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<2x3x4xf16>, tensor<1x3x1xsi32>, tensor<1x3x1xf16>, tensor<2xsi32> -> tensor<2x3x4xf16>
    return %0 : tensor<2x3x4xf16>
}

// -----

// CHECK-LABEL: @AmbiguousAxis
func.func @AmbiguousAxis(%arg0: tensor<2x3x4xf16>, %arg1: tensor<1x3x1xf16>) -> tensor<2x3x4xf16> {
    %cst = const.Declare tensor<1x3x1xsi32> = dense<[[[1], [0], [1]]]> : tensor<1x3x1xsi32>
    %cst_0 = const.Declare tensor<1xsi32> = dense<1> : tensor<si32>, [#const.Reshape<[1]>]
    // expected-error@+1 {{Ambiguous axis representation}}
    %0 = IE.ScatterElementsUpdate(%arg0, %cst, %arg1, %cst_0) {axis_value = 1 : i64, reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<2x3x4xf16>, tensor<1x3x1xsi32>, tensor<1x3x1xf16>, tensor<1xsi32> -> tensor<2x3x4xf16>
    return %0 : tensor<2x3x4xf16>
}

// -----

// CHECK-LABEL: @AxisNotProvided
func.func @AxisNotProvided(%arg0: tensor<2x3x4xf16>, %arg1: tensor<1x3x1xf16>) -> tensor<2x3x4xf16> {
    %cst = const.Declare tensor<1x3x1xsi32> = dense<[[[1], [0], [1]]]> : tensor<1x3x1xsi32>
    // expected-error@+1 {{Axis was not provided}}
    %0 = IE.ScatterElementsUpdate(%arg0, %cst, %arg1) {reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<2x3x4xf16>, tensor<1x3x1xsi32>, tensor<1x3x1xf16> -> tensor<2x3x4xf16>
    return %0 : tensor<2x3x4xf16>
}
