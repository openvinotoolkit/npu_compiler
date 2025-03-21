//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @MinNotSingleElement
func.func @MinNotSingleElement(%arg0: tensor<1x200xf16>) -> tensor<1x200xf16> {
    %min = const.Declare tensor<2xf16> = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf16>
    %max = const.Declare tensor<f16> = dense<1.000000e+00> : tensor<f16>
    // expected-error@+1 {{Min should have only 1 element, while it has 2}}
    %0 = IE.RandomUniform(%min, %max) {global_seed = 0 : i64, op_seed = 11 : i64, outputType = f16, output_shape = [1, 200]} : tensor<2xf16>, tensor<f16> -> tensor<1x200xf16>
    return %0 : tensor<1x200xf16>
}

// -----

// CHECK-LABEL: @MaxNotSingleElement
func.func @MaxNotSingleElement(%arg0: tensor<1x200xf16>) -> tensor<1x200xf16> {
    %min = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
    %max = const.Declare tensor<2xf16> = dense<[2.000000e+00, 1.000000e+00]> : tensor<2xf16>
    // expected-error@+1 {{Max should have only 1 element, while it has 2}}
    %0 = IE.RandomUniform(%min, %max) {global_seed = 0 : i64, op_seed = 11 : i64, outputType = f16, output_shape = [1, 200]} : tensor<f16>, tensor<2xf16> -> tensor<1x200xf16>
    return %0 : tensor<1x200xf16>
}
