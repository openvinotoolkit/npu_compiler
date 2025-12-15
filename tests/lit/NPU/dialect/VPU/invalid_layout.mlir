//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SameLayoutModelTests(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x3x3xf16> {
// expected-error@+1 {{Operation's input/output layout mismatch}}
    %0 = VPU.LeakyRelu(%arg0) {negative_slope = 1.500000e-01 : f64} : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x3x3xf16>

    return %0 : tensor<1x16x3x3xf16>
}

// -----
