//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SameLayoutModelTests(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x3x3xf16> {
// expected-error@+2 {{VPU::verifyOpLayout() failed for VPU.LeakyRelu}}
// expected-error@+1 {{Operation's input/output layout mismatch}}
    %0 = VPU.LeakyRelu(%arg0) {negative_slope = 1.500000e-01 : f64} : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x3x3xf16>

    return %0 : tensor<1x16x3x3xf16>
}

// -----
