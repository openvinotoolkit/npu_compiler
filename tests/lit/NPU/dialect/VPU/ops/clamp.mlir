//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=NPU3720" %s | FileCheck %s

module @test {
  // CHECK-LABEL: func.func @ClampU8
  func.func @ClampU8(%arg0: tensor<1x1x1x4xui8>) -> tensor<1x1x1x4xui8> {
    // CHECK: VPU.Clamp(%arg0) {{.*}} : tensor<1x1x1x4xui8> -> tensor<1x1x1x4xui8>
    %0 = VPU.Clamp(%arg0) {max = 255.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x1x1x4xui8> -> tensor<1x1x1x4xui8>
    return %0 : tensor<1x1x1x4xui8>
  }
}