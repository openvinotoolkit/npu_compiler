//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @BatchNormAttr
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x3x256x256xf16>)
func.func @BatchNormAttr(%arg0: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf16> {
  %0 = IE.BatchNormInference(%arg0) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16>
  return %0 : tensor<1x3x256x256xf16>

  //CHECK: [[VAL0:%.+]] = IE.BatchNormInference([[ARG_0]]) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16>
  //CHECK: return [[VAL0]]
}
