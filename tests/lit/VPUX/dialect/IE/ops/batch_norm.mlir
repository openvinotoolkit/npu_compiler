//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @BatchNormAttr
func @BatchNormAttr(%arg0: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf16> {
  %cst = const.Declare tensor<3xf16> = dense<[0.000000e+00, 4.169920e-01, 1.000000e+00]> : tensor<3xf16>
  %cst_0 = const.Declare tensor<3xf16> = dense<[0.000000e+00, 4.169920e-01, 1.000000e+00]> : tensor<3xf16>
  %cst_1 = const.Declare tensor<3xf16> = dense<[0.000000e+00, 4.169920e-01, 1.000000e+00]> : tensor<3xf16>
  %cst_2 = const.Declare tensor<3xf16> = dense<[7.826080e-05, 1.315430e+00, 7.554680e+00]> : tensor<3xf16>
  %0 = IE.BatchNormInference(%arg0, %cst, %cst_0, %cst_1, %cst_2) {eps = 1.000000e-03 : f64, operand_segment_sizes = dense<1> : vector<5xi32>} : tensor<1x3x256x256xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16> -> tensor<1x3x256x256xf16>
  return %0 : tensor<1x3x256x256xf16>

  //CHECK: [[VAL0:%.*]] = IE.BatchNormInference(%arg0) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], operand_segment_sizes = dense<[1, 0, 0, 0, 0]> : vector<5xi32>, variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16>
  //CHECK: return [[VAL0]]
}
