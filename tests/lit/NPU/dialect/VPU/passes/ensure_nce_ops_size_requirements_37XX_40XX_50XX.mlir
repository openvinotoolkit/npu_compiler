//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --ensure-nce-ops-size-requirements --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @DoNotSplitNCEDepthConvOverChannel
func.func @DoNotSplitNCEDepthConvOverChannel(%arg0: tensor<1x9216x4x1xf16, {order = #NHWC}>) -> tensor<1x9216x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<9216x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16x1x1xf16, {order = #NHWC}>
  %weights_table = const.Declare tensor<9216x1x1x4xsi32> = dense<10> : tensor<9216x1x1x4xsi32>
  %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %weights_table) rawFilterShape [9216, 1, 1, 1] {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEStub<>,
    
    strides = [1, 1]
  } -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK-NOT:  VPU.Slice
}
