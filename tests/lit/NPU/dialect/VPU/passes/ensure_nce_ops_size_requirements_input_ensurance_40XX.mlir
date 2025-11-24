//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ensure-nce-ops-size-requirements="enable-output-ensurance=false" --mlir-print-elementsattrs-with-hex-if-larger=-1 --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverICOnly
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x9728x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverICOnly(%arg0: tensor<1x9728x4x1xf16, {order = #NHWC}>) -> tensor<1x9728x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<9728x9728x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9728x9728x1x1xf16, {order = #NHWC}>
  %weights_table = const.Declare tensor<9728x1x1x4xsi32> = dense<1097072640> : tensor<9728x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [9728, 9728, 1, 1],
    strides = [1, 1]
  } : tensor<1x9728x4x1xf16, {order = #NHWC}>, tensor<9728x9728x1x1xf16, {order = #NHWC}>, tensor<9728x1x1x4xsi32> -> tensor<1x9728x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x9728x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:      [[FILTER0:%.+]] = const.Declare tensor<9728x4864x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9728x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 4864, 0, 0], [9728, 4864, 1, 1]>]
  // CHECK-DAG:      [[FILTER1:%.+]] = const.Declare tensor<9728x4864x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9728x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [9728, 4864, 1, 1]>]

  // CHECK-DAG:      [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<9728x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[9728, 608, 1065353216, 0]]], [[[19456, 1216, 1065353216, 0]]]
  // CHECK-DAG:      [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<9728x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 1097072640]]], [[[9728, 608, 1065353216, 1097072640]]], [[[19456, 1216, 1065353216, 1097072640]]]

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x4864x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER1]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [9728, 4864, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x9728x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 4864, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x4864x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER0]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [9728, 4864, 1, 1], strides = [1, 1]
  // CHECK-SAME: }
  // CHECK-SAME: -> tensor<1x9728x4x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                      quant_scale = [1.425000e+01], fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x9728x4x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1:%.+]] : tensor<1x9728x4x1xf16, {order = #NHWC}>
}
