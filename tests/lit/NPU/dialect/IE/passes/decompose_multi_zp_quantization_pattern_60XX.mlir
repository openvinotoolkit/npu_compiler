//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --run-initial-low-precision-transformations-rewriters="rewriter=decompose-multi-zp-quantization" %s | FileCheck %s
// REQUIRES: platform-NPU6010

// AsymmetricPerChannelZP is enabled by default on NPU6010; this test ensures the
// weight dequantize pattern (Subtract + Multiply + AffineReshape) is kept intact.
module @Module {
  config.PipelineOptions @Options {
        config.Option @config.AsymmetricPerChannelZP : true
  }
  // CHECK-LABEL: @NotDecomposeWACStaticScaleStaticZP
  // CHECK-SAME:  ([[ACT:%.+]]: tensor<1x1x8192xf16>)
  func.func @NotDecomposeWACStaticScaleStaticZP(%input: tensor<1x1x8192xf16>) -> tensor<1x1x3072xf16> {
    %scale = const.Declare tensor<3072x128x1xf16> = dense<2.0> : tensor<3072x128x1xf16>
    %weights = const.Declare tensor<3072x128x64xf16> = dense<1> : tensor<3072x128x64xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f16>]
    %zeroPoint = const.Declare tensor<3072x128x1xf16> = dense<1> : tensor<3072x128x1xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f16>]
    %subtract = IE.Subtract(%weights, %zeroPoint) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3072x128x64xf16>, tensor<3072x128x1xf16> -> tensor<3072x128x64xf16>
    %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3072x128x64xf16>, tensor<3072x128x1xf16> -> tensor<3072x128x64xf16>
    %affine_reshape = IE.AffineReshape(%multiply) {dim_mapping = [[0], [1], [1]], shape_value = [3072, 8192]} : tensor<3072x128x64xf16> -> tensor<3072x8192xf16>
    %reshape_0 = IE.Reshape(%input) {shape_value = [1, 8192]} : tensor<1x1x8192xf16> -> tensor<1x8192xf16>
    %fully_connected = IE.FullyConnected(%reshape_0, %affine_reshape) : tensor<1x8192xf16>, tensor<3072x8192xf16> -> tensor<1x3072xf16>
    %reshape_1 = IE.Reshape(%fully_connected) {shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>
    return %reshape_1 : tensor<1x1x3072xf16>

    // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<3072x128x1xf16> = dense<2.000000e+00> : tensor<3072x128x1xf16>
    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<3072x128x64xf16> = dense<1> : tensor<3072x128x64xui2>,
    // CHECK-SAME:      [#const.ConvertElemType<ui8>, #const.CastElemType<f16>]
    // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<3072x128x1xf16> = dense<1> : tensor<3072x128x1xui2>,
    // CHECK-SAME:      [#const.ConvertElemType<ui8>, #const.CastElemType<f16>]
    // CHECK:     [[SUBTRACT:%.+]] = IE.Subtract([[WEIGHTS]], [[ZP]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<3072x128x64xf16>, tensor<3072x128x1xf16> -> tensor<3072x128x64xf16>
    // CHECK:     [[MULTIPLY:%.+]] = IE.Multiply([[SUBTRACT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<3072x128x64xf16>, tensor<3072x128x1xf16> -> tensor<3072x128x64xf16>
    // CHECK:     [[RESHAPE_0:%.+]] = IE.AffineReshape([[MULTIPLY]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [1]], shape_value = [3072, 8192]} : tensor<3072x128x64xf16> -> tensor<3072x8192xf16>
    // CHECK:     [[RESHAPE_1:%.+]] = IE.Reshape([[ACT]]) {shape_value = [1, 8192]} : tensor<1x1x8192xf16> -> tensor<1x8192xf16>
    // CHECK:     [[FC:%.+]] = IE.FullyConnected([[RESHAPE_1]], [[RESHAPE_0]]) : tensor<1x8192xf16>, tensor<3072x8192xf16> -> tensor<1x3072xf16>
    // CHECK:     [[RESHAPE_2:%.+]] = IE.Reshape([[FC]]) {shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>
    // CHECK:     return [[RESHAPE_2]] : tensor<1x1x3072xf16>
  }
}
