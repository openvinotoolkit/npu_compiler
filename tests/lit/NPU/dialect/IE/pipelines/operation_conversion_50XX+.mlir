//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --operation-conversion --mlir-elide-elementsattrs-if-larger 64 %s | FileCheck %s
// REQUIRES: platform-NPU5010

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

module @module {
config.PipelineOptions @Options {
    config.Option @config.SprLUTEnabled : true
}

// CHECK-LABEL: @FuseDynamicDequantizedFullyConnectedWithSlice
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1024x3072xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<16384x3072xsi4>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: tensor<16384x1xf16>
func.func @FuseDynamicDequantizedFullyConnectedWithSlice(%arg0: tensor<1024x3072xf16>, %arg1: tensor<16384x3072xsi4>, %arg2: tensor<16384x1xf16>) -> (tensor<1x1024x8192xf16>, tensor<1x1024x8192xf16>) {
  %0 = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<16384x3072xsi4>, tensor<16384x1xf16> -> tensor<16384x3072xf16>
  %1 = IE.FullyConnected(%arg0, %0) : tensor<1024x3072xf16>, tensor<16384x3072xf16> -> tensor<1024x16384xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1024, 16384]} : tensor<1024x16384xf16> -> tensor<1x1024x16384xf16>
  %3 = IE.Slice %2 [0, 0, 0] [1, 1024, 8192] : tensor<1x1024x16384xf16> to tensor<1x1024x8192xf16>
  %4 = IE.Swish(%3) {beta_value = 1.000000e+00 : f64} : tensor<1x1024x8192xf16> -> tensor<1x1024x8192xf16>
  %5 = IE.Slice %2 [0, 0, 8192] [1, 1024, 8192] : tensor<1x1024x16384xf16> to tensor<1x1024x8192xf16>
  return %4, %5 : tensor<1x1024x8192xf16>, tensor<1x1024x8192xf16>

  // CHECK:       [[QUANT_CAST:%.+]] = IE.QuantizeCast([[ARG_1]]) {dstElemType = !qElemType} : tensor<16384x3072xsi4> -> tensor<16384x3072x!qElemType>

  // CHECK:       [[SLICE_W0:%.+]] = IE.Slice [[QUANT_CAST]] [0, 0] [8192, 3072] : tensor<16384x3072x!qElemType> to tensor<8192x3072x!qElemType>
  // CHECK:       [[SLICE_S0:%.+]] = IE.Slice [[ARG_2]] [0, 0] [8192, 1] : tensor<16384x1xf16> to tensor<8192x1xf16>
  // CHECK:       [[SCALE_0:%.+]] = IE.Reshape([[SLICE_S0]]) {shape_value = [1, 8192]} : tensor<8192x1xf16> -> tensor<1x8192xf16>
  // CHECK:       [[DEQUANT_0:%.+]] = IE.Dequantize([[SLICE_W0]]) {dstElemType = f16} : tensor<8192x3072x!qElemType> -> tensor<8192x3072xf16>

  // CHECK:       [[SLICE_W1:%.+]] = IE.Slice [[QUANT_CAST]] [8192, 0] [8192, 3072] : tensor<16384x3072x!qElemType> to tensor<8192x3072x!qElemType>
  // CHECK:       [[SLICE_S1:%.+]] = IE.Slice [[ARG_2]] [8192, 0] [8192, 1] : tensor<16384x1xf16> to tensor<8192x1xf16>
  // CHECK:       [[SCALE_1:%.+]] = IE.Reshape([[SLICE_S1]]) {shape_value = [1, 8192]} : tensor<8192x1xf16> -> tensor<1x8192xf16>
  // CHECK:       [[DEQUANT_1:%.+]] = IE.Dequantize([[SLICE_W1]]) {dstElemType = f16} : tensor<8192x3072x!qElemType> -> tensor<8192x3072xf16>

  // CHECK:       [[IN_RESHAPE_0:%.+]] = IE.AffineReshape([[ARG_0]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2, 3]], shape_value = [1024, 3072, 1, 1]} : tensor<1024x3072xf16> -> tensor<1024x3072x1x1xf16>
  // CHECK:       [[W_RESHAPE_1:%.+]] = IE.AffineReshape([[DEQUANT_1]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2, 3]], shape_value = [8192, 3072, 1, 1]} : tensor<8192x3072xf16> -> tensor<8192x3072x1x1xf16>
  // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[IN_RESHAPE_0]], [[W_RESHAPE_1]])
  // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK-SAME:      : tensor<1024x3072x1x1xf16>, tensor<8192x3072x1x1xf16> -> tensor<1024x8192x1x1xf16>
  // CHECK:       [[OUT_RESHAPE_1:%.+]] = IE.AffineReshape([[CONV_1]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [1024, 8192]} : tensor<1024x8192x1x1xf16> -> tensor<1024x8192xf16>
  // CHECK:       [[SCALED_1:%.+]] = IE.Multiply([[OUT_RESHAPE_1]], [[SCALE_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x8192xf16>, tensor<1x8192xf16> -> tensor<1024x8192xf16>
  // CHECK:       [[FC_OUT_1:%.+]] = IE.AffineReshape([[SCALED_1]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2]], shape_value = [1, 1024, 8192]} : tensor<1024x8192xf16> -> tensor<1x1024x8192xf16>

  // CHECK:       [[IN_RESHAPE_1:%.+]] = IE.AffineReshape([[ARG_0]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2, 3]], shape_value = [1024, 3072, 1, 1]} : tensor<1024x3072xf16> -> tensor<1024x3072x1x1xf16>
  // CHECK:       [[W_RESHAPE_0:%.+]] = IE.AffineReshape([[DEQUANT_0]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2, 3]], shape_value = [8192, 3072, 1, 1]} : tensor<8192x3072xf16> -> tensor<8192x3072x1x1xf16>
  // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[IN_RESHAPE_1]], [[W_RESHAPE_0]])
  // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK-SAME:      : tensor<1024x3072x1x1xf16>, tensor<8192x3072x1x1xf16> -> tensor<1024x8192x1x1xf16>
  // CHECK:       [[OUT_RESHAPE_0:%.+]] = IE.AffineReshape([[CONV_0]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [1024, 8192]} : tensor<1024x8192x1x1xf16> -> tensor<1024x8192xf16>
  // CHECK:       [[SCALED_0:%.+]] = IE.Multiply([[OUT_RESHAPE_0]], [[SCALE_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x8192xf16>, tensor<1x8192xf16> -> tensor<1024x8192xf16>
  // CHECK:       [[FC_OUT_0:%.+]] = IE.AffineReshape([[SCALED_0]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2]], shape_value = [1, 1024, 8192]} : tensor<1024x8192xf16> -> tensor<1x1024x8192xf16>

  // CHECK:       [[SWISH:%.+]] = IE.Swish([[FC_OUT_0]]) {beta_value = 1.000000e+00 : f64} : tensor<1x1024x8192xf16> -> tensor<1x1024x8192xf16>

  // CHECK:       return [[SWISH]], [[FC_OUT_1]] : tensor<1x1024x8192xf16>, tensor<1x1024x8192xf16>
}
}
