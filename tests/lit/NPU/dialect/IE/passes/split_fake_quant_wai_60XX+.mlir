//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --split-fake-quant %s | FileCheck %s
// REQUIRES: platform-NPU6010

module @TestSplitFakeQuantToI8ForSI8WeightsAsInputs {
  // CHECK: !qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
  // CHECK: !qElemType1 = !quant.uniform<i8:f32, 1.000000e+00>
  // CHECK-LABEL: @SplitFakeQuantToI8ForSI8WeightsAsInputs
  // CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x256x2048xf32>
  // CHECK-SAME:  [[INPUT1:%.+]]: tensor<2048x2048xsi8>
  func.func @SplitFakeQuantToI8ForSI8WeightsAsInputs(%arg0: tensor<1x256x2048xf32>, %arg1: tensor<2048x2048xsi8>) -> tensor<256x2048x1x1xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
    %2 = IE.FakeQuantize(%1, %cst_3, %cst_2, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x1x256x2048xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x2048xf16>
    %3 = IE.QuantizeCast(%arg1) {dstElemType = !quant.uniform<i8:f32, 1.000000e+00>} : tensor<2048x2048xsi8> -> tensor<2048x2048x!quant.uniform<i8:f32, 1.000000e+00>>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<2048x2048x!quant.uniform<i8:f32, 1.000000e+00>> -> tensor<2048x2048xf16>
    %5 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048xf16> -> tensor<256x2048x1x1xf16>
    %6 = IE.AffineReshape(%4) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
    %7 = IE.Convolution(%5, %6) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

    return %7 : tensor<256x2048x1x1xf16>

    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[RESHAPE0]]) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
    // CHECK: [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x1x256x2048x!qElemType> -> tensor<1x1x256x2048xf16>
    // CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[INPUT1]]) {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>
    // CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048xf16>
    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[DEQUANTIZE0]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048xf16> -> tensor<256x2048x1x1xf16>
    // CHECK: [[RESHAPE2:%.+]] = IE.AffineReshape([[DEQUANTIZE1]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[RESHAPE1]], [[RESHAPE2]])
    // CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

    // CHECK: return [[CONV]] : tensor<256x2048x1x1xf16>
  }
}

// -----

// Asymmetric ZP: input FakeQuantize range [-7.9375, 8.0] is not symmetric around zero.
// Signed conversion would produce non-zero ZP, so keep u8 with ZP=127.

module @TestSplitFakeQuantToU8ForAsymmetricZP {
  // CHECK: !qElemType = !quant.uniform<u8:f16, {{.*}}:127>
  // CHECK: !qElemType1 = !quant.uniform<i8:f32, 1.000000e+00>
  // CHECK-LABEL: @SplitFakeQuantToU8ForAsymmetricZP
  // CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x256x2048xf32>
  // CHECK-SAME:  [[INPUT1:%.+]]: tensor<2048x2048xsi8>
  func.func @SplitFakeQuantToU8ForAsymmetricZP(%arg0: tensor<1x256x2048xf32>, %arg1: tensor<2048x2048xsi8>) -> tensor<256x2048x1x1xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<8.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-7.937500e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<8.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-7.937500e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
    %2 = IE.FakeQuantize(%1, %cst_3, %cst_2, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x1x256x2048xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x2048xf16>
    %3 = IE.QuantizeCast(%arg1) {dstElemType = !quant.uniform<i8:f32, 1.000000e+00>} : tensor<2048x2048xsi8> -> tensor<2048x2048x!quant.uniform<i8:f32, 1.000000e+00>>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<2048x2048x!quant.uniform<i8:f32, 1.000000e+00>> -> tensor<2048x2048xf16>
    %5 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048xf16> -> tensor<256x2048x1x1xf16>
    %6 = IE.AffineReshape(%4) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
    %7 = IE.Convolution(%5, %6) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

    return %7 : tensor<256x2048x1x1xf16>

    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[RESHAPE0]]) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
    // CHECK: [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x1x256x2048x!qElemType> -> tensor<1x1x256x2048xf16>
    // CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[INPUT1]]) {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>
    // CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048xf16>
    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[DEQUANTIZE0]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048xf16> -> tensor<256x2048x1x1xf16>
    // CHECK: [[RESHAPE2:%.+]] = IE.AffineReshape([[DEQUANTIZE1]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[RESHAPE1]], [[RESHAPE2]])
    // CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>

    // CHECK: return [[CONV]] : tensor<256x2048x1x1xf16>
  }
}

// -----

!qU2Weights = !quant.uniform<u2:f16, 1.000000e+00:2>

module @TestSplitFakeQuantToI8ForU2WeightsAsInputsWithZP {

  // CHECK: !qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
  // CHECK: !qElemType1 = !quant.uniform<u2:f16, 1.000000e+00:2>
  // CHECK-LABEL: @SplitFakeQuantToI8ForU2WeightsAsInputsWithZP
  // CHECK-SAME:  [[ACT:%.+]]: tensor<1x1x1536xf32>
  // CHECK-SAME:  [[W_IN:%.+]]: tensor<12288x1536xui2>
  func.func @SplitFakeQuantToI8ForU2WeightsAsInputsWithZP(%arg0: tensor<1x1x1536xf32>, %arg1: tensor<12288x1536xui2>) -> tensor<1x12288x1x1xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 1536]} : tensor<1x1x1536xf32> -> tensor<1x1x1x1536xf32>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x1x1536xf32> -> tensor<1x1x1x1536xf16>
    %2 = IE.FakeQuantize(%1, %cst_3, %cst_2, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x1x1x1536xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1536xf16>

    %3 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 12288, 1536]} : tensor<12288x1536xui2> -> tensor<1x1x12288x1536xui2>
    %4 = IE.QuantizeCast(%3) {dstElemType = !qU2Weights} : tensor<1x1x12288x1536xui2> -> tensor<1x1x12288x1536x!qU2Weights>
    %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x1x12288x1536x!qU2Weights> -> tensor<1x1x12288x1536xf16>

    %6 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 1536, 1, 1]} : tensor<1x1x1x1536xf16> -> tensor<1x1536x1x1xf16>
    %7 = IE.AffineReshape(%5) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12288, 1536, 1, 1]} : tensor<1x1x12288x1536xf16> -> tensor<12288x1536x1x1xf16>
    %8 = IE.Convolution(%6, %7) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1536x1x1xf16>, tensor<12288x1536x1x1xf16> -> tensor<1x12288x1x1xf16>
    return %8 : tensor<1x12288x1x1xf16>

    // CHECK: [[A0:%.+]] = IE.AffineReshape([[ACT]])
    // CHECK: [[A1:%.+]] = IE.Convert([[A0]]) {dstElemType = f16}
    // CHECK: [[Q0:%.+]] = IE.Quantize([[A1]]) {dstElemType = !qElemType}
    // CHECK: [[DQ0:%.+]] = IE.Dequantize([[Q0]]) {dstElemType = f16}

    // CHECK: [[W0:%.+]] = IE.AffineReshape([[W_IN]])
    // CHECK: [[W1:%.+]] = IE.QuantizeCast([[W0]]) {dstElemType = !qElemType1}
    // CHECK: [[W2:%.+]] = IE.Dequantize([[W1]]) {dstElemType = f16}

    // CHECK: [[R0:%.+]] = IE.AffineReshape([[DQ0]])
    // CHECK: [[R1:%.+]] = IE.AffineReshape([[W2]])
    // CHECK: [[CONV:%.+]] = IE.Convolution([[R0]], [[R1]])
    // CHECK: return [[CONV]] : tensor<1x12288x1x1xf16>
  }
}
