//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-weights-to-unsigned --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU6010

!qElemTypeI8Act = !quant.uniform<i8:f16, 0.060064338235294119>
!qElemTypeI8Wt = !quant.uniform<i8:f32, 1.000000e+00:-1>

module @TestKeepSI8WeightsAsInputs {
  // CHECK:       !qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
  // CHECK:       !qElemType1 = !quant.uniform<i8:f32, 1.000000e+00:-1>

  // CHECK:      func.func @KeepSI8WeightsAsInputs
  // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x256x2048xf32>
  // CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<2048x2048xsi8>
  func.func @KeepSI8WeightsAsInputs(%arg0: tensor<1x256x2048xf32>, %arg1: tensor<2048x2048xsi8>) -> tensor<256x2048x1x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
    %2 = IE.Quantize(%1) {dstElemType = !qElemTypeI8Act} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemTypeI8Act>
    %3 = IE.QuantizeCast(%arg1) {dstElemType = !qElemTypeI8Wt} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemTypeI8Wt>
    %4 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048x!qElemTypeI8Act> -> tensor<256x2048x1x1x!qElemTypeI8Act>
    %5 = IE.AffineReshape(%3) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemTypeI8Wt> -> tensor<2048x2048x1x1x!qElemTypeI8Wt>
    %6 = IE.Convolution(%4, %5) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemTypeI8Act>, tensor<2048x2048x1x1x!qElemTypeI8Wt> -> tensor<256x2048x1x1xf16>

    return %6 : tensor<256x2048x1x1xf16>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[RESHAPE_0]]) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
    // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
    // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[QUANTIZE]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048x!qElemType> -> tensor<256x2048x1x1x!qElemType>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[QUANTIZE_CAST]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048x1x1x!qElemType1>
    // CHECK:       [[CONV_RESULT:%.+]] = IE.Convolution([[RESHAPE_1]], [[RESHAPE_2]])
    // CHECK-SAME:            {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<2048x2048x1x1x!qElemType1> -> tensor<256x2048x1x1xf16>

    // CHECK:       return [[CONV_RESULT]] : tensor<256x2048x1x1xf16>
  }
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195:-5>
module @TestKeepMixedPrecisionKeepI8ConstantsAsymmetric {
  // CHECK: !qElemType = !quant.uniform<i8:f16, 1.1534313725490195:-5>
  // CHECK-LABEL: @KeepMixedPrecisionKeepI8ConstantsAsymmetric
  // CHECK-SAME:      [[ARG0:%.+]]: tensor<1x16x16x16xf16>
  func.func @KeepMixedPrecisionKeepI8ConstantsAsymmetric(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %qweights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
    %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
    return %result : tensor<1x16x16x16xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
    // CHECK-SAME{LITERAL}: dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
    // CHECK: return [[CONV]]
  }
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.250000e-02:-10>

module @TestDoNotConvertAsymmetricI8ConstWeightsToU8WithDequantize {
  // CHECK: !qElemType = !quant.uniform<i8:f16, 1.250000e-02:-10>
  // CHECK-LABEL: @DoNotConvertAsymmetricI8ConstWeightsToU8WithDequantize
  // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x8x8xf16>
  func.func @DoNotConvertAsymmetricI8ConstWeightsToU8WithDequantize(%arg0: tensor<1x8x8x8xf16>) -> tensor<1x4x8x8xf16> {
    %weights = const.Declare tensor<4x8x1x1x!qElemType> = dense<1> : tensor<4x8x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>]
    %weights_dq = IE.Dequantize(%weights) {dstElemType = f16} : tensor<4x8x1x1x!qElemType> -> tensor<4x8x1x1xf16>
    %conv = IE.Convolution(%arg0, %weights_dq) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x8x8x8xf16>, tensor<4x8x1x1xf16> -> tensor<1x4x8x8xf16>
    return %conv : tensor<1x4x8x8xf16>

    // CHECK-DAG:      [[CST:%.+]] = const.Declare tensor<4x8x1x1x!qElemType>
    // CHECK-SAME{LITERAL}: dense<1> : tensor<4x8x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>]
    // CHECK:      [[WEIGHTS_DQ:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<4x8x1x1x!qElemType> -> tensor<4x8x1x1xf16>
    // CHECK:      [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS_DQ]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x8x8x8xf16>, tensor<4x8x1x1xf16> -> tensor<1x4x8x8xf16>
    // CHECK:      return [[CONV]] : tensor<1x4x8x8xf16>
  }
}

// -----

// Both activation (i8 per-tensor) and weights (i8 per-tensor) are signed.
// convert-weights-to-unsigned should keep both as i8 without introducing
// builtin.unrealized_conversion_cast.

!qElemTypeI8Act = !quant.uniform<i8:f16, 0.060054928648705576>
!qElemTypeI8Wt = !quant.uniform<i8:f32, 1.000000e+00>

module @TestKeepSignedI8WhenBothInputAndWeightsAreSigned {
  // CHECK:       !qElemType = !quant.uniform<i8:f16, 0.060054928648705576>
  // CHECK:       !qElemType1 = !quant.uniform<i8:f32, 1.000000e+00>

  // CHECK-LABEL: @KeepSignedI8WhenBothInputAndWeightsAreSigned
  // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x256x2048xf32>
  // CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<2048x2048xsi8>
  func.func @KeepSignedI8WhenBothInputAndWeightsAreSigned(%arg0: tensor<1x256x2048xf32>, %arg1: tensor<2048x2048xsi8>) -> tensor<256x2048x1x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
    %2 = IE.Quantize(%1) {dstElemType = !qElemTypeI8Act} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemTypeI8Act>
    %3 = IE.QuantizeCast(%arg1) {dstElemType = !qElemTypeI8Wt} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemTypeI8Wt>
    %4 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048x!qElemTypeI8Act> -> tensor<256x2048x1x1x!qElemTypeI8Act>
    %5 = IE.AffineReshape(%3) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemTypeI8Wt> -> tensor<2048x2048x1x1x!qElemTypeI8Wt>
    %6 = IE.Convolution(%4, %5) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemTypeI8Act>, tensor<2048x2048x1x1x!qElemTypeI8Wt> -> tensor<256x2048x1x1xf16>

    return %6 : tensor<256x2048x1x1xf16>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[RESHAPE_0]]) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
    // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
    // CHECK-NOT:   builtin.unrealized_conversion_cast
    // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType1} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType1>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[QUANTIZE]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048x!qElemType> -> tensor<256x2048x1x1x!qElemType>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[QUANTIZE_CAST]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048x!qElemType1> -> tensor<2048x2048x1x1x!qElemType1>
    // CHECK:       [[CONV_RESULT:%.+]] = IE.Convolution([[RESHAPE_1]], [[RESHAPE_2]])
    // CHECK-SAME:            {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1x!qElemType>, tensor<2048x2048x1x1x!qElemType1> -> tensor<256x2048x1x1xf16>

    // CHECK:       return [[CONV_RESULT]] : tensor<256x2048x1x1xf16>
  }
}

// -----
!qElemTypeI8Act = !quant.uniform<i8:f16, 0.060064338235294119>
!qElemTypeU2Wt = !quant.uniform<u2:f16, 1.000000e+00:2>
!qElemTypeU2LUTWt = !quant.uniform<!quant.quantile<ui2:si8, {-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00}>:f16, 1.000000e+00>

module @TestKeepSignedI8ActWhenWeightsAreU2LUT {
  // CHECK: !qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
  // CHECK: !qElemType1 = !quant.uniform<u2:f16, 1.000000e+00:2>
  // CHECK: !qElemType2 = !quant.uniform<!quant.quantile<ui2:si8, {-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00}>:f16, 1.000000e+00>

  // CHECK-LABEL: @KeepSignedI8ActWhenWeightsAreU2LUT
  // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x1536xf32>
  // CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<16x1536xui2>
  func.func @KeepSignedI8ActWhenWeightsAreU2LUT(%arg0: tensor<1x1x1536xf32>, %arg1: tensor<16x1536xui2>) -> tensor<1x16x1x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 1536]} : tensor<1x1x1536xf32> -> tensor<1x1x1x1536xf32>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x1x1536xf32> -> tensor<1x1x1x1536xf16>
    %2 = IE.Quantize(%1) {dstElemType = !qElemTypeI8Act} : tensor<1x1x1x1536xf16> -> tensor<1x1x1x1536x!qElemTypeI8Act>
    %3 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 1536]} : tensor<16x1536xui2> -> tensor<1x1x16x1536xui2>
    %4 = IE.QuantizeCast(%3) {dstElemType = !qElemTypeU2Wt} : tensor<1x1x16x1536xui2> -> tensor<1x1x16x1536x!qElemTypeU2Wt>
    %5 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 1536, 1, 1]} : tensor<1x1x1x1536x!qElemTypeI8Act> -> tensor<1x1536x1x1x!qElemTypeI8Act>
    %6 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [16, 1536, 1, 1]} : tensor<1x1x16x1536x!qElemTypeU2Wt> -> tensor<16x1536x1x1x!qElemTypeU2Wt>
    %7 = IE.QuantizeCast(%6) {dstElemType = !qElemTypeU2LUTWt} : tensor<16x1536x1x1x!qElemTypeU2Wt> -> tensor<16x1536x1x1x!qElemTypeU2LUTWt>
    %8 = IE.Convolution(%5, %7) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1536x1x1x!qElemTypeI8Act>, tensor<16x1536x1x1x!qElemTypeU2LUTWt> -> tensor<1x16x1x1xf16>
    return %8 : tensor<1x16x1x1xf16>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 1536]} : tensor<1x1x1536xf32> -> tensor<1x1x1x1536xf32>
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[RESHAPE_0]]) {dstElemType = f16} : tensor<1x1x1x1536xf32> -> tensor<1x1x1x1536xf16>
    // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x1x1x1536xf16> -> tensor<1x1x1x1536x!qElemType>
    // CHECK-NOT:   builtin.unrealized_conversion_cast
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[WEIGHTS]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 1536]} : tensor<16x1536xui2> -> tensor<1x1x16x1536xui2>
    // CHECK:       [[QCAST_0:%.+]] = IE.QuantizeCast([[RESHAPE_1]]) {dstElemType = !qElemType1} : tensor<1x1x16x1536xui2> -> tensor<1x1x16x1536x!qElemType1>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[QUANTIZE]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 1536, 1, 1]} : tensor<1x1x1x1536x!qElemType> -> tensor<1x1536x1x1x!qElemType>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.AffineReshape([[QCAST_0]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [16, 1536, 1, 1]} : tensor<1x1x16x1536x!qElemType1> -> tensor<16x1536x1x1x!qElemType1>
    // CHECK:       [[QCAST_1:%.+]] = IE.QuantizeCast([[RESHAPE_3]]) {dstElemType = !qElemType2} : tensor<16x1536x1x1x!qElemType1> -> tensor<16x1536x1x1x!qElemType2>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE_2]], [[QCAST_1]])
    // CHECK-SAME:            {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1536x1x1x!qElemType>, tensor<16x1536x1x1x!qElemType2> -> tensor<1x16x1x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x16x1x1xf16>
  }
}
