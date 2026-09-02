//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-to-mixed-precision %s | FileCheck %s
// REQUIRES: platform-NPU5010

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.500000e+00>

// CHECK-LABEL: @MixedPrecisionMultiplyForDifferentScales
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x3x3xf16>
func.func @MixedPrecisionMultiplyForDifferentScales(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<1x16x3x3x!qElemType> = dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %3 = IE.Multiply(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x16x3x3x!qElemType> =
    // CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1} :
    // CHECK-SAME:  tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>

    // CHECK: [[DQUANT_LHS:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    // CHECK: [[DQUANT_RHS:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    // CHECK: [[MULT:%.+]] = IE.Multiply([[DQUANT_LHS]], [[DQUANT_RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:  tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    // CHECK: return [[MULT]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, -0.500000e+00>

// CHECK-LABEL: @MixedPrecisionMultiplyForDifferentNegativeScales
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x3x3xf16>
func.func @MixedPrecisionMultiplyForDifferentNegativeScales(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<1x16x3x3x!qElemType> = dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %3 = IE.Multiply(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x16x3x3x!qElemType> =
    // CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    // CHECK: [[DQUANT_LHS:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    // CHECK: [[DQUANT_RHS:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    // CHECK: [[MULT:%.+]] = IE.Multiply([[DQUANT_LHS]], [[DQUANT_RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:  tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    // CHECK: return [[MULT]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
//CHECK: !qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionFp16InputQuantBf8WeightsConv
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>)
func.func @MixedPrecisionFp16InputQuantBf8WeightsConv(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<!qElemType>, #const.CastElemType<f16>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>

  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<!QuantileType.quantile<ui4:f8E5M2, {-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0}>:f16, 1.000000e+00>
// CHECK: !qElemType = !quant.uniform<!QuantileType.quantile<ui4:f8E5M2, {-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionFp16InputBf8WeightsQuantileConv
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>)
func.func @MixedPrecisionFp16InputBf8WeightsQuantileConv(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<!qElemType>, #const.CastElemType<ui4>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>

  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<f8E5M2:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<f8E5M2:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionHf8InputQuantBf8WeightsConv
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>)
func.func @MixedPrecisionHf8InputQuantBf8WeightsConv(%arg0: tensor<1x16x16x16x!qElemType>) -> tensor<1x16x16x16x!qElemType> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType1> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<!qElemType1>, #const.CastElemType<f16>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16x!qElemType>

  return %result : tensor<1x16x16x16x!qElemType>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1>

  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1x!qElemType1> -> tensor<1x16x16x16x!qElemType>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<!QuantileType.quantile<ui4:f8E4M3FN, {0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0}>:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<!QuantileType.quantile<ui4:f8E4M3FN, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01}>:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionBf8InputQuantHf8WeightsQuantileConv
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x1x1xf16>)
func.func @MixedPrecisionBf8InputQuantHf8WeightsQuantileConv(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1x!qElemType> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  %weights = const.Declare tensor<16x16x1x1x!qElemType1> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType1>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>

  return %4 : tensor<1x16x1x1x!qElemType>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<16x16x1x1xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui4>, #const.CastElemType<!qElemType1>]

  //CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[QUANT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType>, tensor<16x16x1x1x!qElemType1> -> tensor<1x16x1x1x!qElemType>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<!QuantileType.quantile<ui4:f8E4M3FN, {0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0}>:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<!QuantileType.quantile<ui4:f8E4M3FN, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01}>:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionBf8InputHf8WeightsQuantileConv
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x1x1x!qElemType>)
func.func @MixedPrecisionBf8InputHf8WeightsQuantileConv(%arg0: tensor<1x16x1x1x!qElemType>) -> tensor<1x16x1x1x!qElemType> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType1> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType1>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  %4 = IE.Convolution(%arg0, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>

  return %4 : tensor<1x16x1x1x!qElemType>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<16x16x1x1xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui4>, #const.CastElemType<!qElemType1>]

  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType>, tensor<16x16x1x1x!qElemType1> -> tensor<1x16x1x1x!qElemType>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<i8:f16:0, {0.1085171568627451,0.0043868719362745098,0.0011484183517156863,0.0015251608455882353,-0.023115808823529413,0.0092486213235294118,-0.0024605545343137254,-5.5218864889705881E-4,0.022280943627450981,-0.0096047794117647065,-0.025742953431372548,0.01208639705882353,4.7164617800245099E-4,-0.022916666666666665,0.01014859068627451,-0.020687806372549019,1.1920928955078125E-7,-9.0451708026960788E-4,-0.0028301164215686274,0.020657169117647058,0.029725796568627449,0.0014466528799019608,0.12061887254901961,6.4505782781862744E-4,0.0023399203431372548,0.003393075980392157,0.0095818014705882359,0.013534007352941177,-0.010497089460784313,0.011251531862745098,0.025314031862745098,0.02688419117647059}>

// CHECK-LABEL: @MixedPrecisionForConvWithPostOpReluAndNegativeScales
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x448x448xf32>)
  func.func @MixedPrecisionForConvWithPostOpReluAndNegativeScales(%arg0: tensor<1x3x448x448xf32>) -> tensor<1x32x224x224xf32> {
    %cst = const.Declare tensor<1x32x1x1xf16> = dense<1.0> : tensor<1x32x1x1xf16>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<32x3x3x3x!qElemType> = dense<1> : tensor<32x3x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%cst_0) {dstElemType = f16} : tensor<32x3x3x3x!qElemType> -> tensor<32x3x3x3xf16>
    %1 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x448x448xf32> -> tensor<1x3x448x448xf16>
    %2 = IE.Convolution(%1, %0, %cst) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        post_op = #IE.Relu<>,
        strides = [2, 2]
    } : tensor<1x3x448x448xf16>, tensor<32x3x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x224x224xf16>
    %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x32x224x224xf16> -> tensor<1x32x224x224xf32>
    return %3 : tensor<1x32x224x224xf32>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>, [#const.CastElemType<f16>]
    // CHECK: [[CST0:%.+]] = const.Declare tensor<32x3x3x3x!qElemType> = dense<1> : tensor<32x3x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>]
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x3x448x448xf32> -> tensor<1x3x448x448xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[CONVERT]], [[CST0]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [2, 2]} : tensor<1x3x448x448xf16>, tensor<32x3x3x3x!qElemType>, tensor<1x32x1x1xf16> -> tensor<1x32x224x224xf16>
    // CHECK: [[OUT:%.+]] = IE.Convert([[CONV]]) {dstElemType = f32} : tensor<1x32x224x224xf16> -> tensor<1x32x224x224xf32>
    // CHECK: return [[OUT]] : tensor<1x32x224x224xf32>
  }

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607:128>
!qElemType1 = !quant.uniform<u8:f16:1, {0.01:128, 0.02:128, 0.03:128, 0.04:128}>
!qElemType2 = !quant.uniform<u8:f16:0, {0.05:128, 0.06:128, 0.07:128, 0.08:128}>

// CHECK-LABEL: @ConvWithSwishPostOpAbsorbDequantize
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x3x224x224x!qElemType>
func.func @ConvWithSwishPostOpAbsorbDequantize(%arg0: tensor<1x3x224x224x!qElemType>) -> tensor<1x4x112x112x!qElemType1> {
    %cst_filter = const.Declare tensor<4x3x3x3x!qElemType2> = dense<1> : tensor<4x3x3x3xui8>, [#const.CastElemType<!qElemType2>]

    %dequant_act = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x224x224x!qElemType> -> tensor<1x3x224x224xf16>
    %dequant_filter = IE.Dequantize(%cst_filter) {dstElemType = f16} : tensor<4x3x3x3x!qElemType2> -> tensor<4x3x3x3xf16>
    %conv = IE.Convolution(%dequant_act, %dequant_filter) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], post_op = #IE.Swish<beta = 1.000000e+00 : f64>, strides = [2, 2]
    } : tensor<1x3x224x224xf16>, tensor<4x3x3x3xf16> -> tensor<1x4x112x112xf16>
    %quant = IE.Quantize(%conv) {dstElemType = !qElemType1} : tensor<1x4x112x112xf16> -> tensor<1x4x112x112x!qElemType1>

    return %quant : tensor<1x4x112x112x!qElemType1>

    // CHECK-DAG: [[CST_FILTER:%.+]] = const.Declare tensor<4x3x3x3x!qElemType2>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST_FILTER]])
    // CHECK-SAME: dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], post_op = #IE.Swish<beta = 1.000000e+00 : f64>, strides = [2, 2]
    // CHECK-SAME: tensor<1x3x224x224x!qElemType>, tensor<4x3x3x3x!qElemType2> -> tensor<1x4x112x112xf16>

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[CONV]]) {dstElemType = !qElemType1}
    // CHECK-SAME: tensor<1x4x112x112xf16> -> tensor<1x4x112x112x!qElemType1>

    // CHECK: return [[QUANT]] : tensor<1x4x112x112x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607:128>

// CHECK-LABEL: @ConvWithSwishPostOpFloatInputNotMatched
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x3x224x224xf16>
func.func @ConvWithSwishPostOpFloatInputNotMatched(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x112x112x!qElemType> {
    %cst_filter = const.Declare tensor<4x3x3x3xf16> = dense<1.0> : tensor<4x3x3x3xf16>

    %conv = IE.Convolution(%arg0, %cst_filter) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], post_op = #IE.Swish<beta = 1.000000e+00 : f64>, strides = [2, 2]
    } : tensor<1x3x224x224xf16>, tensor<4x3x3x3xf16> -> tensor<1x4x112x112xf16>
    %quant = IE.Quantize(%conv) {dstElemType = !qElemType} : tensor<1x4x112x112xf16> -> tensor<1x4x112x112x!qElemType>

    return %quant : tensor<1x4x112x112x!qElemType>

    // CHECK-DAG: [[CST_FILTER:%.+]] = const.Declare tensor<4x3x3x3xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST_FILTER]])
    // CHECK-SAME: post_op = #IE.Swish
    // CHECK-SAME: tensor<1x3x224x224xf16>, tensor<4x3x3x3xf16> -> tensor<1x4x112x112x!qElemType>
    // CHECK: return [[CONV]] : tensor<1x4x112x112x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607:128>
!qElemType1 = !quant.uniform<u8:f16:1, {0.01:128, 0.02:128, 0.03:128, 0.04:128}>
!qElemType2 = !quant.uniform<u8:f16:0, {0.05:128, 0.06:128, 0.07:128, 0.08:128}>

// CHECK-LABEL: @ConvWithSigmoidPostOpAbsorbDequantize
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x3x224x224x!qElemType>
func.func @ConvWithSigmoidPostOpAbsorbDequantize(%arg0: tensor<1x3x224x224x!qElemType>) -> tensor<1x4x112x112x!qElemType1> {
    %cst_filter = const.Declare tensor<4x3x3x3x!qElemType2> = dense<1> : tensor<4x3x3x3xui8>, [#const.CastElemType<!qElemType2>]

    %dequant_act = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x224x224x!qElemType> -> tensor<1x3x224x224xf16>
    %dequant_filter = IE.Dequantize(%cst_filter) {dstElemType = f16} : tensor<4x3x3x3x!qElemType2> -> tensor<4x3x3x3xf16>
    %conv = IE.Convolution(%dequant_act, %dequant_filter) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], post_op = #IE.Sigmoid<>, strides = [2, 2]
    } : tensor<1x3x224x224xf16>, tensor<4x3x3x3xf16> -> tensor<1x4x112x112xf16>
    %quant = IE.Quantize(%conv) {dstElemType = !qElemType1} : tensor<1x4x112x112xf16> -> tensor<1x4x112x112x!qElemType1>

    return %quant : tensor<1x4x112x112x!qElemType1>

    // CHECK-DAG: [[CST_FILTER:%.+]] = const.Declare tensor<4x3x3x3x!qElemType2>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST_FILTER]])
    // CHECK-SAME: post_op = #IE.Sigmoid
    // CHECK-SAME: tensor<1x3x224x224x!qElemType>, tensor<4x3x3x3x!qElemType2> -> tensor<1x4x112x112xf16>
    // CHECK: [[QUANT:%.+]] = IE.Quantize([[CONV]]) {dstElemType = !qElemType1}
    // CHECK: return [[QUANT]] : tensor<1x4x112x112x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607:128>
!qElemType1 = !quant.uniform<u8:f16:1, {0.01:128, 0.02:128, 0.03:128, 0.04:128}>
!qElemType2 = !quant.uniform<u8:f16:0, {0.05:128, 0.06:128, 0.07:128, 0.08:128}>

// CHECK-LABEL: @ConvWithGeluPostOpAbsorbDequantize
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x3x224x224x!qElemType>
func.func @ConvWithGeluPostOpAbsorbDequantize(%arg0: tensor<1x3x224x224x!qElemType>) -> tensor<1x4x112x112x!qElemType1> {
    %cst_filter = const.Declare tensor<4x3x3x3x!qElemType2> = dense<1> : tensor<4x3x3x3xui8>, [#const.CastElemType<!qElemType2>]

    %dequant_act = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x224x224x!qElemType> -> tensor<1x3x224x224xf16>
    %dequant_filter = IE.Dequantize(%cst_filter) {dstElemType = f16} : tensor<4x3x3x3x!qElemType2> -> tensor<4x3x3x3xf16>
    %conv = IE.Convolution(%dequant_act, %dequant_filter) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], post_op = #IE.Gelu<>, strides = [2, 2]
    } : tensor<1x3x224x224xf16>, tensor<4x3x3x3xf16> -> tensor<1x4x112x112xf16>
    %quant = IE.Quantize(%conv) {dstElemType = !qElemType1} : tensor<1x4x112x112xf16> -> tensor<1x4x112x112x!qElemType1>

    return %quant : tensor<1x4x112x112x!qElemType1>

    // CHECK-DAG: [[CST_FILTER:%.+]] = const.Declare tensor<4x3x3x3x!qElemType2>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST_FILTER]])
    // CHECK-SAME: post_op = #IE.Gelu
    // CHECK-SAME: tensor<1x3x224x224x!qElemType>, tensor<4x3x3x3x!qElemType2> -> tensor<1x4x112x112xf16>
    // CHECK: [[QUANT:%.+]] = IE.Quantize([[CONV]]) {dstElemType = !qElemType1}
    // CHECK: return [[QUANT]] : tensor<1x4x112x112x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607:128>
!qElemType1 = !quant.uniform<u8:f16:1, {0.01:128, 0.02:128, 0.03:128, 0.04:128}>
!qElemType2 = !quant.uniform<u8:f16:0, {0.05:128, 0.06:128, 0.07:128, 0.08:128}>

// CHECK-LABEL: @ConvWithSwishPostOpAndBias
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x3x224x224x!qElemType>
func.func @ConvWithSwishPostOpAndBias(%arg0: tensor<1x3x224x224x!qElemType>) -> tensor<1x4x112x112x!qElemType1> {
    %cst_filter = const.Declare tensor<4x3x3x3x!qElemType2> = dense<1> : tensor<4x3x3x3xui8>, [#const.CastElemType<!qElemType2>]
    %cst_bias = const.Declare tensor<1x4x1x1xf16> = dense<0.5> : tensor<1x4x1x1xf16>

    %dequant_act = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x224x224x!qElemType> -> tensor<1x3x224x224xf16>
    %dequant_filter = IE.Dequantize(%cst_filter) {dstElemType = f16} : tensor<4x3x3x3x!qElemType2> -> tensor<4x3x3x3xf16>
    %conv = IE.Convolution(%dequant_act, %dequant_filter, %cst_bias) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], post_op = #IE.Swish<beta = 1.000000e+00 : f64>, strides = [2, 2]
    } : tensor<1x3x224x224xf16>, tensor<4x3x3x3xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x112x112xf16>
    %quant = IE.Quantize(%conv) {dstElemType = !qElemType1} : tensor<1x4x112x112xf16> -> tensor<1x4x112x112x!qElemType1>

    return %quant : tensor<1x4x112x112x!qElemType1>

    // CHECK-DAG: [[CST_FILTER:%.+]] = const.Declare tensor<4x3x3x3x!qElemType2>
    // CHECK-DAG: [[CST_BIAS:%.+]] = const.Declare tensor<1x4x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST_FILTER]], [[CST_BIAS]])
    // CHECK-SAME: post_op = #IE.Swish<beta = 1.000000e+00 : f64>
    // CHECK-SAME: tensor<1x3x224x224x!qElemType>, tensor<4x3x3x3x!qElemType2>, tensor<1x4x1x1xf16> -> tensor<1x4x112x112xf16>
    // CHECK: [[QUANT:%.+]] = IE.Quantize([[CONV]]) {dstElemType = !qElemType1}
    // CHECK: return [[QUANT]] : tensor<1x4x112x112x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @ConvertConvWithLeakyRelu
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @ConvertConvWithLeakyRelu(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %1 = IE.Quantize(%arg0) {
      dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %2 = IE.Dequantize(%1) {
      dstElemType = f16
    } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %WEIGHTS = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [
        #const.CastElemType<ui8>, #const.CastElemType<!qElemType>
    ]

    %3 = IE.Dequantize(%WEIGHTS) {
        dstElemType = f16
    } : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>

    %4 = IE.Convolution(%2, %3) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1],
        post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>
    } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    return %4 : tensor<1x16x3x3xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<16x16x1x1xf16>, [
    // CHECK-SAME:      #const.CastElemType<ui8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK-SAME:  ]

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG_0]]) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK-NOT: IE.Dequantize([[QUANT]])

    // CHECK: [[CONV:%.+]] = IE.Convolution([[QUANT]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x3x3xf16>

    // CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @ConvertConvWithLeakyReluConsumer
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @ConvertConvWithLeakyReluConsumer(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %1 = IE.Quantize(%arg0) {
      dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %2 = IE.Dequantize(%1) {
      dstElemType = f16
    } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %WEIGHTS = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [
        #const.CastElemType<ui8>, #const.CastElemType<!qElemType>
    ]

    %3 = IE.Dequantize(%WEIGHTS) {
        dstElemType = f16
    } : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>

    %4 = IE.Convolution(%2, %3) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    %5 = IE.LeakyRelu(%4) {negative_slope = 2.500000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %5 : tensor<1x16x3x3xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<16x16x1x1xf16>, [
    // CHECK-SAME:      #const.CastElemType<ui8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK-SAME:  ]

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG_0]]) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK-NOT: IE.Dequantize([[QUANT]])

    // CHECK: [[CONV:%.+]] = IE.Convolution([[QUANT]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x3x3xf16>

    // CHECK: [[LEAKY:%.+]] = IE.LeakyRelu([[CONV]]) {negative_slope = 2.500000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    // CHECK: return [[LEAKY]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.28679281496534159>

// CHECK-LABEL: @FuseQuantizeIntoMultiply
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x4x4xf16>
// CHECK-SAME:     [[ARG_1:%[^:]+]]: tensor<1x16x4x4xf16>
func.func @FuseQuantizeIntoMultiply(%arg0: tensor<1x16x4x4xf16>, %arg1: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4x!qElemType> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x16x4x4xf16> -> tensor<1x16x4x4x!qElemType>
    return %1 : tensor<1x16x4x4x!qElemType>

    // CHECK-NOT: IE.Quantize
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK-SAME:     : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4x!qElemType>
    // CHECK: return [[MULTIPLY]]
}

// -----

!qElemType = !quant.uniform<i8:f16:1, {0.10000000000000001,0.20000000000000001,0.30000000000000004}>

// CHECK-LABEL: @DoNotFuseQuantizeIntoMultiplyPerAxis
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x3x4x4xf16>
// CHECK-SAME:     [[ARG_1:%[^:]+]]: tensor<1x3x4x4xf16>
func.func @DoNotFuseQuantizeIntoMultiplyPerAxis(%arg0: tensor<1x3x4x4xf16>, %arg1: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4x!qElemType> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x3x4x4xf16>, tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4x!qElemType>
    return %1 : tensor<1x3x4x4x!qElemType>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG_0]], [[ARG_1]])
    // CHECK-SAME:     : tensor<1x3x4x4xf16>, tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf16>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[MULTIPLY]]) {dstElemType = !qElemType}
    // CHECK: return [[QUANTIZE]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.28679281496534159>
!qElemType1 = !quant.uniform<u8:f16, 0.081176862529679844:128>

// CHECK-LABEL: @DoNotFuseQuantizeIntoMultiplyMultipleConsumers
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x4x4xf16>
// CHECK-SAME:     [[ARG_1:%[^:]+]]: tensor<1x16x4x4xf16>
func.func @DoNotFuseQuantizeIntoMultiplyMultipleConsumers(%arg0: tensor<1x16x4x4xf16>, %arg1: tensor<1x16x4x4xf16>) -> (tensor<1x16x4x4x!qElemType>, tensor<1x16x4x4x!qElemType1>) {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x16x4x4xf16> -> tensor<1x16x4x4x!qElemType>
    %2 = IE.Quantize(%0) {dstElemType = !qElemType1} : tensor<1x16x4x4xf16> -> tensor<1x16x4x4x!qElemType1>
    return %1, %2 : tensor<1x16x4x4x!qElemType>, tensor<1x16x4x4x!qElemType1>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG_0]], [[ARG_1]])
    // CHECK-SAME:     : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    // CHECK: [[Q1:%.+]] = IE.Quantize([[MULTIPLY]]) {dstElemType = !qElemType}
    // CHECK: [[Q2:%.+]] = IE.Quantize([[MULTIPLY]]) {dstElemType = !qElemType1}
    // CHECK: return [[Q1]], [[Q2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0013184806879828958:128>

// CHECK-LABEL: @DoNotFuseQuantizeIntoMultiplyBroadcastInputs
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x1x4x4xf16>
// CHECK-SAME:     [[ARG_1:%[^:]+]]: tensor<1x4x4x4xf16>
func.func @DoNotFuseQuantizeIntoMultiplyBroadcastInputs(%arg0: tensor<1x1x4x4xf16>,
                                                        %arg1: tensor<1x4x4x4xf16>) -> tensor<1x4x4x4x!qElemType> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x1x4x4xf16>, tensor<1x4x4x4xf16> -> tensor<1x4x4x4xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x4x4x4xf16> -> tensor<1x4x4x4x!qElemType>
    return %1 : tensor<1x4x4x4x!qElemType>

    // Fusion must be skipped: inputs have different shapes (broadcast over C dim).
    // IE.Multiply and IE.Quantize must remain as separate ops.
    // CHECK-NOT: IE.Multiply{{.*}}-> tensor<1x4x4x4x!qElemType>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:     : tensor<1x1x4x4xf16>, tensor<1x4x4x4xf16> -> tensor<1x4x4x4xf16>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[MULTIPLY]]) {dstElemType = !qElemType}
    // CHECK-SAME:     : tensor<1x4x4x4xf16> -> tensor<1x4x4x4x!qElemType>
    // CHECK: return [[QUANTIZE]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.250000e-01:123>

// CHECK-LABEL: @RejectFusionOnConvSoftmaxViewConvQuantize
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x128x32x32xf16>)
func.func @RejectFusionOnConvSoftmaxViewConvQuantize(%arg0: tensor<1x128x32x32xf16>) -> tensor<1x8x1024x1x!qElemType> {
    %weights0 = const.Declare tensor<128x128x1x1xf16> = dense<1.0> : tensor<128x128x1x1xf16>
    %weights1 = const.Declare tensor<8x128x1x1xf16> = dense<1.0> : tensor<8x128x1x1xf16>

    %conv0 = IE.Convolution(%arg0, %weights0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x32x32xf16>, tensor<128x128x1x1xf16> -> tensor<1x128x32x32xf16>
    %softmax = IE.SoftMax(%conv0) {axisInd = 1 : i64} : tensor<1x128x32x32xf16> -> tensor<1x128x32x32xf16>
    %reshape = IE.Reshape(%softmax) {shape_value = [1, 128, 1024, 1]} : tensor<1x128x32x32xf16> -> tensor<1x128x1024x1xf16>
    %conv1 = IE.Convolution(%reshape, %weights1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1024x1xf16>, tensor<8x128x1x1xf16> -> tensor<1x8x1024x1xf16>
    %quantize = IE.Quantize(%conv1) {dstElemType = !qElemType} : tensor<1x8x1024x1xf16> -> tensor<1x8x1024x1x!qElemType>

    return %quantize : tensor<1x8x1024x1x!qElemType>

    // CHECK-DAG: [[WEIGHTS0:%.+]] = const.Declare tensor<128x128x1x1xf16>
    // CHECK-DAG: [[WEIGHTS1:%.+]] = const.Declare tensor<8x128x1x1xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[ARG0]], [[WEIGHTS0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x32x32xf16>, tensor<128x128x1x1xf16> -> tensor<1x128x32x32xf16>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[CONV0]]) {axisInd = 1 : i64} : tensor<1x128x32x32xf16> -> tensor<1x128x32x32xf16>
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[SOFTMAX]]) {shape_value = [1, 128, 1024, 1]} : tensor<1x128x32x32xf16> -> tensor<1x128x1024x1xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[RESHAPE]], [[WEIGHTS1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x1024x1xf16>, tensor<8x128x1x1xf16> -> tensor<1x8x1024x1xf16>
    // CHECK: [[QUANT:%.+]] = IE.Quantize([[CONV1]]) {dstElemType = !qElemType} : tensor<1x8x1024x1xf16> -> tensor<1x8x1024x1x!qElemType>
    // CHECK: return [[QUANT]] : tensor<1x8x1024x1x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.250000e-01:123>

// CHECK-LABEL: @RejectFusionOnConvSoftmaxConvQuantize
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x128x32x32xf16>)
func.func @RejectFusionOnConvSoftmaxConvQuantize(%arg0: tensor<1x128x32x32xf16>) -> tensor<1x8x32x32x!qElemType> {
    %weights0 = const.Declare tensor<128x128x1x1xf16> = dense<1.0> : tensor<128x128x1x1xf16>
    %weights1 = const.Declare tensor<8x128x1x1xf16> = dense<1.0> : tensor<8x128x1x1xf16>

    %conv0 = IE.Convolution(%arg0, %weights0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x32x32xf16>, tensor<128x128x1x1xf16> -> tensor<1x128x32x32xf16>
    %softmax = IE.SoftMax(%conv0) {axisInd = 1 : i64} : tensor<1x128x32x32xf16> -> tensor<1x128x32x32xf16>
    %conv1 = IE.Convolution(%softmax, %weights1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x32x32xf16>, tensor<8x128x1x1xf16> -> tensor<1x8x32x32xf16>
    %quantize = IE.Quantize(%conv1) {dstElemType = !qElemType} : tensor<1x8x32x32xf16> -> tensor<1x8x32x32x!qElemType>

    return %quantize : tensor<1x8x32x32x!qElemType>

    // CHECK-DAG: [[WEIGHTS0:%.+]] = const.Declare tensor<128x128x1x1xf16>
    // CHECK-DAG: [[WEIGHTS1:%.+]] = const.Declare tensor<8x128x1x1xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[ARG0]], [[WEIGHTS0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x32x32xf16>, tensor<128x128x1x1xf16> -> tensor<1x128x32x32xf16>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[CONV0]]) {axisInd = 1 : i64} : tensor<1x128x32x32xf16> -> tensor<1x128x32x32xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SOFTMAX]], [[WEIGHTS1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x32x32xf16>, tensor<8x128x1x1xf16> -> tensor<1x8x32x32xf16>
    // CHECK: [[QUANT:%.+]] = IE.Quantize([[CONV1]]) {dstElemType = !qElemType} : tensor<1x8x32x32xf16> -> tensor<1x8x32x32x!qElemType>
    // CHECK: return [[QUANT]] : tensor<1x8x32x32x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.250000e-01:123>

// CHECK-LABEL: @DoNotRejectFusionOnConvSoftmaxConvQuantizeSmallSoftmax
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x64x16x16xf16>)
func.func @DoNotRejectFusionOnConvSoftmaxConvQuantizeSmallSoftmax(%arg0: tensor<1x64x16x16xf16>) -> tensor<1x8x16x16x!qElemType> {
    %weights0 = const.Declare tensor<64x64x1x1xf16> = dense<1.0> : tensor<64x64x1x1xf16>
    %weights1 = const.Declare tensor<8x64x1x1xf16> = dense<1.0> : tensor<8x64x1x1xf16>

    %conv0 = IE.Convolution(%arg0, %weights0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x16x16xf16>, tensor<64x64x1x1xf16> -> tensor<1x64x16x16xf16>
    %softmax = IE.SoftMax(%conv0) {axisInd = 1 : i64} : tensor<1x64x16x16xf16> -> tensor<1x64x16x16xf16>
    %conv1 = IE.Convolution(%softmax, %weights1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x16x16xf16>, tensor<8x64x1x1xf16> -> tensor<1x8x16x16xf16>
    %quantize = IE.Quantize(%conv1) {dstElemType = !qElemType} : tensor<1x8x16x16xf16> -> tensor<1x8x16x16x!qElemType>

    return %quantize : tensor<1x8x16x16x!qElemType>

    // CHECK-DAG: [[WEIGHTS0:%.+]] = const.Declare tensor<64x64x1x1xf16>
    // CHECK-DAG: [[WEIGHTS1:%.+]] = const.Declare tensor<8x64x1x1xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[ARG0]], [[WEIGHTS0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x16x16xf16>, tensor<64x64x1x1xf16> -> tensor<1x64x16x16xf16>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[CONV0]]) {axisInd = 1 : i64} : tensor<1x64x16x16xf16> -> tensor<1x64x16x16xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SOFTMAX]], [[WEIGHTS1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x16x16xf16>, tensor<8x64x1x1xf16> -> tensor<1x8x16x16x!qElemType>
    // CHECK-NOT: IE.Quantize
    // CHECK: return [[CONV1]] : tensor<1x8x16x16x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.250000e-01:123>

// CHECK-LABEL: @DoNotRejectFusionOnConvSoftmaxConvQuantizeLowRatio
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x64x32x32xf16>)
func.func @DoNotRejectFusionOnConvSoftmaxConvQuantizeLowRatio(%arg0: tensor<1x64x32x32xf16>) -> tensor<1x8x32x32x!qElemType> {
    %weights0 = const.Declare tensor<64x64x1x1xf16> = dense<1.0> : tensor<64x64x1x1xf16>
    %weights1 = const.Declare tensor<8x64x1x1xf16> = dense<1.0> : tensor<8x64x1x1xf16>

    %conv0 = IE.Convolution(%arg0, %weights0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x32x32xf16>, tensor<64x64x1x1xf16> -> tensor<1x64x32x32xf16>
    %softmax = IE.SoftMax(%conv0) {axisInd = 1 : i64} : tensor<1x64x32x32xf16> -> tensor<1x64x32x32xf16>
    %conv1 = IE.Convolution(%softmax, %weights1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x32x32xf16>, tensor<8x64x1x1xf16> -> tensor<1x8x32x32xf16>
    %quantize = IE.Quantize(%conv1) {dstElemType = !qElemType} : tensor<1x8x32x32xf16> -> tensor<1x8x32x32x!qElemType>

    return %quantize : tensor<1x8x32x32x!qElemType>

    // CHECK-DAG: [[WEIGHTS0:%.+]] = const.Declare tensor<64x64x1x1xf16>
    // CHECK-DAG: [[WEIGHTS1:%.+]] = const.Declare tensor<8x64x1x1xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[ARG0]], [[WEIGHTS0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x32x32xf16>, tensor<64x64x1x1xf16> -> tensor<1x64x32x32xf16>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[CONV0]]) {axisInd = 1 : i64} : tensor<1x64x32x32xf16> -> tensor<1x64x32x32xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SOFTMAX]], [[WEIGHTS1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x32x32xf16>, tensor<8x64x1x1xf16> -> tensor<1x8x32x32x!qElemType>
    // CHECK-NOT: IE.Quantize
    // CHECK: return [[CONV1]] : tensor<1x8x32x32x!qElemType>
}

// -----
!qElemType = !quant.uniform<u8:f16, 1.250000e-01:123>
// CHECK-LABEL: @FuseQuantizeOnMatMulSoftmaxMatMul
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x8x1x64xf16>)
func.func @FuseQuantizeOnMatMulSoftmaxMatMul(%arg0: tensor<1x8x1x64xf16>) -> tensor<1x8x1x1024x!qElemType> {
    %weights0 = const.Declare tensor<1x8x16384x64xf16> = dense<1.0> : tensor<1x8x16384x64xf16>
    %weights1 = const.Declare tensor<1x8x16384x1024xf16> = dense<1.0> : tensor<1x8x16384x1024xf16>
    %matmul0 = IE.MatMul(%arg0, %weights0) {transpose_b} : tensor<1x8x1x64xf16>, tensor<1x8x16384x64xf16> -> tensor<1x8x1x16384xf16>
    %softmax = IE.SoftMax(%matmul0) {axisInd = 1 : i64} : tensor<1x8x1x16384xf16> -> tensor<1x8x1x16384xf16>
    %matmul1 = IE.MatMul(%softmax, %weights1) : tensor<1x8x1x16384xf16>, tensor<1x8x16384x1024xf16> -> tensor<1x8x1x1024xf16>
    %quantize = IE.Quantize(%matmul1) {dstElemType = !qElemType} : tensor<1x8x1x1024xf16> -> tensor<1x8x1x1024x!qElemType>

    return %quantize : tensor<1x8x1x1024x!qElemType>

    // CHECK-DAG: [[WEIGHTS0:%.+]] = const.Declare tensor<1x8x16384x64xf16> = dense<1.000000e+00> : tensor<1x8x16384x64xf16>
    // CHECK-DAG: [[WEIGHTS1:%.+]] = const.Declare tensor<1x8x16384x1024xf16> = dense<1.000000e+00> : tensor<1x8x16384x1024xf16>
    // CHECK: [[MATMUL0:%.+]] = IE.MatMul([[ARG0]], [[WEIGHTS0]]) {transpose_b} : tensor<1x8x1x64xf16>, tensor<1x8x16384x64xf16> -> tensor<1x8x1x16384xf16>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[MATMUL0]]) {axisInd = 1 : i64} : tensor<1x8x1x16384xf16> -> tensor<1x8x1x16384xf16>
    // CHECK: [[MATMUL1:%.+]] = IE.MatMul([[SOFTMAX]], [[WEIGHTS1]]) : tensor<1x8x1x16384xf16>, tensor<1x8x16384x1024xf16> -> tensor<1x8x1x1024x!qElemType>
    // CHECK-NOT: IE.Quantize
    // CHECK: return [[MATMUL1]] : tensor<1x8x1x1024x!qElemType>
}
