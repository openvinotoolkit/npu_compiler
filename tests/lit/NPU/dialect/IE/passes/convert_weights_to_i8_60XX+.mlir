//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-weights-to-i8 --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU6010

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195:-3>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:125>
// CHECK: !qElemType = !quant.uniform<i8:f16, 1.1534313725490195:-3>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:125>
// There is support for per channel zero point with NPU6+
// CHECK-LABEL: @ConvertAsymmetricPerTensorQuantizedU8Weights
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
module @TestConvertAsymmetricPerTensorQuantizedU8Weights {
  func.func @ConvertAsymmetricPerTensorQuantizedU8Weights(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
      %wgt = const.Declare tensor<3x3x3x3x!qElemType1> =
          dense<9.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]
      %dequant = IE.Dequantize(%wgt) {dstElemType = f16} : tensor<3x3x3x3x!qElemType1> -> tensor<3x3x3x3xf16>
      %conv = IE.Convolution(%arg0, %dequant) {
          dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
      } : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
      return %conv : tensor<1x3x14x14xf16>

      // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
      // CHECK-SAME:      dense<9.000000e+00> : tensor<3x3x3x3xf16>,
      // CHECK-SAME:      #const.CastElemType<ui8>,
      // CHECK-SAME:      #const.CastElemType<!qElemType1>,
      // CHECK-SAME:      #const.ConvertElemType<!qElemType>
      // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
      // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]])
      // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
      // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
      // CHECK:       return [[CONV]]
  }
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195:-3>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:125>
// CHECK: !qElemType = !quant.uniform<i8:f16, 1.1534313725490195:-3>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:125>
// There is support for per channel zero point with NPU6+
// CHECK-LABEL: @ConvertAsymmetricPerTensorQuantizedU8WeightsFusedDequant
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
module @TestConvertAsymmetricPerTensorQuantizedU8WeightsFusedDequant {
  func.func @ConvertAsymmetricPerTensorQuantizedU8WeightsFusedDequant(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
      %wgt = const.Declare tensor<3x3x3x3x!qElemType1> =
          dense<9.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]
      %conv = IE.Convolution(%arg0, %wgt) {
          dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
      } : tensor<1x3x16x16xf16>, tensor<3x3x3x3x!qElemType1> -> tensor<1x3x14x14xf16>
      return %conv : tensor<1x3x14x14xf16>

      // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
      // CHECK-SAME:      dense<9.000000e+00> : tensor<3x3x3x3xf16>,
      // CHECK-SAME:      #const.CastElemType<ui8>,
      // CHECK-SAME:      #const.CastElemType<!qElemType1>,
      // CHECK-SAME:      #const.ConvertElemType<!qElemType>
      // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]])
      // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
      // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14xf16>
      // CHECK:       return [[CONV]]
  }
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {0.010680671751968504:125,0.0081200787401574797:50,0.010596087598425197:70}>
// CHECK: !qElemType = !quant.uniform<u8:f16:0, {0.010680671751968504:125,0.0081200787401574797:50,0.010596087598425197:70}>

// CHECK-LABEL: @DontConvertFromPerAxisTypeAsymmetricU8ToI8
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
module @TestDontConvertFromPerAxisTypeAsymmetricU8ToI8 {
  config.PipelineOptions @Options {
        config.Option @config.AsymmetricPerChannelZP : false
  }
  func.func @DontConvertFromPerAxisTypeAsymmetricU8ToI8(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
      %wgt = const.Declare tensor<3x3x3x3x!qElemType> =
          dense<3.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
      %dequant = IE.Dequantize(%wgt) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
      %conv = IE.Convolution(%arg0, %dequant) {
          dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
      } : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
      return %conv : tensor<1x3x14x14xf16>

      // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
      // CHECK-SAME:       dense<3.000000e+00> : tensor<3x3x3x3xf16>,
      // CHECK-SAME:       [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
      // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CST]])
      // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]])
      // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
      // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
      // CHECK:       return [[CONV]]
  }
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {0.010680671751968504:125,0.0081200787401574797:50,0.010596087598425197:70}>
// CHECK: !qElemType = !quant.uniform<u8:f16:0, {0.010680671751968504:125,0.0081200787401574797:50,0.010596087598425197:70}>

// CHECK-LABEL: @DontConvertFromPerAxisTypeAsymmetricU8ToI8FusedDequant
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
module @TestDontConvertFromPerAxisTypeAsymmetricU8ToI8FusedDequant {
  config.PipelineOptions @Options {
        config.Option @config.AsymmetricPerChannelZP : false
  }
  func.func @DontConvertFromPerAxisTypeAsymmetricU8ToI8FusedDequant(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
      %wgt = const.Declare tensor<3x3x3x3x!qElemType> =
          dense<3.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
      %conv = IE.Convolution(%arg0, %wgt) {
          dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
      } : tensor<1x3x16x16xf16>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14xf16>
      return %conv : tensor<1x3x14x14xf16>

      // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
      // CHECK-SAME:       dense<3.000000e+00> : tensor<3x3x3x3xf16>,
      // CHECK-SAME:       [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
      // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]])
      // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
      // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14xf16>
      // CHECK:       return [[CONV]]
  }
}
