//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-to-mixed-precision="enable-float-in-quant-weights-mixed-mode=true" %s | FileCheck %s
// REQUIRES: platform-NPU6010


!qElemType = !quant.uniform<i8:f16, 1.1534313725490195:10>

module @TestPerTensorAsymmetricMixedPrecisionFloatInputQuantWeightsConv {
  // CHECK-LABEL: @PerTensorAsymmetricMixedPrecisionFloatInputQuantWeightsConv
  // CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16>)
  func.func @PerTensorAsymmetricMixedPrecisionFloatInputQuantWeightsConv(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
    %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
    %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    return %result : tensor<1x16x16x16xf16>

    //CHECK: [[VAL0:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.Convolution([[ARG_0]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
    //CHECK: return [[VAL1]]
  }
}

// -----

!qElemType = !quant.uniform<i8:f16:0, {0.010680671751968504:-3,0.0081200787401574797:-78,0.010596087598425197:-58}>
!qElemType1 = !quant.uniform<u8:f16:0, {0.010680671751968504:125,0.0081200787401574797:50,0.010596087598425197:70}>

module @TestDontConvertAsymmetricPerChannelQuantizedU8Weights {
  config.PipelineOptions @Options {
        config.Option @config.AsymmetricPerChannelZP : false
  }
  // CHECK-LABEL: @DontConvertAsymmetricPerChannelQuantizedU8Weights
  func.func @DontConvertAsymmetricPerChannelQuantizedU8Weights(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %cst = const.Declare tensor<3x3x3x3x!qElemType> = dense<9.000000e+00> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>]
    %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    return %1 : tensor<1x3x14x14xf16>
    //CHECK:  const.Declare tensor<3x3x3x3x!qElemType>
    //CHECK:  IE.Dequantize
    //CHECK: IE.Convolution
  }
}
