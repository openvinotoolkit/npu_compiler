//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --u16-quantize-fallback --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU6010

!qElemType  = !quant.uniform<u16:f16, 1.2090807877116612E-5:222>
!qElemType1 = !quant.uniform<u8:f16, 3.4662022017965132E-4:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @ConvTest attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL:    func.func @FallbackAndConvertsWeightsToI8ForConv
    // CHECK-SAME:     ([[ARG_0:%.+]]: tensor<1x3x16x16x!qElemType, {order = #NHWC}>)
    func.func @FallbackAndConvertsWeightsToI8ForConv(%arg0: tensor<1x3x16x16x!qElemType, {order = #NHWC}>) -> tensor<1x16x18x18x!qElemType, {order = #NHWC}> {
        %weights = const.Declare tensor<16x3x1x1x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<16x3x1x1xf16>, [#const.ConvertElemType<!qElemType1>, #const.Reorder<#NHWC>]
        %0 = IE.Convolution(%arg0, %weights)
              {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
              : tensor<1x3x16x16x!qElemType, {order = #NHWC}>, tensor<16x3x1x1x!qElemType1, {order = #NHWC}>
                -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>

        return %0 : tensor<1x16x18x18x!qElemType, {order = #NHWC}>

        // CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<16x3x1x1x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x3x1x1xf16>, [#const.ConvertElemType<!qElemType2>, #const.Reorder<#NHWC>, #const.ConvertElemType<!qElemType1>]
        // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[ARG_0]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType, {order = #NHWC}> -> tensor<1x3x16x16xf16, {order = #NHWC}>

        // CHECK:     [[CONV:%.+]] = IE.Convolution([[DEQUANT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x16xf16, {order = #NHWC}>, tensor<16x3x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x18x18xf16, {order = #NHWC}>
        // CHECK:     [[QUANT:%.+]] = IE.Quantize([[CONV]]) {dstElemType = !qElemType} : tensor<1x16x18x18xf16, {order = #NHWC}> -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>
        // CHECK-NEXT:  return [[QUANT]] : tensor<1x16x18x18x!qElemType, {order = #NHWC}>
    }
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
!qElemType  = !quant.uniform<u16:f16, 1.2090807877116612E-5:222>
!qElemType1 = !quant.uniform<u8:f16, 3.4662022017965132E-4:128>


module @MatMulTest attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: func.func @FallbackAndConvertsWeightsToI8ForMatMul()
    func.func @FallbackAndConvertsWeightsToI8ForMatMul() -> tensor<1x16x18x18x!qElemType> {
        %INP1 = const.Declare tensor<1x16x18x18x!qElemType> = dense<1.0> : tensor<1x16x18x18xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType> ]
        %INP2 = const.Declare tensor<1x16x18x18x!qElemType1> = dense<1.0> : tensor<1x16x18x18xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1> ]
        %MATMUL = IE.MatMul(%INP1, %INP2) {transpose_b} : tensor<1x16x18x18x!qElemType>, tensor<1x16x18x18x!qElemType1> -> tensor<1x16x18x18x!qElemType>
        return %MATMUL : tensor<1x16x18x18x!qElemType>

        // CHECK:      [[CST_0:%.+]] = const.Declare tensor<1x16x18x18x!qElemType> = dense<1.000000e+00> : tensor<1x16x18x18xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
        // CHECK-NEXT: [[CST_1:%.+]] = const.Declare tensor<1x16x18x18x!qElemType1> = dense<1.000000e+00> : tensor<1x16x18x18xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.ConvertElemType<!qElemType1>]
        // CHECK-NEXT: [[DEQUANT:%.+]] = IE.Dequantize([[CST_0]]) {dstElemType = f16} : tensor<1x16x18x18x!qElemType> -> tensor<1x16x18x18xf16>
        // CHECK-NEXT: [[MATMUL:%.+]] = IE.MatMul([[DEQUANT]], [[CST_1]]) {transpose_b} : tensor<1x16x18x18xf16>, tensor<1x16x18x18x!qElemType1> -> tensor<1x16x18x18xf16>
        // CHECK-NEXT: [[QUANT:%.+]] = IE.Quantize([[MATMUL]]) {dstElemType = !qElemType} : tensor<1x16x18x18xf16> -> tensor<1x16x18x18x!qElemType>
        // CHECK-NEXT: return [[QUANT]] : tensor<1x16x18x18x!qElemType>
    }
}
