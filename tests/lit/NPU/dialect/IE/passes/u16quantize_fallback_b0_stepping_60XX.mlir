//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% revision-id=3" --u16-quantize-fallback --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU6010
// Baseline: REVISION_B via revision-id=3.

!qElemType  = !quant.uniform<u16:f16, 1.2090807877116612E-5>
!qElemType1 = !quant.uniform<u8:f16, 3.4662022017965132E-4:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:    func.func @FallbackAndConvertsWeightsToI8
// CHECK-SAME:     ([[ARG_0:%.+]]: tensor<1x3x16x16x!qElemType, {order = #NHWC}>)
func.func @FallbackAndConvertsWeightsToI8(%arg0: tensor<1x3x16x16x!qElemType, {order = #NHWC}>) -> tensor<1x16x18x18x!qElemType, {order = #NHWC}> {

    %weights = const.Declare tensor<16x3x1x1x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<16x3x1x1xf16>, [#const.ConvertElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %weights)
          {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
          : tensor<1x3x16x16x!qElemType, {order = #NHWC}>, tensor<16x3x1x1x!qElemType1, {order = #NHWC}>
            -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x16x18x18x!qElemType, {order = #NHWC}>

    // CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<16x3x1x1x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x3x1x1xf16>, [#const.ConvertElemType<!qElemType2>, #const.Reorder<#NHWC>, #const.ConvertElemType<!qElemType1>]
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[ARG_0]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType, {order = #NHWC}> -> tensor<1x3x16x16xf16, {order = #NHWC}>

    // CHECK:         [[CONV:%.+]] = IE.Convolution([[DEQUANT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x16xf16, {order = #NHWC}>, tensor<16x3x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>

    // CHECK-NEXT:  return [[CONV]] : tensor<1x16x18x18x!qElemType, {order = #NHWC}>
}

// -----

!qElemType  = !quant.uniform<u16:f16, 1.2090807877116612E-5>
!qElemType1 = !quant.uniform<u8:f16, 3.4662022017965132E-4:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:    func.func @SwHeuristicNotMet
// CHECK-SAME:     ([[ARG_0:%.+]]: tensor<1x3x16x16x!qElemType, {order = #NHWC}>, [[ARG_1:%.+]]: tensor<16x3x1x1x!qElemType1, {order = #NHWC}>, [[ARG_2:%.+]]:  tensor<16x16x1x1x!qElemType1, {order = #NHWC}>)
func.func @SwHeuristicNotMet(%arg0: tensor<1x3x16x16x!qElemType, {order = #NHWC}>,
                                     %arg1: tensor<16x3x1x1x!qElemType1, {order = #NHWC}>,
                                     %arg2: tensor<16x16x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x16x18x18x!qElemType, {order = #NHWC}> {
    %2 = IE.Convolution(%arg0, %arg1)
          {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
          : tensor<1x3x16x16x!qElemType, {order = #NHWC}>, tensor<16x3x1x1x!qElemType1, {order = #NHWC}>
            -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>

    %3 = IE.Convolution(%2, %arg2)
        {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x18x18x!qElemType, {order = #NHWC}>, tensor<16x16x1x1x!qElemType1, {order = #NHWC}>
          -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>

    return %3 : tensor<1x16x18x18x!qElemType, {order = #NHWC}>

    // CHECK:         [[DEQUANT_0:%.+]] = IE.Dequantize([[ARG_0]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType, {order = #NHWC}> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // CHECK-NEXT:    [[CONV_0:%.+]] = IE.Convolution([[DEQUANT_0]], [[ARG_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x16xf16, {order = #NHWC}>, tensor<16x3x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x18x18xf16, {order = #NHWC}>
    // CHECK-NEXT:    [[CLAMP:%.+]] = IE.Clamp([[CONV_0]]) {max = 0.79237109422683716 : f64, min = 0.000000e+00 : f64} : tensor<1x16x18x18xf16, {order = #NHWC}> -> tensor<1x16x18x18xf16, {order = #NHWC}>
    // CHECK-NEXT:    [[CONV_1:%.+]] = IE.Convolution([[CLAMP]], [[ARG_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x18x18xf16, {order = #NHWC}>, tensor<16x16x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>
    // CHECK-NEXT:    return [[CONV_1]] : tensor<1x16x18x18x!qElemType, {order = #NHWC}>
}

// -----

!qElemType  = !quant.uniform<u16:f16, 1.2090807877116612E-5>
!qElemType1 = !quant.uniform<u8:f16, 3.4662022017965132E-4:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:    func.func @DedupDuplicateDequantizeForSharedProducer
// CHECK-SAME:     ([[ARG_0:%.+]]: tensor<1x3x16x16x!qElemType, {order = #NHWC}>, [[ARG_1:%.+]]: tensor<16x3x1x1x!qElemType1, {order = #NHWC}>, [[ARG_2:%.+]]: tensor<16x16x1x1x!qElemType1, {order = #NHWC}>, [[ARG_3:%.+]]: tensor<16x16x1x1x!qElemType1, {order = #NHWC}>)
func.func @DedupDuplicateDequantizeForSharedProducer(%arg0: tensor<1x3x16x16x!qElemType, {order = #NHWC}>,
                                     %arg1: tensor<16x3x1x1x!qElemType1, {order = #NHWC}>,
                                     %arg2: tensor<16x16x1x1x!qElemType1, {order = #NHWC}>,
                                     %arg3: tensor<16x16x1x1x!qElemType1, {order = #NHWC}>)
        -> (tensor<1x16x18x18x!qElemType, {order = #NHWC}>, tensor<1x16x18x18x!qElemType, {order = #NHWC}>) {
    %0 = IE.Convolution(%arg0, %arg1)
          {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
          : tensor<1x3x16x16x!qElemType, {order = #NHWC}>, tensor<16x3x1x1x!qElemType1, {order = #NHWC}>
            -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>

    %1 = IE.Convolution(%0, %arg2)
        {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x18x18x!qElemType, {order = #NHWC}>, tensor<16x16x1x1x!qElemType1, {order = #NHWC}>
          -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>

    %2 = IE.Convolution(%0, %arg3)
        {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x18x18x!qElemType, {order = #NHWC}>, tensor<16x16x1x1x!qElemType1, {order = #NHWC}>
          -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>

    return %1, %2 : tensor<1x16x18x18x!qElemType, {order = #NHWC}>, tensor<1x16x18x18x!qElemType, {order = #NHWC}>

    // CHECK:         [[DEQUANT_0:%.+]] = IE.Dequantize([[ARG_0]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType, {order = #NHWC}> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // CHECK-NEXT:    [[CONV_0:%.+]] = IE.Convolution([[DEQUANT_0]], [[ARG_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x16xf16, {order = #NHWC}>, tensor<16x3x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x18x18xf16, {order = #NHWC}>
    // CHECK-NEXT:    [[CLAMP:%.+]] = IE.Clamp([[CONV_0]]) {max = 0.79237109422683716 : f64, min = 0.000000e+00 : f64} : tensor<1x16x18x18xf16, {order = #NHWC}> -> tensor<1x16x18x18xf16, {order = #NHWC}>
    // CHECK-NOT:     IE.Dequantize
    // CHECK-DAG:     [[CONV_1:%.+]] = IE.Convolution([[CLAMP]], [[ARG_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x18x18xf16, {order = #NHWC}>, tensor<16x16x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>
    // CHECK-DAG:     [[CONV_2:%.+]] = IE.Convolution([[CLAMP]], [[ARG_3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x18x18xf16, {order = #NHWC}>, tensor<16x16x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>
    // CHECK:         return [[CONV_1]], [[CONV_2]]
}

// -----

!qElemType  = !quant.uniform<u16:f16, 1.2090807877116612E-5>
!qElemType1 = !quant.uniform<u8:f16, 3.4662022017965132E-4:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:    func.func @SwHeuristicMet
// CHECK-SAME:     ([[ARG_0:%.+]]: tensor<1x512x120x16x!qElemType, {order = #NHWC}>, [[ARG_1:%.+]]: tensor<512x512x1x1x!qElemType1, {order = #NHWC}>, [[ARG_2:%.+]]: tensor<512x512x1x1x!qElemType1, {order = #NHWC}>)
func.func @SwHeuristicMet(%arg0: tensor<1x512x120x16x!qElemType, {order = #NHWC}>,
                                     %arg1: tensor<512x512x1x1x!qElemType1, {order = #NHWC}>,
                                     %arg2: tensor<512x512x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x512x122x18x!qElemType, {order = #NHWC}> {
    %2 = IE.Convolution(%arg0, %arg1)
          {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
          : tensor<1x512x120x16x!qElemType, {order = #NHWC}>, tensor<512x512x1x1x!qElemType1, {order = #NHWC}>
            -> tensor<1x512x122x18x!qElemType, {order = #NHWC}>

    %3 = IE.Convolution(%2, %arg2)
        {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x512x122x18x!qElemType, {order = #NHWC}>, tensor<512x512x1x1x!qElemType1, {order = #NHWC}>
          -> tensor<1x512x122x18x!qElemType, {order = #NHWC}>

    return %3 : tensor<1x512x122x18x!qElemType, {order = #NHWC}>
    // CHECK-NOT:     IE.Dequantize
    // CHECK:         [[CONV_0:%.+]] = IE.Convolution([[ARG_0]], [[ARG_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x120x16x!qElemType, {order = #NHWC}>, tensor<512x512x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x512x122x18x!qElemType, {order = #NHWC}>
    // CHECK-NEXT:    [[CONV_1:%.+]] = IE.Convolution([[CONV_0]], [[ARG_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x122x18x!qElemType, {order = #NHWC}>, tensor<512x512x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x512x122x18x!qElemType, {order = #NHWC}>
    // CHECK-NEXT:    return [[CONV_1]] : tensor<1x512x122x18x!qElemType, {order = #NHWC}>
}

// -----

!qElemType  = !quant.uniform<u16:f16, 1.2090807877116612E-5>
!qElemType1 = !quant.uniform<u8:f16, 3.4662022017965132E-4:128>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:    func.func @HwTest
// CHECK:     ([[ARG_0:%.+]]: tensor<1x512x120x16x!qElemType, {order = #NCHW}>, [[ARG_1:%.+]]: tensor<512x512x1x1x!qElemType1, {order = #NCHW}>, [[ARG_2:%.+]]: tensor<512x512x1x1x!qElemType1, {order = #NCHW}>)
func.func @HwTest(%arg0: tensor<1x512x120x16x!qElemType, {order = #NCHW}>,
                                     %arg1: tensor<512x512x1x1x!qElemType1, {order = #NCHW}>,
                                     %arg2: tensor<512x512x1x1x!qElemType1, {order = #NCHW}>) -> tensor<1x512x122x18x!qElemType, {order = #NCHW}> {
    %2 = IE.Convolution(%arg0, %arg1)
          {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
          : tensor<1x512x120x16x!qElemType, {order = #NCHW}>, tensor<512x512x1x1x!qElemType1, {order = #NCHW}>
            -> tensor<1x512x122x18x!qElemType, {order = #NCHW}>

    %3 = IE.Convolution(%2, %arg2)
        {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x512x122x18x!qElemType, {order = #NCHW}>, tensor<512x512x1x1x!qElemType1, {order = #NCHW}>
          -> tensor<1x512x122x18x!qElemType, {order = #NCHW}>

    return %3 : tensor<1x512x122x18x!qElemType, {order = #NCHW}>

    // Since DCIM requires NHWC layout, we fallback

    // CHECK:         [[DEQUANT_0:%.+]] = IE.Dequantize([[ARG_0]]) {dstElemType = f16} : tensor<1x512x120x16x!qElemType, {order = #NCHW}> -> tensor<1x512x120x16xf16>
    // CHECK-NEXT:    [[CONV_0:%.+]] = IE.Convolution([[DEQUANT_0]], [[ARG_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x120x16xf16>, tensor<512x512x1x1x!qElemType1, {order = #NCHW}> -> tensor<1x512x122x18xf16>
    // CHECK-NEXT:    [[CLAMP:%.+]] = IE.Clamp([[CONV_0]]) {max = 0.79237109422683716 : f64, min = 0.000000e+00 : f64} : tensor<1x512x122x18xf16> -> tensor<1x512x122x18xf16>
    // CHECK-NEXT:    [[CONV_1:%.+]] = IE.Convolution([[CLAMP]], [[ARG_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x122x18xf16>, tensor<512x512x1x1x!qElemType1, {order = #NCHW}> -> tensor<1x512x122x18x!qElemType, {order = #NCHW}>
    // CHECK-NEXT:    return [[CONV_1]] : tensor<1x512x122x18x!qElemType, {order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
!qElemType = !quant.uniform<i16:f16, 5.000000e-01:2888>
!qElemType1 = !quant.uniform<i8:f16, 1.0000152590218967>

// CHECK: func.func @SwHeuristicNotMetForMatMul()
func.func @SwHeuristicNotMetForMatMul() -> tensor<1x16x18x18x!qElemType> {
    %INP1 = const.Declare tensor<1x16x18x18x!qElemType> = dense<1.0> : tensor<1x16x18x18xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType> ]
    %INP2 = const.Declare tensor<1x16x18x18x!qElemType1> = dense<1.0> : tensor<1x16x18x18xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1> ]
    %MATMUL = IE.MatMul(%INP1, %INP2) {transpose_b} : tensor<1x16x18x18x!qElemType>,
     tensor<1x16x18x18x!qElemType1> -> tensor<1x16x18x18x!qElemType>
    return %MATMUL : tensor<1x16x18x18x!qElemType>

    // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x16x18x18x!qElemType> = dense<1.000000e+00> : tensor<1x16x18x18xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<1x16x18x18x!qElemType1> = dense<1.000000e+00> : tensor<1x16x18x18xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]

    // CHECK-NEXT: [[DEQUANT_0:%.+]] = IE.Dequantize([[CST_0]]) {dstElemType = f16} : tensor<1x16x18x18x!qElemType> -> tensor<1x16x18x18xf16>
    // CHECK-NEXT: [[MATMUL:%.+]] = IE.MatMul([[DEQUANT_0]], [[CST_1]]) {transpose_b} : tensor<1x16x18x18xf16>, tensor<1x16x18x18x!qElemType1> -> tensor<1x16x18x18x!qElemType>
    // CHECK-NEXT: return [[MATMUL]] : tensor<1x16x18x18x!qElemType>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
!qElemType = !quant.uniform<i16:f16, 5.000000e-01:2888>
!qElemType1 = !quant.uniform<i8:f16, 1.0000152590218967>

// CHECK: func.func @HwTestForMatMul()
func.func @HwTestForMatMul() -> tensor<1x4x18x18x!qElemType> {
    %INP1 = const.Declare tensor<1x4x18x18x!qElemType> = dense<1.0> : tensor<1x4x18x18xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType> ]
    %INP2 = const.Declare tensor<1x4x18x18x!qElemType1> = dense<1.0> : tensor<1x4x18x18xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1> ]
    %MATMUL = IE.MatMul(%INP1, %INP2) {transpose_b} : tensor<1x4x18x18x!qElemType>,
     tensor<1x4x18x18x!qElemType1> -> tensor<1x4x18x18x!qElemType>
    return %MATMUL : tensor<1x4x18x18x!qElemType>

    // Since IC < 16, we fallback

    // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x4x18x18x!qElemType> = dense<1.000000e+00> : tensor<1x4x18x18xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<1x4x18x18x!qElemType1> = dense<1.000000e+00> : tensor<1x4x18x18xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]

    // CHECK-NEXT: [[DEQUANT_0:%.+]] = IE.Dequantize([[CST_0]]) {dstElemType = f16} : tensor<1x4x18x18x!qElemType> -> tensor<1x4x18x18xf16>
    // CHECK-NEXT: [[MATMUL:%.+]] = IE.MatMul([[DEQUANT_0]], [[CST_1]]) {transpose_b} : tensor<1x4x18x18xf16>, tensor<1x4x18x18x!qElemType1> -> tensor<1x4x18x18x!qElemType>
    // CHECK-NEXT: return [[MATMUL]] : tensor<1x4x18x18x!qElemType>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
!qElemType = !quant.uniform<u16:f16, 5.000000e-01:2888>
!qElemType1 = !quant.uniform<u8:f16, 1.0000152590218967>

// CHECK: func.func @FallbackAndConvertWeightsToI8ForMatMul()
func.func @FallbackAndConvertWeightsToI8ForMatMul() -> tensor<1x16x18x18x!qElemType> {
    %INP1 = const.Declare tensor<1x16x18x18x!qElemType> = dense<1.0> : tensor<1x16x18x18xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType> ]
    %INP2 = const.Declare tensor<1x16x18x18x!qElemType1> = dense<1.0> : tensor<1x16x18x18xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1> ]
    %MATMUL = IE.MatMul(%INP1, %INP2) {transpose_b} : tensor<1x16x18x18x!qElemType>,
     tensor<1x16x18x18x!qElemType1> -> tensor<1x16x18x18x!qElemType>
    return %MATMUL : tensor<1x16x18x18x!qElemType>

    // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x16x18x18x!qElemType> = dense<1.000000e+00> : tensor<1x16x18x18xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<1x16x18x18x!qElemType1> = dense<1.000000e+00> : tensor<1x16x18x18xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.ConvertElemType<!qElemType1>]

    // CHECK-NEXT: [[DEQUANT_0:%.+]] = IE.Dequantize([[CST_0]]) {dstElemType = f16} : tensor<1x16x18x18x!qElemType> -> tensor<1x16x18x18xf16>
    // CHECK-NEXT: [[MATMUL:%.+]] = IE.MatMul([[DEQUANT_0]], [[CST_1]]) {transpose_b} : tensor<1x16x18x18xf16>, tensor<1x16x18x18x!qElemType1> -> tensor<1x16x18x18x!qElemType>
    // CHECK-NEXT: return [[MATMUL]] : tensor<1x16x18x18x!qElemType>
}
