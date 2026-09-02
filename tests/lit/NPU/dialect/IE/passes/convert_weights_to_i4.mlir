//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-weights-to-i4 --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

!qElemType = !quant.uniform<i4<-8:7>:f16:0, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>
!qElemType1 = !quant.uniform<u4<0:15>:f16:0, {0.010680671751968504:8,0.0081200787401574797:8,0.010596087598425197:8}>
!qElemType2 = !quant.uniform<i4:f16, 1.1534313725490195>
!qElemType3 = !quant.uniform<i4:f16, 2.4627450980392158>

// CHECK-LABEL: @ConvertU4WeightsToI4
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @ConvertU4WeightsToI4(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!qElemType1> =
        dense<-1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType1>]
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType2>
    %2 = IE.Convolution(%1, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qElemType2>, tensor<3x3x3x3x!qElemType1> -> tensor<1x3x14x14x!qElemType3>
    %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x3x14x14x!qElemType3> -> tensor<1x3x14x14xf16>

    return %3 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<-1.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<ui4>,
    // CHECK-SAME:      #const.CastElemType<!qElemType1>,
    // CHECK-SAME:      #const.ConvertElemType<!qElemType>

    // CHECK:       [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType2} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType2>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[QUANT]], [[CST]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType2>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType3>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CONV]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType3> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[DEQUANT]]
}

// -----

!qElemType = !quant.uniform<u4:f16, 1.1534313725490195>
// We don't convert u4 to i4 because of the bad zero point value of U4.
// CHECK-LABEL: @NotConvertU4Weights
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16x!qElemType>)
func.func @NotConvertU4Weights(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<-1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>]
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    return %2 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<-1.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<ui4>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CONV]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[DEQUANT]]
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.1534313725490195>
// CHECK-LABEL: @KeepI4Weights
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16x!qElemType>)
func.func @KeepI4Weights(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<-1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>]
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    return %2 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<-1.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<si4>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CONV]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[DEQUANT]]
}

// -----

!qElemType = !quant.uniform<u4<0:15>:f16:0, {0.010680671751968504:8,0.0081200787401574797:8,0.010596087598425197:8}>
// CHECK: !qElemType = !quant.uniform<i4:f16:0, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>

// CHECK-LABEL: @ConvertFromPerAxisTypeU4ToI4
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @ConvertFromPerAxisTypeU4ToI4(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %cst = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<3.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    return %1 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:       dense<3.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:       [#const.CastElemType<ui4>, #const.CastElemType<!qElemType1>,
    // CHECK-SAME:        #const.ConvertElemType<!qElemType>]
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[CONV]]
}

// -----

!qElemType = !quant.uniform<u4<0:15>:f16:0, {0.010680671751968504:8,0.0081200787401574797:8,0.010596087598425197:7}>
// CHECK: !qElemType = !quant.uniform<u4:f16:0, {0.010680671751968504:8,0.0081200787401574797:8,0.010596087598425197:7}>

// CHECK-LABEL: @DontConvertFromPerAxisTypeU4ToI4NotAllZeroPointAre8
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @DontConvertFromPerAxisTypeU4ToI4NotAllZeroPointAre8(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %cst = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<3.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    return %1 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:       dense<3.000000e+00> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>]
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[CONV]]
}

// -----

!qElemType = !quant.uniform<u4:f16:1, {0.010680671751968504:8,0.0081200787401574797:8,0.010596087598425197:8}>
!qElemType1 = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK: [[Q_ELEM_TYPE0:!.+]] = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK: [[Q_ELEM_TYPE1:!.+]] = !quant.uniform<i4:f16:1, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>

// CHECK: @ConvertQuantizeCastPerAxisU4ToI4AllZeroPoints8
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<16x3x1x1x[[Q_ELEM_TYPE0]]>)
func.func @ConvertQuantizeCastPerAxisU4ToI4AllZeroPoints8(%arg0: tensor<16x3x1x1x!qElemType1>) -> tensor<16x3x1x1x!qElemType> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<16x3x1x1x!qElemType1> -> tensor<16x3x1x1x!qElemType>
    return %0 : tensor<16x3x1x1x!qElemType>

    // CHECK:       [[QUANTCAST:%.+]] = IE.QuantizeCast([[ARG0]]) {dstElemType = [[Q_ELEM_TYPE1]]}
    // CHECK-SAME:      : tensor<16x3x1x1x[[Q_ELEM_TYPE0]]> -> tensor<16x3x1x1x[[Q_ELEM_TYPE1]]>
}

// -----

!qElemType = !quant.uniform<u4:f16:1, {0.010680671751968504:8,0.0081200787401574797:8,0.010596087598425197:8}>
// CHECK: [[Q_ELEM_TYPE1:!.+]] = !quant.uniform<i4:f16:1, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>

// CHECK: @ConvertQuantizePerAxisU4ToI4AllZeroPoints8
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<16x3x1x1xf16>)
func.func @ConvertQuantizePerAxisU4ToI4AllZeroPoints8(%arg0: tensor<16x3x1x1xf16>) -> tensor<16x3x1x1x!qElemType> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<16x3x1x1xf16> -> tensor<16x3x1x1x!qElemType>
    return %0 : tensor<16x3x1x1x!qElemType>

    // CHECK:       [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[Q_ELEM_TYPE1]]}
    // CHECK-SAME:      : tensor<16x3x1x1xf16> -> tensor<16x3x1x1x[[Q_ELEM_TYPE1]]>
}

// -----

!qElemType = !quant.uniform<u4:f16, 0.010680671751968504:8>
!qElemType1 = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK: [[Q_ELEM_TYPE0:!.+]] = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK: [[Q_ELEM_TYPE1:!.+]] = !quant.uniform<i4:f16, 0.010680671751968504>

// CHECK: @ConvertQuantizeCastUniformQuantU4ToI4
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<16x3x1x1x[[Q_ELEM_TYPE0]]>)
func.func @ConvertQuantizeCastUniformQuantU4ToI4(%arg0: tensor<16x3x1x1x!qElemType1>) -> tensor<16x3x1x1x!qElemType> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<16x3x1x1x!qElemType1> -> tensor<16x3x1x1x!qElemType>
    return %0 : tensor<16x3x1x1x!qElemType>

    // CHECK:       [[QUANTCAST:%.+]] = IE.QuantizeCast([[ARG0]]) {dstElemType = [[Q_ELEM_TYPE1]]}
    // CHECK-SAME:      : tensor<16x3x1x1x[[Q_ELEM_TYPE0]]> -> tensor<16x3x1x1x[[Q_ELEM_TYPE1]]>
}

// -----

!qElemType = !quant.uniform<u4:f16, 0.0081200787401574797:8>
// CHECK: [[Q_ELEM_TYPE1:!.+]] = !quant.uniform<i4:f16, 0.0081200787401574797>

// CHECK: @ConvertQuantizeUniformQuantU4ToI4
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<16x3x1x1xf16>)
func.func @ConvertQuantizeUniformQuantU4ToI4(%arg0: tensor<16x3x1x1xf16>) -> tensor<16x3x1x1x!qElemType> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<16x3x1x1xf16> -> tensor<16x3x1x1x!qElemType>
    return %0 : tensor<16x3x1x1x!qElemType>

    // CHECK:       [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[Q_ELEM_TYPE1]]}
    // CHECK-SAME:      : tensor<16x3x1x1xf16> -> tensor<16x3x1x1x[[Q_ELEM_TYPE1]]>
}

// -----

// We don't convert u4 storageElement in quant.quantile types
!qElemType = !quant.uniform<!QuantileType.quantile<ui4:ui8, {0.0,16.0,32.0,48.0,64.0,80.0,96.0,112.0,128.0,144.0,160.0,176.0,192.0,208.0,224.0,240.0}>:f16, 2.000000e+00:128>

// CHECK-LABEL: @NotConvertQuantileU4StorageElement
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16x!qElemType>)
func.func @NotConvertQuantileU4StorageElement(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>]
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    return %2 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<ui4>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CONV]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[DEQUANT]]
}

// -----

!qU4weights = !quant.uniform<u4:f16, 0.10000000000000001:8>
!qU8act = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: @KeepU4WeightsWithU8Activations
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x32x16x16xf16>)
func.func @KeepU4WeightsWithU8Activations(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x64x14x14xf16> {
    %weights = const.Declare tensor<64x32x3x3x!qU4weights> = dense<1> : tensor<64x32x3x3xui4>,
        [#const.CastElemType<ui4>, #const.CastElemType<!qU4weights>]
    %act_q = IE.Quantize(%arg0) {dstElemType = !qU8act} : tensor<1x32x16x16xf16> -> tensor<1x32x16x16x!qU8act>
    %conv = IE.Convolution(%act_q, %weights) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x32x16x16x!qU8act>, tensor<64x32x3x3x!qU4weights> -> tensor<1x64x14x14xf16>
    return %conv : tensor<1x64x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<64x32x3x3x!qElemType> = dense<1> : tensor<64x32x3x3xui4>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>]
    // CHECK-NOT:   ConvertElemType
    // CHECK:       [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1}
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[QUANT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:       return [[CONV]]
}

// -----

!qU4weights_sym = !quant.uniform<u4:f16, 0.0625:8>
!qU16act = !quant.uniform<u16:f16, 1.52590218966964e-05>

// CHECK-LABEL: @KeepU4WeightsWithU16Activations
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x32x16x16xf16>)
func.func @KeepU4WeightsWithU16Activations(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x64x14x14xf16> {
    %weights = const.Declare tensor<64x32x3x3x!qU4weights_sym> = dense<2> : tensor<64x32x3x3xui4>,
        [#const.CastElemType<ui4>, #const.CastElemType<!qU4weights_sym>]
    %act_q = IE.Quantize(%arg0) {dstElemType = !qU16act} : tensor<1x32x16x16xf16> -> tensor<1x32x16x16x!qU16act>
    %conv = IE.Convolution(%act_q, %weights) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x32x16x16x!qU16act>, tensor<64x32x3x3x!qU4weights_sym> -> tensor<1x64x14x14xf16>
    return %conv : tensor<1x64x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<64x32x3x3x!qElemType> = dense<2> : tensor<64x32x3x3xui4>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>]
    // CHECK-NOT:   ConvertElemType
    // CHECK:       [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1}
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[QUANT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:       return [[CONV]]
}

// -----

!qU4weights_perchannel = !quant.uniform<u4:f16:0, {0.1:8, 0.2:8, 0.15:8}>
!qU8act_pc = !quant.uniform<u8:f16, 0.01>

// CHECK-LABEL: @KeepU4PerChannelWeightsWithU8Activations
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @KeepU4PerChannelWeightsWithU8Activations(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %weights = const.Declare tensor<3x3x3x3x!qU4weights_perchannel> = dense<1> : tensor<3x3x3x3xui4>,
        [#const.CastElemType<ui4>, #const.CastElemType<!qU4weights_perchannel>]
    %act_q = IE.Quantize(%arg0) {dstElemType = !qU8act_pc} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qU8act_pc>
    %conv = IE.Convolution(%act_q, %weights) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qU8act_pc>, tensor<3x3x3x3x!qU4weights_perchannel> -> tensor<1x3x14x14xf16>
    return %conv : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> = dense<1> : tensor<3x3x3x3xui4>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>]
    // CHECK-NOT:   ConvertElemType
    // CHECK:       [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1}
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[QUANT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:       return [[CONV]]
}

// -----

!qU4weights_gconv = !quant.uniform<u4:f16, 0.1:8>
!qU8act_gconv = !quant.uniform<u8:f16, 0.005>

// CHECK-LABEL: @KeepU4WeightsInGroupConvolutionWithU8
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x4x56x56xf16>)
func.func @KeepU4WeightsInGroupConvolutionWithU8(%arg0: tensor<1x4x56x56xf16>) -> tensor<1x4x56x56xf16> {
    %weights = const.Declare tensor<4x1x3x3x!qU4weights_gconv> = dense<2> : tensor<4x1x3x3xui4>,
        [#const.CastElemType<ui4>, #const.CastElemType<!qU4weights_gconv>]
    %act_q = IE.Quantize(%arg0) {dstElemType = !qU8act_gconv} : tensor<1x4x56x56xf16> -> tensor<1x4x56x56x!qU8act_gconv>
    %conv = IE.GroupConvolution(%act_q, %weights) {
        dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
    } : tensor<1x4x56x56x!qU8act_gconv>, tensor<4x1x3x3x!qU4weights_gconv> -> tensor<1x4x56x56xf16>
    return %conv : tensor<1x4x56x56xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<4x1x3x3x!qElemType> = dense<2> : tensor<4x1x3x3xui4>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>]
    // CHECK-NOT:   ConvertElemType
    // CHECK:       [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1}
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[QUANT]], [[CST]]) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK:       return [[CONV]]
}

// -----

!qU4weights_dequant = !quant.uniform<u4:f16, 0.066650390625:8>

// CHECK: !qElemType = !quant.uniform<i4:f16, 0.066650390625>
// CHECK: !qElemType1 = !quant.uniform<u4:f16, 0.066650390625:8>

// CHECK-LABEL: @ConvertU4WeightsWithDequantizeToConv
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x128x2x2xf16>)
func.func @ConvertU4WeightsWithDequantizeToConv(%arg0: tensor<1x128x2x2xf16>) -> tensor<1x64x2x2xf16> {
    %weights = const.Declare tensor<64x128x3x3x!qU4weights_dequant> = dense<1> : tensor<64x128x3x3xui4>,
        [#const.CastElemType<ui4>, #const.CastElemType<!qU4weights_dequant>]
    %0 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<64x128x3x3x!qU4weights_dequant> -> tensor<64x128x3x3xf16>
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
    } : tensor<1x128x2x2xf16>, tensor<64x128x3x3xf16> -> tensor<1x64x2x2xf16>
    return %1 : tensor<1x64x2x2xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<64x128x3x3x!qElemType> = dense<1> : tensor<64x128x3x3xui4>,
    // CHECK-SAME:      [#const.CastElemType<ui4>, #const.CastElemType<!qElemType1>,
    // CHECK-SAME:       #const.ConvertElemType<!qElemType>]
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<64x128x3x3x!qElemType> -> tensor<64x128x3x3xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x128x2x2xf16>, tensor<64x128x3x3xf16> -> tensor<1x64x2x2xf16>
    // CHECK:       return [[CONV]]
}

// -----

!qU4weights_dequant_matmul = !quant.uniform<u4:f16, 0.05:8>

// CHECK: !qElemType = !quant.uniform<i4:f16, 5.000000e-02>
// CHECK: !qElemType1 = !quant.uniform<u4:f16, 5.000000e-02:8>

// CHECK-LABEL: @ConvertU4WeightsWithDequantizeToMatMul
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x1x128xf16>)
func.func @ConvertU4WeightsWithDequantizeToMatMul(%arg0: tensor<1x1x128xf16>) -> tensor<1x1x64xf16> {
    %weights = const.Declare tensor<64x128x!qU4weights_dequant_matmul> = dense<1> : tensor<64x128xui4>,
        [#const.CastElemType<ui4>, #const.CastElemType<!qU4weights_dequant_matmul>]
    %0 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<64x128x!qU4weights_dequant_matmul> -> tensor<64x128xf16>
    %1 = IE.MatMul(%arg0, %0) {transpose_b} : tensor<1x1x128xf16>, tensor<64x128xf16> -> tensor<1x1x64xf16>
    return %1 : tensor<1x1x64xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<64x128x!qElemType> = dense<1> : tensor<64x128xui4>,
    // CHECK-SAME:      [#const.CastElemType<ui4>, #const.CastElemType<!qElemType1>,
    // CHECK-SAME:       #const.ConvertElemType<!qElemType>]
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<64x128x!qElemType> -> tensor<64x128xf16>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[ARG0]], [[DEQUANT]]) {transpose_b}
    // CHECK-SAME:      : tensor<1x1x128xf16>, tensor<64x128xf16> -> tensor<1x1x64xf16>
    // CHECK:       return [[MATMUL]]
}
