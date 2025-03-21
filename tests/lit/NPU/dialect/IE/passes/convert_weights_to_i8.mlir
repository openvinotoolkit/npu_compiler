//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-weights-to-i8 --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<u8:f16, 2.000000e+00:128>
// CHECK: !qElemType = !quant.uniform<i8:f16, 2.000000e+00>


// CHECK-LABEL: @ConvertFromU8ToI8
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x16x1x1xf32>)
func.func @ConvertFromU8ToI8(%arg0: tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32> {
  %cst = const.Declare tensor<16x16x1x1x!qElemType> = dense<127.0> : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %1 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x16x1x1xf32> -> tensor<1x16x1x1xf16>
  %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1,1]}:  tensor<1x16x1x1xf16> , tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf32>
  return %3 : tensor<1x16x1x1xf32>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
    // CHECK-SAME:      dense<1.270000e+02> : tensor<16x16x1x1xf32>
    // CHECK-SAME:      [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1>,
    // CHECK-SAME:       #const.ConvertElemType<!qElemType>]
    // CHECK:       [[CNVRT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x16x1x1xf32> -> tensor<1x16x1x1xf16>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[CNVRT]], [[DEQUANT]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK:       [[CNVRT2:%.+]] = IE.Convert([[CONV]]) {dstElemType = f32} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf32>
    // CHECK:       return [[CNVRT2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:127>
// CHECK: !qElemType = !quant.uniform<u8:f16, 1.1534313725490195:127>
// We don't convert u8 to i8 because of the zero point value of U8 which must be 128.
// Conversion converts from u8 ZP = 128 to i8 ZP = 0
// CHECK-LABEL: @NotConvertU8Weights
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @NotConvertU8Weights(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %cst = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<9.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    return %1 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<9.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<ui8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[CONV]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
// CHECK-LABEL: @KeepI8Weights
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16x!qElemType>)
func.func @KeepI8Weights(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<-1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    return %2 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<-1.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<si8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CONV]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[DEQUANT]]
}

// -----

!qElemType = !quant.uniform<u8<0:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
// CHECK: !qElemType = !quant.uniform<i8:f16:0, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>

// CHECK-LABEL: @ConvertFromPerAxisTypeU8ToI8
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @ConvertFromPerAxisTypeU8ToI8(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %cst = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<3.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    return %1 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:       dense<3.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:       [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1>,
    // CHECK-SAME:        #const.ConvertElemType<!qElemType>]
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CST]])
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[CONV]]
}

// -----

!qElemType = !quant.quantile<i4:u8:f16, {0.0,16.0,32.0,48.0,64.0,80.0,96.0,112.0,128.0,144.0,160.0,176.0,192.0,208.0,224.0,240.0}:2.000000e+00:128>
// CHECK: !qElemType = !quant.quantile<i4:i8:f16, {-1.280000e+02,-1.120000e+02,-9.600000e+01,-8.000000e+01,-6.400000e+01,-4.800000e+01,-3.200000e+01,-1.600000e+01,0.000000e+00,1.600000e+01,3.200000e+01,4.800000e+01,6.400000e+01,8.000000e+01,9.600000e+01,1.120000e+02}:2.000000e+00>

// CHECK-LABEL: @ConvertQuantileFromU8ToI8
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x16x1x1xf32>)
func.func @ConvertQuantileFromU8ToI8(%arg0: tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32> {
  %cst = const.Declare tensor<16x16x1x1x!qElemType> = dense<127.0> : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %1 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x16x1x1xf32> -> tensor<1x16x1x1xf16>
  %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1,1]}:  tensor<1x16x1x1xf16> , tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf32>
  return %3 : tensor<1x16x1x1xf32>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
    // CHECK-SAME:      dense<1.270000e+02> : tensor<16x16x1x1xf32>
    // CHECK-SAME:      [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1>,
    // CHECK-SAME:       #const.ConvertElemType<!qElemType>]
    // CHECK:       [[CNVRT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x16x1x1xf32> -> tensor<1x16x1x1xf16>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[CNVRT]], [[DEQUANT]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK:       [[CNVRT2:%.+]] = IE.Convert([[CONV]]) {dstElemType = f32} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf32>
    // CHECK:       return [[CNVRT2]]
}

// -----

!qElemType = !quant.quantile<i4:u8:f16, {0.0,16.0,32.0,48.0,64.0,80.0,96.0,112.0,128.0,144.0,160.0,176.0,192.0,208.0,224.0,240.0}:2.000000e+00:127>
// CHECK: !qElemType = !quant.quantile<i4:u8:f16, {0.000000e+00,1.600000e+01,3.200000e+01,4.800000e+01,6.400000e+01,8.000000e+01,9.600000e+01,1.120000e+02,1.280000e+02,1.440000e+02,1.600000e+02,1.760000e+02,1.920000e+02,2.080000e+02,2.240000e+02,2.400000e+02}:2.000000e+00:127>

// Don't convert u8 to i8 because of the zero point value of U8 which must be 128.
// Conversion converts from u8 ZP = 128 to i8 ZP = 0
// CHECK-LABEL: @DontConvertQuantileU8Weights
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @DontConvertQuantileU8Weights(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %cst = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<9.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    return %1 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<9.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<ui8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[CONV]]
}

// -----

!qElemType = !quant.quantile<i4:i8:f16, {-1.280000e+02,-1.120000e+02,-9.600000e+01,-8.000000e+01,-6.400000e+01,-4.800000e+01,-3.200000e+01,-1.600000e+01,0.000000e+00,1.600000e+01,3.200000e+01,4.800000e+01,6.400000e+01,8.000000e+01,9.600000e+01,1.120000e+02}:2.000000e+00:0>
// CHECK: !qElemType = !quant.quantile<i4:i8:f16, {-1.280000e+02,-1.120000e+02,-9.600000e+01,-8.000000e+01,-6.400000e+01,-4.800000e+01,-3.200000e+01,-1.600000e+01,0.000000e+00,1.600000e+01,3.200000e+01,4.800000e+01,6.400000e+01,8.000000e+01,9.600000e+01,1.120000e+02}:2.000000e+00>

// CHECK-LABEL: @KeepQuantileI8Weights
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16x!qElemType>)
func.func @KeepQuantileI8Weights(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<-1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    return %2 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<-1.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<si8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CONV]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[DEQUANT]]
}
