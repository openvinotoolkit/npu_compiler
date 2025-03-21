//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-dequantize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertToDequantize
func.func @ConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi8>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.QuantizeCast([[ARG1:%.*]]) {dstElemType = !qElemType} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1x!qElemType>
  // CHECK:              [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<64x64x1x1x!qElemType> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL2:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-DAG:  [[Q_ELEM_TYPE0:!.*]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG:  [[Q_ELEM_TYPE1:!.*]] = !quant.uniform<u8:f16, 1.000000e+00:128>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK:      ConvertToDequantizeWithQuantInput
// CHECK-SAME: ([[ARG0:%.*]]: tensor<1x64x64x100x[[Q_ELEM_TYPE1]], {order = #NHWC}>, [[ARG1:%.*]]: tensor<64x64x1x1xsi8>) -> tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @ConvertToDequantizeWithQuantInput(%arg0: tensor<1x64x64x100x!qElemType1, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi8>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100x!qElemType1, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.QuantizeCast([[ARG1:%.*]]) {dstElemType = [[Q_ELEM_TYPE0]]} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1x[[Q_ELEM_TYPE0]]>
  // CHECK:              [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<64x64x1x1x[[Q_ELEM_TYPE0]]> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL2:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100x[[Q_ELEM_TYPE1]], {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertToDequantizeWithMiddleOp
func.func @ConvertToDequantizeWithMiddleOp(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1xsi8>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1xsi8> -> tensor<64x64x1xf16>
  %1 = IE.Reshape(%0) { shape_value = [64, 64, 1, 1] } : tensor<64x64x1xf16> -> tensor<64x64x1x1xf16>
  %2 = IE.Convolution(%arg0, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %2 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.QuantizeCast([[ARG1:%.*]]) {dstElemType = !qElemType} : tensor<64x64x1xsi8> -> tensor<64x64x1x!qElemType>
  // CHECK:              [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<64x64x1x!qElemType> -> tensor<64x64x1xf16>
  // CHECK:              [[VAL2:%.*]] = IE.Reshape([[VAL1]]) {shape_value = [64, 64, 1, 1]} : tensor<64x64x1xf16> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL3:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL3]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertU8ToQuant
func.func @ConvertU8ToQuant(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xui8>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xui8> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK-NOT:  IE.Convert

  // CHECK:       [[VAL0:%.*]] = IE.QuantizeCast([[ARG1:%.*]]) {dstElemType = !qElemType}
  // CHECK-SAME:      : tensor<64x64x1x1xui8> -> tensor<64x64x1x1x!qElemType>
  // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16}
  // CHECK-SAME:      : tensor<64x64x1x1x!qElemType> -> tensor<64x64x1x1xf16>
  // CHECK:  [[VAL2:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL1]])
  // CHECK-SAME:      : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:  return [[VAL2]]
}

// -----

// CHECK-LABEL: @KeepConvertIfNotFilter
func.func @KeepConvertIfNotFilter(%arg0: tensor<1x64x32x200xsi8>, %arg1: tensor<1x64x64x64xf16>) -> tensor<1x1x1x37xf16> {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x64x32x200xsi8> -> tensor<1x64x32x200xf16>
  %1 = IE.Reshape(%0) { shape_value = [1, 64, 64, 100] } : tensor<1x64x32x200xf16> -> tensor<1x64x64x100xf16>
  %2 = IE.Convolution(%1, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16>, tensor<1x64x64x64xf16> -> tensor<1x1x1x37xf16>
  return %2 : tensor<1x1x1x37xf16>

  // CHECK:  [[VAL0:%.*]] = IE.Convert([[ARG0:%.*]]) {dstElemType = f16} : tensor<1x64x32x200xsi8> -> tensor<1x64x32x200xf16>
  // CHECK:  [[VAL1:%.*]] = IE.Reshape([[VAL0]]) {shape_value = [1, 64, 64, 100]} : tensor<1x64x32x200xf16> -> tensor<1x64x64x100xf16>
  // CHECK:  [[VAL2:%.*]] = IE.Convolution([[VAL1]], [[ARG1:%.*]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16>, tensor<1x64x64x64xf16> -> tensor<1x1x1x37xf16>
  // CHECK:  return [[VAL2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotConvertToDequantizeIfInputIsConstants
func.func @NotConvertToDequantizeIfInputIsConstants(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<64x64x1x1xsi8> = dense<1> : tensor<64x64x1x1xsi8>
  %0 = IE.Convert(%cst) {dstElemType = f16} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              [[CST:%.*]] = const.Declare tensor<64x64x1x1xf16> = dense<1> : tensor<64x64x1x1xsi8>, [#const.CastElemType<f16>]
  // CHECK-NOT:          IE.QuantizeCast
  // CHECK:              [[VAL0:%.*]] = IE.Convolution([[ARG0:%.*]], [[CST:%.*]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @I4ConvertToDequantize
func.func @I4ConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi4>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi4> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.QuantizeCast([[ARG1:%.*]]) {dstElemType = !qElemType} : tensor<64x64x1x1xsi4> -> tensor<64x64x1x1x!qElemType>
  // CHECK:              [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<64x64x1x1x!qElemType> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL2:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @I1DontConvertToDequantize
func.func @I1DontConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xi1>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xi1> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.Convert([[ARG1:%.*]]) {dstElemType = f16} : tensor<64x64x1x1xi1> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @I32DontConvertToDequantize
func.func @I32DontConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi32>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi32> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.Convert([[ARG1:%.*]]) {dstElemType = f16} : tensor<64x64x1x1xsi32> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @I16DontConvertToDequantize
func.func @I16DontConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi16>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi16> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.Convert([[ARG1:%.*]]) {dstElemType = f16} : tensor<64x64x1x1xsi16> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: !qElemType = !quant.quantile<i4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:1.000000e+00>
!quantileFloatType = !QuantileFloat.quantileFloat<4, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>
!qElemType = !quant.quantile<i4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:1.000000e+00>

// CHECK-LABEL: @NF4ConvertToDequantize
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>
// CHECK-SAME:  [[ARG1:%.+]]: tensor<64x64x1x1x!QuantileFloat.quantileFloat<4, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>
func.func @NF4ConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1x!quantileFloatType>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1x!quantileFloatType> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[QUANT_CAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType} : tensor<64x64x1x1x!QuantileFloat.quantileFloat<4, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<64x64x1x1x!qElemType>
  // CHECK:              [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f16} : tensor<64x64x1x1x!qElemType> -> tensor<64x64x1x1xf16>
  // CHECK:              [[RES:%.+]] = IE.Convolution([[ARG0]], [[DEQUANTIZE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[RES]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-DAG:  [[Q_ELEM_TYPE0:!.*]] = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK-DAG:  [[Q_ELEM_TYPE1:!.*]] = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK-DAG:  [[Q_ELEM_TYPE2:!.*]] = !quant.uniform<u4:f16, 0.0057189941406250002:8>
!qElemType = !quant.uniform<u4:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u4:f16, 0.0057189941406250002:8>

// CHECK:      ConvertToDequantizeForU4Weights
// CHECK-SAME: ([[ARG0:%.*]]: tensor<1x64x64x100xf16, {order = #NHWC}>, [[ARG1:%.*]]: tensor<64x64x1x1xui4>) -> tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @ConvertToDequantizeForU4WeightsWithQuantDequant(
    %arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xui4>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xui4> -> tensor<64x64x1x1xf16>
  %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<64x64x1x1xf16> -> tensor<64x64x1x1x!qElemType>
  %2 = IE.QuantizeCast(%1) {dstElemType = !qElemType1} : tensor<64x64x1x1x!qElemType> -> tensor<64x64x1x1x!qElemType1>
  %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<64x64x1x1x!qElemType1> -> tensor<64x64x1x1xf16>
  %4 = IE.Convolution(%arg0, %3)
      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %4 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK-NOT: IE.Convert

  // CHECK:       [[VAL0:%.*]] = IE.QuantizeCast([[ARG1:%.*]])
  // CHECK-SAME:      {dstElemType = [[Q_ELEM_TYPE0]]}
  // CHECK-SAME:        : tensor<64x64x1x1xui4> -> tensor<64x64x1x1x[[Q_ELEM_TYPE0]]>
  // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16}
  // CHECK-SAME:        : tensor<64x64x1x1x[[Q_ELEM_TYPE0]]> -> tensor<64x64x1x1xf16>

  // CHECK:       [[VAL2:%.*]] = IE.Quantize([[VAL1]]) {dstElemType = [[Q_ELEM_TYPE1]]}

  // CHECK:       [[VAL3:%.*]] = IE.QuantizeCast([[VAL2]]) {dstElemType = [[Q_ELEM_TYPE2]]}

  // CHECK:       [[VAL4:%.*]] = IE.Dequantize([[VAL3]]) {dstElemType = f16}

  // CHECK:       [[VAL5:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL4]])
  // CHECK-SAME:       : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:       return [[VAL5]]
}
