//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-to-quantized-ops %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertToDequantize
func.func @ConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi8>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.+]] = IE.QuantizeCast([[ARG1:%.+]]) {dstElemType = !qElemType} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1x!qElemType>
  // CHECK:              [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<64x64x1x1x!qElemType> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL2:%.+]] = IE.Convolution([[ARG0:%.+]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-DAG:  [[Q_ELEM_TYPE0:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG:  [[Q_ELEM_TYPE1:!.+]] = !quant.uniform<u8:f16, 1.000000e+00:128>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK:      ConvertToDequantizeWithQuantInput
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x64x64x100x[[Q_ELEM_TYPE1]], {order = #NHWC}>, [[ARG1:%.+]]: tensor<64x64x1x1xsi8>) -> tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @ConvertToDequantizeWithQuantInput(%arg0: tensor<1x64x64x100x!qElemType1, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi8>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100x!qElemType1, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.+]] = IE.QuantizeCast([[ARG1:%.+]]) {dstElemType = [[Q_ELEM_TYPE0]]} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1x[[Q_ELEM_TYPE0]]>
  // CHECK:              [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<64x64x1x1x[[Q_ELEM_TYPE0]]> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL2:%.+]] = IE.Convolution([[ARG0:%.+]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100x[[Q_ELEM_TYPE1]], {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
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

  // CHECK:              [[VAL0:%.+]] = IE.QuantizeCast([[ARG1:%.+]]) {dstElemType = !qElemType} : tensor<64x64x1xsi8> -> tensor<64x64x1x!qElemType>
  // CHECK:              [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<64x64x1x!qElemType> -> tensor<64x64x1xf16>
  // CHECK:              [[VAL2:%.+]] = IE.Reshape([[VAL1]]) {shape_value = [64, 64, 1, 1]} : tensor<64x64x1xf16> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL3:%.+]] = IE.Convolution([[ARG0:%.+]], [[VAL2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
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

  // CHECK:       [[VAL0:%.+]] = IE.QuantizeCast([[ARG1:%.+]]) {dstElemType = !qElemType}
  // CHECK-SAME:      : tensor<64x64x1x1xui8> -> tensor<64x64x1x1x!qElemType>
  // CHECK:       [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16}
  // CHECK-SAME:      : tensor<64x64x1x1x!qElemType> -> tensor<64x64x1x1xf16>
  // CHECK:  [[VAL2:%.+]] = IE.Convolution([[ARG0:%.+]], [[VAL1]])
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

  // CHECK:  [[VAL0:%.+]] = IE.Convert([[ARG0:%.+]]) {dstElemType = f16} : tensor<1x64x32x200xsi8> -> tensor<1x64x32x200xf16>
  // CHECK:  [[VAL1:%.+]] = IE.Reshape([[VAL0]]) {shape_value = [1, 64, 64, 100]} : tensor<1x64x32x200xf16> -> tensor<1x64x64x100xf16>
  // CHECK:  [[VAL2:%.+]] = IE.Convolution([[VAL1]], [[ARG1:%.+]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16>, tensor<1x64x64x64xf16> -> tensor<1x1x1x37xf16>
  // CHECK:  return [[VAL2]]
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

  // CHECK:              [[VAL0:%.+]] = IE.QuantizeCast([[ARG1:%.+]]) {dstElemType = !qElemType} : tensor<64x64x1x1xsi4> -> tensor<64x64x1x1x!qElemType>
  // CHECK:              [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<64x64x1x1x!qElemType> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL2:%.+]] = IE.Convolution([[ARG0:%.+]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @I1DontConvertToDequantize
func.func @I1DontConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xi1>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xi1> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.+]] = IE.Convert([[ARG1:%.+]]) {dstElemType = f16} : tensor<64x64x1x1xi1> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL1:%.+]] = IE.Convolution([[ARG0:%.+]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @I32DontConvertToDequantize
func.func @I32DontConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi32>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi32> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.+]] = IE.Convert([[ARG1:%.+]]) {dstElemType = f16} : tensor<64x64x1x1xsi32> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL1:%.+]] = IE.Convolution([[ARG0:%.+]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @I16DontConvertToDequantize
func.func @I16DontConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi16>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi16> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.+]] = IE.Convert([[ARG1:%.+]]) {dstElemType = f16} : tensor<64x64x1x1xsi16> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL1:%.+]] = IE.Convolution([[ARG0:%.+]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: !qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-6.961928e-01,-5.250731e-01,-3.949175e-01,-2.844414e-01,-1.847734e-01,-9.105004e-02,0.000000e+00,7.958030e-02,1.609302e-01,2.461123e-01,3.379152e-01,4.407098e-01,5.626170e-01,7.229568e-01,1.000000e+00}>:f16, 1.000000e+00>
!quantileType = !quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>
!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @NF4ConvertToDequantize
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>
// CHECK-SAME:  [[ARG1:%.+]]: tensor<64x64x1x1x!quantileType>
func.func @NF4ConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1x!quantileType>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1x!quantileType> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[QUANT_CAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType} : tensor<64x64x1x1x!quantileType> -> tensor<64x64x1x1x!qElemType>
  // CHECK:              [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f16} : tensor<64x64x1x1x!qElemType> -> tensor<64x64x1x1xf16>
  // CHECK:              [[RES:%.+]] = IE.Convolution([[ARG0]], [[DEQUANTIZE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[RES]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-DAG:  [[Q_ELEM_TYPE0:!.+]] = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK-DAG:  [[Q_ELEM_TYPE1:!.+]] = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK-DAG:  [[Q_ELEM_TYPE2:!.+]] = !quant.uniform<u4:f16, 0.0057189941406250002:8>
!qElemType = !quant.uniform<u4:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u4:f16, 0.0057189941406250002:8>

// CHECK:      ConvertToDequantizeForU4Weights
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<64x64x1x1xui4>) -> tensor<1x64x64x100xf16, {order = #NHWC}>
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

  // CHECK:       [[VAL0:%.+]] = IE.QuantizeCast([[ARG1:%.+]])
  // CHECK-SAME:      {dstElemType = [[Q_ELEM_TYPE0]]}
  // CHECK-SAME:        : tensor<64x64x1x1xui4> -> tensor<64x64x1x1x[[Q_ELEM_TYPE0]]>
  // CHECK:       [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16}
  // CHECK-SAME:        : tensor<64x64x1x1x[[Q_ELEM_TYPE0]]> -> tensor<64x64x1x1xf16>

  // CHECK:       [[VAL2:%.+]] = IE.Quantize([[VAL1]]) {dstElemType = [[Q_ELEM_TYPE1]]}

  // CHECK:       [[VAL3:%.+]] = IE.QuantizeCast([[VAL2]]) {dstElemType = [[Q_ELEM_TYPE2]]}

  // CHECK:       [[VAL4:%.+]] = IE.Dequantize([[VAL3]]) {dstElemType = f16}

  // CHECK:       [[VAL5:%.+]] = IE.Convolution([[ARG0:%.+]], [[VAL4]])
  // CHECK-SAME:       : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:       return [[VAL5]]
}

// -----

// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: func.func @ConvertToQuantizeBasicSI8
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16>
func.func @ConvertToQuantizeBasicSI8(%arg0: tensor<1x64x64x100xf16>) -> tensor<1x64x64x100xsi8> {
  %0 = IE.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %1 = IE.Clamp(%0) {min = -128.0, max = 127.0} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %2 = IE.Convert(%1) {dstElemType = si8} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xsi8>
  return %2 : tensor<1x64x64x100xsi8>

  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QELEMTYPE]]}
  // CHECK-SAME:      : tensor<1x64x64x100xf16> -> tensor<1x64x64x100x[[QELEMTYPE]]>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = si8}
  // CHECK-SAME:      : tensor<1x64x64x100x[[QELEMTYPE]]> -> tensor<1x64x64x100xsi8>
  // CHECK:       return [[QUANTIZE_CAST]]
}

// -----

// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK: func.func @ConvertToQuantizeBasicUI8
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16>
func.func @ConvertToQuantizeBasicUI8(%arg0: tensor<1x64x64x100xf16>) -> tensor<1x64x64x100xui8> {
  %0 = IE.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %1 = IE.Clamp(%0) {min = 0.0, max = 255.0} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %2 = IE.Convert(%1) {dstElemType = ui8} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xui8>
  return %2 : tensor<1x64x64x100xui8>

  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QELEMTYPE]]}
  // CHECK-SAME:      : tensor<1x64x64x100xf16> -> tensor<1x64x64x100x[[QELEMTYPE]]>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = ui8}
  // CHECK-SAME:      : tensor<1x64x64x100x[[QELEMTYPE]]> -> tensor<1x64x64x100xui8>
  // CHECK:       return [[QUANTIZE_CAST]]
}

// -----

// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: func.func @ConvertToQuantizeAlternatePattern
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16>
func.func @ConvertToQuantizeAlternatePattern(%arg0: tensor<1x64x64x100xf16>) -> tensor<1x64x64x100xsi8> {
  %0 = IE.Clamp(%arg0) {min = -128.0, max = 127.0} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %1 = IE.Round(%0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %2 = IE.Convert(%1) {dstElemType = si8} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xsi8>
  return %2 : tensor<1x64x64x100xsi8>

  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QELEMTYPE]]}
  // CHECK-SAME:      : tensor<1x64x64x100xf16> -> tensor<1x64x64x100x[[QELEMTYPE]]>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = si8}
  // CHECK-SAME:      : tensor<1x64x64x100x[[QELEMTYPE]]> -> tensor<1x64x64x100xsi8>
  // CHECK:       return [[QUANTIZE_CAST]]
}

// -----

// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK: func.func @ConvertToQuantizeSI4
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16>
func.func @ConvertToQuantizeSI4(%arg0: tensor<1x64x64x100xf16>) -> tensor<1x64x64x100xsi4> {
  %0 = IE.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %1 = IE.Clamp(%0) {min = -8.0, max = 7.0} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %2 = IE.Convert(%1) {dstElemType = si4} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xsi4>
  return %2 : tensor<1x64x64x100xsi4>

  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QELEMTYPE]]}
  // CHECK-SAME:      : tensor<1x64x64x100xf16> -> tensor<1x64x64x100x[[QELEMTYPE]]>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = si4}
  // CHECK-SAME:      : tensor<1x64x64x100x[[QELEMTYPE]]> -> tensor<1x64x64x100xsi4>
  // CHECK:       return [[QUANTIZE_CAST]]
}

// -----

// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<u4:f16, 1.000000e+00>

// CHECK: func.func @ConvertToQuantizeUI4
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16>
func.func @ConvertToQuantizeUI4(%arg0: tensor<1x64x64x100xf16>) -> tensor<1x64x64x100xui4> {
  %0 = IE.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %1 = IE.Clamp(%0) {min = 0.0, max = 15.0} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %2 = IE.Convert(%1) {dstElemType = ui4} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xui4>
  return %2 : tensor<1x64x64x100xui4>

  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QELEMTYPE]]}
  // CHECK-SAME:      : tensor<1x64x64x100xf16> -> tensor<1x64x64x100x[[QELEMTYPE]]>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = ui4}
  // CHECK-SAME:      : tensor<1x64x64x100x[[QELEMTYPE]]> -> tensor<1x64x64x100xui4>
  // CHECK:       return [[QUANTIZE_CAST]]
}

// -----

// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 5.000000e-01>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ConvertToQuantizeWithGroupConvScale
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @ConvertToQuantizeWithGroupConvScale(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>) -> tensor<1x64x64x100xui8, {order = #NHWC}> {
  %cst_filter = const.Declare tensor<64x1x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<64x1x1x1xf16, {order = #NHWC}>
  %0 = IE.GroupConvolution(%arg0, %cst_filter) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x1x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %1 = IE.Round(%0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %2 = IE.Clamp(%1) {min = 0.0, max = 255.0} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %3 = IE.Convert(%2) {dstElemType = ui8} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  return %3 : tensor<1x64x64x100xui8, {order = #NHWC}>

  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QELEMTYPE]]}
  // CHECK-SAME:      : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100x[[QELEMTYPE]], {order = #NHWC}>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = ui8}
  // CHECK-SAME:      : tensor<1x64x64x100x[[QELEMTYPE]], {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  // CHECK:       return [[QUANTIZE_CAST]]
}

// -----

// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 1.000000e+00>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ConvertToQuantizeWithGroupConvZeroScale
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @ConvertToQuantizeWithGroupConvZeroScale(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>) -> tensor<1x64x64x100xui8, {order = #NHWC}> {
  %cst_filter = const.Declare tensor<64x1x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<64x1x1x1xf16, {order = #NHWC}>
  %0 = IE.GroupConvolution(%arg0, %cst_filter) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x1x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %1 = IE.Round(%0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %2 = IE.Clamp(%1) {min = 0.0, max = 255.0} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %3 = IE.Convert(%2) {dstElemType = ui8} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  return %3 : tensor<1x64x64x100xui8, {order = #NHWC}>

  // CHECK:       [[CST_FILTER:%.+]] = const.Declare
  // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[ARG0]], [[CST_FILTER]])
  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[GROUPCONV]]) {dstElemType = [[QELEMTYPE]]}
  // CHECK-SAME:      : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100x[[QELEMTYPE]], {order = #NHWC}>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = ui8}
  // CHECK-SAME:      : tensor<1x64x64x100x[[QELEMTYPE]], {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  // CHECK:       return [[QUANTIZE_CAST]]
}

// -----

// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 5.000000e-01:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ConvertToQuantizeWithGroupConvScaleAndBias
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @ConvertToQuantizeWithGroupConvScaleAndBias(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>) -> tensor<1x64x64x100xui8, {order = #NHWC}> {
  %cst_filter = const.Declare tensor<64x1x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<64x1x1x1xf16, {order = #NHWC}>
  %cst_bias = const.Declare tensor<1x64x1x1xf16> = dense<128.0> : tensor<1x64x1x1xf16>
  %0 = IE.GroupConvolution(%arg0, %cst_filter, %cst_bias) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x1x1x1xf16, {order = #NHWC}>, tensor<1x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %1 = IE.Round(%0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %2 = IE.Clamp(%1) {min = 0.0, max = 255.0} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %3 = IE.Convert(%2) {dstElemType = ui8} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  return %3 : tensor<1x64x64x100xui8, {order = #NHWC}>

  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QELEMTYPE]]}
  // CHECK-SAME:      : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100x[[QELEMTYPE]], {order = #NHWC}>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = ui8}
  // CHECK-SAME:      : tensor<1x64x64x100x[[QELEMTYPE]], {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  // CHECK:       return [[QUANTIZE_CAST]]
}

// -----

// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 5.000000e-01:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ConvertToQuantizeWithGroupConvBiasHalfToEvenRound
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @ConvertToQuantizeWithGroupConvBiasHalfToEvenRound(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>) -> tensor<1x64x64x100xui8, {order = #NHWC}> {
  %cst_filter = const.Declare tensor<64x1x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<64x1x1x1xf16, {order = #NHWC}>
  %cst_bias = const.Declare tensor<1x64x1x1xf16> = dense<128.5> : tensor<1x64x1x1xf16>
  %0 = IE.GroupConvolution(%arg0, %cst_filter, %cst_bias) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x1x1x1xf16, {order = #NHWC}>, tensor<1x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %1 = IE.Round(%0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %2 = IE.Clamp(%1) {min = 0.0, max = 255.0} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %3 = IE.Convert(%2) {dstElemType = ui8} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  return %3 : tensor<1x64x64x100xui8, {order = #NHWC}>

  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QELEMTYPE]]}
}

// -----

// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 5.000000e-01:129>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ConvertToQuantizeWithGroupConvBiasHalfAwayFromZeroRound
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @ConvertToQuantizeWithGroupConvBiasHalfAwayFromZeroRound(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>) -> tensor<1x64x64x100xui8, {order = #NHWC}> {
  %cst_filter = const.Declare tensor<64x1x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<64x1x1x1xf16, {order = #NHWC}>
  %cst_bias = const.Declare tensor<1x64x1x1xf16> = dense<128.5> : tensor<1x64x1x1xf16>
  %0 = IE.GroupConvolution(%arg0, %cst_filter, %cst_bias) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x1x1x1xf16, {order = #NHWC}>, tensor<1x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %1 = IE.Round(%0) {mode = #IE.round_mode<HALF_AWAY_FROM_ZERO>} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %2 = IE.Clamp(%1) {min = 0.0, max = 255.0} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %3 = IE.Convert(%2) {dstElemType = ui8} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  return %3 : tensor<1x64x64x100xui8, {order = #NHWC}>

  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QELEMTYPE]]}
}

// -----

// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 1.000000e+00>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ConvertToQuantizeWithGroupConvWithPostOp
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @ConvertToQuantizeWithGroupConvWithPostOp(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>) -> tensor<1x64x64x100xui8, {order = #NHWC}> {
  %cst_filter = const.Declare tensor<64x1x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<64x1x1x1xf16, {order = #NHWC}>
  %0 = IE.GroupConvolution(%arg0, %cst_filter) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1], post_op = #IE.Relu<>} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x1x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %1 = IE.Round(%0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %2 = IE.Clamp(%1) {min = 0.0, max = 255.0} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %3 = IE.Convert(%2) {dstElemType = ui8} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  return %3 : tensor<1x64x64x100xui8, {order = #NHWC}>

  // CHECK:       [[CST_FILTER:%.+]] = const.Declare
  // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[ARG0]], [[CST_FILTER]])
  // CHECK-SAME:      post_op = #IE.Relu<>
  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[GROUPCONV]]) {dstElemType = [[QELEMTYPE]]}
  // CHECK-SAME:      : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100x[[QELEMTYPE]], {order = #NHWC}>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = ui8}
  // CHECK-SAME:      : tensor<1x64x64x100x[[QELEMTYPE]], {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  // CHECK:       return [[QUANTIZE_CAST]]
}

// -----

// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 1.000000e+00>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ConvertToQuantizeWithGroupConvWithClamp
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @ConvertToQuantizeWithGroupConvWithClamp(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>) -> tensor<1x64x64x100xui8, {order = #NHWC}> {
  %cst_filter = const.Declare tensor<64x1x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<64x1x1x1xf16, {order = #NHWC}>
  %0 = IE.GroupConvolution(%arg0, %cst_filter) {
    clamp = {min = 0.000000e+00 : f64, max = 1.000000e+00 : f64},
    dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
  } : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x1x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %1 = IE.Round(%0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %2 = IE.Clamp(%1) {min = 0.0, max = 255.0} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %3 = IE.Convert(%2) {dstElemType = ui8} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  return %3 : tensor<1x64x64x100xui8, {order = #NHWC}>

  // CHECK:       [[CST_FILTER:%.+]] = const.Declare
  // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[ARG0]], [[CST_FILTER]])
  // CHECK-SAME:      clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}
  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[GROUPCONV]]) {dstElemType = [[QELEMTYPE]]}
  // CHECK-SAME:      : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100x[[QELEMTYPE]], {order = #NHWC}>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = ui8}
  // CHECK-SAME:      : tensor<1x64x64x100x[[QELEMTYPE]], {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  // CHECK:       return [[QUANTIZE_CAST]]
}

// -----

// CHECK-DAG: [[QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 1.000000e+00>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ConvertToQuantizeWithNonEltwiseGroupConv
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @ConvertToQuantizeWithNonEltwiseGroupConv(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>) -> tensor<1x64x64x100xui8, {order = #NHWC}> {
  %cst_filter = const.Declare tensor<64x1x1x1xf16, {order = #NHWC}> = dense<[[[[2.0]]], [[[3.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]], [[[2.0]]]]> : tensor<64x1x1x1xf16, {order = #NHWC}>
  %0 = IE.GroupConvolution(%arg0, %cst_filter) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x1x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %1 = IE.Round(%0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %2 = IE.Clamp(%1) {min = 0.0, max = 255.0} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %3 = IE.Convert(%2) {dstElemType = ui8} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  return %3 : tensor<1x64x64x100xui8, {order = #NHWC}>

  // CHECK:       [[CST_FILTER:%.+]] = const.Declare
  // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[ARG0]], [[CST_FILTER]])
  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[GROUPCONV]]) {dstElemType = [[QELEMTYPE]]}
  // CHECK-SAME:      : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x64x100x[[QELEMTYPE]], {order = #NHWC}>
  // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = ui8}
  // CHECK-SAME:      : tensor<1x64x64x100x[[QELEMTYPE]], {order = #NHWC}> -> tensor<1x64x64x100xui8, {order = #NHWC}>
  // CHECK:       return [[QUANTIZE_CAST]]
}

// -----

// CHECK-LABEL: @DontConvertToQuantizeWithWrongClampRange
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16>
func.func @DontConvertToQuantizeWithWrongClampRange(%arg0: tensor<1x64x64x100xf16>) -> tensor<1x64x64x100xsi8> {
  %0 = IE.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %1 = IE.Clamp(%0) {min = -100.0, max = 100.0} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %2 = IE.Convert(%1) {dstElemType = si8} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xsi8>
  return %2 : tensor<1x64x64x100xsi8>

  // CHECK:       [[ROUND:%.+]] = IE.Round([[ARG0]])
  // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[ROUND]])
  // CHECK:       [[CONVERT:%.+]] = IE.Convert([[CLAMP]])
  // CHECK:       return [[CONVERT]]
}

// -----

// CHECK-LABEL: @DontConvertToQuantizeWithoutRound
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16>
func.func @DontConvertToQuantizeWithoutRound(%arg0: tensor<1x64x64x100xf16>) -> tensor<1x64x64x100xsi8> {
  %0 = IE.Clamp(%arg0) {min = -128.0, max = 127.0} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %1 = IE.Convert(%0) {dstElemType = si8} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xsi8>
  return %1 : tensor<1x64x64x100xsi8>

  // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[ARG0]])
  // CHECK:       [[CONVERT:%.+]] = IE.Convert([[CLAMP]])
  // CHECK:       return [[CONVERT]]
}

// -----

// CHECK-LABEL: @DontConvertToQuantizeWithoutClamp
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16>
func.func @DontConvertToQuantizeWithoutClamp(%arg0: tensor<1x64x64x100xf16>) -> tensor<1x64x64x100xsi8> {
  %0 = IE.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %1 = IE.Convert(%0) {dstElemType = si8} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xsi8>
  return %1 : tensor<1x64x64x100xsi8>

  // CHECK:       [[ROUND:%.+]] = IE.Round([[ARG0]])
  // CHECK:       [[CONVERT:%.+]] = IE.Convert([[ROUND]])
  // CHECK:       return [[CONVERT]]
}

// -----

// CHECK-LABEL: @I16DontConvertToQuantize
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x64x100xf16>
func.func @I16DontConvertToQuantize(%arg0: tensor<1x64x64x100xf16>) -> tensor<1x64x64x100xsi16> {
  %0 = IE.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %1 = IE.Clamp(%0) {min = -32768.0, max = 32767.0} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xf16>
  %2 = IE.Convert(%1) {dstElemType = si16} : tensor<1x64x64x100xf16> -> tensor<1x64x64x100xsi16>
  return %2 : tensor<1x64x64x100xsi16>

  // CHECK:       [[ROUND:%.+]] = IE.Round([[ARG0]])
  // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[ROUND]])
  // CHECK:       [[CONVERT:%.+]] = IE.Convert([[CLAMP]])
  // CHECK:       return [[CONVERT]]
}
