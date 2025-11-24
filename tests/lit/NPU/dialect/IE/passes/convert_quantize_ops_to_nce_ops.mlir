//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-quantize-ops-to-nce-ops  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK:  !qElemType = !quant.uniform<u8:f32, 2.000000e+00>
// CHECK:  !qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>
// CHECK:  !qElemType2 = !quant.uniform<u8:f32, 5.000000e-01>

// CHECK-LABEL:  func.func @ConvertQuantizeToEltwise
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x4xf32>)
func.func @ConvertQuantizeToEltwise(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf32> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x4x!qElemType> -> tensor<1x4xf32>

    // CHECK:  [[VAL0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4xf32>, tensor<1x4xf32> -> tensor<1x4x!qElemType>
    // CHECK:  [[VAL1:%.+]] = IE.QuantizeCast([[VAL0]]) {dstElemType = !qElemType1} : tensor<1x4x!qElemType> -> tensor<1x4x!qElemType1>
    // CHECK:  [[VAL2:%.+]] = IE.QuantizeCast([[VAL1]]) {dstElemType = !qElemType2} : tensor<1x4x!qElemType1> -> tensor<1x4x!qElemType2>
    // CHECK:  [[VAL3:%.+]] = IE.Add([[VAL2]], [[VAL2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x!qElemType2>, tensor<1x4x!qElemType2> -> tensor<1x4xf32>

    return %1 : tensor<1x4xf32>

    // CHECK:  return [[VAL3]] : tensor<1x4xf32>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>

// CHECK:  !qElemType = !quant.uniform<u8:f16:1, {0.95599999999999996:128,7.850000e-01:128,5.670000e-01:128}>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL:  func.func @ConvertQuantizeToDwConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x3x16x16xf16>)
func.func @ConvertQuantizeToDwConv(%arg0 : tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    // CHECK-DAG:   [[VAL0:%.+]] = const.Declare tensor<3x1x1x1xf16> = dense<1.000000e+00> : tensor<3x1x1x1xf16>
    // CHECK:       [[VAL1:%.+]] = IE.GroupConvolution([[INPUT]], [[VAL0]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 3 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x3x16x16xf16>, tensor<3x1x1x1xf16> -> tensor<1x3x16x16x!qElemType>

    // CHECK-DAG:   [[VAL2:%.+]] = const.Declare tensor<3x1x1x1x!qElemType1> = dense<1.000000e+00> : tensor<3x1x1x1xf16>, [#const.CastElemType<!qElemType1>]
    // CHECK:       [[VAL3:%.+]] = IE.GroupConvolution([[VAL1]], [[VAL2]])  {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 3 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x3x16x16x!qElemType>, tensor<3x1x1x1x!qElemType1> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>

    // CHECK:  return [[VAL3]] : tensor<1x3x16x16xf16>
}

// -----

!qElemType = !quant.uniform<u8:f32:1, {0.956:128, 0.785:128, 0.567:128}>

// CHECK:  !qElemType = !quant.uniform<u8:f32:1, {0.95599999999999996:128,7.850000e-01:128,5.670000e-01:128}>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL:  func.func @ConvertQuantizeToDwConvFP32Output
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x3x16x16xf32>)
func.func @ConvertQuantizeToDwConvFP32Output(%arg0 : tensor<1x3x16x16xf32>) -> tensor<1x3x16x16xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf32> -> tensor<1x3x16x16x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf32>

    // CHECK-DAG:   [[VAL0:%.+]] = const.Declare tensor<3x1x1x1xf32> = dense<1.000000e+00> : tensor<3x1x1x1xf32>
    // CHECK:       [[VAL1:%.+]] = IE.GroupConvolution([[INPUT]], [[VAL0]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 3 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x3x16x16xf32>, tensor<3x1x1x1xf32> -> tensor<1x3x16x16x!qElemType>

    // CHECK-DAG:   [[VAL2:%.+]] = const.Declare tensor<3x1x1x1x!qElemType1> = dense<1.000000e+00> : tensor<3x1x1x1xf16>, [#const.CastElemType<!qElemType1>]
    // CHECK:       [[VAL3:%.+]] = IE.GroupConvolution([[VAL1]], [[VAL2]])  {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 3 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x3x16x16x!qElemType>, tensor<3x1x1x1x!qElemType1> -> tensor<1x3x16x16xf32>

    return %1 : tensor<1x3x16x16xf32>

    // CHECK:  return [[VAL3]] : tensor<1x3x16x16xf32>
}

// -----

!qElemType = !quant.uniform<u8:f32, 2.000000e+00>

// CHECK:  !qElemType = !quant.uniform<u8:f32, 2.000000e+00>

// CHECK-LABEL:  func.func @ConvertQuantizeToAvgPool
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x400x800x400xf32>)
func.func @ConvertQuantizeToAvgPool(%arg0 : tensor<1x400x800x400xf32>) -> tensor<1x400x800x400xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x400x800x400xf32> -> tensor<1x400x800x400x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x400x800x400x!qElemType> -> tensor<1x400x800x400xf32>
    return %1 : tensor<1x400x800x400xf32>

    // CHECK:  [[AVGPOOL_0:%.+]] = IE.AvgPool([[INPUT]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x400x800x400xf32> -> tensor<1x400x800x400x!qElemType>
    // CHECK:  [[AVGPOOL_1:%.+]] = IE.AvgPool([[AVGPOOL_0]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x400x800x400x!qElemType> -> tensor<1x400x800x400xf32>
    // CHECK:  return [[AVGPOOL_1]] : tensor<1x400x800x400xf32>
}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.004>

// CHECK:  !qElemType = !quant.uniform<f8E4M3FN:f16, 4.000000e-03>

// CHECK-LABEL:  func.func @Float8
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x8x4x64xf16>)
func.func @Float8(%arg0 : tensor<1x8x4x64xf16>) -> tensor<1x8x4x64xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x8x4x64xf16> -> tensor<1x8x4x64x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x8x4x64x!qElemType> -> tensor<1x8x4x64xf16>
    return %1 : tensor<1x8x4x64xf16>

    // CHECK:  [[AVGPOOL_0:%.+]] = IE.AvgPool([[INPUT]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x8x4x64xf16> -> tensor<1x8x4x64x!qElemType>
    // CHECK:  [[AVGPOOL_1:%.+]] = IE.AvgPool([[AVGPOOL_0]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x8x4x64x!qElemType> -> tensor<1x8x4x64xf16>
    // CHECK:  return [[AVGPOOL_1]] : tensor<1x8x4x64xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:3, {
    0.956:128, 0.785:128, 0.567:128, 0.487:128,
    0.957:128, 0.786:128, 0.568:128, 0.488:128,
    0.958:128, 0.787:128, 0.569:128, 0.489:128,
    0.959:128, 0.788:128, 0.560:128, 0.480:128}>
!qElemType1 = !quant.uniform<u8<0:254>:f16, 1.000000e+00>

// CHECK-LABEL:  func.func @NotConvertQuantizeToDwConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x3x16x16xf16>)
func.func @NotConvertQuantizeToDwConv(%arg0 : tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>

    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    // CHECK:  return [[DEQUANTIZE]] : tensor<1x3x16x16xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 0.013961060374390846:109>
!dynType = tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
!dynQuantType = tensor<1x1x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:  !qElemType = !quant.uniform<u8:f16, 0.027922120748781691:109>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16, 0.013961060374390846:109>
// CHECK:  !qElemType2 = !quant.uniform<u8:f16, 0.0069805301871954228:109>

// CHECK-LABEL:  func.func @DynamicShapeToAdd
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>)
func.func @DynamicShapeToAdd(%arg0 : !dynType) -> !dynType {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : !dynType -> !dynQuantType
    %1 = IE.Dequantize(%0) {dstElemType = f16} : !dynQuantType -> !dynType
    return %1 : !dynType

    // CHECK:       [[Q_ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME:    tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[Q_CAST:%.+]] = IE.QuantizeCast([[Q_ADD]]) {dstElemType = !qElemType1} :
    // CHECK-SAME:    tensor<1x1x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x2x!qElemType1, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:       [[D_CAST:%.+]] = IE.QuantizeCast([[Q_CAST]]) {dstElemType = !qElemType2} :
    // CHECK-SAME:    tensor<1x1x?x2x!qElemType1, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x2x!qElemType2, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[D_ADD:%.+]] = IE.Add([[D_CAST]], [[D_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME:    tensor<1x1x?x2x!qElemType2, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x2x!qElemType2, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:  return [[D_ADD]] : tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 0.013961060374390846:109>
!dynType = tensor<1x800x?x800xf16, {bounds = #const.OpaqueI64Elements<[1, 800, 500, 800]> : tensor<4xsi64>, order = #NCHW}>
!dynQuantType = tensor<1x800x?x800x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 800, 500, 800]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:  !qElemType = !quant.uniform<u8:f16, 0.013961060374390846:109>

// CHECK-LABEL:  func.func @DynamicShapeToAvgPool
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x800x?x800xf16, {bounds = #const.OpaqueI64Elements<[1, 800, 500, 800]> : tensor<4xsi64>, order = #NCHW}>)
func.func @DynamicShapeToAvgPool(%arg0 : !dynType) -> !dynType {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : !dynType -> !dynQuantType
    %1 = IE.Dequantize(%0) {dstElemType = f16} : !dynQuantType -> !dynType
    return %1 : !dynType

    // CHECK:       [[Q_AVGPOOL:%.+]] = IE.AvgPool([[INPUT]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} :
    // CHECK-SAME:    tensor<1x800x?x800xf16, {bounds = #const.OpaqueI64Elements<[1, 800, 500, 800]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x800x?x800x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 800, 500, 800]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:       [[D_AVGPOOL:%.+]] = IE.AvgPool([[Q_AVGPOOL]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} :
    // CHECK-SAME:     tensor<1x800x?x800x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 800, 500, 800]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x800x?x800xf16, {bounds = #const.OpaqueI64Elements<[1, 800, 500, 800]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:  return [[D_AVGPOOL]] : tensor<1x800x?x800xf16, {bounds = #const.OpaqueI64Elements<[1, 800, 500, 800]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>
!dynType = tensor<4x3x?x2xf16, {bounds = #const.OpaqueI64Elements<[4, 3, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
!dynQuantType = tensor<4x3x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[4, 3, 500, 2]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:  !qElemType = !quant.uniform<u8:f16:1, {0.95599999999999996:128,7.850000e-01:128,5.670000e-01:128}>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL:  func.func @DynamicShapeToDwConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<4x3x?x2xf16, {bounds = #const.OpaqueI64Elements<[4, 3, 500, 2]> : tensor<4xsi64>, order = #NCHW}>)
func.func @DynamicShapeToDwConv(%arg0 : !dynType) -> !dynType {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : !dynType -> !dynQuantType
    %1 = IE.Dequantize(%0) {dstElemType = f16} : !dynQuantType -> !dynType
    return %1 : !dynType

    // CHECK:       [[Q_FILTER:%.+]] = const.Declare tensor<3x1x1x1xf16> = dense<1.000000e+00> : tensor<3x1x1x1xf16>
    // CHECK:       [[Q_DWCONV:%.+]] = IE.GroupConvolution([[INPUT]], [[Q_FILTER]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:    tensor<4x3x?x2xf16, {bounds = #const.OpaqueI64Elements<[4, 3, 500, 2]> : tensor<4xsi64>, order = #NCHW}>, tensor<3x1x1x1xf16> -> tensor<4x3x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[4, 3, 500, 2]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:       [[D_FILTER:%.+]] = const.Declare tensor<3x1x1x1x!qElemType1> = dense<1.000000e+00> : tensor<3x1x1x1xf16>, [#const.CastElemType<!qElemType1>]
    // CHECK:       [[D_DWCONV:%.+]] = IE.GroupConvolution([[Q_DWCONV]], [[D_FILTER]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:    tensor<4x3x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[4, 3, 500, 2]> : tensor<4xsi64>, order = #NCHW}>, tensor<3x1x1x1x!qElemType1> -> tensor<4x3x?x2xf16, {bounds = #const.OpaqueI64Elements<[4, 3, 500, 2]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:  return [[D_DWCONV]] : tensor<4x3x?x2xf16, {bounds = #const.OpaqueI64Elements<[4, 3, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

!qElemType = !quant.uniform<i16:f16, 1.000000e+00>

// CHECK:  !qElemType = !quant.uniform<i16:f16, 1.000000e+00>

// CHECK-LABEL:  func.func @NotConvertI16Quantize2DShape
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x4xf16>)
func.func @NotConvertI16Quantize2DShape(%arg0 : tensor<1x4xf16>) -> tensor<1x4xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf16> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x4x!qElemType> -> tensor<1x4xf16>

    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x4xf16> -> tensor<1x4x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x4x!qElemType> -> tensor<1x4xf16>

    return %1 : tensor<1x4xf16>

    // CHECK:  return [[DEQUANTIZE]] : tensor<1x4xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK:  !qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL:  func.func @NotConvertSubByteI4Quantize
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x4xf16>)
func.func @NotConvertSubByteI4Quantize(%arg0 : tensor<1x4xf16>) -> tensor<1x4xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf16> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x4x!qElemType> -> tensor<1x4xf16>

    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x4xf16> -> tensor<1x4x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x4x!qElemType> -> tensor<1x4xf16>

    return %1 : tensor<1x4xf16>

    // CHECK:  return [[DEQUANTIZE]] : tensor<1x4xf16>
}

// -----

!qElemType = !quant.uniform<u4:f16, 1.000000e+00>

// CHECK:  !qElemType = !quant.uniform<u4:f16, 1.000000e+00>

// CHECK-LABEL:  func.func @NotConvertSubByteI4Quantize
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x4xf16>)
func.func @NotConvertSubByteI4Quantize(%arg0 : tensor<1x4xf16>) -> tensor<1x4xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf16> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x4x!qElemType> -> tensor<1x4xf16>

    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x4xf16> -> tensor<1x4x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x4x!qElemType> -> tensor<1x4xf16>

    return %1 : tensor<1x4xf16>

    // CHECK:  return [[DEQUANTIZE]] : tensor<1x4xf16>
}

// -----

!qElemType = !quant.uniform<i2:f16, 1.000000e+00>

// CHECK:   !qElemType = !quant.uniform<i2:f16, 1.000000e+00>

// CHECK-LABEL:  func.func @NotConvertSubByteI2Quantize
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x4xf16>)
func.func @NotConvertSubByteI2Quantize(%arg0 : tensor<1x4xf16>) -> tensor<1x4xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf16> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x4x!qElemType> -> tensor<1x4xf16>

    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x4xf16> -> tensor<1x4x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x4x!qElemType> -> tensor<1x4xf16>

    return %1 : tensor<1x4xf16>

    // CHECK:  return [[DEQUANTIZE]] : tensor<1x4xf16>
}

// -----

!qElemType = !quant.quantile<u4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:0.07874348958333334>

// CHECK:   !qElemType = !quant.quantile<u4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:0.07874348958333334>

// CHECK-LABEL:  func.func @NotConvertSubByteNF4Quantize
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x4xf16>)
func.func @NotConvertSubByteNF4Quantize(%arg0 : tensor<1x4xf16>) -> tensor<1x4xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf16> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x4x!qElemType> -> tensor<1x4xf16>

    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x4xf16> -> tensor<1x4x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x4x!qElemType> -> tensor<1x4xf16>

    return %1 : tensor<1x4xf16>

    // CHECK:  return [[DEQUANTIZE]] : tensor<1x4xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16:1, {0.956:8, 0.785:8, 0.567:8, 0.956:8}>

// CHECK:  !qElemType = !quant.uniform<i4:f16:1, {0.95599999999999996:8,7.850000e-01:8,5.670000e-01:8,0.95599999999999996:8}>

// CHECK-LABEL:  func.func @NotConvertSubByteI4PerAxisQuantize
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x4xf16>)
func.func @NotConvertSubByteI4PerAxisQuantize(%arg0 : tensor<1x4xf16>) -> tensor<1x4xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf16> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x4x!qElemType> -> tensor<1x4xf16>

    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x4xf16> -> tensor<1x4x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x4x!qElemType> -> tensor<1x4xf16>

    return %1 : tensor<1x4xf16>

    // CHECK:  return [[DEQUANTIZE]] : tensor<1x4xf16>
}
