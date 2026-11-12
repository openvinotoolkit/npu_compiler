//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% revision-id=3" --convert-to-mixed-precision %s | FileCheck %s
// REQUIRES: platform-NPU6010
// Baseline: REVISION_B via revision-id=3.

!qElemType = !quant.uniform<u16:f32, 8.0507534851041623E-5:0>

// CHECK-LABEL: @GroupConvWithQuantizeSymmetricU16Quant
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x3x3xf16>
func.func @GroupConvWithQuantizeSymmetricU16Quant(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %cst = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1x1xf16>

    %0 = IE.GroupConvolution(%arg0, %cst) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<16x1x1x1xf16>

    // CHECK:   [[GCONV:%.+]] = IE.GroupConvolution([[ARG0]], [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 16 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[GCONV]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u16:f32, 8.0507534851041623E-5:3325>

// CHECK-LABEL: @GroupConvWithQuantizeAsymmetricU16Quant
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x3x3xf16>
func.func @GroupConvWithQuantizeAsymmetricU16Quant(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %cst = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1x1xf16>

    %0 = IE.GroupConvolution(%arg0, %cst) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<16x1x1x1xf16>

    // CHECK:   [[GCONV:%.+]] = IE.GroupConvolution([[ARG0]], [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 16 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[GCONV]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0}>:f16, 1.000000e+00>
// CHECK: !qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionDynamicDequantizeMaxpoolConvert
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType>,
// CHECK-SAME:  [[ARG2:%.+]]: tensor<1x16x1x1xf16>)
func.func @MixedPrecisionDynamicDequantizeMaxpoolConvert(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1x!qElemType>, %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
  %qweights = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  // CHECK:   [[MAXPOOL_CONVERT_SCALE:%.+]] = IE.MaxPool([[ARG2]]) {
   // CHECK-SAME:    kernel_size = [1, 1],
   // CHECK-SAME:    pads_begin = [0, 0],
   // CHECK-SAME:    pads_end = [0, 0],
   // CHECK-SAME:    strides = [1, 1]
   // CHECK-SAME:   }  : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf32>

  // CHECK:   [[RESHAPE_SCALE:%.+]] = IE.Reshape([[MAXPOOL_CONVERT_SCALE]]) {shape_value = [16, 1, 1, 1]} : tensor<1x16x1x1xf32> -> tensor<16x1x1x1xf32>

  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]], [[RESHAPE_SCALE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType>, tensor<16x1x1x1xf32> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0}>:f16, 1.000000e+00>
// CHECK: !qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionDynamicDequantizeNoMaxpoolConvert
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType>,
// CHECK-SAME:  [[ARG2:%.+]]: tensor<1x16x1x1xf32>)
func.func @MixedPrecisionDynamicDequantizeNoMaxpoolConvert(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1x!qElemType>, %arg2: tensor<1x16x1x1xf32>) -> tensor<1x16x16x16xf16> {
  %qweights = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf32> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]], [[ARG2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf32> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.0045223289302417213>
!qElemType1 = !quant.uniform<!quant.quantile<ui4:f16, {-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0}>:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<!quant.quantile<ui4:f16, {-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00}>:f16, 1.000000e+00>


// CHECK-LABEL: @MixedPrecisionDynamicDequantizeFP8U4
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType1>,
// CHECK-SAME:  [[ARG2:%.+]]: tensor<1x16x1x1xf16>)
func.func @MixedPrecisionDynamicDequantizeFP8U4(%arg0: tensor<1x16x16x16x!qElemType>, %arg1: tensor<16x16x1x1x!qElemType1>, %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
  %qactivation = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16xf16>
  %qweights = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%qactivation, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  // CHECK-NOT:     IE.Dequantize
  // CHECK:   [[MAXPOOL_CONVERT_SCALE:%.+]] = IE.MaxPool([[ARG2]]) {
   // CHECK-SAME:    kernel_size = [1, 1],
   // CHECK-SAME:    pads_begin = [0, 0],
   // CHECK-SAME:    pads_end = [0, 0],
   // CHECK-SAME:    strides = [1, 1]
   // CHECK-SAME:   }  : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf32>

  // CHECK:   [[RESHAPE_SCALE:%.+]] = IE.Reshape([[MAXPOOL_CONVERT_SCALE]]) {shape_value = [16, 1, 1, 1]} : tensor<1x16x1x1xf32> -> tensor<16x1x1x1xf32>

  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]], [[RESHAPE_SCALE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1x!qElemType1>, tensor<16x1x1x1xf32> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}


// -----

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.0045223289302417213>
!qElemType1 = !quant.uniform<ui8:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionDynamicDequantizeFP8U8
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16x!qElemType>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType1>,
// CHECK-SAME:  [[ARG2:%.+]]: tensor<1x16x1x1xf16>)
func.func @MixedPrecisionDynamicDequantizeFP8U8(%arg0: tensor<1x16x16x16x!qElemType>, %arg1: tensor<16x16x1x1x!qElemType1>, %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
  %qactivation = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16xf16>
  %qweights = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%qactivation, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  // CHECK:     [[DEQUANT:%.+]] = IE.Dequantize([[ARG0]]) {dstElemType = f16} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16xf16>
  // CHECK:     [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[ARG1]], [[ARG2]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

  //CHECK: [[CONV:%.+]] = IE.Convolution([[DEQUANT]], [[DYN_DEQUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0}>:f16, 1.000000e+00>
// CHECK: !qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionDynamicDequantizeMaxpoolConvert
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<16x16x1x1x!qElemType>,
// CHECK-SAME:  [[ARG2:%.+]]: tensor<1x16x1x1xf16>,
// CHECK-SAME:  [[ARG3:%.+]]: tensor<1x16x1x1xf16>)
func.func @MixedPrecisionDynamicDequantizeMaxpoolConvert(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1x!qElemType>, %arg2: tensor<1x16x1x1xf16>, %arg3: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
  %qweights = IE.DynamicDequantize(%arg1, %arg2) {dstElemType = f16} : tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>
  %scale_table = IE.MaxPool(%arg3) {
      kernel_size = [1, 1],
      pads_begin = [0, 0],
      pads_end = [0, 0],
      strides = [1, 1],
      rounding_type = #IE.rounding_type<FLOOR>
  } : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf32>
  %affine_reshape = IE.AffineReshape(%scale_table) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [16, 1, 1, 1]} : tensor<1x16x1x1xf32> -> tensor<16x1x1x1xf32>
  %result = IE.Convolution(%arg0, %qweights, , %affine_reshape) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>, tensor<16x1x1x1xf32> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  // CHECK:   [[LAYOUT_CAST_IN_1:%.+]] = IE.LayoutCast([[ARG3]]
  // CHECK:   [[LAYOUT_CAST_IN_2:%.+]] = IE.LayoutCast([[ARG2]]

  // CHECK:   [[MAXPOOL_CONVERT_SCALE:%.+]] = IE.Multiply([[LAYOUT_CAST_IN_1]], [[LAYOUT_CAST_IN_2]]) {
  // CHECK-SAME:   : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf32, {order = #NHWC}>

  // CHECK:   [[LAYOUT_CAST_OUT:%.+]] = IE.LayoutCast([[MAXPOOL_CONVERT_SCALE]]
  // CHECK:   [[RESHAPE:%.+]] = IE.Reshape([[LAYOUT_CAST_OUT]]

  //CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]], [[RESHAPE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType>, tensor<16x1x1x1xf32> -> tensor<1x16x16x16xf16>
  //CHECK: return [[CONV]]
}
