//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-and-fuse-quantize-dequantize="se-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateDequantInterpolate
func.func @PropagateDequantInterpolate(%arg0: tensor<1x16x48x80x!qElemType>) -> tensor<1x16x96x160xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x16x48x80x!qElemType> -> tensor<1x16x48x80xf16>
  %2 = IE.Interpolate(%1) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]
         } : tensor<1x16x48x80xf16> -> tensor<1x16x96x160xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x16x96x160xf16>, tensor<1x16x96x160xf16> -> tensor<1x16x96x160xf16>

  return %3 : tensor<1x16x96x160xf16>

  //CHECK: [[INTERPOLATE:%.*]] = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]} : tensor<1x16x48x80x!qElemType> -> tensor<1x16x96x160x!qElemType>
  //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[INTERPOLATE]]) {dstElemType = f16} : tensor<1x16x96x160x!qElemType> -> tensor<1x16x96x160xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x96x160xf16>, tensor<1x16x96x160xf16> -> tensor<1x16x96x160xf16>
  //CHECK: return [[ADD]] : tensor<1x16x96x160xf16>
 }

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantInterpolate
func.func @PropagateQuantInterpolate(%arg0: tensor<1x16x48x80xf16>) -> tensor<1x16x96x160x!qElemType> {
  %1 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]
         } : tensor<1x16x48x80xf16> -> tensor<1x16x96x160xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x16x96x160xf16> -> tensor<1x16x96x160x!qElemType>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x16x96x160x!qElemType>, tensor<1x16x96x160x!qElemType> -> tensor<1x16x96x160x!qElemType>

  return %3 : tensor<1x16x96x160x!qElemType>

  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x48x80xf16> -> tensor<1x16x48x80x!qElemType>
  //CHECK: [[INTERPOLATE:%.*]] = IE.Interpolate([[QUANTIZE]]) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]} : tensor<1x16x48x80x!qElemType> -> tensor<1x16x96x160x!qElemType>
  //CHECK: [[ADD:%.*]] = IE.Add([[INTERPOLATE]], [[INTERPOLATE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x96x160x!qElemType>, tensor<1x16x96x160x!qElemType> -> tensor<1x16x96x160x!qElemType>
  //CHECK: return [[ADD]] : tensor<1x16x96x160x!qElemType>
 }

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @DoNotPropagateDequantInterpolateBilinear
func.func @DoNotPropagateDequantInterpolateBilinear(%arg0: tensor<1x16x48x80x!qElemType>) -> tensor<1x16x96x160xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x16x48x80x!qElemType> -> tensor<1x16x48x80xf16>
  %2 = IE.Interpolate(%1) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]
         } : tensor<1x16x48x80xf16> -> tensor<1x16x96x160xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x16x96x160xf16>, tensor<1x16x96x160xf16> -> tensor<1x16x96x160xf16>

  return %3 : tensor<1x16x96x160xf16>

  //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x16x48x80x!qElemType> -> tensor<1x16x48x80xf16>
  //CHECK: [[INTERPOLATE:%.*]] = IE.Interpolate([[DEQUANTIZE]]) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]} : tensor<1x16x48x80xf16> -> tensor<1x16x96x160xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[INTERPOLATE]], [[INTERPOLATE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x96x160xf16>, tensor<1x16x96x160xf16> -> tensor<1x16x96x160xf16>
  //CHECK: return [[ADD]] : tensor<1x16x96x160xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @DoNotPropagateQuantInterpolateBilinear
func.func @DoNotPropagateQuantInterpolateBilinear(%arg0: tensor<1x16x48x80xf16>) -> tensor<1x16x96x160x!qElemType> {
  %1 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]
         } : tensor<1x16x48x80xf16> -> tensor<1x16x96x160xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x16x96x160xf16> -> tensor<1x16x96x160x!qElemType>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x16x96x160x!qElemType>, tensor<1x16x96x160x!qElemType> -> tensor<1x16x96x160x!qElemType>

  return %3 : tensor<1x16x96x160x!qElemType>

  //CHECK: [[INTERPOLATE:%.*]] = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]} : tensor<1x16x48x80xf16> -> tensor<1x16x96x160xf16>
  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[INTERPOLATE]]) {dstElemType = !qElemType} : tensor<1x16x96x160xf16> -> tensor<1x16x96x160x!qElemType>
  //CHECK: [[ADD:%.*]] = IE.Add([[QUANTIZE]], [[QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x96x160x!qElemType>, tensor<1x16x96x160x!qElemType> -> tensor<1x16x96x160x!qElemType>
  //CHECK: return [[ADD]] : tensor<1x16x96x160x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @DoNotPropagateDequantInterpolateBicubic
func.func @DoNotPropagateDequantInterpolateBicubic(%arg0: tensor<1x16x48x80x!qElemType>) -> tensor<1x16x96x160xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x16x48x80x!qElemType> -> tensor<1x16x48x80xf16>
  %2 = IE.Interpolate(%1) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <CUBIC>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]
         } : tensor<1x16x48x80xf16> -> tensor<1x16x96x160xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x16x96x160xf16>, tensor<1x16x96x160xf16> -> tensor<1x16x96x160xf16>

  return %3 : tensor<1x16x96x160xf16>

  //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x16x48x80x!qElemType> -> tensor<1x16x48x80xf16>
  //CHECK: [[INTERPOLATE:%.*]] = IE.Interpolate([[DEQUANTIZE]]) {attr = #IE.Interpolate<mode = <CUBIC>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]} : tensor<1x16x48x80xf16> -> tensor<1x16x96x160xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[INTERPOLATE]], [[INTERPOLATE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x96x160xf16>, tensor<1x16x96x160xf16> -> tensor<1x16x96x160xf16>
  //CHECK: return [[ADD]] : tensor<1x16x96x160xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @DoNotPropagateQuantInterpolateBicubic
func.func @DoNotPropagateQuantInterpolateBicubic(%arg0: tensor<1x16x48x80xf16>) -> tensor<1x16x96x160x!qElemType> {
  %1 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <CUBIC>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]
         } : tensor<1x16x48x80xf16> -> tensor<1x16x96x160xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x16x96x160xf16> -> tensor<1x16x96x160x!qElemType>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x16x96x160x!qElemType>, tensor<1x16x96x160x!qElemType> -> tensor<1x16x96x160x!qElemType>

  return %3 : tensor<1x16x96x160x!qElemType>

  //CHECK: [[INTERPOLATE:%.*]] = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <CUBIC>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]} : tensor<1x16x48x80xf16> -> tensor<1x16x96x160xf16>
  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[INTERPOLATE]]) {dstElemType = !qElemType} : tensor<1x16x96x160xf16> -> tensor<1x16x96x160x!qElemType>
  //CHECK: [[ADD:%.*]] = IE.Add([[QUANTIZE]], [[QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x96x160x!qElemType>, tensor<1x16x96x160x!qElemType> -> tensor<1x16x96x160x!qElemType>
  //CHECK: return [[ADD]] : tensor<1x16x96x160x!qElemType>
}
