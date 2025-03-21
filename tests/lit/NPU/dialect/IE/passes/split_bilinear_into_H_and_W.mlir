//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-bilinear-into-H-and-W %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @SplitBilinearIntoHAndW
func.func @SplitBilinearIntoHAndW(%arg0: tensor<1x32x68x120xf16>) -> tensor<1x32x136x240xf16> {

    %0 = IE.Interpolate(%arg0)
    {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false,
    pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 32, 136, 240]
    } : tensor<1x32x68x120xf16> -> tensor<1x32x136x240xf16>
    return %0 : tensor<1x32x136x240xf16>

  // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate({{[^:]+}}) {
  // CHECK-SAME:  attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false,
  // CHECK-SAME:  pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
  // CHECK-SAME:  scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 32, 136, 120]} : tensor<1x32x68x120xf16> -> tensor<1x32x136x120xf16>
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[INTERPOLATE]], {{[^:]+}}) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
  // CHECK-SAME:  tensor<1x32x136x120xf16>, tensor<64x32x1x2xf16, {order = #NHWC}> -> tensor<1x64x136x119xf16>
  // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[CONV]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 32, 136, 238]} : tensor<1x64x136x119xf16> -> tensor<1x32x136x238xf16>
  // CHECK:       [[SLICE0:%.+]] = IE.Slice [[INTERPOLATE]] [0, 0, 0, 0] [1, 32, 136, 1] : tensor<1x32x136x120xf16> to tensor<1x32x136x1xf16>
  // CHECK:       [[SLICE1:%.+]] = IE.Slice [[INTERPOLATE]] [0, 0, 0, 119] [1, 32, 136, 1] : tensor<1x32x136x120xf16> to tensor<1x32x136x1xf16>
  // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SLICE0]], [[RESHAPE]], [[SLICE1]]) {per_axis = #IE.Concat<axis = 3 : i64>} :
  // CHECK-SAME:  tensor<1x32x136x1xf16>, tensor<1x32x136x238xf16>, tensor<1x32x136x1xf16> -> tensor<1x32x136x240xf16>

  // CHECK:       return [[CONCAT]] : tensor<1x32x136x240xf16>

}

// -----

// CHECK-LABEL: @SplitBilinearIntoHAndWChannel8
func.func @SplitBilinearIntoHAndWChannel8(%arg0: tensor<1x8x68x120xf16>) -> tensor<1x8x136x240xf16> {

    %0 = IE.Interpolate(%arg0)
    {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false,
    pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 8, 136, 240]
    } : tensor<1x8x68x120xf16> -> tensor<1x8x136x240xf16>
    return %0 : tensor<1x8x136x240xf16>

  // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate({{[^:]+}}) {
  // CHECK-SAME:  attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false,
  // CHECK-SAME:  pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
  // CHECK-SAME:  scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 8, 136, 120]} : tensor<1x8x68x120xf16> -> tensor<1x8x136x120xf16>
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[INTERPOLATE]], {{[^:]+}}) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
  // CHECK-SAME:  tensor<1x8x136x120xf16>, tensor<16x8x1x2xf16, {order = #NHWC}> -> tensor<1x16x136x119xf16>
  // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[CONV]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 8, 136, 238]} : tensor<1x16x136x119xf16> -> tensor<1x8x136x238xf16>
  // CHECK:       [[SLICE0:%.+]] = IE.Slice [[INTERPOLATE]] [0, 0, 0, 0] [1, 8, 136, 1] : tensor<1x8x136x120xf16> to tensor<1x8x136x1xf16>
  // CHECK:       [[SLICE1:%.+]] = IE.Slice [[INTERPOLATE]] [0, 0, 0, 119] [1, 8, 136, 1] : tensor<1x8x136x120xf16> to tensor<1x8x136x1xf16>
  // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SLICE0]], [[RESHAPE]], [[SLICE1]]) {per_axis = #IE.Concat<axis = 3 : i64>} :
  // CHECK-SAME:  tensor<1x8x136x1xf16>, tensor<1x8x136x238xf16>, tensor<1x8x136x1xf16> -> tensor<1x8x136x240xf16>

  // CHECK:       return [[CONCAT]] : tensor<1x8x136x240xf16>

}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
  // CHECK: !qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @DoNotSplitBilinearIntoHAndWQuantize
func.func @DoNotSplitBilinearIntoHAndWQuantize(%arg0: tensor<1x8x68x120x!qElemType>) -> tensor<1x8x136x240x!qElemType> {

    %0 = IE.Interpolate(%arg0)
    {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false,
    pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 8, 136, 240]
    } : tensor<1x8x68x120x!qElemType> -> tensor<1x8x136x240x!qElemType>
    return %0 : tensor<1x8x136x240x!qElemType>

  // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate({{[^:]+}}) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false,
  // CHECK-SAME:  pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
  // CHECK-SAME:  scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 8, 136, 240]} : tensor<1x8x68x120x!qElemType> -> tensor<1x8x136x240x!qElemType>
  // CHECK:       return [[INTERPOLATE]] : tensor<1x8x136x240x!qElemType>

}

// -----

// CHECK-LABEL: @DoNotSplitBilinearIntoHAndWChannel1
func.func @DoNotSplitBilinearIntoHAndWChannel1(%arg0: tensor<1x1x68x10xf16>) -> tensor<1x1x136x240xf16> {

    %0 = IE.Interpolate(%arg0)
    {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false,
    pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 1, 136, 240]
    } : tensor<1x1x68x10xf16> -> tensor<1x1x136x240xf16>
    return %0 : tensor<1x1x136x240xf16>


  // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate({{[^:]+}}) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false,
  // CHECK-SAME:  pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
  // CHECK-SAME:  scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 1, 136, 240]} : tensor<1x1x68x10xf16> -> tensor<1x1x136x240xf16>
  // CHECK:       return [[INTERPOLATE]] : tensor<1x1x136x240xf16>

}

// CHECK-LABEL: @DoNotSplitBilinearIntoHAndWNonIntegerScale
func.func @DoNotSplitBilinearIntoHAndWNonIntegerScale(%arg0: tensor<1x400x2x230xf16>) -> tensor<1x400x4x600xf16> {
    %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false,
    pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [4, 600]
    } : tensor<1x400x2x230xf16> -> tensor<1x400x4x600xf16>
    return %0 : tensor<1x400x4x600xf16>

  // CHECK:       [[INTERPOLATE:%.+]] = IE.Interpolate({{[^:]+}}) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false,
  // CHECK-SAME:  pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
  // CHECK-SAME:  scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [4, 600]} : tensor<1x400x2x230xf16> -> tensor<1x400x4x600xf16>
  // CHECK:       return [[INTERPOLATE]] : tensor<1x400x4x600xf16>

  }
