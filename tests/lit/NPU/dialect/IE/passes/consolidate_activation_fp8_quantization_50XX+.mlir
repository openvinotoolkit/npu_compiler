//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --consolidate-activation-fp8-quantization %s | FileCheck %s
// REQUIRES: arch-NPU50XX

!qElemType = !quant.uniform<i4:f32, 1.000000e+00:8>
// CHECK:  !qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

// CHECK:  func.func @FP8ActivationI4WeightsPostFC
// CHECK-SAME:   ([[IN:%.+]]: tensor<1x3072xf32>, [[IN_SCALE:%.+]]: tensor<1x3072xf32>, [[FP8_SCALE:%.+]]: tensor<1xf32>,
// CHECK-SAME:    [[WT:%.+]]: tensor<9216x3072xsi4>, [[WT_SCALE:%.+]]: tensor<9216x1xf32>)
func.func @FP8ActivationI4WeightsPostFC(%in: tensor<1x3072xf32>, %in_scale: tensor<1x3072xf32>, %fp8_scale: tensor<1xf32>,
                                  %wt: tensor<9216x3072xsi4>, %wt_scale: tensor<9216x1xf32>) -> tensor<1x9216xf32> {
    %in_mul = IE.Multiply(%in, %in_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072xf32>, tensor<1x3072xf32> -> tensor<1x3072xf32>
    %in_fp8 = IE.FakeConvert(%in_mul, %fp8_scale) {dst_type = f8E4M3FN} : tensor<1x3072xf32>, tensor<1xf32> -> tensor<1x3072xf32>

    %wt_cast = IE.QuantizeCast(%wt) {dstElemType = !qElemType} : tensor<9216x3072xsi4> -> tensor<9216x3072x!qElemType>
    %wt_dynquant = IE.DynamicDequantize(%wt_cast, %wt_scale) {dstElemType = f32} : tensor<9216x3072x!qElemType>, tensor<9216x1xf32> -> tensor<9216x3072xf32>

    %fc = IE.FullyConnected(%in_fp8, %wt_dynquant) : tensor<1x3072xf32>, tensor<9216x3072xf32> -> tensor<1x9216xf32>

    return %fc : tensor<1x9216xf32>

    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[IN]], [[IN_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072xf32>, tensor<1x3072xf32> -> tensor<1x3072xf32>
    // CHECK:  [[ACT_RESHAPE:%.+]] = IE.Reshape([[MULTIPLY]]) {shape_value = [1, 1, 1, 3072]} : tensor<1x3072xf32> -> tensor<1x1x1x3072xf32>
    // CHECK:  [[WT_RESHAPE:%.+]] = IE.Reshape([[FP8_SCALE]]) {shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:  [[CONV:%.+]] = IE.GroupConvolution([[ACT_RESHAPE]], [[WT_RESHAPE]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x1x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x3072xf32>
    // CHECK:  [[OUT_RESHAPE:%.+]] = IE.Reshape([[CONV]]) {shape_value = [1, 3072]} : tensor<1x1x1x3072xf32> -> tensor<1x3072xf32>
    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[OUT_RESHAPE]]) {dstElemType = !qElemType} : tensor<1x3072xf32> -> tensor<1x3072x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f32} : tensor<1x3072x!qElemType> -> tensor<1x3072xf32>
    // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WT]]) {dstElemType = !qElemType1} : tensor<9216x3072xsi4> -> tensor<9216x3072x!qElemType1>
    // CHECK:  [[DYNDEQUANTIZE:%.+]] = IE.DynamicDequantize([[QUANTCAST]], [[WT_SCALE]]) {dstElemType = f32} : tensor<9216x3072x!qElemType1>, tensor<9216x1xf32> -> tensor<9216x3072xf32>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[DEQUANTIZE]], [[DYNDEQUANTIZE]]) : tensor<1x3072xf32>, tensor<9216x3072xf32> -> tensor<1x9216xf32>
    // CHECK:  [[DIVIDE:%.+]] = IE.Divide([[FC]], [[FP8_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x9216xf32>, tensor<1xf32> -> tensor<1x9216xf32>
    // CHECK:  return [[DIVIDE]]
}

// -----

!qElemType = !quant.uniform<i4:f32, 1.000000e+00:8>
// CHECK:  !qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

// CHECK:  func.func @FP8ActivationI4WeightsPostMatMul
// CHECK-SAME:   ([[IN:%.+]]: tensor<1x3072xf32>, [[IN_SCALE:%.+]]: tensor<1x3072xf32>, [[FP8_SCALE:%.+]]: tensor<1xf32>,
// CHECK-SAME:    [[WT:%.+]]: tensor<9216x3072xsi4>, [[WT_SCALE:%.+]]: tensor<9216x1xf32>)
func.func @FP8ActivationI4WeightsPostMatMul(%in: tensor<1x3072xf32>, %in_scale: tensor<1x3072xf32>, %fp8_scale: tensor<1xf32>,
                                  %wt: tensor<9216x3072xsi4>, %wt_scale: tensor<9216x1xf32>) -> tensor<1x9216xf32> {
    %in_mul = IE.Multiply(%in, %in_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072xf32>, tensor<1x3072xf32> -> tensor<1x3072xf32>
    %in_fp8 = IE.FakeConvert(%in_mul, %fp8_scale) {dst_type = f8E4M3FN} : tensor<1x3072xf32>, tensor<1xf32> -> tensor<1x3072xf32>

    %wt_cast = IE.QuantizeCast(%wt) {dstElemType = !qElemType} : tensor<9216x3072xsi4> -> tensor<9216x3072x!qElemType>
    %wt_dynquant = IE.DynamicDequantize(%wt_cast, %wt_scale) {dstElemType = f32} : tensor<9216x3072x!qElemType>, tensor<9216x1xf32> -> tensor<9216x3072xf32>

    %fc = IE.MatMul(%in_fp8, %wt_dynquant) {transpose_b} : tensor<1x3072xf32>, tensor<9216x3072xf32> -> tensor<1x9216xf32>

    return %fc : tensor<1x9216xf32>

    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[IN]], [[IN_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072xf32>, tensor<1x3072xf32> -> tensor<1x3072xf32>
    // CHECK:  [[ACT_RESHAPE:%.+]] = IE.Reshape([[MULTIPLY]]) {shape_value = [1, 1, 1, 3072]} : tensor<1x3072xf32> -> tensor<1x1x1x3072xf32>
    // CHECK:  [[WT_RESHAPE:%.+]] = IE.Reshape([[FP8_SCALE]]) {shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:  [[CONV:%.+]] = IE.GroupConvolution([[ACT_RESHAPE]], [[WT_RESHAPE]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x1x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x3072xf32>
    // CHECK:  [[OUT_RESHAPE:%.+]] = IE.Reshape([[CONV]]) {shape_value = [1, 3072]} : tensor<1x1x1x3072xf32> -> tensor<1x3072xf32>
    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[OUT_RESHAPE]]) {dstElemType = !qElemType} : tensor<1x3072xf32> -> tensor<1x3072x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f32} : tensor<1x3072x!qElemType> -> tensor<1x3072xf32>
    // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WT]]) {dstElemType = !qElemType1} : tensor<9216x3072xsi4> -> tensor<9216x3072x!qElemType1>
    // CHECK:  [[DYNDEQUANTIZE:%.+]] = IE.DynamicDequantize([[QUANTCAST]], [[WT_SCALE]]) {dstElemType = f32} : tensor<9216x3072x!qElemType1>, tensor<9216x1xf32> -> tensor<9216x3072xf32>
    // CHECK:  [[MATMUL:%.+]] = IE.MatMul([[DEQUANTIZE]], [[DYNDEQUANTIZE]]) {transpose_b} : tensor<1x3072xf32>, tensor<9216x3072xf32> -> tensor<1x9216xf32>
    // CHECK:  [[DIVIDE:%.+]] = IE.Divide([[FC]], [[FP8_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x9216xf32>, tensor<1xf32> -> tensor<1x9216xf32>
    // CHECK:  return [[DIVIDE]]
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00:8>
// CHECK:  !qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

// CHECK:  func.func @FP8ActivationI4WeightsChannelwiseQuantPostFC
// CHECK-SAME:   ([[IN:%.+]]: tensor<1x1x3072xf32>, [[IN_SCALE:%.+]]: tensor<1x1x3072xf32>, [[FP8_SCALE:%.+]]: tensor<1xf32>,
// CHECK-SAME:    [[WT:%.+]]: tensor<9216x3072xsi4>, [[WT_SCALE:%.+]]: tensor<9216x1xf16>)
func.func @FP8ActivationI4WeightsChannelwiseQuantPostFC(%input: tensor<1x1x3072xf32>, %in_scale: tensor<1x1x3072xf32>, %fp8_scale: tensor<1xf32>,
                                                  %wt: tensor<9216x3072xsi4>, %wt_scale: tensor<9216x1xf16>) -> tensor<1x9216xf32> {
    %cst_0 = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    %in_mul = IE.Multiply(%input, %in_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    %in_fp8 = IE.FakeConvert(%in_mul, %fp8_scale, %cst_0) {dst_type = f8E4M3FN} : tensor<1x1x3072xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x1x3072xf32>
    %in_reshape = IE.Reshape(%in_fp8) {shape_value = [1, 3072]} : tensor<1x1x3072xf32> -> tensor<1x3072xf32>

    %wt_cast = IE.QuantizeCast(%wt) {dstElemType = !qElemType} : tensor<9216x3072xsi4> -> tensor<9216x3072x!qElemType>
    %wt_dynquant = IE.DynamicDequantize(%wt_cast, %wt_scale) {dstElemType = f16} : tensor<9216x3072x!qElemType>, tensor<9216x1xf16> -> tensor<9216x3072xf16>
    %wt_convert = IE.Convert(%wt_dynquant) {dstElemType = f32} : tensor<9216x3072xf16> -> tensor<9216x3072xf32>

    %fc = IE.FullyConnected(%in_reshape, %wt_convert) : tensor<1x3072xf32>, tensor<9216x3072xf32> -> tensor<1x9216xf32>

    return %fc : tensor<1x9216xf32>

    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[IN]], [[IN_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    // CHECK:  [[ACT_RESHAPE:%.+]] = IE.Reshape([[MULTIPLY]]) {shape_value = [1, 1, 1, 3072]} : tensor<1x1x3072xf32> -> tensor<1x1x1x3072xf32>
    // CHECK:  [[WT_RESHAPE:%.+]] = IE.Reshape([[FP8_SCALE]]) {shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:  [[CONV:%.+]] = IE.GroupConvolution([[ACT_RESHAPE]], [[WT_RESHAPE]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x1x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x3072xf32>
    // CHECK:  [[OUT_RESHAPE_1:%.+]] = IE.Reshape([[CONV]]) {shape_value = [1, 1, 3072]} : tensor<1x1x1x3072xf32> -> tensor<1x1x3072xf32>
    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[OUT_RESHAPE_1]]) {dstElemType = !qElemType} : tensor<1x1x3072xf32> -> tensor<1x1x3072x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f32} : tensor<1x1x3072x!qElemType> -> tensor<1x1x3072xf32>
    // CHECK:  [[OUT_RESHAPE_2:%.+]] = IE.Reshape([[DEQUANTIZE]]) {shape_value = [1, 3072]} : tensor<1x1x3072xf32> -> tensor<1x3072xf32>
    // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WT]]) {dstElemType = !qElemType1} : tensor<9216x3072xsi4> -> tensor<9216x3072x!qElemType1>
    // CHECK:  [[DYNDEQUANTIZE:%.+]] = IE.DynamicDequantize([[QUANTCAST]], [[WT_SCALE]]) {dstElemType = f16} : tensor<9216x3072x!qElemType1>, tensor<9216x1xf16> -> tensor<9216x3072xf16>
    // CHECK:  [[WT_CONVERT:%.+]] = IE.Convert([[DYNDEQUANTIZE]]) {dstElemType = f32} : tensor<9216x3072xf16> -> tensor<9216x3072xf32>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[OUT_RESHAPE_2]], [[WT_CONVERT]]) : tensor<1x3072xf32>, tensor<9216x3072xf32> -> tensor<1x9216xf32>
    // CHECK:  [[DIVIDE:%.+]] = IE.Divide([[FC]], [[FP8_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x9216xf32>, tensor<1xf32> -> tensor<1x9216xf32>
    // CHECK:  return [[DIVIDE]]
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00:8>
// CHECK:  !qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

// CHECK:  func.func @FP8ActivationI4WeightsChannelwiseQuantPostMatMul
// CHECK-SAME:   ([[IN:%.+]]: tensor<1x1x3072xf32>, [[IN_SCALE:%.+]]: tensor<1x1x3072xf32>, [[FP8_SCALE:%.+]]: tensor<1xf32>,
// CHECK-SAME:    [[WT:%.+]]: tensor<9216x3072xsi4>, [[WT_SCALE:%.+]]: tensor<9216x1xf16>)
func.func @FP8ActivationI4WeightsChannelwiseQuantPostMatMul(%input: tensor<1x1x3072xf32>, %in_scale: tensor<1x1x3072xf32>, %fp8_scale: tensor<1xf32>,
                                                  %wt: tensor<9216x3072xsi4>, %wt_scale: tensor<9216x1xf16>) -> tensor<1x9216xf32> {
    %cst_0 = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    %in_mul = IE.Multiply(%input, %in_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    %in_fp8 = IE.FakeConvert(%in_mul, %fp8_scale, %cst_0) {dst_type = f8E4M3FN} : tensor<1x1x3072xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x1x3072xf32>
    %in_reshape = IE.Reshape(%in_fp8) {shape_value = [1, 3072]} : tensor<1x1x3072xf32> -> tensor<1x3072xf32>

    %wt_cast = IE.QuantizeCast(%wt) {dstElemType = !qElemType} : tensor<9216x3072xsi4> -> tensor<9216x3072x!qElemType>
    %wt_dynquant = IE.DynamicDequantize(%wt_cast, %wt_scale) {dstElemType = f16} : tensor<9216x3072x!qElemType>, tensor<9216x1xf16> -> tensor<9216x3072xf16>
    %wt_convert = IE.Convert(%wt_dynquant) {dstElemType = f32} : tensor<9216x3072xf16> -> tensor<9216x3072xf32>

    %fc = IE.MatMul(%in_reshape, %wt_convert) {transpose_b} : tensor<1x3072xf32>, tensor<9216x3072xf32> -> tensor<1x9216xf32>

    return %fc : tensor<1x9216xf32>

    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[IN]], [[IN_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    // CHECK:  [[ACT_RESHAPE:%.+]] = IE.Reshape([[MULTIPLY]]) {shape_value = [1, 1, 1, 3072]} : tensor<1x1x3072xf32> -> tensor<1x1x1x3072xf32>
    // CHECK:  [[WT_RESHAPE:%.+]] = IE.Reshape([[FP8_SCALE]]) {shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:  [[CONV:%.+]] = IE.GroupConvolution([[ACT_RESHAPE]], [[WT_RESHAPE]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x1x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x3072xf32>
    // CHECK:  [[OUT_RESHAPE_1:%.+]] = IE.Reshape([[CONV]]) {shape_value = [1, 1, 3072]} : tensor<1x1x1x3072xf32> -> tensor<1x1x3072xf32>
    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[OUT_RESHAPE_1]]) {dstElemType = !qElemType} : tensor<1x1x3072xf32> -> tensor<1x1x3072x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f32} : tensor<1x1x3072x!qElemType> -> tensor<1x1x3072xf32>
    // CHECK:  [[OUT_RESHAPE_2:%.+]] = IE.Reshape([[DEQUANTIZE]]) {shape_value = [1, 3072]} : tensor<1x1x3072xf32> -> tensor<1x3072xf32>
    // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WT]]) {dstElemType = !qElemType1} : tensor<9216x3072xsi4> -> tensor<9216x3072x!qElemType1>
    // CHECK:  [[DYNDEQUANTIZE:%.+]] = IE.DynamicDequantize([[QUANTCAST]], [[WT_SCALE]]) {dstElemType = f16} : tensor<9216x3072x!qElemType1>, tensor<9216x1xf16> -> tensor<9216x3072xf16>
    // CHECK:  [[WT_CONVERT:%.+]] = IE.Convert([[DYNDEQUANTIZE]]) {dstElemType = f32} : tensor<9216x3072xf16> -> tensor<9216x3072xf32>
    // CHECK:  [[MATMUL:%.+]] = IE.MatMul([[OUT_RESHAPE_2]], [[WT_CONVERT]]) {transpose_b} : tensor<1x3072xf32>, tensor<9216x3072xf32> -> tensor<1x9216xf32>
    // CHECK:  [[DIVIDE:%.+]] = IE.Divide([[MATMUL]], [[FP8_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x9216xf32>, tensor<1xf32> -> tensor<1x9216xf32>
    // CHECK:  return [[DIVIDE]]
}

// -----

!qElemType = !quant.uniform<i4:f32, 1.000000e+00:8>
// CHECK:  !qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

// CHECK:  func.func @FP8ActivationI4WeightsGroupQuantPostFC
// CHECK-SAME:   ([[IN:%.+]]: tensor<1x1x3072xf32>, [[IN_SCALE:%.+]]: tensor<1x1x3072xf32>, [[FP8_SCALE:%.+]]: tensor<1xf32>,
// CHECK-SAME:    [[WT:%.+]]: tensor<9216x24x128xsi4>, [[WT_SCALE:%.+]]: tensor<9216x24x1xf32>)
func.func @FP8ActivationI4WeightsGroupQuantPostFC(%input: tensor<1x1x3072xf32>, %in_scale: tensor<1x1x3072xf32>, %fp8_scale: tensor<1xf32>,
                                            %wt: tensor<9216x24x128xsi4>, %wt_scale: tensor<9216x24x1xf32>) -> tensor<1x9216xf32> {
    %in_mul = IE.Multiply(%input, %in_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    %in_fp8 = IE.FakeConvert(%in_mul, %fp8_scale) {dst_type = f8E4M3FN} : tensor<1x1x3072xf32>, tensor<1xf32> -> tensor<1x1x3072xf32>
    %in_reshape = IE.Reshape(%in_fp8) {shape_value = [1, 3072]} : tensor<1x1x3072xf32> -> tensor<1x3072xf32>
    %wt_cast = IE.QuantizeCast(%wt) {dstElemType = !qElemType} : tensor<9216x24x128xsi4> -> tensor<9216x24x128x!qElemType>
    %wt_dynquant = IE.DynamicDequantize(%wt_cast, %wt_scale) {dstElemType = f32} : tensor<9216x24x128x!qElemType>, tensor<9216x24x1xf32> -> tensor<9216x24x128xf32>
    %wt_reshape = IE.AffineReshape(%wt_dynquant) {dim_mapping = [[0], [1], [1]], shape_value = [9216, 3072]} : tensor<9216x24x128xf32> -> tensor<9216x3072xf32>
    %fc = IE.FullyConnected(%in_reshape, %wt_reshape) : tensor<1x3072xf32>, tensor<9216x3072xf32> -> tensor<1x9216xf32>

    return %fc : tensor<1x9216xf32>

    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[IN]], [[IN_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    // CHECK:  [[ACT_RESHAPE:%.+]] = IE.Reshape([[MULTIPLY]]) {shape_value = [1, 1, 1, 3072]} : tensor<1x1x3072xf32> -> tensor<1x1x1x3072xf32>
    // CHECK:  [[WT_RESHAPE:%.+]] = IE.Reshape([[FP8_SCALE]]) {shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:  [[CONV:%.+]] = IE.GroupConvolution([[ACT_RESHAPE]], [[WT_RESHAPE]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x1x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x3072xf32>
    // CHECK:  [[OUT_RESHAPE_1:%.+]] = IE.Reshape([[CONV]]) {shape_value = [1, 1, 3072]} : tensor<1x1x1x3072xf32> -> tensor<1x1x3072xf32>
    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[OUT_RESHAPE_1]]) {dstElemType = !qElemType} : tensor<1x1x3072xf32> -> tensor<1x1x3072x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f32} : tensor<1x1x3072x!qElemType> -> tensor<1x1x3072xf32>
    // CHECK:  [[OUT_RESHAPE_2:%.+]] = IE.Reshape([[DEQUANTIZE]]) {shape_value = [1, 3072]} : tensor<1x1x3072xf32> -> tensor<1x3072xf32>
    // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WT]]) {dstElemType = !qElemType1} : tensor<9216x24x128xsi4> -> tensor<9216x24x128x!qElemType1>
    // CHECK:  [[DYNDEQUANTIZE:%.+]] = IE.DynamicDequantize([[QUANTCAST]], [[WT_SCALE]]) {dstElemType = f32} : tensor<9216x24x128x!qElemType1>, tensor<9216x24x1xf32> -> tensor<9216x24x128xf32>
    // CHECK:  [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[DYNDEQUANTIZE]])
    // CHECK-SAME{LITERAL}:        {dim_mapping = [[0], [1], [1]], shape_value = [9216, 3072]} : tensor<9216x24x128xf32> -> tensor<9216x3072xf32>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[OUT_RESHAPE_2]], [[AFFINERESHAPE]]) : tensor<1x3072xf32>, tensor<9216x3072xf32> -> tensor<1x9216xf32>
    // CHECK:  [[DIVIDE:%.+]] = IE.Divide([[FC]], [[FP8_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x9216xf32>, tensor<1xf32> -> tensor<1x9216xf32>
    // CHECK:  return [[DIVIDE]]
}

// -----

!qElemType = !quant.uniform<i4:f32, 1.000000e+00:8>
// CHECK:  !qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

// CHECK:  func.func @FP8ActivationI4WeightsGroupQuantPostMatMul
// CHECK-SAME:   ([[IN:%.+]]: tensor<1x1x3072xf32>, [[IN_SCALE:%.+]]: tensor<1x1x3072xf32>, [[FP8_SCALE:%.+]]: tensor<1xf32>,
// CHECK-SAME:    [[WT:%.+]]: tensor<9216x24x128xsi4>, [[WT_SCALE:%.+]]: tensor<9216x24x1xf32>)
func.func @FP8ActivationI4WeightsGroupQuantPostMatMul(%input: tensor<1x1x3072xf32>, %in_scale: tensor<1x1x3072xf32>, %fp8_scale: tensor<1xf32>,
                                            %wt: tensor<9216x24x128xsi4>, %wt_scale: tensor<9216x24x1xf32>) -> tensor<1x9216xf32> {
    %in_mul = IE.Multiply(%input, %in_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    %in_fp8 = IE.FakeConvert(%in_mul, %fp8_scale) {dst_type = f8E4M3FN} : tensor<1x1x3072xf32>, tensor<1xf32> -> tensor<1x1x3072xf32>
    %in_reshape = IE.Reshape(%in_fp8) {shape_value = [1, 3072]} : tensor<1x1x3072xf32> -> tensor<1x3072xf32>
    %wt_cast = IE.QuantizeCast(%wt) {dstElemType = !qElemType} : tensor<9216x24x128xsi4> -> tensor<9216x24x128x!qElemType>
    %wt_dynquant = IE.DynamicDequantize(%wt_cast, %wt_scale) {dstElemType = f32} : tensor<9216x24x128x!qElemType>, tensor<9216x24x1xf32> -> tensor<9216x24x128xf32>
    %wt_reshape = IE.AffineReshape(%wt_dynquant) {dim_mapping = [[0], [1], [1]], shape_value = [9216, 3072]} : tensor<9216x24x128xf32> -> tensor<9216x3072xf32>
    %fc = IE.MatMul(%in_reshape, %wt_reshape) {transpose_b} : tensor<1x3072xf32>, tensor<9216x3072xf32> -> tensor<1x9216xf32>

    return %fc : tensor<1x9216xf32>

    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[IN]], [[IN_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    // CHECK:  [[ACT_RESHAPE:%.+]] = IE.Reshape([[MULTIPLY]]) {shape_value = [1, 1, 1, 3072]} : tensor<1x1x3072xf32> -> tensor<1x1x1x3072xf32>
    // CHECK:  [[WT_RESHAPE:%.+]] = IE.Reshape([[FP8_SCALE]]) {shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:  [[CONV:%.+]] = IE.GroupConvolution([[ACT_RESHAPE]], [[WT_RESHAPE]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x1x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x3072xf32>
    // CHECK:  [[OUT_RESHAPE_1:%.+]] = IE.Reshape([[CONV]]) {shape_value = [1, 1, 3072]} : tensor<1x1x1x3072xf32> -> tensor<1x1x3072xf32>
    // CHECK:  [[QUANTIZE:%.+]] = IE.Quantize([[OUT_RESHAPE_1]]) {dstElemType = !qElemType} : tensor<1x1x3072xf32> -> tensor<1x1x3072x!qElemType>
    // CHECK:  [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f32} : tensor<1x1x3072x!qElemType> -> tensor<1x1x3072xf32>
    // CHECK:  [[OUT_RESHAPE_2:%.+]] = IE.Reshape([[DEQUANTIZE]]) {shape_value = [1, 3072]} : tensor<1x1x3072xf32> -> tensor<1x3072xf32>
    // CHECK:  [[QUANTCAST:%.+]] = IE.QuantizeCast([[WT]]) {dstElemType = !qElemType1} : tensor<9216x24x128xsi4> -> tensor<9216x24x128x!qElemType1>
    // CHECK:  [[DYNDEQUANTIZE:%.+]] = IE.DynamicDequantize([[QUANTCAST]], [[WT_SCALE]]) {dstElemType = f32} : tensor<9216x24x128x!qElemType1>, tensor<9216x24x1xf32> -> tensor<9216x24x128xf32>
    // CHECK:  [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[DYNDEQUANTIZE]])
    // CHECK-SAME{LITERAL}:        {dim_mapping = [[0], [1], [1]], shape_value = [9216, 3072]} : tensor<9216x24x128xf32> -> tensor<9216x3072xf32>
    // CHECK:  [[MATMUL:%.+]] = IE.MatMul([[OUT_RESHAPE_2]], [[AFFINERESHAPE]]) {transpose_b} : tensor<1x3072xf32>, tensor<9216x3072xf32> -> tensor<1x9216xf32>
    // CHECK:  [[DIVIDE:%.+]] = IE.Divide([[MATMUL]], [[FP8_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x9216xf32>, tensor<1xf32> -> tensor<1x9216xf32>
    // CHECK:  return [[DIVIDE]]
}

// -----

// CHECK:  !qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>
// CHECK:  func.func @DecomposeFakeConvert
// CHECK-SAME:   ([[IN:%.+]]: tensor<1x1x8192xf32>, [[SCALE:%.+]]: tensor<1xf32>)
func.func @DecomposeFakeConvert(%input: tensor<1x1x8192xf32>, %in_scale: tensor<1xf32>) -> tensor<1x1x8192xf32> {
    %zp = const.Declare tensor<8192xf32> = dense<0.0> : tensor<f32> isSplat, [#const.Reshape<[8192]>]
    %res = IE.FakeConvert(%input, %in_scale, %zp) {dst_type = f8E4M3FN} : tensor<1x1x8192xf32>, tensor<1xf32>, tensor<8192xf32> -> tensor<1x1x8192xf32>

    return %res : tensor<1x1x8192xf32>

    // CHECK: [[RESHAPE_1:%.+]] = IE.Reshape([[IN]]) {shape_value = [1, 1, 1, 8192]} : tensor<1x1x8192xf32> -> tensor<1x1x1x8192xf32>
    // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[SCALE]]) {shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK: [[DWCONV:%.+]] = IE.GroupConvolution([[RESHAPE_1]], [[RESHAPE_2]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x1x8192xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x8192xf32>
    // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[DWCONV]]) {shape_value = [1, 1, 8192]} : tensor<1x1x1x8192xf32> -> tensor<1x1x8192xf32>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[RESHAPE_3]]) {dstElemType = !qElemType} : tensor<1x1x8192xf32> -> tensor<1x1x8192x!qElemType>
    // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f32} : tensor<1x1x8192x!qElemType> -> tensor<1x1x8192xf32>
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[DEQUANTIZE]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x8192xf32>, tensor<1xf32> -> tensor<1x1x8192xf32>

    // CHECK: return [[DIVIDE]]
}

// -----

// CHECK:  !qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>
// CHECK:  func.func @DecomposeDualFakeConvertPostFC
// CHECK-SAME:   ([[IN_1:%.+]]: tensor<1x8192xf32>, [[SCALE_1:%.+]]: tensor<1xf32>, [[IN_2:%.+]]: tensor<3072x8192xf32>, [[SCALE_2:%.+]]: tensor<1xf32>)
func.func @DecomposeDualFakeConvertPostFC(%input_1: tensor<1x8192xf32>, %in_scale_1: tensor<1xf32>, %input_2: tensor<3072x8192xf32>, %in_scale_2: tensor<1xf32>) -> tensor<1x3072xf32> {
    %zp = const.Declare tensor<1xf32> = dense<0.0> : tensor<f32> isSplat, [#const.Reshape<[1]>]
    %fake_convert_1 = IE.FakeConvert(%input_1, %in_scale_1, %zp) {dst_type = f8E4M3FN} : tensor<1x8192xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x8192xf32>
    %fake_convert_2 = IE.FakeConvert(%input_2, %in_scale_2, %zp) {dst_type = f8E4M3FN} : tensor<3072x8192xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<3072x8192xf32>
    %res = IE.FullyConnected(%fake_convert_1, %fake_convert_2) : tensor<1x8192xf32>, tensor<3072x8192xf32> -> tensor<1x3072xf32>

    return %res : tensor<1x3072xf32>

    // CHECK: [[ACT_RESHAPE_1:%.+]] = IE.Reshape([[IN_1]]) {shape_value = [1, 1, 1, 8192]} : tensor<1x8192xf32> -> tensor<1x1x1x8192xf32>
    // CHECK: [[WT_RESHAPE_1:%.+]] = IE.Reshape([[SCALE_1]]) {shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK: [[CONV_1:%.+]] = IE.GroupConvolution([[ACT_RESHAPE_1]], [[WT_RESHAPE_1]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x1x8192xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x8192xf32>
    // CHECK: [[CONV_OUT_RESHAPE_1:%.+]] = IE.Reshape([[CONV_1]]) {shape_value = [1, 8192]} : tensor<1x1x1x8192xf32> -> tensor<1x8192xf32>
    // CHECK: [[QUANTIZE_1:%.+]] = IE.Quantize([[CONV_OUT_RESHAPE_1]]) {dstElemType = !qElemType} : tensor<1x8192xf32> -> tensor<1x8192x!qElemType>
    // CHECK: [[DEQUANTIZE_1:%.+]] = IE.Dequantize([[QUANTIZE_1]]) {dstElemType = f32} : tensor<1x8192x!qElemType> -> tensor<1x8192xf32>

    // CHECK: [[ACT_RESHAPE_2:%.+]] = IE.Reshape([[IN_2]]) {shape_value = [1, 1, 3072, 8192]} : tensor<3072x8192xf32> -> tensor<1x1x3072x8192xf32>
    // CHECK: [[WT_RESHAPE_2:%.+]] = IE.Reshape([[SCALE_2]]) {shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK: [[CONV_2:%.+]] = IE.GroupConvolution([[ACT_RESHAPE_2]], [[WT_RESHAPE_2]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x3072x8192xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x3072x8192xf32>
    // CHECK: [[CONV_OUT_RESHAPE_2:%.+]] = IE.Reshape([[CONV_2]]) {shape_value = [3072, 8192]} : tensor<1x1x3072x8192xf32> -> tensor<3072x8192xf32>
    // CHECK: [[QUANTIZE_2:%.+]] = IE.Quantize([[CONV_OUT_RESHAPE_2]]) {dstElemType = !qElemType} : tensor<3072x8192xf32> -> tensor<3072x8192x!qElemType>
    // CHECK: [[DEQUANTIZE_2:%.+]] = IE.Dequantize([[QUANTIZE_2]]) {dstElemType = f32} : tensor<3072x8192x!qElemType> -> tensor<3072x8192xf32>

    // CHECK: [[FC:%.+]] = IE.FullyConnected([[DEQUANTIZE_1]], [[DEQUANTIZE_2]]) : tensor<1x8192xf32>, tensor<3072x8192xf32> -> tensor<1x3072xf32>
    // CHECK: [[DIVIDE_1:%.+]] = IE.Divide([[FC]], [[SCALE_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072xf32>, tensor<1xf32> -> tensor<1x3072xf32>
    // CHECK: [[DIVIDE_2:%.+]] = IE.Divide([[DIVIDE_1]], [[SCALE_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072xf32>, tensor<1xf32> -> tensor<1x3072xf32>
    // CHECK: return [[DIVIDE_2]]
}

// -----

// CHECK:  !qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>
// CHECK:  func.func @DecomposeDualFakeConvertPostMatMul
// CHECK-SAME:   ([[IN_1:%.+]]: tensor<1x8192xf32>, [[SCALE_1:%.+]]: tensor<1xf32>, [[IN_2:%.+]]: tensor<3072x8192xf32>, [[SCALE_2:%.+]]: tensor<1xf32>)
func.func @DecomposeDualFakeConvertPostMatMul(%input_1: tensor<1x8192xf32>, %in_scale_1: tensor<1xf32>, %input_2: tensor<3072x8192xf32>, %in_scale_2: tensor<1xf32>) -> tensor<1x3072xf32> {
    %zp = const.Declare tensor<1xf32> = dense<0.0> : tensor<f32> isSplat, [#const.Reshape<[1]>]
    %fake_convert_1 = IE.FakeConvert(%input_1, %in_scale_1, %zp) {dst_type = f8E4M3FN} : tensor<1x8192xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x8192xf32>
    %fake_convert_2 = IE.FakeConvert(%input_2, %in_scale_2, %zp) {dst_type = f8E4M3FN} : tensor<3072x8192xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<3072x8192xf32>
    %res = IE.MatMul(%fake_convert_1, %fake_convert_2) {transpose_b} : tensor<1x8192xf32>, tensor<3072x8192xf32> -> tensor<1x3072xf32>

    return %res : tensor<1x3072xf32>

    // CHECK: [[ACT_RESHAPE_1:%.+]] = IE.Reshape([[IN_1]]) {shape_value = [1, 1, 1, 8192]} : tensor<1x8192xf32> -> tensor<1x1x1x8192xf32>
    // CHECK: [[WT_RESHAPE_1:%.+]] = IE.Reshape([[SCALE_1]]) {shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK: [[CONV_1:%.+]] = IE.GroupConvolution([[ACT_RESHAPE_1]], [[WT_RESHAPE_1]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x1x8192xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x8192xf32>
    // CHECK: [[CONV_OUT_RESHAPE_1:%.+]] = IE.Reshape([[CONV_1]]) {shape_value = [1, 8192]} : tensor<1x1x1x8192xf32> -> tensor<1x8192xf32>
    // CHECK: [[QUANTIZE_1:%.+]] = IE.Quantize([[CONV_OUT_RESHAPE_1]]) {dstElemType = !qElemType} : tensor<1x8192xf32> -> tensor<1x8192x!qElemType>
    // CHECK: [[DEQUANTIZE_1:%.+]] = IE.Dequantize([[QUANTIZE_1]]) {dstElemType = f32} : tensor<1x8192x!qElemType> -> tensor<1x8192xf32>

    // CHECK: [[ACT_RESHAPE_2:%.+]] = IE.Reshape([[IN_2]]) {shape_value = [1, 1, 3072, 8192]} : tensor<3072x8192xf32> -> tensor<1x1x3072x8192xf32>
    // CHECK: [[WT_RESHAPE_2:%.+]] = IE.Reshape([[SCALE_2]]) {shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK: [[CONV_2:%.+]] = IE.GroupConvolution([[ACT_RESHAPE_2]], [[WT_RESHAPE_2]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x3072x8192xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x3072x8192xf32>
    // CHECK: [[CONV_OUT_RESHAPE_2:%.+]] = IE.Reshape([[CONV_2]]) {shape_value = [3072, 8192]} : tensor<1x1x3072x8192xf32> -> tensor<3072x8192xf32>
    // CHECK: [[QUANTIZE_2:%.+]] = IE.Quantize([[CONV_OUT_RESHAPE_2]]) {dstElemType = !qElemType} : tensor<3072x8192xf32> -> tensor<3072x8192x!qElemType>
    // CHECK: [[DEQUANTIZE_2:%.+]] = IE.Dequantize([[QUANTIZE_2]]) {dstElemType = f32} : tensor<3072x8192x!qElemType> -> tensor<3072x8192xf32>

    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[DEQUANTIZE_1]], [[DEQUANTIZE_2]]) {transpose_b} : tensor<1x8192xf32>, tensor<3072x8192xf32> -> tensor<1x3072xf32>
    // CHECK: [[DIVIDE_1:%.+]] = IE.Divide([[MATMUL]], [[SCALE_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072xf32>, tensor<1xf32> -> tensor<1x3072xf32>
    // CHECK: [[DIVIDE_2:%.+]] = IE.Divide([[DIVIDE_1]], [[SCALE_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072xf32>, tensor<1xf32> -> tensor<1x3072xf32>
    // CHECK: return [[DIVIDE_2]]
}

// -----

// CHECK:  func.func @DecomposeFakeConvertMultiConsumers
// CHECK-SAME:   ([[IN:%.+]]: tensor<1x1x8192xf32>, [[SCALE:%.+]]: tensor<1xf32>)
func.func @DecomposeFakeConvertMultiConsumers(%input: tensor<1x1x8192xf32>, %in_scale: tensor<1xf32>) -> (tensor<1x1x8192xf16>, tensor<1x1x8192xf16>) {
    %zp = const.Declare tensor<1xf32> = dense<0.0> : tensor<f32> isSplat, [#const.Reshape<[1]>]
    %fake_convert = IE.FakeConvert(%input, %in_scale, %zp) {dst_type = f8E4M3FN} : tensor<1x1x8192xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x1x8192xf32>
    %consumer_1 = IE.Convert(%fake_convert) {dstElemType = f16} : tensor<1x1x8192xf32> -> tensor<1x1x8192xf16>
    %consumer_2 = IE.Convert(%fake_convert) {dstElemType = f16} : tensor<1x1x8192xf32> -> tensor<1x1x8192xf16>
    return %consumer_1, %consumer_2 : tensor<1x1x8192xf16>, tensor<1x1x8192xf16>

    // CHECK: [[ACT_RESHAPE:%.+]] = IE.Reshape([[IN]]) {shape_value = [1, 1, 1, 8192]} : tensor<1x1x8192xf32> -> tensor<1x1x1x8192xf32>
    // CHECK: [[WT_RESHAPE:%.+]] = IE.Reshape([[SCALE]]) {shape_value = [1, 1, 1, 1]} : tensor<1xf32> -> tensor<1x1x1x1xf32>
    // CHECK: [[CONV:%.+]] = IE.GroupConvolution([[ACT_RESHAPE]], [[WT_RESHAPE]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x1x8192xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x8192xf32>
    // CHECK: [[CONV_OUT_RESHAPE:%.+]] = IE.Reshape([[CONV]]) {shape_value = [1, 1, 8192]} : tensor<1x1x1x8192xf32> -> tensor<1x1x8192xf32>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[CONV_OUT_RESHAPE]]) {dstElemType = !qElemType} : tensor<1x1x8192xf32> -> tensor<1x1x8192x!qElemType>
    // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f32} : tensor<1x1x8192x!qElemType> -> tensor<1x1x8192xf32>
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[DEQUANTIZE]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x8192xf32>, tensor<1xf32> -> tensor<1x1x8192xf32>
    // CHECK: [[CONVERT_1:%.+]] = IE.Convert([[DIVIDE]]) {dstElemType = f16} : tensor<1x1x8192xf32> -> tensor<1x1x8192xf16>
    // CHECK: [[CONVERT_2:%.+]] = IE.Convert([[DIVIDE]]) {dstElemType = f16} : tensor<1x1x8192xf32> -> tensor<1x1x8192xf16>
    // CHECK: return [[CONVERT_1]], [[CONVERT_2]]
}

// -----

// CHECK:  func.func @DoNotDecomposeFakeConvertWithConstScale
// CHECK-SAME:   ([[IN:%.+]]: tensor<1x1x8192xf32>)
func.func @DoNotDecomposeFakeConvertWithConstScale(%input: tensor<1x1x8192xf32>) -> tensor<1x1x8192xf32> {
    %scale = const.Declare tensor<1xf32> = dense<1.0> : tensor<f32> isSplat, [#const.Reshape<[1]>]
    %zp = const.Declare tensor<1xf32> = dense<0.0> : tensor<f32> isSplat, [#const.Reshape<[1]>]
    %res = IE.FakeConvert(%input, %scale, %zp) {dst_type = f8E4M3FN} : tensor<1x1x8192xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x1x8192xf32>

    return %res : tensor<1x1x8192xf32>

    // CHECK: IE.FakeConvert
}

// -----

// CHECK:  func.func @DoNotDecomposeFakeConvertMultiScales
// CHECK-SAME:   ([[IN:%.+]]: tensor<1x1x8192xf32>, [[SCALE:%.+]]: tensor<8192xf32>)
func.func @DoNotDecomposeFakeConvertMultiScales(%input: tensor<1x1x8192xf32>, %in_scale: tensor<8192xf32>) -> tensor<1x1x8192xf32> {
    %zp = const.Declare tensor<1xf32> = dense<0.0> : tensor<f32> isSplat, [#const.Reshape<[1]>]
    %res = IE.FakeConvert(%input, %in_scale, %zp) {dst_type = f8E4M3FN} : tensor<1x1x8192xf32>, tensor<8192xf32>, tensor<1xf32> -> tensor<1x1x8192xf32>

    return %res : tensor<1x1x8192xf32>

    // CHECK: IE.FakeConvert
}

// -----

// CHECK:  func.func @DoNotDecomposeFakeConvertNonSplatConstShift
// CHECK-SAME:   ([[IN:%.+]]: tensor<1x1x4xf32>, [[SCALE:%.+]]: tensor<1xf32>)
func.func @DoNotDecomposeFakeConvertNonSplatConstShift(%input: tensor<1x1x4xf32>, %in_scale: tensor<1xf32>) -> tensor<1x1x4xf32> {
    %zp = const.Declare tensor<4xf32> = dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>, [#const.Reshape<[4]>]
    %res = IE.FakeConvert(%input, %in_scale, %zp) {dst_type = f8E4M3FN} : tensor<1x1x4xf32>, tensor<1xf32>, tensor<4xf32> -> tensor<1x1x4xf32>

    return %res : tensor<1x1x4xf32>

    // CHECK: IE.FakeConvert
}

// -----

// CHECK:  func.func @DoNotDecomposeFakeConvertNonZeroConstShift
// CHECK-SAME:   ([[IN:%.+]]: tensor<1x1x8192xf32>, [[SCALE:%.+]]: tensor<1xf32>)
func.func @DoNotDecomposeFakeConvertNonZeroConstShift(%input: tensor<1x1x8192xf32>, %in_scale: tensor<1xf32>) -> tensor<1x1x8192xf32> {
    %zp = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32> isSplat, [#const.Reshape<[1]>]
    %res = IE.FakeConvert(%input, %in_scale, %zp) {dst_type = f8E4M3FN} : tensor<1x1x8192xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x1x8192xf32>

    return %res : tensor<1x1x8192xf32>

    // CHECK: IE.FakeConvert
}

// -----

// CHECK:  func.func @DoNotDecomposeFakeConvertNonConstShift
// CHECK-SAME:   ([[IN:%.+]]: tensor<1x1x8192xf32>, [[SCALE:%.+]]: tensor<1xf32>, [[ZP:%.+]]: tensor<1xf32>)
func.func @DoNotDecomposeFakeConvertNonConstShift(%input: tensor<1x1x8192xf32>, %in_scale: tensor<1xf32>, %zp: tensor<1xf32>) -> tensor<1x1x8192xf32> {
    %res = IE.FakeConvert(%input, %in_scale, %zp) {dst_type = f8E4M3FN} : tensor<1x1x8192xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x1x8192xf32>

    return %res : tensor<1x1x8192xf32>

    // CHECK: IE.FakeConvert
}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

#CNH = affine_map<(d0, d1, d2) -> (d1, d0, d2)>

// CHECK:  func.func @MoveDividePostFC
// CHECK-SAME:   ([[IN_1:%.+]]: tensor<1x1x8192x!qElemType>, [[SCALE_1:%.+]]: tensor<1xf32>, [[IN_2:%.+]]: tensor<1x3072x8192x!qElemType>, [[SCALE_2:%.+]]: tensor<1xf32>)
func.func @MoveDividePostFC(%input_1: tensor<1x1x8192x!qElemType>, %in_scale_1: tensor<1xf32>, %input_2: tensor<1x3072x8192x!qElemType>, %in_scale_2: tensor<1xf32>) -> tensor<1x3072xf16> {
    %dequantize_1 = IE.Dequantize(%input_1) {dstElemType = f32} : tensor<1x1x8192x!qElemType> -> tensor<1x1x8192xf32>
    %divide_1 = IE.Divide(%dequantize_1, %in_scale_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x8192xf32>, tensor<1xf32> -> tensor<1x1x8192xf32>
    %transpose_1 = IE.Transpose(%divide_1) {order_value = #CNH} : tensor<1x1x8192xf32> -> tensor<1x1x8192xf32>
    %reshape_1 = IE.Reshape(%transpose_1) { shape_value = [1, 8192] } : tensor<1x1x8192xf32> -> tensor<1x8192xf32>
    %convert_1 = IE.Convert(%reshape_1) {dstElemType = f16} : tensor<1x8192xf32> -> tensor<1x8192xf16>
    %dequantize_2 = IE.Dequantize(%input_2) {dstElemType = f32} : tensor<1x3072x8192x!qElemType> -> tensor<1x3072x8192xf32>
    %divide_2 = IE.Divide(%dequantize_2, %in_scale_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072x8192xf32>, tensor<1xf32> -> tensor<1x3072x8192xf32>
    %transpose_2 = IE.Transpose(%divide_2) {order_value = #CNH} : tensor<1x3072x8192xf32> -> tensor<3072x1x8192xf32>
    %reshape_2 = IE.Reshape(%transpose_2) { shape_value = [3072, 8192] } : tensor<3072x1x8192xf32> -> tensor<3072x8192xf32>
    %convert_2 = IE.Convert(%reshape_2) {dstElemType = f16} : tensor<3072x8192xf32> -> tensor<3072x8192xf16>
    %res = IE.FullyConnected(%convert_1, %convert_2) : tensor<1x8192xf16>, tensor<3072x8192xf16> -> tensor<1x3072xf16>

    return %res : tensor<1x3072xf16>

    // CHECK: IE.FullyConnected
    // CHECK: IE.Divide
    // CHECK: IE.Divide
}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

#CNH = affine_map<(d0, d1, d2) -> (d1, d0, d2)>

// CHECK:  func.func @MoveDividePostMatMul
// CHECK-SAME:   ([[IN_1:%.+]]: tensor<1x1x8192x!qElemType>, [[SCALE_1:%.+]]: tensor<1xf32>, [[IN_2:%.+]]: tensor<1x3072x8192x!qElemType>, [[SCALE_2:%.+]]: tensor<1xf32>)
func.func @MoveDividePostMatMul(%input_1: tensor<1x1x8192x!qElemType>, %in_scale_1: tensor<1xf32>, %input_2: tensor<1x3072x8192x!qElemType>, %in_scale_2: tensor<1xf32>) -> tensor<1x3072xf16> {
    %dequantize_1 = IE.Dequantize(%input_1) {dstElemType = f32} : tensor<1x1x8192x!qElemType> -> tensor<1x1x8192xf32>
    %divide_1 = IE.Divide(%dequantize_1, %in_scale_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x8192xf32>, tensor<1xf32> -> tensor<1x1x8192xf32>
    %transpose_1 = IE.Transpose(%divide_1) {order_value = #CNH} : tensor<1x1x8192xf32> -> tensor<1x1x8192xf32>
    %reshape_1 = IE.Reshape(%transpose_1) { shape_value = [1, 8192] } : tensor<1x1x8192xf32> -> tensor<1x8192xf32>
    %convert_1 = IE.Convert(%reshape_1) {dstElemType = f16} : tensor<1x8192xf32> -> tensor<1x8192xf16>
    %dequantize_2 = IE.Dequantize(%input_2) {dstElemType = f32} : tensor<1x3072x8192x!qElemType> -> tensor<1x3072x8192xf32>
    %divide_2 = IE.Divide(%dequantize_2, %in_scale_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072x8192xf32>, tensor<1xf32> -> tensor<1x3072x8192xf32>
    %transpose_2 = IE.Transpose(%divide_2) {order_value = #CNH} : tensor<1x3072x8192xf32> -> tensor<3072x1x8192xf32>
    %reshape_2 = IE.Reshape(%transpose_2) { shape_value = [3072, 8192] } : tensor<3072x1x8192xf32> -> tensor<3072x8192xf32>
    %convert_2 = IE.Convert(%reshape_2) {dstElemType = f16} : tensor<3072x8192xf32> -> tensor<3072x8192xf16>
    %res = IE.MatMul(%convert_1, %convert_2) {transpose_b} : tensor<1x8192xf16>, tensor<3072x8192xf16> -> tensor<1x3072xf16>

    return %res : tensor<1x3072xf16>

    // CHECK: IE.MatMul
    // CHECK: IE.Divide
    // CHECK: IE.Divide
}

// -----

// CHECK:  func.func @DoNotMoveDivideNoDequantizePostFC
// CHECK-SAME:   ([[IN_1:%.+]]: tensor<1x8192xf32>, [[SCALE_1:%.+]]: tensor<1xf32>, [[IN_2:%.+]]: tensor<3072x8192xf32>, [[SCALE_2:%.+]]: tensor<1xf32>)
func.func @DoNotMoveDivideNoDequantizePostFC(%input_1: tensor<1x8192xf32>, %in_scale_1: tensor<1xf32>, %input_2: tensor<3072x8192xf32>, %in_scale_2: tensor<1xf32>) -> tensor<1x3072xf32> {
    %divide_1 = IE.Divide(%input_1, %in_scale_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8192xf32>, tensor<1xf32> -> tensor<1x8192xf32>
    %divide_2 = IE.Divide(%input_2, %in_scale_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3072x8192xf32>, tensor<1xf32> -> tensor<3072x8192xf32>
    %res = IE.FullyConnected(%divide_1, %divide_2) : tensor<1x8192xf32>, tensor<3072x8192xf32> -> tensor<1x3072xf32>

    return %res : tensor<1x3072xf32>

    // CHECK: IE.Divide
    // CHECK: IE.Divide
    // CHECK: IE.FullyConnected
}

// -----

// CHECK:  func.func @DoNotMoveDivideNoDequantizePostMatMul
// CHECK-SAME:   ([[IN_1:%.+]]: tensor<1x8192xf32>, [[SCALE_1:%.+]]: tensor<1xf32>, [[IN_2:%.+]]: tensor<3072x8192xf32>, [[SCALE_2:%.+]]: tensor<1xf32>)
func.func @DoNotMoveDivideNoDequantizePostMatMul(%input_1: tensor<1x8192xf32>, %in_scale_1: tensor<1xf32>, %input_2: tensor<3072x8192xf32>, %in_scale_2: tensor<1xf32>) -> tensor<1x3072xf32> {
    %divide_1 = IE.Divide(%input_1, %in_scale_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8192xf32>, tensor<1xf32> -> tensor<1x8192xf32>
    %divide_2 = IE.Divide(%input_2, %in_scale_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3072x8192xf32>, tensor<1xf32> -> tensor<3072x8192xf32>
    %res = IE.MatMul(%divide_1, %divide_2) {transpose_b} : tensor<1x8192xf32>, tensor<3072x8192xf32> -> tensor<1x3072xf32>

    return %res : tensor<1x3072xf32>

    // CHECK: IE.Divide
    // CHECK: IE.Divide
    // CHECK: IE.MatMul
}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

// CHECK:  func.func @DoNotMoveDivideMultiScalesPostFC
// CHECK-SAME:   ([[IN_1:%.+]]: tensor<1x8192x!qElemType>, [[SCALE_1:%.+]]: tensor<8192xf32>, [[IN_2:%.+]]: tensor<3072x8192x!qElemType>, [[SCALE_2:%.+]]: tensor<8192xf32>)
func.func @DoNotMoveDivideMultiScalesPostFC(%input_1: tensor<1x8192x!qElemType>, %in_scale_1: tensor<8192xf32>, %input_2: tensor<3072x8192x!qElemType>, %in_scale_2: tensor<8192xf32>) -> tensor<1x3072xf32> {
    %dequantize_1 = IE.Dequantize(%input_1) {dstElemType = f32} : tensor<1x8192x!qElemType> -> tensor<1x8192xf32>
    %divide_1 = IE.Divide(%dequantize_1, %in_scale_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8192xf32>, tensor<8192xf32> -> tensor<1x8192xf32>
    %dequantize_2 = IE.Dequantize(%input_2) {dstElemType = f32} : tensor<3072x8192x!qElemType> -> tensor<3072x8192xf32>
    %divide_2 = IE.Divide(%dequantize_2, %in_scale_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3072x8192xf32>, tensor<8192xf32> -> tensor<3072x8192xf32>
    %res = IE.FullyConnected(%divide_1, %divide_2) : tensor<1x8192xf32>, tensor<3072x8192xf32> -> tensor<1x3072xf32>

    return %res : tensor<1x3072xf32>

    // CHECK: IE.Divide
    // CHECK: IE.Divide
    // CHECK: IE.FullyConnected
}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

// CHECK:  func.func @DoNotMoveDivideMultiScalesPostMatMul
// CHECK-SAME:   ([[IN_1:%.+]]: tensor<1x8192x!qElemType>, [[SCALE_1:%.+]]: tensor<8192xf32>, [[IN_2:%.+]]: tensor<3072x8192x!qElemType>, [[SCALE_2:%.+]]: tensor<8192xf32>)
func.func @DoNotMoveDivideMultiScalesPostMatMul(%input_1: tensor<1x8192x!qElemType>, %in_scale_1: tensor<8192xf32>, %input_2: tensor<3072x8192x!qElemType>, %in_scale_2: tensor<8192xf32>) -> tensor<1x3072xf32> {
    %dequantize_1 = IE.Dequantize(%input_1) {dstElemType = f32} : tensor<1x8192x!qElemType> -> tensor<1x8192xf32>
    %divide_1 = IE.Divide(%dequantize_1, %in_scale_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8192xf32>, tensor<8192xf32> -> tensor<1x8192xf32>
    %dequantize_2 = IE.Dequantize(%input_2) {dstElemType = f32} : tensor<3072x8192x!qElemType> -> tensor<3072x8192xf32>
    %divide_2 = IE.Divide(%dequantize_2, %in_scale_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3072x8192xf32>, tensor<8192xf32> -> tensor<3072x8192xf32>
    %res = IE.MatMul(%divide_1, %divide_2) {transpose_b} : tensor<1x8192xf32>, tensor<3072x8192xf32> -> tensor<1x3072xf32>

    return %res : tensor<1x3072xf32>

    // CHECK: IE.Divide
    // CHECK: IE.Divide
    // CHECK: IE.MatMul
}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

// CHECK:  func.func @DoNotMoveDivideFCWithBias
// CHECK-SAME:   ([[IN_1:%.+]]: tensor<1x8192x!qElemType>, [[SCALE_1:%.+]]: tensor<1xf32>, [[IN_2:%.+]]: tensor<3072x8192x!qElemType>, [[SCALE_2:%.+]]: tensor<1xf32>)
func.func @DoNotMoveDivideFCWithBias(%input_1: tensor<1x8192x!qElemType>, %in_scale_1: tensor<1xf32>, %input_2: tensor<3072x8192x!qElemType>, %in_scale_2: tensor<1xf32>) -> tensor<1x3072xf32> {
    %bias = const.Declare tensor<1xf32> = dense<0.1> : tensor<1xf32>
    %dequantize_1 = IE.Dequantize(%input_1) {dstElemType = f32} : tensor<1x8192x!qElemType> -> tensor<1x8192xf32>
    %divide_1 = IE.Divide(%dequantize_1, %in_scale_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8192xf32>, tensor<1xf32> -> tensor<1x8192xf32>
    %dequantize_2 = IE.Dequantize(%input_2) {dstElemType = f32} : tensor<3072x8192x!qElemType> -> tensor<3072x8192xf32>
    %divide_2 = IE.Divide(%dequantize_2, %in_scale_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3072x8192xf32>, tensor<1xf32> -> tensor<3072x8192xf32>
    %res = IE.FullyConnected(%divide_1, %divide_2, %bias) : tensor<1x8192xf32>, tensor<3072x8192xf32>, tensor<1xf32> -> tensor<1x3072xf32>

    return %res : tensor<1x3072xf32>

    // CHECK: IE.Divide
    // CHECK: IE.Divide
    // CHECK: IE.FullyConnected
}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

// CHECK:  func.func @DoNotMoveDivideMatMulWithPostOp
// CHECK-SAME:   ([[IN_1:%.+]]: tensor<1x8192x!qElemType>, [[SCALE_1:%.+]]: tensor<1xf32>, [[IN_2:%.+]]: tensor<3072x8192x!qElemType>, [[SCALE_2:%.+]]: tensor<1xf32>)
func.func @DoNotMoveDivideMatMulWithPostOp(%input_1: tensor<1x8192x!qElemType>, %in_scale_1: tensor<1xf32>, %input_2: tensor<3072x8192x!qElemType>, %in_scale_2: tensor<1xf32>) -> tensor<1x3072xf32> {
    %dequantize_1 = IE.Dequantize(%input_1) {dstElemType = f32} : tensor<1x8192x!qElemType> -> tensor<1x8192xf32>
    %divide_1 = IE.Divide(%dequantize_1, %in_scale_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8192xf32>, tensor<1xf32> -> tensor<1x8192xf32>
    %dequantize_2 = IE.Dequantize(%input_2) {dstElemType = f32} : tensor<3072x8192x!qElemType> -> tensor<3072x8192xf32>
    %divide_2 = IE.Divide(%dequantize_2, %in_scale_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3072x8192xf32>, tensor<1xf32> -> tensor<3072x8192xf32>
    %res = IE.MatMul(%divide_1, %divide_2) {post_op = #IE.Relu<>, transpose_b} : tensor<1x8192xf32>, tensor<3072x8192xf32> -> tensor<1x3072xf32>

    return %res : tensor<1x3072xf32>

    // CHECK: IE.Divide
    // CHECK: IE.Divide
    // CHECK: IE.MatMul
}

// -----

!qElemType = !quant.uniform<f8E4M3FN:f32, 1.000000e+00>

#CNH = affine_map<(d0, d1, d2) -> (d1, d0, d2)>

// CHECK:  func.func @MoveDividePostFCWithMultiply
// CHECK-SAME:   ([[IN_1:%.+]]: tensor<1x1x8192x!qElemType>, [[SCALE_1:%.+]]: tensor<1xf32>, [[IN_2:%.+]]: tensor<1x3072x8192x!qElemType>, [[SCALE_2:%.+]]: tensor<1xf32>)
func.func @MoveDividePostFCWithMultiply(%input_1: tensor<1x1x8192x!qElemType>, %in_scale_1: tensor<1xf32>, %input_2: tensor<1x3072x8192x!qElemType>, %in_scale_2: tensor<1xf32>) -> tensor<1x3072xf16> {
    %cst = const.Declare tensor<1x1x1xf32> = dense<2.0> : tensor<1x1x1xf32>
    %dequantize_1 = IE.Dequantize(%input_1) {dstElemType = f32} : tensor<1x1x8192x!qElemType> -> tensor<1x1x8192xf32>
    %divide_1 = IE.Divide(%dequantize_1, %in_scale_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x8192xf32>, tensor<1xf32> -> tensor<1x1x8192xf32>
    %multiply = IE.Multiply(%divide_1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x1x8192xf32>, tensor<1x1x1xf32> -> tensor<1x1x8192xf32>
    %transpose_1 = IE.Transpose(%multiply) {order_value = #CNH} : tensor<1x1x8192xf32> -> tensor<1x1x8192xf32>
    %reshape_1 = IE.Reshape(%transpose_1) { shape_value = [1, 8192] } : tensor<1x1x8192xf32> -> tensor<1x8192xf32>
    %convert_1 = IE.Convert(%reshape_1) {dstElemType = f16} : tensor<1x8192xf32> -> tensor<1x8192xf16>
    %dequantize_2 = IE.Dequantize(%input_2) {dstElemType = f32} : tensor<1x3072x8192x!qElemType> -> tensor<1x3072x8192xf32>
    %divide_2 = IE.Divide(%dequantize_2, %in_scale_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072x8192xf32>, tensor<1xf32> -> tensor<1x3072x8192xf32>
    %transpose_2 = IE.Transpose(%divide_2) {order_value = #CNH} : tensor<1x3072x8192xf32> -> tensor<3072x1x8192xf32>
    %reshape_2 = IE.Reshape(%transpose_2) { shape_value = [3072, 8192] } : tensor<3072x1x8192xf32> -> tensor<3072x8192xf32>
    %convert_2 = IE.Convert(%reshape_2) {dstElemType = f16} : tensor<3072x8192xf32> -> tensor<3072x8192xf16>
    %res = IE.FullyConnected(%convert_1, %convert_2) : tensor<1x8192xf16>, tensor<3072x8192xf16> -> tensor<1x3072xf16>

    return %res : tensor<1x3072xf16>

    // CHECK: IE.FullyConnected
    // CHECK: IE.Divide
    // CHECK: IE.Divide
}
