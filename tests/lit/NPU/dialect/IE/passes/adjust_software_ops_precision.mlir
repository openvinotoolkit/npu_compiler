//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-software-ops-precision --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @TopK_KeepSI32Precision
func.func @TopK_KeepSI32Precision(%arg0: tensor<1x77xsi32>) -> (tensor<1x1xsi32>, tensor<1x1xsi32>) {
    %cst_K = const.Declare tensor<si64> = dense<1> : tensor<si64>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x77xsi32>, tensor<si64> -> tensor<1x1xsi32>, tensor<1x1xsi32>
    return %output_values, %target_shape : tensor<1x1xsi32>, tensor<1x1xsi32>

    // CHECK:       [[VALUES:%.+]], [[SHAPE:%.+]] = IE.TopK
    // CHECK-SAME:      {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x77xsi32> -> tensor<1x1xsi32>, tensor<1x1xsi32>
    // CHECK:       return [[VALUES]], [[SHAPE]] : tensor<1x1xsi32>, tensor<1x1xsi32>
}

// -----

// CHECK-LABEL: @TopK_SI64toFP16
func.func @TopK_SI64toFP16(%arg0: tensor<1x77xsi64>) -> (tensor<1x1xsi64>, tensor<1x1xsi32>) {
    %cst_K = const.Declare tensor<si64> = dense<1> : tensor<si64>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x77xsi64>, tensor<si64> -> tensor<1x1xsi64>, tensor<1x1xsi32>
    return %output_values, %target_shape : tensor<1x1xsi64>, tensor<1x1xsi32>

    // CHECK: [[INPUT_CVT:%.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x77xsi64> -> tensor<1x77xf16>
    // CHECK: [[VALUES:%.*]], [[SHAPE:%.*]] = IE.TopK([[INPUT_CVT]]
    // CHECK: [[OUT_CVT:%.*]] = IE.Convert([[VALUES]]) {dstElemType = si64} : tensor<1x1xf16> -> tensor<1x1xsi64>
    // CHECK: return [[OUT_CVT]], [[SHAPE]] : tensor<1x1xsi64>, tensor<1x1xsi32>
}


// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @DequantizeScaletoFP16
// CHECK-SAME:   [[INPUT0:%.+]]: tensor<28x512x128xi4>
// CHECK-SAME:   [[INPUT1:%.+]]: tensor<28x1x128xf32>
func.func @DequantizeScaletoFP16(%arg0: tensor<28x512x128xi4>, %arg1: tensor<28x1x128xf32>) -> tensor<28x512x128xf32> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<28x512x128xi4> -> tensor<28x512x128x!qElemType>
    %1 = IE.DynamicDequantize(%0, %arg1) {dstElemType = f32} : tensor<28x512x128x!qElemType>, tensor<28x1x128xf32> -> tensor<28x512x128xf32>
    return %1 : tensor<28x512x128xf32>

    // CHECK: [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[INPUT0]]) {dstElemType = !qElemType} : tensor<28x512x128xi4> -> tensor<28x512x128x!qElemType>
    // CHECK: [[CONVERT_IN:%.+]] = IE.Convert([[INPUT1]]) {dstElemType = f16} : tensor<28x1x128xf32> -> tensor<28x1x128xf16>
    // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANTIZE_CAST]], [[CONVERT_IN]]) {dstElemType = f16} : tensor<28x512x128x!qElemType>, tensor<28x1x128xf16> -> tensor<28x512x128xf16>
    // CHECK: [[CONVERT_OUT:%.+]] = IE.Convert([[DYN_DEQUANT]]) {dstElemType = f32} : tensor<28x512x128xf16> -> tensor<28x512x128xf32>
    // CHECK:       return [[CONVERT_OUT]] : tensor<28x512x128xf32>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @DequantizeStaticScaleToFP16
// CHECK-SAME:   [[INPUT0:%.+]]: tensor<28x512x128xi4>
func.func @DequantizeStaticScaleToFP16(%arg0: tensor<28x512x128xi4>) -> tensor<28x512x128xf32> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<28x512x128xi4> -> tensor<28x512x128x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<28x512x128x!qElemType> -> tensor<28x512x128xf32>
    return %1 : tensor<28x512x128xf32>
    
    // CHECK:       [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[INPUT0]]) {dstElemType = !qElemType} : tensor<28x512x128xi4> -> tensor<28x512x128x!qElemType>
    // CHECK-NEXT:  [[DEQUANT:%.+]] = IE.Dequantize([[QUANTIZE_CAST]]) {dstElemType = f16} : tensor<28x512x128x!qElemType> -> tensor<28x512x128xf16>
    // CHECK-NEXT:  [[CONVERT_OUT:%.+]] = IE.Convert([[DEQUANT]]) {dstElemType = f32} : tensor<28x512x128xf16> -> tensor<28x512x128xf32>
    // CHECK-NEXT:  return [[CONVERT_OUT]] : tensor<28x512x128xf32>
}
