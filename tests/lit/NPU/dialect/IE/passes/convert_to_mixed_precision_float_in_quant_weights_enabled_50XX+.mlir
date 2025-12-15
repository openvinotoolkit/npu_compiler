//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-mixed-precision="enable-float-in-quant-weights-mixed-mode=true" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.0021300796153289931>
// CHECK: !qElemType = !quant.uniform<f8E4M3FN:f16, 0.0021300796153289931>
// CHECK: !qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.002645585685968399>
// CHECK-LABEL: @MixedPrecisionFloatInputQuantWeightsMatMulF8E4M3FN
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x32x1x1152xf16>)
func.func @MixedPrecisionFloatInputQuantWeightsMatMulF8E4M3FN(%arg0: tensor<1x32x1x1152xf16>) -> tensor<1x32x1x96x!qElemType> {
  %1 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x32x1x1152xf16> -> tensor<1x32x1x1152xf16>
  %2 = const.Declare tensor<1x32x96x1152x!quant.uniform<f8E4M3FN:f16, 0.002645585685968399>> = 
      dense<0.0> : tensor<1x32x96x1152xf16>, 
      [#const.CastElemType<!quant.uniform<f8E4M3FN:f16, 0.002645585685968399>>]
  %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x32x96x1152x!quant.uniform<f8E4M3FN:f16, 0.002645585685968399>> -> tensor<1x32x96x1152xf16>
  %4 = IE.MatMul(%1, %3) {transpose_b} : tensor<1x32x1x1152xf16>, tensor<1x32x96x1152xf16> -> tensor<1x32x1x96x!qElemType>

  return %4 : tensor<1x32x1x96x!qElemType>

  //CHECK: [[CST:%.*]] = const.Declare tensor<1x32x96x1152x!qElemType1> = dense<0.000000e+00>
  //CHECK: [[VAL0:%.*]] = IE.SoftMax([[ARG0]]) {axisInd = 3 : i64} : tensor<1x32x1x1152xf16> -> tensor<1x32x1x1152xf16>
  //CHECK: [[VAL1:%.*]] = IE.MatMul([[VAL0]], [[CST]]) {transpose_b} : tensor<1x32x1x1152xf16>, tensor<1x32x96x1152x!qElemType1> -> tensor<1x32x1x96x!qElemType>
  //CHECK: return [[VAL1]]
}

// -----

!qElemTypeE5M2 = !quant.uniform<f8E5M2:f16, 0.0021300796153289931>
// CHECK: !qElemType = !quant.uniform<f8E5M2:f16, 0.0021300796153289931>
// CHECK: !qElemType1 = !quant.uniform<f8E5M2:f16, 0.002645585685968399>
// CHECK-LABEL: @MixedPrecisionFloatInputQuantWeightsMatMulF8E5M2
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x32x1x1152xf16>)
func.func @MixedPrecisionFloatInputQuantWeightsMatMulF8E5M2(%arg0: tensor<1x32x1x1152xf16>) -> tensor<1x32x1x96x!qElemTypeE5M2> {
  %1 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x32x1x1152xf16> -> tensor<1x32x1x1152xf16>
  %2 = const.Declare tensor<1x32x96x1152x!quant.uniform<f8E5M2:f16, 0.002645585685968399>> =
      dense<0.0> : tensor<1x32x96x1152xf16>,
      [#const.CastElemType<!quant.uniform<f8E5M2:f16, 0.002645585685968399>>]
  %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x32x96x1152x!quant.uniform<f8E5M2:f16, 0.002645585685968399>> -> tensor<1x32x96x1152xf16>
  %4 = IE.MatMul(%1, %3) {transpose_b} : tensor<1x32x1x1152xf16>, tensor<1x32x96x1152xf16> -> tensor<1x32x1x96x!qElemTypeE5M2>

  return %4 : tensor<1x32x1x96x!qElemTypeE5M2>

  //CHECK: [[CST:%.*]] = const.Declare tensor<1x32x96x1152x!qElemType1> = dense<0.000000e+00>
  //CHECK: [[VAL0:%.*]] = IE.SoftMax([[ARG0]]) {axisInd = 3 : i64} : tensor<1x32x1x1152xf16> -> tensor<1x32x1x1152xf16>
  //CHECK: [[VAL1:%.*]] = IE.MatMul([[VAL0]], [[CST]]) {transpose_b} : tensor<1x32x1x1152xf16>, tensor<1x32x96x1152x!qElemType1> -> tensor<1x32x1x96x!qElemType>
  //CHECK: return [[VAL1]]
}
