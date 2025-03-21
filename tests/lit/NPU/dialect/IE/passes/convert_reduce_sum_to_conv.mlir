//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-reduce-sum-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @ConvertReduceSumToConv4D
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x32x31xf16>
func.func @ConvertReduceSumToConv4D(%arg0: tensor<1x4x32x31xf16>) -> tensor<1x1x32x31xf16> {
  %1 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x4x32x31xf16> -> tensor<1x1x32x31xf16>
  return %1 : tensor<1x1x32x31xf16>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x4x1x1xf16> = dense<1.000000e+00> : tensor<1x4x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x32x31xf16>, tensor<1x4x1x1xf16> -> tensor<1x1x32x31xf16>

  // CHECK:       return [[CONV]] : tensor<1x1x32x31xf16>
}

// -----

// CHECK-LABEL: @NotConvertReduceSumToConv4DIfNotReduceChannel
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x32x32xf16>
func.func @NotConvertReduceSumToConv4DIfNotReduceChannel(%arg0: tensor<1x4x32x32xf16>) -> tensor<1x4x32x1xf16> {
  %1 = IE.ReduceSum(%arg0) {axes_value = [3], keep_dims} : tensor<1x4x32x32xf16> -> tensor<1x4x32x1xf16>
  return %1 : tensor<1x4x32x1xf16>

  // CHECK-NOT:   Convolution
}


// -----

!qElemType = !quant.uniform<u8:f16, 0.010857077205882353:127>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertReduceSumToConvU8
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x32x31x!qElemType>
func.func @ConvertReduceSumToConvU8(%arg0: tensor<1x4x32x31x!qElemType>) -> tensor<1x1x32x31x!qElemType> {
  %1 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x4x32x31x!qElemType> -> tensor<1x1x32x31x!qElemType>
  return %1 : tensor<1x1x32x31x!qElemType>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x4x1x1x!qElemType1>
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x32x31x!qElemType>, tensor<1x4x1x1x!qElemType1> -> tensor<1x1x32x31x!qElemType>

  // CHECK:       return [[CONV]] : tensor<1x1x32x31x!qElemType>
}


// CHECK-LABEL: @ConvetWithNCEParent
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x34x34xf16>
func.func @ConvetWithNCEParent(%arg0: tensor<1x4x34x34xf16>) -> tensor<1x1x32x32xf16> {
  %0 = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [3, 3],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x4x34x34xf16> -> tensor<1x4x32x32xf16>

  %1 = IE.ReduceSum(%0) {axes_value = [1], keep_dims} : tensor<1x4x32x32xf16> -> tensor<1x1x32x32xf16>
  return %1 : tensor<1x1x32x32xf16>

  // CHECK-DAG:[[CST:%.+]] = const.Declare tensor<1x4x1x1xf16> = dense<1.000000e+00> : tensor<1x4x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK:    [[POOL:%.+]] = IE.AvgPool([[INPUT]]) {exclude_pads, kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x34x34xf16> -> tensor<1x4x32x32xf16>
  // CHECK:    [[CONV:%.+]] = IE.Convolution([[POOL]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x32x32xf16>, tensor<1x4x1x1xf16> -> tensor<1x1x32x32xf16>
  // CHECK:    return [[CONV]] : tensor<1x1x32x32xf16>
}



// CHECK-LABEL: @NotConverNoNCEParent
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x32x32xf16>
func.func @NotConverNoNCEParent(%arg0: tensor<1x4x32x32xf16>) -> tensor<1x1x32x32xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x4x32x32xf16> -> tensor<1x1x32x32xf16>
  return %0 : tensor<1x1x32x32xf16>

  // CHECK:       [[REDUCE:%.+]] = IE.ReduceSum([[INPUT]]) {axes_value = [1], keep_dims} : tensor<1x4x32x32xf16> -> tensor<1x1x32x32xf16>
  // CHECK:       return  [[REDUCE]] : tensor<1x1x32x32xf16>
}
