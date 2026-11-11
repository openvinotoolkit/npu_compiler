//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --outline-dim-operations --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: module @dynamicShape {
module @dynamicShape {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
    DataInfo "out_0" : tensor<4xi64>
  }

// CHECK:   net.NetworkInfo entryPoint : @main inputsInfo : {
// CHECK:     DataInfo "input" : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:   } outputsInfo : {
// CHECK:     DataInfo "output" : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:   func.func @output_shape
// CHECK:     [[ARG:%.+]]: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:     [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK:     [[CST_16_i64:%.+]] = arith.constant 16 : i64
// CHECK:     [[CST_32_i64:%.+]] = arith.constant 32 : i64
// CHECK:     [[CST_3:%.+]] = arith.constant 3 : index
// CHECK:     [[CST_2:%.+]] = arith.constant 2 : index
// CHECK:     [[DIM:%.+]] = tensor.dim [[ARG]], [[CST_3]] : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:     [[DIV:%.+]] = arith.divsi [[DIM]], [[CST_2]] : index
// CHECK:     [[IND_CAST:%.+]] = arith.index_cast [[DIV]] : index to i64
// CHECK:     [[FROM_ELEMENTS:%.+]] = tensor.from_elements [[CST_1_i64]], [[CST_16_i64]], [[CST_32_i64]], [[IND_CAST]] : tensor<4xi64>
// CHECK:     return [[FROM_ELEMENTS]] : tensor<4xi64>

// CHECK:   func.func @main
// CHECK:   [[IN:%.+]]: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
func.func @main(%arg: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>)
  -> (tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xi64>) {
  %maxpool = IE.MaxPool(%arg) {
            kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
      -> tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:     [[MAXPOOL:%.+]] = IE.MaxPool

  // these operations below come from `extract-return-shapes` and `resolve-shaped-type-result-dims`
  // passes that resolve `tensor.dim` of result of operations
  %cst_1_i64 = arith.constant 1 : i64
  %cst_16_i64 = arith.constant 16 : i64
  %cst_32_i64 = arith.constant 32 : i64
  // CHECK-NOT: arith.constant

  %cst_3 = arith.constant 3 : index
  %cst_2 = arith.constant 2 : index
  %dim = tensor.dim %arg, %cst_3 : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
  %div = arith.divsi %dim, %cst_2 : index
  %idx_cast = arith.index_cast %div : index to i64
  // CHECK-NOT: tensor.dim
  // CHECK-NOT: arith.divsi
  // CHECK-NOT: arith.index_cast

  %from_elements = tensor.from_elements %cst_1_i64, %cst_16_i64, %cst_32_i64, %idx_cast : tensor<4xi64>
  // CHECK-NOT: tensor.from_elements

  return %maxpool, %from_elements : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xi64>
  // CHECK:     return [[MAXPOOL]] : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @dynamicShapeChainOfOperations
module @dynamicShapeChainOfOperations {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 16, 64]> : tensor<4xsi64>, order = #NCHW}>
    DataInfo "out_0" : tensor<4xi64>
  }

// CHECK:   net.NetworkInfo entryPoint : @main inputsInfo : {
// CHECK:     DataInfo "input" : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:   } outputsInfo : {
// CHECK:     DataInfo "output" : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 16, 64]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:   func.func @output_shape
// CHECK:     [[ARG:%.+]]: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:     [[CST_2:%.+]] = arith.constant 2 : index
// CHECK:     [[CST_3:%.+]] = arith.constant 3 : index
// CHECK:     [[CST_16_i64:%.+]] = arith.constant 16 : i64
// CHECK:     [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK:     [[DIM:%.+]] = tensor.dim [[ARG]], [[CST_3]] : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:     [[DIV_0:%.+]] = arith.divsi [[DIM]], [[CST_2]] : index
// CHECK:     [[DIV_1:%.+]] = arith.divsi [[DIV_0]], [[CST_2]] : index
// CHECK:     [[IND_CAST:%.+]] = arith.index_cast [[DIV_1]] : index to i64
// CHECK:     [[FROM_ELEMENTS:%.+]] = tensor.from_elements [[CST_1_i64]], [[CST_16_i64]], [[CST_16_i64]], [[IND_CAST]] : tensor<4xi64>
// CHECK:     return [[FROM_ELEMENTS]] : tensor<4xi64>

// CHECK:   func.func @main
// CHECK:   [[ARG:%.+]]: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
func.func @main(%arg: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>)
    -> (tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xi64>) {
    %cst_1 = const.Declare tensor<32x16x3x3xf16, {order = #NCHW}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
    %cst_2 = const.Declare tensor<16x32x1x1xf16, {order = #NCHW}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>

    %maxpool_1 = IE.MaxPool(%arg) {
            kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
      -> tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[MAXPOOL_0:%.+]] = IE.MaxPool

    %conv_1 = IE.Convolution(%maxpool_1, %cst_1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
                tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>,
                tensor<32x16x3x3xf16, {order = #NCHW}> -> tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[CONV_0:%.+]] = IE.Convolution

    %relu = IE.ReLU(%conv_1) : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
                           -> tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[RELU:%.+]] = IE.ReLU

    %maxpool_2 = IE.MaxPool(%relu) {
            kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
      -> tensor<1x32x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 16, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[MAXPOOL_1:%.+]] = IE.MaxPool

    %conv_2 = IE.Convolution(%maxpool_2, %cst_2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
                tensor<1x32x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 16, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<16x32x1x1xf16, {order = #NCHW}>
                -> tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[CONV_1:%.+]] = IE.Convolution

    // these operations below come from `extract-return-shapes` and `resolve-shaped-type-result-dims`
    // passes that resolve `tensor.dim` of result of operations
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c16_i64 = arith.constant 16 : i64
    %c1_i64 = arith.constant 1 : i64
    // CHECK-NOT: arith.constant

    %dim = tensor.dim %arg, %c3 : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
    %div_0 = arith.divsi %dim, %c2 : index
    %div_1 = arith.divsi %div_0, %c2 : index
    %idx_cast = arith.index_cast %div_1 : index to i64
    // CHECK-NOT: tensor.dim
    // CHECK-NOT: arith.divsi
    // CHECK-NOT: arith.index_cast

    %from_elements = tensor.from_elements %c1_i64, %c16_i64, %c16_i64, %idx_cast : tensor<4xi64>
    // CHECK-NOT: tensor.from_elements

    return %conv_2, %from_elements : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xi64>
    // CHECK:   return [[CONV_1]] : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: module @staticShape {
module @staticShape {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x64x64xf16, {order = #NCHW}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x32x32xf16, {order = #NCHW}>
    DataInfo "out_0" : tensor<4xi64>
  }

// CHECK:   net.NetworkInfo entryPoint : @main inputsInfo : {
// CHECK:     DataInfo "input" : tensor<1x16x64x64xf16, {order = #NCHW}>
// CHECK:   } outputsInfo : {
// CHECK:     DataInfo "output" : tensor<1x16x32x32xf16, {order = #NCHW}>

// CHECK:   func.func @output_shape
// CHECK:     [[ARG:%.+]]: tensor<1x16x64x64xf16, {order = #NCHW}>
// CHECK:     [[CST:%.+]] = arith.constant dense<[1, 16, 32, 32]> : tensor<4xi64>
// CHECK:     return [[CST]] : tensor<4xi64>

// CHECK:     func.func @main
// CHECK:     [[IN:%.+]]: tensor<1x16x64x64xf16, {order = #NCHW}>
func.func @main(%arg: tensor<1x16x64x64xf16, {order = #NCHW}>) -> (tensor<1x16x32x32xf16, {order = #NCHW}>, tensor<4xi64>) {
  // this `arith.constant` operation comes from `extract-return-shapes` and `resolve-shaped-type-result-dims`
  // passes that resolve `tensor.dim` of result of operations
  %cst = arith.constant dense<[1, 16, 32, 32]> : tensor<4xi64>
  // CHECK-NOT: arith.constant
  %maxpool = IE.MaxPool(%arg) {
            kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : tensor<1x16x64x64xf16, {order = #NCHW}>
      -> tensor<1x16x32x32xf16, {order = #NCHW}>
  // CHECK:     [[MAXPOOL:%.+]] = IE.MaxPool

  return %maxpool, %cst : tensor<1x16x32x32xf16, {order = #NCHW}>, tensor<4xi64>
  // CHECK:     return [[MAXPOOL]] : tensor<1x16x32x32xf16, {order = #NCHW}>
}

}
