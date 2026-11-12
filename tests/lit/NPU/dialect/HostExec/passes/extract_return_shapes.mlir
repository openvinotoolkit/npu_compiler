//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --extract-return-shapes %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: module @dynamicShape {
module @dynamicShape {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  }

// CHECK:   net.NetworkInfo entryPoint : @main inputsInfo : {
// CHECK:     DataInfo "input" : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:   } outputsInfo : {
// CHECK:     DataInfo "output" : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:     DataInfo "out_0" : tensor<4xi64>

// CHECK:   func.func @main
// CHECK:   [[ARG:%.+]]: tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
func.func @main(%arg: tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>)
  -> tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}> {
  %cst = const.Declare tensor<32x16x3x3xf16, {order = #NCHW}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
  // CHECK:   const.Declare

  %conv = IE.Convolution(%arg, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
            tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>,
            tensor<32x16x3x3xf16, {order = #NCHW}> -> tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[CONV:%.+]] = IE.Convolution

  // CHECK:   [[CST_0:%.+]] = arith.constant 0 : index
  // CHECK:   [[DIM_0:%.+]] = tensor.dim [[CONV]], [[CST_0]] : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[IND_CAST_0:%.+]] = arith.index_cast [[DIM_0]] : index to i64

  // CHECK:   [[CST_1:%.+]] = arith.constant 1 : index
  // CHECK:   [[DIM_1:%.+]] = tensor.dim [[CONV]], [[CST_1]] : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[IND_CAST_1:%.+]] = arith.index_cast [[DIM_1]] : index to i64

  // CHECK:   [[CST_2:%.+]] = arith.constant 2 : index
  // CHECK:   [[DIM_2:%.+]] = tensor.dim [[CONV]], [[CST_2]] : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[IND_CAST_2:%.+]] = arith.index_cast [[DIM_2]] : index to i64

  // CHECK:   [[CST_3:%.+]] = arith.constant 3 : index
  // CHECK:   [[DIM_3:%.+]] = tensor.dim [[CONV]], [[CST_3]] : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[IND_CAST_3:%.+]] = arith.index_cast [[DIM_3]] : index to i64

  // CHECK:   [[FROM_ELEMENTS:%.+]] = tensor.from_elements [[IND_CAST_0]], [[IND_CAST_1]], [[IND_CAST_2]], [[IND_CAST_3]] : tensor<4xi64>

  return %conv : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   return [[CONV]], [[FROM_ELEMENTS]] : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xi64>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: module @dynamicShapeMultipleOutputs {
module @dynamicShapeMultipleOutputs {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
    DataInfo "output_1" : tensor<1x32x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 30, 30]> : tensor<4xsi64>, order = #NCHW}>
  }

// CHECK:   net.NetworkInfo entryPoint : @main inputsInfo : {
// CHECK:     DataInfo "input" : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:   } outputsInfo : {
// CHECK:     DataInfo "output_0" : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:     DataInfo "output_1" : tensor<1x32x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 30, 30]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:     DataInfo "out_0" : tensor<4xi64>
// CHECK:     DataInfo "out_1" : tensor<4xi64>

// CHECK:   func.func @main
// CHECK:   [[ARG:%.+]]: tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
func.func @main(%arg: tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>)
  -> (tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x32x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 30, 30]> : tensor<4xsi64>, order = #NCHW}>) {
  %cst_0 = const.Declare tensor<32x16x3x3xf16, {order = #NCHW}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
  // CHECK:   const.Declare
  %cst_1 = const.Declare tensor<32x16x5x5xf16, {order = #NCHW}> = dense<1.000000e+00> : tensor<32x16x5x5xf16>
  // CHECK:   const.Declare

  %conv_0 = IE.Convolution(%arg, %cst_0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
            tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>,
            tensor<32x16x3x3xf16, {order = #NCHW}> -> tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[CONV_0:%.+]] = IE.Convolution

  %conv_1 = IE.Convolution(%arg, %cst_1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
            tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>,
            tensor<32x16x5x5xf16, {order = #NCHW}> -> tensor<1x32x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 30, 30]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[CONV_1:%.+]] = IE.Convolution

  // CHECK:   [[CST_0_0:%.+]] = arith.constant 0 : index
  // CHECK:   [[DIM_0_0:%.+]] = tensor.dim [[CONV_0]], [[CST_0_0]] : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[IND_CAST_0_0:%.+]] = arith.index_cast [[DIM_0_0]] : index to i64

  // CHECK:   [[CST_0_1:%.+]] = arith.constant 1 : index
  // CHECK:   [[DIM_0_1:%.+]] = tensor.dim [[CONV_0]], [[CST_0_1]] : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[IND_CAST_0_1:%.+]] = arith.index_cast [[DIM_0_1]] : index to i64

  // CHECK:   [[CST_0_2:%.+]] = arith.constant 2 : index
  // CHECK:   [[DIM_0_2:%.+]] = tensor.dim [[CONV_0]], [[CST_0_2]] : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[IND_CAST_0_2:%.+]] = arith.index_cast [[DIM_0_2]] : index to i64

  // CHECK:   [[CST_0_3:%.+]] = arith.constant 3 : index
  // CHECK:   [[DIM_0_3:%.+]] = tensor.dim [[CONV_0]], [[CST_0_3]] : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[IND_CAST_0_3:%.+]] = arith.index_cast [[DIM_0_3]] : index to i64

  // CHECK:   [[FROM_ELEMENTS_0:%.+]] = tensor.from_elements [[IND_CAST_0_0]], [[IND_CAST_0_1]], [[IND_CAST_0_2]], [[IND_CAST_0_3]] : tensor<4xi64>

  // CHECK:   [[CST_1_0:%.+]] = arith.constant 0 : index
  // CHECK:   [[DIM_1_0:%.+]] = tensor.dim [[CONV_1]], [[CST_1_0]] : tensor<1x32x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 30, 30]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[IND_CAST_1_0:%.+]] = arith.index_cast [[DIM_1_0]] : index to i64

  // CHECK:   [[CST_1_1:%.+]] = arith.constant 1 : index
  // CHECK:   [[DIM_1_1:%.+]] = tensor.dim [[CONV_1]], [[CST_1_1]] : tensor<1x32x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 30, 30]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[IND_CAST_1_1:%.+]] = arith.index_cast [[DIM_1_1]] : index to i64

  // CHECK:   [[CST_1_2:%.+]] = arith.constant 2 : index
  // CHECK:   [[DIM_1_2:%.+]] = tensor.dim [[CONV_1]], [[CST_1_2]] : tensor<1x32x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 30, 30]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[IND_CAST_1_2:%.+]] = arith.index_cast [[DIM_1_2]] : index to i64

  // CHECK:   [[CST_1_3:%.+]] = arith.constant 3 : index
  // CHECK:   [[DIM_1_3:%.+]] = tensor.dim [[CONV_1]], [[CST_1_3]] : tensor<1x32x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 30, 30]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   [[IND_CAST_1_3:%.+]] = arith.index_cast [[DIM_1_3]] : index to i64

  // CHECK:   [[FROM_ELEMENTS_1:%.+]] = tensor.from_elements [[IND_CAST_1_0]], [[IND_CAST_1_1]], [[IND_CAST_1_2]], [[IND_CAST_1_3]] : tensor<4xi64>

  return %conv_0, %conv_1 : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>,
                            tensor<1x32x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 30, 30]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   return [[CONV_0]], [[CONV_1]], [[FROM_ELEMENTS_0]], [[FROM_ELEMENTS_1]]
  // CHECK-SAME:   : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>,
  // CHECK-SAME:     tensor<1x32x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 30, 30]> : tensor<4xsi64>, order = #NCHW}>,
  // CHECK-SAME:     tensor<4xi64>, tensor<4xi64>
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
  }

// CHECK:   net.NetworkInfo entryPoint : @main inputsInfo : {
// CHECK:     DataInfo "input" : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:   } outputsInfo : {
// CHECK:     DataInfo "output" : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 16, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:     DataInfo "out_0" : tensor<4xi64>

// CHECK:   func.func @main
// CHECK:   [[ARG:%.+]]: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
func.func @main(%arg: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>)
    -> (tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>) {
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

    // CHECK:   [[CST_0:%.+]] = arith.constant 0 : index
    // CHECK:   [[DIM_0:%.+]] = tensor.dim [[CONV_1]], [[CST_0]] : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[IND_CAST_0:%.+]] = arith.index_cast [[DIM_0]] : index to i64

    // CHECK:   [[CST_1:%.+]] = arith.constant 1 : index
    // CHECK:   [[DIM_1:%.+]] = tensor.dim [[CONV_1]], [[CST_1]] : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[IND_CAST_1:%.+]] = arith.index_cast [[DIM_1]] : index to i64

    // CHECK:   [[CST_2:%.+]] = arith.constant 2 : index
    // CHECK:   [[DIM_2:%.+]] = tensor.dim [[CONV_1]], [[CST_2]] : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[IND_CAST_2:%.+]] = arith.index_cast [[DIM_2]] : index to i64

    // CHECK:   [[CST_3:%.+]] = arith.constant 3 : index
    // CHECK:   [[DIM_3:%.+]] = tensor.dim [[CONV_1]], [[CST_3]] : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[IND_CAST_3:%.+]] = arith.index_cast [[DIM_3]] : index to i64

    // CHECK:   [[FROM_ELEMENTS:%.+]] = tensor.from_elements [[IND_CAST_0]], [[IND_CAST_1]], [[IND_CAST_2]], [[IND_CAST_3]] : tensor<4xi64>

    return %conv_2 : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   return [[CONV_1]], [[FROM_ELEMENTS]] : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xi64>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: module @staticShape {
module @staticShape {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x32x32xf16, {order = #NCHW}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x32x32xf16, {order = #NCHW}>
  }

// CHECK:   func.func @main
// CHECK:   [[ARG:%.+]]: tensor<1x16x32x32xf16, {order = #NCHW}>
func.func @main(%arg: tensor<1x16x32x32xf16, {order = #NCHW}>)
  -> tensor<1x32x32x32xf16, {order = #NCHW}> {
  %cst = const.Declare tensor<32x16x3x3xf16, {order = #NCHW}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
  // CHECK:   const.Declare

  %conv = IE.Convolution(%arg, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
            tensor<1x16x32x32xf16, {order = #NCHW}>,
            tensor<32x16x3x3xf16, {order = #NCHW}> -> tensor<1x32x32x32xf16, {order = #NCHW}>
  // CHECK:   [[CONV:%.+]] = IE.Convolution

  // CHECK:   [[CST_0:%.+]] = arith.constant 0 : index
  // CHECK:   [[DIM_0:%.+]] = tensor.dim [[CONV]], [[CST_0]] : tensor<1x32x32x32xf16, {order = #NCHW}>
  // CHECK:   [[IND_CAST_0:%.+]] = arith.index_cast [[DIM_0]] : index to i64

  // CHECK:   [[CST_1:%.+]] = arith.constant 1 : index
  // CHECK:   [[DIM_1:%.+]] = tensor.dim [[CONV]], [[CST_1]] : tensor<1x32x32x32xf16, {order = #NCHW}>
  // CHECK:   [[IND_CAST_1:%.+]] = arith.index_cast [[DIM_1]] : index to i64

  // CHECK:   [[CST_2:%.+]] = arith.constant 2 : index
  // CHECK:   [[DIM_2:%.+]] = tensor.dim [[CONV]], [[CST_2]] : tensor<1x32x32x32xf16, {order = #NCHW}>
  // CHECK:   [[IND_CAST_2:%.+]] = arith.index_cast [[DIM_2]] : index to i64

  // CHECK:   [[CST_3:%.+]] = arith.constant 3 : index
  // CHECK:   [[DIM_3:%.+]] = tensor.dim [[CONV]], [[CST_3]] : tensor<1x32x32x32xf16, {order = #NCHW}>
  // CHECK:   [[IND_CAST_3:%.+]] = arith.index_cast [[DIM_3]] : index to i64

  // CHECK:   [[FROM_ELEMENTS:%.+]] = tensor.from_elements [[IND_CAST_0]], [[IND_CAST_1]], [[IND_CAST_2]], [[IND_CAST_3]] : tensor<4xi64>

  return %conv : tensor<1x32x32x32xf16, {order = #NCHW}>
  // CHECK:   return [[CONV]], [[FROM_ELEMENTS]] : tensor<1x32x32x32xf16, {order = #NCHW}>, tensor<4xi64>
}

}
