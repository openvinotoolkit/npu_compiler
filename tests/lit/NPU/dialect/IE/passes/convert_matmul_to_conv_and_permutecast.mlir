//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-matmul-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @Convert3dMatMulToConvAndPermutecast_transpose_b
func.func @Convert3dMatMulToConvAndPermutecast_transpose_b(%arg0: tensor<64x100x64xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x100x64xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<64x100x64xf16>, tensor<64x64xf16> -> tensor<64x100x64xf16>

  return %0 : tensor<64x100x64xf16>

    // CHECK:       %[[RESHAPE_1:.*]] = IE.Reshape(%arg1) {shape_value = [64, 64, 1, 1]} : tensor<64x64xf16> -> tensor<64x64x1x1xf16>
    // CHECK:       %[[RESHAPE_2:.*]] = IE.Reshape(%arg0) {shape_value = [1, 64, 100, 64]} : tensor<64x100x64xf16> -> tensor<1x64x100x64xf16>
    // CHECK:       %[[PERMUTE_2:.*]] = IE.PermuteCast(%[[RESHAPE_2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x64x100x64xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
    // CHECK:       %[[CONV:.*]] = IE.Convolution(%[[PERMUTE_2]], %[[RESHAPE_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}
    // CHECK:       %[[PERMUTE_3:.*]] = IE.PermuteCast(%[[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x100x64xf16>
    // CHECK:       %[[RESHAPE_3:.*]] = IE.Reshape(%[[PERMUTE_3]]) {shape_value = [64, 100, 64]} : tensor<1x64x100x64xf16> -> tensor<64x100x64xf16>
    // CHECK:       return %[[RESHAPE_3]] : tensor<64x100x64xf16>
}

// CHECK-LABEL: @Convert3dMatMulToConvAndPermutecast_transpose_a

#CN = affine_map<(d0, d1) -> (d1, d0)>
func.func @Convert3dMatMulToConvAndPermutecast_transpose_a(%arg0: tensor<64x64x100xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x100x64xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_a} : tensor<64x64x100xf16>, tensor<64x64xf16> -> tensor<64x100x64xf16>

  return %0 : tensor<64x100x64xf16>

    // CHECK:       %[[TRANSPOSE_1:.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<64x64x100xf16> -> tensor<64x100x64xf16>
    // CHECK:       %[[TRANSPOSE_2:.*]] = IE.Transpose(%arg1) {order_value = #CN} : tensor<64x64xf16> -> tensor<64x64xf16>
    // CHECK:       %[[RESHAPE_1:.*]] = IE.Reshape(%[[TRANSPOSE_2]]) {shape_value = [64, 64, 1, 1]} : tensor<64x64xf16> -> tensor<64x64x1x1xf16>
    // CHECK:       %[[RESHAPE_2:.*]] = IE.Reshape(%[[TRANSPOSE_1]]) {shape_value = [1, 64, 100, 64]} : tensor<64x100x64xf16> -> tensor<1x64x100x64xf16>
    // CHECK:       %[[PERMUTE_2:.*]] = IE.PermuteCast(%[[RESHAPE_2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x64x100x64xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
    // CHECK:       %[[CONV:.*]] = IE.Convolution(%[[PERMUTE_2]], %[[RESHAPE_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}
    // CHECK:       %[[PERMUTE_3:.*]] = IE.PermuteCast(%[[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x100x64xf16>
    // CHECK:       %[[RESHAPE_3:.*]] = IE.Reshape(%[[PERMUTE_3]]) {shape_value = [64, 100, 64]} : tensor<1x64x100x64xf16> -> tensor<64x100x64xf16>
    // CHECK:       return %[[RESHAPE_3]] : tensor<64x100x64xf16>
}

// -----

// CHECK-LABEL: @Convert4dMatMulToConvAndPermutecast
func.func @Convert4dMatMulToConvAndPermutecast(%arg0: tensor<1x1x1x3xf16>) -> tensor<1x1x1x12xf16> {
    %cst = const.Declare tensor<12x3xf16> = dense<2.000000e+00> : tensor<12x3xf32>, [#const.CastElemType<f16>]
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<1x1x1x3xf16>, tensor<12x3xf16> -> tensor<1x1x1x12xf16>
    return %0 : tensor<1x1x1x12xf16>
    // CHECK:       [[CST_0:%.*]] = const.Declare tensor<12x3xf16> = dense<2.000000e+00> : tensor<12x3xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[RESHAPED_CST:%.*]] = IE.Reshape([[CST_0]]) {shape_value = [12, 3, 1, 1]} : tensor<12x3xf16> -> tensor<12x3x1x1xf16>
    // CHECK-DAG:   [[RESHAPED_ARG0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 1, 1, 3]} : tensor<1x1x1x3xf16> -> tensor<1x1x1x3xf16>
    // CHECK:       [[PERMUTED_CST:%.*]] = IE.PermuteCast([[RESHAPED_ARG0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x1x3xf16> -> tensor<1x3x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CONV_RET:%.*]] = IE.Convolution([[PERMUTED_CST]], [[RESHAPED_CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:       [[PERMUTED_RET:%.*]] = IE.PermuteCast([[CONV_RET]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x12x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x12xf16>
    // CHECK:       [[RET:%.*]] = IE.Reshape([[PERMUTED_RET]]) {shape_value = [1, 1, 1, 12]} : tensor<1x1x1x12xf16> -> tensor<1x1x1x12xf16>
    // CHECK:       return [[RET]]
}

// -----

// CHECK-LABEL: @Convert4dMatMulToConvAndPermutecastCGreatThan1
func.func @Convert4dMatMulToConvAndPermutecastCGreatThan1(%arg0: tensor<1x2x1x3xf16>) -> tensor<1x2x1x12xf16> {
    %cst = const.Declare tensor<12x3xf16> = dense<2.000000e+00> : tensor<12x3xf32>, [#const.CastElemType<f16>]
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<1x2x1x3xf16>, tensor<12x3xf16> -> tensor<1x2x1x12xf16>
    return %0 : tensor<1x2x1x12xf16>

    // CHECK:       [[CST_0:%.*]] = const.Declare tensor<12x3xf16> = dense<2.000000e+00> : tensor<12x3xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[RESHAPED_CST:%.*]] = IE.Reshape([[CST_0]]) {shape_value = [12, 3, 1, 1]} : tensor<12x3xf16> -> tensor<12x3x1x1xf16>
    // CHECK-DAG:   [[RESHAPED_ARG0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 2, 1, 3]} : tensor<1x2x1x3xf16> -> tensor<1x2x1x3xf16>
    // CHECK:       [[PERMUTED_CST:%.*]] = IE.PermuteCast([[RESHAPED_ARG0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x2x1x3xf16> -> tensor<1x3x2x1xf16, {order = #NHWC}>
    // CHECK:       [[CONV_RET:%.*]] = IE.Convolution([[PERMUTED_CST]], [[RESHAPED_CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:       [[PERMUTED_RET:%.*]] = IE.PermuteCast([[CONV_RET]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x12x2x1xf16, {order = #NHWC}> -> tensor<1x2x1x12xf16>
    // CHECK:       [[RET:%.*]] = IE.Reshape([[PERMUTED_RET]]) {shape_value = [1, 2, 1, 12]} : tensor<1x2x1x12xf16> -> tensor<1x2x1x12xf16>
    // CHECK:       return [[RET]]
}

// -----

// CHECK-LABEL: @FailConvert4dMatMulToConvAndPermutecast
func.func @FailConvert4dMatMulToConvAndPermutecast(%arg0: tensor<2x1x1x3xf16>) -> tensor<2x1x1x12xf16> {
    %cst = const.Declare tensor<12x3xf16> = dense<2.000000e+00> : tensor<12x3xf32>, [#const.CastElemType<f16>]
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<2x1x1x3xf16>, tensor<12x3xf16> -> tensor<2x1x1x12xf16>
    return %0 : tensor<2x1x1x12xf16>
    // CHECK:       [[CST_0:%.*]] = const.Declare tensor<12x3xf16> = dense<2.000000e+00> : tensor<12x3xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[RET:%.*]] = IE.MatMul(%arg0, [[CST_0]]) {transpose_b} : tensor<2x1x1x3xf16>, tensor<12x3xf16> -> tensor<2x1x1x12xf16>
    // CHECK:       return [[RET]]
}
