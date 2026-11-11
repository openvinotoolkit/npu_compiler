//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-matmul-to-conv %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @Convert3dMatMulToConvAndPermutecast_transpose_b
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<64x100x64xf16>, [[ARG_1:%[^:]+]]: tensor<64x64xf16>)
func.func @Convert3dMatMulToConvAndPermutecast_transpose_b(%arg0: tensor<64x100x64xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x100x64xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<64x100x64xf16>, tensor<64x64xf16> -> tensor<64x100x64xf16>

  return %0 : tensor<64x100x64xf16>

    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[ARG_1]]) {shape_value = [64, 64, 1, 1]} : tensor<64x64xf16> -> tensor<64x64x1x1xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[ARG_0]]) {shape_value = [1, 64, 100, 64]} : tensor<64x100x64xf16> -> tensor<1x64x100x64xf16>
    // CHECK:       [[PERMUTE_2:%.+]] = IE.PermuteCast([[RESHAPE_2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x64x100x64xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[PERMUTE_2]], [[RESHAPE_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}
    // CHECK:       [[PERMUTE_3:%.+]] = IE.PermuteCast([[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x100x64xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[PERMUTE_3]]) {shape_value = [64, 100, 64]} : tensor<1x64x100x64xf16> -> tensor<64x100x64xf16>
    // CHECK:       return [[RESHAPE_3]] : tensor<64x100x64xf16>
}

// CHECK-LABEL: @Convert3dMatMulToConvAndPermutecast_transpose_a
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<64x64x100xf16>, [[ARG_1:%[^:]+]]: tensor<64x64xf16>)

#CN = affine_map<(d0, d1) -> (d1, d0)>
func.func @Convert3dMatMulToConvAndPermutecast_transpose_a(%arg0: tensor<64x64x100xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x100x64xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_a} : tensor<64x64x100xf16>, tensor<64x64xf16> -> tensor<64x100x64xf16>

  return %0 : tensor<64x100x64xf16>

    // CHECK:       [[TRANSPOSE_1:%.+]] = IE.Transpose([[ARG_0]]) {order_value = #map} : tensor<64x64x100xf16> -> tensor<64x100x64xf16>
    // CHECK:       [[TRANSPOSE_2:%.+]] = IE.Transpose([[ARG_1]]) {order_value = #CN} : tensor<64x64xf16> -> tensor<64x64xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[TRANSPOSE_2]]) {shape_value = [64, 64, 1, 1]} : tensor<64x64xf16> -> tensor<64x64x1x1xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[TRANSPOSE_1]]) {shape_value = [1, 64, 100, 64]} : tensor<64x100x64xf16> -> tensor<1x64x100x64xf16>
    // CHECK:       [[PERMUTE_2:%.+]] = IE.PermuteCast([[RESHAPE_2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x64x100x64xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[PERMUTE_2]], [[RESHAPE_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}
    // CHECK:       [[PERMUTE_3:%.+]] = IE.PermuteCast([[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x100x64xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[PERMUTE_3]]) {shape_value = [64, 100, 64]} : tensor<1x64x100x64xf16> -> tensor<64x100x64xf16>
    // CHECK:       return [[RESHAPE_3]] : tensor<64x100x64xf16>
}

// -----

// CHECK-LABEL: @Convert4dMatMulToConvAndPermutecast
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x1x3xf16>)
func.func @Convert4dMatMulToConvAndPermutecast(%arg0: tensor<1x1x1x3xf16>) -> tensor<1x1x1x12xf16> {
    %cst = const.Declare tensor<12x3xf16> = dense<2.000000e+00> : tensor<12x3xf32>, [#const.CastElemType<f16>]
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<1x1x1x3xf16>, tensor<12x3xf16> -> tensor<1x1x1x12xf16>
    return %0 : tensor<1x1x1x12xf16>
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<12x3xf16> = dense<2.000000e+00> : tensor<12x3xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[RESHAPED_CST:%.+]] = IE.Reshape([[CST_0]]) {shape_value = [12, 3, 1, 1]} : tensor<12x3xf16> -> tensor<12x3x1x1xf16>
    // CHECK-DAG:   [[RESHAPED_ARG0:%.+]] = IE.Reshape([[ARG_0]]) {shape_value = [1, 1, 1, 3]} : tensor<1x1x1x3xf16> -> tensor<1x1x1x3xf16>
    // CHECK:       [[PERMUTED_CST:%.+]] = IE.PermuteCast([[RESHAPED_ARG0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x1x3xf16> -> tensor<1x3x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[PERMUTED_CST]], [[RESHAPED_CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:       [[PERMUTED_RET:%.+]] = IE.PermuteCast([[CONV_RET]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x12x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x12xf16>
    // CHECK:       [[RET:%.+]] = IE.Reshape([[PERMUTED_RET]]) {shape_value = [1, 1, 1, 12]} : tensor<1x1x1x12xf16> -> tensor<1x1x1x12xf16>
    // CHECK:       return [[RET]]
}

// -----

// CHECK-LABEL: @Convert4dMatMulToConvAndPermutecastCGreatThan1
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x2x1x3xf16>)
func.func @Convert4dMatMulToConvAndPermutecastCGreatThan1(%arg0: tensor<1x2x1x3xf16>) -> tensor<1x2x1x12xf16> {
    %cst = const.Declare tensor<12x3xf16> = dense<2.000000e+00> : tensor<12x3xf32>, [#const.CastElemType<f16>]
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<1x2x1x3xf16>, tensor<12x3xf16> -> tensor<1x2x1x12xf16>
    return %0 : tensor<1x2x1x12xf16>

    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<12x3xf16> = dense<2.000000e+00> : tensor<12x3xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[RESHAPED_CST:%.+]] = IE.Reshape([[CST_0]]) {shape_value = [12, 3, 1, 1]} : tensor<12x3xf16> -> tensor<12x3x1x1xf16>
    // CHECK-DAG:   [[RESHAPED_ARG0:%.+]] = IE.Reshape([[ARG_0]]) {shape_value = [1, 2, 1, 3]} : tensor<1x2x1x3xf16> -> tensor<1x2x1x3xf16>
    // CHECK:       [[PERMUTED_CST:%.+]] = IE.PermuteCast([[RESHAPED_ARG0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x2x1x3xf16> -> tensor<1x3x2x1xf16, {order = #NHWC}>
    // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[PERMUTED_CST]], [[RESHAPED_CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK:       [[PERMUTED_RET:%.+]] = IE.PermuteCast([[CONV_RET]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x12x2x1xf16, {order = #NHWC}> -> tensor<1x2x1x12xf16>
    // CHECK:       [[RET:%.+]] = IE.Reshape([[PERMUTED_RET]]) {shape_value = [1, 2, 1, 12]} : tensor<1x2x1x12xf16> -> tensor<1x2x1x12xf16>
    // CHECK:       return [[RET]]
}

// -----

// CHECK-LABEL: @FailConvert4dMatMulToConvAndPermutecast
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<2x1x1x3xf16>)
func.func @FailConvert4dMatMulToConvAndPermutecast(%arg0: tensor<2x1x1x3xf16>) -> tensor<2x1x1x12xf16> {
    %cst = const.Declare tensor<12x3xf16> = dense<2.000000e+00> : tensor<12x3xf32>, [#const.CastElemType<f16>]
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<2x1x1x3xf16>, tensor<12x3xf16> -> tensor<2x1x1x12xf16>
    return %0 : tensor<2x1x1x12xf16>
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<12x3xf16> = dense<2.000000e+00> : tensor<12x3xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[RET:%.+]] = IE.MatMul([[ARG_0]], [[CST_0]]) {transpose_b} : tensor<2x1x1x3xf16>, tensor<12x3xf16> -> tensor<2x1x1x12xf16>
    // CHECK:       return [[RET]]
}
