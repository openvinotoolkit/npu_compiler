//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --low-precision %s | FileCheck %s
// REQUIRES: platform-NPU5010

!qElemType = !quant.uniform<i8:f16, 0.12012867647058824>
!qElemType1 = !quant.uniform<i8:f16, 0.060064338235294119>
!qElemType2 = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x256x2048xf32>, [[ARG1:%.+]]: tensor<2048x2048xsi8>)
func.func @main(%arg0: tensor<1x256x2048xf32>, %arg1: tensor<2048x2048xsi8>) -> tensor<256x2048x1x1xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<7.62890625> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-7.6875> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x2048xf32> -> tensor<1x1x256x2048xf32>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x256x2048xf32> -> tensor<1x1x256x2048xf16>
    %2 = IE.FakeQuantize(%1, %cst_3, %cst_2, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x1x256x2048xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x2048xf16>
    %3 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType2} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType2>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<2048x2048x!qElemType2> -> tensor<2048x2048xf16>
    %5 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]} : tensor<1x1x256x2048xf16> -> tensor<256x2048x1x1xf16>
    %6 = IE.AffineReshape(%4) {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]} : tensor<2048x2048xf16> -> tensor<2048x2048x1x1xf16>
    %7 = IE.Convolution(%5, %6) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x2048x1x1xf16>, tensor<2048x2048x1x1xf16> -> tensor<256x2048x1x1xf16>
    %8 = IE.FakeQuantize(%7, %cst_3, %cst_2, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<256x2048x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<256x2048x1x1xf16>
    return %8 : tensor<256x2048x1x1xf16>

    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 256, 2048]}
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[RESHAPE0]]) {dstElemType = f16}
    // CHECK:       [[ADD:%.+]] = IE.Add([[CONVERT]], [[CONVERT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x256x2048xf16>, tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048x!qElemType>
    // CHECK:       [[QCAST_IN:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = !qElemType1} : tensor<1x1x256x2048x!qElemType> -> tensor<1x1x256x2048x!qElemType1>
    // CHECK:       [[QCAST_W:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType2} : tensor<2048x2048xsi8> -> tensor<2048x2048x!qElemType2>
    // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[QCAST_IN]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [256, 2048, 1, 1]}
    // CHECK:       [[RESHAPE_W:%.+]] = IE.AffineReshape([[QCAST_W]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [2048, 2048, 1, 1]}
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE_IN]], [[RESHAPE_W]]) {
    // CHECK-SAME:      clamp = {max = 7.6281709671020508 : f64, min = -7.6882352828979492 : f64}
    // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME: } : tensor<256x2048x1x1x!qElemType1>, tensor<2048x2048x1x1x!qElemType2> -> tensor<256x2048x1x1xf16>
    // CHECK:       return [[CONV]] : tensor<256x2048x1x1xf16>
}
