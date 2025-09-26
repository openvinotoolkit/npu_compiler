//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-mvn6-to-mvn1 %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1Case2D
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<5x17xf16>)
func.func @ConvertMVN6ToMVN1Case2D(%arg0: tensor<5x17xf16>) -> tensor<5x17xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [1], eps = 9.9999997473787516E-5 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<5x17xf16> -> tensor<5x17xf16>
    return %0 : tensor<5x17xf16>

    // CHECK:       [[INPUT_RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 5, 17, 1]} : tensor<5x17xf16> -> tensor<1x5x17x1xf16>
    // CHECK:       [[MVN:%.+]] = IE.MVN([[INPUT_RESHAPE]]) {across_channels = false, eps = 9.9999997473787516E-5 : f64, normalize_variance = true} : tensor<1x5x17x1xf16> -> tensor<1x5x17x1xf16>
    // CHECK:       [[OUTPUT:%.+]] = IE.Reshape([[MVN]]) {shape_value = [5, 17]} : tensor<1x5x17x1xf16> -> tensor<5x17xf16>
    // CHECK:       return [[OUTPUT]] : tensor<5x17xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1Case3D
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x48x48xf16>)
func.func @ConvertMVN6ToMVN1Case3D(%arg0: tensor<1x48x48xf16>) -> tensor<1x48x48xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [1, 2], eps = 9.9999997473787516E-5 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x48x48xf16> -> tensor<1x48x48xf16>
    return %0 : tensor<1x48x48xf16>

    // CHECK:       [[INPUT_RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 48, 48]} : tensor<1x48x48xf16> -> tensor<1x1x48x48xf16>
    // CHECK:       [[MVN:%.+]] = IE.MVN([[INPUT_RESHAPE]]) {across_channels = false, eps = 9.9999997473787516E-5 : f64, normalize_variance = true} : tensor<1x1x48x48xf16> -> tensor<1x1x48x48xf16>
    // CHECK:       [[OUTPUT:%.+]] = IE.Reshape([[MVN]]) {shape_value = [1, 48, 48]} : tensor<1x1x48x48xf16> -> tensor<1x48x48xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x48x48xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1Case4D
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<4x10x5x17xf16>)
func.func @ConvertMVN6ToMVN1Case4D(%arg0: tensor<4x10x5x17xf16>) -> tensor<4x10x5x17xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [3], eps = 5.000000e-01 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<4x10x5x17xf16> -> tensor<4x10x5x17xf16>
    return %0 : tensor<4x10x5x17xf16>

    // CHECK:       [[INPUT_RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [4, 50, 1, 17]} : tensor<4x10x5x17xf16> -> tensor<4x50x1x17xf16>
    // CHECK:       [[MVN:%.+]] = IE.MVN([[INPUT_RESHAPE]]) {across_channels = false, eps = 5.000000e-01 : f64, normalize_variance = true} : tensor<4x50x1x17xf16> -> tensor<4x50x1x17xf16>
    // CHECK:       [[OUTPUT:%.+]] = IE.Reshape([[MVN]]) {shape_value = [4, 10, 5, 17]} : tensor<4x50x1x17xf16> -> tensor<4x10x5x17xf16>
    // CHECK:       return [[OUTPUT]] : tensor<4x10x5x17xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1Case5D
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x32x20x20x20xf16>)
func.func @ConvertMVN6ToMVN1Case5D(%arg0: tensor<1x32x20x20x20xf16>) -> tensor<1x32x20x20x20xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [2, 3, 4], eps = 9.9999997473787516E-5 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x32x20x20x20xf16> -> tensor<1x32x20x20x20xf16>
    return %0 : tensor<1x32x20x20x20xf16>

    // CHECK:       [[INPUT_RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 32, 20, 400]} : tensor<1x32x20x20x20xf16> -> tensor<1x32x20x400xf16>
    // CHECK:       [[MVN:%.+]] = IE.MVN([[INPUT_RESHAPE]]) {across_channels = false, eps = 9.9999997473787516E-5 : f64, normalize_variance = true} : tensor<1x32x20x400xf16> -> tensor<1x32x20x400xf16>
    // CHECK:       [[OUTPUT:%.+]] = IE.Reshape([[MVN]]) {shape_value = [1, 32, 20, 20, 20]} : tensor<1x32x20x400xf16> -> tensor<1x32x20x20x20xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x32x20x20x20xf16>
}

// -----

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1NotApplied
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<4x10x5x17xf16>)
func.func @ConvertMVN6ToMVN1NotApplied(%arg0: tensor<4x10x5x17xf16>) -> tensor<4x10x5x17xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [0], eps = 5.000000e-01 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<4x10x5x17xf16> -> tensor<4x10x5x17xf16>
    return %0 : tensor<4x10x5x17xf16>

    // CHECK:       [[MVN:%.+]] = IE.MVN6([[INPUT]]) {axes_value = [0], eps = 5.000000e-01 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<4x10x5x17xf16> -> tensor<4x10x5x17xf16>
    // CHECK:       return [[MVN]] : tensor<4x10x5x17xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1TransposeAxisCToSpatial
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x13x28x14xf32>)
func.func @ConvertMVN6ToMVN1TransposeAxisCToSpatial(%arg0: tensor<1x13x28x14xf32>) -> tensor<1x13x28x14xf32> {
    %0 = IE.MVN6(%arg0) {axes_value = [1], eps = 9.9999999747524271E-7 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x13x28x14xf32> -> tensor<1x13x28x14xf32>
    return %0 : tensor<1x13x28x14xf32>

    // CHECK:       [[TRANS_IN:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NHWC} : tensor<1x13x28x14xf32> -> tensor<1x28x14x13xf32>
    // CHECK:       [[RESHAPE_IN:%.+]] = IE.Reshape([[TRANS_IN]]) {shape_value = [1, 392, 1, 13]} : tensor<1x28x14x13xf32> -> tensor<1x392x1x13xf32>
    // CHECK:       [[MVN:%.+]] = IE.MVN([[RESHAPE_IN]]) {across_channels = false, eps = 9.9999999747524271E-7 : f64, normalize_variance = true} : tensor<1x392x1x13xf32> -> tensor<1x392x1x13xf32>
    // CHECK:       [[RESHAPE_OUT:%.+]] = IE.Reshape([[MVN]]) {shape_value = [1, 28, 14, 13]} : tensor<1x392x1x13xf32> -> tensor<1x28x14x13xf32>
    // CHECK:       [[TRANS_OUT:%.+]] = IE.Transpose([[RESHAPE_OUT]]) {order_value = #NWCH} : tensor<1x28x14x13xf32> -> tensor<1x13x28x14xf32>
    // CHECK:       return [[TRANS_OUT]] : tensor<1x13x28x14xf32>
}

// -----

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1Case2DDynamic
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 32]> : tensor<2xsi64>, order = #NC}>)
!dynType2D = tensor<1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 32]> : tensor<2xsi64>, order = affine_map<(d0, d1) -> (d0, d1)>}>
func.func @ConvertMVN6ToMVN1Case2DDynamic(%arg0: !dynType2D) -> !dynType2D {
    %0 = IE.MVN6(%arg0) {axes_value = [1], eps = 9.9999997171806853E-10 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : !dynType2D -> !dynType2D
    return %0 : !dynType2D

    // CHECK: [[RESHAPE_CST_2D:%.+]] = const.Declare tensor<2xsi32> = dense<[1, -1]> : tensor<2xsi32>
    // CHECK: [[RESHAPE_CST_4D:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 1, -1, 1]> : tensor<4xsi32>
    // CHECK: [[RESHAPE_IN:%.+]] = IE.DynamicReshape([[INPUT]], [[RESHAPE_CST_4D]]) {output_bounds = [1, 1, 1, 1], output_shape = [1, 1, -9223372036854775808, 1]} : tensor<1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 32]> : tensor<2xsi64>, order = #NC}>, tensor<4xsi32> -> tensor<1x1x?x1xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 1]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[MVN:%.+]] = IE.MVN([[RESHAPE_IN]]) {across_channels = false, eps = 9.9999997171806853E-10 : f64, normalize_variance = true} : tensor<1x1x?x1xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 1]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x1xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 1]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[MVN]], [[RESHAPE_CST_2D]]) {output_bounds = [1, 32], output_shape = [1, -9223372036854775808]} : tensor<1x1x?x1xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 1]> : tensor<4xsi64>, order = #NCHW}>, tensor<2xsi32> -> tensor<1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 32]> : tensor<2xsi64>, order = #NC}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 32]> : tensor<2xsi64>, order = #NC}>
}

// -----

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1Case3DDynamic
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 32]> : tensor<3xsi64>, order = #CHW}>)
!dynType3D = tensor<1x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 32]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
func.func @ConvertMVN6ToMVN1Case3DDynamic(%arg0: !dynType3D) -> !dynType3D {
    %0 = IE.MVN6(%arg0) {axes_value = [1, 2], eps = 9.9999997171806853E-10 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : !dynType3D -> !dynType3D
    return %0 : !dynType3D

    // CHECK: [[RESHAPE_CST_3D:%.+]] = const.Declare tensor<3xsi32> = dense<[1, -1, 32]> : tensor<3xsi32>
    // CHECK: [[RESHAPE_CST_4D:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 1, -1, 32]> : tensor<4xsi32>
    // CHECK: [[RESHAPE_IN:%.+]] = IE.DynamicReshape([[INPUT]], [[RESHAPE_CST_4D]]) {output_bounds = [1, 1, 5, 32], output_shape = [1, 1, -9223372036854775808, 32]} : tensor<1x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 32]> : tensor<3xsi64>, order = #CHW}>, tensor<4xsi32> -> tensor<1x1x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 5, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[MVN:%.+]] = IE.MVN([[RESHAPE_IN]]) {across_channels = false, eps = 9.9999997171806853E-10 : f64, normalize_variance = true} : tensor<1x1x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 5, 32]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 5, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[MVN]], [[RESHAPE_CST_3D]]) {output_bounds = [1, 5, 32], output_shape = [1, -9223372036854775808, 32]} : tensor<1x1x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 5, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<3xsi32> -> tensor<1x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 32]> : tensor<3xsi64>, order = #CHW}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 32]> : tensor<3xsi64>, order = #CHW}>
}

// -----

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1Case4DDynamic
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x5x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 7, 32]> : tensor<4xsi64>, order = #NCHW}>)
!dynType4D = tensor<1x5x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 7, 32]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
func.func @ConvertMVN6ToMVN1Case4DDynamic(%arg0: !dynType4D) -> !dynType4D {
    %0 = IE.MVN6(%arg0) {axes_value = [1, 2, 3], eps = 9.9999997171806853E-10 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : !dynType4D -> !dynType4D
    return %0 : !dynType4D

    // CHECK: [[RESHAPE_CST_4D:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 5, -1, 32]> : tensor<4xsi32>
    // CHECK: [[RESHAPE_IN:%.+]] = IE.DynamicReshape([[INPUT]], [[RESHAPE_CST_4D]]) {output_bounds = [1, 5, 7, 32], output_shape = [1, 5, -9223372036854775808, 32]} : tensor<1x5x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 7, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi32> -> tensor<1x5x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 7, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[MVN:%.+]] = IE.MVN([[RESHAPE_IN]]) {across_channels = true, eps = 9.9999997171806853E-10 : f64, normalize_variance = true} : tensor<1x5x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 7, 32]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x5x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 7, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[MVN]], [[RESHAPE_CST_4D]]) {output_bounds = [1, 5, 7, 32], output_shape = [1, 5, -9223372036854775808, 32]} : tensor<1x5x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 7, 32]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi32> -> tensor<1x5x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 7, 32]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x5x?x32xf32, {bounds = #const.OpaqueI64Elements<[1, 5, 7, 32]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1Case5DDynamic
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x32x?x20x20xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 20, 20, 20]> : tensor<5xsi64>, order = #NCDHW}>)
!dynType5D = tensor<1x32x?x20x20xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 20, 20, 20]> : tensor<5xsi64>, order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>}>
func.func @ConvertMVN6ToMVN1Case5DDynamic(%arg0: !dynType5D) -> !dynType5D {
    %0 = IE.MVN6(%arg0) {axes_value = [2, 3, 4], eps = 9.9999997171806853E-10 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : !dynType5D -> !dynType5D
    return %0 : !dynType5D

    // CHECK: [[RESHAPE_CST_5D:%.+]] = const.Declare tensor<5xsi32> = dense<[1, 32, -1, 20, 20]> : tensor<5xsi32>
    // CHECK: [[RESHAPE_CST_4D:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 32, -1, 400]> : tensor<4xsi32>
    // CHECK: [[RESHAPE_IN:%.+]] = IE.DynamicReshape([[INPUT]], [[RESHAPE_CST_4D]]) {output_bounds = [1, 32, 20, 400], output_shape = [1, 32, -9223372036854775808, 400]} : tensor<1x32x?x20x20xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 20, 20, 20]> : tensor<5xsi64>, order = #NCDHW}>, tensor<4xsi32> -> tensor<1x32x?x400xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 20, 400]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[MVN:%.+]] = IE.MVN([[RESHAPE_IN]]) {across_channels = false, eps = 9.9999997171806853E-10 : f64, normalize_variance = true} : tensor<1x32x?x400xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 20, 400]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x32x?x400xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 20, 400]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[MVN]], [[RESHAPE_CST_5D]]) {output_bounds = [1, 32, 20, 20, 20], output_shape = [1, 32, -9223372036854775808, 20, 20]} : tensor<1x32x?x400xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 20, 400]> : tensor<4xsi64>, order = #NCHW}>, tensor<5xsi32> -> tensor<1x32x?x20x20xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 20, 20, 20]> : tensor<5xsi64>, order = #NCDHW}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x32x?x20x20xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 20, 20, 20]> : tensor<5xsi64>, order = #NCDHW}>
}
