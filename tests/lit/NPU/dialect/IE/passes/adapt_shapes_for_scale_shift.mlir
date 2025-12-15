//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adapt-shapes-for-scale-shift --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @ConvertAdd
func.func @ConvertAdd(%arg0: tensor<19x80xf16>) -> tensor<19x80xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x80xf16> = dense<2.000000e+00> : tensor<1x80xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<19x80xf16>, tensor<1x80xf16> -> tensor<19x80xf16>

    return %ADD : tensor<19x80xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> = dense<2.000000e+00>
    // CHECK-SAME:  : tensor<1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 19, 80, 1]
    // CHECK-SAME:  } : tensor<19x80xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[TRANSPOSE_INPUT:%.*]] = IE.Transpose([[RESHAPE_INPUT]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[TRANSPOSE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x19x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[TRANSPOSE_OUT:%.*]] = IE.Transpose([[ADD]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x80x19x1xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[TRANSPOSE_OUT]]) {
    // CHECK-SAME:      shape_value = [19, 80]
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<19x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<19x80xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @ConvertMul
func.func @ConvertMul(%arg0: tensor<19x80xf16>) -> tensor<19x80xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x80xf16> = dense<2.000000e+00> : tensor<1x80xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<19x80xf16>, tensor<1x80xf16> -> tensor<19x80xf16>

    return %MUL : tensor<19x80xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> = dense<2.000000e+00>
    // CHECK-SAME:  : tensor<1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 19, 80, 1]
    // CHECK-SAME:  } : tensor<19x80xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[TRANSPOSE_INPUT:%.*]] = IE.Transpose([[RESHAPE_INPUT]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[TRANSPOSE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x19x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[TRANSPOSE_OUT:%.*]] = IE.Transpose([[MUL]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x80x19x1xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[TRANSPOSE_OUT]]) {
    // CHECK-SAME:      shape_value = [19, 80]
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<19x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<19x80xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @ConvertAddWithMul
func.func @ConvertAddWithMul(%arg0: tensor<19x80xf16>) -> tensor<19x80xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x80xf16> = dense<1.000000e+00> : tensor<1x80xf16>
    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<19x80xf16>, tensor<1x80xf16> -> tensor<19x80xf16>

    %MUL_WEIGHTS = const.Declare tensor<1x80xf16> = dense<2.000000e+00> : tensor<1x80xf16>
    %MUL = IE.Multiply(%ADD, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<19x80xf16>, tensor<1x80xf16> -> tensor<19x80xf16>

    return %MUL : tensor<19x80xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> = dense<2.000000e+00>
    // CHECK-SAME:  : tensor<1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:  : tensor<1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 19, 80, 1]
    // CHECK-SAME:  } : tensor<19x80xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[TRANSPOSE_INPUT:%.*]] = IE.Transpose([[RESHAPE_INPUT]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[TRANSPOSE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x19x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[ADD]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x19x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[TRANSPOSE_OUT:%.*]] = IE.Transpose([[MUL]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x80x19x1xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[TRANSPOSE_OUT]]) {
    // CHECK-SAME:      shape_value = [19, 80]
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<19x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<19x80xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @ConvertMulWithAdd
func.func @ConvertMulWithAdd(%arg0: tensor<19x80xf16>) -> tensor<19x80xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x80xf16> = dense<2.000000e+00> : tensor<1x80xf16>
    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<19x80xf16>, tensor<1x80xf16> -> tensor<19x80xf16>

    %ADD_WEIGHTS = const.Declare tensor<1x80xf16> = dense<1.000000e+00> : tensor<1x80xf16>
    %ADD = IE.Add(%MUL, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<19x80xf16>, tensor<1x80xf16> -> tensor<19x80xf16>

    return %ADD : tensor<19x80xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:  : tensor<1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> = dense<2.000000e+00>
    // CHECK-SAME:  : tensor<1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 19, 80, 1]
    // CHECK-SAME:  } : tensor<19x80xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[TRANSPOSE_INPUT:%.*]] = IE.Transpose([[RESHAPE_INPUT]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[TRANSPOSE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x19x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[MUL]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x19x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[TRANSPOSE_OUT:%.*]] = IE.Transpose([[ADD]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x80x19x1xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[TRANSPOSE_OUT]]) {
    // CHECK-SAME:      shape_value = [19, 80]
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<19x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<19x80xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @Convert3dAdd
func.func @Convert3dAdd(%arg0: tensor<1x19x80xf16>) -> tensor<1x19x80xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x1x80xf16> = dense<2.000000e+00> : tensor<1x1x80xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x19x80xf16>, tensor<1x1x80xf16> -> tensor<1x19x80xf16>

    return %ADD : tensor<1x19x80xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> = dense<2.000000e+00>
    // CHECK-SAME:  : tensor<1x1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 19, 80, 1]
    // CHECK-SAME:  } : tensor<1x19x80xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[TRANSPOSE_INPUT:%.*]] = IE.Transpose([[RESHAPE_INPUT]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[TRANSPOSE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x19x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[TRANSPOSE_OUT:%.*]] = IE.Transpose([[ADD]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x80x19x1xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[TRANSPOSE_OUT]]) {
    // CHECK-SAME:      shape_value = [1, 19, 80]
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<1x19x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x19x80xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @Convert3dMul
func.func @Convert3dMul(%arg0: tensor<1x19x80xf16>) -> tensor<1x19x80xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x1x80xf16> = dense<2.000000e+00> : tensor<1x1x80xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x19x80xf16>, tensor<1x1x80xf16> -> tensor<1x19x80xf16>

    return %MUL : tensor<1x19x80xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> = dense<2.000000e+00>
    // CHECK-SAME:  : tensor<1x1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 19, 80, 1]
    // CHECK-SAME:  } : tensor<1x19x80xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[TRANSPOSE_INPUT:%.*]] = IE.Transpose([[RESHAPE_INPUT]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[TRANSPOSE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x19x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[TRANSPOSE_OUT:%.*]] = IE.Transpose([[MUL]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x80x19x1xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[TRANSPOSE_OUT]]) {
    // CHECK-SAME:      shape_value = [1, 19, 80]
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<1x19x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x19x80xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @Convert3dAddWithMul
func.func @Convert3dAddWithMul(%arg0: tensor<1x19x80xf16>) -> tensor<1x19x80xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x1x80xf16> = dense<1.000000e+00> : tensor<1x1x80xf16>
    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x19x80xf16>, tensor<1x1x80xf16> -> tensor<1x19x80xf16>

    %MUL_WEIGHTS = const.Declare tensor<1x1x80xf16> = dense<2.000000e+00> : tensor<1x1x80xf16>
    %MUL = IE.Multiply(%ADD, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x19x80xf16>, tensor<1x1x80xf16> -> tensor<1x19x80xf16>

    return %MUL : tensor<1x19x80xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> = dense<2.000000e+00>
    // CHECK-SAME:  : tensor<1x1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:  : tensor<1x1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 19, 80, 1]
    // CHECK-SAME:  } : tensor<1x19x80xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[TRANSPOSE_INPUT:%.*]] = IE.Transpose([[RESHAPE_INPUT]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[TRANSPOSE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x19x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[ADD]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x19x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[TRANSPOSE_OUT:%.*]] = IE.Transpose([[MUL]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x80x19x1xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[TRANSPOSE_OUT]]) {
    // CHECK-SAME:      shape_value = [1, 19, 80]
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<1x19x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x19x80xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @Convert3dMulWithAdd
func.func @Convert3dMulWithAdd(%arg0: tensor<1x19x80xf16>) -> tensor<1x19x80xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x1x80xf16> = dense<2.000000e+00> : tensor<1x1x80xf16>
    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x19x80xf16>, tensor<1x1x80xf16> -> tensor<1x19x80xf16>

    %ADD_WEIGHTS = const.Declare tensor<1x1x80xf16> = dense<1.000000e+00> : tensor<1x1x80xf16>
    %ADD = IE.Add(%MUL, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x19x80xf16>, tensor<1x1x80xf16> -> tensor<1x19x80xf16>

    return %ADD : tensor<1x19x80xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:  : tensor<1x1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> = dense<2.000000e+00>
    // CHECK-SAME:  : tensor<1x1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 19, 80, 1]
    // CHECK-SAME:  } : tensor<1x19x80xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[TRANSPOSE_INPUT:%.*]] = IE.Transpose([[RESHAPE_INPUT]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[TRANSPOSE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x19x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[MUL]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x19x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x19x1xf16>

    // CHECK:   [[TRANSPOSE_OUT:%.*]] = IE.Transpose([[ADD]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x80x19x1xf16> -> tensor<1x19x80x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[TRANSPOSE_OUT]]) {
    // CHECK-SAME:      shape_value = [1, 19, 80]
    // CHECK-SAME:  } : tensor<1x19x80x1xf16> -> tensor<1x19x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x19x80xf16>
}

// -----

// CHECK-LABEL: @Convert4dMul
func.func @Convert4dMul(%arg0: tensor<1x19x80x80xf16>) -> tensor<1x19x80x80xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x1x80x80xf16> = dense<2.000000e+00> : tensor<1x1x80x80xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x19x80x80xf16>, tensor<1x1x80x80xf16> -> tensor<1x19x80x80xf16>

    return %MUL : tensor<1x19x80x80xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x6400x1x1xf16> = dense<2.000000e+00>
    // CHECK-SAME:  : tensor<1x1x80x80xf16>, [#const.Reshape<[1, 6400, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 19, 6400]
    // CHECK-SAME:  } : tensor<1x19x80x80xf16> -> tensor<1x1x19x6400xf16>

    // CHECK:   [[TRANSPOSE_INPUT:%.*]] = IE.Transpose([[RESHAPE_INPUT]]) {
    // CHECK-SAME:      order_value = #NWCH
    // CHECK-SAME:  } : tensor<1x1x19x6400xf16> -> tensor<1x6400x1x19xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[TRANSPOSE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x6400x1x19xf16>, tensor<1x6400x1x1xf16> -> tensor<1x6400x1x19xf16>

    // CHECK:   [[TRANSPOSE_OUT:%.*]] = IE.Transpose([[MUL]]) {
    // CHECK-SAME:      order_value = #NHWC
    // CHECK-SAME:  } : tensor<1x6400x1x19xf16> -> tensor<1x1x19x6400xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[TRANSPOSE_OUT]]) {
    // CHECK-SAME:      shape_value = [1, 19, 80, 80]
    // CHECK-SAME:  } : tensor<1x1x19x6400xf16> -> tensor<1x19x80x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x19x80x80xf16>
}

// -----

// CHECK-LABEL: @Convert4dMulWithTwoActivations
func.func @Convert4dMulWithTwoActivations(%arg0: tensor<1x19x80x80xf16>, %arg1: tensor<1x1x80x80xf16>) -> tensor<1x19x80x80xf16> {
    %MUL = IE.Multiply(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x19x80x80xf16>, tensor<1x1x80x80xf16> -> tensor<1x19x80x80xf16>

    return %MUL : tensor<1x19x80x80xf16>

    // CHECK:   [[RESHAPE_INPUT1:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 19, 6400]
    // CHECK-SAME:  } : tensor<1x19x80x80xf16> -> tensor<1x1x19x6400xf16>

    // CHECK:   [[TRANSPOSE_INPUT1:%.*]] = IE.Transpose([[RESHAPE_INPUT1]]) {
    // CHECK-SAME:      order_value = #NWCH
    // CHECK-SAME:  } : tensor<1x1x19x6400xf16> -> tensor<1x6400x1x19xf16>

    // CHECK:   [[RESHAPE_INPUT2:%.*]] = IE.AffineReshape(%arg1) {
    // CHECK-SAME:      shape_value = [1, 6400, 1, 1]
    // CHECK-SAME:  } : tensor<1x1x80x80xf16> -> tensor<1x6400x1x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[TRANSPOSE_INPUT1]], [[RESHAPE_INPUT2]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x6400x1x19xf16>, tensor<1x6400x1x1xf16> -> tensor<1x6400x1x19xf16>

    // CHECK:   [[TRANSPOSE_OUT:%.*]] = IE.Transpose([[MUL]]) {
    // CHECK-SAME:      order_value = #NHWC
    // CHECK-SAME:  } : tensor<1x6400x1x19xf16> -> tensor<1x1x19x6400xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[TRANSPOSE_OUT]]) {
    // CHECK-SAME:      shape_value = [1, 19, 80, 80]
    // CHECK-SAME:  } : tensor<1x1x19x6400xf16> -> tensor<1x19x80x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x19x80x80xf16>
}

// -----

// CHECK-LABEL: @DoNotConvert4dMul
func.func @DoNotConvert4dMul(%arg0: tensor<1x20x80x80xf16>) -> tensor<2x20x80x80xf16> {
    %MUL_WEIGHTS = const.Declare tensor<2x1x80x80xf16> = dense<2.000000e+00> : tensor<2x1x80x80xf16>
    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x20x80x80xf16>, tensor<2x1x80x80xf16> -> tensor<2x20x80x80xf16>
    return %MUL : tensor<2x20x80x80xf16>
    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<2x1x80x80xf16> = dense<2.000000e+00> : tensor<2x1x80x80xf16>
    // CHECK:   [[MUL:%.*]] = IE.Multiply(%arg0, [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x20x80x80xf16>, tensor<2x1x80x80xf16> -> tensor<2x20x80x80xf16>
    // CHECK:   return [[MUL]] : tensor<2x20x80x80xf16>
}

// -----

// CHECK-LABEL: @DoNotConvertMulChannelOverVPUDimLimit
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x1x1080x1920xf16>
func.func @DoNotConvertMulChannelOverVPUDimLimit(%arg0: tensor<1x1x1080x1920xf16>) -> tensor<1x1x1080x1920xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x1x1080x1920xf16> = dense<2.000000e+00> : tensor<1x1x1080x1920xf16>
    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x1x1080x1920xf16>, tensor<1x1x1080x1920xf16> -> tensor<1x1x1080x1920xf16>
    return %MUL : tensor<1x1x1080x1920xf16>

    // CHECK-DAG: [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x1x1080x1920xf16> = dense<2.000000e+00> : tensor<1x1x1080x1920xf16>
    // CHECK:     [[MUL:%.*]] = IE.Multiply([[INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x1080x1920xf16>, tensor<1x1x1080x1920xf16> -> tensor<1x1x1080x1920xf16>
    // CHECK:   return [[MUL]] : tensor<1x1x1080x1920xf16>
}

// -----

// CHECK-LABEL: @DoNotConvert2dAdd
func.func @DoNotConvert2dAdd(%arg0: tensor<512x1xf16>) -> tensor<512x1xf16> {
    %ADD_WEIGHTS = const.Declare tensor<512x1xf16> = dense<1.000000e+00> : tensor<512x1xf16>
    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<512x1xf16>, tensor<512x1xf16> -> tensor<512x1xf16>

    return %ADD : tensor<512x1xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<512x1xf16> = dense<1.000000e+00> : tensor<512x1xf16>
    // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<512x1xf16>, tensor<512x1xf16> -> tensor<512x1xf16>

    // CHECK:   return [[ADD]] : tensor<512x1xf16>
}

// -----

// CHECK-LABEL: @DoNotConvert3dAdd
func.func @DoNotConvert3dAdd(%arg0: tensor<1x512x1xf16>) -> tensor<1x512x1xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x512x1xf16> = dense<1.000000e+00> : tensor<1x512x1xf16>
    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x512x1xf16>, tensor<1x512x1xf16> -> tensor<1x512x1xf16>

    return %ADD : tensor<1x512x1xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x512x1xf16> = dense<1.000000e+00> : tensor<1x512x1xf16>
    // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x512x1xf16>, tensor<1x512x1xf16> -> tensor<1x512x1xf16>

    // CHECK:   return [[ADD]] : tensor<1x512x1xf16>
}

// -----

// CHECK-LABEL: @DoNotConvertTrivialShape
func.func @DoNotConvertTrivialShape(%arg0: tensor<1x1x512xf16>) -> tensor<1x1x512xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x1x512xf16> = dense<1.000000e+00> : tensor<1x1x512xf16>
    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x1x512xf16>, tensor<1x1x512xf16> -> tensor<1x1x512xf16>

    return %ADD : tensor<1x1x512xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x1x512xf16> = dense<1.000000e+00> : tensor<1x1x512xf16>
    // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x512xf16>, tensor<1x1x512xf16> -> tensor<1x1x512xf16>

    // CHECK:   return [[ADD]] : tensor<1x1x512xf16>
}

// -----

// CHECK-LABEL: @TransposeMultiply
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x64x128x64xf16>
func.func @TransposeMultiply(%arg0: tensor<1x64x128x64xf16>) -> tensor<1x64x128x64xf16> {
    %cst = const.Declare tensor<1x1x1x64xf16> = dense<1.0> : tensor<1x1x1x64xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %0 = IE.FakeQuantize(%cst, %cst_0, %cst_0, %cst_0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64}
        : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
                -> tensor<1x1x1x64xf16>
    %1 = IE.Multiply(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x128x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x64x128x64xf16>
    return %1 : tensor<1x64x128x64xf16>

    // CHECK-DAG:   [[CONST_INPUT:%.+]] = const.Declare tensor<1x1x1x64xf16> = dense<1.000000e+00> : tensor<1x1x1x64xf16>
    // CHECK-DAG:   [[CONST_FQ:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[FQ_INPUT:%.+]] = IE.FakeQuantize([[CONST_INPUT]], [[CONST_FQ]], [[CONST_FQ]], [[CONST_FQ]], [[CONST_FQ]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:      : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
    // CHECK-SAME:          -> tensor<1x1x1x64xf16>
    // CHECK:       [[ACT_TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x64x128x64xf16> -> tensor<1x64x64x128xf16>
    // CHECK:       [[CONST_RESHAPE:%.+]] = IE.AffineReshape([[FQ_INPUT]]) {
    // CHECK-SAME:       shape_value = [1, 64, 1, 1]} : tensor<1x1x1x64xf16> -> tensor<1x64x1x1xf16>

    // CHECK:       [[MUL:%.+]] = IE.Multiply([[ACT_TRANSPOSE]], [[CONST_RESHAPE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x64x64x128xf16>, tensor<1x64x1x1xf16>
    // CHECK-SAME:          -> tensor<1x64x64x128xf16>

    // CHECK:       [[OUTPUT:%.+]] = IE.Transpose([[MUL]]) {order_value = #NHWC} : tensor<1x64x64x128xf16> -> tensor<1x64x128x64xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x64x128x64xf16>

}

// -----

// CHECK-LABEL: @TransposeSubtract
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x64x128x64xf16>
func.func @TransposeSubtract(%arg0: tensor<1x64x128x64xf16>) -> tensor<1x64x128x64xf16> {
    %cst = const.Declare tensor<1x1x1x64xf16> = dense<1.0> : tensor<1x1x1x64xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %0 = IE.FakeQuantize(%cst, %cst_0, %cst_0, %cst_0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64}
        : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
                -> tensor<1x1x1x64xf16>
    %1 = IE.Subtract(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x128x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x64x128x64xf16>
    return %1 : tensor<1x64x128x64xf16>

    // CHECK-DAG:   [[CONST_INPUT:%.+]] = const.Declare tensor<1x1x1x64xf16> = dense<1.000000e+00> : tensor<1x1x1x64xf16>
    // CHECK-DAG:   [[CONST_FQ:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[FQ_INPUT:%.+]] = IE.FakeQuantize([[CONST_INPUT]], [[CONST_FQ]], [[CONST_FQ]], [[CONST_FQ]], [[CONST_FQ]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:      : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
    // CHECK-SAME:          -> tensor<1x1x1x64xf16>
    // CHECK:       [[ACT_TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x64x128x64xf16> -> tensor<1x64x64x128xf16>
    // CHECK:       [[CONST_RESHAPE:%.+]] = IE.AffineReshape([[FQ_INPUT]]) {
    // CHECK-SAME:       shape_value = [1, 64, 1, 1]} : tensor<1x1x1x64xf16> -> tensor<1x64x1x1xf16>

    // CHECK:       [[SUB:%.+]] = IE.Subtract([[ACT_TRANSPOSE]], [[CONST_RESHAPE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x64x64x128xf16>, tensor<1x64x1x1xf16>
    // CHECK-SAME:          -> tensor<1x64x64x128xf16>

    // CHECK:       [[OUTPUT:%.+]] = IE.Transpose([[SUB]]) {order_value = #NHWC} : tensor<1x64x64x128xf16> -> tensor<1x64x128x64xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x64x128x64xf16>

}

// -----

// CHECK-LABEL: @TransposeAdd
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x64x128x64xf16>
func.func @TransposeAdd(%arg0: tensor<1x64x128x64xf16>) -> tensor<1x64x128x64xf16> {
    %cst = const.Declare tensor<1x1x1x64xf16> = dense<1.0> : tensor<1x1x1x64xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %0 = IE.FakeQuantize(%cst, %cst_0, %cst_0, %cst_0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64}
        : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
                -> tensor<1x1x1x64xf16>
    %1 = IE.Add(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x128x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x64x128x64xf16>
    return %1 : tensor<1x64x128x64xf16>

    // CHECK-DAG:   [[CONST_INPUT:%.+]] = const.Declare tensor<1x1x1x64xf16> = dense<1.000000e+00> : tensor<1x1x1x64xf16>
    // CHECK-DAG:   [[CONST_FQ:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[FQ_INPUT:%.+]] = IE.FakeQuantize([[CONST_INPUT]], [[CONST_FQ]], [[CONST_FQ]], [[CONST_FQ]], [[CONST_FQ]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:      : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
    // CHECK-SAME:          -> tensor<1x1x1x64xf16>
    // CHECK:       [[ACT_TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x64x128x64xf16> -> tensor<1x64x64x128xf16>
    // CHECK:       [[CONST_RESHAPE:%.+]] = IE.AffineReshape([[FQ_INPUT]]) {
    // CHECK-SAME:       shape_value = [1, 64, 1, 1]} : tensor<1x1x1x64xf16> -> tensor<1x64x1x1xf16>

    // CHECK:       [[ADD:%.+]] = IE.Add([[ACT_TRANSPOSE]], [[CONST_RESHAPE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x64x64x128xf16>, tensor<1x64x1x1xf16>
    // CHECK-SAME:          -> tensor<1x64x64x128xf16>

    // CHECK:       [[OUTPUT:%.+]] = IE.Transpose([[ADD]]) {order_value = #NHWC} : tensor<1x64x64x128xf16> -> tensor<1x64x128x64xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x64x128x64xf16>

}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @Convert3dMulWithAdd
func.func @Convert3dMulWithAdd(%arg0: tensor<325x1x768xf16>) -> tensor<325x1x768xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x1x768xf16> = dense<1.100000e+00> : tensor<1x1x768xf16>
    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<325x1x768xf16>, tensor<1x1x768xf16> -> tensor<325x1x768xf16>

    %ADD_WEIGHTS = const.Declare tensor<1x1x768xf16> = dense<1.200000e+00> : tensor<1x1x768xf16>
    %ADD = IE.Add(%MUL, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<325x1x768xf16>, tensor<1x1x768xf16> -> tensor<325x1x768xf16>

    return %ADD : tensor<325x1x768xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.+]] = const.Declare tensor<1x768x1x1xf16> = dense<1.200200e+00>
    // CHECK-SAME:  : tensor<1x1x768xf16>, [#const.Reshape<[1, 768, 1, 1]>]

    // CHECK-DAG:   [[MUL_WEIGHTS:%.+]] = const.Declare tensor<1x768x1x1xf16> = dense<1.099610e+00>
    // CHECK-SAME:  : tensor<1x1x768xf16>, [#const.Reshape<[1, 768, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 325, 768, 1]
    // CHECK-SAME:  } : tensor<325x1x768xf16> -> tensor<1x325x768x1xf16>

    // CHECK:   [[TRANSPOSE_INPUT:%.+]] = IE.Transpose([[RESHAPE_INPUT]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } : tensor<1x325x768x1xf16> -> tensor<1x768x325x1xf16>

    // CHECK:   [[MUL:%.+]] = IE.Multiply([[TRANSPOSE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x768x325x1xf16>, tensor<1x768x1x1xf16> -> tensor<1x768x325x1xf16>

    // CHECK:   [[ADD:%.+]] = IE.Add([[MUL]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x768x325x1xf16>, tensor<1x768x1x1xf16> -> tensor<1x768x325x1xf16>

    // CHECK:   [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[ADD]]) {
    // CHECK-SAME:      order_value = #NHCW
    // CHECK-SAME:  } :  tensor<1x768x325x1xf16> -> tensor<1x325x768x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.+]] = IE.Reshape([[TRANSPOSE_OUT]]) {
    // CHECK-SAME:      shape_value = [325, 1, 768]
    // CHECK-SAME:  } : tensor<1x325x768x1xf16> -> tensor<325x1x768xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<325x1x768xf16>
}
