//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --decompose-istft %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @DecomposeISTFT
func.func @DecomposeISTFT(%arg0: tensor<2x3x2xf16>) -> tensor<4xf16> {
    %cst = const.Declare tensor<2xf16> =
        dense<1.0> : tensor<2xf16>
    %cst_0 = const.Declare tensor<1xsi32> =
        dense<2> : tensor<si64>,
          [
            #const.Reshape<[1]>
          ]
    %cst_1 = const.Declare tensor<1xsi32> =
        dense<4> : tensor<si64>,
          [
            #const.Reshape<[1]>
          ]
    %cst_2 = const.Declare tensor<1xsi32> =
        dense<4> : tensor<si64>,
          [
            #const.Reshape<[1]>
          ]
    %0 = IE.ISTFT(%arg0, %cst, %cst_0, %cst_1, %cst_2) : tensor<2x3x2xf16>, tensor<2xf16>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32> -> tensor<4xf16>
    return %0 : tensor<4xf16>


    // CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x2xf16>) -> tensor<4xf16>
    // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<2xf16> = dense<1.000000e+00> : tensor<2xf16>
    // CHECK-DAG:  [[CST_0:%.+]] = const.Declare tensor<1xsi32> = dense<2> : tensor<si64>, [#const.Reshape<[1]>]
    // CHECK-DAG:  [[CST_1:%.+]] = const.Declare tensor<1xsi32> = dense<4> : tensor<si64>, [#const.Reshape<[1]>]
    // CHECK-DAG:  [[CST_2:%.+]] = const.Declare tensor<1xsi32> = dense<4> : tensor<si64>, [#const.Reshape<[1]>]

    // CHECK:  [[TRANSPOSE:%.+]] = IE.Transpose([[ARG0]]) {order_value = #HCW} : tensor<2x3x2xf16> -> tensor<3x2x2xf16>
    // CHECK:  [[IRDFT:%.+]] = IE.IRDFT([[TRANSPOSE]]) {axes_attr = [1], operandSegmentSizes = array<i32: 1, 0, 0>, signal_size_attr = [2]} : tensor<3x2x2xf16> -> tensor<3x2xf16>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[IRDFT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x2xf16>, tensor<2xf16> -> tensor<3x2xf16>

    // CHECK:  [[RESHAPE:%.+]] = IE.Reshape([[MULTIPLY]]) {shape_value = [1, 3, 2]} : tensor<3x2xf16> -> tensor<1x3x2xf16>
    // CHECK:  [[SLICE:%.+]] = IE.Slice [[RESHAPE]] [0, 0, 0] [1, 3, 2] : tensor<1x3x2xf16> to tensor<1x3x2xf16>
    // CHECK:  [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE]]) {shape_value = [3, 2]} : tensor<1x3x2xf16> -> tensor<3x2xf16>
    // CHECK-DAG:  [[CST_3:%.+]] = const.Declare tensor<10xf16> = dense<0.000000e+00> : tensor<10xf16>
    // CHECK:  [[SLICE_0:%.+]] = IE.Slice [[RESHAPE_1]] [0, 0] [1, 2] : tensor<3x2xf16> to tensor<1x2xf16>
    // CHECK:  [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [2]} : tensor<1x2xf16> -> tensor<2xf16>
    // CHECK:  [[PAD_0:%.+]] = IE.Pad([[RESHAPE_2]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0], pads_end_attr = [8]} : tensor<2xf16> -> tensor<10xf16>
    // CHECK:  [[ADD_0:%.+]] = IE.Add([[CST_3]], [[PAD_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10xf16>, tensor<10xf16> -> tensor<10xf16>
    // CHECK:  [[SLICE_1:%.+]] = IE.Slice [[RESHAPE_1]] [1, 0] [1, 2] : tensor<3x2xf16> to tensor<1x2xf16>
    // CHECK:  [[RESHAPE_3:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [2]} : tensor<1x2xf16> -> tensor<2xf16>
    // CHECK:  [[PAD_1:%.+]] = IE.Pad([[RESHAPE_3]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [4], pads_end_attr = [4]} : tensor<2xf16> -> tensor<10xf16>
    // CHECK:  [[ADD_1:%.+]] = IE.Add([[ADD_0]], [[PAD_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10xf16>, tensor<10xf16> -> tensor<10xf16>
    // CHECK:  [[SLICE_2:%.+]] = IE.Slice [[RESHAPE_1]] [2, 0] [1, 2] : tensor<3x2xf16> to tensor<1x2xf16>
    // CHECK:      [[RESHAPE_4:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [2]} : tensor<1x2xf16> -> tensor<2xf16>
    // CHECK:  [[PAD_2:%.+]] = IE.Pad([[RESHAPE_4]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [8], pads_end_attr = [0]} : tensor<2xf16> -> tensor<10xf16>
    // CHECK:  [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[PAD_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10xf16>, tensor<10xf16> -> tensor<10xf16>
    // CHECK:  [[RESHAPE_5:%.+]] = IE.Reshape([[ADD_2]]) {shape_value = [1, 10]} : tensor<10xf16> -> tensor<1x10xf16>
    // CHECK:  [[RESHAPE_6:%.+]] = IE.Reshape([[RESHAPE_5]]) {shape_value = [10]} : tensor<1x10xf16> -> tensor<10xf16>
    // CHECK-DAG:  [[CST_4:%.+]] = const.Declare tensor<10xf16> = dense<[1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<10xf16>
    // CHECK:  [[DIVIDE:%.+]] = IE.Divide([[RESHAPE_6]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10xf16>, tensor<10xf16> -> tensor<10xf16>
    // CHECK:  [[SLICE_3:%.+]] = IE.Slice [[DIVIDE]] [0] [4] : tensor<10xf16> to tensor<4xf16
    // CHECK:  return [[SLICE_3]] : tensor<4xf16>

}
