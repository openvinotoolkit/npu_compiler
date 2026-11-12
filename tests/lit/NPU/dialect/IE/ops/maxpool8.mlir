//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --canonicalize --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// -----

// CHECK-LABEL: @MaxPool8NormalizeAxisToPositive
module @MaxPool8NormalizeAxisToPositive {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x100x17x17xf32>)
func.func @main(%arg0: tensor<1x100x17x17xf32>) -> (tensor<1x100x8x8xf32>, tensor<1x100x8x8xsi64>) {
  %output, %output_index = IE.MaxPool8(%arg0) {axis = -1 : i64, dilations = [1, 1], index_element_type = si64, kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x100x17x17xf32> -> tensor<1x100x8x8xf32>, tensor<1x100x8x8xsi64>
  return %output, %output_index : tensor<1x100x8x8xf32>, tensor<1x100x8x8xsi64>
  }

  // CHECK:        [[MAX_POOL_8:%.+]], [[MAX_POOL_8_INDEX:%.+]] = IE.MaxPool8([[ARG0]]) {
  // CHECK-SAME:       axis = 3 : i64,
  // CHECK-SAME:       dilations = [1, 1],
  // CHECK-SAME:       index_element_type = si64,
  // CHECK-SAME:       kernel_size = [3, 3],
  // CHECK-SAME:       pads_begin = [0, 0],
  // CHECK-SAME:       pads_end = [0, 0],
  // CHECK-SAME:       rounding_type = #IE.rounding_type<FLOOR>,
  // CHECK-SAME:       strides = [2, 2]
  // CHECK-SAME:   } :
  // CHECK:        tensor<1x100x17x17xf32> -> tensor<1x100x8x8xf32>, tensor<1x100x8x8xsi64>
  // CHECK:        return [[MAX_POOL_8]], [[MAX_POOL_8_INDEX]] : tensor<1x100x8x8xf32>, tensor<1x100x8x8xsi64>

}

// -----

// CHECK-LABEL: @MaxPool8RemoveIdenticalOp
module @MaxPool8RemoveIdenticalOp {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x6x2x2xf32>)
func.func @main(%arg0: tensor<1x6x2x2xf32>) -> (tensor<1x6x2x2xf32>, tensor<1x6x2x2xsi64>) {
  %output, %output_index = IE.MaxPool8(%arg0) {axis = 0 : i64, dilations = [1, 1], index_element_type = si64, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x6x2x2xf32> -> tensor<1x6x2x2xf32>, tensor<1x6x2x2xsi64>
  return %output, %output_index : tensor<1x6x2x2xf32>, tensor<1x6x2x2xsi64>
  }

  // CHECK:        [[CST_INDEX:%.+]] = const.Declare tensor<1x6x2x2xsi64>
  // CHECK-SAME:                     = dense<[
  // CHECK-SAME{LITERAL}:            [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]], [[12, 13], [14, 15]], [[16, 17], [18, 19]], [[20, 21], [22, 23]]]
  // CHECK-SAME:                     ]> : tensor<1x6x2x2xsi64>
  // CHECK:        return [[ARG0]], [[CST_INDEX]] : tensor<1x6x2x2xf32>, tensor<1x6x2x2xsi64>

}

// -----

// CHECK-LABEL: @MaxPool8RemoveIdenticalOp2
module @MaxPool8RemoveIdenticalOp2 {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x6x2x2xf32>)
func.func @main(%arg0: tensor<1x6x2x2xf32>) -> (tensor<1x6x2x2xf32>, tensor<1x6x2x2xsi64>) {
  %output, %output_index = IE.MaxPool8(%arg0) {axis = 2 : i64, dilations = [1, 1], index_element_type = si64, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x6x2x2xf32> -> tensor<1x6x2x2xf32>, tensor<1x6x2x2xsi64>
  return %output, %output_index : tensor<1x6x2x2xf32>, tensor<1x6x2x2xsi64>
  }

  // CHECK:        [[CST_INDEX:%.+]] = const.Declare tensor<1x6x2x2xsi64>
  // CHECK-SAME:                     = dense<[
  // CHECK-SAME{LITERAL}:            [[[0, 1], [2, 3]], [[0, 1], [2, 3]], [[0, 1], [2, 3]], [[0, 1], [2, 3]], [[0, 1], [2, 3]], [[0, 1], [2, 3]]]
  // CHECK-SAME:                     ]> : tensor<1x6x2x2xsi64>
  // CHECK:        return [[ARG0]], [[CST_INDEX]] : tensor<1x6x2x2xf32>, tensor<1x6x2x2xsi64>

}
