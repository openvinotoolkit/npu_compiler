//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --canonicalize --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

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
