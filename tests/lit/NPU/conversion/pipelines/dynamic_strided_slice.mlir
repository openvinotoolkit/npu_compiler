//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=NPU37XX --import-IE ./slice.xml | FileCheck %s

// CHECK:   IE.CNNNetwork entryPoint : @main inputsInfo : {
// CHECK:       DataInfo "param_node_0" tensorNames = ["param_node_0"] : tensor<?xf32
// CHECK:   } outputsInfo : {
// CHECK:       DataInfo "StridedSlice_5" friendlyName = "Result_6" : tensor<?xf32
// CHECK:   }
// CHECK:   func.func @main([[ARG:%.+]]: tensor<?xf32, {bounds = [32], order = #C}>)
// CHECK-SAME:      -> tensor<?xf32, {bounds = [19], order = #C}> {
// CHECK:       [[BEGINS:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
// CHECK:       [[ENDS:%.+]] = const.Declare tensor<1xsi64> = dense<20> : tensor<1xsi64>
// CHECK:       [[STRIDES:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
// CHECK:       [[SLICE:%.+]] = IE.StridedSlice([[ARG]], [[BEGINS]], [[ENDS]], [[STRIDES]]) {
// CHECK-SAME:      begin_mask = [0],
// CHECK-SAME:      ellipsis_mask = [],
// CHECK-SAME:      end_mask = [0],
// CHECK-SAME:      new_axis_mask = [],
// CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 1, 1>,
// CHECK-SAME:      shrink_axis_mask = [0]
// CHECK-SAME:  } : tensor<?xf32, {bounds = [32], order = #C}>, tensor<1xsi64>, tensor<1xsi64>,
// CHECK-SAME:      tensor<1xsi64> -> tensor<?xf32, {bounds = [19], order = #C}>
// CHECK:       return [[SLICE]] : tensor<?xf32, {bounds = [19], order = #C}>
// CHECK:   }
