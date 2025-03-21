//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=NPU37XX --import-IE ./dynamic_reshape.xml | FileCheck %s

// CHECK: module @dynamic_reshape {
// CHECK:   IE.CNNNetwork entryPoint : @main inputsInfo : {
// CHECK:       DataInfo "Parameter_1" : tensor<1x8x?x?xf32
// CHECK:   } outputsInfo : {
// CHECK:       DataInfo "Reshape_5" friendlyName = "Result_12" : tensor<1x8x?x?x1xf32
// CHECK:   }
// CHECK:   func.func @main([[ARG:%.*]]: tensor<1x8x?x?xf32, {bounds = [1, 8, 48, 48], order = #NCHW}>)
// CHECK-SAME:      -> tensor<1x8x?x?x1xf32, {bounds = [1, 8, 48, 48, 1], order = #NCDHW}> {
// CHECK:       [[CST:%.+]] = const.Declare tensor<5xsi64> = dense<[1, 8, 0, 0, 1]> : tensor<5xsi64>
// CHECK:       [[RESHAPE:%.+]] = IE.DynamicReshape([[ARG]], [[CST]]) {
// CHECK-SAME:      output_bounds = [1, 8, 48, 48, 1]
// CHECK-SAME:      output_shape = [1, 8, -9223372036854775808, -9223372036854775808, 1]
// CHECK-SAME:  } : tensor<1x8x?x?xf32, {bounds = [1, 8, 48, 48], order = #NCHW}>, tensor<5xsi64>
// CHECK-SAME:      -> tensor<1x8x?x?x1xf32, {bounds = [1, 8, 48, 48, 1], order = #NCDHW}>
// CHECK:       return [[RESHAPE]] : tensor<1x8x?x?x1xf32, {bounds = [1, 8, 48, 48, 1], order = #NCDHW}>
// CHECK:   }
// CHECK: }
