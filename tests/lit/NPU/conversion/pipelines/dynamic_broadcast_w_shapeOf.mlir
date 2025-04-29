//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=NPU37XX --import-IE ./dynamic_broadcast_with_shapeOf.xml | FileCheck %s

// CHECK: #C = affine_map<(d0) -> (d0)>
// CHECK: #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK:   IE.CNNNetwork entryPoint : @main inputsInfo : {
// CHECK:     DataInfo "input_0" : tensor<1x3x?x?xf32
// CHECK:     DataInfo "input_1" : tensor<?xf32
// CHECK:   } outputsInfo : {
// CHECK:     DataInfo "Broadcast_60" friendlyName = "Result_64" : tensor<1x3x?x?xf32
// CHECK:   }
// CHECK:   func.func @main(%arg0: tensor<1x3x?x?xf32, {bounds = [1, 3, 10, 16], order = #NCHW}>, %arg1: tensor<?xf32, {bounds = [10], order = #C}>) -> tensor<1x3x?x?xf32, {bounds = [1, 3, 10, 16], order = #NCHW}> {
// CHECK:   [[SHAPEOF:%.+]] = IE.ShapeOf(%arg0) {dstElemType = si64} : tensor<1x3x?x?xf32, {bounds = [1, 3, 10, 16], order = #NCHW}> -> tensor<4xsi64>
// CHECK:   [[BROADCAST:%.+]] = IE.DynamicBroadcast(%arg1, [[SHAPEOF]]) {mode = #IE.broadcast_type<NUMPY>, output_bounds = [1, 3, 10, 16], output_shape = [1, 3, -9223372036854775808, -9223372036854775808]} : tensor<?xf32, {bounds = [10], order = #C}>, tensor<4xsi64> -> tensor<1x3x?x?xf32, {bounds = [1, 3, 10, 16], order = #NCHW}>
// CHECK:   return [[BROADCAST]] : tensor<1x3x?x?xf32, {bounds = [1, 3, 10, 16], order = #NCHW}>
// CHECK:   }
