//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-identity-pools %s | FileCheck %s
// REQUIRES: arch-NPU50XX

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @FuseAddIdentityAvgPool([[ARG0:%.+]]: tensor<1x16x320x320x!qElemType, {order = #NHWC}>) -> tensor<1x16x320x320xf16, {order = #NHWC}> {
func.func @FuseAddIdentityAvgPool(%arg0 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>) -> tensor<1x16x320x320xf16, {order = #NHWC}> {
    %add = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320x!qElemType, {order = #NHWC}>, tensor<1x16x320x320x!qElemType, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>

    %ave_pool = IE.AvgPool(%add) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}  :
                  tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>

    return %ave_pool : tensor<1x16x320x320xf16, {order = #NHWC}>
    // CHECK:       IE.Add([[ARG0]], [[ARG0]])
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>
    // CHECK-NOT:       IE.AvgPool

}
