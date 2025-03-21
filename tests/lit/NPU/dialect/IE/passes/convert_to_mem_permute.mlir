//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-mem-permute %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvertToMemPermute
func.func @ConvertToMemPermute(%arg0: tensor<1x2x3x4xf32>,
                            %arg1: tensor<1x2x3x4xf32, {order = #NHWC}>,
                            %arg2: tensor<1x2x3x4xf32>) ->
                        (tensor<1x4x2x3xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32, {order = #NHWC}>) {
    %0 = IE.Transpose(%arg0) {order_value = #NWCH} : tensor<1x2x3x4xf32> -> tensor<1x4x2x3xf32>

    %1 = IE.Reorder(%arg1) {dstOrder = #NCHW} : tensor<1x2x3x4xf32, {order = #NHWC}> -> tensor<1x2x3x4xf32>

    %2 = IE.Reorder(%arg2) {dstOrder = #NHWC} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32, {order = #NHWC}>
    return %0, %1, %2 : tensor<1x4x2x3xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32, {order = #NHWC}>

    // CHECK-NOT: IE.Transpose
    // CHECK-NOT: IE.Reorder
    // CHECK-NOT: IE.Reorder
    // CHECK:     %[[VAL0:.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x3x4xf32> -> tensor<1x4x2x3xf32>
    // CHECK:     %[[VAL1:.*]] = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x3x4xf32, {order = #NHWC}> -> tensor<1x2x3x4xf32>
    // CHECK:     %[[VAL2:.*]] = IE.MemPermute(%arg2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32, {order = #NHWC}>
    // CHECK:     return %[[VAL0]], %[[VAL1:.*]], %[[VAL2:.*]] : tensor<1x4x2x3xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32, {order = #NHWC}>
}
