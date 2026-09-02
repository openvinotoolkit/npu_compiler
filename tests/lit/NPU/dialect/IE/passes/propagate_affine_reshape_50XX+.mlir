//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --propagate-affine-reshape %s | FileCheck %s
// REQUIRES: platform-NPU5010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotSwapAffineReshapeMultiplyAlignedWithHWConsumer
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x1xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<1x16x16x1xf16, {order = #NHWC}>)
func.func @DoNotSwapAffineReshapeMultiplyAlignedWithHWConsumer(
        %arg0: tensor<1x16x16x1xf16, {order = #NHWC}>,
        %arg1: tensor<1x16x16x1xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 16, 4, 4]} : tensor<1x16x16x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 16, 4, 4]} : tensor<1x16x16x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x4x4xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x4x4xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    return %3 : tensor<1x16x4x4xf16, {order = #NHWC}>

    // Multiply output 1x16x4x4 NHWC: C=16 (divisible by channel alignment 16),
    // H=4 and W=4 (both divisible by VPU_SPATIAL_ALIGNMENT=4). The IE.Add consumer
    // is NCE-compatible and aligned → runsOnHWAlignedWithConsumer returns true →
    // propagation is blocked and the IR is left unchanged.

    // CHECK:       [[R1:%.+]] = IE.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 16, 4, 4]}
    // CHECK-SAME:             tensor<1x16x16x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    // CHECK:       [[R2:%.+]] = IE.AffineReshape([[ARG_1]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 16, 4, 4]}
    // CHECK-SAME:             tensor<1x16x16x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[R1]], [[R2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x4x4xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[MUL]], [[MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x4x4xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    // CHECK:       return [[ADD]] : tensor<1x16x4x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapAffineReshapeMultiplyWhenSpatialNotAligned
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x12x1xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<1x16x12x1xf16, {order = #NHWC}>)
func.func @SwapAffineReshapeMultiplyWhenSpatialNotAligned(
        %arg0: tensor<1x16x12x1xf16, {order = #NHWC}>,
        %arg1: tensor<1x16x12x1xf16, {order = #NHWC}>) -> tensor<1x16x4x3xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 16, 4, 3]} : tensor<1x16x12x1xf16, {order = #NHWC}> -> tensor<1x16x4x3xf16, {order = #NHWC}>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 16, 4, 3]} : tensor<1x16x12x1xf16, {order = #NHWC}> -> tensor<1x16x4x3xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4x3xf16, {order = #NHWC}>, tensor<1x16x4x3xf16, {order = #NHWC}> -> tensor<1x16x4x3xf16, {order = #NHWC}>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4x3xf16, {order = #NHWC}>, tensor<1x16x4x3xf16, {order = #NHWC}> -> tensor<1x16x4x3xf16, {order = #NHWC}>
    return %3 : tensor<1x16x4x3xf16, {order = #NHWC}>

    // CHECK:       [[MUL:%.+]] = IE.Multiply([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x12x1xf16, {order = #NHWC}>, tensor<1x16x12x1xf16, {order = #NHWC}> -> tensor<1x16x12x1xf16, {order = #NHWC}>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[MUL]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 16, 4, 3]}
    // CHECK-SAME:             tensor<1x16x12x1xf16, {order = #NHWC}> -> tensor<1x16x4x3xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[RESHAPE]], [[RESHAPE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4x3xf16, {order = #NHWC}>, tensor<1x16x4x3xf16, {order = #NHWC}> -> tensor<1x16x4x3xf16, {order = #NHWC}>
    // CHECK:       return [[ADD]] : tensor<1x16x4x3xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapAffineReshapeMultiplyAlignedWithSWConsumer
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x1xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<1x16x16x1xf16, {order = #NHWC}>)
func.func @SwapAffineReshapeMultiplyAlignedWithSWConsumer(
        %arg0: tensor<1x16x16x1xf16, {order = #NHWC}>,
        %arg1: tensor<1x16x16x1xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 16, 4, 4]} : tensor<1x16x16x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 16, 4, 4]} : tensor<1x16x16x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x4x4xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    %3 = IE.SoftMax(%2) {axisInd = 1 : i64} : tensor<1x16x4x4xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    return %3 : tensor<1x16x4x4xf16, {order = #NHWC}>

    // CHECK:       [[MUL:%.+]] = IE.Multiply([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x1xf16, {order = #NHWC}>, tensor<1x16x16x1xf16, {order = #NHWC}> -> tensor<1x16x16x1xf16, {order = #NHWC}>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[MUL]]) {axisInd = 1 : i64} : tensor<1x16x16x1xf16, {order = #NHWC}> -> tensor<1x16x16x1xf16, {order = #NHWC}>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 16, 4, 4]}
    // CHECK-SAME:             tensor<1x16x16x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    // CHECK:       return [[RESHAPE]] : tensor<1x16x4x4xf16, {order = #NHWC}>
}
