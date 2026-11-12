//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @BroadcastFoldFold
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x8x4x4xf32>)
func.func @BroadcastFoldFold(%arg0 : tensor<1x8x4x4xf32>)-> tensor<1x8x4x4xf32> {
    %0 = const.Declare tensor<4xsi64> = dense<1> : tensor<4xsi64>
    %1 = IE.Broadcast(%arg0, %0) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x8x4x4xf32>, tensor<4xsi64> -> tensor<1x8x4x4xf32>
    return %1 : tensor<1x8x4x4xf32>

    // CHECK-NOT: IE.Broadcast
    // CHECK:     return [[ARG_0]]
}

// CHECK-LABEL: @ConstBroadcastFuseBidirectional
func.func @ConstBroadcastFuseBidirectional()-> tensor<1x8x4x4xf32> {
    %0 = const.Declare tensor<4x1xf32> = dense<1.0> : tensor<4x1xf32>
    %1 = const.Declare tensor<4xsi64> = dense<[1, 8, 4, 4]> : tensor<4xsi64>

    %2 = IE.Broadcast(%0, %1) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<4x1xf32>, tensor<4xsi64> -> tensor<1x8x4x4xf32>
    return %2 : tensor<1x8x4x4xf32>

    // CHECK-NOT: IE.Broadcast
    // CHECK:     [[CST:%.+]] = const.Declare tensor<1x8x4x4xf32> = dense<1.000000e+00> :
    // CHECK-SAME:          tensor<4x1xf32>, [#const.Reshape<[1, 1, 4, 1]>, #const.Broadcast<1 : i64, 8 : i64>, #const.Broadcast<3 : i64, 4 : i64>]
    // CHECK:     return [[CST]] : tensor<1x8x4x4xf32>
}

// CHECK-LABEL: @ConstBroadcastFuseNumpy
func.func @ConstBroadcastFuseNumpy()-> tensor<1x8x4x4xf32> {
    %0 = const.Declare tensor<4x1xf32> = dense<1.0> : tensor<4x1xf32>
    %1 = const.Declare tensor<4xsi64> = dense<[1, 8, 4, 4]> : tensor<4xsi64>

    %2 = IE.Broadcast(%0, %1) {mode = #IE.broadcast_type<NUMPY>} : tensor<4x1xf32>, tensor<4xsi64> -> tensor<1x8x4x4xf32>
    return %2 : tensor<1x8x4x4xf32>

    // CHECK-NOT: IE.Broadcast
    // CHECK:     [[CST:%.+]] = const.Declare tensor<1x8x4x4xf32> = dense<1.000000e+00> :
    // CHECK-SAME:          tensor<4x1xf32>, [#const.Reshape<[1, 1, 4, 1]>, #const.Broadcast<1 : i64, 8 : i64>, #const.Broadcast<3 : i64, 4 : i64>]
    // CHECK:     return [[CST]] : tensor<1x8x4x4xf32>
}

// CHECK-LABEL: @ConstBroadcastFuseExplicit
func.func @ConstBroadcastFuseExplicit()-> tensor<1x8x4x4xf32> {
    %0 = const.Declare tensor<4x1xf32> = dense<1.0> : tensor<4x1xf32>
    %1 = const.Declare tensor<4xsi64> = dense<[1, 8, 4, 4]> : tensor<4xsi64>
    %2 = const.Declare tensor<2xsi64> = dense<[1, 2]> : tensor<2xsi64>

    %3 = IE.Broadcast(%0, %1, %2) {mode = #IE.broadcast_type<EXPLICIT>} : tensor<4x1xf32>, tensor<4xsi64>, tensor<2xsi64> -> tensor<1x8x4x4xf32>
    return %3 : tensor<1x8x4x4xf32>

    // CHECK-NOT: IE.Broadcast
    // CHECK:     [[CST:%.+]] = const.Declare tensor<1x8x4x4xf32> = dense<1.000000e+00> :
    // CHECK-SAME:          tensor<4x1xf32>, [#const.Reshape<[1, 4, 1, 1]>, #const.Broadcast<1 : i64, 8 : i64>, #const.Broadcast<2 : i64, 4 : i64>, #const.Broadcast<3 : i64, 4 : i64>]
    // CHECK:     return [[CST]] : tensor<1x8x4x4xf32>
}
