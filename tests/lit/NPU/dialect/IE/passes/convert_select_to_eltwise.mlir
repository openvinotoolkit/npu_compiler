//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-select-to-eltwise %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// -----

// CHECK-LABEL: @FuseSelectLogicalNotBroadcastToMultiply
// CHECK-SAME:  [[DATA:%.+]]: tensor<1x64x5x1xf16>
func.func @FuseSelectLogicalNotBroadcastToMultiply(%arg0: tensor<1x64x5x1xf16>) -> tensor<1x64x5x5xf16> {
    %cst_bool = const.Declare tensor<1x1x5x5xf16> = dense<[[[[0.0, 1.0, 0.0, 1.0, 1.0],
                                                              [0.0, 0.0, 1.0, 1.0, 0.0],
                                                              [1.0, 0.0, 0.0, 1.0, 1.0],
                                                              [0.0, 1.0, 1.0, 0.0, 0.0],
                                                              [1.0, 1.0, 0.0, 0.0, 1.0]]]]> : tensor<1x1x5x5xf16>
    %cst_shape = const.Declare tensor<4xsi32> = dense<[1, 64, 5, 5]> : tensor<4xsi32>
    %cst_zero  = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>

    %not = IE.LogicalNot(%cst_bool) : tensor<1x1x5x5xf16> -> tensor<1x1x5x5xf16>
    %bc  = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} :
               tensor<1x64x5x1xf16>, tensor<4xsi32> -> tensor<1x64x5x5xf16>
    %sel = IE.Select(%not, %cst_zero, %bc) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
               tensor<1x1x5x5xf16>, tensor<1x1x1x1xf16>, tensor<1x64x5x5xf16> -> tensor<1x64x5x5xf16>
    return %sel : tensor<1x64x5x5xf16>

    // CHECK-NOT: IE.LogicalNot
    // CHECK-NOT: IE.Broadcast
    // CHECK-NOT: IE.Select
    // CHECK-DAG: [[MASK:%.+]] = const.Declare tensor<1x1x5x5xf16>
    // CHECK:     [[MUL:%.+]] = IE.Multiply([[DATA]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x64x5x1xf16>, tensor<1x1x5x5xf16> -> tensor<1x64x5x5xf16>
    // CHECK:     return [[MUL]] : tensor<1x64x5x5xf16>
}

// -----

// CHECK-LABEL: @FuseMamba256BatchMismatch
// CHECK-SAME:  [[DATA:%.+]]: tensor<64x4x4x1xf16>
func.func @FuseMamba256BatchMismatch(%arg0: tensor<64x4x4x1xf16>) -> tensor<64x4x4x4xf16> {
    %cst_bool  = const.Declare tensor<1x1x4x4xf16> = dense<[[[[0.0, 1.0, 0.0, 1.0],
                                                               [1.0, 0.0, 1.0, 0.0],
                                                               [0.0, 1.0, 0.0, 1.0],
                                                               [1.0, 0.0, 1.0, 0.0]]]]> : tensor<1x1x4x4xf16>
    %cst_shape = const.Declare tensor<4xsi32> = dense<[64, 4, 4, 4]> : tensor<4xsi32>
    %cst_zero  = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %not = IE.LogicalNot(%cst_bool) : tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
    %bc  = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} :
               tensor<64x4x4x1xf16>, tensor<4xsi32> -> tensor<64x4x4x4xf16>
    %sel = IE.Select(%not, %cst_zero, %bc) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
               tensor<1x1x4x4xf16>, tensor<1x1x1x1xf16>, tensor<64x4x4x4xf16> -> tensor<64x4x4x4xf16>
    return %sel : tensor<64x4x4x4xf16>

    // CHECK-NOT: IE.LogicalNot
    // CHECK-NOT: IE.Broadcast
    // CHECK-NOT: IE.Select
    // CHECK-DAG: [[MASK:%.+]] = const.Declare tensor<1x1x4x4xf16>
    // CHECK:     [[MUL:%.+]] = IE.Multiply([[DATA]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<64x4x4x1xf16>, tensor<1x1x4x4xf16> -> tensor<64x4x4x4xf16>
    // CHECK:     return [[MUL]] : tensor<64x4x4x4xf16>
}

// -----

// CHECK-LABEL: @FuseSelectConstDirectToMultiply
// CHECK-SAME:  [[DATA:%.+]]: tensor<1x64x5x1xf16>
func.func @FuseSelectConstDirectToMultiply(%arg0: tensor<1x64x5x1xf16>) -> tensor<1x64x5x5xf16> {
    %cst_mask  = const.Declare tensor<1x1x5x5xf16> = dense<[[[[0.0, 1.0, 0.0, 1.0, 1.0],
                                                               [0.0, 0.0, 1.0, 1.0, 0.0],
                                                               [1.0, 0.0, 0.0, 1.0, 1.0],
                                                               [0.0, 1.0, 1.0, 0.0, 0.0],
                                                               [1.0, 1.0, 0.0, 0.0, 1.0]]]]> : tensor<1x1x5x5xf16>
    %cst_shape = const.Declare tensor<4xsi32> = dense<[1, 64, 5, 5]> : tensor<4xsi32>
    %cst_zero  = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %bc  = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} :
               tensor<1x64x5x1xf16>, tensor<4xsi32> -> tensor<1x64x5x5xf16>
    %sel = IE.Select(%cst_mask, %cst_zero, %bc) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
               tensor<1x1x5x5xf16>, tensor<1x1x1x1xf16>, tensor<1x64x5x5xf16> -> tensor<1x64x5x5xf16>
    return %sel : tensor<1x64x5x5xf16>

    // CHECK-NOT: IE.Broadcast
    // CHECK-NOT: IE.Select
    // CHECK-DAG: [[MASK:%.+]] = const.Declare tensor<1x1x5x5xf16>
    // CHECK:     [[MUL:%.+]] = IE.Multiply([[DATA]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x64x5x1xf16>, tensor<1x1x5x5xf16> -> tensor<1x64x5x5xf16>
    // CHECK:     return [[MUL]] : tensor<1x64x5x5xf16>
}

// -----

// CHECK-LABEL: @NoFuseInput3NotBroadcast
// CHECK-SAME:  [[DATA:%.+]]: tensor<1x64x5x5xf16>
func.func @NoFuseInput3NotBroadcast(%arg0: tensor<1x64x5x5xf16>) -> tensor<1x64x5x5xf16> {
    %cst_bool = const.Declare tensor<1x1x5x5xf16> = dense<1.0> : tensor<1x1x5x5xf16>
    %cst_zero = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %not = IE.LogicalNot(%cst_bool) : tensor<1x1x5x5xf16> -> tensor<1x1x5x5xf16>
    %sel = IE.Select(%not, %cst_zero, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
               tensor<1x1x5x5xf16>, tensor<1x1x1x1xf16>, tensor<1x64x5x5xf16> -> tensor<1x64x5x5xf16>
    return %sel : tensor<1x64x5x5xf16>

    // CHECK-NOT: IE.Multiply
    // CHECK:     IE.Select
}

// -----

// CHECK-LABEL: @NoFuseTrueBranchNonZero
func.func @NoFuseTrueBranchNonZero(%arg0: tensor<1x64x5x1xf16>) -> tensor<1x64x5x5xf16> {
    %cst_bool  = const.Declare tensor<1x1x5x5xf16> = dense<1.0> : tensor<1x1x5x5xf16>
    %cst_shape = const.Declare tensor<4xsi32> = dense<[1, 64, 5, 5]> : tensor<4xsi32>
    %cst_fill  = const.Declare tensor<1x1x1x1xf16> = dense<-6.55e+04> : tensor<1x1x1x1xf16>
    %not = IE.LogicalNot(%cst_bool) : tensor<1x1x5x5xf16> -> tensor<1x1x5x5xf16>
    %bc  = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} :
               tensor<1x64x5x1xf16>, tensor<4xsi32> -> tensor<1x64x5x5xf16>
    %sel = IE.Select(%not, %cst_fill, %bc) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
               tensor<1x1x5x5xf16>, tensor<1x1x1x1xf16>, tensor<1x64x5x5xf16> -> tensor<1x64x5x5xf16>
    return %sel : tensor<1x64x5x5xf16>

    // CHECK-NOT: IE.Multiply
    // CHECK:     IE.Select
}

// -----

// CHECK-LABEL: @NoFuseDynamicMask
// CHECK-SAME:  [[MASK:%.+]]: tensor<1x1x5x5xf16>
// CHECK-SAME:  [[DATA:%.+]]: tensor<1x64x5x1xf16>
func.func @NoFuseDynamicMask(%mask: tensor<1x1x5x5xf16>, %arg0: tensor<1x64x5x1xf16>) -> tensor<1x64x5x5xf16> {
    %cst_shape = const.Declare tensor<4xsi32> = dense<[1, 64, 5, 5]> : tensor<4xsi32>
    %cst_zero  = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %not = IE.LogicalNot(%mask) : tensor<1x1x5x5xf16> -> tensor<1x1x5x5xf16>
    %bc  = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} :
               tensor<1x64x5x1xf16>, tensor<4xsi32> -> tensor<1x64x5x5xf16>
    %sel = IE.Select(%not, %cst_zero, %bc) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
               tensor<1x1x5x5xf16>, tensor<1x1x1x1xf16>, tensor<1x64x5x5xf16> -> tensor<1x64x5x5xf16>
    return %sel : tensor<1x64x5x5xf16>

    // CHECK-NOT: IE.Multiply
    // CHECK:     IE.LogicalNot
    // CHECK:     IE.Select
}

// -----

// CHECK-LABEL: @NoFuseTileBroadcastNonOneDim
// CHECK-SAME:  [[DATA:%.+]]: tensor<1x4x5x2xf16>
func.func @NoFuseTileBroadcastNonOneDim(%arg0: tensor<1x4x5x2xf16>) -> tensor<1x4x5x4xf16> {
    // Broadcast tiles dim=3 from 2 -> 4: not a 1->N expansion, so transformation is unsafe.
    %cst_bool  = const.Declare tensor<1x1x5x4xf16> = dense<1.0> : tensor<1x1x5x4xf16>
    %cst_shape = const.Declare tensor<4xsi32> = dense<[1, 4, 5, 4]> : tensor<4xsi32>
    %cst_zero  = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %not = IE.LogicalNot(%cst_bool) : tensor<1x1x5x4xf16> -> tensor<1x1x5x4xf16>
    %bc  = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<NUMPY>} :
               tensor<1x4x5x2xf16>, tensor<4xsi32> -> tensor<1x4x5x4xf16>
    %sel = IE.Select(%not, %cst_zero, %bc) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
               tensor<1x1x5x4xf16>, tensor<1x1x1x1xf16>, tensor<1x4x5x4xf16> -> tensor<1x4x5x4xf16>
    return %sel : tensor<1x4x5x4xf16>

    // CHECK-NOT: IE.Multiply
    // CHECK:     IE.Broadcast
    // CHECK:     IE.Select
}

// -----

// CHECK-LABEL: @FuseSelectI8ConstDirectToMultiply
// CHECK-SAME:  [[DATA:%.+]]: tensor<1x64x5x1xf16>
func.func @FuseSelectI8ConstDirectToMultiply(%arg0: tensor<1x64x5x1xf16>) -> tensor<1x64x5x5xf16> {
    %cst_mask  = const.Declare tensor<1x1x5x5xi8> = dense<[[[[0, 1, 0, 1, 1],
                                                              [0, 0, 1, 1, 0],
                                                              [1, 0, 0, 1, 1],
                                                              [0, 1, 1, 0, 0],
                                                              [1, 1, 0, 0, 1]]]]> : tensor<1x1x5x5xi8>
    %cst_shape = const.Declare tensor<4xsi32> = dense<[1, 64, 5, 5]> : tensor<4xsi32>
    %cst_zero  = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %bc  = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} :
               tensor<1x64x5x1xf16>, tensor<4xsi32> -> tensor<1x64x5x5xf16>
    %sel = IE.Select(%cst_mask, %cst_zero, %bc) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
               tensor<1x1x5x5xi8>, tensor<1x1x1x1xf16>, tensor<1x64x5x5xf16> -> tensor<1x64x5x5xf16>
    return %sel : tensor<1x64x5x5xf16>

    // CHECK-NOT: IE.Broadcast
    // CHECK-NOT: IE.Select
    // CHECK-DAG: [[MASK:%.+]] = const.Declare tensor<1x1x5x5xf16>
    // CHECK:     [[MUL:%.+]] = IE.Multiply([[DATA]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x64x5x1xf16>, tensor<1x1x5x5xf16> -> tensor<1x64x5x5xf16>
    // CHECK:     return [[MUL]] : tensor<1x64x5x5xf16>
}

// -----

// CHECK-LABEL: @FuseSelectI8LogicalNotToMultiply
// CHECK-SAME:  [[DATA:%.+]]: tensor<1x64x5x1xf16>
func.func @FuseSelectI8LogicalNotToMultiply(%arg0: tensor<1x64x5x1xf16>) -> tensor<1x64x5x5xf16> {
    %cst_bool  = const.Declare tensor<1x1x5x5xi8> = dense<[[[[0, 1, 0, 1, 1],
                                                              [0, 0, 1, 1, 0],
                                                              [1, 0, 0, 1, 1],
                                                              [0, 1, 1, 0, 0],
                                                              [1, 1, 0, 0, 1]]]]> : tensor<1x1x5x5xi8>
    %cst_shape = const.Declare tensor<4xsi32> = dense<[1, 64, 5, 5]> : tensor<4xsi32>
    %cst_zero  = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %not = IE.LogicalNot(%cst_bool) : tensor<1x1x5x5xi8> -> tensor<1x1x5x5xi8>
    %bc  = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} :
               tensor<1x64x5x1xf16>, tensor<4xsi32> -> tensor<1x64x5x5xf16>
    %sel = IE.Select(%not, %cst_zero, %bc) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
               tensor<1x1x5x5xi8>, tensor<1x1x1x1xf16>, tensor<1x64x5x5xf16> -> tensor<1x64x5x5xf16>
    return %sel : tensor<1x64x5x5xf16>

    // CHECK-NOT: IE.LogicalNot
    // CHECK-NOT: IE.Broadcast
    // CHECK-NOT: IE.Select
    // CHECK-DAG: [[MASK:%.+]] = const.Declare tensor<1x1x5x5xf16>
    // CHECK:     [[MUL:%.+]] = IE.Multiply([[DATA]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x64x5x1xf16>, tensor<1x1x5x5xf16> -> tensor<1x64x5x5xf16>
    // CHECK:     return [[MUL]] : tensor<1x64x5x5xf16>
}
