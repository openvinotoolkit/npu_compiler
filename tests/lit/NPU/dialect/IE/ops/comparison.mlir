//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @LessBroadcastable
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<10x1xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x50xf16>
func.func @LessBroadcastable(%arg0: tensor<10x1xf16>, %arg1: tensor<1x50xf16>) -> tensor<10x50xi8> {
    %0 = IE.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10x1xf16>, tensor<1x50xf16> -> tensor<10x50xi8>
    return %0 : tensor<10x50xi8>

    // CHECK:       [[VAL0:%.+]] =   IE.Less([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10x1xf16>, tensor<1x50xf16> -> tensor<10x50xi8>
    // CHECK-NOT:   IE.Less
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @GreaterEqualBroadcastable
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<10x1xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x50xf16>
func.func @GreaterEqualBroadcastable(%arg0: tensor<10x1xf16>, %arg1: tensor<1x50xf16>) -> tensor<10x50xi8> {
    %0 = IE.GreaterEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10x1xf16>, tensor<1x50xf16> -> tensor<10x50xi8>
    return %0 : tensor<10x50xi8>

    // CHECK:       [[VAL0:%.+]] =   IE.GreaterEqual([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10x1xf16>, tensor<1x50xf16> -> tensor<10x50xi8>
    // CHECK-NOT:   IE.GreaterEqual
    // CHECK:       return [[VAL0]]
}
