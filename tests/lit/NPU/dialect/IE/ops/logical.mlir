//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @LogicalAndBroadcastable
// CHECK-SAME:       ([[ARG0:%.+]]: tensor<1x28x300x1xf16>, [[ARG1:%.+]]: tensor<1x1x300x28xf16>) -> tensor<1x28x300x28xi8>
func.func @LogicalAndBroadcastable(%arg0: tensor<1x28x300x1xf16>, %arg1: tensor<1x1x300x28xf16>) -> tensor<1x28x300x28xi8> {
    %0 = IE.Convert(%arg0) {dstElemType = i8} : tensor<1x28x300x1xf16> -> tensor<1x28x300x1xi8>
    %1 = IE.Convert(%arg1) {dstElemType = i8} : tensor<1x1x300x28xf16> -> tensor<1x1x300x28xi8>
    %2 = IE.And(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x28x300x1xi8>, tensor<1x1x300x28xi8> -> tensor<1x28x300x28xi8>
    return %2 : tensor<1x28x300x28xi8>

    // CHECK:       [[CONVERT0:%.+]] = IE.Convert([[ARG0:%.+]]) {dstElemType = i8} : tensor<1x28x300x1xf16> -> tensor<1x28x300x1xi8>
    // CHECK:       [[CONVERT1:%.+]] = IE.Convert([[ARG1:%.+]]) {dstElemType = i8} : tensor<1x1x300x28xf16> -> tensor<1x1x300x28xi8>
    // CHECK:       [[VAL0:%.+]] = IE.And([[CONVERT0]], [[CONVERT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x28x300x1xi8>, tensor<1x1x300x28xi8> -> tensor<1x28x300x28xi8>
    // CHECK-NOT:   IE.And
    // CHECK:       return [[VAL0]]
}
