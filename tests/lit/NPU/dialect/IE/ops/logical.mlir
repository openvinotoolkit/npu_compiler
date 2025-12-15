//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @LogicalAndBroadcastable
func.func @LogicalAndBroadcastable(%arg0: tensor<1x28x300x1xf16>, %arg1: tensor<1x1x300x28xf16>) -> tensor<1x28x300x28xi8> {
    %0 = IE.Convert(%arg0) {dstElemType = i8} : tensor<1x28x300x1xf16> -> tensor<1x28x300x1xi8>
    %1 = IE.Convert(%arg1) {dstElemType = i8} : tensor<1x1x300x28xf16> -> tensor<1x1x300x28xi8>
    %2 = IE.And(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x28x300x1xi8>, tensor<1x1x300x28xi8> -> tensor<1x28x300x28xi8>
    return %2 : tensor<1x28x300x28xi8>

    // CHECK:       %[[VAL0:.*]] = IE.And(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x28x300x1xi8>, tensor<1x1x300x28xi8> -> tensor<1x28x300x28xi8>
    // CHECK-NOT:   IE.And
    // CHECK:       return %[[VAL0]]
}
