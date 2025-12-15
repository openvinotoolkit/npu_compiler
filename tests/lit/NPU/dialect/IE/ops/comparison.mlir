//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @LessBroadcastable
func.func @LessBroadcastable(%arg0: tensor<10x1xf16>, %arg1: tensor<1x50xf16>) -> tensor<10x50xi8> {
    %0 = IE.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10x1xf16>, tensor<1x50xf16> -> tensor<10x50xi8>
    return %0 : tensor<10x50xi8>

    // CHECK:       %[[VAL0:.*]] =   IE.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10x1xf16>, tensor<1x50xf16> -> tensor<10x50xi8>
    // CHECK-NOT:   IE.Less
    // CHECK:       return %[[VAL0]]
}
