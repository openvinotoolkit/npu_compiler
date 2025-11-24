//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-reduce-mean-square --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @FuseReduceMeanSquare
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x32x96xf32>)
func.func @FuseReduceMeanSquare(%arg0: tensor<1x32x32x96xf32>) -> tensor<1x32x32x1xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32> isSplat
    %0 = IE.Power(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x96xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x32x96xf32>
    %1 = IE.ReduceMean(%0) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf32> -> tensor<1x32x32x1xf32>
    %2 = IE.Sqrt(%1) : tensor<1x32x32x1xf32> -> tensor<1x32x32x1xf32>
    return %2 : tensor<1x32x32x1xf32>

    // CHECK: [[ReduceMeanSquare:%.+]] = IE.ReduceMeanSquare([[ARG0]]) {axes_value = [3], keep_dims} : tensor<1x32x32x96xf32> -> tensor<1x32x32x1xf32>
    // CHECK: return [[ReduceMeanSquare]] : tensor<1x32x32x1xf32>
}
