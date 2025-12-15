//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL:  func.func @GatherNDWithDifferentInOutRanksAndExplicitOrder
// CHECK-SAME:      ([[INPUT0:%.+]]: tensor<5x7x3xsi32, {order = #CHW}>,
// CHECK-SAME:       [[INPUT1:%.+]]: tensor<5x1xsi32>)
func.func @GatherNDWithDifferentInOutRanksAndExplicitOrder(%input0: tensor<5x7x3xsi32, {order = #CHW}>, %input1: tensor<5x1xsi32>)
        -> tensor<5x3xsi32, {order = #NC}> {
    %gather_nd = VPU.GatherND(%input0, %input1) {batch_dims = 1 : i64} : tensor<5x7x3xsi32, {order = #CHW}>, tensor<5x1xsi32> -> tensor<5x3xsi32, {order = #NC}>
    return %gather_nd: tensor<5x3xsi32, {order = #NC}>

    // CHECK:       [[GATHER_ND:%.+]] = VPU.GatherND([[INPUT0]], [[INPUT1]]) {
    // CHECK-SAME:    batch_dims = 1 : i64
    // CHECK-SAME:  } : tensor<5x7x3xsi32, {order = #CHW}>, tensor<5x1xsi32> -> tensor<5x3xsi32, {order = #NC}>
    //              return [[GATHER_ND]]
}
