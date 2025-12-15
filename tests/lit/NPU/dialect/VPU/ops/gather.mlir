//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL:  func.func @GatherWithDifferentInOutRanksAndExplicitOrder
// CHECK-SAME:      ([[INPUT0:%.+]]: tensor<51865x512xf16, {order = #NC}>,
// CHECK-SAME:       [[INPUT1:%.+]]: tensor<1x16xsi32>)
func.func @GatherWithDifferentInOutRanksAndExplicitOrder(%input0: tensor<51865x512xf16, {order = #NC}>, %input1: tensor<1x16xsi32>)
        -> tensor<1x16x512xf16, {order = #CHW}> {
    %gather = VPU.Gather(%input0, %input1) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64}
        : tensor<51865x512xf16, {order = #NC}>, tensor<1x16xsi32> -> tensor<1x16x512xf16, {order = #CHW}>
    return %gather: tensor<1x16x512xf16, {order = #CHW}>

    // CHECK:       [[GATHER:%.+]] = VPU.Gather([[INPUT0]], [[INPUT1]]) {
    // CHECK-SAME:    axis_value = 0 : i64,
    // CHECK-SAME:    batch_dims = 0 : i64,
    // CHECK-SAME:    indices_rank = 2 : i64
    // CHECK-SAME:  } : tensor<51865x512xf16, {order = #NC}>, tensor<1x16xsi32> -> tensor<1x16x512xf16, {order = #CHW}>
    //              return [[GATHER]]
}
