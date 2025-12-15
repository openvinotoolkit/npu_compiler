//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @Fold
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x8x4x4xf16>
func.func @Fold(%arg0: tensor<1x8x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>

    // CHECK-NOT:  VPU.Expand
    // CHECK:      return [[INPUT]]
}

// -----

// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<1x11x12x12xf16> {
    %cst = const.Declare tensor<1x5x10x11xf16> = dense<1.0> : tensor<1x5x10x11xf16>
    %0 = VPU.Expand(%cst) {pads_begin = [0, 3, 0, 1], pads_end = [0, 3, 2, 0]} : tensor<1x5x10x11xf16> -> tensor<1x11x12x12xf16>
    return %0 : tensor<1x11x12x12xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x11x12x12xf16> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x5x10x11xf16>, [#const.PadWithZero<[0, 3, 0, 1], [0, 3, 2, 0]>]
    // CHECK-NOT:   VPU.Expand
    // CHECK:       return [[CST]]
}
