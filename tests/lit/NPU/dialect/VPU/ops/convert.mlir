//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<1x16xf16> {
    %0 = const.Declare tensor<1x16xf32> = dense<1.0> : tensor<1x16xf32>
    %1 = VPU.Convert(%0) { dstElemType = f16 } : tensor<1x16xf32> -> tensor<1x16xf16>
    return %1 : tensor<1x16xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x16xf16> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x16xf32>, [#const.CastElemType<f16>]
    // CHECK-NOT:   VPU.Convert
    // CHECK:       return [[CST]]
}

// -----

// CHECK-LABEL: @SameType
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x3x4xf32>
func.func @SameType(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32>
    return %0 : tensor<1x2x3x4xf32>

    // CHECK-NOT:   VPU.Convert
    // CHECK:       return [[INPUT]] : tensor<1x2x3x4xf32>
}
