//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConstFold
func @ConstFold() -> tensor<1x16xf16> {
    %0 = const.Declare tensor<1x16xf32> = dense<1.0> : tensor<1x16xf32>
    %1 = IE.Convert(%0) { dstElemType = f16 } : tensor<1x16xf32> -> tensor<1x16xf16>
    return %1 : tensor<1x16xf16>

    // CHECK:       %[[VAL0:.*]] = const.Declare tensor<1x16xf16> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x16xf32>, [#const.ConvertElemType<f16>]
    // CHECK-NOT:   IE.Convert
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @SameType
func @SameType(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32>
    return %0 : tensor<1x2x3x4xf32>

    // CHECK-NOT:   IE.Convert
    // CHECK:       return %arg0 : tensor<1x2x3x4xf32>
}
