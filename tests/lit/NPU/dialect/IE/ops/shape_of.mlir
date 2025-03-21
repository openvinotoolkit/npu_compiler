//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @FoldSi32
func.func @FoldSi32(%arg0: tensor<1x8x4x4xf32>) -> tensor<4xsi32> {
    %shape_of = IE.ShapeOf(%arg0) {dstElemType = si32} : tensor<1x8x4x4xf32> -> tensor<4xsi32>
    return %shape_of : tensor<4xsi32>

    // CHECK:   [[CONST:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 8, 4, 4]> : tensor<4xsi32>
    // CHECK:   return [[CONST]]
}

// CHECK-LABEL: @FoldSi64
func.func @FoldSi64(%arg0: tensor<1x8x4x4xf32>) -> tensor<4xsi64> {
    %shape_of = IE.ShapeOf(%arg0) {dstElemType = si64} : tensor<1x8x4x4xf32> -> tensor<4xsi64>
    return %shape_of : tensor<4xsi64>

    // CHECK:   [[CONST:%.+]] = const.Declare tensor<4xsi64> = dense<[1, 8, 4, 4]> : tensor<4xsi64>
    // CHECK:   return [[CONST]]
}
