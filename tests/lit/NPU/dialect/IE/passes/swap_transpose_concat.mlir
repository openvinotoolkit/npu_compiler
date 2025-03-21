//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-transpose-concat %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

func.func @SwapTransposeConcatWithOffsets(%arg0: tensor<1x76x4x1xf16>, %arg1: tensor<1x76x4x1xf16> ) -> tensor<1x8x76x1xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHCW} : tensor<1x76x4x1xf16> -> tensor<1x4x76x1xf16>
    %1 = IE.Transpose(%arg1) {order_value = #NHCW} : tensor<1x76x4x1xf16> -> tensor<1x4x76x1xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]]} : tensor<1x4x76x1xf16>, tensor<1x4x76x1xf16> -> tensor<1x8x76x1xf16>
    return %2: tensor<1x8x76x1xf16>

    // CHECK:                IE.Concat(%arg0, %arg1)
    // CHECK-SAME{LITERAL} : {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x76x4x1xf16>, tensor<1x76x4x1xf16> -> tensor<1x76x8x1xf16>
    // CHECK:                IE.Transpose
    // CHECK-SAME:           {order_value = #NHCW} : tensor<1x76x8x1xf16> -> tensor<1x8x76x1xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

func.func @SwapTransposeConcatWithAxis(%arg0: tensor<1x76x4x1xf16>, %arg1: tensor<1x76x4x1xf16> ) -> tensor<1x8x76x1xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHCW} : tensor<1x76x4x1xf16> -> tensor<1x4x76x1xf16>
    %1 = IE.Transpose(%arg1) {order_value = #NHCW} : tensor<1x76x4x1xf16> -> tensor<1x4x76x1xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x4x76x1xf16>, tensor<1x4x76x1xf16> -> tensor<1x8x76x1xf16>
    return %2: tensor<1x8x76x1xf16>

    // CHECK:                IE.Concat(%arg0, %arg1)
    // CHECK-SAME{LITERAL} : {per_axis = #IE.Concat<axis = 2>} : tensor<1x76x4x1xf16>, tensor<1x76x4x1xf16> -> tensor<1x76x8x1xf16>
    // CHECK:                IE.Transpose
    // CHECK-SAME:           {order_value = #NHCW} : tensor<1x76x8x1xf16> -> tensor<1x8x76x1xf16>
}
