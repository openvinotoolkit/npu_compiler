//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Fold
func @Fold() -> tensor<1x3x8x4xf32, {order = #NHWC}> {
    %0 = const.Declare tensor<1x3x16x16xf32, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x3x16x16xf32>, [#const.Reorder<#NHWC>]

    %1 = IE.Slice %0 [0, 0, 8, 12] [1, 3, 8, 4] :
        tensor<1x3x16x16xf32, {order = #NHWC}> to
        tensor<1x3x8x4xf32, {order = #NHWC}>

    return %1 : tensor<1x3x8x4xf32, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = const.Declare tensor<1x3x8x4xf32, {order = #NHWC}> =
    // CHECK-SAME:      [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 8, 12], [1, 3, 8, 4]>]
    // CHECK-NOT:   IE.Slice

    // CHECK:       return [[VAR0]] : tensor<1x3x8x4xf32, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @ComposeSlice
func @ComposeSlice(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x3x8x4xf32> {
    %1 = IE.Slice %arg0 [0, 0, 0, 8] [1, 3, 16, 8] : tensor<1x3x16x16xf32> to tensor<1x3x16x8xf32>
    %2 = IE.Slice %1 [0, 0, 8, 4] [1, 3, 8, 4] : tensor<1x3x16x8xf32> to tensor<1x3x8x4xf32>
    return %2 : tensor<1x3x8x4xf32>

    // CHECK:       [[VAR0:%.*]] = IE.Slice %arg0 [0, 0, 8, 12] [1, 3, 8, 4] :
    // CHECK-SAME:      tensor<1x3x16x16xf32> to tensor<1x3x8x4xf32>

    // CHECK:       return [[VAR0]] : tensor<1x3x8x4xf32>
}
