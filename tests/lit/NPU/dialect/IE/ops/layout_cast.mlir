//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @FoldLayoutCast
func.func @FoldLayoutCast(%arg0: tensor<1x4x8x64xf32>) -> tensor<1x4x8x64xf32> {
    %0 = IE.LayoutCast(%arg0) {dst_order = #NCHW} : tensor<1x4x8x64xf32> -> tensor<1x4x8x64xf32>
    return %0 : tensor<1x4x8x64xf32>

    // CHECK-NOT:   IE.LayoutCast
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<1x2x3x4xf32> {
    %0 = const.Declare tensor<1x2x3x4xf32, {order = #NHWC}> = dense<5.0> : tensor<1x2x3x4xf32>, [#const.Reorder<#NHWC>]
    %1 = IE.LayoutCast(%0) {dst_order = #NCHW} : tensor<1x2x3x4xf32, {order = #NHWC}> -> tensor<1x2x3x4xf32>
    return %1 : tensor<1x2x3x4xf32>

    // CHECK-NOT:   IE.LayoutCast
    // CHECK:      const.Declare tensor<1x2x3x4xf32> = dense<5.000000e+00> : tensor<1x2x3x4xf32>,
    // CHECK-SAME:      [#const.Reorder<#NHWC>, #const.LayoutCast<#NCHW>]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseLayoutCasts
func.func @FuseLayoutCasts(%arg0: tensor<1x4x8x64xf32>) -> tensor<1x4x8x64xf32, {order = #NHWC}> {
    %0 = IE.LayoutCast(%arg0) {
        dst_order = #NCWH
    } : tensor<1x4x8x64xf32> -> tensor<1x4x8x64xf32, {order = #NCWH}>

    %1 = IE.LayoutCast(%0) {
        dst_order = #NHWC
    } : tensor<1x4x8x64xf32, {order = #NCWH}> -> tensor<1x4x8x64xf32, {order = #NHWC}>

    return %1 : tensor<1x4x8x64xf32, {order = #NHWC}>

    // CHECK:   [[LAYOUT_CAST:%.*]] = IE.LayoutCast(%arg0) {
    // CHECK-SAME:      order = #NHWC
    // CHECK-SAME:  } : tensor<1x4x8x64xf32> -> tensor<1x4x8x64xf32, {order = #NHWC}>

    // CHECK:   return [[LAYOUT_CAST]]
}
