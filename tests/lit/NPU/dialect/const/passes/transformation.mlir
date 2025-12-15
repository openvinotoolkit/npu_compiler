//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @MoveSubviewBeforeMempermuteNoMemoryReordering
func.func @MoveSubviewBeforeMempermuteNoMemoryReordering() -> tensor<1x512x12x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x512x12x512xf16, {order = #NHWC}> = dense<3.56> : tensor<1x12x512x512xf32>, [#const.CastElemType<f16>, #const.MemPermute<#NHWC, #NCHW>]
    %slice = VPU.Slice %cst [0, 0, 0, 456] [1, 512, 12, 56] : tensor<1x512x12x512xf16, {order = #NHWC}> to tensor<1x512x12x56xf16, {order = #NHWC}>
    return %slice : tensor<1x512x12x56xf16, {order = #NHWC}>

    // CHECK:      [[CST:%.+]] = const.Declare tensor<1x512x12x56xf16, {order = #NHWC}> = dense<3.560000e+00> : tensor<1x12x512x512xf32>,
    // CHECK-SAME:  [#const.SubView<[0, 0, 456, 0], [1, 12, 56, 512]>, #const.CastElemType<f16>, #const.MemPermute<#NHWC, #NCHW>]
    // CHECK:      return [[CST]] : tensor<1x512x12x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK-LABEL:   @MoveSubviewBeforeMempermuteMemoryReordering
func.func @MoveSubviewBeforeMempermuteMemoryReordering() -> tensor<1x512x56x12xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x512x512x12xf16, {order = #NHWC}> = dense<0.0> : tensor<1x12x512x512xf32>, [#const.CastElemType<f16>, #const.MemPermute<#NHWC, #NWCH>]
    %slice = VPU.Slice %cst [0, 0, 456, 0] [1, 512, 56, 12] : tensor<1x512x512x12xf16, {order = #NHWC}> to tensor<1x512x56x12xf16, {order = #NHWC}>

    return %slice : tensor<1x512x56x12xf16, {order = #NHWC}>
    // CHECK:      [[CST:%.+]] = const.Declare tensor<1x512x56x12xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x12x512x512xf32>,
    // CHECK-SAME:  [#const.SubView<[0, 0, 0, 456], [1, 12, 512, 56]>, #const.CastElemType<f16>, #const.MemPermute<#NHWC, #NWCH>]
    // CHECK:      return [[CST]] : tensor<1x512x56x12xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-LABEL:   @MoveSubviewBeforeMempermuteMemoryReorderingAndInitialLayout
func.func @MoveSubviewBeforeMempermuteMemoryReorderingAndInitialLayout() ->  tensor<1x512x12x56xf16, {order = #NWCH}> {
    %cst = const.Declare tensor<1x512x12x512xf16, {order = #NWCH}> = dense<0.0> : tensor<1x12x512x512xf32>, [#const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.MemPermute<#NWCH, #NHCW>]
    %slice = VPU.Slice %cst [0, 0, 0, 456] [1, 512, 12, 56] : tensor<1x512x12x512xf16, {order = #NWCH}> to tensor<1x512x12x56xf16, {order = #NWCH}>

    return %slice :  tensor<1x512x12x56xf16, {order = #NWCH}>
    // CHECK:      [[CST:%.+]] = const.Declare tensor<1x512x12x56xf16, {order = #NWCH}> = dense<0.000000e+00> : tensor<1x12x512x512xf32>,
    // CHECK-SAME:  [#const.SubView<[0, 0, 0, 456], [1, 12, 512, 56]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.MemPermute<#NWCH, #NHCW>]
    // CHECK:      return [[CST]] : tensor<1x512x12x56xf16, {order = #NWCH}>
}

// -----


#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
// CHECK-LABEL:   @MoveSubviewBeforeMempermuteMemoryReorderingNonSymmetric
func.func @MoveSubviewBeforeMempermuteMemoryReorderingNonSymmetric() ->  tensor<1x512x12x56xf16, {order = #NWCH}> {
    %cst = const.Declare tensor<1x512x12x512xf16, {order = #NWCH}> = dense<0.0> : tensor<1x12x512x512xf32>, [ #const.Reorder<#NWHC>,#const.CastElemType<f16>, #const.MemPermute<#NWCH, #NHCW>]
    %slice = VPU.Slice %cst [0, 0, 0, 456] [1, 512, 12, 56] : tensor<1x512x12x512xf16, {order = #NWCH}> to tensor<1x512x12x56xf16, {order = #NWCH}>

    return %slice :  tensor<1x512x12x56xf16, {order = #NWCH}>
    // CHECK:      [[CST:%.+]] = const.Declare tensor<1x512x12x56xf16, {order = #NWCH}> = dense<0.000000e+00> : tensor<1x12x512x512xf32>,
    // CHECK-SAME:  [#const.SubView<[0, 0, 456, 0], [1, 12, 56, 512]>, #const.Reorder<#NWHC>, #const.CastElemType<f16>, #const.MemPermute<#NWCH, #NHCW>]

    // CHECK:      return [[CST]] : tensor<1x512x12x56xf16, {order = #NWCH}>
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
// CHECK-LABEL:   @MoveSubviewBeforeMempermuteMemoryReorderingSame
func.func @MoveSubviewBeforeMempermuteMemoryReorderingSame() ->  tensor<1x12x512x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x12x512x512xf16, {order = #NHWC}> = dense<0.0> : tensor<1x12x512x512xf32>, [ #const.Reorder<#NWHC>,#const.CastElemType<f16>, #const.MemPermute<#NHWC, #NHCW>]
    %slice = VPU.Slice %cst [0, 0, 0, 456] [1, 12, 512, 56] : tensor<1x12x512x512xf16, {order = #NHWC}> to tensor<1x12x512x56xf16, {order = #NHWC}>

    return %slice :  tensor<1x12x512x56xf16, {order = #NHWC}>
    // CHECK:      [[CST:%.+]] = const.Declare tensor<1x12x512x56xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x12x512x512xf32>,
    // CHECK-SAME:  [#const.SubView<[0, 0, 0, 456], [1, 12, 512, 56]>, #const.Reorder<#NWHC>, #const.CastElemType<f16>, #const.MemPermute<#NHWC, #NHCW>]

    // CHECK:      return [[CST]] : tensor<1x12x512x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL:   @MoveSubviewBeforeMempermuteTrivial
func.func @MoveSubviewBeforeMempermuteTrivial() ->  tensor<1x1x12x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x12x512xf16, {order = #NHWC}> = dense<0.0> : tensor<1x12x1x512xf32>, [ #const.CastElemType<f16>, #const.MemPermute<#NHWC, #NCWH>]
    %slice = VPU.Slice %cst [0, 0, 0, 456] [1, 1, 12, 56] : tensor<1x1x12x512xf16, {order = #NHWC}> to tensor<1x1x12x56xf16, {order = #NHWC}>

    return %slice :  tensor<1x1x12x56xf16, {order = #NHWC}>
    // CHECK:      [[CST:%.+]] = const.Declare tensor<1x1x12x56xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x12x1x512xf32>,
    // CHECK-SAME:  [#const.SubView<[0, 0, 0, 456], [1, 12, 1, 56]>, #const.CastElemType<f16>, #const.MemPermute<#NHWC, #NCWH>]

    // CHECK:      return [[CST]] : tensor<1x1x12x56xf16, {order = #NHWC}>
}
