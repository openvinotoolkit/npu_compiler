//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --adjust-mem-permute-around-op %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @AdjustMemPermutesAroundMultiply
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x51x1xf16, {order = #NCWH}>, [[ARG_1:%[^:]+]]: tensor<1x128x51x64xf16, {order = #NHWC}>)
func.func @AdjustMemPermutesAroundMultiply(%arg0: tensor<1x1x51x1xf16, {order = #NCWH}>, %arg1: tensor<1x128x51x64xf16, {order = #NHWC}>) -> tensor<1x128x51x64xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg1) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x128x51x64xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %1 = IE.Multiply(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16, {order = #NCWH}>, tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NHWC}>

    return %2 : tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:        [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[ARG_0]])
    // CHECK:            {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x1x51x1xf16, {order = #NCWH}> -> tensor<1x1x51x1xf16, {order = #NHWC}>
    // CHECK:        [[MULTIPLY:%.+]] = IE.Multiply([[PERMUTE_CAST]], [[ARG_1]])
    // CHECK:            {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16, {order = #NHWC}>, tensor<1x128x51x64xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    // CHECK:        return [[MULTIPLY]] : tensor<1x128x51x64xf16, {order = #NHWC}>
}

// CHECK-LABEL: @AdjustMemPermutesAroundMultiplyWithConstInput
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x128x51x64xf16, {order = #NHWC}>)
func.func @AdjustMemPermutesAroundMultiplyWithConstInput(%arg0: tensor<1x128x51x64xf16, {order = #NHWC}>) -> tensor<1x128x51x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x51x1xf16, {order = #NCWH}> = dense<2.0> : tensor<1x1x51x1xf16>, [#const.Reorder<#NCWH>]
    %0 = IE.MemPermute(%arg0) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x128x51x64xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %1 = IE.Multiply(%cst, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16, {order = #NCWH}>, tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NHWC}>

    return %2 : tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:        [[CST:%.+]] = const.Declare tensor<1x1x51x1xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<1x1x51x1xf16>,
    // CHECK-SAME:            [#const.MemPermute<#NHWC, #NHWC>]
    // CHECK:        [[MULTIPLY:%.+]] = IE.Multiply([[ARG_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x51x64xf16, {order = #NHWC}>, tensor<1x1x51x1xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    // CHECK:        return [[MULTIPLY]] : tensor<1x128x51x64xf16, {order = #NHWC}>
}

// CHECK-LABEL: @AdjustMemPermutesAroundMultiplyWithPermuteQuantizeInput
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x51x1xf16, {order = #NCWH}>, [[ARG_1:%[^:]+]]: tensor<1x128x51x64xf16, {order = #NHWC}>)
func.func @AdjustMemPermutesAroundMultiplyWithPermuteQuantizeInput(%arg0: tensor<1x1x51x1xf16, {order = #NCWH}>, %arg1: tensor<1x128x51x64xf16, {order = #NHWC}>) -> tensor<1x128x51x64xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NCWH, mem_perm = #NWHC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x128x51x64xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %1 = IE.Multiply(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16, {order = #NCWH}>, tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    return %2 : tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:        [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[ARG_0]])
    // CHECK:            {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x1x51x1xf16, {order = #NCWH}> -> tensor<1x1x51x1xf16, {order = #NHWC}>
    // CHECK:        [[MULTIPLY:%.+]] = IE.Multiply([[PERMUTE_CAST]], [[ARG_1]])
    // CHECK:            {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16, {order = #NHWC}>, tensor<1x128x51x64xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    // CHECK:        return [[MULTIPLY]] : tensor<1x128x51x64xf16, {order = #NHWC}>
}

// CHECK-LABEL: @AdjustMemPermutesAroundMultiplyWithoutDeadLoop
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x128x16x64xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<1x16x1x128xf16, {order = #NHWC}>)
func.func @AdjustMemPermutesAroundMultiplyWithoutDeadLoop(%arg0: tensor<1x128x16x64xf16, {order = #NHWC}>, %arg1: tensor<1x16x1x128xf16, {order = #NHWC}>) -> tensor<1x128x16x64xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NWCH, mem_perm = #NWCH} : tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x16x64x128xf16, {order = #NWCH}>
    %1 = IE.PermuteCast(%arg1) {dst_order = #NWCH, mem_perm = #NHWC} : tensor<1x16x1x128xf16, {order = #NHWC}> -> tensor<1x16x1x128xf16, {order = #NWCH}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x64x128xf16, {order = #NWCH}>, tensor<1x16x1x128xf16, {order = #NWCH}> -> tensor<1x16x64x128xf16, {order = #NWCH}>
    %3 = IE.MemPermute(%2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x64x128xf16, {order = #NWCH}> -> tensor<1x128x16x64xf16, {order = #NHWC}>
    return %3 : tensor<1x128x16x64xf16, {order = #NHWC}>

    // CHECK:        [[IN_PERMUTE_CAST:%.+]] = IE.PermuteCast([[ARG_0]])
    // CHECK:            {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x16x64x128xf16>
    // CHECK:        [[MEM_PERMUTE:%.+]] = IE.MemPermute([[ARG_1]])
    // CHECK:            {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x1x128xf16, {order = #NHWC}> -> tensor<1x16x1x128xf16>
    // CHECK:        [[MULTIPLY:%.+]] = IE.Multiply([[IN_PERMUTE_CAST]], [[MEM_PERMUTE]])
    // CHECK:            {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x64x128xf16>, tensor<1x16x1x128xf16> -> tensor<1x16x64x128xf16>
    // CHECK:        [[OUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]])
    // CHECK:            {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x64x128xf16> -> tensor<1x128x16x64xf16, {order = #NHWC}>
    // CHECK:        return [[OUT_PERMUTE_CAST]] : tensor<1x128x16x64xf16, {order = #NHWC}>
}

// CHECK-LABEL: @NotAdjustMemPermutesAroundMultiply
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x51x1xf16>, [[ARG_1:%[^:]+]]: tensor<1x128x51x64xf16>)
func.func @NotAdjustMemPermutesAroundMultiply(%arg0: tensor<1x1x51x1xf16>, %arg1: tensor<1x128x51x64xf16>) -> tensor<1x128x51x64xf16, {order = #NHWC}> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16>, tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    return %1 : tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:        [[MULTIPLY:%.+]] = IE.Multiply([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16>, tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16>
    // CHECK:        [[PERMUTE:%.+]] = IE.MemPermute([[MULTIPLY]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    // CHECK:        return [[PERMUTE]] : tensor<1x128x51x64xf16, {order = #NHWC}>
}

// CHECK-LABEL: @NotAdjustInputMemPermutesToOutput
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x2x16x16xf16>, [[ARG_1:%[^:]+]]: tensor<1x2x16x16xf16, {order = #NHWC}>)
func.func @NotAdjustInputMemPermutesToOutput(%arg0: tensor<1x2x16x16xf16>, %arg1: tensor<1x2x16x16xf16, {order = #NHWC}>) -> tensor<1x2x16x16xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16, {order = #NHWC}>, tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>

    return %2 : tensor<1x2x16x16xf16, {order = #NHWC}>

    // CHECK:        [[PERMUTE_L:%.+]] = IE.MemPermute([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        [[PERMUTE_R:%.+]] = IE.MemPermute([[ARG_1]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        [[MULTIPLY:%.+]] = IE.Multiply([[PERMUTE_L]], [[PERMUTE_R]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16, {order = #NHWC}>, tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        return [[MULTIPLY]] : tensor<1x2x16x16xf16, {order = #NHWC}>
}

// CHECK-LABEL: @AdjustMemPermutesAfterTile
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x1x512xf16, {order = #NHWC}>)
func.func @AdjustMemPermutesAfterTile(%arg0: tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 512, 1]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    return %1 : tensor<1x2x512x512xf16>

    // CHECK:        [[PERMUTE:%.+]] = IE.PermuteCast([[ARG_0]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x1x1x512xf16>
    // CHECK:        [[TILE:%.+]] = IE.Tile([[PERMUTE]]) {repeats_values = [1, 2, 512, 1]} : tensor<1x1x1x512xf16> -> tensor<1x2x512x512xf16>
    // CHECK:        return [[TILE]] : tensor<1x2x512x512xf16>
}

// CHECK-LABEL: @NotAdjustMemPermutesAfterTile
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x2x256x512xf16, {order = #NHWC}>)
func.func @NotAdjustMemPermutesAfterTile(%arg0: tensor<1x2x256x512xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 2, 1]} : tensor<1x2x256x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    return %1 : tensor<1x2x512x512xf16>

    // CHECK:        [[TILE:%.+]] = IE.Tile([[ARG_0]]) {repeats_values = [1, 1, 2, 1]} : tensor<1x2x256x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:        [[PERMUTE:%.+]] = IE.MemPermute([[TILE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    // CHECK:        return [[PERMUTE]] : tensor<1x2x512x512xf16>
}

// CHECK-LABEL: @NotAdjustMemPermutesLayoutNotSupportInnerDimNotAligned
// CHECK-SAME: [[INPUT1:%.+]]: tensor<1x32x16x17xf16>, [[INPUT2:%.+]]: tensor<1x32x16x17xf16>
func.func @NotAdjustMemPermutesLayoutNotSupportInnerDimNotAligned(%arg0: tensor<1x32x16x17xf16>, %arg1: tensor<1x32x16x17xf16>) -> tensor<1x32x16x17xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x32x16x17xf16> -> tensor<1x32x16x17xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x32x16x17xf16> -> tensor<1x32x16x17xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x16x17xf16, {order = #NHWC}>, tensor<1x32x16x17xf16, {order = #NHWC}> -> tensor<1x32x16x17xf16, {order = #NHWC}>

    return %2 : tensor<1x32x16x17xf16, {order = #NHWC}>

    // CHECK:        [[PERMUTE0:%.+]] = IE.MemPermute([[INPUT1]])
    // CHECK:        [[PERMUTE1:%.+]] = IE.MemPermute([[INPUT2]])
    // CHECK:        [[ADD:%.+]] = IE.Add([[PERMUTE0]], [[PERMUTE1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        return [[ADD]] : tensor<1x32x16x17xf16, {order = #NHWC}>
}

// CHECK-LABEL: @AdjustMemPermutesWithPermuteCast
// CHECK-SAME: [[INPUT1:%.+]]: tensor<1x32x16x16xf16>, [[INPUT2:%.+]]: tensor<1x32x16x16xf16>
func.func @AdjustMemPermutesWithPermuteCast(%arg0: tensor<1x32x16x16xf16>, %arg1: tensor<1x32x16x16xf16>) -> tensor<1x32x16x16xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x32x16x16xf16> -> tensor<1x32x16x16xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x32x16x16xf16> -> tensor<1x32x16x16xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<1x32x16x16xf16, {order = #NHWC}> -> tensor<1x32x16x16xf16, {order = #NHWC}>

    return %2 : tensor<1x32x16x16xf16, {order = #NHWC}>

    // CHECK-NOT:    IE.MemPermute
    // CHECK:        [[PERMUTE_CAST1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x16x16xf16> -> tensor<1x16x32x16xf16, {order = #NHWC}>
    // CHECK:        [[PERMUTE_CAST2:%.+]] = IE.PermuteCast([[INPUT2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x16x16xf16> -> tensor<1x16x32x16xf16, {order = #NHWC}>
    // CHECK:        [[ADD:%.+]] = IE.Add([[PERMUTE_CAST1]], [[PERMUTE_CAST2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        [[PERMUTE:%.+]] = IE.MemPermute([[ADD]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x32x16xf16, {order = #NHWC}> -> tensor<1x32x16x16xf16, {order = #NHWC}>
    // CHECK:        return [[PERMUTE]] : tensor<1x32x16x16xf16, {order = #NHWC}>
}

// CHECK-LABEL: @AdjustMemPermutesLayoutSubtract
func.func @AdjustMemPermutesLayoutSubtract(%arg0: tensor<1x10x192x16xf16, {order = #NHWC}>, %arg1: tensor<1x10x192x16xf16, {order = #NHWC}>) -> tensor<1x10x192x16xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x10x192x16xf16, {order = #NHWC}> -> tensor<1x10x192x16xf16>
    %1 = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x10x192x16xf16, {order = #NHWC}> -> tensor<1x10x192x16xf16>
    %2 = IE.Subtract(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x192x16xf16>, tensor<1x10x192x16xf16> -> tensor<1x10x192x16xf16>
    %3 = IE.MemPermute(%2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x10x192x16xf16> -> tensor<1x10x192x16xf16, {order = #NHWC}>

    return %3 : tensor<1x10x192x16xf16, {order = #NHWC}>
    // CHECK-NOT:    IE.MemPermute
    // CHECK:        [[SUB:%.+]] = IE.Subtract([[ARG1:%.+]], [[ARG2:%.+]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        return [[SUB]] : tensor<1x10x192x16xf16, {order = #NHWC}>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustForConvert
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x21x513x513xf16, {order = #NHWC}>)
func.func @AdjustForConvert(%arg0: tensor<1x21x513x513xf16, {order = #NHWC}>) -> tensor<1x513x513x21xf32> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x21x513x513xf16, {order = #NHWC}> -> tensor<1x513x513x21xf16>
    %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x513x513x21xf16> -> tensor<1x513x513x21xf32>
    return %1 : tensor<1x513x513x21xf32>

    // CHECK:        [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x21x513x513xf16, {order = #NHWC}> -> tensor<1x21x513x513xf32, {order = #NHWC}>
    // CHECK:        [[PERMUTE:%.+]] = IE.PermuteCast([[CONVERT]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x21x513x513xf32, {order = #NHWC}> -> tensor<1x513x513x21xf32>
    // CHECK:        return [[PERMUTE]] : tensor<1x513x513x21xf32>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @AdjustForConcat
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x128x3000x1xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x128x3000x1xf16, {order = #NHWC}>, [[INPUT3:%.+]]: tensor<1x128x19x1xf16>
func.func @AdjustForConcat(%arg0: tensor<1x128x3000x1xf16, {order = #NHWC}>, %arg1: tensor<1x128x3000x1xf16, {order = #NHWC}>, %arg2: tensor<1x128x19x1xf16>)
-> tensor<1x128x6019x1xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x128x3000x1xf16, {order = #NHWC}> -> tensor<1x128x3000x1xf16>
    %1 = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x128x3000x1xf16, {order = #NHWC}> -> tensor<1x128x3000x1xf16>
    %2 = IE.Concat(%arg2, %0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 3019, 0]]} : tensor<1x128x19x1xf16>, tensor<1x128x3000x1xf16>, tensor<1x128x3000x1xf16> -> tensor<1x128x6019x1xf16>
    %3 = IE.MemPermute(%2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x128x6019x1xf16> -> tensor<1x128x6019x1xf16, {order = #NHWC}>
    return %3 : tensor<1x128x6019x1xf16, {order = #NHWC}>

    // CHECK: [[BA_PERMUTE:%.+]] = IE.MemPermute([[INPUT3]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x128x19x1xf16> -> tensor<1x128x19x1xf16, {order = #NHWC}>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[BA_PERMUTE]], [[INPUT1]], [[INPUT2]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK:    tensor<1x128x19x1xf16, {order = #NHWC}>, tensor<1x128x3000x1xf16, {order = #NHWC}>, tensor<1x128x3000x1xf16, {order = #NHWC}>
    // CHECK:    -> tensor<1x128x6019x1xf16, {order = #NHWC}>
    // CHECK: return [[CONCAT]] : tensor<1x128x6019x1xf16, {order = #NHWC}>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @AdjustForConcatToPermuteCast
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x128x3000x1xf16>, [[INPUT2:%.+]]: tensor<1x128x3000x1xf16>, [[INPUT3:%.+]]: tensor<1x1x19x128xf16>
func.func @AdjustForConcatToPermuteCast(%arg0: tensor<1x128x3000x1xf16>, %arg1: tensor<1x128x3000x1xf16>, %arg2: tensor<1x1x19x128xf16>)
-> tensor<1x1x128x6019xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x128x3000x1xf16> -> tensor<1x1x3000x128xf16>
    %1 = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x128x3000x1xf16> -> tensor<1x1x3000x128xf16>
    %2 = IE.Concat(%arg2, %0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 3019, 0]]} : tensor<1x1x19x128xf16>, tensor<1x1x3000x128xf16>, tensor<1x1x3000x128xf16> -> tensor<1x1x6019x128xf16>
    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x6019x128xf16> -> tensor<1x1x128x6019xf16>
    return %3 : tensor<1x1x128x6019xf16>

    // CHECK: [[BA_PERMUTE:%.+]] = IE.MemPermute([[INPUT3]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x19x128xf16> -> tensor<1x1x128x19xf16>
    // CHECK: [[PERMUTE_CAST1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x128x3000x1xf16> -> tensor<1x1x128x3000xf16>
    // CHECK: [[PERMUTE_CAST2:%.+]] = IE.PermuteCast([[INPUT2]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x128x3000x1xf16> -> tensor<1x1x128x3000xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[BA_PERMUTE]], [[PERMUTE_CAST1]], [[PERMUTE_CAST2]]) {per_axis = #IE.Concat<axis = 3 : i64>}
    // CHECK:    tensor<1x1x128x19xf16>, tensor<1x1x128x3000xf16>, tensor<1x1x128x3000xf16>
    // CHECK:    -> tensor<1x1x128x6019xf16>
    // CHECK: return [[CONCAT]] : tensor<1x1x128x6019xf16>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustForSoftmax
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x512x12x512xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x512x12x512xf16, {order = #NHWC}>)
func.func @AdjustForSoftmax(%arg0: tensor<1x512x12x512xf16, {order = #NHWC}>, %arg1: tensor<1x512x12x512xf16, {order = #NHWC}>) -> tensor<1x12x512x512xf16> {
    %add = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
         tensor<1x512x12x512xf16, {order = #NHWC}>, tensor<1x512x12x512xf16, {order = #NHWC}>
         -> tensor<1x512x12x512xf16, {order = #NHWC}>
    %permute = IE.PermuteCast(%add) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x12x512xf16, {order = #NHWC}> -> tensor<1x12x512x512xf16>
    %softmax = IE.SoftMax(%permute) {axisInd = 3} : tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %softmax : tensor<1x12x512x512xf16>

    // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK:            tensor<1x512x12x512xf16, {order = #NHWC}>, tensor<1x512x12x512xf16, {order = #NHWC}> -> tensor<1x512x12x512xf16, {order = #NHWC}>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]])
    // CHECK:            {axisInd = 1 : i64} : tensor<1x512x12x512xf16, {order = #NHWC}> -> tensor<1x512x12x512xf16, {order = #NHWC}>

    // CHECK:        [[PERMUTE:%.+]] = IE.PermuteCast([[SOFTMAX]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x12x512xf16, {order = #NHWC}> -> tensor<1x12x512x512xf16>
    // CHECK:        return [[PERMUTE]] : tensor<1x12x512x512xf16>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustForSoftmaxWithShapeCast
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x12x512x512xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x12x512x512xf16, {order = #NHWC}>)
func.func @AdjustForSoftmaxWithShapeCast(%arg0: tensor<1x12x512x512xf16, {order = #NHWC}>, %arg1: tensor<1x12x512x512xf16, {order = #NHWC}>) -> tensor<1x12x512x512xf16> {
    %add = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
         tensor<1x12x512x512xf16, {order = #NHWC}>, tensor<1x12x512x512xf16, {order = #NHWC}>
         -> tensor<1x12x512x512xf16, {order = #NHWC}>
    %shapecast = IE.ShapeCast { shape = [1, 512, 12, 512] } inputs(%add : tensor<1x12x512x512xf16, {order = #NHWC}>) -> tensor<1x512x12x512xf16, {order = #NHWC}>
    %permute = IE.PermuteCast(%shapecast) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x12x512xf16, {order = #NHWC}> -> tensor<1x12x512x512xf16>
    %softmax = IE.SoftMax(%permute) {axisInd = 3} : tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %softmax : tensor<1x12x512x512xf16>

    // CHECK:       [[SHAPECAST0:%.+]] = IE.ShapeCast {shape = [1, 512, 12, 512]}
    // CHECK:       [[SHAPECAST1:%.+]] = IE.ShapeCast {shape = [1, 512, 12, 512]}

    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPECAST0]], [[SHAPECAST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK:            tensor<1x512x12x512xf16, {order = #NHWC}>, tensor<1x512x12x512xf16, {order = #NHWC}> -> tensor<1x512x12x512xf16, {order = #NHWC}>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]])
    // CHECK:            {axisInd = 1 : i64} : tensor<1x512x12x512xf16, {order = #NHWC}> -> tensor<1x512x12x512xf16, {order = #NHWC}>

    // CHECK:        [[PERMUTE:%.+]] = IE.PermuteCast([[SOFTMAX]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x12x512xf16, {order = #NHWC}> -> tensor<1x12x512x512xf16>
    // CHECK:        return [[PERMUTE]] : tensor<1x12x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @FuseMemPermutesWithMultiUsers
// CHECK-SAME:    ([[INPUT_0:%.+]]: tensor<1x16x32x64xf16, {order = #NHWC}>, [[INPUT_1:%.+]]: tensor<1x16x32x64xf16, {order = #NHWC}>)
func.func @FuseMemPermutesWithMultiUsers(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>, %arg1: tensor<1x16x32x64xf16, {order = #NHWC}>) -> (tensor<1x32x64x16xf16, {order = #NHWC}>, tensor<1x32x64x16xf16>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x64x16xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x64x16xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x32x64x16xf16, {order = #NHWC}>, tensor<1x32x64x16xf16, {order = #NHWC}> -> tensor<1x32x64x16xf16, {order = #NHWC}>
    %3 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x32x64x16xf16, {order = #NHWC}> -> tensor<1x32x64x16xf16>

    return %2, %3 : tensor<1x32x64x16xf16, {order = #NHWC}>, tensor<1x32x64x16xf16>

    // CHECK:     [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT_0]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x64x16xf16>
    // CHECK:     [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[PERMUTE_CAST_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x64x16xf16> -> tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:     [[PERMUTE_CAST_2:%.+]] = IE.PermuteCast([[INPUT_1]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x64x16xf16>
    // CHECK:     [[PERMUTE_CAST_3:%.+]] = IE.PermuteCast([[PERMUTE_CAST_2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x64x16xf16> -> tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:     [[ADD:%.+]] = IE.Add([[PERMUTE_CAST_1]], [[PERMUTE_CAST_3]])
    // CHECK:     [[MEM_PERMUTE:%.+]] = IE.MemPermute([[ADD]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x64x16xf16, {order = #NHWC}>
    // CHECK:     [[PERMUTE_CAST_4:%.+]] = IE.PermuteCast([[INPUT_0]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x64x16xf16>

    // CHECK:     return [[MEM_PERMUTE]], [[PERMUTE_CAST_4]] : tensor<1x32x64x16xf16, {order = #NHWC}>, tensor<1x32x64x16xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @DoNotAdjustForDynamicDequantize
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x16x255x128xsi4>, [[INPUT2:%.+]]: tensor<16x255x1xf16>
func.func @DoNotAdjustForDynamicDequantize(%arg0: tensor<1x16x255x128xsi4>, %arg1: tensor<16x255x1xf16>) -> tensor<1x255x16x128xf16> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !quant.uniform<i4:f16, 1.000000e+00>} : tensor<1x16x255x128xsi4> -> tensor<1x16x255x128x!quant.uniform<i4:f16, 1.000000e+00>>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 16, 255, 1]} : tensor<16x255x1xf16> -> tensor<1x16x255x1xf16>
    %2 = IE.DynamicDequantize(%0, %1) {dstElemType = f16} : tensor<1x16x255x128x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<1x16x255x1xf16> -> tensor<1x16x255x128xf16>
    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x16x255x128xf16> -> tensor<1x255x16x128xf16>
    return %3 : tensor<1x255x16x128xf16>

    // CHECK:        [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[INPUT1]])
    // CHECK:        [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[INPUT2]])
    // CHECK:        [[DYNAMIC_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANTIZE_CAST]], [[AFFINE_RESHAPE]])
    // CHECK:        [[MEM_PERMUTE:%.+]] = IE.MemPermute([[DYNAMIC_DEQUANT]])
    // CHECK:        return [[MEM_PERMUTE]] : tensor<1x255x16x128xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseAffineReshapePermuteCastAroundAdd
// CHECK-SAME:  ([[INPUT1:%.+]]: tensor<1x128x16x64xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x128x16x64xf16, {order = #NHWC}>)
func.func @FuseAffineReshapePermuteCastAroundAdd(%arg0: tensor<1x128x16x64xf16, {order = #NHWC}>, %arg1: tensor<1x128x16x64xf16, {order = #NHWC}>) -> tensor<1x128x16x64xf16, {order = #NHWC}> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x16x64x128xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1024, 128]} : tensor<1x16x64x128xf16> -> tensor<1024x128xf16>
    %2 = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x16x64x128xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1024, 128]} : tensor<1x16x64x128xf16> -> tensor<1024x128xf16>
    %4 = IE.Add(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x128xf16>, tensor<1024x128xf16> -> tensor<1024x128xf16>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 16, 64, 128]} : tensor<1024x128xf16> -> tensor<1x16x64x128xf16>
    %6 = IE.PermuteCast(%5) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x64x128xf16> -> tensor<1x128x16x64xf16, {order = #NHWC}>
    return %6 : tensor<1x128x16x64xf16, {order = #NHWC}>

    // CHECK-NOT:  IE.PermuteCast
    // CHECK-NOT:  IE.AffineReshape
    // CHECK:      [[ADD:%.+]] = IE.Add([[INPUT1]], [[INPUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x16x64xf16, {order = #NHWC}>, tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x128x16x64xf16, {order = #NHWC}>
    // CHECK:      return [[ADD]] : tensor<1x128x16x64xf16, {order = #NHWC}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustMemPermutesAroundMultiplyReuseSiblingMemPermute
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x8x51x4xf16, {order = #NCWH}>, [[INPUT1:%.+]]: tensor<1x8x51x4xf16, {order = #NHWC}>)
func.func @AdjustMemPermutesAroundMultiplyReuseSiblingMemPermute(%arg0: tensor<1x8x51x4xf16, {order = #NCWH}>, %arg1: tensor<1x8x51x4xf16, {order = #NHWC}>) -> (tensor<1x8x51x4xf16, {order = #NHWC}>, tensor<1x8x51x4xf16, {order = #NHWC}>) {
    %reuse = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x8x51x4xf16, {order = #NCWH}> -> tensor<1x8x51x4xf16, {order = #NHWC}>
    %0 = IE.MemPermute(%arg1) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x8x51x4xf16, {order = #NHWC}> -> tensor<1x8x51x4xf16, {order = #NCWH}>
    %1 = IE.Multiply(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x51x4xf16, {order = #NCWH}>, tensor<1x8x51x4xf16, {order = #NCWH}> -> tensor<1x8x51x4xf16, {order = #NCWH}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x8x51x4xf16, {order = #NCWH}> -> tensor<1x8x51x4xf16, {order = #NHWC}>

    return %2, %reuse : tensor<1x8x51x4xf16, {order = #NHWC}>, tensor<1x8x51x4xf16, {order = #NHWC}>

    // CHECK:        [[REUSE:%.+]] = IE.MemPermute([[INPUT0]])
    // CHECK-SAME:       {dst_order = #NHWC, mem_perm = #NWHC}
    // CHECK:        [[MUL:%.+]] = IE.Multiply([[REUSE]], [[INPUT1]])
    // CHECK-SAME:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        return [[MUL]], [[REUSE]] : tensor<1x8x51x4xf16, {order = #NHWC}>, tensor<1x8x51x4xf16, {order = #NHWC}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustMemPermutesAroundMultiplyWithSameInput
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x8x51x4xf16, {order = #NCWH}>)
func.func @NotAdjustMemPermutesAroundMultiplyWithSameInput(%arg0: tensor<1x8x51x4xf16, {order = #NCWH}>) -> tensor<1x8x51x4xf16, {order = #NHWC}> {
    %shared = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x8x51x4xf16, {order = #NCWH}> -> tensor<1x8x51x4xf16, {order = #NHWC}>
    %1 = IE.Multiply(%shared, %shared) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x51x4xf16, {order = #NHWC}>, tensor<1x8x51x4xf16, {order = #NHWC}> -> tensor<1x8x51x4xf16, {order = #NHWC}>

    return %1 : tensor<1x8x51x4xf16, {order = #NHWC}>

    // CHECK:        [[SHARED:%.+]] = IE.MemPermute([[INPUT0]])
    // CHECK-SAME:       {dst_order = #NHWC, mem_perm = #NWHC}
    // CHECK:        [[MUL:%.+]] = IE.Multiply([[SHARED]], [[SHARED]])
    // CHECK-SAME:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        return [[MUL]] : tensor<1x8x51x4xf16, {order = #NHWC}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustMemPermutesAroundMultiplyWithSameInputReuseSiblingMemPermute
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x8x51x4xf16, {order = #NCWH}>)
func.func @AdjustMemPermutesAroundMultiplyWithSameInputReuseSiblingMemPermute(%arg0: tensor<1x8x51x4xf16, {order = #NCWH}>) -> (tensor<1x8x51x4xf16, {order = #NHWC}>, tensor<1x8x51x4xf16, {order = #NHWC}>) {
    %reuse = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x8x51x4xf16, {order = #NCWH}> -> tensor<1x8x51x4xf16, {order = #NHWC}>
    %1 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x51x4xf16, {order = #NCWH}>, tensor<1x8x51x4xf16, {order = #NCWH}> -> tensor<1x8x51x4xf16, {order = #NCWH}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x8x51x4xf16, {order = #NCWH}> -> tensor<1x8x51x4xf16, {order = #NHWC}>

    return %2, %reuse : tensor<1x8x51x4xf16, {order = #NHWC}>, tensor<1x8x51x4xf16, {order = #NHWC}>

    // CHECK:        [[REUSE:%.+]] = IE.MemPermute([[INPUT0]])
    // CHECK-SAME:       {dst_order = #NHWC, mem_perm = #NWHC}
    // CHECK:        [[MUL:%.+]] = IE.Multiply([[REUSE]], [[REUSE]])
    // CHECK-SAME:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        return [[MUL]], [[REUSE]] : tensor<1x8x51x4xf16, {order = #NHWC}>, tensor<1x8x51x4xf16, {order = #NHWC}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustMemPermutesAroundMultiplyReuseLateSiblingMemPermute
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x8x51x4xf16, {order = #NCWH}>, [[INPUT1:%.+]]: tensor<1x8x51x4xf16, {order = #NHWC}>)
func.func @AdjustMemPermutesAroundMultiplyReuseLateSiblingMemPermute(%arg0: tensor<1x8x51x4xf16, {order = #NCWH}>, %arg1: tensor<1x8x51x4xf16, {order = #NHWC}>) -> (tensor<1x8x51x4xf16, {order = #NHWC}>, tensor<1x8x51x4xf16, {order = #NHWC}>) {
    %0 = IE.MemPermute(%arg1) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x8x51x4xf16, {order = #NHWC}> -> tensor<1x8x51x4xf16, {order = #NCWH}>
    %1 = IE.Multiply(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x51x4xf16, {order = #NCWH}>, tensor<1x8x51x4xf16, {order = #NCWH}> -> tensor<1x8x51x4xf16, {order = #NCWH}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x8x51x4xf16, {order = #NCWH}> -> tensor<1x8x51x4xf16, {order = #NHWC}>
    %reuse = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x8x51x4xf16, {order = #NCWH}> -> tensor<1x8x51x4xf16, {order = #NHWC}>

    return %2, %reuse : tensor<1x8x51x4xf16, {order = #NHWC}>, tensor<1x8x51x4xf16, {order = #NHWC}>

    // CHECK:        [[REUSE:%.+]] = IE.MemPermute([[INPUT0]])
    // CHECK-SAME:       {dst_order = #NHWC, mem_perm = #NWHC}
    // CHECK:        [[MUL:%.+]] = IE.Multiply([[REUSE]], [[INPUT1]])
    // CHECK-SAME:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        return [[MUL]], [[REUSE]] : tensor<1x8x51x4xf16, {order = #NHWC}>, tensor<1x8x51x4xf16, {order = #NHWC}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @AdjustMemPermutesAroundMultiplyReuseSiblingMemPermuteDifferentOrders
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x8x51x4xf16, {order = #NCWH}>, [[INPUT1:%.+]]: tensor<1x8x51x4xf16, {order = #NHWC}>)
func.func @AdjustMemPermutesAroundMultiplyReuseSiblingMemPermuteDifferentOrders(%arg0: tensor<1x8x51x4xf16, {order = #NCWH}>, %arg1: tensor<1x8x51x4xf16, {order = #NHWC}>) -> (tensor<1x8x51x4xf16, {order = #NHWC}>, tensor<1x51x4x8xf16>) {
    %reuse = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x8x51x4xf16, {order = #NCWH}> -> tensor<1x51x4x8xf16>
    %0 = IE.MemPermute(%arg1) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x8x51x4xf16, {order = #NHWC}> -> tensor<1x8x51x4xf16, {order = #NCWH}>
    %1 = IE.Multiply(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x51x4xf16, {order = #NCWH}>, tensor<1x8x51x4xf16, {order = #NCWH}> -> tensor<1x8x51x4xf16, {order = #NCWH}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x8x51x4xf16, {order = #NCWH}> -> tensor<1x8x51x4xf16, {order = #NHWC}>

    return %2, %reuse : tensor<1x8x51x4xf16, {order = #NHWC}>, tensor<1x51x4x8xf16>

    // CHECK:        [[REUSE:%.+]] = IE.MemPermute([[INPUT0]])
    // CHECK-SAME:       {dst_order = #NCHW, mem_perm = #NWHC}
    // CHECK:        [[REUSE_CAST:%.+]] = IE.PermuteCast([[REUSE]])
    // CHECK-SAME:       {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK:        [[MUL:%.+]] = IE.Multiply([[REUSE_CAST]], [[INPUT1]])
    // CHECK-SAME:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        return [[MUL]], [[REUSE]] : tensor<1x8x51x4xf16, {order = #NHWC}>, tensor<1x51x4x8xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustMemPermutesAroundAddReuseSiblingMemPermute
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x128x32xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x16x128x32xf16, {order = #NCWH}>)
func.func @AdjustMemPermutesAroundAddReuseSiblingMemPermute(%arg0: tensor<1x16x128x32xf16, {order = #NHWC}>, %arg1: tensor<1x16x128x32xf16, {order = #NCWH}>) -> (tensor<1x16x128x32xf16, {order = #NCWH}>, tensor<1x16x128x32xf16, {order = #NCWH}>) {
    %reuse = IE.MemPermute(%arg0) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x16x128x32xf16, {order = #NHWC}> -> tensor<1x16x128x32xf16, {order = #NCWH}>
    %0 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x16x128x32xf16, {order = #NCWH}> -> tensor<1x16x128x32xf16, {order = #NHWC}>
    %1 = IE.Add(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x128x32xf16, {order = #NHWC}>, tensor<1x16x128x32xf16, {order = #NHWC}> -> tensor<1x16x128x32xf16, {order = #NHWC}>
    %2 = IE.MemPermute(%1) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x16x128x32xf16, {order = #NHWC}> -> tensor<1x16x128x32xf16, {order = #NCWH}>

    return %2, %reuse : tensor<1x16x128x32xf16, {order = #NCWH}>, tensor<1x16x128x32xf16, {order = #NCWH}>

    // CHECK:        [[REUSE:%.+]] = IE.MemPermute([[INPUT0]])
    // CHECK-SAME:       {dst_order = #NCWH, mem_perm = #NWHC}
    // CHECK:        [[REUSE_CAST:%.+]] = IE.PermuteCast([[REUSE]])
    // CHECK-SAME:       {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK:        [[INPUT1_CAST:%.+]] = IE.PermuteCast([[INPUT1]])
    // CHECK-SAME:       {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK:        [[ADD:%.+]] = IE.Add([[REUSE_CAST]], [[INPUT1_CAST]])
    // CHECK-SAME:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        [[OUT:%.+]] = IE.PermuteCast([[ADD]])
    // CHECK-SAME:       {dst_order = #NCWH, mem_perm = #NCHW}
    // CHECK:        return [[OUT]], [[REUSE]] : tensor<1x16x128x32xf16, {order = #NCWH}>, tensor<1x16x128x32xf16, {order = #NCWH}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @AdjustMemPermutesAroundAddReuseSiblingMemPermuteDifferentOrders
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x128x32xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x16x128x32xf16, {order = #NCWH}>)
func.func @AdjustMemPermutesAroundAddReuseSiblingMemPermuteDifferentOrders(%arg0: tensor<1x16x128x32xf16, {order = #NHWC}>, %arg1: tensor<1x16x128x32xf16, {order = #NCWH}>) -> (tensor<1x16x128x32xf16, {order = #NCWH}>, tensor<1x16x32x128xf16>) {
    %reuse = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x16x128x32xf16, {order = #NHWC}> -> tensor<1x16x32x128xf16>
    %0 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x16x128x32xf16, {order = #NCWH}> -> tensor<1x16x128x32xf16, {order = #NHWC}>
    %1 = IE.Add(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x128x32xf16, {order = #NHWC}>, tensor<1x16x128x32xf16, {order = #NHWC}> -> tensor<1x16x128x32xf16, {order = #NHWC}>
    %2 = IE.MemPermute(%1) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x16x128x32xf16, {order = #NHWC}> -> tensor<1x16x128x32xf16, {order = #NCWH}>

    return %2, %reuse : tensor<1x16x128x32xf16, {order = #NCWH}>, tensor<1x16x32x128xf16>

    // CHECK:        [[REUSE:%.+]] = IE.MemPermute([[INPUT0]])
    // CHECK-SAME:       {dst_order = #NCHW, mem_perm = #NWHC}
    // CHECK:        [[REUSE_TO_NHWC:%.+]] = IE.PermuteCast([[REUSE]])
    // CHECK-SAME:       {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK:        [[INPUT1_CAST:%.+]] = IE.PermuteCast([[INPUT1]])
    // CHECK-SAME:       {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK:        [[ADD:%.+]] = IE.Add([[REUSE_TO_NHWC]], [[INPUT1_CAST]])
    // CHECK-SAME:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        [[OUT:%.+]] = IE.PermuteCast([[ADD]])
    // CHECK-SAME:       {dst_order = #NCWH, mem_perm = #NCHW}
    // CHECK:        return [[OUT]], [[REUSE]] : tensor<1x16x128x32xf16, {order = #NCWH}>, tensor<1x16x32x128xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @AdjustMemPermutesAroundMulWhenInputAddWithTwo1DInputs
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x128x1x1xf16>, [[INPUT1:%.+]]: tensor<1x1x1x64xf16>, [[INPUT2:%.+]]: tensor<1x1x1x64xf16>)
func.func @AdjustMemPermutesAroundMulWhenInputAddWithTwo1DInputs(%arg0: tensor<1x128x1x1xf16>, %arg1: tensor<1x1x1x64xf16>, %arg2: tensor<1x1x1x64xf16>) -> tensor<1x64x128x1xf16> {
    %add1 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x1xf16>, tensor<1x1x1x64xf16> -> tensor<1x128x1x64xf16>
    %pc1 = IE.PermuteCast(%add1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x128x1x64xf16> -> tensor<1x64x128x1xf16, {order = #NHWC}>
    %add2 = IE.Add(%arg0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x1xf16>, tensor<1x1x1x64xf16> -> tensor<1x128x1x64xf16>
    %pc2 = IE.PermuteCast(%add2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x128x1x64xf16> -> tensor<1x64x128x1xf16, {order = #NHWC}>
    %mul = IE.Multiply(%pc1, %pc2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x128x1xf16, {order = #NHWC}>, tensor<1x64x128x1xf16, {order = #NHWC}> -> tensor<1x64x128x1xf16, {order = #NHWC}>
    %result = IE.MemPermute(%mul) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x64x128x1xf16, {order = #NHWC}> -> tensor<1x64x128x1xf16>
    return %result : tensor<1x64x128x1xf16>

    // CHECK-NOT:    IE.MemPermute
    // CHECK:        [[PC0:%.+]] = IE.PermuteCast([[INPUT0]])
    // CHECK-SAME:       {dst_order = #NWCH, mem_perm = #NWCH}
    // CHECK:        [[PC1:%.+]] = IE.PermuteCast([[INPUT1]])
    // CHECK-SAME:       {dst_order = #NWCH, mem_perm = #NWCH}
    // CHECK:        [[ADD0:%.+]] = IE.Add([[PC0]], [[PC1]])
    // CHECK-SAME:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        [[PC2:%.+]] = IE.PermuteCast([[INPUT0]])
    // CHECK-SAME:       {dst_order = #NWCH, mem_perm = #NWCH}
    // CHECK:        [[PC3:%.+]] = IE.PermuteCast([[INPUT2]])
    // CHECK-SAME:       {dst_order = #NWCH, mem_perm = #NWCH}
    // CHECK:        [[ADD1:%.+]] = IE.Add([[PC2]], [[PC3]])
    // CHECK-SAME:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        [[PC4:%.+]] = IE.PermuteCast([[ADD0]])
    // CHECK-SAME:       {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK:        [[PC5:%.+]] = IE.PermuteCast([[ADD1]])
    // CHECK-SAME:       {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK:        [[MUL:%.+]] = IE.Multiply([[PC4]], [[PC5]])
    // CHECK-SAME:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        return [[MUL]] : tensor<1x64x128x1xf16>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @AdjustConcatLayoutForMultiUser
// CHECK-SAME:    [[INPUT_0:%[^:]+]]: tensor<1x16x4x1xf16>, [[INPUT_1:%[^:]+]]: tensor<1x16x1x1xf16>
func.func @AdjustConcatLayoutForMultiUser(%input0: tensor<1x16x4x1xf16>, %input1: tensor<1x16x1x1xf16>) -> (tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x16x4x1xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x5x1xf16>
    %permute = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16, {order = #NHWC}>
    %slice = IE.Slice %concat [0, 0, 0, 0] [1, 16, 4, 1] : tensor<1x16x5x1xf16> to tensor<1x16x4x1xf16>
    return %permute, %slice : tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x16x4x1xf16>

    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x16x4x1xf16> -> tensor<1x16x4x1xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[MEM_PERMUTE]], [[PERMUTE_CAST]]) {per_axis = #IE.Concat<axis = 2 : i64>}
    // CHECK-SAME:      : tensor<1x16x4x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x5x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 16, 4, 1]
    // CHECK-SAME:      : tensor<1x16x5x1xf16, {order = #NHWC}> to tensor<1x16x4x1xf16, {order = #NHWC}>
    // CHECK:       [[INV_PERMUTE:%.+]] = IE.MemPermute([[SLICE]]) {dst_order = #NCHW, mem_perm = #NWCH}
    // CHECK-SAME:      : tensor<1x16x4x1xf16, {order = #NHWC}> -> tensor<1x16x4x1xf16>
    // CHECK:       return [[CONCAT]], [[INV_PERMUTE]]
    // CHECK-SAME:      : tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x16x4x1xf16>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @AdjustConcatLayoutWithRemappedSlice
// CHECK-SAME:    [[INPUT_0:%[^:]+]]: tensor<1x16x4x1xf16>, [[INPUT_1:%[^:]+]]: tensor<1x16x1x1xf16>
func.func @AdjustConcatLayoutWithRemappedSlice(%input0: tensor<1x16x4x1xf16>, %input1: tensor<1x16x1x1xf16>) -> (tensor<1x5x16x1xf16>, tensor<1x5x16x1xf16>, tensor<1x16x3x1xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x5x1xf16>
    %permute0 = IE.MemPermute(%concat) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x16x5x1xf16> -> tensor<1x5x16x1xf16>
    %permute1 = IE.MemPermute(%concat) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x16x5x1xf16> -> tensor<1x5x16x1xf16>
    %slice = IE.Slice %concat [0, 0, 1, 0] [1, 16, 3, 1] : tensor<1x16x5x1xf16> to tensor<1x16x3x1xf16>
    return %permute0, %permute1, %slice : tensor<1x5x16x1xf16>, tensor<1x5x16x1xf16>, tensor<1x16x3x1xf16>

    // CHECK:       [[INPUT_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]])
    // CHECK:       [[INPUT_CAST:%.+]] = IE.PermuteCast([[INPUT_1]])
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_PERMUTE]], [[INPUT_CAST]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:      -> tensor<1x5x16x1xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]] [0, 1, 0, 0] [1, 3, 16, 1]
    // CHECK-SAME:      : tensor<1x5x16x1xf16> to tensor<1x3x16x1xf16>
    // CHECK:       [[INV_PERMUTE:%.+]] = IE.MemPermute([[SLICE]]) {dst_order = #NCHW, mem_perm = #NHCW}
    // CHECK-SAME:      : tensor<1x3x16x1xf16> -> tensor<1x16x3x1xf16>
    // CHECK:       return [[CONCAT]], [[CONCAT]], [[INV_PERMUTE]]
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustConcatWithUnsupportedUser
// CHECK-SAME:    [[INPUT_0:%[^:]+]]: tensor<1x16x4x1xf16>, [[INPUT_1:%[^:]+]]: tensor<1x16x1x1xf16>, [[INPUT_2:%[^:]+]]: tensor<1x16x5x1xf16>
func.func @NotAdjustConcatWithUnsupportedUser(%input0: tensor<1x16x4x1xf16>, %input1: tensor<1x16x1x1xf16>, %input2: tensor<1x16x5x1xf16>) -> (tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x16x5x1xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x5x1xf16>
    %permute = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16, {order = #NHWC}>
    %add = IE.Add(%concat, %input2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x5x1xf16>, tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16>
    return %permute, %add : tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x16x5x1xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]])
    // CHECK:       [[PERMUTE:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[CONCAT]], [[INPUT_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x16x5x1xf16>, tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16>
    // CHECK:       return [[PERMUTE]], [[ADD]]
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NotAdjustConcatWithDiffMemPerm
// CHECK-SAME:    [[INPUT_0:%[^:]+]]: tensor<1x16x4x3xf16>, [[INPUT_1:%[^:]+]]: tensor<1x16x1x3xf16>
func.func @NotAdjustConcatWithDiffMemPerm(%input0: tensor<1x16x4x3xf16>, %input1: tensor<1x16x1x3xf16>) -> (tensor<1x16x5x3xf16, {order = #NHWC}>, tensor<1x3x16x5xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x3xf16>, tensor<1x16x1x3xf16> -> tensor<1x16x5x3xf16>
    %permute1 = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x5x3xf16> -> tensor<1x16x5x3xf16, {order = #NHWC}>
    %permute2 = IE.MemPermute(%concat) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x5x3xf16> -> tensor<1x3x16x5xf16>
    return %permute1, %permute2 : tensor<1x16x5x3xf16, {order = #NHWC}>, tensor<1x3x16x5xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]])
    // CHECK:       [[PERMUTE_0:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x16x5x3xf16> -> tensor<1x16x5x3xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_1:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NCHW, mem_perm = #NWCH}
    // CHECK-SAME:      : tensor<1x16x5x3xf16> -> tensor<1x3x16x5xf16>
    // CHECK:       return [[PERMUTE_0]], [[PERMUTE_1]]
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Slice on outermost non-trivial dim (C) in NCHW: contiguous access, no benefit from transform
// CHECK-LABEL: @NotAdjustConcatWithContiguousSlice
// CHECK-SAME:    [[INPUT_0:%[^:]+]]: tensor<1x4x1x1xf16>, [[INPUT_1:%[^:]+]]: tensor<1x1x1x1xf16>
func.func @NotAdjustConcatWithContiguousSlice(%input0: tensor<1x4x1x1xf16>, %input1: tensor<1x1x1x1xf16>) -> (tensor<1x5x1x1xf16, {order = #NHWC}>, tensor<1x4x1x1xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]]} : tensor<1x4x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x5x1x1xf16>
    %permute = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x5x1x1xf16> -> tensor<1x5x1x1xf16, {order = #NHWC}>
    %slice = IE.Slice %concat [0, 0, 0, 0] [1, 4, 1, 1] : tensor<1x5x1x1xf16> to tensor<1x4x1x1xf16>
    return %permute, %slice : tensor<1x5x1x1xf16, {order = #NHWC}>, tensor<1x4x1x1xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]])
    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x5x1x1xf16> -> tensor<1x5x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 4, 1, 1]
    // CHECK-SAME:      : tensor<1x5x1x1xf16> to tensor<1x4x1x1xf16>
    // CHECK:       return [[PERMUTE_CAST]], [[SLICE]]
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Concat output MemPermute is trivial (only H>1), so there is nothing to optimize.
// CHECK-LABEL: @NotAdjustConcatWithTrivialPermute
// CHECK-SAME:    [[INPUT_0:%[^:]+]]: tensor<1x1x4x1xf16>, [[INPUT_1:%[^:]+]]: tensor<1x1x1x1xf16>
func.func @NotAdjustConcatWithTrivialPermute(%input0: tensor<1x1x4x1xf16>, %input1: tensor<1x1x1x1xf16>) -> (tensor<1x1x5x1xf16, {order = #NHWC}>, tensor<1x1x4x1xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x1x4x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x5x1xf16>
    %permute = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x5x1xf16> -> tensor<1x1x5x1xf16, {order = #NHWC}>
    %slice = IE.Slice %concat [0, 0, 0, 0] [1, 1, 4, 1] : tensor<1x1x5x1xf16> to tensor<1x1x4x1xf16>
    return %permute, %slice : tensor<1x1x5x1xf16, {order = #NHWC}>, tensor<1x1x4x1xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]])
    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x1x5x1xf16> -> tensor<1x1x5x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 1, 4, 1]
    // CHECK-SAME:      : tensor<1x1x5x1xf16> to tensor<1x1x4x1xf16>
    // CHECK:       return [[PERMUTE_CAST]], [[SLICE]]
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// All inputs are non-trivial (C>1 and H>1 for both), so no MemPermute folds to PermuteCast.
// CHECK-LABEL: @NotAdjustConcatNoTrivialInput
// CHECK-SAME:    [[INPUT_0:%[^:]+]]: tensor<1x16x4x1xf16>, [[INPUT_1:%[^:]+]]: tensor<1x16x2x1xf16>
func.func @NotAdjustConcatNoTrivialInput(%input0: tensor<1x16x4x1xf16>, %input1: tensor<1x16x2x1xf16>) -> (tensor<1x16x6x1xf16, {order = #NHWC}>, tensor<1x16x4x1xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x1xf16>, tensor<1x16x2x1xf16> -> tensor<1x16x6x1xf16>
    %permute = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x6x1xf16> -> tensor<1x16x6x1xf16, {order = #NHWC}>
    %slice = IE.Slice %concat [0, 0, 0, 0] [1, 16, 4, 1] : tensor<1x16x6x1xf16> to tensor<1x16x4x1xf16>
    return %permute, %slice : tensor<1x16x6x1xf16, {order = #NHWC}>, tensor<1x16x4x1xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]])
    // CHECK:       [[PERMUTE:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x16x6x1xf16> -> tensor<1x16x6x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 16, 4, 1]
    // CHECK-SAME:      : tensor<1x16x6x1xf16> to tensor<1x16x4x1xf16>
    // CHECK:       return [[PERMUTE]], [[SLICE]]
}

// -----
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// New DMA cost exceeds eliminated volume because input0 and the post-Slice inv_MemPermute both need real DMA.
// Input0: 1x16x4x1=64 (non-trivial, DMA cost), Input1: 1x16x1x1=16 (trivial, cost 0)
// Eliminated: 1x16x5x1=80, inv_MemPermute: 1x16x4x1=64 (non-trivial)
// newDmaCost = 64 + 64 = 128 >= 80, so reject.
// CHECK-LABEL: @NotAdjustConcatDmaCostTooHigh
// CHECK-SAME:    [[INPUT_0:%[^:]+]]: tensor<1x16x4x1xf16>, [[INPUT_1:%[^:]+]]: tensor<1x16x1x1xf16>
func.func @NotAdjustConcatDmaCostTooHigh(%input0: tensor<1x16x4x1xf16>, %input1: tensor<1x16x1x1xf16>) -> (tensor<1x5x16x1xf16>, tensor<1x16x4x1xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x5x1xf16>
    %permute = IE.MemPermute(%concat) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = #NHCW} : tensor<1x16x5x1xf16> -> tensor<1x5x16x1xf16>
    %slice = IE.Slice %concat [0, 0, 0, 0] [1, 16, 4, 1] : tensor<1x16x5x1xf16> to tensor<1x16x4x1xf16>
    return %permute, %slice : tensor<1x5x16x1xf16>, tensor<1x16x4x1xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]])
    // CHECK:       [[PERMUTE:%.+]] = IE.MemPermute([[CONCAT]])
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 16, 4, 1]
    // CHECK-SAME:      : tensor<1x16x5x1xf16> to tensor<1x16x4x1xf16>
    // CHECK:       return [[PERMUTE]], [[SLICE]]
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Concat on C axis in NCHW is already flat because C is the outermost non-batch dimension.
// CHECK-LABEL: @NotAdjustConcatAlreadyFlat
// CHECK-SAME:    [[INPUT_0:%[^:]+]]: tensor<1x4x16x1xf16>, [[INPUT_1:%[^:]+]]: tensor<1x1x16x1xf16>
func.func @NotAdjustConcatAlreadyFlat(%input0: tensor<1x4x16x1xf16>, %input1: tensor<1x1x16x1xf16>) -> (tensor<1x5x16x1xf16, {order = #NHWC}>, tensor<1x4x16x1xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]]} : tensor<1x4x16x1xf16>, tensor<1x1x16x1xf16> -> tensor<1x5x16x1xf16>
    %permute = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x5x16x1xf16> -> tensor<1x5x16x1xf16, {order = #NHWC}>
    %slice = IE.Slice %concat [0, 0, 0, 0] [1, 4, 16, 1] : tensor<1x5x16x1xf16> to tensor<1x4x16x1xf16>
    return %permute, %slice : tensor<1x5x16x1xf16, {order = #NHWC}>, tensor<1x4x16x1xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]])
    // CHECK:       [[PERMUTE:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x5x16x1xf16> -> tensor<1x5x16x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 4, 16, 1]
    // CHECK-SAME:      : tensor<1x5x16x1xf16> to tensor<1x4x16x1xf16>
    // CHECK:       return [[PERMUTE]], [[SLICE]]
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// N=2 keeps concat strided after transform because H moves to memPos=1 in NHWC but N is non-trivial.
// CHECK-LABEL: @NotAdjustConcatStillStrided
// CHECK-SAME:    [[INPUT_0:%[^:]+]]: tensor<2x16x4x1xf16>, [[INPUT_1:%[^:]+]]: tensor<2x16x1x1xf16>
func.func @NotAdjustConcatStillStrided(%input0: tensor<2x16x4x1xf16>, %input1: tensor<2x16x1x1xf16>) -> (tensor<2x16x5x1xf16, {order = #NHWC}>, tensor<2x16x4x1xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<2x16x4x1xf16>, tensor<2x16x1x1xf16> -> tensor<2x16x5x1xf16>
    %permute = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<2x16x5x1xf16> -> tensor<2x16x5x1xf16, {order = #NHWC}>
    %slice = IE.Slice %concat [0, 0, 0, 0] [2, 16, 4, 1] : tensor<2x16x5x1xf16> to tensor<2x16x4x1xf16>
    return %permute, %slice : tensor<2x16x5x1xf16, {order = #NHWC}>, tensor<2x16x4x1xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]])
    // CHECK:       [[PERMUTE:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<2x16x5x1xf16> -> tensor<2x16x5x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [2, 16, 4, 1]
    // CHECK-SAME:      : tensor<2x16x5x1xf16> to tensor<2x16x4x1xf16>
    // CHECK:       return [[PERMUTE]], [[SLICE]]
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Inputs differ on both C and H axes, so this is not a single-axis concat.
// CHECK-LABEL: @NotAdjustConcatMultiAxis
// CHECK-SAME:    [[INPUT_0:%[^:]+]]: tensor<1x8x4x1xf16>, [[INPUT_1:%[^:]+]]: tensor<1x16x1x1xf16>
func.func @NotAdjustConcatMultiAxis(%input0: tensor<1x8x4x1xf16>, %input1: tensor<1x16x1x1xf16>) -> (tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x16x4x1xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x8x4x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x5x1xf16>
    %permute = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16, {order = #NHWC}>
    %slice = IE.Slice %concat [0, 0, 0, 0] [1, 16, 4, 1] : tensor<1x16x5x1xf16> to tensor<1x16x4x1xf16>
    return %permute, %slice : tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x16x4x1xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]])
    // CHECK:       [[PERMUTE:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 16, 4, 1]
    // CHECK-SAME:      : tensor<1x16x5x1xf16> to tensor<1x16x4x1xf16>
    // CHECK:       return [[PERMUTE]], [[SLICE]]
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustConcatWithGap
func.func @NotAdjustConcatWithGap(%input0: tensor<1x16x4x1xf16>, %input1: tensor<1x16x1x1xf16>) -> (tensor<1x16x6x1xf16, {order = #NHWC}>, tensor<1x16x4x1xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]]} : tensor<1x16x4x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x6x1xf16>
    %permute = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x6x1xf16> -> tensor<1x16x6x1xf16, {order = #NHWC}>
    %slice = IE.Slice %concat [0, 0, 0, 0] [1, 16, 4, 1] : tensor<1x16x6x1xf16> to tensor<1x16x4x1xf16>
    return %permute, %slice : tensor<1x16x6x1xf16, {order = #NHWC}>, tensor<1x16x4x1xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat
    // CHECK-SAME:      {static_offsets =
    // CHECK-SAME:      -> tensor<1x16x6x1xf16>
    // CHECK:       [[PERMUTE:%.+]] = IE.MemPermute([[CONCAT]])
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]]
    // CHECK:       return [[PERMUTE]], [[SLICE]]
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustConcatWithDifferentDstOrder
func.func @NotAdjustConcatWithDifferentDstOrder(%input0: tensor<1x16x4x1xf16>, %input1: tensor<1x16x1x1xf16>) -> (tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x5x1x16xf16>, tensor<1x16x4x1xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x5x1xf16>
    %permute0 = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16, {order = #NHWC}>
    %permute1 = IE.MemPermute(%concat) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x5x1xf16> -> tensor<1x5x1x16xf16>
    %slice = IE.Slice %concat [0, 0, 0, 0] [1, 16, 4, 1] : tensor<1x16x5x1xf16> to tensor<1x16x4x1xf16>
    return %permute0, %permute1, %slice : tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x5x1x16xf16>, tensor<1x16x4x1xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat
    // CHECK:       [[PERMUTE_0:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       [[PERMUTE_1:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NCHW, mem_perm = #NHWC}
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]]
    // CHECK:       return [[PERMUTE_0]], [[PERMUTE_1]], [[SLICE]]
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustConcatWithTwoPermuteUsers
// CHECK-SAME:    [[INPUT_0:%[^:]+]]: tensor<1x16x4x1xf16>, [[INPUT_1:%[^:]+]]: tensor<1x16x1x1xf16>
func.func @AdjustConcatWithTwoPermuteUsers(%input0: tensor<1x16x4x1xf16>, %input1: tensor<1x16x1x1xf16>) -> (tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x16x4x1xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x5x1xf16>
    %permute0 = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16, {order = #NHWC}>
    %permute1 = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16, {order = #NHWC}>
    %slice = IE.Slice %concat [0, 0, 0, 0] [1, 16, 4, 1] : tensor<1x16x5x1xf16> to tensor<1x16x4x1xf16>
    return %permute0, %permute1, %slice : tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x16x4x1xf16>

    // CHECK:       [[INPUT_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]])
    // CHECK:       [[INPUT_CAST:%.+]] = IE.PermuteCast([[INPUT_1]])
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_PERMUTE]], [[INPUT_CAST]])
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]]
    // CHECK:       [[INV_PERMUTE:%.+]] = IE.MemPermute([[SLICE]])
    // CHECK:       return [[CONCAT]], [[CONCAT]], [[INV_PERMUTE]]
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @AdjustConcatWithFusedInputPermute
// CHECK-SAME:    [[INPUT_0:%[^:]+]]: tensor<1x16x4x1xf16, {order = #NHWC}>, [[INPUT_1:%[^:]+]]: tensor<1x16x1x1xf16>
func.func @AdjustConcatWithFusedInputPermute(%input0: tensor<1x16x4x1xf16, {order = #NHWC}>, %input1: tensor<1x16x1x1xf16>) -> (tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x16x4x1xf16>) {
    %inputPermute = IE.MemPermute(%input0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x4x1xf16, {order = #NHWC}> -> tensor<1x16x4x1xf16>
    %concat = IE.Concat(%inputPermute, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x5x1xf16>
    %outputPermute = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16, {order = #NHWC}>
    %slice = IE.Slice %concat [0, 0, 0, 0] [1, 16, 4, 1] : tensor<1x16x5x1xf16> to tensor<1x16x4x1xf16>
    return %outputPermute, %slice : tensor<1x16x5x1xf16, {order = #NHWC}>, tensor<1x16x4x1xf16>

    // CHECK:       [[INPUT_CAST:%.+]] = IE.PermuteCast([[INPUT_1]])
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_CAST]])
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]]
    // CHECK:       [[INV_PERMUTE:%.+]] = IE.MemPermute([[SLICE]])
    // CHECK:       return [[CONCAT]], [[INV_PERMUTE]]
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Original output MemPermutes can be converted to MaxPool, so their eliminated DMA cost is zero.
// CHECK-LABEL: @NotAdjustConcatWhenOutputPermutesConvertToPool
func.func @NotAdjustConcatWhenOutputPermutesConvertToPool(%input0: tensor<1x16x4096x1xf16>, %input1: tensor<1x16x4096x1xf16>) -> (tensor<1x16x8192x1xf16, {order = #NHWC}>, tensor<1x16x8192x1xf16, {order = #NHWC}>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4096, 0]]} : tensor<1x16x4096x1xf16>, tensor<1x16x4096x1xf16> -> tensor<1x16x8192x1xf16>
    %permute0 = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x8192x1xf16> -> tensor<1x16x8192x1xf16, {order = #NHWC}>
    %permute1 = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x8192x1xf16> -> tensor<1x16x8192x1xf16, {order = #NHWC}>
    return %permute0, %permute1 : tensor<1x16x8192x1xf16, {order = #NHWC}>, tensor<1x16x8192x1xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat
    // CHECK:       [[PERMUTE_0:%.+]] = IE.MemPermute([[CONCAT]])
    // CHECK:       [[PERMUTE_1:%.+]] = IE.MemPermute([[CONCAT]])
    // CHECK:       return [[PERMUTE_0]], [[PERMUTE_1]]
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustConcatDynamicShape
func.func @NotAdjustConcatDynamicShape(%input0: tensor<1x16x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 1]> : tensor<4xsi64>, order = #NCHW}>, %input1: tensor<1x16x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1]> : tensor<4xsi64>, order = #NCHW}>)
        -> (tensor<1x16x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 1]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x4x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 1]> : tensor<4xsi64>, order = #NCHW}>) {
    %concat = IE.Concat(%input0, %input1) {per_axis = #IE.Concat<axis = 2 : i64>}
        : tensor<1x16x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 1]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x16x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x16x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 1]> : tensor<4xsi64>, order = #NCHW}>
    %permute = IE.MemPermute(%concat) {dst_order = #NHWC, mem_perm = #NHWC}
        : tensor<1x16x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 1]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x16x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 1]> : tensor<4xsi64>, order = #NHWC}>
    %slice = IE.Slice %concat [0, 0, 0, 0] [1, 16, 4, 1]
        : tensor<1x16x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 1]> : tensor<4xsi64>, order = #NCHW}>
        to tensor<1x16x4x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 1]> : tensor<4xsi64>, order = #NCHW}>
    return %permute, %slice
        : tensor<1x16x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 1]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x4x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat
    // CHECK:       [[PERMUTE:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]]
    // CHECK:       return [[PERMUTE]], [[SLICE]]
}

// -----
#ABC = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#ACB = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// Rank-3 permutations are not eligible for the MaxPool conversion query and use regular DMA cost.
// CHECK-LABEL: @NotAdjustRank3ConcatDmaCostTooHigh
func.func @NotAdjustRank3ConcatDmaCostTooHigh(%input0: tensor<1x16x4xf16>, %input1: tensor<1x16x1xf16>) -> (tensor<1x5x16xf16>, tensor<1x16x4xf16>) {
    %concat = IE.Concat(%input0, %input1) {static_offsets = [[0, 0, 0], [0, 0, 4]]} : tensor<1x16x4xf16>, tensor<1x16x1xf16> -> tensor<1x16x5xf16>
    %permute = IE.MemPermute(%concat) {dst_order = #ABC, mem_perm = #ACB} : tensor<1x16x5xf16> -> tensor<1x5x16xf16>
    %slice = IE.Slice %concat [0, 0, 0] [1, 16, 4] : tensor<1x16x5xf16> to tensor<1x16x4xf16>
    return %permute, %slice : tensor<1x5x16xf16>, tensor<1x16x4xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat
    // CHECK:       [[PERMUTE:%.+]] = IE.MemPermute([[CONCAT]])
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONCAT]]
    // CHECK:       return [[PERMUTE]], [[SLICE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 0.023263396468817019:128>

// CHECK-LABEL: @AdjustMemPermuteAroundIdentityAvgPool
// CHECK-SAME:    ([[INPUT:%[^:]+]]: tensor<1x384x4x145x!qElemType, {order = #NHWC}>)
func.func @AdjustMemPermuteAroundIdentityAvgPool(
        %arg0: tensor<1x384x4x145x!qElemType, {order = #NHWC}>) ->
        (tensor<1x4x145x384x!qElemType>, tensor<1x4x145x384xf16>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} :
        tensor<1x384x4x145x!qElemType, {order = #NHWC}> ->
        tensor<1x4x145x384x!qElemType, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x4x145x384x!qElemType, {order = #NHWC}> -> tensor<1x4x145x384x!qElemType>
    %2 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} :
        tensor<1x4x145x384x!qElemType, {order = #NHWC}> ->
        tensor<1x4x145x384xf16, {order = #NHWC}>
    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x4x145x384xf16, {order = #NHWC}> -> tensor<1x4x145x384xf16>
    return %1, %3 : tensor<1x4x145x384x!qElemType>, tensor<1x4x145x384xf16>

    // CHECK-NOT:    IE.MemPermute
    // CHECK:        [[OUTPUT0:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:       : tensor<1x384x4x145x!qElemType, {order = #NHWC}> -> tensor<1x4x145x384x!qElemType>
    // CHECK:        [[POOL_INPUT:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:       : tensor<1x384x4x145x!qElemType, {order = #NHWC}> -> tensor<1x4x145x384x!qElemType>
    // CHECK:        [[POOL_INPUT_NHWC:%.+]] = IE.PermuteCast([[POOL_INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:       : tensor<1x4x145x384x!qElemType> -> tensor<1x384x4x145x!qElemType, {order = #NHWC}>
    // CHECK:        [[POOL:%.+]] = IE.AvgPool([[POOL_INPUT_NHWC]])
    // CHECK-SAME:       {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:       : tensor<1x384x4x145x!qElemType, {order = #NHWC}> -> tensor<1x384x4x145xf16, {order = #NHWC}>
    // CHECK:        [[OUTPUT1:%.+]] = IE.PermuteCast([[POOL]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:       : tensor<1x384x4x145xf16, {order = #NHWC}> -> tensor<1x4x145x384xf16>
    // CHECK:        return [[OUTPUT0]], [[OUTPUT1]] : tensor<1x4x145x384x!qElemType>, tensor<1x4x145x384xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// The pool input has no permute-like producer, so the selected mem_perm comes from the output
// MemPermute. A sibling already applies that exact mem_perm to the same input, so the pool must reuse
// it instead of materializing a second identical permute.
// CHECK-LABEL: @ReuseSiblingPermuteForIdentityAvgPool
// CHECK-SAME:    ([[INPUT:%[^:]+]]: tensor<1x384x4x144xf16, {order = #NHWC}>)
func.func @ReuseSiblingPermuteForIdentityAvgPool(
        %arg0: tensor<1x384x4x144xf16, {order = #NHWC}>) ->
        (tensor<1x384x4x144xf16>, tensor<1x384x4x144xf16>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x384x4x144xf16, {order = #NHWC}> -> tensor<1x384x4x144xf16>
    %1 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} :
        tensor<1x384x4x144xf16, {order = #NHWC}> -> tensor<1x384x4x144xf16, {order = #NHWC}>
    %2 = IE.MemPermute(%1) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x384x4x144xf16, {order = #NHWC}> -> tensor<1x384x4x144xf16>
    return %0, %2 : tensor<1x384x4x144xf16>, tensor<1x384x4x144xf16>

    // Exactly one MemPermute survives: the sibling, shared by both branches.
    // CHECK:        [[SIBLING:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCHW, mem_perm = #NWCH}
    // CHECK-SAME:       : tensor<1x384x4x144xf16, {order = #NHWC}> -> tensor<1x384x4x144xf16>
    // CHECK-NOT:    IE.MemPermute
    // CHECK:        [[POOL_INPUT:%.+]] = IE.PermuteCast([[SIBLING]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:       : tensor<1x384x4x144xf16> -> tensor<1x144x384x4xf16, {order = #NHWC}>
    // CHECK:        [[POOL:%.+]] = IE.AvgPool([[POOL_INPUT]])
    // CHECK-SAME:       {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:       : tensor<1x144x384x4xf16, {order = #NHWC}> -> tensor<1x144x384x4xf16, {order = #NHWC}>
    // CHECK:        [[OUTPUT:%.+]] = IE.PermuteCast([[POOL]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:       : tensor<1x144x384x4xf16, {order = #NHWC}> -> tensor<1x384x4x144xf16>
    // CHECK:        return [[SIBLING]], [[OUTPUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// The permuted layout would put 145 on the channel dim, which is not a multiple of the f16 channel
// alignment, so the identity pool must stay as is.
// CHECK-LABEL: @NotAdjustIdentityAvgPoolWithUnalignedChannel
// CHECK-SAME:    ([[INPUT:%[^:]+]]: tensor<1x384x4x145xf16, {order = #NHWC}>)
func.func @NotAdjustIdentityAvgPoolWithUnalignedChannel(
        %arg0: tensor<1x384x4x145xf16, {order = #NHWC}>) ->
        (tensor<1x384x4x145xf16>, tensor<1x384x4x145xf16>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x384x4x145xf16, {order = #NHWC}> -> tensor<1x384x4x145xf16>
    %1 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} :
        tensor<1x384x4x145xf16, {order = #NHWC}> -> tensor<1x384x4x145xf16, {order = #NHWC}>
    %2 = IE.MemPermute(%1) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x384x4x145xf16, {order = #NHWC}> -> tensor<1x384x4x145xf16>
    return %0, %2 : tensor<1x384x4x145xf16>, tensor<1x384x4x145xf16>

    // CHECK:        [[SIBLING:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCHW, mem_perm = #NWCH}
    // CHECK:        [[POOL:%.+]] = IE.AvgPool([[INPUT]])
    // CHECK-SAME:       : tensor<1x384x4x145xf16, {order = #NHWC}> -> tensor<1x384x4x145xf16, {order = #NHWC}>
    // CHECK:        [[OUTPUT:%.+]] = IE.MemPermute([[POOL]]) {dst_order = #NCHW, mem_perm = #NWCH}
    // CHECK:        return [[SIBLING]], [[OUTPUT]]
}
