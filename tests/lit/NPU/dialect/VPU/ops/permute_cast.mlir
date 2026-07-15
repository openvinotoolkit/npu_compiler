//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagatePermuteCastThroughExpand
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<100x1x1x1xf16>)
func.func @PropagatePermuteCastThroughExpand(%arg0 : tensor<100x1x1x1xf16>) -> tensor<112x1x1x1xf16> {
    %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<100x1x1x1xf16> -> tensor<100x1x1x1xf16, {order = #NHWC}>
    %1 = VPU.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16, {order = #NHWC}> -> tensor<112x1x1x1xf16, {order = #NHWC}>
    %2 = VPU.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<112x1x1x1xf16, {order = #NHWC}> -> tensor<112x1x1x1xf16>

    return %2 : tensor<112x1x1x1xf16>

    // CHECK:    [[EXPAND:%.+]] = VPU.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16> -> tensor<112x1x1x1xf16>
    // CHECK:    return [[EXPAND]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MergeParallelPermuteCast
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<112x1x1x1xf16, {order = #NHWC}>)
func.func @MergeParallelPermuteCast(%arg0 : tensor<112x1x1x1xf16, {order = #NHWC}>) -> (tensor<112x1x1x1xf16>, tensor<112x1x1x1xf16>, tensor<100x1x1x1xf16, {order = #NHWC}>) {
    %0 = VPU.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<112x1x1x1xf16, {order = #NHWC}> -> tensor<112x1x1x1xf16>
    %1 = VPU.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<112x1x1x1xf16, {order = #NHWC}> -> tensor<112x1x1x1xf16>
    %2 = VPU.Slice %arg0 [0, 0, 0, 0] [100, 1, 1, 1] : tensor<112x1x1x1xf16, {order = #NHWC}> to tensor<100x1x1x1xf16, {order = #NHWC}>

    return %0, %1, %2 : tensor<112x1x1x1xf16>, tensor<112x1x1x1xf16>, tensor<100x1x1x1xf16, {order = #NHWC}>

    // CHECK:    [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<112x1x1x1xf16, {order = #NHWC}> -> tensor<112x1x1x1xf16>
    // CHECK:    [[SLICE:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [100, 1, 1, 1] : tensor<112x1x1x1xf16, {order = #NHWC}> to tensor<100x1x1x1xf16, {order = #NHWC}>
    // CHECK:    return [[PERMUTE_CAST]], [[PERMUTE_CAST]], [[SLICE]]
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

func.func @PermuteCastMemPermute() -> tensor<1x2xf32, { order = #CN }> {
    %cst = const.Declare tensor<1x2xf32> = dense<[[1.0, 2.0]]> : tensor<1x2xf32>
    %permute_cast = VPU.PermuteCast(%cst) {dst_order = #CN, mem_perm = #CN} : tensor<1x2xf32> -> tensor<1x2xf32, { order = #CN }>
    return %permute_cast : tensor<1x2xf32, { order = #CN }>
}

// CHECK: func.func @PermuteCastMemPermute() -> tensor<1x2xf32, {order = #CN}> {
// CHECK:    [[CST:%.+]] = const.Declare tensor<1x2xf32, {order = #CN}> = dense<{{\[\[}}1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32>, [#const.MemPermute<#CN, #CN>]
// CHECK:    return [[CST]] : tensor<1x2xf32, {order = #CN}>
// CHECK: }

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#CN = affine_map<(d0, d1) -> (d1, d0)>

func.func @PermuteCastNoOp() -> tensor<1x2xf32> {
    %cst = const.Declare tensor<1x2xf32> = dense<[[1.0, 2.0]]> : tensor<1x2xf32>
    %permute_cast_0 = VPU.PermuteCast(%cst) {dst_order = #NC, mem_perm = #NC} : tensor<1x2xf32> -> tensor<1x2xf32>
    return %permute_cast_0 : tensor<1x2xf32>
}

// CHECK: func.func @PermuteCastNoOp() -> tensor<1x2xf32> {
// CHECK:     [[CST:%.+]] = const.Declare tensor<1x2xf32> = dense<{{\[\[}}1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32>
// CHECK:     return [[CST]] : tensor<1x2xf32>
// CHECK: }

// -----

// The earlier PermuteCast must survive; replacing it with the later one would make
// the QuantizeCast (its immediate user) reference a value defined after it — a
// dominance violation.
!qElemType = !quant.uniform<u8:f16, 0.003921568627450980>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeParallelPermuteCastAfterIdentitySliceFold
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x4x4x3xui8>)
func.func @MergeParallelPermuteCastAfterIdentitySliceFold(
        %arg0 : tensor<1x4x4x3xui8>)
        -> (tensor<1x4x4x4x!qElemType, {order = #NHWC}>, tensor<1x3x4x4xui8, {order = #NHWC}>) {
    // Sub-graph 1: identity slice folds to %arg0
    %slice_1 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 4, 4, 3]
             : tensor<1x4x4x3xui8> to tensor<1x4x4x3xui8>
    %permute_cast_earliest = VPU.PermuteCast(%slice_1) {dst_order = #NHWC, mem_perm = #NCHW}
             : tensor<1x4x4x3xui8> -> tensor<1x3x4x4xui8, {order = #NHWC}>
    %2 = VPU.QuantizeCast(%permute_cast_earliest) {dstElemType = !qElemType}
             : tensor<1x3x4x4xui8, {order = #NHWC}> -> tensor<1x3x4x4x!qElemType, {order = #NHWC}>
    %3 = VPU.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
             : tensor<1x3x4x4x!qElemType, {order = #NHWC}> -> tensor<1x4x4x4x!qElemType, {order = #NHWC}>

    // Sub-graph 2: identical identity slice + duplicate PermuteCast
    %slice_2 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 4, 4, 3]
             : tensor<1x4x4x3xui8> to tensor<1x4x4x3xui8>
    %permute_cast_latest = VPU.PermuteCast(%slice_2) {dst_order = #NHWC, mem_perm = #NCHW}
             : tensor<1x4x4x3xui8> -> tensor<1x3x4x4xui8, {order = #NHWC}>

    return %3, %permute_cast_latest
             : tensor<1x4x4x4x!qElemType, {order = #NHWC}>, tensor<1x3x4x4xui8, {order = #NHWC}>

    // CHECK:    [[PC_EARLIEST:%.+]] = VPU.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x4x3xui8> -> tensor<1x3x4x4xui8, {order = #NHWC}>
    // CHECK:    [[QC:%.+]] = VPU.QuantizeCast([[PC_EARLIEST]]) {dstElemType = !qElemType} : tensor<1x3x4x4xui8, {order = #NHWC}> -> tensor<1x3x4x4x!qElemType, {order = #NHWC}>
    // CHECK:    [[EXP:%.+]] = VPU.Expand([[QC]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x4x4x!qElemType, {order = #NHWC}> -> tensor<1x4x4x4x!qElemType, {order = #NHWC}>
    // CHECK:    return [[EXP]], [[PC_EARLIEST]]
}
