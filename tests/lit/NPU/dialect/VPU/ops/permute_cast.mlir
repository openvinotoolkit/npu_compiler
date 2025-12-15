//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

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
