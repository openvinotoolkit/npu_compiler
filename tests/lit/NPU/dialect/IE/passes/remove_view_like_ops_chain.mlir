//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --remove-view-like-ops-chain %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @RemovePermuteCastAffineReshapePermuteCastChain
func.func @RemovePermuteCastAffineReshapePermuteCastChain(%arg0: tensor<1x3x1x1xf16, {order = #NHWC}>) -> tensor<1x3x1x1xf16, {order = #NHWC}> {
    %2 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x1x1xf16, {order = #NCHW}>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 3]} :
        tensor<1x3x1x1xf16, {order = #NCHW}> -> tensor<1x1x1x3xf16, {order = #NCHW}>
    %4 = IE.PermuteCast(%3) {dst_order = #NHWC, mem_perm = #NCHW} :
        tensor<1x1x1x3xf16, {order = #NCHW}> -> tensor<1x3x1x1xf16, {order = #NHWC}>

    return %4 : tensor<1x3x1x1xf16, {order = #NHWC}>

    // CHECK: return %arg0 : tensor<1x3x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @RemoveAffineReshapePermuteCastx2Chain
func.func @RemoveAffineReshapePermuteCastx2Chain(%arg0: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
    %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4096, 4096, 1]} :
        tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x4096x1xf16, {order = #NHWC}>

    %2 = IE.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NHCW} :
        tensor<1x4096x4096x1xf16, {order = #NHWC}> -> tensor<1x1x4096x4096xf16, {order = #NCHW}>

    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 1024, 4, 4096]} :
        tensor<1x1x4096x4096xf16, {order = #NCHW}> -> tensor<1x1024x4x4096xf16, {order = #NCHW}>

    %4 = IE.PermuteCast(%3) {dst_order = #NHWC, mem_perm = #NCHW} :
        tensor<1x1024x4x4096xf16, {order = #NCHW}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    return %4 : tensor<1x4096x1024x4xf16, {order = #NHWC}>

    // CHECK: return %arg0 : tensor<1x4096x1024x4xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @RemovePermuteCastAffineReshapePermuteCastSubChain
func.func @RemovePermuteCastAffineReshapePermuteCastSubChain(%arg0: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x4096x1xf16, {order = #NHWC}> {
    %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4096, 4096, 1]} :
        tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x4096x1xf16, {order = #NHWC}>

    %2 = IE.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NHCW} :
        tensor<1x4096x4096x1xf16, {order = #NHWC}> -> tensor<1x1x4096x4096xf16, {order = #NCHW}>

    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 1024, 4, 4096]} :
        tensor<1x1x4096x4096xf16, {order = #NCHW}> -> tensor<1x1024x4x4096xf16, {order = #NCHW}>

    %4 = IE.PermuteCast(%3) {dst_order = #NHWC, mem_perm = #NCHW} :
        tensor<1x1024x4x4096xf16, {order = #NCHW}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4096, 4096, 1]} :
        tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x4096x1xf16, {order = #NHWC}>

    return %5 : tensor<1x4096x4096x1xf16, {order = #NHWC}>

    // CHECK: [[VAL0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4096, 4096, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x4096x1xf16, {order = #NHWC}>
    // CHECK: return [[VAL0]] : tensor<1x4096x4096x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @RemovePermuteCastAffineReshapePermuteCastx2SubChain
func.func @RemovePermuteCastAffineReshapePermuteCastx2SubChain(%arg0: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
    %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4096, 4096, 1]} :
        tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x4096x1xf16, {order = #NHWC}>

    %2 = IE.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NHCW} :
        tensor<1x4096x4096x1xf16, {order = #NHWC}> -> tensor<1x1x4096x4096xf16, {order = #NCHW}>

    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 1024, 4, 4096]} :
        tensor<1x1x4096x4096xf16, {order = #NCHW}> -> tensor<1x1024x4x4096xf16, {order = #NCHW}>

    %4 = IE.PermuteCast(%3) {dst_order = #NHWC, mem_perm = #NCHW} :
        tensor<1x1024x4x4096xf16, {order = #NCHW}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4096, 4096, 1]} :
        tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x4096x1xf16, {order = #NHWC}>

    %6 = IE.PermuteCast(%5) {dst_order = #NCHW, mem_perm = #NHCW} :
        tensor<1x4096x4096x1xf16, {order = #NHWC}> -> tensor<1x1x4096x4096xf16, {order = #NCHW}>

    %7 = IE.AffineReshape(%6) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 1024, 4, 4096]} :
        tensor<1x1x4096x4096xf16, {order = #NCHW}> -> tensor<1x1024x4x4096xf16, {order = #NCHW}>

    %8 = IE.PermuteCast(%7) {dst_order = #NHWC, mem_perm = #NCHW} :
        tensor<1x1024x4x4096xf16, {order = #NCHW}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    return %8 : tensor<1x4096x1024x4xf16, {order = #NHWC}>

    // CHECK: return %arg0 : tensor<1x4096x1024x4xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveShapeCastLayoutCastChain
func.func @RemoveShapeCastLayoutCastChain(%arg0: tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
    %1 = IE.ShapeCast {shape = [1, 1, 4096, 4096]}
        inputs(%arg0 : tensor<1x4096x1024x4xf16, {order = #NHWC}>) -> tensor<1x1x4096x4096xf16, {order = #NHWC}>

    %2 = IE.LayoutCast(%1) {dst_order = #NCHW} :
        tensor<1x1x4096x4096xf16, {order = #NHWC}> -> tensor<1x1x4096x4096xf16, {order = #NCHW}>

    %3 = IE.ShapeCast {shape = [1, 4096, 1024, 4]}
        inputs(%2 : tensor<1x1x4096x4096xf16, {order = #NCHW}>) -> tensor<1x4096x1024x4xf16, {order = #NCHW}>

    %4 = IE.LayoutCast(%3) {dst_order = #NHWC} :
        tensor<1x4096x1024x4xf16, {order = #NCHW}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>

    return %4 : tensor<1x4096x1024x4xf16, {order = #NHWC}>

    // CHECK: return %arg0 : tensor<1x4096x1024x4xf16, {order = #NHWC}>
}
