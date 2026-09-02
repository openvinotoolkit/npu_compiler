//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile" --fold-shapeof-gather-dynamic-reshape-to-transpose %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Recover an IE.Transpose(perm = [0, 3, 1, 2]) that OpenVINO decomposed into
// IE.ShapeOf -> IE.Gather(perm_const) -> IE.DynamicReshape when the input is
// fully dynamic. The recovered IE.Transpose implements
// ReifyRankedShapedTypeOpInterface, unblocking HostCompile output-shape prediction.
// CHECK-LABEL: @FoldNHWCTransposeChain
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x?x?x1xf16
func.func @FoldNHWCTransposeChain(%arg0: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}>)
        -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}> {
    %shape = IE.ShapeOf(%arg0) {dstElemType = si64} :
        tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    %perm = const.Declare tensor<4xsi64> = dense<[0, 3, 1, 2]> : tensor<4xsi64>
    %new_shape = IE.Gather(%shape, %perm) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} :
        tensor<4xsi64>, tensor<4xsi64> -> tensor<4xsi64>
    %out = IE.DynamicReshape(%arg0, %new_shape)
        {output_bounds = [1, 1, 2160, 3840],
         output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} :
        tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}>
    return %out : tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-NOT: IE.ShapeOf
    // CHECK-NOT: IE.Gather
    // CHECK-NOT: IE.DynamicReshape
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[ARG0]]) {order_value = {{#.+}}} : tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: return [[TRANSPOSE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Rewrite is skipped when the ShapeOf input differs from the DynamicReshape input,
// which would mean the reshape target shape is unrelated to the data tensor.
// CHECK-LABEL: @DoNotFoldWhenShapeOfIsFromDifferentValue
func.func @DoNotFoldWhenShapeOfIsFromDifferentValue(
        %arg0: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}>,
        %arg1: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}>)
        -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}> {
    %shape = IE.ShapeOf(%arg1) {dstElemType = si64} :
        tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    %perm = const.Declare tensor<4xsi64> = dense<[0, 3, 1, 2]> : tensor<4xsi64>
    %new_shape = IE.Gather(%shape, %perm) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} :
        tensor<4xsi64>, tensor<4xsi64> -> tensor<4xsi64>
    %out = IE.DynamicReshape(%arg0, %new_shape)
        {output_bounds = [1, 1, 2160, 3840],
         output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} :
        tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}>
    return %out : tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: IE.DynamicReshape
    // CHECK-NOT: IE.Transpose
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Rewrite is skipped when the Gather indices are not a valid permutation
// (contains a duplicate entry), so the reshape does not correspond to a transpose.
// CHECK-LABEL: @DoNotFoldWhenIndicesAreNotAPermutation
func.func @DoNotFoldWhenIndicesAreNotAPermutation(%arg0: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}>)
        -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}> {
    %shape = IE.ShapeOf(%arg0) {dstElemType = si64} :
        tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    %perm = const.Declare tensor<4xsi64> = dense<[0, 3, 1, 3]> : tensor<4xsi64>
    %new_shape = IE.Gather(%shape, %perm) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} :
        tensor<4xsi64>, tensor<4xsi64> -> tensor<4xsi64>
    %out = IE.DynamicReshape(%arg0, %new_shape)
        {output_bounds = [1, 1, 2160, 3840],
         output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} :
        tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}>
    return %out : tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: IE.DynamicReshape
    // CHECK-NOT: IE.Transpose
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Rewrite is skipped when the recorded DynamicReshape output_shape/bounds do
// not match the permutation of the input's shape/bounds. This guards against
// rewriting a chain that happens to look like the pattern but semantically
// reshapes to a shape unrelated to a transpose of the input.
// CHECK-LABEL: @DoNotFoldWhenOutputShapeMismatchesPermutation
func.func @DoNotFoldWhenOutputShapeMismatchesPermutation(%arg0: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}>)
        -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 3840, 2160]> : tensor<4xsi64>, order = #NCHW}> {
    %shape = IE.ShapeOf(%arg0) {dstElemType = si64} :
        tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    %perm = const.Declare tensor<4xsi64> = dense<[0, 3, 1, 2]> : tensor<4xsi64>
    %new_shape = IE.Gather(%shape, %perm) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} :
        tensor<4xsi64>, tensor<4xsi64> -> tensor<4xsi64>
    %out = IE.DynamicReshape(%arg0, %new_shape)
        {output_bounds = [1, 1, 3840, 2160],
         output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} :
        tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 1]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64>
        -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 3840, 2160]> : tensor<4xsi64>, order = #NCHW}>
    return %out : tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 3840, 2160]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: IE.DynamicReshape
    // CHECK-NOT: IE.Transpose
}
