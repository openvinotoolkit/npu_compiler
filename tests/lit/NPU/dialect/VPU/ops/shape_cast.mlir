//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @Eliminate
func.func @Eliminate(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16> {
    %0 = VPU.ShapeCast {shape = [1, 2, 3, 4]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16>
    return %0 : tensor<1x2x3x4xf16>

    // CHECK-NOT:   VPU.ShapeCast
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @Fuse
func.func @Fuse(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16> {
    %0 = VPU.ShapeCast {shape = [1, 3, 2, 4]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16>
    %1 = VPU.ShapeCast {shape = [1, 2, 3, 4]} inputs(%0 : tensor<1x3x2x4xf16>) -> tensor<1x2x3x4xf16>
    return %1 : tensor<1x2x3x4xf16>

    // CHECK-NOT:   VPU.ShapeCast
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @FuseSequence
func.func @FuseSequence(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x4x3x2xf16> {
    %0 = VPU.ShapeCast {shape = [1, 3, 2, 4]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16>
    %1 = VPU.ShapeCast {shape = [1, 3, 4, 2]} inputs(%0 : tensor<1x3x2x4xf16>) -> tensor<1x3x4x2xf16>
    %2 = VPU.ShapeCast {shape = [1, 4, 3, 2]} inputs(%1 : tensor<1x3x4x2xf16>) -> tensor<1x4x3x2xf16>
    return %2 : tensor<1x4x3x2xf16>

    // CHECK:       [[SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 4, 3, 2]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x4x3x2xf16>
    // CHECK:       return [[SHAPE_CAST]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @FuseShapeCastWithMultipleBranches
func.func @FuseShapeCastWithMultipleBranches(%arg0 : tensor<1x2x3x4xf16>) ->
    (tensor<1x3x2x4xf16>, tensor<1x3x4x2xf16, { order = #NCWH }>)
{
    %0 = VPU.ShapeCast { shape = [1, 3, 2, 4] } inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16>

    %1 = VPU.ShapeCast { shape = [1, 3, 2, 4] } inputs(%0 : tensor<1x3x2x4xf16>) -> tensor<1x3x2x4xf16>
    %2 = VPU.PermuteCast(%0) { dst_order = #NCWH, mem_perm = #NCHW } : tensor<1x3x2x4xf16> -> tensor<1x3x4x2xf16, { order = #NCWH }>

    return %0, %2 : tensor<1x3x2x4xf16>, tensor<1x3x4x2xf16, { order = #NCWH }>

    // CHECK: [[SHAPECAST:%.+]] = VPU.ShapeCast {shape = [1, 3, 2, 4]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16>
    // CHECK: [[PERMUTECAST:%.+]] = VPU.PermuteCast([[SHAPECAST]]) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x3x2x4xf16> -> tensor<1x3x4x2xf16, {order = #NCWH}>
    // CHECK: return [[SHAPECAST]], [[PERMUTECAST]] : tensor<1x3x2x4xf16>, tensor<1x3x4x2xf16, {order = #NCWH}>
}

// -----

// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<1x3x2x4xf16> {
    %0 = const.Declare tensor<1x2x3x4xf16> = dense<1.0> : tensor<1x2x3x4xf16>
    %1 = IE.ShapeCast { shape = [1, 3, 2, 4] } inputs(%0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16>
    return %1 : tensor<1x3x2x4xf16>

    // CHECK:       [[VAR0:%.+]] = const.Declare tensor<1x3x2x4xf16> = dense<1.000000e+00> : tensor<1x2x3x4xf16>, [#const.Reshape<[1, 3, 2, 4]>]
    // CHECK-NOT:   VPU.ShapeCast
    // CHECK:       return [[VAR0]] : tensor<1x3x2x4xf16>
}
