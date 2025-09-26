//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
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

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UniquifyShapeCastWithSliceOnEnlargeDim
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x512x360x2xf16, {order = #NHWC}>
func.func @UniquifyShapeCastWithSliceOnEnlargeDim(%arg0: tensor<1x512x360x2xf16, {order = #NHWC}>) -> (tensor<1x32x360x16xf16, {order = #NHWC}>, tensor<1x32x360x16xf16, {order = #NHWC}>) {
    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 512, 360, 1] : tensor<1x512x360x2xf16, {order = #NHWC}> to tensor<1x512x360x1xf16, {order = #NHWC}>
    %1 = VPU.ShapeCast {shape = [1, 32, 360, 16]} inputs(%0 : tensor<1x512x360x1xf16, {order = #NHWC}>) -> tensor<1x32x360x16xf16, {order = #NHWC}>
    %2 = VPU.Slice %arg0 [0, 0, 0, 1] [1, 512, 360, 1] : tensor<1x512x360x2xf16, {order = #NHWC}> to tensor<1x512x360x1xf16, {order = #NHWC}>
    %3 = VPU.ShapeCast {shape = [1, 32, 360, 16]} inputs(%2 : tensor<1x512x360x1xf16, {order = #NHWC}>) -> tensor<1x32x360x16xf16, {order = #NHWC}>
    return %1, %3 : tensor<1x32x360x16xf16, {order = #NHWC}>, tensor<1x32x360x16xf16, {order = #NHWC}>

    // CHECK:       [[SHAPECAST:%.+]] = VPU.ShapeCast {shape = [1, 32, 360, 32]} inputs([[ARG0]] : tensor<1x512x360x2xf16, {order = #NHWC}>) -> tensor<1x32x360x32xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[SHAPECAST]] [0, 0, 0, 0] [1, 32, 360, 16] : tensor<1x32x360x32xf16, {order = #NHWC}> to tensor<1x32x360x16xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[SHAPECAST]] [0, 0, 0, 16] [1, 32, 360, 16] : tensor<1x32x360x32xf16, {order = #NHWC}> to tensor<1x32x360x16xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x32x360x16xf16, {order = #NHWC}>, tensor<1x32x360x16xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @UniquifyShapeCastWithSliceOnShrinkDim
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x512x360x2xf16>
func.func @UniquifyShapeCastWithSliceOnShrinkDim(%arg0: tensor<1x512x360x2xf16>) -> (tensor<1x32x360x16xf16>, tensor<1x32x360x16xf16>) {
    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 256, 360, 2] : tensor<1x512x360x2xf16> to tensor<1x256x360x2xf16>
    %1 = VPU.ShapeCast {shape = [1, 32, 360, 16]} inputs(%0 : tensor<1x256x360x2xf16>) -> tensor<1x32x360x16xf16>
    %2 = VPU.Slice %arg0 [0, 256, 0, 0] [1, 256, 360, 2] : tensor<1x512x360x2xf16> to tensor<1x256x360x2xf16>
    %3 = VPU.ShapeCast {shape = [1, 32, 360, 16]} inputs(%2 : tensor<1x256x360x2xf16>) -> tensor<1x32x360x16xf16>
    return %1, %3 : tensor<1x32x360x16xf16>, tensor<1x32x360x16xf16>

    // CHECK:       [[SHAPECAST:%.+]] = VPU.ShapeCast {shape = [1, 64, 360, 16]} inputs([[ARG0]] : tensor<1x512x360x2xf16>) -> tensor<1x64x360x16xf16>
    // CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[SHAPECAST]] [0, 0, 0, 0] [1, 32, 360, 16] : tensor<1x64x360x16xf16> to tensor<1x32x360x16xf16>
    // CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[SHAPECAST]] [0, 32, 0, 0] [1, 32, 360, 16] : tensor<1x64x360x16xf16> to tensor<1x32x360x16xf16>
    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x32x360x16xf16>, tensor<1x32x360x16xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UniquifyShapeCastWithSliceOnUnChangedDim
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x512x360x2xf16, {order = #NHWC}>
func.func @UniquifyShapeCastWithSliceOnUnChangedDim(%arg0: tensor<1x512x360x2xf16, {order = #NHWC}>) -> (tensor<1x32x180x32xf16, {order = #NHWC}>, tensor<1x32x180x32xf16, {order = #NHWC}>) {
    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 512, 180, 2] : tensor<1x512x360x2xf16, {order = #NHWC}> to tensor<1x512x180x2xf16, {order = #NHWC}>
    %1 = VPU.ShapeCast {shape = [1, 32, 180, 32]} inputs(%0 : tensor<1x512x180x2xf16, {order = #NHWC}>) -> tensor<1x32x180x32xf16, {order = #NHWC}>
    %2 = VPU.Slice %arg0 [0, 0, 180, 0] [1, 512, 180, 2] : tensor<1x512x360x2xf16, {order = #NHWC}> to tensor<1x512x180x2xf16, {order = #NHWC}>
    %3 = VPU.ShapeCast {shape = [1, 32, 180, 32]} inputs(%2 : tensor<1x512x180x2xf16, {order = #NHWC}>) -> tensor<1x32x180x32xf16, {order = #NHWC}>
    return %1, %3 : tensor<1x32x180x32xf16, {order = #NHWC}>, tensor<1x32x180x32xf16, {order = #NHWC}>

    // CHECK:       [[SHAPECAST:%.+]] = VPU.ShapeCast {shape = [1, 32, 360, 32]} inputs([[ARG0]] : tensor<1x512x360x2xf16, {order = #NHWC}>) -> tensor<1x32x360x32xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[SHAPECAST]] [0, 0, 0, 0] [1, 32, 180, 32] : tensor<1x32x360x32xf16, {order = #NHWC}> to tensor<1x32x180x32xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[SHAPECAST]] [0, 0, 180, 0] [1, 32, 180, 32] : tensor<1x32x360x32xf16, {order = #NHWC}> to tensor<1x32x180x32xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x32x180x32xf16, {order = #NHWC}>, tensor<1x32x180x32xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotUniquifyShapeCastWithUnsupportedDim
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x512x360x2xf16, {order = #NHWC}>
func.func @NotUniquifyShapeCastWithUnsupportedDim(%arg0: tensor<1x512x360x2xf16, {order = #NHWC}>) -> (tensor<1x32x360x16xf16, {order = #NHWC}>, tensor<1x32x360x16xf16, {order = #NHWC}>) {
    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 256, 360, 2] : tensor<1x512x360x2xf16, {order = #NHWC}> to tensor<1x256x360x2xf16, {order = #NHWC}>
    %1 = VPU.ShapeCast {shape = [1, 32, 360, 16]} inputs(%0 : tensor<1x256x360x2xf16, {order = #NHWC}>) -> tensor<1x32x360x16xf16, {order = #NHWC}>
    %2 = VPU.Slice %arg0 [0, 256, 0, 0] [1, 256, 360, 2] : tensor<1x512x360x2xf16, {order = #NHWC}> to tensor<1x256x360x2xf16, {order = #NHWC}>
    %3 = VPU.ShapeCast {shape = [1, 32, 360, 16]} inputs(%2 : tensor<1x256x360x2xf16, {order = #NHWC}>) -> tensor<1x32x360x16xf16, {order = #NHWC}>
    return %1, %3 : tensor<1x32x360x16xf16, {order = #NHWC}>, tensor<1x32x360x16xf16, {order = #NHWC}>

    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 256, 360, 2] : tensor<1x512x360x2xf16, {order = #NHWC}> to tensor<1x256x360x2xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST0:%.+]] = VPU.ShapeCast {shape = [1, 32, 360, 16]} inputs([[SLICE0]] : tensor<1x256x360x2xf16, {order = #NHWC}>) -> tensor<1x32x360x16xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[ARG0]] [0, 256, 0, 0] [1, 256, 360, 2] : tensor<1x512x360x2xf16, {order = #NHWC}> to tensor<1x256x360x2xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST1:%.+]] = VPU.ShapeCast {shape = [1, 32, 360, 16]} inputs([[SLICE1]] : tensor<1x256x360x2xf16, {order = #NHWC}>) -> tensor<1x32x360x16xf16, {order = #NHWC}>
    // CHECK:       return [[SHAPECAST0]], [[SHAPECAST1]] : tensor<1x32x360x16xf16, {order = #NHWC}>, tensor<1x32x360x16xf16, {order = #NHWC}>
}
