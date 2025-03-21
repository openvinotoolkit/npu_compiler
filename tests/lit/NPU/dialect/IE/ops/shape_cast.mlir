//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @Eliminate
func.func @Eliminate(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16> {
    %0 = IE.ShapeCast {shape = [1, 2, 3, 4]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16>
    return %0 : tensor<1x2x3x4xf16>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @Fuse
func.func @Fuse(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16> {
    %0 = IE.ShapeCast {shape = [1, 3, 2, 4]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16>
    %1 = IE.ShapeCast {shape = [1, 2, 3, 4]} inputs(%0 : tensor<1x3x2x4xf16>) -> tensor<1x2x3x4xf16>
    return %1 : tensor<1x2x3x4xf16>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @FuseSequence
func.func @FuseSequence(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x4x3x2xf16> {
    %0 = IE.ShapeCast {shape = [1, 3, 2, 4]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16>
    %1 = IE.ShapeCast {shape = [1, 3, 4, 2]} inputs(%0 : tensor<1x3x2x4xf16>) -> tensor<1x3x4x2xf16>
    %2 = IE.ShapeCast {shape = [1, 4, 3, 2]} inputs(%1 : tensor<1x3x4x2xf16>) -> tensor<1x4x3x2xf16>
    return %2 : tensor<1x4x3x2xf16>

    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 4, 3, 2]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x4x3x2xf16>
    // CHECK:       return [[SHAPE_CAST]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @FuseShapeCastWithMultipleBranches
func.func @FuseShapeCastWithMultipleBranches(%arg0 : tensor<1x2x3x4xf16>) ->
    (tensor<1x3x2x4xf16>, tensor<1x3x4x2xf16, { order = #NCWH }>)
{
    %0 = IE.ShapeCast { shape = [1, 3, 2, 4] } inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16>

    %1 = IE.ShapeCast { shape = [1, 3, 2, 4] } inputs(%0 : tensor<1x3x2x4xf16>) -> tensor<1x3x2x4xf16>
    %2 = IE.PermuteCast(%0) { dst_order = #NCWH, mem_perm = #NCHW } : tensor<1x3x2x4xf16> -> tensor<1x3x4x2xf16, { order = #NCWH }>

    return %0, %2 : tensor<1x3x2x4xf16>, tensor<1x3x4x2xf16, { order = #NCWH }>

    // CHECK: [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 3, 2, 4]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16>
    // CHECK: [[PERMUTECAST:%.+]] = IE.PermuteCast([[SHAPECAST]]) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x3x2x4xf16> -> tensor<1x3x4x2xf16, {order = #NCWH}>
    // CHECK: return [[SHAPECAST]], [[PERMUTECAST]] : tensor<1x3x2x4xf16>, tensor<1x3x4x2xf16, {order = #NCWH}>
}

// -----

// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<1x3x2x4xf16> {
    %0 = const.Declare tensor<1x2x3x4xf16> = dense<1.0> : tensor<1x2x3x4xf16>
    %1 = IE.ShapeCast { shape = [1, 3, 2, 4] } inputs(%0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16>
    return %1 : tensor<1x3x2x4xf16>

    // CHECK:       [[VAR0:%.+]] = const.Declare tensor<1x3x2x4xf16> = dense<1.000000e+00> : tensor<1x2x3x4xf16>, [#const.Reshape<[1, 3, 2, 4]>]
    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       return [[VAR0]] : tensor<1x3x2x4xf16>
}

// -----

!qElemType = !quant.uniform<i8<-127:127>:f16:1, {0.01, 0.02}>
// CHECK-LABEL: @ConstFoldQuantPerChannel
func.func @ConstFoldQuantPerChannel() -> tensor<1x2x4x3xf16> {
    %cst = const.Declare tensor<1x2x3x4x!qElemType> = dense<1.000000e+00> : tensor<1x2x3x4xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
    %1 = IE.ShapeCast { shape = [1, 2, 4, 3] } inputs(%cst : tensor<1x2x3x4x!qElemType>) -> tensor<1x2x4x3x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x2x4x3x!qElemType> -> tensor<1x2x4x3xf16>
    return %2 : tensor<1x2x4x3xf16>

    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<1x2x3x4x!qElemType> = dense<1.000000e+00> : tensor<1x2x3x4xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
    // CHECK:       [[VAR1:%.+]] = IE.ShapeCast {shape = [1, 2, 4, 3]} inputs([[CST_0]] : tensor<1x2x3x4x!qElemType>) -> tensor<1x2x4x3x!qElemType>
    // CHECK:       [[VAR2:%.+]] = IE.Dequantize([[VAR1]]) {dstElemType = f16} : tensor<1x2x4x3x!qElemType> -> tensor<1x2x4x3xf16>
    // CHECK:       return [[VAR2]] : tensor<1x2x4x3xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.002>
// CHECK-LABEL: @ConstFoldQuantPerTensor
func.func @ConstFoldQuantPerTensor() -> tensor<2x3x2x2xf16> {
    %cst = const.Declare tensor<1x2x3x4x!qElemType> = dense<1.000000e+00> : tensor<1x2x3x4xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
    %1 = IE.ShapeCast { shape = [2, 3, 2, 2] } inputs(%cst : tensor<1x2x3x4x!qElemType>) -> tensor<2x3x2x2x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<2x3x2x2x!qElemType> -> tensor<2x3x2x2xf16>
    return %2 : tensor<2x3x2x2xf16>

    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<2x3x2x2x!qElemType> = dense<1.000000e+00> : tensor<1x2x3x4xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reshape<[2, 3, 2, 2]>]
    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       [[VAR1:%.+]] = IE.Dequantize([[CST_0]]) {dstElemType = f16} : tensor<2x3x2x2x!qElemType> -> tensor<2x3x2x2xf16>
    // CHECK:       return [[VAR1]] : tensor<2x3x2x2xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseAdjacentAffineReshapeAndShapeCastOps
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x3584x256x4xf16, {order = #NHWC}>
func.func @FuseAdjacentAffineReshapeAndShapeCastOps(%arg0: tensor<1x3584x256x4xf16, {order = #NHWC}>) -> tensor<1x3584x256x4xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 3584, 1024, 1]} : tensor<1x3584x256x4xf16, {order = #NHWC}> -> tensor<1x3584x1024x1xf16, {order = #NHWC}>
    %1 = IE.ShapeCast {shape = [1, 3584, 256, 4]} inputs(%0 : tensor<1x3584x1024x1xf16, {order = #NHWC}>) -> tensor<1x3584x256x4xf16, {order = #NHWC}>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 3584, 1024, 1]} : tensor<1x3584x256x4xf16, {order = #NHWC}> -> tensor<1x3584x1024x1xf16, {order = #NHWC}>
    %3 = IE.ShapeCast {shape = [1, 3584, 256, 4]} inputs(%2 : tensor<1x3584x1024x1xf16, {order = #NHWC}>) -> tensor<1x3584x256x4xf16, {order = #NHWC}>

    return %3 : tensor<1x3584x256x4xf16, {order = #NHWC}>

    // CHECK:   return [[INPUT]] : tensor<1x3584x256x4xf16, {order = #NHWC}>
}
