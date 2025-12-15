
//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --move-convert-around-viewlike-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveConvertAfterPermuteCast
func.func @MoveConvertAfterPermuteCast(%arg0: tensor<1x16x1x1xf32, {order = #NCHW}>) -> tensor<1x1x16x1xf16, {order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x16x1x1xf32, {order = #NCHW}> -> tensor<1x16x1x1xf16, {order = #NCHW}>
    %1 = VPU.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x1x1xf16, {order = #NCHW}> -> tensor<1x1x16x1xf16, {order = #NHWC}>

    return %1 : tensor<1x1x16x1xf16, {order = #NHWC}>

    //CHECK: [[VAL0:%.*]] = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x1x1xf32, {order = #NCHW}> -> tensor<1x1x16x1xf32, {order = #NHWC}>
    //CHECK: [[VAL1:%.*]] = VPU.Convert([[VAL0]]) {dstElemType = f16} : tensor<1x1x16x1xf32, {order = #NHWC}> -> tensor<1x1x16x1xf16, {order = #NHWC}>
    //CHECK: return [[VAL1]] : tensor<1x1x16x1xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @MoveConvertAfterShapeCast
func.func @MoveConvertAfterShapeCast(%arg0: tensor<1x16x2x3xf32>) -> tensor<1x16x3x2xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x16x2x3xf32> -> tensor<1x16x2x3xf16>
    %1 = VPU.ShapeCast {shape = [1, 16, 3, 2]} inputs(%0 : tensor<1x16x2x3xf16>) -> tensor<1x16x3x2xf16>

    return %1 : tensor<1x16x3x2xf16>

    //CHECK: [[VAL0:%.*]] =  VPU.ShapeCast {shape = [1, 16, 3, 2]} inputs(%arg0 : tensor<1x16x2x3xf32>) -> tensor<1x16x3x2xf32>
    //CHECK: [[VAL1:%.*]] =  VPU.Convert([[VAL0]]) {dstElemType = f16} : tensor<1x16x3x2xf32> -> tensor<1x16x3x2xf16>
    //CHECK: return [[VAL1]] :  tensor<1x16x3x2xf16>
}

// -----

// CHECK-LABEL: @MoveConvertBeforeAffineReshape
func.func @MoveConvertBeforeAffineReshape(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16xf32> {
   %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 16]} : tensor<1x16x1x1xf16> -> tensor<1x16xf16>
   %1 = VPU.Convert(%0) {dstElemType = f32} : tensor<1x16xf16> -> tensor<1x16xf32>
   return %1 : tensor<1x16xf32>

   //CHECK: [[VAL0:%.*]] = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf32>
   //CHECK: [[VAL1:%.*]] = VPU.AffineReshape([[VAL0]])
   //CHECK: return [[VAL1]] : tensor<1x16xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotMoveConvertAfterPermuteCastForNonDMAConvert
func.func @NotMoveConvertAfterPermuteCastForNonDMAConvert(%arg0: tensor<1x16x1x1xf16, {order = #NCHW}>) -> tensor<1x1x16x1xf32, {order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x16x1x1xf16, {order = #NCHW}> -> tensor<1x16x1x1xf32, {order = #NCHW}>
    %1 = VPU.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x1x1xf32, {order = #NCHW}> -> tensor<1x1x16x1xf32, {order = #NHWC}>

    return %1 : tensor<1x1x16x1xf32, {order = #NHWC}>

    //CHECK: [[VAL0:%.*]] = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x16x1x1xf16, {order = #NCHW}> -> tensor<1x16x1x1xf32, {order = #NCHW}>
    //CHECK: [[VAL1:%.*]] = VPU.PermuteCast([[VAL0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x1x1xf32, {order = #NCHW}> -> tensor<1x1x16x1xf32, {order = #NHWC}>
    //CHECK: return [[VAL1]] : tensor<1x1x16x1xf32, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @NotMoveConvertOpBeforeAffineReshape
func.func @NotMoveConvertOpBeforeAffineReshape(%arg0: tensor<1x16x2x2xf16>) -> tensor<1x64x1x1xf32> {
   %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 64, 1, 1]} : tensor<1x16x2x2xf16> -> tensor<1x64x1x1xf16>
   %1 = VPU.Convert(%0) {dstElemType = f32} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1xf32>
   return %1 : tensor<1x64x1x1xf32>


  //CHECK: [[VAL0:%.*]] =  VPU.AffineReshape(%arg0)
  //CHECK: [[VAL1:%.*]] =  VPU.Convert([[VAL0]]) {dstElemType = f32} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1xf32>
  //CHECK: return [[VAL1]] :  tensor<1x64x1x1xf32>
}

// -----

// CHECK-LABEL: @NotMoveConvertAfterPermuteCastForNonDMAConvert
func.func @NotMoveConvertAfterPermuteCastForNonDMAConvert(%arg0: tensor<1x16x2x3xbf16>) -> tensor<1x16x3x2xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x16x2x3xbf16> -> tensor<1x16x2x3xf32>
    %1 = VPU.ShapeCast {shape = [1, 16, 3, 2]} inputs(%0 : tensor<1x16x2x3xf32>) -> tensor<1x16x3x2xf32>

    return %1 : tensor<1x16x3x2xf32>

    //CHECK: [[VAL0:%.*]] =  VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x16x2x3xbf16> -> tensor<1x16x2x3xf32>
    //CHECK: [[VAL1:%.*]] =  VPU.ShapeCast {shape = [1, 16, 3, 2]} inputs([[VAL0]] : tensor<1x16x2x3xf32>) -> tensor<1x16x3x2xf32>
    //CHECK: return [[VAL1]] :  tensor<1x16x3x2xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveConvertAfterShapeCastMultipleUsers
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x32x8x8xf32, {order = #NHWC}>)
func.func @MoveConvertAfterShapeCastMultipleUsers(%arg0: tensor<1x32x8x8xf32, {order = #NHWC}>) -> (tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x32x8x8xf16, {order = #NHWC}>) {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x32x8x8xf32, {order = #NHWC}> -> tensor<1x32x8x8xf16, {order = #NHWC}>
    %1 = VPU.ShapeCast {shape = [1, 32, 8, 8]} inputs(%0 : tensor<1x32x8x8xf16, {order = #NHWC}>) -> tensor<1x32x8x8xf16, {order = #NHWC}>

    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %2 = VPU.NCE.MaxPool(%1, %weights_table) {
        kernel_size = [3, 3],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
    } -> tensor<1x32x8x8xf16, {order = #NHWC}>

    %3 = VPU.NCE.MaxPool(%1, %weights_table) {
        kernel_size = [3, 3],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
    } -> tensor<1x32x8x8xf16, {order = #NHWC}>

    return %2, %3 : tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x32x8x8xf16, {order = #NHWC}>

    //CHECK: [[SHAPECAST:%.*]] = VPU.ShapeCast {shape = [1, 32, 8, 8]} inputs([[INPUT]] : tensor<1x32x8x8xf32, {order = #NHWC}>) -> tensor<1x32x8x8xf32, {order = #NHWC}>
    //CHECK: [[CONVERT:%.*]] = VPU.Convert([[SHAPECAST]]) {dstElemType = f16}
    //CHECK: [[WEIGHT_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32>

    //CHECK: [[MP1:%.*]] = VPU.NCE.MaxPool([[CONVERT]], [[WEIGHT_TABLE]] )
    //CHECK: [[MP2:%.*]] = VPU.NCE.MaxPool([[CONVERT]], [[WEIGHT_TABLE]] )
}

// -----

// CHECK-LABEL: @NotMoveConvertOpBeforeAffineReshapeNon4DInput
func.func @NotMoveConvertOpBeforeAffineReshapeNon4DInput(%arg0: tensor<1x16x2xf16>) -> tensor<1x32x1xf32> {
   %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2]], shape_value = [1, 32, 1]} : tensor<1x16x2xf16> -> tensor<1x32x1xf16>
   %1 = VPU.Convert(%0) {dstElemType = f32} : tensor<1x32x1xf16> -> tensor<1x32x1xf32>
   return %1 : tensor<1x32x1xf32>


  //CHECK: [[VAL0:%.*]] =  VPU.AffineReshape(%arg0)
  //CHECK: [[VAL1:%.*]] =  VPU.Convert([[VAL0]]) {dstElemType = f32} : tensor<1x32x1xf16> -> tensor<1x32x1xf32>
  //CHECK: return [[VAL1]] :  tensor<1x32x1xf32>
}
