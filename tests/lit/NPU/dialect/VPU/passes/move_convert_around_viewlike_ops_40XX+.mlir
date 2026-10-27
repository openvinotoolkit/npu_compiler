//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --move-convert-around-viewlike-ops %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveConvertAfterPermuteCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x1x1xf32, {order = #NCHW}>)
func.func @MoveConvertAfterPermuteCast(%arg0: tensor<1x16x1x1xf32, {order = #NCHW}>) -> tensor<1x1x16x1xf16, {order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x16x1x1xf32, {order = #NCHW}> -> tensor<1x16x1x1xf16, {order = #NCHW}>
    %1 = VPU.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x1x1xf16, {order = #NCHW}> -> tensor<1x1x16x1xf16, {order = #NHWC}>

    return %1 : tensor<1x1x16x1xf16, {order = #NHWC}>

    //CHECK: [[VAL0:%.+]] = VPU.PermuteCast([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x1x1xf32, {order = #NCHW}> -> tensor<1x1x16x1xf32, {order = #NHWC}>
    //CHECK: [[VAL1:%.+]] = VPU.Convert([[VAL0]]) {dstElemType = f16} : tensor<1x1x16x1xf32, {order = #NHWC}> -> tensor<1x1x16x1xf16, {order = #NHWC}>
    //CHECK: return [[VAL1]] : tensor<1x1x16x1xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @MoveConvertAfterShapeCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x2x3xf32>)
func.func @MoveConvertAfterShapeCast(%arg0: tensor<1x16x2x3xf32>) -> tensor<1x16x3x2xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x16x2x3xf32> -> tensor<1x16x2x3xf16>
    %1 = VPU.ShapeCast {shape = [1, 16, 3, 2]} inputs(%0 : tensor<1x16x2x3xf16>) -> tensor<1x16x3x2xf16>

    return %1 : tensor<1x16x3x2xf16>

    //CHECK: [[VAL0:%.+]] =  VPU.ShapeCast {shape = [1, 16, 3, 2]} inputs([[ARG_0]] : tensor<1x16x2x3xf32>) -> tensor<1x16x3x2xf32>
    //CHECK: [[VAL1:%.+]] =  VPU.Convert([[VAL0]]) {dstElemType = f16} : tensor<1x16x3x2xf32> -> tensor<1x16x3x2xf16>
    //CHECK: return [[VAL1]] :  tensor<1x16x3x2xf16>
}

// -----

// CHECK-LABEL: @MoveConvertBeforeAffineReshape
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @MoveConvertBeforeAffineReshape(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16xf32> {
   %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 16]} : tensor<1x16x1x1xf16> -> tensor<1x16xf16>
   %1 = VPU.Convert(%0) {dstElemType = f32} : tensor<1x16xf16> -> tensor<1x16xf32>
   return %1 : tensor<1x16xf32>

   //CHECK: [[VAL0:%.+]] = VPU.Convert([[ARG_0]]) {dstElemType = f32} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf32>
   //CHECK: [[VAL1:%.+]] = VPU.AffineReshape([[VAL0]])
   //CHECK: return [[VAL1]] : tensor<1x16xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotMoveConvertAfterPermuteCastForNonDMAConvert
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NCHW}>)
func.func @NotMoveConvertAfterPermuteCastForNonDMAConvert(%arg0: tensor<1x16x1x1xf16, {order = #NCHW}>) -> tensor<1x1x16x1xf32, {order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x16x1x1xf16, {order = #NCHW}> -> tensor<1x16x1x1xf32, {order = #NCHW}>
    %1 = VPU.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x1x1xf32, {order = #NCHW}> -> tensor<1x1x16x1xf32, {order = #NHWC}>

    return %1 : tensor<1x1x16x1xf32, {order = #NHWC}>

    //CHECK: [[VAL0:%.+]] = VPU.Convert([[ARG_0]]) {dstElemType = f32} : tensor<1x16x1x1xf16, {order = #NCHW}> -> tensor<1x16x1x1xf32, {order = #NCHW}>
    //CHECK: [[VAL1:%.+]] = VPU.PermuteCast([[VAL0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x1x1xf32, {order = #NCHW}> -> tensor<1x1x16x1xf32, {order = #NHWC}>
    //CHECK: return [[VAL1]] : tensor<1x1x16x1xf32, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @NotMoveConvertOpBeforeAffineReshape
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x2x2xf16>)
func.func @NotMoveConvertOpBeforeAffineReshape(%arg0: tensor<1x16x2x2xf16>) -> tensor<1x64x1x1xf32> {
   %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 64, 1, 1]} : tensor<1x16x2x2xf16> -> tensor<1x64x1x1xf16>
   %1 = VPU.Convert(%0) {dstElemType = f32} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1xf32>
   return %1 : tensor<1x64x1x1xf32>


  //CHECK: [[VAL0:%.+]] =  VPU.AffineReshape([[ARG_0]])
  //CHECK: [[VAL1:%.+]] =  VPU.Convert([[VAL0]]) {dstElemType = f32} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1xf32>
  //CHECK: return [[VAL1]] :  tensor<1x64x1x1xf32>
}

// -----

// CHECK-LABEL: @NotMoveConvertAfterShapeCastForNonDMAConvert
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x2x3xbf16>)
func.func @NotMoveConvertAfterShapeCastForNonDMAConvert(%arg0: tensor<1x16x2x3xbf16>) -> tensor<1x16x3x2xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x16x2x3xbf16> -> tensor<1x16x2x3xf32>
    %1 = VPU.ShapeCast {shape = [1, 16, 3, 2]} inputs(%0 : tensor<1x16x2x3xf32>) -> tensor<1x16x3x2xf32>

    return %1 : tensor<1x16x3x2xf32>

    //CHECK: [[VAL0:%.+]] =  VPU.Convert([[ARG_0]]) {dstElemType = f32} : tensor<1x16x2x3xbf16> -> tensor<1x16x2x3xf32>
    //CHECK: [[VAL1:%.+]] =  VPU.ShapeCast {shape = [1, 16, 3, 2]} inputs([[VAL0]] : tensor<1x16x2x3xf32>) -> tensor<1x16x3x2xf32>
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

    %2 = VPU.NCE.MaxPool(%1, %weights_table) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [3, 3],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
    } -> tensor<1x32x8x8xf16, {order = #NHWC}>

    %3 = VPU.NCE.MaxPool(%1, %weights_table) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [3, 3],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
    } -> tensor<1x32x8x8xf16, {order = #NHWC}>

    return %2, %3 : tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x32x8x8xf16, {order = #NHWC}>

    //CHECK: [[SHAPECAST:%.+]] = VPU.ShapeCast {shape = [1, 32, 8, 8]} inputs([[INPUT]] : tensor<1x32x8x8xf32, {order = #NHWC}>) -> tensor<1x32x8x8xf32, {order = #NHWC}>
    //CHECK: [[CONVERT:%.+]] = VPU.Convert([[SHAPECAST]]) {dstElemType = f16}
    //CHECK: [[WEIGHT_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>

    //CHECK: [[MP1:%.+]] = VPU.NCE.MaxPool([[CONVERT]], [[WEIGHT_TABLE]] )
    //CHECK: [[MP2:%.+]] = VPU.NCE.MaxPool([[CONVERT]], [[WEIGHT_TABLE]] )
}

// -----

// CHECK-LABEL: @NotMoveConvertOpBeforeAffineReshapeNon4DInput
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x2xf16>)
func.func @NotMoveConvertOpBeforeAffineReshapeNon4DInput(%arg0: tensor<1x16x2xf16>) -> tensor<1x32x1xf32> {
   %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2]], shape_value = [1, 32, 1]} : tensor<1x16x2xf16> -> tensor<1x32x1xf16>
   %1 = VPU.Convert(%0) {dstElemType = f32} : tensor<1x32x1xf16> -> tensor<1x32x1xf32>
   return %1 : tensor<1x32x1xf32>


  //CHECK: [[VAL0:%.+]] =  VPU.AffineReshape([[ARG_0]])
  //CHECK: [[VAL1:%.+]] =  VPU.Convert([[VAL0]]) {dstElemType = f32} : tensor<1x32x1xf16> -> tensor<1x32x1xf32>
  //CHECK: return [[VAL1]] :  tensor<1x32x1xf32>
}

// -----

// CHECK-LABEL: @MergeConvertBeforeSliceBranchesEightBranches
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x1x1x8448xsi32>)
func.func @MergeConvertBeforeSliceBranchesEightBranches(%arg0: tensor<1x1x1x8448xsi32>) ->
    (tensor<1056x1x1x1xi64>, tensor<1056x1x1x1xi64>, tensor<1056x1x1x1xi64>, tensor<1056x1x1x1xi64>,
        tensor<1056x1x1x1xi64>, tensor<1056x1x1x1xi64>, tensor<1056x1x1x1xi64>, tensor<1056x1x1x1xi64>) {
    %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [0]], shape_value = [8448]} : tensor<1x1x1x8448xsi32> -> tensor<8448xsi32>

    %1 = VPU.Slice %0 [0] [1056] : tensor<8448xsi32> to tensor<1056xsi32>
    %2 = VPU.Reshape(%1) {shape_value = [1056, 1, 1, 1]} : tensor<1056xsi32> -> tensor<1056x1x1x1xsi32>
    %3 = VPU.Convert(%2) {dstElemType = i64} : tensor<1056x1x1x1xsi32> -> tensor<1056x1x1x1xi64>

    %4 = VPU.Slice %0 [1056] [1056] : tensor<8448xsi32> to tensor<1056xsi32>
    %5 = VPU.Reshape(%4) {shape_value = [1056, 1, 1, 1]} : tensor<1056xsi32> -> tensor<1056x1x1x1xsi32>
    %6 = VPU.Convert(%5) {dstElemType = i64} : tensor<1056x1x1x1xsi32> -> tensor<1056x1x1x1xi64>

    %7 = VPU.Slice %0 [2112] [1056] : tensor<8448xsi32> to tensor<1056xsi32>
    %8 = VPU.Reshape(%7) {shape_value = [1056, 1, 1, 1]} : tensor<1056xsi32> -> tensor<1056x1x1x1xsi32>
    %9 = VPU.Convert(%8) {dstElemType = i64} : tensor<1056x1x1x1xsi32> -> tensor<1056x1x1x1xi64>

    %10 = VPU.Slice %0 [3168] [1056] : tensor<8448xsi32> to tensor<1056xsi32>
    %11 = VPU.Reshape(%10) {shape_value = [1056, 1, 1, 1]} : tensor<1056xsi32> -> tensor<1056x1x1x1xsi32>
    %12 = VPU.Convert(%11) {dstElemType = i64} : tensor<1056x1x1x1xsi32> -> tensor<1056x1x1x1xi64>

    %13 = VPU.Slice %0 [4224] [1056] : tensor<8448xsi32> to tensor<1056xsi32>
    %14 = VPU.Reshape(%13) {shape_value = [1056, 1, 1, 1]} : tensor<1056xsi32> -> tensor<1056x1x1x1xsi32>
    %15 = VPU.Convert(%14) {dstElemType = i64} : tensor<1056x1x1x1xsi32> -> tensor<1056x1x1x1xi64>

    %16 = VPU.Slice %0 [5280] [1056] : tensor<8448xsi32> to tensor<1056xsi32>
    %17 = VPU.Reshape(%16) {shape_value = [1056, 1, 1, 1]} : tensor<1056xsi32> -> tensor<1056x1x1x1xsi32>
    %18 = VPU.Convert(%17) {dstElemType = i64} : tensor<1056x1x1x1xsi32> -> tensor<1056x1x1x1xi64>

    %19 = VPU.Slice %0 [6336] [1056] : tensor<8448xsi32> to tensor<1056xsi32>
    %20 = VPU.Reshape(%19) {shape_value = [1056, 1, 1, 1]} : tensor<1056xsi32> -> tensor<1056x1x1x1xsi32>
    %21 = VPU.Convert(%20) {dstElemType = i64} : tensor<1056x1x1x1xsi32> -> tensor<1056x1x1x1xi64>

    %22 = VPU.Slice %0 [7392] [1056] : tensor<8448xsi32> to tensor<1056xsi32>
    %23 = VPU.Reshape(%22) {shape_value = [1056, 1, 1, 1]} : tensor<1056xsi32> -> tensor<1056x1x1x1xsi32>
    %24 = VPU.Convert(%23) {dstElemType = i64} : tensor<1056x1x1x1xsi32> -> tensor<1056x1x1x1xi64>

    return %3, %6, %9, %12, %15, %18, %21, %24 :
        tensor<1056x1x1x1xi64>, tensor<1056x1x1x1xi64>, tensor<1056x1x1x1xi64>, tensor<1056x1x1x1xi64>,
        tensor<1056x1x1x1xi64>, tensor<1056x1x1x1xi64>, tensor<1056x1x1x1xi64>, tensor<1056x1x1x1xi64>

    // CHECK: [[CONVERT:%.+]] = VPU.Convert([[ARG_0]]) {dstElemType = i64} : tensor<1x1x1x8448xsi32> -> tensor<1x1x1x8448xi64>
    // CHECK: [[RESHAPE:%.+]] = VPU.AffineReshape([[CONVERT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [0]], shape_value = [8448]}
    // CHECK-SAME: tensor<1x1x1x8448xi64> -> tensor<8448xi64>

    // CHECK-DAG: [[S0:%.+]] = VPU.Slice [[RESHAPE]] [0] [1056] : tensor<8448xi64> to tensor<1056xi64>
    // CHECK-DAG: [[R0:%.+]] = VPU.Reshape([[S0]]) {shape_value = [1056, 1, 1, 1]} : tensor<1056xi64> -> tensor<1056x1x1x1xi64>
    // CHECK-DAG: [[S1:%.+]] = VPU.Slice [[RESHAPE]] [1056] [1056] : tensor<8448xi64> to tensor<1056xi64>
    // CHECK-DAG: [[R1:%.+]] = VPU.Reshape([[S1]]) {shape_value = [1056, 1, 1, 1]} : tensor<1056xi64> -> tensor<1056x1x1x1xi64>
    // CHECK-DAG: [[S2:%.+]] = VPU.Slice [[RESHAPE]] [2112] [1056] : tensor<8448xi64> to tensor<1056xi64>
    // CHECK-DAG: [[R2:%.+]] = VPU.Reshape([[S2]]) {shape_value = [1056, 1, 1, 1]} : tensor<1056xi64> -> tensor<1056x1x1x1xi64>
    // CHECK-DAG: [[S3:%.+]] = VPU.Slice [[RESHAPE]] [3168] [1056] : tensor<8448xi64> to tensor<1056xi64>
    // CHECK-DAG: [[R3:%.+]] = VPU.Reshape([[S3]]) {shape_value = [1056, 1, 1, 1]} : tensor<1056xi64> -> tensor<1056x1x1x1xi64>
    // CHECK-DAG: [[S4:%.+]] = VPU.Slice [[RESHAPE]] [4224] [1056] : tensor<8448xi64> to tensor<1056xi64>
    // CHECK-DAG: [[R4:%.+]] = VPU.Reshape([[S4]]) {shape_value = [1056, 1, 1, 1]} : tensor<1056xi64> -> tensor<1056x1x1x1xi64>
    // CHECK-DAG: [[S5:%.+]] = VPU.Slice [[RESHAPE]] [5280] [1056] : tensor<8448xi64> to tensor<1056xi64>
    // CHECK-DAG: [[R5:%.+]] = VPU.Reshape([[S5]]) {shape_value = [1056, 1, 1, 1]} : tensor<1056xi64> -> tensor<1056x1x1x1xi64>
    // CHECK-DAG: [[S6:%.+]] = VPU.Slice [[RESHAPE]] [6336] [1056] : tensor<8448xi64> to tensor<1056xi64>
    // CHECK-DAG: [[R6:%.+]] = VPU.Reshape([[S6]]) {shape_value = [1056, 1, 1, 1]} : tensor<1056xi64> -> tensor<1056x1x1x1xi64>
    // CHECK-DAG: [[S7:%.+]] = VPU.Slice [[RESHAPE]] [7392] [1056] : tensor<8448xi64> to tensor<1056xi64>
    // CHECK-DAG: [[R7:%.+]] = VPU.Reshape([[S7]]) {shape_value = [1056, 1, 1, 1]} : tensor<1056xi64> -> tensor<1056x1x1x1xi64>

    // CHECK: return [[R0]], [[R1]], [[R2]], [[R3]], [[R4]], [[R5]], [[R6]], [[R7]]
}

// -----

// CHECK-LABEL: @NotMergeConvertBeforeSliceBranchesInsufficientBranchCount
// CHECK-SAME: ([[IDX_4D:%[^:]+]]: tensor<1x1x1x23040xsi32>, [[TABLE:%[^:]+]]: tensor<184320x2880xf16>)
func.func @NotMergeConvertBeforeSliceBranchesInsufficientBranchCount(
    %idx4d: tensor<1x1x1x23040xsi32>, %table: tensor<184320x2880xf16>) ->
    (tensor<5760x2880x1x1xf16>, tensor<5760x2880x1x1xf16>, tensor<5760x2880x1x1xf16>, tensor<5760x2880x1x1xf16>) {
    %0 = VPU.AffineReshape(%idx4d) {dim_mapping = [[0], [0], [0], [0]], shape_value = [23040]} : tensor<1x1x1x23040xsi32> -> tensor<23040xsi32>

    %1 = VPU.AffineReshape(%table) {dim_mapping = [[0, 1], [2]], shape_value = [1, 184320, 2880]} : tensor<184320x2880xf16> -> tensor<1x184320x2880xf16>
    %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [184320, 2880, 1, 1]} : tensor<1x184320x2880xf16> -> tensor<184320x2880x1x1xf16>

    %3 = VPU.Slice %0 [0] [5760] : tensor<23040xsi32> to tensor<5760xsi32>
    %4 = VPU.Reshape(%3) {shape_value = [5760, 1, 1, 1]} : tensor<5760xsi32> -> tensor<5760x1x1x1xsi32>
    %5 = VPU.Convert(%4) {dstElemType = i64} : tensor<5760x1x1x1xsi32> -> tensor<5760x1x1x1xi64>
    %6 = VPU.GatherDMA(%2, %5) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<184320x2880x1x1xf16>, tensor<5760x1x1x1xi64> -> tensor<5760x2880x1x1xf16>

    %7 = VPU.Slice %0 [5760] [5760] : tensor<23040xsi32> to tensor<5760xsi32>
    %8 = VPU.Reshape(%7) {shape_value = [5760, 1, 1, 1]} : tensor<5760xsi32> -> tensor<5760x1x1x1xsi32>
    %9 = VPU.Convert(%8) {dstElemType = i64} : tensor<5760x1x1x1xsi32> -> tensor<5760x1x1x1xi64>
    %10 = VPU.GatherDMA(%2, %9) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<184320x2880x1x1xf16>, tensor<5760x1x1x1xi64> -> tensor<5760x2880x1x1xf16>

    %11 = VPU.Slice %0 [11520] [5760] : tensor<23040xsi32> to tensor<5760xsi32>
    %12 = VPU.Reshape(%11) {shape_value = [5760, 1, 1, 1]} : tensor<5760xsi32> -> tensor<5760x1x1x1xsi32>
    %13 = VPU.Convert(%12) {dstElemType = i64} : tensor<5760x1x1x1xsi32> -> tensor<5760x1x1x1xi64>
    %14 = VPU.GatherDMA(%2, %13) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<184320x2880x1x1xf16>, tensor<5760x1x1x1xi64> -> tensor<5760x2880x1x1xf16>

    %15 = VPU.Slice %0 [17280] [5760] : tensor<23040xsi32> to tensor<5760xsi32>
    %16 = VPU.Reshape(%15) {shape_value = [5760, 1, 1, 1]} : tensor<5760xsi32> -> tensor<5760x1x1x1xsi32>
    %17 = VPU.Convert(%16) {dstElemType = i64} : tensor<5760x1x1x1xsi32> -> tensor<5760x1x1x1xi64>
    %18 = VPU.GatherDMA(%2, %17) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<184320x2880x1x1xf16>, tensor<5760x1x1x1xi64> -> tensor<5760x2880x1x1xf16>

    return %6, %10, %14, %18 : tensor<5760x2880x1x1xf16>, tensor<5760x2880x1x1xf16>, tensor<5760x2880x1x1xf16>, tensor<5760x2880x1x1xf16>

    // CHECK: [[IDX_1D:%.+]] = VPU.AffineReshape([[IDX_4D]])
    // CHECK-NOT: VPU.Convert([[IDX_4D]]) {dstElemType = i64}
    // CHECK: [[SLICE0:%.+]] = VPU.Slice [[IDX_1D]] [0] [5760] : tensor<23040xsi32> to tensor<5760xsi32>
    // CHECK: [[RESHAPE0:%.+]] = VPU.Reshape([[SLICE0]]) {shape_value = [5760, 1, 1, 1]} : tensor<5760xsi32> -> tensor<5760x1x1x1xsi32>
    // CHECK: [[CVT0:%.+]] = VPU.Convert([[RESHAPE0]]) {dstElemType = i64} : tensor<5760x1x1x1xsi32> -> tensor<5760x1x1x1xi64>
    // CHECK: VPU.GatherDMA({{.*}}, [[CVT0]])
}

// -----

// CHECK-LABEL: @MergeConvertBeforeSliceBranchesMixedReshapeChains
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x1x1x4096xsi32>)
func.func @MergeConvertBeforeSliceBranchesMixedReshapeChains(%arg0: tensor<1x1x1x4096xsi32>) ->
    (tensor<512x1x1x1xi64>, tensor<512xi64>, tensor<512x1x1x1xi64>, tensor<512xi64>, tensor<512x1x1x1xi64>,
        tensor<512xi64>, tensor<512x1x1x1xi64>, tensor<512xi64>) {
    %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [0]], shape_value = [4096]} : tensor<1x1x1x4096xsi32> -> tensor<4096xsi32>

    %1 = VPU.Slice %0 [0] [512] : tensor<4096xsi32> to tensor<512xsi32>
    %2 = VPU.Reshape(%1) {shape_value = [512, 1, 1, 1]} : tensor<512xsi32> -> tensor<512x1x1x1xsi32>
    %3 = VPU.Convert(%2) {dstElemType = i64} : tensor<512x1x1x1xsi32> -> tensor<512x1x1x1xi64>

    %4 = VPU.Slice %0 [512] [512] : tensor<4096xsi32> to tensor<512xsi32>
    %5 = VPU.Convert(%4) {dstElemType = i64} : tensor<512xsi32> -> tensor<512xi64>

    %6 = VPU.Slice %0 [1024] [512] : tensor<4096xsi32> to tensor<512xsi32>
    %7 = VPU.Reshape(%6) {shape_value = [512, 1, 1, 1]} : tensor<512xsi32> -> tensor<512x1x1x1xsi32>
    %8 = VPU.Convert(%7) {dstElemType = i64} : tensor<512x1x1x1xsi32> -> tensor<512x1x1x1xi64>

    %9 = VPU.Slice %0 [1536] [512] : tensor<4096xsi32> to tensor<512xsi32>
    %10 = VPU.Convert(%9) {dstElemType = i64} : tensor<512xsi32> -> tensor<512xi64>

    %11 = VPU.Slice %0 [2048] [512] : tensor<4096xsi32> to tensor<512xsi32>
    %12 = VPU.Reshape(%11) {shape_value = [512, 1, 1, 1]} : tensor<512xsi32> -> tensor<512x1x1x1xsi32>
    %13 = VPU.Convert(%12) {dstElemType = i64} : tensor<512x1x1x1xsi32> -> tensor<512x1x1x1xi64>

    %14 = VPU.Slice %0 [2560] [512] : tensor<4096xsi32> to tensor<512xsi32>
    %15 = VPU.Convert(%14) {dstElemType = i64} : tensor<512xsi32> -> tensor<512xi64>

    %16 = VPU.Slice %0 [3072] [512] : tensor<4096xsi32> to tensor<512xsi32>
    %17 = VPU.Reshape(%16) {shape_value = [512, 1, 1, 1]} : tensor<512xsi32> -> tensor<512x1x1x1xsi32>
    %18 = VPU.Convert(%17) {dstElemType = i64} : tensor<512x1x1x1xsi32> -> tensor<512x1x1x1xi64>

    %19 = VPU.Slice %0 [3584] [512] : tensor<4096xsi32> to tensor<512xsi32>
    %20 = VPU.Convert(%19) {dstElemType = i64} : tensor<512xsi32> -> tensor<512xi64>

    return %3, %5, %8, %10, %13, %15, %18, %20 :
        tensor<512x1x1x1xi64>, tensor<512xi64>, tensor<512x1x1x1xi64>, tensor<512xi64>, tensor<512x1x1x1xi64>,
        tensor<512xi64>, tensor<512x1x1x1xi64>, tensor<512xi64>

    // CHECK: [[ROOT_CVT:%.+]] = VPU.Convert([[ARG_0]]) {dstElemType = i64} : tensor<1x1x1x4096xsi32> -> tensor<1x1x1x4096xi64>
    // CHECK: [[RESHAPE_1D:%.+]] = VPU.AffineReshape([[ROOT_CVT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [0]], shape_value = [4096]}
    // CHECK-SAME: tensor<1x1x1x4096xi64> -> tensor<4096xi64>

    // CHECK-DAG: [[S0:%.+]] = VPU.Slice [[RESHAPE_1D]] [0] [512] : tensor<4096xi64> to tensor<512xi64>
    // CHECK-DAG: [[R0:%.+]] = VPU.Reshape([[S0]]) {shape_value = [512, 1, 1, 1]} : tensor<512xi64> -> tensor<512x1x1x1xi64>
    // CHECK-DAG: [[S1:%.+]] = VPU.Slice [[RESHAPE_1D]] [512] [512] : tensor<4096xi64> to tensor<512xi64>

    // CHECK: return [[R0]], [[S1]]
}

    // -----

    // CHECK-LABEL: @NotMergeConvertBeforeSliceBranchesWithEmptySliceUsers
    // CHECK-SAME: ([[IDX_4D:%[^:]+]]: tensor<1x1x1x23040xsi32>, [[TABLE:%[^:]+]]: tensor<184320x2880xf16>)
    func.func @NotMergeConvertBeforeSliceBranchesWithEmptySliceUsers(
        %idx4d: tensor<1x1x1x23040xsi32>, %table: tensor<184320x2880xf16>) ->
        (tensor<2880x2880x1x1xf16>, tensor<2880x2880x1x1xf16>, tensor<2880x2880x1x1xf16>, tensor<2880x2880x1x1xf16>) {
        %0 = VPU.AffineReshape(%idx4d) {dim_mapping = [[0], [0], [0], [0]], shape_value = [23040]} : tensor<1x1x1x23040xsi32> -> tensor<23040xsi32>

        %1 = VPU.AffineReshape(%table) {dim_mapping = [[0, 1], [2]], shape_value = [1, 184320, 2880]} : tensor<184320x2880xf16> -> tensor<1x184320x2880xf16>
        %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [184320, 2880, 1, 1]} : tensor<1x184320x2880xf16> -> tensor<184320x2880x1x1xf16>

        // 4 meaningful branches: Slice -> Reshape -> Convert -> GatherDMA.
        %3 = VPU.Slice %0 [0] [2880] : tensor<23040xsi32> to tensor<2880xsi32>
        %4 = VPU.Reshape(%3) {shape_value = [2880, 1, 1, 1]} : tensor<2880xsi32> -> tensor<2880x1x1x1xsi32>
        %5 = VPU.Convert(%4) {dstElemType = i64} : tensor<2880x1x1x1xsi32> -> tensor<2880x1x1x1xi64>
        %6 = VPU.GatherDMA(%2, %5) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<184320x2880x1x1xf16>, tensor<2880x1x1x1xi64> -> tensor<2880x2880x1x1xf16>

        %7 = VPU.Slice %0 [2880] [2880] : tensor<23040xsi32> to tensor<2880xsi32>
        %8 = VPU.Reshape(%7) {shape_value = [2880, 1, 1, 1]} : tensor<2880xsi32> -> tensor<2880x1x1x1xsi32>
        %9 = VPU.Convert(%8) {dstElemType = i64} : tensor<2880x1x1x1xsi32> -> tensor<2880x1x1x1xi64>
        %10 = VPU.GatherDMA(%2, %9) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<184320x2880x1x1xf16>, tensor<2880x1x1x1xi64> -> tensor<2880x2880x1x1xf16>

        %11 = VPU.Slice %0 [5760] [2880] : tensor<23040xsi32> to tensor<2880xsi32>
        %12 = VPU.Reshape(%11) {shape_value = [2880, 1, 1, 1]} : tensor<2880xsi32> -> tensor<2880x1x1x1xsi32>
        %13 = VPU.Convert(%12) {dstElemType = i64} : tensor<2880x1x1x1xsi32> -> tensor<2880x1x1x1xi64>
        %14 = VPU.GatherDMA(%2, %13) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<184320x2880x1x1xf16>, tensor<2880x1x1x1xi64> -> tensor<2880x2880x1x1xf16>

        %15 = VPU.Slice %0 [8640] [2880] : tensor<23040xsi32> to tensor<2880xsi32>
        %16 = VPU.Reshape(%15) {shape_value = [2880, 1, 1, 1]} : tensor<2880xsi32> -> tensor<2880x1x1x1xsi32>
        %17 = VPU.Convert(%16) {dstElemType = i64} : tensor<2880x1x1x1xsi32> -> tensor<2880x1x1x1xi64>
        %18 = VPU.GatherDMA(%2, %17) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<184320x2880x1x1xf16>, tensor<2880x1x1x1xi64> -> tensor<2880x2880x1x1xf16>

        // 4 empty branches: any empty Slice branch makes the merge pattern ineligible.
        %19 = VPU.Slice %0 [11520] [2880] : tensor<23040xsi32> to tensor<2880xsi32>
        %20 = VPU.Slice %0 [14400] [2880] : tensor<23040xsi32> to tensor<2880xsi32>
        %21 = VPU.Slice %0 [17280] [2880] : tensor<23040xsi32> to tensor<2880xsi32>
        %22 = VPU.Slice %0 [20160] [2880] : tensor<23040xsi32> to tensor<2880xsi32>

        return %6, %10, %14, %18 : tensor<2880x2880x1x1xf16>, tensor<2880x2880x1x1xf16>, tensor<2880x2880x1x1xf16>, tensor<2880x2880x1x1xf16>

        // CHECK: [[IDX_1D:%.+]] = VPU.AffineReshape([[IDX_4D]])
        // CHECK-NOT: VPU.Convert([[IDX_4D]]) {dstElemType = i64}
        // CHECK: [[SLICE0:%.+]] = VPU.Slice [[IDX_1D]] [0] [2880] : tensor<23040xsi32> to tensor<2880xsi32>
        // CHECK: [[RESHAPE0:%.+]] = VPU.Reshape([[SLICE0]]) {shape_value = [2880, 1, 1, 1]} : tensor<2880xsi32> -> tensor<2880x1x1x1xsi32>
        // CHECK: [[CVT0:%.+]] = VPU.Convert([[RESHAPE0]]) {dstElemType = i64} : tensor<2880x1x1x1xsi32> -> tensor<2880x1x1x1xi64>
        // CHECK: VPU.GatherDMA({{.*}}, [[CVT0]])
    }
