//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --initial-transformations %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @FullyConnected
func.func @FullyConnected(%arg0: tensor<1x16xf32>) -> tensor<1x64xf32> {
    %weights = const.Declare tensor<64x16xf32> = dense<1.0> : tensor<64x16xf32>
    %bias = const.Declare tensor<1x64xf32> = dense<1.0> : tensor<1x64xf32>
    %0 = IE.FullyConnected(%arg0, %weights, %bias) : tensor<1x16xf32>, tensor<64x16xf32>, tensor<1x64xf32> -> tensor<1x64xf32>

    return %0 : tensor<1x64xf32>

    // CHECK-NOT:   IE.Convolution
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<64x16xf32> = dense<1.000000e+00> : tensor<64x16xf32>
    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x64xf32> = dense<1.000000e+00> : tensor<1x64xf32>
    // CHECK:       [[FC:%.+]] = IE.FullyConnected(%arg0, [[WEIGHTS]], [[BIAS]])
    // CHECK:       return [[FC]]
}

// -----

// CHECK-LABEL: @MatMul4dInputsTo2d
func.func @MatMul4dInputsTo2d(%arg0: tensor<1x2x1x512xf32>) -> tensor<1x2x1x40xf32> {
    %cst = const.Declare tensor<1x2x512x40xf32> = dense<1.0> : tensor<1x2x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x1x512xf32>, tensor<1x2x512x40xf32> -> tensor<1x2x1x40xf32>

    return %0 : tensor<1x2x1x40xf32>

    // CHECK-DAG:      [[CST_0:%.+]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00>
    // CHECK-DAG:      [[CST_1:%.+]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00>
    // CHECK:          [[IN_1:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:          [[IN_1_2D:%.+]] = IE.AffineReshape([[IN_1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:          [[IN_2:%.+]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:          [[IN_2_2D:%.+]] = IE.AffineReshape([[IN_2]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>

    // CHECK:          [[FC_1:%.+]] = IE.FullyConnected([[IN_1_2D]], [[CST_1]])
    // CHECK-SAME:           : tensor<1x512xf32>, tensor<40x512xf32> -> tensor<1x40xf32>

    // CHECK:          [[FC_2:%.+]] = IE.FullyConnected([[IN_2_2D]], [[CST_0]])
    // CHECK-SAME:           : tensor<1x512xf32>, tensor<40x512xf32> -> tensor<1x40xf32>

    // CHECK:          [[OUT_1_4D:%.+]] = IE.AffineReshape([[FC_1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 40]} : tensor<1x40xf32> -> tensor<1x1x1x40xf32>
    // CHECK:          [[OUT_2_4D:%.+]] = IE.AffineReshape([[FC_2]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 40]} : tensor<1x40xf32> -> tensor<1x1x1x40xf32>

    // CHECK:          [[CONCAT:%.+]] = IE.Concat([[OUT_1_4D]], [[OUT_2_4D]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x40xf32>, tensor<1x1x1x40xf32> -> tensor<1x2x1x40xf32>

    // CHECK:          return [[CONCAT]] : tensor<1x2x1x40xf32>
}

// -----

// CHECK-LABEL: @UnrollMatMulAndPropagate
func.func @UnrollMatMulAndPropagate(%arg0: tensor<1x8x4096x40xf32>, %arg1: tensor<1x8x4096x40xf32>) -> tensor<8x4096x4096xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x4096x40xf32>, tensor<1x8x4096x40xf32> -> tensor<1x8x4096x4096xf32>
    %1 = IE.Reshape(%0) {shape_value = [8, 4096, 4096]} : tensor<1x8x4096x4096xf32> -> tensor<8x4096x4096xf32>
    %2 = IE.SoftMax(%1) {axisInd = 2 : i64} : tensor<8x4096x4096xf32> -> tensor<8x4096x4096xf32>
    return %2 : tensor<8x4096x4096xf32>

    // CHECK:       [[FC:%.+]] = IE.FullyConnected
    // CHECK:       [[AFFINE:%.+]] = IE.AffineReshape([[FC]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 4096, 4096]} : tensor<4096x4096xf32> -> tensor<1x1x4096x4096xf32>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[AFFINE]])
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SOFTMAX]],
}

// -----

// CHECK-LABEL: @MergeParallelLayers
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x2x256xf32>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1537x256xf32>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x1537xf32>,
// CHECK-SAME:      [[INPUT_3:%.+]]: tensor<1x1537x256xf32>
func.func @MergeParallelLayers(%arg0: tensor<1x1x2x256xf32>, %arg1: tensor<1x1537x256xf32>, %arg2: tensor<1x1537xf32>, %arg3: tensor<1x1537x256xf32>) -> tensor<1x1x512xf32> {
    %cst_20 = const.Declare tensor<1x1x1xf32> = dense<0.1> : tensor<1x1x1xf32> isSplat
    %cst_21 = const.Declare tensor<1x1x1xf32> = dense<0.2> : tensor<1x1x1xf32> isSplat
    %cst_22 = const.Declare tensor<1x1x1xf32> = dense<0.3> : tensor<1x1x1xf32> isSplat
    %cst_23 = const.Declare tensor<1x1x1xf32> = dense<0.4> : tensor<1x1x1xf32> isSplat

    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 1, 512]} : tensor<1x1x2x256xf32> -> tensor<1x1x512xf32>

    %1 = IE.StridedSlice(%0) {begin_mask = [1, 1], begins_attr = [0, 0, 0], ellipsis_mask = [0, 0], end_mask = [1, 1], ends_attr = [0, 0, 256], new_axis_mask = [0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0], strides_attr = [1, 1, 1]} : tensor<1x1x512xf32> -> tensor<1x1x256xf32>
    %2 = IE.Multiply(%1, %cst_23) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256xf32>, tensor<1x1x1xf32> -> tensor<1x1x256xf32>
    %3 = IE.Reshape(%arg1) {shape_value = [1537, 256]} : tensor<1x1537x256xf32> -> tensor<1537x256xf32>
    %4 = IE.Reshape(%2) {shape_value = [1, 256]} : tensor<1x1x256xf32> -> tensor<1x256xf32>
    %5 = IE.FullyConnected(%3, %4) : tensor<1537x256xf32>, tensor<1x256xf32> -> tensor<1537x1xf32>
    %6 = IE.Reshape(%5) {shape_value = [1, 1537, 1]} : tensor<1537x1xf32> -> tensor<1x1537x1xf32>
    %7 = IE.Tanh(%6) : tensor<1x1537x1xf32> -> tensor<1x1537x1xf32>
    %8 = IE.Multiply(%7, %cst_22) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1537x1xf32>, tensor<1x1x1xf32> -> tensor<1x1537x1xf32>
    %9 = IE.AffineReshape(%8) {dim_mapping = [[0, 1], [2], [2]], shape_value = [1, 1, 1537]} : tensor<1x1537x1xf32> -> tensor<1x1x1537xf32>
    %10 = IE.Add(%9, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1537xf32>, tensor<1x1537xf32> -> tensor<1x1x1537xf32>
    %11 = IE.SoftMax(%10) {axisInd = 2 : i64} : tensor<1x1x1537xf32> -> tensor<1x1x1537xf32>
    %12 = IE.Transpose(%arg3) {order_value = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : tensor<1x1537x256xf32> -> tensor<1x256x1537xf32>
    %13 = IE.Reshape(%11) {shape_value = [1, 1537]} : tensor<1x1x1537xf32> -> tensor<1x1537xf32>
    %14 = IE.Reshape(%12) {shape_value = [256, 1537]} : tensor<1x256x1537xf32> -> tensor<256x1537xf32>
    %15 = IE.FullyConnected(%13, %14) : tensor<1x1537xf32>, tensor<256x1537xf32> -> tensor<1x256xf32>
    %16 = IE.Reshape(%15) {shape_value = [1, 1, 256]} : tensor<1x256xf32> -> tensor<1x1x256xf32>


    %17 = IE.StridedSlice(%0) {begin_mask = [1, 1], begins_attr = [0, 0, 256], ellipsis_mask = [0, 0], end_mask = [1, 1], ends_attr = [0, 0, 512], new_axis_mask = [0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0], strides_attr = [1, 1, 1]} : tensor<1x1x512xf32> -> tensor<1x1x256xf32>
    %18 = IE.Multiply(%17, %cst_21) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256xf32>, tensor<1x1x1xf32> -> tensor<1x1x256xf32>
    %19 = IE.Reshape(%arg1) {shape_value = [1537, 256]} : tensor<1x1537x256xf32> -> tensor<1537x256xf32>
    %20 = IE.Reshape(%18) {shape_value = [1, 256]} : tensor<1x1x256xf32> -> tensor<1x256xf32>
    %21 = IE.FullyConnected(%19, %20) : tensor<1537x256xf32>, tensor<1x256xf32> -> tensor<1537x1xf32>
    %22 = IE.Reshape(%21) {shape_value = [1, 1537, 1]} : tensor<1537x1xf32> -> tensor<1x1537x1xf32>
    %23 = IE.Tanh(%22) : tensor<1x1537x1xf32> -> tensor<1x1537x1xf32>
    %24 = IE.Multiply(%23, %cst_20) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1537x1xf32>, tensor<1x1x1xf32> -> tensor<1x1537x1xf32>
    %25 = IE.AffineReshape(%24) {dim_mapping = [[0, 1], [2], [2]], shape_value = [1, 1, 1537]} : tensor<1x1537x1xf32> -> tensor<1x1x1537xf32>
    %26 = IE.Add(%25, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1537xf32>, tensor<1x1537xf32> -> tensor<1x1x1537xf32>
    %27 = IE.SoftMax(%26) {axisInd = 2 : i64} : tensor<1x1x1537xf32> -> tensor<1x1x1537xf32>
    %28 = IE.Transpose(%arg3) {order_value = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : tensor<1x1537x256xf32> -> tensor<1x256x1537xf32>
    %29 = IE.Reshape(%27) {shape_value = [1, 1537]} : tensor<1x1x1537xf32> -> tensor<1x1537xf32>
    %30 = IE.Reshape(%28) {shape_value = [256, 1537]} : tensor<1x256x1537xf32> -> tensor<256x1537xf32>
    %31 = IE.FullyConnected(%29, %30) : tensor<1x1537xf32>, tensor<256x1537xf32> -> tensor<1x256xf32>
    %32 = IE.Reshape(%31) {shape_value = [1, 1, 256]} : tensor<1x256xf32> -> tensor<1x1x256xf32>

    %33 = IE.Concat(%16, %32) {static_offsets = [[0, 0, 0], [0, 0, 256]]} : tensor<1x1x256xf32>, tensor<1x1x256xf32> -> tensor<1x1x512xf32>

    return %33 : tensor<1x1x512xf32>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x2x1xf32> =
    // CHECK-SAME{LITERAL}:     dense<[[[[3.000000e-01], [1.000000e-01]]]]> : tensor<1x1x2x1xf32>
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<1x1x2x1xf32> =
    // CHECK-SAME{LITERAL}:     dense<[[[[4.000000e-01], [2.000000e-01]]]]> : tensor<1x1x2x1xf32>

    // CHECK:       [[MUL_0:%.+]] = IE.Multiply([[INPUT_0]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x256xf32>, tensor<1x1x2x1xf32> -> tensor<1x1x2x256xf32>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1]], shape_value = [1537, 256]} : tensor<1x1537x256xf32> -> tensor<1537x256xf32>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[MUL_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [2, 256]} : tensor<1x1x2x256xf32> -> tensor<2x256xf32>
    // CHECK:       [[FC_0:%.+]] = IE.FullyConnected([[RESHAPE_0]], [[RESHAPE_1]]) : tensor<1537x256xf32>, tensor<2x256xf32> -> tensor<1537x2xf32>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[FC_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2]], shape_value = [1, 1537, 2]} : tensor<1537x2xf32> -> tensor<1x1537x2xf32>

    // CHECK:       [[TANH:%.+]] = IE.Tanh([[RESHAPE_2]]) : tensor<1x1537x2xf32> -> tensor<1x1537x2xf32>

    // CHECK:       [[RESHAPE_3:%.+]] = IE.AffineReshape([[TANH]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1537, 2, 1]} : tensor<1x1537x2xf32> -> tensor<1x1537x2x1xf32>
    // CHECK:       [[MUL_1:%.+]] = IE.Multiply([[RESHAPE_3]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1537x2x1xf32>, tensor<1x1x2x1xf32> -> tensor<1x1537x2x1xf32>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.AffineReshape([[MUL_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 1537, 2]} : tensor<1x1537x2x1xf32> -> tensor<1x1537x2xf32>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[RESHAPE_4]] [1, 0, 0] [1, 1537, 1] : tensor<1x1537x2xf32> to tensor<1x1537x1xf32>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[RESHAPE_4]] [1, 0, 1] [1, 1537, 1] : tensor<1x1537x2xf32> to tensor<1x1537x1xf32>

    // CHECK:       [[RESHAPE_5:%.+]] = IE.AffineReshape([[SLICE_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [2]], shape_value = [1, 1, 1537]} : tensor<1x1537x1xf32> -> tensor<1x1x1537xf32>

    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT_3]]) {order_value = #map} : tensor<1x1537x256xf32> -> tensor<1x256x1537xf32>
    // CHECK:       [[RESHAPE_6:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1]], shape_value = [256, 1537]} : tensor<1x256x1537xf32> -> tensor<256x1537xf32>

    // CHECK:       [[RESHAPE_7:%.+]] = IE.AffineReshape([[SLICE_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [2]], shape_value = [1, 1, 1537]} : tensor<1x1537x1xf32> -> tensor<1x1x1537xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_5]], [[RESHAPE_7]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0], [0, 1, 0]]} : tensor<1x1x1537xf32>, tensor<1x1x1537xf32> -> tensor<1x2x1537xf32>

    // CHECK:       [[ADD:%.+]] = IE.Add([[CONCAT]], [[INPUT_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1537xf32>, tensor<1x1537xf32> -> tensor<1x2x1537xf32>

    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 2 : i64} : tensor<1x2x1537xf32> -> tensor<1x2x1537xf32>

    // CHECK:       [[RESHAPE_8:%.+]] = IE.AffineReshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1]], shape_value = [2, 1537]} : tensor<1x2x1537xf32> -> tensor<2x1537xf32>
    // CHECK:       [[FC_1:%.+]] = IE.FullyConnected([[RESHAPE_8]], [[RESHAPE_6]]) : tensor<2x1537xf32>, tensor<256x1537xf32> -> tensor<2x256xf32>
    // CHECK:       [[OUT_RESHAPE:%.+]] = IE.Reshape([[FC_1]]) {shape_value = [1, 1, 512]} : tensor<2x256xf32> -> tensor<1x1x512xf32>

    // CHECK:       return [[OUT_RESHAPE]] : tensor<1x1x512xf32>
}
