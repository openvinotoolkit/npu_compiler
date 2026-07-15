//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --initial-transformations %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FullyConnected
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16xf32>
func.func @FullyConnected(%arg0: tensor<1x16xf32>) -> tensor<1x64xf32> {
    %weights = const.Declare tensor<64x16xf32> = dense<1.0> : tensor<64x16xf32>
    %bias = const.Declare tensor<1x64xf32> = dense<1.0> : tensor<1x64xf32>
    %0 = IE.FullyConnected(%arg0, %weights, %bias) : tensor<1x16xf32>, tensor<64x16xf32>, tensor<1x64xf32> -> tensor<1x64xf32>

    return %0 : tensor<1x64xf32>

    // CHECK-NOT:   IE.Convolution
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<64x16xf32> = dense<1.000000e+00> : tensor<64x16xf32>
    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x64xf32> = dense<1.000000e+00> : tensor<1x64xf32>
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[ARG_0]], [[WEIGHTS]], [[BIAS]])
    // CHECK:       return [[FC]]
}

// -----

// CHECK-LABEL: @MatMul4dInputsTo2d
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x2x1x512xf32>
func.func @MatMul4dInputsTo2d(%arg0: tensor<1x2x1x512xf32>) -> tensor<1x2x1x40xf32> {
    %cst = const.Declare tensor<1x2x512x40xf32> = dense<1.0> : tensor<1x2x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x1x512xf32>, tensor<1x2x512x40xf32> -> tensor<1x2x1x40xf32>

    return %0 : tensor<1x2x1x40xf32>

    // CHECK-DAG:      [[CST_0:%.+]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00>
    // CHECK-DAG:      [[CST_1:%.+]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00>
    // CHECK:          [[IN_1:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:          [[IN_1_2D:%.+]] = IE.AffineReshape([[IN_1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:          [[IN_2:%.+]] = IE.Slice [[ARG_0]] [0, 1, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
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
    %cst_20 = const.Declare tensor<1x1x1xf32> = dense<0.1> : tensor<1x1x1xf32>
    %cst_21 = const.Declare tensor<1x1x1xf32> = dense<0.2> : tensor<1x1x1xf32>
    %cst_22 = const.Declare tensor<1x1x1xf32> = dense<0.3> : tensor<1x1x1xf32>
    %cst_23 = const.Declare tensor<1x1x1xf32> = dense<0.4> : tensor<1x1x1xf32>

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

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x2x1xf32>
    // CHECK-SAME:       #const.Concat
    // CHECK-SAME:       dense<3.000000e-01>
    // CHECK-SAME:       dense<1.000000e-01>
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<1x1x2x1xf32>
    // CHECK-SAME:       #const.Concat
    // CHECK-SAME:       dense<4.000000e-01>
    // CHECK-SAME:       dense<2.000000e-01>

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

// -----

// CHECK-LABEL: @UnrollSDPAPattern
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x8x128x128xf32>, [[ARG1:%.+]]: tensor<1x8x128x128xf32>, [[MASK:%.+]]: tensor<1x1x128x128xf32>, [[V_IN:%.+]]: tensor<1x1x1x128x128xf32>)
func.func @UnrollSDPAPattern(%arg0: tensor<1x8x128x128xf32>, %arg1: tensor<1x8x128x128xf32>, %mask: tensor<1x1x128x128xf32>, %v_in: tensor<1x1x1x128x128xf32>) -> tensor<1x8x128x128xf32> {
  %target_shape = const.Declare tensor<5xsi64> = dense<[1, 1, 8, 128, 128]> : tensor<5xsi64>
  %broadcast = IE.Broadcast(%v_in, %target_shape) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x1x1x128x128xf32>, tensor<5xsi64> -> tensor<1x1x8x128x128xf32>
  %reshape = IE.AffineReshape(%broadcast) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [1, 8, 128, 128]} : tensor<1x1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %transpose = IE.Transpose(%reshape) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %1 = IE.Add(%0, %mask) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x128x128xf32>, tensor<1x1x128x128xf32> -> tensor<1x8x128x128xf32>
  %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  %3 = IE.MatMul(%2, %transpose) {transpose_b} : tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
  return %3 : tensor<1x8x128x128xf32>

  // CHECK: [[TARGET_SHAPE:%.+]] = const.Declare
  // CHECK: [[BROADCAST:%.+]] = IE.Broadcast([[V_IN]], [[TARGET_SHAPE]])
  // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[BROADCAST]])
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])

  // CHECK: [[SLICE_Q_0:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_1:%.+]] = IE.Slice [[ARG0]] [0, 1, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_2:%.+]] = IE.Slice [[ARG0]] [0, 2, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_3:%.+]] = IE.Slice [[ARG0]] [0, 3, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_4:%.+]] = IE.Slice [[ARG0]] [0, 4, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_5:%.+]] = IE.Slice [[ARG0]] [0, 5, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_6:%.+]] = IE.Slice [[ARG0]] [0, 6, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_Q_7:%.+]] = IE.Slice [[ARG0]] [0, 7, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_0:%.+]] = IE.Slice [[ARG1]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_1:%.+]] = IE.Slice [[ARG1]] [0, 1, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_2:%.+]] = IE.Slice [[ARG1]] [0, 2, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_3:%.+]] = IE.Slice [[ARG1]] [0, 3, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_4:%.+]] = IE.Slice [[ARG1]] [0, 4, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_5:%.+]] = IE.Slice [[ARG1]] [0, 5, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_6:%.+]] = IE.Slice [[ARG1]] [0, 6, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_K_7:%.+]] = IE.Slice [[ARG1]] [0, 7, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_0:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_1:%.+]] = IE.Slice [[TRANSPOSE]] [0, 1, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_2:%.+]] = IE.Slice [[TRANSPOSE]] [0, 2, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_3:%.+]] = IE.Slice [[TRANSPOSE]] [0, 3, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_4:%.+]] = IE.Slice [[TRANSPOSE]] [0, 4, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_5:%.+]] = IE.Slice [[TRANSPOSE]] [0, 5, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_6:%.+]] = IE.Slice [[TRANSPOSE]] [0, 6, 0, 0] [1, 1, 128, 128]
  // CHECK: [[SLICE_V_7:%.+]] = IE.Slice [[TRANSPOSE]] [0, 7, 0, 0] [1, 1, 128, 128]

  // CHECK:       [[RESHAPE_Q_0:%.+]] = IE.AffineReshape([[SLICE_Q_0]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_K_0:%.+]] = IE.AffineReshape([[SLICE_K_0]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_0:%.+]] = IE.FullyConnected([[RESHAPE_Q_0]], [[RESHAPE_K_0]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[FC_0]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>
  // CHECK:       [[ADD_0:%.+]] = IE.Add([[RESHAPE_0]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[SOFTMAX_0:%.+]] = IE.SoftMax([[ADD_0]]) {axisInd = 3 : i64}
  // CHECK:       [[RESHAPE_S0:%.+]] = IE.AffineReshape([[SOFTMAX_0]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_V_0:%.+]] = IE.AffineReshape([[SLICE_V_0]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_V_0:%.+]] = IE.FullyConnected([[RESHAPE_S0]], [[RESHAPE_V_0]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_OUT_0:%.+]] = IE.AffineReshape([[FC_V_0]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>

  // CHECK:       [[RESHAPE_Q_1:%.+]] = IE.AffineReshape([[SLICE_Q_1]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_K_1:%.+]] = IE.AffineReshape([[SLICE_K_1]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_1:%.+]] = IE.FullyConnected([[RESHAPE_Q_1]], [[RESHAPE_K_1]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[FC_1]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>
  // CHECK:       [[ADD_1:%.+]] = IE.Add([[RESHAPE_1]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[SOFTMAX_1:%.+]] = IE.SoftMax([[ADD_1]]) {axisInd = 3 : i64}
  // CHECK:       [[RESHAPE_S1:%.+]] = IE.AffineReshape([[SOFTMAX_1]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_V_1:%.+]] = IE.AffineReshape([[SLICE_V_1]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_V_1:%.+]] = IE.FullyConnected([[RESHAPE_S1]], [[RESHAPE_V_1]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_OUT_1:%.+]] = IE.AffineReshape([[FC_V_1]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>

  // CHECK:       [[RESHAPE_Q_2:%.+]] = IE.AffineReshape([[SLICE_Q_2]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_K_2:%.+]] = IE.AffineReshape([[SLICE_K_2]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_2:%.+]] = IE.FullyConnected([[RESHAPE_Q_2]], [[RESHAPE_K_2]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[FC_2]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>
  // CHECK:       [[ADD_2:%.+]] = IE.Add([[RESHAPE_2]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[SOFTMAX_2:%.+]] = IE.SoftMax([[ADD_2]]) {axisInd = 3 : i64}
  // CHECK:       [[RESHAPE_S2:%.+]] = IE.AffineReshape([[SOFTMAX_2]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_V_2:%.+]] = IE.AffineReshape([[SLICE_V_2]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_V_2:%.+]] = IE.FullyConnected([[RESHAPE_S2]], [[RESHAPE_V_2]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_OUT_2:%.+]] = IE.AffineReshape([[FC_V_2]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>

  // CHECK:       [[RESHAPE_Q_3:%.+]] = IE.AffineReshape([[SLICE_Q_3]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_K_3:%.+]] = IE.AffineReshape([[SLICE_K_3]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_3:%.+]] = IE.FullyConnected([[RESHAPE_Q_3]], [[RESHAPE_K_3]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_3:%.+]] = IE.AffineReshape([[FC_3]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>
  // CHECK:       [[ADD_3:%.+]] = IE.Add([[RESHAPE_3]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[SOFTMAX_3:%.+]] = IE.SoftMax([[ADD_3]]) {axisInd = 3 : i64}
  // CHECK:       [[RESHAPE_S3:%.+]] = IE.AffineReshape([[SOFTMAX_3]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_V_3:%.+]] = IE.AffineReshape([[SLICE_V_3]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_V_3:%.+]] = IE.FullyConnected([[RESHAPE_S3]], [[RESHAPE_V_3]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_OUT_3:%.+]] = IE.AffineReshape([[FC_V_3]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>

  // CHECK:       [[RESHAPE_Q_4:%.+]] = IE.AffineReshape([[SLICE_Q_4]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_K_4:%.+]] = IE.AffineReshape([[SLICE_K_4]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_4:%.+]] = IE.FullyConnected([[RESHAPE_Q_4]], [[RESHAPE_K_4]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_4:%.+]] = IE.AffineReshape([[FC_4]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>
  // CHECK:       [[ADD_4:%.+]] = IE.Add([[RESHAPE_4]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[SOFTMAX_4:%.+]] = IE.SoftMax([[ADD_4]]) {axisInd = 3 : i64}
  // CHECK:       [[RESHAPE_S4:%.+]] = IE.AffineReshape([[SOFTMAX_4]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_V_4:%.+]] = IE.AffineReshape([[SLICE_V_4]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_V_4:%.+]] = IE.FullyConnected([[RESHAPE_S4]], [[RESHAPE_V_4]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_OUT_4:%.+]] = IE.AffineReshape([[FC_V_4]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>

  // CHECK:       [[RESHAPE_Q_5:%.+]] = IE.AffineReshape([[SLICE_Q_5]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_K_5:%.+]] = IE.AffineReshape([[SLICE_K_5]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_5:%.+]] = IE.FullyConnected([[RESHAPE_Q_5]], [[RESHAPE_K_5]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_5:%.+]] = IE.AffineReshape([[FC_5]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>
  // CHECK:       [[ADD_5:%.+]] = IE.Add([[RESHAPE_5]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[SOFTMAX_5:%.+]] = IE.SoftMax([[ADD_5]]) {axisInd = 3 : i64}
  // CHECK:       [[RESHAPE_S5:%.+]] = IE.AffineReshape([[SOFTMAX_5]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_V_5:%.+]] = IE.AffineReshape([[SLICE_V_5]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_V_5:%.+]] = IE.FullyConnected([[RESHAPE_S5]], [[RESHAPE_V_5]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_OUT_5:%.+]] = IE.AffineReshape([[FC_V_5]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>

  // CHECK:       [[RESHAPE_Q_6:%.+]] = IE.AffineReshape([[SLICE_Q_6]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_K_6:%.+]] = IE.AffineReshape([[SLICE_K_6]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_6:%.+]] = IE.FullyConnected([[RESHAPE_Q_6]], [[RESHAPE_K_6]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_6:%.+]] = IE.AffineReshape([[FC_6]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>
  // CHECK:       [[ADD_6:%.+]] = IE.Add([[RESHAPE_6]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[SOFTMAX_6:%.+]] = IE.SoftMax([[ADD_6]]) {axisInd = 3 : i64}
  // CHECK:       [[RESHAPE_S6:%.+]] = IE.AffineReshape([[SOFTMAX_6]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_V_6:%.+]] = IE.AffineReshape([[SLICE_V_6]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_V_6:%.+]] = IE.FullyConnected([[RESHAPE_S6]], [[RESHAPE_V_6]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_OUT_6:%.+]] = IE.AffineReshape([[FC_V_6]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>

  // CHECK:       [[RESHAPE_Q_7:%.+]] = IE.AffineReshape([[SLICE_Q_7]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_K_7:%.+]] = IE.AffineReshape([[SLICE_K_7]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_7:%.+]] = IE.FullyConnected([[RESHAPE_Q_7]], [[RESHAPE_K_7]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_7:%.+]] = IE.AffineReshape([[FC_7]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>
  // CHECK:       [[ADD_7:%.+]] = IE.Add([[RESHAPE_7]], [[MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[SOFTMAX_7:%.+]] = IE.SoftMax([[ADD_7]]) {axisInd = 3 : i64}
  // CHECK:       [[RESHAPE_S7:%.+]] = IE.AffineReshape([[SOFTMAX_7]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_V_7:%.+]] = IE.AffineReshape([[SLICE_V_7]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 128]} : tensor<1x1x128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[FC_V_7:%.+]] = IE.FullyConnected([[RESHAPE_S7]], [[RESHAPE_V_7]]) : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK:       [[RESHAPE_OUT_7:%.+]] = IE.AffineReshape([[FC_V_7]])
  // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 128, 128]} : tensor<128x128xf32> -> tensor<1x1x128x128xf32>

  // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_OUT_0]], [[RESHAPE_OUT_1]], [[RESHAPE_OUT_2]], [[RESHAPE_OUT_3]], [[RESHAPE_OUT_4]], [[RESHAPE_OUT_5]], [[RESHAPE_OUT_6]], [[RESHAPE_OUT_7]])
  // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0], [0, 4, 0, 0], [0, 5, 0, 0], [0, 6, 0, 0], [0, 7, 0, 0]]} : tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32>, tensor<1x1x128x128xf32> -> tensor<1x8x128x128xf32>
  // CHECK:       return [[CONCAT]]
}
