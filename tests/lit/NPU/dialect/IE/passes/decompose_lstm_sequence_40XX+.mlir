//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-lstm-sequence %s | FileCheck %s
// REQUIRES: arch-NPU40XX
// CHECK-LABEL: func.func @DecomposeLSTMSequence(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x64xf32>) -> (tensor<1x2x3x128xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>) {
func.func @DecomposeLSTMSequence(%arg0: tensor<1x3x64xf32>) -> (tensor<1x2x3x128xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>) {
    %cst_0 = const.Declare tensor<1x2x128xf32> = dense<1.000000e+00> : tensor<1x2x128xf32>
    %cst_1 = const.Declare tensor<2x512x64xf32> = dense<2.000000e+00> : tensor<2x512x64xf32>
    %cst_2 = const.Declare tensor<2x512x128xf32> = dense<3.000000e+00> : tensor<2x512x128xf32>
    %cst_3 = const.Declare tensor<2x512xf32> = dense<4.000000e+00> : tensor<2x512xf32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_1, %cst_2, %cst_3) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 3 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : tensor<1x3x64xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>, tensor<2x512x64xf32>, tensor<2x512x128xf32>, tensor<2x512xf32> -> tensor<1x2x3x128xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x2x3x128xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>

// CHECK:   %[[VAL_1:.*]] = const.Declare tensor<2x1x512xf32> = dense<4.000000e+00> : tensor<2x512xf32>, [#const.Reshape<[2, 1, 512]>]
// CHECK:   %[[VAL_2:.*]] = const.Declare tensor<4xsi32> = dense<[1, 2, 3, 64]> : tensor<4xsi64>, [#const.CastElemType<si32>]
// CHECK:   %[[VAL_3:.*]] = const.Declare tensor<1x2x512x64xf32> = dense<2.000000e+00> : tensor<2x512x64xf32>, [#const.Reshape<[1, 2, 512, 64]>]
// CHECK:   %[[VAL_4:.*]] = const.Declare tensor<1x2x128xf32> = dense<1.000000e+00> : tensor<1x2x128xf32>
// CHECK:   %[[VAL_5:.*]] = const.Declare tensor<2x512x128xf32> = dense<3.000000e+00> : tensor<2x512x128xf32>
// CHECK:   %[[VAL_6:.*]] = IE.Unsqueeze(%[[VAL_0]]) {axes_value = [1]} : tensor<1x3x64xf32> -> tensor<1x1x3x64xf32>
// CHECK:   %[[VAL_7:.*]] = IE.Broadcast(%[[VAL_6]], %[[VAL_2]]) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x1x3x64xf32>, tensor<4xsi32> -> tensor<1x2x3x64xf32>
// CHECK:   %[[VAL_8:.*]] = IE.MatMul(%[[VAL_7]], %[[VAL_3]]) {transpose_b} : tensor<1x2x3x64xf32>, tensor<1x2x512x64xf32> -> tensor<1x2x3x512xf32>
// CHECK:   %[[VAL_9:.*]] = IE.Add(%[[VAL_8]], %[[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x3x512xf32>, tensor<2x1x512xf32> -> tensor<1x2x3x512xf32>
// CHECK:   %[[VAL_10:.*]], %[[VAL_11:.*]], %[[VAL_12:.*]] = IE.LSTMSequence(%[[VAL_9]], %[[VAL_4]], %[[VAL_4]], %[[VAL_5]]) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>, sequenceLength = 3 : i64} : tensor<1x2x3x512xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>, tensor<2x512x128xf32> -> tensor<1x2x3x128xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>
// CHECK:   return %[[VAL_10]], %[[VAL_11]], %[[VAL_12]] : tensor<1x2x3x128xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>
}

// -----

// CHECK-LABEL: func.func @DecomposeUnsupportedLSTMSequenceToLSTMCells(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x64xf32>) -> (tensor<1x2x3x300xf32>, tensor<1x2x300xf32>, tensor<1x2x300xf32>) {
func.func @DecomposeUnsupportedLSTMSequenceToLSTMCells(%arg0: tensor<1x3x64xf32>) -> (tensor<1x2x3x300xf32>, tensor<1x2x300xf32>, tensor<1x2x300xf32>) {
    %cst_0 = const.Declare tensor<1x2x300xf32> = dense<1.000000e+00> : tensor<1x2x300xf32>
    %cst_1 = const.Declare tensor<2x512x64xf32> = dense<2.000000e+00> : tensor<2x512x64xf32>
    %cst_2 = const.Declare tensor<2x512x300xf32> = dense<3.000000e+00> : tensor<2x512x300xf32>
    %cst_3 = const.Declare tensor<2x512xf32> = dense<4.000000e+00> : tensor<2x512xf32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_1, %cst_2, %cst_3) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 3 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : tensor<1x3x64xf32>, tensor<1x2x300xf32>, tensor<1x2x300xf32>, tensor<2x512x64xf32>, tensor<2x512x300xf32>, tensor<2x512xf32> -> tensor<1x2x3x300xf32>, tensor<1x2x300xf32>, tensor<1x2x300xf32>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x2x3x300xf32>, tensor<1x2x300xf32>, tensor<1x2x300xf32>

// CHECK:   %[[VAL_1:.*]] = const.Declare tensor<512x300xf32> = dense<3.000000e+00> : tensor<2x512x300xf32>, [#const.SubView<[1, 0, 0], [1, 512, 300]>, #const.Reshape<[512, 300]>]
// CHECK:   %[[VAL_2:.*]] = const.Declare tensor<1x300xf32> = dense<1.000000e+00> : tensor<1x2x300xf32>, [#const.SubView<[0, 1, 0], [1, 1, 300]>, #const.Reshape<[1, 300]>]
// CHECK:   %[[VAL_3:.*]] = const.Declare tensor<512x300xf32> = dense<3.000000e+00> : tensor<2x512x300xf32>, [#const.SubView<[0, 0, 0], [1, 512, 300]>, #const.Reshape<[512, 300]>]
// CHECK:   %[[VAL_4:.*]] = const.Declare tensor<1x300xf32> = dense<1.000000e+00> : tensor<1x2x300xf32>, [#const.SubView<[0, 0, 0], [1, 1, 300]>, #const.Reshape<[1, 300]>]
// CHECK:   %[[VAL_5:.*]] = const.Declare tensor<2x1x512xf32> = dense<4.000000e+00> : tensor<2x512xf32>, [#const.Reshape<[2, 1, 512]>]
// CHECK:   %[[VAL_6:.*]] = const.Declare tensor<4xsi32> = dense<[1, 2, 3, 64]> : tensor<4xsi64>, [#const.CastElemType<si32>]
// CHECK:   %[[VAL_7:.*]] = const.Declare tensor<1x2x512x64xf32> = dense<2.000000e+00> : tensor<2x512x64xf32>, [#const.Reshape<[1, 2, 512, 64]>]
// CHECK:   %[[VAL_8:.*]] = IE.Unsqueeze(%[[VAL_0]]) {axes_value = [1]} : tensor<1x3x64xf32> -> tensor<1x1x3x64xf32>
// CHECK:   %[[VAL_9:.*]] = IE.Broadcast(%[[VAL_8]], %[[VAL_6]]) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x1x3x64xf32>, tensor<4xsi32> -> tensor<1x2x3x64xf32>
// CHECK:   %[[VAL_10:.*]] = IE.MatMul(%[[VAL_9]], %[[VAL_7]]) {transpose_b} : tensor<1x2x3x64xf32>, tensor<1x2x512x64xf32> -> tensor<1x2x3x512xf32>
// CHECK:   %[[VAL_11:.*]] = IE.Add(%[[VAL_10]], %[[VAL_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x3x512xf32>, tensor<2x1x512xf32> -> tensor<1x2x3x512xf32>
// CHECK:   %[[VAL_12:.*]] = IE.Slice %[[VAL_11]] [0, 0, 0, 0] [1, 1, 3, 512] : tensor<1x2x3x512xf32> to tensor<1x1x3x512xf32>
// CHECK:   %[[VAL_13:.*]] = IE.Slice %[[VAL_11]] [0, 1, 0, 0] [1, 1, 3, 512] : tensor<1x2x3x512xf32> to tensor<1x1x3x512xf32>
// CHECK:   %[[VAL_14:.*]] = IE.Squeeze(%[[VAL_12]]) {axes_value = [1]} : tensor<1x1x3x512xf32> -> tensor<1x3x512xf32>
// CHECK:   %[[VAL_15:.*]] = IE.Slice %[[VAL_14]] [0, 0, 0] [1, 1, 512] : tensor<1x3x512xf32> to tensor<1x1x512xf32>
// CHECK:   %[[VAL_16:.*]] = IE.Squeeze(%[[VAL_15]]) {axes_value = [1]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
// CHECK:   %[[VAL_17:.*]], %[[VAL_18:.*]] = IE.LSTMCell(%[[VAL_16]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]]) {hiddenSize = 300 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x512xf32>, tensor<1x300xf32>, tensor<1x300xf32>, tensor<512x300xf32> -> tensor<1x300xf32>, tensor<1x300xf32>
// CHECK:   %[[VAL_19:.*]] = IE.Unsqueeze(%[[VAL_17]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_20:.*]] = IE.Slice %[[VAL_14]] [0, 1, 0] [1, 1, 512] : tensor<1x3x512xf32> to tensor<1x1x512xf32>
// CHECK:   %[[VAL_21:.*]] = IE.Squeeze(%[[VAL_20]]) {axes_value = [1]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
// CHECK:   %[[VAL_22:.*]], %[[VAL_23:.*]] = IE.LSTMCell(%[[VAL_21]], %[[VAL_17]], %[[VAL_18]], %[[VAL_3]]) {hiddenSize = 300 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x512xf32>, tensor<1x300xf32>, tensor<1x300xf32>, tensor<512x300xf32> -> tensor<1x300xf32>, tensor<1x300xf32>
// CHECK:   %[[VAL_24:.*]] = IE.Unsqueeze(%[[VAL_22]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_25:.*]] = IE.Slice %[[VAL_14]] [0, 2, 0] [1, 1, 512] : tensor<1x3x512xf32> to tensor<1x1x512xf32>
// CHECK:   %[[VAL_26:.*]] = IE.Squeeze(%[[VAL_25]]) {axes_value = [1]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
// CHECK:   %[[VAL_27:.*]], %[[VAL_28:.*]] = IE.LSTMCell(%[[VAL_26]], %[[VAL_22]], %[[VAL_23]], %[[VAL_3]]) {hiddenSize = 300 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x512xf32>, tensor<1x300xf32>, tensor<1x300xf32>, tensor<512x300xf32> -> tensor<1x300xf32>, tensor<1x300xf32>
// CHECK:   %[[VAL_29:.*]] = IE.Unsqueeze(%[[VAL_27]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_30:.*]] = IE.Concat(%[[VAL_19]], %[[VAL_24]], %[[VAL_29]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x300xf32>, tensor<1x1x300xf32>, tensor<1x1x300xf32> -> tensor<1x3x300xf32>
// CHECK:   %[[VAL_31:.*]] = IE.Unsqueeze(%[[VAL_30]]) {axes_value = [1]} : tensor<1x3x300xf32> -> tensor<1x1x3x300xf32>
// CHECK:   %[[VAL_32:.*]] = IE.Unsqueeze(%[[VAL_27]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_33:.*]] = IE.Unsqueeze(%[[VAL_28]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_34:.*]] = IE.Squeeze(%[[VAL_13]]) {axes_value = [1]} : tensor<1x1x3x512xf32> -> tensor<1x3x512xf32>
// CHECK:   %[[VAL_35:.*]] = IE.Slice %[[VAL_34]] [0, 2, 0] [1, 1, 512] : tensor<1x3x512xf32> to tensor<1x1x512xf32>
// CHECK:   %[[VAL_36:.*]] = IE.Squeeze(%[[VAL_35]]) {axes_value = [1]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
// CHECK:   %[[VAL_37:.*]], %[[VAL_38:.*]] = IE.LSTMCell(%[[VAL_36]], %[[VAL_2]], %[[VAL_2]], %[[VAL_1]]) {hiddenSize = 300 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x512xf32>, tensor<1x300xf32>, tensor<1x300xf32>, tensor<512x300xf32> -> tensor<1x300xf32>, tensor<1x300xf32>
// CHECK:   %[[VAL_39:.*]] = IE.Unsqueeze(%[[VAL_37]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_40:.*]] = IE.Slice %[[VAL_34]] [0, 1, 0] [1, 1, 512] : tensor<1x3x512xf32> to tensor<1x1x512xf32>
// CHECK:   %[[VAL_41:.*]] = IE.Squeeze(%[[VAL_40]]) {axes_value = [1]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
// CHECK:   %[[VAL_42:.*]], %[[VAL_43:.*]] = IE.LSTMCell(%[[VAL_41]], %[[VAL_37]], %[[VAL_38]], %[[VAL_1]]) {hiddenSize = 300 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x512xf32>, tensor<1x300xf32>, tensor<1x300xf32>, tensor<512x300xf32> -> tensor<1x300xf32>, tensor<1x300xf32>
// CHECK:   %[[VAL_44:.*]] = IE.Unsqueeze(%[[VAL_42]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_45:.*]] = IE.Slice %[[VAL_34]] [0, 0, 0] [1, 1, 512] : tensor<1x3x512xf32> to tensor<1x1x512xf32>
// CHECK:   %[[VAL_46:.*]] = IE.Squeeze(%[[VAL_45]]) {axes_value = [1]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
// CHECK:   %[[VAL_47:.*]], %[[VAL_48:.*]] = IE.LSTMCell(%[[VAL_46]], %[[VAL_42]], %[[VAL_43]], %[[VAL_1]]) {hiddenSize = 300 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x512xf32>, tensor<1x300xf32>, tensor<1x300xf32>, tensor<512x300xf32> -> tensor<1x300xf32>, tensor<1x300xf32>
// CHECK:   %[[VAL_49:.*]] = IE.Unsqueeze(%[[VAL_47]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_50:.*]] = IE.Concat(%[[VAL_49]], %[[VAL_44]], %[[VAL_39]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x300xf32>, tensor<1x1x300xf32>, tensor<1x1x300xf32> -> tensor<1x3x300xf32>
// CHECK:   %[[VAL_51:.*]] = IE.Unsqueeze(%[[VAL_50]]) {axes_value = [1]} : tensor<1x3x300xf32> -> tensor<1x1x3x300xf32>
// CHECK:   %[[VAL_52:.*]] = IE.Unsqueeze(%[[VAL_47]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_53:.*]] = IE.Unsqueeze(%[[VAL_48]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_54:.*]] = IE.Concat(%[[VAL_31]], %[[VAL_51]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x3x300xf32>, tensor<1x1x3x300xf32> -> tensor<1x2x3x300xf32>
// CHECK:   %[[VAL_55:.*]] = IE.Concat(%[[VAL_32]], %[[VAL_52]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x300xf32>, tensor<1x1x300xf32> -> tensor<1x2x300xf32>
// CHECK:   %[[VAL_56:.*]] = IE.Concat(%[[VAL_33]], %[[VAL_53]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x300xf32>, tensor<1x1x300xf32> -> tensor<1x2x300xf32>
// CHECK:   return %[[VAL_54]], %[[VAL_55]], %[[VAL_56]] : tensor<1x2x3x300xf32>, tensor<1x2x300xf32>, tensor<1x2x300xf32>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @DecomposeDynamicLSTMSequence(
// CHECK-SAME:  [[ARG_0:.+]]: tensor<1x?x512xf32, {bounds = [1, 35, 512], order = #CHW}>, [[ARG_1:.+]]: tensor<1x1x128xf32>, [[ARG_2:.+]]: tensor<1x1x128xf32>) -> (tensor<1x1x?x128xf32, {bounds = [1, 1, 35, 128], order = #NCHW}>, tensor<1x1x128xf32>, tensor<1x1x128xf32>) {
func.func @DecomposeDynamicLSTMSequence(%arg0: tensor<1x?x512xf32, {bounds = [1, 35, 512], order = #CHW}>, %arg1: tensor<1x1x128xf32>, %arg2: tensor<1x1x128xf32>) -> (tensor<1x1x?x128xf32, {bounds = [1, 1, 35, 128], order = #NCHW}>, tensor<1x1x128xf32>, tensor<1x1x128xf32>) {
    %0 = IE.ShapeOf(%arg0) {dstElemType = si64} : tensor<1x?x512xf32, {bounds = [1, 35, 512], order = #CHW}> -> tensor<3xsi64>
    %cst = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    %cst_0 = const.Declare tensor<si64> = dense<0> : tensor<si64>
    %cst_1 = const.Declare tensor<1x512x512xf32> = dense<0.000000e+00> : tensor<1x512x512xf32>
    %cst_2 = const.Declare tensor<1x512x128xf32> = dense<0.000000e+00> : tensor<1x512x128xf32>
    %cst_3 = const.Declare tensor<1x512xf32> = dense<0.000000e+00> : tensor<1x512xf32>
    %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %arg1, %arg2, %cst_1, %cst_2, %cst_3) {direction = #IE.rnn_seq_direction<REVERSE>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : tensor<1x?x512xf32, {bounds = [1, 35, 512], order = #CHW}>, tensor<1x1x128xf32>, tensor<1x1x128xf32>, tensor<1x512x512xf32>, tensor<1x512x128xf32>, tensor<1x512xf32> -> tensor<1x1x?x128xf32, {bounds = [1, 1, 35, 128], order = #NCHW}>, tensor<1x1x128xf32>, tensor<1x1x128xf32>
    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x1x?x128xf32, {bounds = [1, 1, 35, 128], order = #NCHW}>, tensor<1x1x128xf32>, tensor<1x1x128xf32>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<1xsi64> = dense<[1, 1, 35, 512]> : tensor<4xsi64>, [#const.SubView<[3], [1]>]
    // CHECK:   [[CST_0:%.+]] = const.Declare tensor<2xsi64> = dense<[1, 1, 512, 512]> : tensor<4xsi64>, [#const.SubView<[0], [2]>]
    // CHECK:   [[CST_1:%.+]] = const.Declare tensor<1x1x512x512xf32> = dense<0.000000e+00> : tensor<1x512x512xf32>, [#const.Reshape<[1, 1, 512, 512]>]
    // CHECK:   [[CST_2:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[CST_3:%.+]] = const.Declare tensor<1x512x128xf32> = dense<0.000000e+00> : tensor<1x512x128xf32>
    // CHECK:   [[SHAPE_OF_0:%.+]] = IE.ShapeOf([[ARG_0]]) {dstElemType = si64} : tensor<1x?x512xf32, {bounds = [1, 35, 512], order = #CHW}> -> tensor<3xsi64>
    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[SHAPE_OF_0]] [0] [1] : tensor<3xsi64> to tensor<1xsi64>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[SHAPE_OF_0]] [1] [2] : tensor<3xsi64> to tensor<2xsi64>
    // CHECK:   [[CONCAT_0:%.+]] = IE.Concat([[SLICE_0]], [[CST_2]], [[SLICE_1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<2xsi64> -> tensor<4xsi64>
    // CHECK:   [[DYN_RESHAPE_0:%.+]] = IE.DynamicReshape([[ARG_0]], [[CONCAT_0]]) {output_bounds = [1, 1, 35, 512], output_shape = [1, 1, -9223372036854775808, 512]} : tensor<1x?x512xf32, {bounds = [1, 35, 512], order = #CHW}>, tensor<4xsi64> -> tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}>
    // CHECK:   [[DYN_EXPAND:%.+]] = IE.DynamicExpand([[DYN_RESHAPE_0]]) : tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}> -> tensor<1x1x35x512xf32>
    // CHECK:   [[MAT_MUL:%.+]] = IE.MatMul([[DYN_EXPAND]], [[CST_1]]) {transpose_b} : tensor<1x1x35x512xf32>, tensor<1x1x512x512xf32> -> tensor<1x1x35x512xf32>
    // CHECK:   [[SHAPE_OF_1:%.+]] = IE.ShapeOf([[DYN_RESHAPE_0]]) {dstElemType = si64} : tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}> -> tensor<4xsi64>
    // CHECK:   [[SLICE_3:%.+]] = IE.Slice [[SHAPE_OF_1]] [2] [1] : tensor<4xsi64> to tensor<1xsi64>
    // CHECK:   [[CONCAT_1:%.+]] = IE.Concat([[CST_0]], [[SLICE_3]], [[CST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<2xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>
    // CHECK:   [[DYN_RESHAPE_1:%.+]] = IE.DynamicReshape([[MAT_MUL]], [[CONCAT_1]]) {output_bounds = [1, 1, 35, 512], output_shape = [1, 1, -9223372036854775808, 512]} : tensor<1x1x35x512xf32>, tensor<4xsi64> -> tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}>
    // CHECK:   [[OUT_HV:%.+]], [[OUT_HS:%.+]], [[OUT_CS:%.+]] = IE.LSTMSequence([[DYN_RESHAPE_1]], [[ARG_1]], [[ARG_2]], [[CST_3]]) {direction = #IE.rnn_seq_direction<REVERSE>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}>, tensor<1x1x128xf32>, tensor<1x1x128xf32>, tensor<1x512x128xf32> -> tensor<1x1x?x128xf32, {bounds = [1, 1, 35, 128], order = #NCHW}>, tensor<1x1x128xf32>, tensor<1x1x128xf32>
    // CHECK:   return [[OUT_HV]], [[OUT_HS]], [[OUT_CS]] : tensor<1x1x?x128xf32, {bounds = [1, 1, 35, 128], order = #NCHW}>, tensor<1x1x128xf32>, tensor<1x1x128xf32>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @TestDecomposeLSTMSequence(
// CHECK-SAME:  [[ARG_0:.+]]: tensor<1x?x512xf32, {bounds = [1, 35, 512], order = #CHW}>, [[ARG_1:.+]]: tensor<1x2x128xf32>, [[ARG_2:.+]]: tensor<1x2x128xf32>) -> (tensor<1x2x?x128xf32, {bounds = [1, 2, 35, 128], order = #NCHW}>, tensor<1x2x128xf32>, tensor<1x2x128xf32>)
func.func @TestDecomposeLSTMSequence(%arg0: tensor<1x?x512xf32, {bounds = [1, 35, 512], order = #CHW}>, %arg1: tensor<1x2x128xf32>, %arg2: tensor<1x2x128xf32>) -> (tensor<1x2x?x128xf32, {bounds = [1, 2, 35, 128], order = #NCHW}>, tensor<1x2x128xf32>, tensor<1x2x128xf32>) {
    %cst = const.Declare tensor<2x512xf32> = dense<0.000000e+00> : tensor<2x512xf32>
    %cst_0 = const.Declare tensor<2x512x128xf32> = dense<0.000000e+00> : tensor<2x512x128xf32>
    %cst_1 = const.Declare tensor<2x512x512xf32> = dense<0.000000e+00> : tensor<2x512x512xf32>
    %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %arg1, %arg2, %cst_1, %cst_0, %cst) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : tensor<1x?x512xf32, {bounds = [1, 35, 512], order = #CHW}>, tensor<1x2x128xf32>, tensor<1x2x128xf32>, tensor<2x512x512xf32>, tensor<2x512x128xf32>, tensor<2x512xf32> -> tensor<1x2x?x128xf32, {bounds = [1, 2, 35, 128], order = #NCHW}>, tensor<1x2x128xf32>, tensor<1x2x128xf32>
    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x2x?x128xf32, {bounds = [1, 2, 35, 128], order = #NCHW}>, tensor<1x2x128xf32>, tensor<1x2x128xf32>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<1x1x512x512xf32> = dense<0.000000e+00> : tensor<2x512x512xf32>, [#const.SubView<[1, 0, 0], [1, 512, 512]>, #const.Reshape<[1, 1, 512, 512]>]
    // CHECK:   [[CST_0:%.+]] = const.Declare tensor<1xsi64> = dense<[1, 1, 35, 512]> : tensor<4xsi64>, [#const.SubView<[3], [1]>]
    // CHECK:   [[CST_1:%.+]] = const.Declare tensor<2xsi64> = dense<[1, 1, 512, 512]> : tensor<4xsi64>, [#const.SubView<[0], [2]>]
    // CHECK:   [[CST_2:%.+]] = const.Declare tensor<1x1x512x512xf32> = dense<0.000000e+00> : tensor<2x512x512xf32>, [#const.SubView<[0, 0, 0], [1, 512, 512]>, #const.Reshape<[1, 1, 512, 512]>]
    // CHECK:   [[CST_3:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:   [[CST_4:%.+]] = const.Declare tensor<1x512x128xf32> = dense<0.000000e+00> : tensor<2x512x128xf32>, [#const.SubView<[1, 0, 0], [1, 512, 128]>]
    // CHECK:   [[CST_5:%.+]] = const.Declare tensor<1x512x128xf32> = dense<0.000000e+00> : tensor<2x512x128xf32>, [#const.SubView<[0, 0, 0], [1, 512, 128]>]
    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[ARG_1]] [0, 0, 0] [1, 1, 128] : tensor<1x2x128xf32> to tensor<1x1x128xf32>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[ARG_1]] [0, 1, 0] [1, 1, 128] : tensor<1x2x128xf32> to tensor<1x1x128xf32>
    // CHECK:   [[SLICE_2:%.+]] = IE.Slice [[ARG_2]] [0, 0, 0] [1, 1, 128] : tensor<1x2x128xf32> to tensor<1x1x128xf32>
    // CHECK:   [[SLICE_3:%.+]] = IE.Slice [[ARG_2]] [0, 1, 0] [1, 1, 128] : tensor<1x2x128xf32> to tensor<1x1x128xf32>
    // CHECK:   [[SHAPE_OF_0:%.+]] = IE.ShapeOf([[ARG_0]]) {dstElemType = si64} : tensor<1x?x512xf32, {bounds = [1, 35, 512], order = #CHW}> -> tensor<3xsi64>
    // CHECK:   [[SLICE_4:%.+]] = IE.Slice [[SHAPE_OF_0]] [0] [1] : tensor<3xsi64> to tensor<1xsi64>
    // CHECK:   [[SLICE_5:%.+]] = IE.Slice [[SHAPE_OF_0]] [1] [2] : tensor<3xsi64> to tensor<2xsi64>
    // CHECK:   [[CONCAT_0:%.+]] = IE.Concat([[SLICE_4]], [[CST_3]], [[SLICE_5]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<2xsi64> -> tensor<4xsi64>
    // CHECK:   [[DYN_RESHAPE_0:%.+]] = IE.DynamicReshape([[ARG_0]], [[CONCAT_0]]) {output_bounds = [1, 1, 35, 512], output_shape = [1, 1, -9223372036854775808, 512]} : tensor<1x?x512xf32, {bounds = [1, 35, 512], order = #CHW}>, tensor<4xsi64> -> tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}>
    // CHECK:   [[EXPAND_0:%.+]] = IE.DynamicExpand([[DYN_RESHAPE_0]]) : tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}> -> tensor<1x1x35x512xf32>
    // CHECK:   [[MATMUL_0:%.+]] = IE.MatMul([[EXPAND_0]], [[CST_2]]) {transpose_b} : tensor<1x1x35x512xf32>, tensor<1x1x512x512xf32> -> tensor<1x1x35x512xf32>
    // CHECK:   [[SHAPE_OF_1:%.+]] = IE.ShapeOf([[DYN_RESHAPE_0]]) {dstElemType = si64} : tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}> -> tensor<4xsi64>
    // CHECK:   [[SLICE_6:%.+]] = IE.Slice [[SHAPE_OF_1]] [2] [1] : tensor<4xsi64> to tensor<1xsi64>
    // CHECK:   [[CONCAT_1:%.+]] = IE.Concat([[CST_1]], [[SLICE_6]], [[CST_0]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<2xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>
    // CHECK:   [[DYN_RESHAPE_1:%.+]] = IE.DynamicReshape([[MATMUL_0]], [[CONCAT_1]]) {output_bounds = [1, 1, 35, 512], output_shape = [1, 1, -9223372036854775808, 512]} : tensor<1x1x35x512xf32>, tensor<4xsi64> -> tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}>
    // CHECK:   [[OUT_HV_0:%.+]], [[OUT_HS_0:%.+]], [[OUT_CS_0:%.+]] = IE.LSTMSequence([[DYN_RESHAPE_1]], [[SLICE_0]], [[SLICE_2]], [[CST_5]]) {direction = #IE.rnn_seq_direction<FORWARD>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}>, tensor<1x1x128xf32>, tensor<1x1x128xf32>, tensor<1x512x128xf32> -> tensor<1x1x?x128xf32, {bounds = [1, 1, 35, 128], order = #NCHW}>, tensor<1x1x128xf32>, tensor<1x1x128xf32>
    // CHECK:   [[SHAPE_OF_2:%.+]] = IE.ShapeOf([[ARG_0]]) {dstElemType = si64} : tensor<1x?x512xf32, {bounds = [1, 35, 512], order = #CHW}> -> tensor<3xsi64>
    // CHECK:   [[SLICE_7:%.+]] = IE.Slice [[SHAPE_OF_2]] [0] [1] : tensor<3xsi64> to tensor<1xsi64>
    // CHECK:   [[SLICE_8:%.+]] = IE.Slice [[SHAPE_OF_2]] [1] [2] : tensor<3xsi64> to tensor<2xsi64>
    // CHECK:   [[CONCAT_2:%.+]] = IE.Concat([[SLICE_7]], [[CST_3]], [[SLICE_8]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<2xsi64> -> tensor<4xsi64>
    // CHECK:   [[DYN_RESHAPE_2:%.+]] = IE.DynamicReshape([[ARG_0]], [[CONCAT_2]]) {output_bounds = [1, 1, 35, 512], output_shape = [1, 1, -9223372036854775808, 512]} : tensor<1x?x512xf32, {bounds = [1, 35, 512], order = #CHW}>, tensor<4xsi64> -> tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}>
    // CHECK:   [[EXPAND_1:%.+]] = IE.DynamicExpand([[DYN_RESHAPE_2]]) : tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}> -> tensor<1x1x35x512xf32>
    // CHECK:   [[MATMUL_1:%.+]] = IE.MatMul([[EXPAND_1]], [[CST]]) {transpose_b} : tensor<1x1x35x512xf32>, tensor<1x1x512x512xf32> -> tensor<1x1x35x512xf32>
    // CHECK:   [[SHAPE_OF_3:%.+]] = IE.ShapeOf([[DYN_RESHAPE_2]]) {dstElemType = si64} : tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}> -> tensor<4xsi64>
    // CHECK:   [[SLICE_9:%.+]] = IE.Slice [[SHAPE_OF_3]] [2] [1] : tensor<4xsi64> to tensor<1xsi64>
    // CHECK:   [[CONCAT_3:%.+]] = IE.Concat([[CST_1]], [[SLICE_9]], [[CST_0]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<2xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>
    // CHECK:   [[DYN_RESHAPE_3:%.+]] = IE.DynamicReshape([[MATMUL_1]], [[CONCAT_3]]) {output_bounds = [1, 1, 35, 512], output_shape = [1, 1, -9223372036854775808, 512]} : tensor<1x1x35x512xf32>, tensor<4xsi64> -> tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}>
    // CHECK:   [[OUT_HV_1:%.+]], [[OUT_HS_1:%.+]], [[OUT_CS_1:%.+]] = IE.LSTMSequence([[DYN_RESHAPE_3]], [[SLICE_1]], [[SLICE_3]], [[CST_4]]) {direction = #IE.rnn_seq_direction<REVERSE>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x1x?x512xf32, {bounds = [1, 1, 35, 512], order = #NCHW}>, tensor<1x1x128xf32>, tensor<1x1x128xf32>, tensor<1x512x128xf32> -> tensor<1x1x?x128xf32, {bounds = [1, 1, 35, 128], order = #NCHW}>, tensor<1x1x128xf32>, tensor<1x1x128xf32>
    // CHECK:   [[CONCAT_4:%.+]] = IE.Concat([[OUT_HV_0]], [[OUT_HV_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x?x128xf32, {bounds = [1, 1, 35, 128], order = #NCHW}>, tensor<1x1x?x128xf32, {bounds = [1, 1, 35, 128], order = #NCHW}> -> tensor<1x2x?x128xf32, {bounds = [1, 2, 35, 128], order = #NCHW}>
    // CHECK:   [[CONCAT_5:%.+]] = IE.Concat([[OUT_HS_0]], [[OUT_HS_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x128xf32>, tensor<1x1x128xf32> -> tensor<1x2x128xf32>
    // CHECK:   [[CONCAT_6:%.+]] = IE.Concat([[OUT_CS_0]], [[OUT_CS_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x128xf32>, tensor<1x1x128xf32> -> tensor<1x2x128xf32>
    // CHECK:   return [[CONCAT_4]], [[CONCAT_5]], [[CONCAT_6]] : tensor<1x2x?x128xf32, {bounds = [1, 2, 35, 128], order = #NCHW}>, tensor<1x2x128xf32>, tensor<1x2x128xf32>
}
