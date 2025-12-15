//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-gru-sequence %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL:   func.func @DecomposeGruResetTrue(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<2x2x256xf16>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: tensor<2x1x256xf16>) -> (tensor<2x1x2x256xf16>, tensor<2x1x256xf16>) {
func.func @DecomposeGruResetTrue(%arg0: tensor<2x2x256xf16>, %arg1: tensor<2x1x256xf16>) -> (tensor<2x1x2x256xf16>, tensor<2x1x256xf16>) {
  %cst = const.Declare tensor<1x768x256xf16> = dense<1.0> : tensor<1x768x256xf16>
  %cst_0 = const.Declare tensor<1x768x256xf16> = dense<1.0> : tensor<1x768x256xf16>
  %cst_1 = const.Declare tensor<1x1024xf16> = dense<1.0> : tensor<1x1024xf16>
  %middle_hidden_state, %output_hidden_state = IE.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 256 : i64, seq_length = 2 : i64, should_linear_before_reset} : tensor<2x2x256xf16>, tensor<2x1x256xf16>, tensor<1x768x256xf16>, tensor<1x768x256xf16>, tensor<1x1024xf16> -> tensor<2x1x2x256xf16>, tensor<2x1x256xf16>
  return %middle_hidden_state, %output_hidden_state : tensor<2x1x2x256xf16>, tensor<2x1x256xf16>
// CHECK:           %[[VAL_2:.*]] = const.Declare tensor<768x256xf16> = dense<1.000000e+00> : tensor<1x768x256xf16>, [#const.Reshape<[768, 256]>]
// CHECK:           %[[VAL_3:.*]] = const.Declare tensor<1024xf16> = dense<1.000000e+00> : tensor<1x1024xf16>, [#const.Reshape<[1024]>]
// CHECK:           %[[VAL_4:.*]] = const.Declare tensor<1x1x768x256xf16> = dense<1.000000e+00> : tensor<1x768x256xf16>,  [#const.Reshape<[1, 1, 768, 256]>]
// CHECK:           %[[VAL_5:.*]] = IE.Unsqueeze(%[[VAL_0]]) {axes_value = [1]} : tensor<2x2x256xf16> -> tensor<2x1x2x256xf16>
// CHECK:           %[[VAL_6:.*]] = IE.MatMul(%[[VAL_5]], %[[VAL_4]]) {transpose_b} : tensor<2x1x2x256xf16>, tensor<1x1x768x256xf16> -> tensor<2x1x2x768xf16>
// CHECK:           %[[VAL_7:.*]] = IE.Squeeze(%[[VAL_6]]) {axes_value = [1]} : tensor<2x1x2x768xf16> -> tensor<2x2x768xf16>
// CHECK:           %[[VAL_8:.*]] = IE.Squeeze(%[[VAL_1]]) {axes_value = [1]} : tensor<2x1x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_9:.*]] = IE.Slice %[[VAL_7]] [0, 0, 0] [2, 1, 768] : tensor<2x2x768xf16> to tensor<2x1x768xf16>
// CHECK:           %[[VAL_10:.*]] = IE.Squeeze(%[[VAL_9]]) {axes_value = [1]} : tensor<2x1x768xf16> -> tensor<2x768xf16>
// CHECK:           %[[VAL_11:.*]] = IE.MatMul(%[[VAL_8]], %[[VAL_2]]) {transpose_b} : tensor<2x256xf16>, tensor<768x256xf16> -> tensor<2x768xf16>
// CHECK:           %[[VAL_12:.*]] = IE.GRUGates(%[[VAL_10]], %[[VAL_8]], %[[VAL_11]], %[[VAL_3]]) : tensor<2x768xf16>, tensor<2x256xf16>, tensor<2x768xf16>, tensor<1024xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_13:.*]] = IE.Unsqueeze(%[[VAL_12]]) {axes_value = [1]} : tensor<2x256xf16> -> tensor<2x1x256xf16>
// CHECK:           %[[VAL_14:.*]] = IE.Slice %[[VAL_7]] [0, 1, 0] [2, 1, 768] : tensor<2x2x768xf16> to tensor<2x1x768xf16>
// CHECK:           %[[VAL_15:.*]] = IE.Squeeze(%[[VAL_14]]) {axes_value = [1]} : tensor<2x1x768xf16> -> tensor<2x768xf16>
// CHECK:           %[[VAL_16:.*]] = IE.MatMul(%[[VAL_12]], %[[VAL_2]]) {transpose_b} : tensor<2x256xf16>, tensor<768x256xf16> -> tensor<2x768xf16>
// CHECK:           %[[VAL_17:.*]] = IE.GRUGates(%[[VAL_15]], %[[VAL_12]], %[[VAL_16]], %[[VAL_3]]) : tensor<2x768xf16>, tensor<2x256xf16>, tensor<2x768xf16>, tensor<1024xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_18:.*]] = IE.Unsqueeze(%[[VAL_17]]) {axes_value = [1]} : tensor<2x256xf16> -> tensor<2x1x256xf16>
// CHECK:           %[[VAL_19:.*]] = IE.Concat(%[[VAL_13]], %[[VAL_18]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<2x1x256xf16>, tensor<2x1x256xf16> -> tensor<2x2x256xf16>
// CHECK:           %[[VAL_20:.*]] = IE.Unsqueeze(%[[VAL_19]]) {axes_value = [1]} : tensor<2x2x256xf16> -> tensor<2x1x2x256xf16>
// CHECK:           %[[VAL_21:.*]] = IE.Unsqueeze(%[[VAL_17]]) {axes_value = [1]} : tensor<2x256xf16> -> tensor<2x1x256xf16>
// CHECK:           return %[[VAL_20]], %[[VAL_21]] : tensor<2x1x2x256xf16>, tensor<2x1x256xf16>
}

// -----

// CHECK-LABEL:   func.func @DecomposeGruResetFalse(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<2x2x256xf16>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: tensor<2x1x256xf16>) -> (tensor<2x1x2x256xf16>, tensor<2x1x256xf16>) {
func.func @DecomposeGruResetFalse(%arg0: tensor<2x2x256xf16>, %arg1: tensor<2x1x256xf16>) -> (tensor<2x1x2x256xf16>, tensor<2x1x256xf16>) {
  %cst = const.Declare tensor<1x768x256xf16> = dense<1.0>  : tensor<1x768x256xf16>
  %cst_0 = const.Declare tensor<1x768x256xf16> = dense<1.0>  : tensor<1x768x256xf16>
  %cst_1 = const.Declare tensor<1x768xf16> = dense<1.0>  : tensor<1x768xf16>
  %middle_hidden_state, %output_hidden_state = IE.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 256 : i64, seq_length = 2 : i64} : tensor<2x2x256xf16>, tensor<2x1x256xf16>, tensor<1x768x256xf16>, tensor<1x768x256xf16>, tensor<1x768xf16> -> tensor<2x1x2x256xf16>, tensor<2x1x256xf16>
  return %middle_hidden_state, %output_hidden_state : tensor<2x1x2x256xf16>, tensor<2x1x256xf16>
// CHECK:           %[[VAL_2:.*]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
// CHECK:           %[[VAL_3:.*]] = const.Declare tensor<256x256xf16> = dense<1.000000e+00> : tensor<1x768x256xf16>, [#const.SubView<[0, 512, 0], [1, 256, 256]>, #const.Reshape<[256, 256]>]
// CHECK:           %[[VAL_4:.*]] = const.Declare tensor<768x256xf16> = dense<1.000000e+00> : tensor<1x768x256xf16>,  [#const.Reshape<[768, 256]>]
// CHECK:           %[[VAL_5:.*]] = const.Declare tensor<768xf16> = dense<1.000000e+00> : tensor<1x768xf16>, [#const.Reshape<[768]>]
// CHECK:           %[[VAL_6:.*]] = const.Declare tensor<1x1x768x256xf16> = dense<1.000000e+00> : tensor<1x768x256xf16>, [#const.Reshape<[1, 1, 768, 256]>]
// CHECK:           %[[VAL_7:.*]] = IE.Unsqueeze(%[[VAL_0]]) {axes_value = [1]} : tensor<2x2x256xf16> -> tensor<2x1x2x256xf16>
// CHECK:           %[[VAL_8:.*]] = IE.MatMul(%[[VAL_7]], %[[VAL_6]]) {transpose_b} : tensor<2x1x2x256xf16>, tensor<1x1x768x256xf16> -> tensor<2x1x2x768xf16>
// CHECK:           %[[VAL_9:.*]] = IE.Add(%[[VAL_8]], %[[VAL_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x1x2x768xf16>, tensor<768xf16> -> tensor<2x1x2x768xf16>
// CHECK:           %[[VAL_10:.*]] = IE.Squeeze(%[[VAL_9]]) {axes_value = [1]} : tensor<2x1x2x768xf16> -> tensor<2x2x768xf16>
// CHECK:           %[[VAL_11:.*]] = IE.Squeeze(%[[VAL_1]]) {axes_value = [1]} : tensor<2x1x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_12:.*]] = IE.Slice %[[VAL_10]] [0, 0, 0] [2, 1, 768] : tensor<2x2x768xf16> to tensor<2x1x768xf16>
// CHECK:           %[[VAL_13:.*]] = IE.Squeeze(%[[VAL_12]]) {axes_value = [1]} : tensor<2x1x768xf16> -> tensor<2x768xf16>
// CHECK:           %[[VAL_14:.*]] = IE.MatMul(%[[VAL_11]], %[[VAL_4]]) {transpose_b} : tensor<2x256xf16>, tensor<768x256xf16> -> tensor<2x768xf16>
// CHECK:           %[[VAL_15:.*]] = IE.Slice %[[VAL_13]] [0, 0] [2, 512] : tensor<2x768xf16> to tensor<2x512xf16>
// CHECK:           %[[VAL_16:.*]] = IE.Slice %[[VAL_14]] [0, 0] [2, 512] : tensor<2x768xf16> to tensor<2x512xf16>
// CHECK:           %[[VAL_17:.*]] = IE.Add(%[[VAL_15]], %[[VAL_16]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x512xf16>, tensor<2x512xf16> -> tensor<2x512xf16>
// CHECK:           %[[VAL_18:.*]] = IE.Sigmoid(%[[VAL_17]]) : tensor<2x512xf16> -> tensor<2x512xf16>
// CHECK:           %[[VAL_19:.*]]:2 = IE.Split(%[[VAL_18]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<2x512xf16> -> tensor<2x256xf16>, tensor<2x256xf16>
// CHECK:           %[[VAL_20:.*]] = IE.Slice %[[VAL_13]] [0, 512] [2, 256] : tensor<2x768xf16> to tensor<2x256xf16>
// CHECK:           %[[VAL_21:.*]] = IE.Multiply(%[[VAL_19]]#1, %[[VAL_11]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_22:.*]] = IE.MatMul(%[[VAL_21]], %[[VAL_3]]) {transpose_b} : tensor<2x256xf16>, tensor<256x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_23:.*]] = IE.Add(%[[VAL_20]], %[[VAL_22]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_24:.*]] = IE.Tanh(%[[VAL_23]]) : tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_25:.*]] = IE.Subtract(%[[VAL_2]], %[[VAL_19]]#0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_26:.*]] = IE.Multiply(%[[VAL_19]]#0, %[[VAL_11]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_27:.*]] = IE.Multiply(%[[VAL_25]], %[[VAL_24]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_28:.*]] = IE.Add(%[[VAL_26]], %[[VAL_27]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_29:.*]] = IE.Unsqueeze(%[[VAL_28]]) {axes_value = [1]} : tensor<2x256xf16> -> tensor<2x1x256xf16>
// CHECK:           %[[VAL_30:.*]] = IE.Slice %[[VAL_10]] [0, 1, 0] [2, 1, 768] : tensor<2x2x768xf16> to tensor<2x1x768xf16>
// CHECK:           %[[VAL_31:.*]] = IE.Squeeze(%[[VAL_30]]) {axes_value = [1]} : tensor<2x1x768xf16> -> tensor<2x768xf16>
// CHECK:           %[[VAL_32:.*]] = IE.MatMul(%[[VAL_28]], %[[VAL_4]]) {transpose_b} : tensor<2x256xf16>, tensor<768x256xf16> -> tensor<2x768xf16>
// CHECK:           %[[VAL_33:.*]] = IE.Slice %[[VAL_31]] [0, 0] [2, 512] : tensor<2x768xf16> to tensor<2x512xf16>
// CHECK:           %[[VAL_34:.*]] = IE.Slice %[[VAL_32]] [0, 0] [2, 512] : tensor<2x768xf16> to tensor<2x512xf16>
// CHECK:           %[[VAL_35:.*]] = IE.Add(%[[VAL_33]], %[[VAL_34]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x512xf16>, tensor<2x512xf16> -> tensor<2x512xf16>
// CHECK:           %[[VAL_36:.*]] = IE.Sigmoid(%[[VAL_35]]) : tensor<2x512xf16> -> tensor<2x512xf16>
// CHECK:           %[[VAL_37:.*]]:2 = IE.Split(%[[VAL_36]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<2x512xf16> -> tensor<2x256xf16>, tensor<2x256xf16>
// CHECK:           %[[VAL_38:.*]] = IE.Slice %[[VAL_31]] [0, 512] [2, 256] : tensor<2x768xf16> to tensor<2x256xf16>
// CHECK:           %[[VAL_39:.*]] = IE.Multiply(%[[VAL_37]]#1, %[[VAL_28]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_40:.*]] = IE.MatMul(%[[VAL_39]], %[[VAL_3]]) {transpose_b} : tensor<2x256xf16>, tensor<256x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_41:.*]] = IE.Add(%[[VAL_38]], %[[VAL_40]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_42:.*]] = IE.Tanh(%[[VAL_41]]) : tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_43:.*]] = IE.Subtract(%[[VAL_2]], %[[VAL_37]]#0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_44:.*]] = IE.Multiply(%[[VAL_37]]#0, %[[VAL_28]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_45:.*]] = IE.Multiply(%[[VAL_43]], %[[VAL_42]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_46:.*]] = IE.Add(%[[VAL_44]], %[[VAL_45]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x256xf16>
// CHECK:           %[[VAL_47:.*]] = IE.Unsqueeze(%[[VAL_46]]) {axes_value = [1]} : tensor<2x256xf16> -> tensor<2x1x256xf16>
// CHECK:           %[[VAL_48:.*]] = IE.Concat(%[[VAL_29]], %[[VAL_47]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<2x1x256xf16>, tensor<2x1x256xf16> -> tensor<2x2x256xf16>
// CHECK:           %[[VAL_49:.*]] = IE.Unsqueeze(%[[VAL_48]]) {axes_value = [1]} : tensor<2x2x256xf16> -> tensor<2x1x2x256xf16>
// CHECK:           %[[VAL_50:.*]] = IE.Unsqueeze(%[[VAL_46]]) {axes_value = [1]} : tensor<2x256xf16> -> tensor<2x1x256xf16>
// CHECK:           return %[[VAL_49]], %[[VAL_50]] : tensor<2x1x2x256xf16>, tensor<2x1x256xf16>

}

// -----

// CHECK-LABEL:   func.func @NotDecomposeGru(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<2x5x10xf16>,
// CHECK-SAME:                               %[[VAL_1:.*]]: tensor<2x1x4xf16>) -> (tensor<2x1x5x4xf16>, tensor<2x1x4xf16>) {
func.func @NotDecomposeGru(%arg0: tensor<2x5x10xf16>, %arg1: tensor<2x1x4xf16>) -> (tensor<2x1x5x4xf16>, tensor<2x1x4xf16>) {
  %cst = const.Declare tensor<1x12x10xf16> = dense<1.0> : tensor<1x12x10xf16>
  %cst_0 = const.Declare tensor<1x12x4xf16> = dense<1.0> : tensor<1x12x4xf16>
  %cst_1 = const.Declare tensor<1x12xf16> = dense<1.0> : tensor<1x12xf16>
  %middle_hidden_state, %output_hidden_state = IE.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 4 : i64, seq_length = 5 : i64} : tensor<2x5x10xf16>, tensor<2x1x4xf16>, tensor<1x12x10xf16>, tensor<1x12x4xf16>, tensor<1x12xf16> -> tensor<2x1x5x4xf16>, tensor<2x1x4xf16>
  return %middle_hidden_state, %output_hidden_state : tensor<2x1x5x4xf16>, tensor<2x1x4xf16>
// CHECK:           %[[VAL_2:.*]] = const.Declare tensor<1x12x10xf16> = dense<1.000000e+00> : tensor<1x12x10xf16>
// CHECK:           %[[VAL_3:.*]] = const.Declare tensor<1x12x4xf16> = dense<1.000000e+00> : tensor<1x12x4xf16>
// CHECK:           %[[VAL_4:.*]] = const.Declare tensor<1x12xf16> = dense<1.000000e+00> : tensor<1x12xf16>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = IE.GRUSequence(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]]) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 4 : i64, seq_length = 5 : i64} : tensor<2x5x10xf16>, tensor<2x1x4xf16>, tensor<1x12x10xf16>, tensor<1x12x4xf16>, tensor<1x12xf16> -> tensor<2x1x5x4xf16>, tensor<2x1x4xf16>
// CHECK:           return %[[VAL_5]], %[[VAL_6]] : tensor<2x1x5x4xf16>, tensor<2x1x4xf16>
}
