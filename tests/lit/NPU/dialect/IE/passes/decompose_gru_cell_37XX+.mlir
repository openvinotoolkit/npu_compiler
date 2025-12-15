//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-gru-cell %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: func.func @DecomposeGRUCellLinearBeforeResetTrue(
// CHECK-SAME:      [[INPUT_DATA:%.+]]: tensor<1x768xf16>,
// CHECK-SAME:      [[INITIAL_HIDDEN_STATE:%.+]]: tensor<1x768xf16>) -> tensor<1x768xf16> {
func.func @DecomposeGRUCellLinearBeforeResetTrue(%input_data: tensor<1x768xf16>, %initial_hidden_state: tensor<1x768xf16>) -> tensor<1x768xf16> {
  %weights = const.Declare tensor<2304x768xf16> = dense<2.0> : tensor<2304x768xf16>
  %recurrence_weights = const.Declare tensor<2304x768xf16> = dense<2.0> : tensor<2304x768xf16>
  %biases = const.Declare tensor<3072xf16> = dense<3.0> : tensor<3072xf16>
  %0 = IE.GRUCell(%input_data, %initial_hidden_state, %weights, %recurrence_weights, %biases) {clip = 0.000000e+00 : f64, hidden_size = 768 : i64, should_linear_before_reset} : tensor<1x768xf16>, tensor<1x768xf16>, tensor<2304x768xf16>, tensor<2304x768xf16>, tensor<3072xf16> -> tensor<1x768xf16>
  return %0 : tensor<1x768xf16>

// CHECK-DAG:   [[WEIGHTS:%.+]]             = const.Declare tensor<2304x768xf16> = dense<2.000000e+00> : tensor<2304x768xf16>
// CHECK-DAG:   [[RECURRENCE_WEIGHTS:%.+]]  = const.Declare tensor<2304x768xf16> = dense<2.000000e+00> : tensor<2304x768xf16>
// CHECK-DAG:   [[BIASES:%.+]]              = const.Declare tensor<3072xf16> = dense<3.000000e+00> : tensor<3072xf16>

// CHECK:   [[MATMUL_1:%.+]]    = IE.MatMul([[INPUT_DATA]], [[WEIGHTS]]) {transpose_b} : tensor<1x768xf16>, tensor<2304x768xf16> -> tensor<1x2304xf16>
// CHECK:   [[MATMUL_2:%.+]]    = IE.MatMul([[INITIAL_HIDDEN_STATE]], [[RECURRENCE_WEIGHTS]]) {transpose_b} : tensor<1x768xf16>, tensor<2304x768xf16> -> tensor<1x2304xf16>
// CHECK:   [[SLICE_CONST_1:%.+]]     = IE.Slice [[BIASES]] [0] [1536] : tensor<3072xf16> to tensor<1536xf16>
// CHECK:   [[SLICE_CONST_2:%.+]]     = IE.Slice [[BIASES]] [2304] [768] : tensor<3072xf16> to tensor<768xf16>
// CHECK:   [[CONCAT:%.+]]      = IE.Concat([[SLICE_CONST_1]], [[SLICE_CONST_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1536xf16>, tensor<768xf16> -> tensor<2304xf16>
// CHECK:   [[ADD_1:%.+]]       = IE.Add([[MATMUL_2]], [[CONCAT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2304xf16>, tensor<2304xf16> -> tensor<1x2304xf16>
// CHECK:   [[SLICE_1:%.+]]     = IE.Slice [[MATMUL_1]] [0, 0] [1, 1536] : tensor<1x2304xf16> to tensor<1x1536xf16>
// CHECK:   [[SLICE_2:%.+]]     = IE.Slice [[ADD_1]] [0, 0] [1, 1536] : tensor<1x2304xf16> to tensor<1x1536xf16>
// CHECK:   [[ADD_2:%.+]]       = IE.Add([[SLICE_1]], [[SLICE_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1536xf16>, tensor<1x1536xf16> -> tensor<1x1536xf16>
// CHECK:   [[SIGMOID:%.+]]     = IE.Sigmoid([[ADD_2]]) : tensor<1x1536xf16> -> tensor<1x1536xf16>
// CHECK:   [[SPLIT:%.+]]:2     = IE.Split([[SIGMOID]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x1536xf16> -> tensor<1x768xf16>, tensor<1x768xf16>
// CHECK:   [[SLICE_3:%.+]]     = IE.Slice [[MATMUL_1]] [0, 1536] [1, 768] : tensor<1x2304xf16> to tensor<1x768xf16>
// CHECK:   [[SLICE_4:%.+]]     = IE.Slice [[ADD_1]] [0, 1536] [1, 768] : tensor<1x2304xf16> to tensor<1x768xf16>
// CHECK:   [[MULTIPLY_1:%.+]]  = IE.Multiply([[SPLIT]]#1, [[SLICE_4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:   [[SLICE_5:%.+]]     = IE.Slice [[BIASES]] [1536] [768] : tensor<3072xf16> to tensor<768xf16>
// CHECK:   [[ADD_3:%.+]]       = IE.Add([[MULTIPLY_1]], [[SLICE_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x768xf16>, tensor<768xf16> -> tensor<1x768xf16>
// CHECK:   [[ADD_4:%.+]]       = IE.Add([[SLICE_3]], [[ADD_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:   [[TANH:%.+]]        = IE.Tanh([[ADD_4]]) : tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK-DAG:[[CST:%.+]]        = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
// CHECK:   [[SUBTRACT:%.+]]    = IE.Subtract([[CST]], [[SPLIT]]#0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:   [[MULTIPLY_2:%.+]]  = IE.Multiply([[SPLIT]]#0, [[INITIAL_HIDDEN_STATE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:   [[MULTIPLY_3:%.+]]  = IE.Multiply([[SUBTRACT]], [[TANH]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:   [[ADD_5:%.+]]       = IE.Add([[MULTIPLY_2]], [[MULTIPLY_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:   return [[ADD_5]] : tensor<1x768xf16>
// CHECK: }
}

// -----

// CHECK-LABEL: func.func @DecomposeGRUCellLinearBeforeResetFalse(
// CHECK-SAME:      [[INPUT_DATA:%.+]]: tensor<1x768xf16>,
// CHECK-SAME:      [[INITIAL_HIDDEN_STATE:%.+]]: tensor<1x768xf16>) -> tensor<1x768xf16> {
func.func @DecomposeGRUCellLinearBeforeResetFalse(%input_data: tensor<1x768xf16>, %initial_hidden_state: tensor<1x768xf16>) -> tensor<1x768xf16> {
  %weights = const.Declare tensor<2304x768xf16> = dense<2.0> : tensor<2304x768xf16>
  %recurrence_weights = const.Declare tensor<2304x768xf16> = dense<2.0> : tensor<2304x768xf16>
  %biases = const.Declare tensor<2304xf16> = dense<3.0> : tensor<2304xf16>
  %0 = IE.GRUCell(%input_data, %initial_hidden_state, %weights, %recurrence_weights, %biases) {clip = 0.000000e+00 : f64, hidden_size = 768 : i64} : tensor<1x768xf16>, tensor<1x768xf16>, tensor<2304x768xf16>, tensor<2304x768xf16>, tensor<2304xf16> -> tensor<1x768xf16>
  return %0 : tensor<1x768xf16>

// CHECK-DAG:   [[WEIGHTS:%.+]]             = const.Declare tensor<2304x768xf16> = dense<2.000000e+00> : tensor<2304x768xf16>
// CHECK-DAG:   [[RECURRENCE_WEIGHTS:%.+]]  = const.Declare tensor<2304x768xf16> = dense<2.000000e+00> : tensor<2304x768xf16>
// CHECK-DAG:   [[BIASES:%.+]]              = const.Declare tensor<2304xf16> = dense<3.000000e+00> : tensor<2304xf16>

// CHECK:       [[MATMUL_1:%.+]]    = IE.MatMul([[INPUT_DATA]], [[WEIGHTS]]) {transpose_b} : tensor<1x768xf16>, tensor<2304x768xf16> -> tensor<1x2304xf16>
// CHECK:       [[MATMUL_2:%.+]]    = IE.MatMul([[INITIAL_HIDDEN_STATE]], [[RECURRENCE_WEIGHTS]]) {transpose_b} : tensor<1x768xf16>, tensor<2304x768xf16> -> tensor<1x2304xf16>
// CHECK:       [[SLICE_1:%.+]]     = IE.Slice [[BIASES]] [0] [1536] : tensor<2304xf16> to tensor<1536xf16>
// CHECK-DAG:   [[CONST_0:%.+]]     = const.Declare tensor<768xf16> = dense<0.000000e+00> : tensor<768xf32>, [#const.CastElemType<f16>]
// CHECK:       [[CONCAT_0:%.+]]    = IE.Concat([[SLICE_1]], [[CONST_0]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1536xf16>, tensor<768xf16> -> tensor<2304xf16>
// CHECK:       [[ADD_0:%.+]]       = IE.Add([[MATMUL_2]], [[CONCAT_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2304xf16>, tensor<2304xf16> -> tensor<1x2304xf16>
// CHECK:       [[SLICE_2:%.+]]     = IE.Slice [[MATMUL_1]] [0, 0] [1, 1536] : tensor<1x2304xf16> to tensor<1x1536xf16>
// CHECK:       [[SLICE_3:%.+]]     = IE.Slice [[ADD_0]] [0, 0] [1, 1536] : tensor<1x2304xf16> to tensor<1x1536xf16>
// CHECK:       [[ADD_1:%.+]]       = IE.Add([[SLICE_2]], [[SLICE_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1536xf16>, tensor<1x1536xf16> -> tensor<1x1536xf16>
// CHECK:       [[SIGMOID:%.+]]     = IE.Sigmoid([[ADD_1]]) : tensor<1x1536xf16> -> tensor<1x1536xf16>
// CHECK:       [[SPLIT:%.+]]:2     = IE.Split([[SIGMOID]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x1536xf16> -> tensor<1x768xf16>, tensor<1x768xf16>
// CHECK:       [[SLICE_4:%.+]]     = IE.Slice [[MATMUL_1]] [0, 1536] [1, 768] : tensor<1x2304xf16> to tensor<1x768xf16>
// CHECK:       [[SLICE_5:%.+]]     = IE.Slice [[RECURRENCE_WEIGHTS]] [1536, 0] [768, 768] : tensor<2304x768xf16> to tensor<768x768xf16>
// CHECK:       [[MULTIPLY_1:%.+]]  = IE.Multiply([[SPLIT]]#1, [[INITIAL_HIDDEN_STATE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:       [[MATMUL_3:%.+]]    = IE.MatMul([[MULTIPLY_1]], [[SLICE_5]]) {transpose_b} : tensor<1x768xf16>, tensor<768x768xf16> -> tensor<1x768xf16>
// CHECK:       [[SLICE_6:%.+]]     = IE.Slice [[BIASES]] [1536] [768] : tensor<2304xf16> to tensor<768xf16>
// CHECK:       [[ADD_2:%.+]]       = IE.Add([[MATMUL_3]], [[SLICE_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x768xf16>, tensor<768xf16> -> tensor<1x768xf16>
// CHECK:       [[ADD_3:%.+]]       = IE.Add([[SLICE_4]], [[ADD_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:       [[TANH:%.+]]        = IE.Tanh([[ADD_3]]) : tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK-DAG:   [[CST_1:%.+]]       = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
// CHECK:       [[SUBTRACT:%.+]]    = IE.Subtract([[CST_1]], [[SPLIT]]#0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:       [[MULTIPLY_2:%.+]]  = IE.Multiply([[SPLIT]]#0, [[INITIAL_HIDDEN_STATE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:       [[MULTIPLY_3:%.+]]  = IE.Multiply([[SUBTRACT]], [[TANH]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:       [[ADD_4:%.+]]       = IE.Add([[MULTIPLY_2]], [[MULTIPLY_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:       return [[ADD_4]] : tensor<1x768xf16>
// CHECK: }
}

// -----

// CHECK-LABEL: func.func @DecomposeGRUCellMissingBiases(
// CHECK-SAME:      [[INPUT_DATA:%.+]]: tensor<1x768xf16>,
// CHECK-SAME:      [[INITIAL_HIDDEN_STATE:%.+]]: tensor<1x768xf16>) -> tensor<1x768xf16> {
func.func @DecomposeGRUCellMissingBiases(%input_data: tensor<1x768xf16>, %initial_hidden_state: tensor<1x768xf16>) -> tensor<1x768xf16> {
  %weights = const.Declare tensor<2304x768xf16> = dense<2.0> : tensor<2304x768xf16>
  %recurrence_weights = const.Declare tensor<2304x768xf16> = dense<2.0> : tensor<2304x768xf16>
  %0 = IE.GRUCell(%input_data, %initial_hidden_state, %weights, %recurrence_weights) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>, clip = 0.000000e+00 : f64, hidden_size = 768 : i64, should_linear_before_reset} : tensor<1x768xf16>, tensor<1x768xf16>, tensor<2304x768xf16>, tensor<2304x768xf16> -> tensor<1x768xf16>
  return %0 : tensor<1x768xf16>

// CHECK-DAG:   [[WEIGHTS:%.+]]             = const.Declare tensor<2304x768xf16> = dense<2.000000e+00> : tensor<2304x768xf16>
// CHECK-DAG:   [[RECURRENCE_WEIGHTS:%.+]]  = const.Declare tensor<2304x768xf16> = dense<2.000000e+00> : tensor<2304x768xf16>

// CHECK:   [[MATMUL_1:%.+]]    = IE.MatMul([[INPUT_DATA]], [[WEIGHTS]]) {transpose_b} : tensor<1x768xf16>, tensor<2304x768xf16> -> tensor<1x2304xf16>
// CHECK:   [[MATMUL_2:%.+]]    = IE.MatMul([[INITIAL_HIDDEN_STATE]], [[RECURRENCE_WEIGHTS]]) {transpose_b} : tensor<1x768xf16>, tensor<2304x768xf16> -> tensor<1x2304xf16>
// CHECK:   [[SLICE_1:%.+]]     = IE.Slice [[MATMUL_1]] [0, 0] [1, 1536] : tensor<1x2304xf16> to tensor<1x1536xf16>
// CHECK:   [[SLICE_2:%.+]]     = IE.Slice [[MATMUL_2]] [0, 0] [1, 1536] : tensor<1x2304xf16> to tensor<1x1536xf16>
// CHECK:   [[ADD_1:%.+]]       = IE.Add([[SLICE_1]], [[SLICE_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1536xf16>, tensor<1x1536xf16> -> tensor<1x1536xf16>
// CHECK:   [[SIGMOID:%.+]]     = IE.Sigmoid([[ADD_1]]) : tensor<1x1536xf16> -> tensor<1x1536xf16>
// CHECK:   [[SPLIT:%.+]]:2     = IE.Split([[SIGMOID]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x1536xf16> -> tensor<1x768xf16>, tensor<1x768xf16>
// CHECK:   [[SLICE_3:%.+]]     = IE.Slice [[MATMUL_1]] [0, 1536] [1, 768] : tensor<1x2304xf16> to tensor<1x768xf16>
// CHECK:   [[SLICE_4:%.+]]     = IE.Slice [[MATMUL_2]] [0, 1536] [1, 768] : tensor<1x2304xf16> to tensor<1x768xf16>
// CHECK:   [[MULTIPLY_1:%.+]]  = IE.Multiply([[SPLIT]]#1, [[SLICE_4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:   [[ADD_2:%.+]]       = IE.Add([[SLICE_3]], [[MULTIPLY_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:   [[TANH:%.+]]        = IE.Tanh([[ADD_2]]) : tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK-DAG:[[CST:%.+]]        = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
// CHECK:   [[SUBTRACT:%.+]]    = IE.Subtract([[CST]], [[SPLIT]]#0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:   [[MULTIPLY_2:%.+]] = IE.Multiply([[SPLIT]]#0, [[INITIAL_HIDDEN_STATE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:   [[MULTIPLY_4:%.+]] = IE.Multiply([[SUBTRACT]], [[TANH]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:   [[ADD_3:%.+]]      = IE.Add([[MULTIPLY_2]], [[MULTIPLY_4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x768xf16>, tensor<1x768xf16> -> tensor<1x768xf16>
// CHECK:   return [[ADD_3]] : tensor<1x768xf16>
// CHECK: }
}

// -----

// CHECK-LABEL: func.func @TransferGRUSequenceToGRUCell(
// CHECK-SAME:      [[INPUT_DATA:%.+]]: tensor<1x1x256xf32>,
// CHECK-SAME:      [[INITIAL_HIDDEN_STATE:%.+]]: tensor<1x1x256xf32>)
func.func @TransferGRUSequenceToGRUCell(%input_data: tensor<1x1x256xf32>, %initial_hidden_state: tensor<1x1x256xf32>) -> (tensor<1x1x1x256xf32>, tensor<1x1x256xf32>) {
  %weights = const.Declare tensor<1x768x256xf32> = dense<2.0> : tensor<1x768x256xf32>
  %recurrence_weights = const.Declare tensor<1x768x256xf32> = dense<2.0> : tensor<1x768x256xf32>
  %biases = const.Declare tensor<1x1024xf32> = dense<3.0> : tensor<1x1024xf32>

  %middle_hidden_state, %output_hidden_state = IE.GRUSequence(%input_data, %initial_hidden_state, %weights, %recurrence_weights, %biases) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 256 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<1x1x256xf32>, tensor<1x1x256xf32>, tensor<1x768x256xf32>, tensor<1x768x256xf32>, tensor<1x1024xf32> -> tensor<1x1x1x256xf32>, tensor<1x1x256xf32>

  return %middle_hidden_state, %output_hidden_state : tensor<1x1x1x256xf32>, tensor<1x1x256xf32>

// CHECK-DAG:   [[WEIGHTS:%.+]]             = const.Declare tensor<1x768x256xf32> = dense<2.000000e+00> : tensor<1x768x256xf32>
// CHECK-DAG:   [[RECURRENCE_WEIGHTS:%.+]]  = const.Declare tensor<1x768x256xf32> = dense<2.000000e+00> : tensor<1x768x256xf32>
// CHECK-DAG:   [[BIASES:%.+]]              = const.Declare tensor<1x1024xf32> = dense<3.000000e+00> : tensor<1x1024xf32>

// CHECK:   [[RESHAPE_INPUT_DATA:%.+]] = IE.Reshape([[INPUT_DATA]]) {shape_value = [1, 256]} : tensor<1x1x256xf32> -> tensor<1x256xf32>
// CHECK:   [[RESHAPE_INITIAL_HIDDEN_STATE:%.+]] = IE.Reshape([[INITIAL_HIDDEN_STATE]]) {shape_value = [1, 256]} : tensor<1x1x256xf32> -> tensor<1x256xf32>
// CHECK:   [[RESHAPE_WEIGHTS:%.+]] = IE.Reshape([[WEIGHTS]]) {shape_value = [768, 256]} : tensor<1x768x256xf32> -> tensor<768x256xf32>
// CHECK:   [[RESHAPE_RECURRENCE_WEIGHTS:%.+]]    = IE.Reshape([[RECURRENCE_WEIGHTS]]) {shape_value = [768, 256]} : tensor<1x768x256xf32> -> tensor<768x256xf32>
// CHECK:   [[RESHAPE_BIASES:%.+]] = IE.Reshape([[BIASES]]) {shape_value = [1024]} : tensor<1x1024xf32> -> tensor<1024xf32>

// CHECK:   [[MATMUL_1:%.+]]     = IE.MatMul([[RESHAPE_INPUT_DATA]], [[RESHAPE_WEIGHTS]]) {transpose_b} : tensor<1x256xf32>, tensor<768x256xf32> -> tensor<1x768xf32>
// CHECK:   [[MATMUL_2:%.+]]     = IE.MatMul([[RESHAPE_INITIAL_HIDDEN_STATE]], [[RESHAPE_RECURRENCE_WEIGHTS]]) {transpose_b} : tensor<1x256xf32>, tensor<768x256xf32> -> tensor<1x768xf32>
// CHECK:   [[SLICE_1:%.+]]      = IE.Slice [[RESHAPE_BIASES]] [0] [512] : tensor<1024xf32> to tensor<512xf32>
// CHECK:   [[SLICE_2:%.+]]      = IE.Slice [[RESHAPE_BIASES]] [768] [256] : tensor<1024xf32> to tensor<256xf32>
// CHECK:   [[CONCAT:%.+]]       = IE.Concat([[SLICE_1]], [[SLICE_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<512xf32>, tensor<256xf32> -> tensor<768xf32>
// CHECK:   [[ADD_1:%.+]]        = IE.Add([[MATMUL_2]], [[CONCAT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x768xf32>, tensor<768xf32> -> tensor<1x768xf32>
// CHECK:   [[SLICE_3:%.+]]      = IE.Slice [[MATMUL_1]] [0, 0] [1, 512] : tensor<1x768xf32> to tensor<1x512xf32>
// CHECK:   [[SLICE_4:%.+]]      = IE.Slice [[ADD_1]] [0, 0] [1, 512] : tensor<1x768xf32> to tensor<1x512xf32>
// CHECK:   [[ADD_2:%.+]]        = IE.Add([[SLICE_3]], [[SLICE_4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512xf32>, tensor<1x512xf32> -> tensor<1x512xf32>
// CHECK:   [[SIGMOID:%.+]]      = IE.Sigmoid([[ADD_2]]) : tensor<1x512xf32> -> tensor<1x512xf32>
// CHECK:   [[SPLIT:%.+]]:2      = IE.Split([[SIGMOID]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x512xf32> -> tensor<1x256xf32>, tensor<1x256xf32>
// CHECK:   [[SLICE_5:%.+]]      = IE.Slice [[MATMUL_1]] [0, 512] [1, 256] : tensor<1x768xf32> to tensor<1x256xf32>
// CHECK:   [[SLICE_6:%.+]]      = IE.Slice [[ADD_1]] [0, 512] [1, 256] : tensor<1x768xf32> to tensor<1x256xf32>
// CHECK:   [[MULTIPLY_1:%.+]]   = IE.Multiply([[SPLIT]]#1, [[SLICE_6]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x256xf32>, tensor<1x256xf32> -> tensor<1x256xf32>
// CHECK:   [[SLICE_7:%.+]]      = IE.Slice [[RESHAPE_BIASES]] [512] [256] : tensor<1024xf32> to tensor<256xf32>
// CHECK:   [[ADD_3:%.+]]        = IE.Add([[MULTIPLY_1]], [[SLICE_7]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256xf32>, tensor<256xf32> -> tensor<1x256xf32>
// CHECK:   [[ADD_4:%.+]]        = IE.Add([[SLICE_5]], [[ADD_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x256xf32>, tensor<1x256xf32> -> tensor<1x256xf32>
// CHECK:   [[TANH:%.+]]         = IE.Tanh([[ADD_4]]) : tensor<1x256xf32> -> tensor<1x256xf32>
// CHECK-DAG:[[CST:%.+]]         = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
// CHECK:   [[SUBTRACT:%.+]]     = IE.Subtract([[CST]], [[SPLIT]]#0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x256xf32> -> tensor<1x256xf32>
// CHECK:   [[MULTIPLY_2:%.+]]   = IE.Multiply([[SPLIT]]#0, [[RESHAPE_INITIAL_HIDDEN_STATE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x256xf32>, tensor<1x256xf32> -> tensor<1x256xf32>
// CHECK:   [[MULTIPLY_3:%.+]]   = IE.Multiply([[SUBTRACT]], [[TANH]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x256xf32>, tensor<1x256xf32> -> tensor<1x256xf32>
// CHECK:   [[ADD_5:%.+]]        = IE.Add([[MULTIPLY_2]], [[MULTIPLY_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x256xf32>, tensor<1x256xf32> -> tensor<1x256xf32>
// CHECK:   [[RESHAPE_5:%.+]]    = IE.Reshape([[ADD_5]]) {shape_value = [1, 1, 1, 256]} : tensor<1x256xf32> -> tensor<1x1x1x256xf32>
// CHECK:   [[RESHAPE_6:%.+]]    = IE.Reshape([[ADD_5]]) {shape_value = [1, 1, 256]} : tensor<1x256xf32> -> tensor<1x1x256xf32>

// CHECK:   return [[RESHAPE_5]], [[RESHAPE_6]] : tensor<1x1x1x256xf32>, tensor<1x1x256xf32>
// CHECK: }
}
