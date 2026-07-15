//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-sub-gru-sequence-to-conv %s | FileCheck %s
// REQUIRES: platform-NPU3720

// CHECK-LABEL: @ConvertSubGRUSequenceToConv
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x80x256xf16>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x1x128xf16>
func.func @ConvertSubGRUSequenceToConv(%arg0: tensor<1x80x256xf16>, %arg1: tensor<1x1x128xf16>) -> (tensor<1x1x80x128xf16>, tensor<1x1x128xf16>) {
  %cst = const.Declare tensor<1x384x256xf16> = dense<1.000000e+00> : tensor<1x384x256xf16>
  %cst_0 = const.Declare tensor<1x384x128xf16> = dense<1.000000e+00> : tensor<1x384x128xf16>
  %cst_1 = const.Declare tensor<1x512xf16> = dense<1.000000e+00> : tensor<1x512xf16>
  %middle_hidden_state, %output_hidden_state = IE.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 128 : i64, seq_length = 80 : i64, should_linear_before_reset} : tensor<1x80x256xf16>, tensor<1x1x128xf16>, tensor<1x384x256xf16>, tensor<1x384x128xf16>, tensor<1x512xf16> -> tensor<1x1x80x128xf16>, tensor<1x1x128xf16>
  return %middle_hidden_state, %output_hidden_state : tensor<1x1x80x128xf16>, tensor<1x1x128xf16>

    // CHECK:         [[CST0:%.+]] = const.Declare tensor<1x384x256xf16> = dense<1.000000e+00> : tensor<1x384x256xf16>
    // CHECK:         [[CST1:%.+]] = const.Declare tensor<1x384x128xf16> = dense<1.000000e+00> : tensor<1x384x128xf16>
    // CHECK:         [[CST2:%.+]] = const.Declare tensor<1x512xf16> = dense<1.000000e+00> : tensor<1x512xf16>
    // CHECK:         [[RESHAPE0:%.+]] = IE.Reshape([[INPUT0]]) {shape_value = [80, 1, 1, 256]} : tensor<1x80x256xf16> -> tensor<80x1x1x256xf16>
    // CHECK:         [[RESHAPE1:%.+]] = IE.Reshape([[CST0]]) {shape_value = [1, 1, 384, 256]} : tensor<1x384x256xf16> -> tensor<1x1x384x256xf16>
    // CHECK:         [[CONVOLUTION:%.+]] = IE.Convolution([[RESHAPE1]], [[RESHAPE0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x384x256xf16>, tensor<80x1x1x256xf16> -> tensor<1x80x384x1xf16>
    // CHECK:         [[RESHAPE2:%.+]] = IE.Reshape([[CONVOLUTION]]) {shape_value = [1, 80, 384, 1]} : tensor<1x80x384x1xf16> -> tensor<1x80x384x1xf16>
    // CHECK:         [[PERMUTECAST:%.+]] = IE.PermuteCast([[RESHAPE2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x80x384x1xf16> -> tensor<1x1x80x384xf16, {order = #NHWC}>
    // CHECK:         [[OUTPUT0:%.+]], [[OUTPUT1:%.+]] = IE.GRUSequenceLastPart([[PERMUTECAST]], [[INPUT1]], [[CST1]], [[CST2]]) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 128 : i64, seq_length = 80 : i64, should_linear_before_reset} : tensor<1x1x80x384xf16, {order = #NHWC}>, tensor<1x1x128xf16>, tensor<1x384x128xf16>, tensor<1x512xf16> -> tensor<1x1x80x128xf16>, tensor<1x1x128xf16>
    // CHECK:         return [[OUTPUT0]], [[OUTPUT1]] : tensor<1x1x80x128xf16>, tensor<1x1x128xf16>
}
