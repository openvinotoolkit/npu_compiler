//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tile-lstm-sequence --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: func.func @TileBidirectionalLSTMSequence(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<1x2x640x512xf16>) -> (tensor<1x2x640x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>) {
func.func @TileBidirectionalLSTMSequence(%arg0: tensor<1x2x640x512xf16>) -> (tensor<1x2x640x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>) {
    %cst_0 = const.Declare tensor<1x2x1x128xf16> = dense<1.000000e+00> : tensor<1x2x1x128xf16>
    %cst_2 = const.Declare tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}> = dense<3.000000e+00> : tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>
    %cst_5 = const.Declare tensor<1x2x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x2x4x128xf16, {order = #NCWH}>
    %cst = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_2, %cst_5, %cst) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 640 : i64} : tensor<1x2x640x512xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>, tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x2x640x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x2x640x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>

// CHECK-DAG:   [[VAL_1:%.+]] = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>
// CHECK-DAG:   [[VAL_2:%.+]] = const.Declare tensor<1x2x1x128xf16> = dense<1.000000e+00> : tensor<1x2x1x128xf16>
// CHECK-DAG:   [[VAL_3:%.+]] = const.Declare tensor<2x4x128x128xf16, {order = #NWHC}> = dense<3.000000e+00> : tensor<2x4x128x128xf16, {order = #NWHC}>
// CHECK-DAG:   [[VAL_33:%.+]] = const.Declare tensor<1x2x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x2x4x128xf16, {order = #NCWH}>
// CHECK-DAG:   [[VAL_4:%.+]] = VPU.Slice [[VAL_0]] [0, 0, 0, 0] [1, 1, 320, 512] : tensor<1x2x640x512xf16> to tensor<1x1x320x512xf16>
// CHECK-DAG:   [[VAL_5:%.+]] = VPU.Slice [[VAL_0]] [0, 1, 320, 0] [1, 1, 320, 512] : tensor<1x2x640x512xf16> to tensor<1x1x320x512xf16>
// CHECK:   [[VAL_8:%.+]] = VPU.Concat([[VAL_4]], [[VAL_5]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x320x512xf16>, tensor<1x1x320x512xf16> -> tensor<1x2x320x512xf16>
// CHECK:   [[VAL_9:%.+]], [[VAL_10:.*]], [[VAL_11:.*]] = VPU.LSTMSequence([[VAL_8]], [[VAL_2]], [[VAL_2]], [[VAL_3]],  [[VAL_33]], [[VAL_1]]) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 320 : i64} : tensor<1x2x320x512xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>, tensor<2x4x128x128xf16, {order = #NWHC}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x2x320x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>
// CHECK:   [[VAL_12:%.+]] = VPU.Slice [[VAL_9]] [0, 0, 0, 0] [1, 1, 320, 128] : tensor<1x2x320x128xf16> to tensor<1x1x320x128xf16>
// CHECK:   [[VAL_13:%.+]] = VPU.Slice [[VAL_9]] [0, 1, 0, 0] [1, 1, 320, 128] : tensor<1x2x320x128xf16> to tensor<1x1x320x128xf16>
// CHECK:   [[VAL_14:%.+]] = VPU.Slice [[VAL_0]] [0, 0, 320, 0] [1, 1, 320, 512] : tensor<1x2x640x512xf16> to tensor<1x1x320x512xf16>
// CHECK:   [[VAL_15:%.+]] = VPU.Slice [[VAL_0]] [0, 1, 0, 0] [1, 1, 320, 512] : tensor<1x2x640x512xf16> to tensor<1x1x320x512xf16>
// CHECK:   [[VAL_16:%.+]] = VPU.Concat([[VAL_14]], [[VAL_15]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x320x512xf16>, tensor<1x1x320x512xf16> -> tensor<1x2x320x512xf16>
// CHECK:   [[VAL_17:%.+]], [[VAL_18:.*]], [[VAL_19:.*]] = VPU.LSTMSequence([[VAL_16]], [[VAL_10]], [[VAL_11]], [[VAL_3]], [[VAL_33]], [[VAL_1]]) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 320 : i64} : tensor<1x2x320x512xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>, tensor<2x4x128x128xf16, {order = #NWHC}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x2x320x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>
// CHECK:   [[VAL_20:%.+]] = VPU.Slice [[VAL_17]] [0, 0, 0, 0] [1, 1, 320, 128] : tensor<1x2x320x128xf16> to tensor<1x1x320x128xf16>
// CHECK:   [[VAL_21:%.+]] = VPU.Slice [[VAL_17]] [0, 1, 0, 0] [1, 1, 320, 128] : tensor<1x2x320x128xf16> to tensor<1x1x320x128xf16>
// CHECK:   [[VAL_22:%.+]] = VPU.Concat([[VAL_12]], [[VAL_20]], [[VAL_21]], [[VAL_13]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 0, 320, 0], [0, 1, 0, 0], [0, 1, 320, 0]]} : tensor<1x1x320x128xf16>, tensor<1x1x320x128xf16>, tensor<1x1x320x128xf16>, tensor<1x1x320x128xf16> -> tensor<1x2x640x128xf16>
// CHECK:   return [[VAL_22]], [[VAL_18]], [[VAL_19]] : tensor<1x2x640x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: func.func @TileForwardLSTMSequence(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<1x1x1280x512xf16>) -> (tensor<1x1x1280x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
func.func @TileForwardLSTMSequence(%arg0: tensor<1x1x1280x512xf16>) -> (tensor<1x1x1280x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
    %cst_0 = const.Declare tensor<1x1x1x128xf16> = dense<1.000000e+00> : tensor<1x1x1x128xf16>
    %cst_2 = const.Declare tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}> = dense<3.000000e+00> : tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>
    %cst_5 = const.Declare tensor<1x1x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x1x4x128xf16, {order = #NCWH}>
    %cst = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_2, %cst_5, %cst) {direction = #IE.rnn_seq_direction<FORWARD>, sequenceLength = 1280 : i64} : tensor<1x1x1280x512xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x1x1280x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x1x1280x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

// CHECK-DAG:   [[VAL_1:%.+]] = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>
// CHECK-DAG:   [[VAL_2:%.+]] = const.Declare tensor<1x1x1x128xf16> = dense<1.000000e+00> : tensor<1x1x1x128xf16>
// CHECK-DAG:   [[VAL_3:%.+]] = const.Declare tensor<1x4x128x128xf16, {order = #NWHC}> = dense<3.000000e+00> : tensor<1x4x128x128xf16, {order = #NWHC}>
// CHECK-DAG:   [[VAL_33:%.+]] = const.Declare tensor<1x1x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x1x4x128xf16, {order = #NCWH}>
// CHECK-DAG:   [[VAL_4:%.+]] = VPU.Slice [[VAL_0]] [0, 0, 0, 0] [1, 1, 640, 512] : tensor<1x1x1280x512xf16> to tensor<1x1x640x512xf16>
// CHECK:   [[VAL_5:%.+]], [[VAL_6:%.+]], [[VAL_7:%.+]] = VPU.LSTMSequence([[VAL_4]], [[VAL_2]], [[VAL_2]], [[VAL_3]], [[VAL_33]], [[VAL_1]]) {direction = #IE.rnn_seq_direction<FORWARD>, sequenceLength = 640 : i64} : tensor<1x1x640x512xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x1x640x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
// CHECK:   [[VAL_8:%.+]] = VPU.Slice [[VAL_0]] [0, 0, 640, 0] [1, 1, 640, 512] : tensor<1x1x1280x512xf16> to tensor<1x1x640x512xf16>
// CHECK:   [[VAL_9:%.+]], [[VAL_10:%.+]], [[VAL_11:%.+]] = VPU.LSTMSequence([[VAL_8]], [[VAL_6]], [[VAL_7]], [[VAL_3]], [[VAL_33]], [[VAL_1]]) {direction = #IE.rnn_seq_direction<FORWARD>, sequenceLength = 640 : i64} : tensor<1x1x640x512xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x1x640x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
// CHECK:   [[VAL_12:%.+]] = VPU.Concat([[VAL_5]], [[VAL_9]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 0, 640, 0]]} : tensor<1x1x640x128xf16>, tensor<1x1x640x128xf16> -> tensor<1x1x1280x128xf16>
// CHECK:   return [[VAL_12]], [[VAL_10]], [[VAL_11]] : tensor<1x1x1280x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: func.func @TileReverseLSTMSequence(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<1x1x1280x512xf16>) -> (tensor<1x1x1280x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
func.func @TileReverseLSTMSequence(%arg0: tensor<1x1x1280x512xf16>) -> (tensor<1x1x1280x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
    %cst_0 = const.Declare tensor<1x1x1x128xf16> = dense<1.000000e+00> : tensor<1x1x1x128xf16>
    %cst_2 = const.Declare tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}> = dense<3.000000e+00> : tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>
    %cst_5 = const.Declare tensor<1x1x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x1x4x128xf16, {order = #NCWH}>
    %cst = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_2, %cst_5, %cst) {direction = #IE.rnn_seq_direction<REVERSE>, sequenceLength = 1280 : i64} : tensor<1x1x1280x512xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x1x1280x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x1x1280x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

// CHECK-DAG:   [[VAL_1:%.+]] = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>
// CHECK-DAG:   [[VAL_2:%.+]] = const.Declare tensor<1x1x1x128xf16> = dense<1.000000e+00> : tensor<1x1x1x128xf16>
// CHECK-DAG:   [[VAL_3:%.+]] = const.Declare tensor<1x4x128x128xf16, {order = #NWHC}> = dense<3.000000e+00> : tensor<1x4x128x128xf16, {order = #NWHC}>
// CHECK-DAG:   [[VAL_33:%.+]] = const.Declare tensor<1x1x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x1x4x128xf16, {order = #NCWH}>
// CHECK-DAG:   [[VAL_4:%.+]] = VPU.Slice [[VAL_0]] [0, 0, 640, 0] [1, 1, 640, 512] : tensor<1x1x1280x512xf16> to tensor<1x1x640x512xf16>
// CHECK:   [[VAL_5:%.+]], [[VAL_6:%.+]], [[VAL_7:%.+]] = VPU.LSTMSequence([[VAL_4]], [[VAL_2]], [[VAL_2]], [[VAL_3]], [[VAL_33]], [[VAL_1]]) {direction = #IE.rnn_seq_direction<REVERSE>, sequenceLength = 640 : i64} : tensor<1x1x640x512xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x1x640x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
// CHECK:   [[VAL_8:%.+]] = VPU.Slice [[VAL_0]] [0, 0, 0, 0] [1, 1, 640, 512] : tensor<1x1x1280x512xf16> to tensor<1x1x640x512xf16>
// CHECK:   [[VAL_9:%.+]], [[VAL_10:%.+]], [[VAL_11:%.+]] = VPU.LSTMSequence([[VAL_8]], [[VAL_6]], [[VAL_7]], [[VAL_3]], [[VAL_33]], [[VAL_1]]) {direction = #IE.rnn_seq_direction<REVERSE>, sequenceLength = 640 : i64} : tensor<1x1x640x512xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x1x640x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
// CHECK:   [[VAL_12:%.+]] = VPU.Concat([[VAL_9]], [[VAL_5]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 0, 640, 0]]} : tensor<1x1x640x128xf16>, tensor<1x1x640x128xf16> -> tensor<1x1x1280x128xf16>
// CHECK:   return [[VAL_12]], [[VAL_10]], [[VAL_11]] : tensor<1x1x1280x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: func.func @TileBidirectionalLSTMSequenceUnevenTiling(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<1x2x962x512xf16>) -> (tensor<1x2x962x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>) {
func.func @TileBidirectionalLSTMSequenceUnevenTiling(%arg0: tensor<1x2x962x512xf16>) -> (tensor<1x2x962x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>) {
    %cst_0 = const.Declare tensor<1x2x1x128xf16> = dense<1.000000e+00> : tensor<1x2x1x128xf16>
    %cst_2 = const.Declare tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}> = dense<3.000000e+00> : tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>
    %cst_5 = const.Declare tensor<1x2x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x2x4x128xf16, {order = #NCWH}>
    %cst = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_2, %cst_5, %cst) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 962 : i64} : tensor<1x2x962x512xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>, tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x2x962x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x2x962x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>

// CHECK:       VPU.LSTMSequence
// CHECK-SAME:      {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 321 : i64} : tensor<1x2x321x512xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>, tensor<2x4x128x128xf16, {order = #NWHC}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x2x321x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>

// CHECK:       VPU.LSTMSequence
// CHECK-SAME:      {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 321 : i64} : tensor<1x2x321x512xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>, tensor<2x4x128x128xf16, {order = #NWHC}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x2x321x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>

// CHECK:       VPU.LSTMSequence
// CHECK-SAME:      {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 320 : i64} : tensor<1x2x320x512xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>, tensor<2x4x128x128xf16, {order = #NWHC}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x2x320x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: func.func @TileForwardLSTMSequenceUnevenTiling(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<1x1x1982x512xf16>) -> (tensor<1x1x1982x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
func.func @TileForwardLSTMSequenceUnevenTiling(%arg0: tensor<1x1x1982x512xf16>) -> (tensor<1x1x1982x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
    %cst_0 = const.Declare tensor<1x1x1x128xf16> = dense<1.000000e+00> : tensor<1x1x1x128xf16>
    %cst_2 = const.Declare tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}> = dense<3.000000e+00> : tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>
    %cst_5 = const.Declare tensor<1x1x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x1x4x128xf16, {order = #NCWH}>
    %cst = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_2, %cst_5, %cst) {direction = #IE.rnn_seq_direction<FORWARD>, sequenceLength = 1982 : i64} : tensor<1x1x1982x512xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x1x1982x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x1x1982x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

// CHECK:       VPU.LSTMSequence
// CHECK-SAME:      {direction = #IE.rnn_seq_direction<FORWARD>, sequenceLength = 661 : i64} : tensor<1x1x661x512xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x1x661x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

// CHECK:       VPU.LSTMSequence
// CHECK-SAME:      {direction = #IE.rnn_seq_direction<FORWARD>, sequenceLength = 661 : i64} : tensor<1x1x661x512xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x1x661x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

// CHECK:       VPU.LSTMSequence
// CHECK-SAME:      {direction = #IE.rnn_seq_direction<FORWARD>, sequenceLength = 660 : i64} : tensor<1x1x660x512xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x1x660x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: func.func @TileBidirectionalLSTMSequenceSplitOverKernel(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<1x2x962x512xf16>) -> (tensor<1x2x962x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>) {
func.func @TileBidirectionalLSTMSequenceSplitOverKernel(%arg0: tensor<1x2x962x512xf16>) -> (tensor<1x2x962x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>) {
    %cst_0 = const.Declare tensor<1x2x1x128xf16> = dense<1.000000e+00> : tensor<1x2x1x128xf16>
    %cst_2 = const.Declare tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}> = dense<3.000000e+00> : tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>
    %cst_5 = const.Declare tensor<1x2x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x2x4x128xf16, {order = #NCWH}>
    %cst = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_2, %cst_5, %cst) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, sequenceLength = 962 : i64} : tensor<1x2x962x512xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>, tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x2x962x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x2x962x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>

// CHECK-DAG:   [[VAL_1:%.+]] = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>
// CHECK-DAG:   [[VAL_2:%.+]] = const.Declare tensor<1x2x1x128xf16> = dense<1.000000e+00> : tensor<1x2x1x128xf16>
// CHECK-DAG:   [[VAL_3:%.+]] = const.Declare tensor<2x4x128x128xf16, {order = #NWHC}> = dense<3.000000e+00> : tensor<2x4x128x128xf16, {order = #NWHC}>
// CHECK-DAG:   [[VAL_33:%.+]] = const.Declare tensor<1x2x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x2x4x128xf16, {order = #NCWH}>
// CHECK-DAG:   [[VAL_4:%.+]] = VPU.Slice [[VAL_0]] [0, 0, 0, 0] [1, 1, 481, 512] : tensor<1x2x962x512xf16> to tensor<1x1x481x512xf16>
// CHECK-DAG:   [[VAL_5:%.+]] = VPU.Slice [[VAL_0]] [0, 1, 481, 0] [1, 1, 481, 512] : tensor<1x2x962x512xf16> to tensor<1x1x481x512xf16>
// CHECK:   [[VAL_8:%.+]] = VPU.Concat([[VAL_4]], [[VAL_5]])
// CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x481x512xf16>, tensor<1x1x481x512xf16> -> tensor<1x2x481x512xf16>
// CHECK:   [[VAL_9:%.+]], [[VAL_10:.*]], [[VAL_11:.*]] = VPU.LSTMSequence([[VAL_8]], [[VAL_2]], [[VAL_2]], [[VAL_3]], [[VAL_33]], [[VAL_1]]) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, sequenceLength = 481 : i64} : tensor<1x2x481x512xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>, tensor<2x4x128x128xf16, {order = #NWHC}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x2x481x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>
// CHECK:   [[VAL_12:%.+]] = VPU.Slice [[VAL_9]] [0, 0, 0, 0] [1, 1, 481, 128] : tensor<1x2x481x128xf16> to tensor<1x1x481x128xf16>
// CHECK:   [[VAL_13:%.+]] = VPU.Slice [[VAL_9]] [0, 1, 0, 0] [1, 1, 481, 128] : tensor<1x2x481x128xf16> to tensor<1x1x481x128xf16>
// CHECK:   [[VAL_14:%.+]] = VPU.Slice [[VAL_0]] [0, 0, 481, 0] [1, 1, 481, 512] : tensor<1x2x962x512xf16> to tensor<1x1x481x512xf16>
// CHECK:   [[VAL_15:%.+]] = VPU.Slice [[VAL_0]] [0, 1, 0, 0] [1, 1, 481, 512] : tensor<1x2x962x512xf16> to tensor<1x1x481x512xf16>
// CHECK:   [[VAL_16:%.+]] = VPU.Concat([[VAL_14]], [[VAL_15]])
// CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x481x512xf16>, tensor<1x1x481x512xf16> -> tensor<1x2x481x512xf16>
// CHECK:   [[VAL_17:%.+]], [[VAL_18:.*]], [[VAL_19:.*]] = VPU.LSTMSequence([[VAL_16]], [[VAL_10]], [[VAL_11]], [[VAL_3]], [[VAL_33]], [[VAL_1]]) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, sequenceLength = 481 : i64} : tensor<1x2x481x512xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>, tensor<2x4x128x128xf16, {order = #NWHC}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x2x481x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>
// CHECK:   [[VAL_20:%.+]] = VPU.Slice [[VAL_17]] [0, 0, 0, 0] [1, 1, 481, 128] : tensor<1x2x481x128xf16> to tensor<1x1x481x128xf16>
// CHECK:   [[VAL_21:%.+]] = VPU.Slice [[VAL_17]] [0, 1, 0, 0] [1, 1, 481, 128] : tensor<1x2x481x128xf16> to tensor<1x1x481x128xf16>
// CHECK:   [[VAL_22:%.+]] = VPU.Concat([[VAL_12]], [[VAL_20]], [[VAL_21]], [[VAL_13]])
// CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 481, 0], [0, 1, 0, 0], [0, 1, 481, 0]]} : tensor<1x1x481x128xf16>, tensor<1x1x481x128xf16>, tensor<1x1x481x128xf16>, tensor<1x1x481x128xf16> -> tensor<1x2x962x128xf16>
// CHECK:   return [[VAL_22]], [[VAL_18]], [[VAL_19]] : tensor<1x2x962x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: func.func @TileReverseLSTMSequenceSplitOverBatch(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<3x1x990x512xf16>) -> (tensor<3x1x990x128xf16>, tensor<3x1x1x128xf16>, tensor<3x1x1x128xf16>) {
func.func @TileReverseLSTMSequenceSplitOverBatch(%arg0: tensor<3x1x990x512xf16>) -> (tensor<3x1x990x128xf16>, tensor<3x1x1x128xf16>, tensor<3x1x1x128xf16>) {
    %cst_0 = const.Declare tensor<3x1x1x128xf16> = dense<1.000000e+00> : tensor<3x1x1x128xf16>
    %cst_2 = const.Declare tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}> = dense<3.000000e+00> : tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>
    %cst_5 = const.Declare tensor<1x1x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x1x4x128xf16, {order = #NCWH}>
    %cst = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_2, %cst_5, %cst) {direction = #IE.rnn_seq_direction<REVERSE>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>, sequenceLength = 990 : i64} : tensor<3x1x990x512xf16>, tensor<3x1x1x128xf16>, tensor<3x1x1x128xf16>, tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<3x1x990x128xf16>, tensor<3x1x1x128xf16>, tensor<3x1x1x128xf16>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<3x1x990x128xf16>, tensor<3x1x1x128xf16>, tensor<3x1x1x128xf16>

// CHECK-DAG:   [[VAL_1:%.+]] = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>
// CHECK-DAG:   [[VAL_2:%.+]] = const.Declare tensor<3x1x1x128xf16> = dense<1.000000e+00> : tensor<3x1x1x128xf16>
// CHECK-DAG:   [[VAL_3:%.+]] = const.Declare tensor<1x4x128x128xf16, {order = #NWHC}> = dense<3.000000e+00> : tensor<1x4x128x128xf16, {order = #NWHC}>
// CHECK-DAG:   [[VAL_33:%.+]] = const.Declare tensor<1x1x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x1x4x128xf16, {order = #NCWH}>
// CHECK-DAG:   [[VAL_4:%.+]] = VPU.Slice [[VAL_0]] [0, 0, 495, 0] [3, 1, 495, 512] : tensor<3x1x990x512xf16> to tensor<3x1x495x512xf16>
// CHECK:   [[VAL_5:%.+]], [[VAL_6:%.+]], [[VAL_7:%.+]] = VPU.LSTMSequence([[VAL_4]], [[VAL_2]], [[VAL_2]], [[VAL_3]], [[VAL_33]], [[VAL_1]]) {direction = #IE.rnn_seq_direction<REVERSE>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>, sequenceLength = 495 : i64} : tensor<3x1x495x512xf16>, tensor<3x1x1x128xf16>, tensor<3x1x1x128xf16>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<3x1x495x128xf16>, tensor<3x1x1x128xf16>, tensor<3x1x1x128xf16>
// CHECK:   [[VAL_8:%.+]] = VPU.Slice [[VAL_0]] [0, 0, 0, 0] [3, 1, 495, 512] : tensor<3x1x990x512xf16> to tensor<3x1x495x512xf16>
// CHECK:   [[VAL_9:%.+]], [[VAL_10:%.+]], [[VAL_11:%.+]] = VPU.LSTMSequence([[VAL_8]], [[VAL_6]], [[VAL_7]], [[VAL_3]], [[VAL_33]], [[VAL_1]]) {direction = #IE.rnn_seq_direction<REVERSE>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>, sequenceLength = 495 : i64} : tensor<3x1x495x512xf16>, tensor<3x1x1x128xf16>, tensor<3x1x1x128xf16>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<3x1x495x128xf16>, tensor<3x1x1x128xf16>, tensor<3x1x1x128xf16>
// CHECK:   [[VAL_12:%.+]] = VPU.Concat([[VAL_9]], [[VAL_5]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 0, 495, 0]]} : tensor<3x1x495x128xf16>, tensor<3x1x495x128xf16> -> tensor<3x1x990x128xf16>
// CHECK:   return [[VAL_12]], [[VAL_10]], [[VAL_11]] : tensor<3x1x990x128xf16>, tensor<3x1x1x128xf16>, tensor<3x1x1x128xf16>
}
