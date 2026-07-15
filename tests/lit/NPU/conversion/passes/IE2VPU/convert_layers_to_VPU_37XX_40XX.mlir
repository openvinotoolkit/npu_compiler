//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --convert-layers-to-VPU %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @dynamicLSTMSequence
// CHECK-SAME: ([[ARG_0:.+]]: tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, [[ARG_1:.+]]: tensor<1x1x1x128xf16>, [[ARG_2:.+]]: tensor<1x1x1x128xf16>) -> (tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
func.func @dynamicLSTMSequence(%arg0: tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, %arg1: tensor<1x1x1x128xf16>, %arg2: tensor<1x1x1x128xf16>) -> (tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
    %cst = const.Declare tensor<1x4x128x128xf16, {order = #NWHC}> = dense<0.000000e+00> : tensor<1x512x128xf32>, [#const.Reshape<[1, 4, 128, 128]>, #const.CastElemType<f16>, #const.Reorder<#NWHC>]
    %cst_1 = const.Declare tensor<1x1x4x128xf16, {order = #NCWH}> = dense<0.000000e+00> : tensor<2x512xf32>, [#const.SubView<[0, 0], [1, 512]>, #const.Reshape<[1, 1, 4, 128]>, #const.CastElemType<f16>, #const.Reorder<#NCWH>]
    %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %arg1, %arg2, %cst, %cst_1) {direction = #IE.rnn_seq_direction<FORWARD>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1, 1>} : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}> -> tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x4x128x128xf16, {order = #NWHC}> = dense<0.000000e+00> : tensor<1x512x128xf32>, [#const.Reshape<[1, 4, 128, 128]>, #const.CastElemType<f16>, #const.Reorder<#NWHC>]
    // CHECK: [[BIASES:%.+]] = const.Declare tensor<1x1x4x128xf16, {order = #NCWH}> = dense<0.000000e+00> : tensor<2x512xf32>, [#const.SubView<[0, 0], [1, 512]>, #const.Reshape<[1, 1, 4, 128]>, #const.CastElemType<f16>, #const.Reorder<#NCWH>]
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>
    // CHECK: [[OUT_HV:%.+]], [[OUT_HS:%.+]], [[OUT_CS:%.+]] = VPU.LSTMSequence([[ARG_0]], [[ARG_1]], [[ARG_2]], [[CST]], [[BIASES]], [[CST_0]]) {direction = #IE.rnn_seq_direction<FORWARD>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 1, 1>} : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
    // CHECK: return [[OUT_HV]], [[OUT_HS]], [[OUT_CS]] : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @SeqLenParamLSTMSequence
// CHECK-SAME: [[ARG0:%[^, ]+]]: tensor<2x2x25x512xf16>,
// CHECK-SAME: [[ARG1:%[^, ]+]]: tensor<2x2x1x128xf16>,
// CHECK-SAME: [[ARG2:%[^, ]+]]: tensor<2x2x1x128xf16>
// CHECK-SAME: [[ARG3:%[^, ]+]]: tensor<2x1x1x1xsi32>
func.func @SeqLenParamLSTMSequence (%arg0: tensor<2x2x25x512xf16>, %arg1: tensor<2x2x1x128xf16>, %arg2: tensor<2x2x1x128xf16>, %arg3: tensor<2x1x1x1xsi32>) -> (tensor<2x2x25x128xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>) {
    %cst = const.Declare tensor<1x2x4x128xf16, {order = #NCWH}> = dense<0.000000e+00> : tensor<2x512xf32>, [#const.Reshape<[1, 2, 4, 128]>, #const.CastElemType<f16>, #const.Reorder<#NCWH>]
    %cst_0 = const.Declare tensor<2x4x128x128xf16, {order = #NWHC}> = dense<0.000000e+00> : tensor<2x512x128xf32>, [#const.Reshape<[2, 4, 128, 128]>, #const.CastElemType<f16>, #const.Reorder<#NWHC>]
    %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %arg1, %arg2, %arg3, %cst_0, %cst) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 1>} : tensor<2x2x25x512xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>, tensor<2x1x1x1xsi32>, tensor<2x4x128x128xf16, {order = #NWHC}>, tensor<1x2x4x128xf16, {order = #NCWH}> -> tensor<2x2x25x128xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>
    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<2x2x25x128xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x2x4x128xf16, {order = #NCWH}> = dense<0.000000e+00> : tensor<2x512xf32>, [#const.Reshape<[1, 2, 4, 128]>, #const.CastElemType<f16>, #const.Reorder<#NCWH>]
    // CHECK: [[BIASES:%.+]] = const.Declare tensor<2x4x128x128xf16, {order = #NWHC}> = dense<0.000000e+00> : tensor<2x512x128xf32>, [#const.Reshape<[2, 4, 128, 128]>, #const.CastElemType<f16>, #const.Reorder<#NWHC>]
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>
    // CHECK: [[OUT_HV:%.+]], [[OUT_HS:%.+]], [[OUT_CS:%.+]] = VPU.LSTMSequence([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]], [[BIASES]], [[CST]], [[CST_1]]) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>} : tensor<2x2x25x512xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>, tensor<2x1x1x1xsi32>, tensor<2x4x128x128xf16, {order = #NWHC}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<2x2x25x128xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>
    // CHECK: return [[OUT_HV]], [[OUT_HS]], [[OUT_CS]] : tensor<2x2x25x128xf16>, tensor<2x2x1x128xf16>, tensor<2x2x1x128xf16>
}
