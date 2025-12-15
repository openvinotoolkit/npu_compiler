//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-layers-to-VPU %s | FileCheck %s
// REQUIRES: arch-NPU50XX


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @dynamicLSTMSequence
// CHECK-SAME: ([[ARG_0:.+]]: tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, [[ARG_1:.+]]: tensor<1x1x1x128xf16>, [[ARG_2:.+]]: tensor<1x1x1x128xf16>) -> (tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
func.func @dynamicLSTMSequence(%arg0: tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, %arg1: tensor<1x1x1x128xf16>, %arg2: tensor<1x1x1x128xf16>) -> (tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
    %cst = const.Declare tensor<1x4x128x128xf16> = dense<0.000000e+00> : tensor<1x512x128xf32>, [#const.Reshape<[1, 4, 128, 128]>, #const.CastElemType<f16>]
    %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %arg1, %arg2, %cst) {direction = #IE.rnn_seq_direction<FORWARD>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>, useDpu = true} : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16> -> tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x4x128x128xf16> = dense<0.000000e+00> : tensor<1x512x128xf32>, [#const.Reshape<[1, 4, 128, 128]>, #const.CastElemType<f16>]
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x2x2432xsi32> = dense<0> : tensor<1x1x2x2432xsi32>
    // CHECK: [[OUT_HV:%.+]], [[OUT_HS:%.+]], [[OUT_CS:%.+]] = VPU.LSTMSequence([[ARG_0]], [[ARG_1]], [[ARG_2]], [[CST]], [[CST_0]]) {direction = #IE.rnn_seq_direction<FORWARD>, useDpu = true} : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 512]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16>, tensor<1x1x2x2432xsi32> -> tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
    // CHECK: return [[OUT_HV]], [[OUT_HS]], [[OUT_CS]] : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 35, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
}

// -----

// CHECK-LABEL: @FlashSDPA
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<1x8x64x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<1x8x32x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<1x8x128x32xf16>
func.func @FlashSDPA(%arg0: tensor<1x8x64x64xf16>, %arg1: tensor<1x8x32x64xf16>, %arg2: tensor<1x8x128x32xf16>) -> tensor<1x8x64x128xf16> {
    %0 = IE.FlashSDPA(%arg0, %arg1, %arg2) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, source_seq_len_pad_size = 0 : i64}
            : tensor<1x8x64x64xf16>, tensor<1x8x32x64xf16>, tensor<1x8x128x32xf16> -> tensor<1x8x64x128xf16>

    return %0 : tensor<1x8x64x128xf16>

    // CHECK-DAG:   [[IN_OUT:%.+]] = const.Declare tensor<1x8x64x128xf16> = dense<0.000000e+00> : tensor<1x8x64x128xf16>
    // CHECK-DAG:   [[IN_MAX:%.+]] = const.Declare tensor<1x8x64x1xf16> = dense<0xFC00> : tensor<1x8x64x1xf16>
    // CHECK-DAG:   [[IN_SUM:%.+]] = const.Declare tensor<1x8x64x1xf16> = dense<0.000000e+00> : tensor<1x8x64x1xf16>
    // CHECK-DAG:   [[AUX_BUF:%.+]] = const.Declare tensor<1x8x64x32xf16> = dense<0.000000e+00> : tensor<1x8x64x32xf16>
    // CHECK-DAG:   [[DPU_DESCRIPTORS_BUF:%.+]] = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    // CHECK-DAG:   [[WEIGHTS_TABLE_0:%.+]] = const.Declare tensor<1x1x32x4xsi32> = dense
    // CHECK-DAG:   [[WEIGHTS_TABLE_1:%.+]] = const.Declare tensor<1x1x128x4xsi32> = dense

    // CHECK:           [[RES_OUT:%[^, ]+]], [[RES_MAX:%[^, ]+]], [[RES_SUM:%[^, ]+]], [[RES_QUERY:%[^, ]+]] =
    // CHECK-SAME:              VPU.FlashSDPA([[QUERY]], [[KEY]], [[VALUE]], [[AUX_BUF]],
    // CHECK-SAME:                            [[DPU_DESCRIPTORS_BUF]], [[WEIGHTS_TABLE_0]], [[WEIGHTS_TABLE_1]],
    // CHECK-SAME:                            [[IN_OUT]], [[IN_MAX]], [[IN_SUM]]) {
    // CHECK-SAME:                      is_head = true,
    // CHECK-SAME:                      is_tail = true,
    // CHECK-SAME:                      source_seq_len_pad_size = 0 : i64
    // CHECK-SAME:                  -> tensor<1x8x64x128xf16>, tensor<1x8x64x1xf16>, tensor<1x8x64x1xf16>, tensor<1x8x64x64xf16>

    // CHECK:   return [[RES_OUT]]
}
