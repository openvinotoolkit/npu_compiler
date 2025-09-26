//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-incremental-sdpa  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @DecomposeIncrementalSdpaWithAttentionMaskNoScale
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<8x512x512xf16>
func.func @DecomposeIncrementalSdpaWithAttentionMaskNoScale(%arg0: tensor<8x512x64xf16>, %arg1: tensor<8x512x64xf16>, %arg2: tensor<8x512x64xf16>, %arg3: tensor<8x512x512xf16>) -> tensor<8x512x64xf16> {
    %cst = const.Declare tensor<8x512x64xf16> = dense<0.000000e+00> : tensor<8x512x64xf16>
    %cst_0 = const.Declare tensor<8x512xf16> = dense<0xFC00> : tensor<8x512xf16>
    %cst_1 = const.Declare tensor<8x512xf16> = dense<0.000000e+00> : tensor<8x512xf16>
    %result_partial_output, %result_running_max, %result_running_sum = IE.IncrementalSDPA(%arg0, %arg1, %arg2, %cst, %cst_0, %cst_1, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0>} : tensor<8x512x64xf16>, tensor<8x512x64xf16>, tensor<8x512x64xf16>, tensor<8x512x64xf16>, tensor<8x512xf16>, tensor<8x512xf16>, tensor<8x512x512xf16> -> tensor<8x512x64xf16>, tensor<8x512xf16>, tensor<8x512xf16>
    %0 = IE.Unsqueeze(%result_running_sum) {axes_value = [2]} : tensor<8x512xf16> -> tensor<8x512x1xf16>
    %1 = IE.Divide(%result_partial_output, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x512x64xf16>, tensor<8x512x1xf16> -> tensor<8x512x64xf16>
    return %1 : tensor<8x512x64xf16>

    // CHECK-DAG:   [[MAX_INITIAL:%.+]] = const.Declare tensor<8x512xf16> = dense<0xFC00> : tensor<8x512xf16>
    // CHECK-DAG:   [[SUM_INITIAL:%.+]] = const.Declare tensor<8x512xf16> = dense<0.000000e+00> : tensor<8x512xf16>
    // CHECK-DAG:   [[OUT_INITIAL:%.+]] = const.Declare tensor<8x512x64xf16> = dense<0.000000e+00> : tensor<8x512x64xf16>
    // CHECK-DAG:   [[CONST_SCALE:%.+]] = const.Declare tensor<1xf16> = dense<1.250000e-01> : tensor<1xf16>

    // CHECK:       [[SCALED_QUERY:%.+]] = IE.Multiply([[QUERY]], [[CONST_SCALE]])
    // CHECK-SAME:      -> tensor<8x512x64xf16>
    // CHECK:       [[QK_MATMUL:%.+]] = IE.MatMul([[SCALED_QUERY]], [[KEY]])
    // CHECK-SAME:      {transpose_b}
    // CHECK-SAME:      -> tensor<8x512x512xf16>
    // CHECK:       [[ADD_ATTENTION_MASK:%.+]] = IE.Add([[QK_MATMUL]], [[ATTENTION_MASK]])
    // CHECK-SAME:      -> tensor<8x512x512xf16>
    // CHECK:       [[REDUCE_MAX:%.+]] = IE.ReduceMax([[ADD_ATTENTION_MASK]])
    // CHECK-SAME:      -> tensor<8x512xf16>
    // CHECK:       [[NEW_MAX:%.+]] = IE.Maximum([[MAX_INITIAL]], [[REDUCE_MAX]])
    // CHECK-SAME:      -> tensor<8x512xf16>

    // CHECK:       [[SUBTRACT_MAX:%.+]] = IE.Subtract([[MAX_INITIAL]], [[NEW_MAX]])
    // CHECK-SAME:      -> tensor<8x512xf16>
    // CHECK:       [[EXP:%.+]] = IE.Exp([[SUBTRACT_MAX]])
    // CHECK-SAME:      -> tensor<8x512xf16>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[SUM_INITIAL]], [[EXP]])
    // CHECK-SAME:      -> tensor<8x512xf16>
    // CHECK:       [[NEW_MAX_3D:%.+]] = IE.Unsqueeze([[NEW_MAX]])
    // CHECK-SAME:      -> tensor<8x512x1xf16>
    // CHECK:       [[SUBTRACT_VALUES:%.+]] = IE.Subtract([[ADD_ATTENTION_MASK]], [[NEW_MAX_3D]])
    // CHECK-SAME:      -> tensor<8x512x512xf16>
    // CHECK:       [[EXP_VALUES:%.+]] = IE.Exp([[SUBTRACT_VALUES]])
    // CHECK-SAME:      -> tensor<8x512x512xf16>
    // CHECK:       [[REDUCE_SUM:%.+]] = IE.ReduceSum([[EXP_VALUES]])
    // CHECK-SAME:      -> tensor<8x512xf16>
    // CHECK:       [[NEW_SUM:%.+]] = IE.Add([[MULTIPLY]], [[REDUCE_SUM]])
    // CHECK-SAME:      -> tensor<8x512xf16>

    // CHECK:       [[EXP_3D:%.+]] = IE.Unsqueeze([[EXP]])
    // CHECK-SAME:      -> tensor<8x512x1xf16>
    // CHECK:       [[PARTIAL_OUT:%.+]] = IE.Multiply([[OUT_INITIAL]], [[EXP_3D]])
    // CHECK-SAME:      -> tensor<8x512x64xf16>
    // CHECK:       [[V_MATMUL:%.+]] = IE.MatMul([[EXP_VALUES]], [[VALUE]])
    // CHECK-SAME:      -> tensor<8x512x64xf16>
    // CHECK:       [[NEW_OUT:%.+]] = IE.Add([[PARTIAL_OUT]], [[V_MATMUL]])
    // CHECK-SAME:      -> tensor<8x512x64xf16>

    // CHECK:       [[NEW_SUM_3D:%.+]] = IE.Unsqueeze([[NEW_SUM]])
    // CHECK-SAME:      -> tensor<8x512x1xf16>
    // CHECK:       [[RESULT:%.+]] = IE.Divide([[NEW_OUT]], [[NEW_SUM_3D]])
    // CHECK-SAME:      -> tensor<8x512x64xf16>

    // CHECK:       return [[RESULT]]
}

