//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-online-sdpa="disable-incremental-sdpa-decomposition=true"  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @OnlineSdpaWithAttentionMaskNoScaleNoDivide
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<8x512x512xf16>
func.func @OnlineSdpaWithAttentionMaskNoScaleNoDivide(%arg0: tensor<8x512x64xf16>, %arg1: tensor<8x512x64xf16>, %arg2: tensor<8x512x64xf16>, %arg3: tensor<8x512x512xf16>) -> tensor<8x512x64xf16> {
    %0 = IE.OnlineSDPA(%arg0, %arg1, %arg2, %arg3) {kv_num_blocks = 1 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>} : tensor<8x512x64xf16>, tensor<8x512x64xf16>, tensor<8x512x64xf16>, tensor<8x512x512xf16> -> tensor<8x512x64xf16>
    return %0 : tensor<8x512x64xf16>

    // CHECK-DAG:   [[MAX_INITIAL:%.+]] = const.Declare tensor<8x512xf16> = dense<0xFC00> : tensor<8x512xf16>
    // CHECK-DAG:   [[SUM_INITIAL:%.+]] = const.Declare tensor<8x512xf16> = dense<0.000000e+00> : tensor<8x512xf16>
    // CHECK-DAG:   [[OUT_INITIAL:%.+]] = const.Declare tensor<8x512x64xf16> = dense<0.000000e+00> : tensor<8x512x64xf16>

    // CHECK:       [[RESULT_OUT:%[^, ]+]], [[RESULT_MAX:%[^, ]+]], [[RESULT_SUM:%[^, ]+]] =
    // CHECK-SAME:      IE.IncrementalSDPA([[QUERY]], [[KEY]], [[VALUE]], [[OUT_INITIAL]], [[MAX_INITIAL]], [[SUM_INITIAL]], [[ATTENTION_MASK]])
    // CHECK-SAME:          -> tensor<8x512x64xf16>, tensor<8x512xf16>, tensor<8x512xf16>

    // CHECK:       return [[RESULT_OUT]]
}

