//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ConvertConstToAttr
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<5x6x4xui8>)
func.func @ConvertConstToAttr(%arg0: tensor<5x6x4xui8>) -> tensor<7x6x4xui8> {
    %cst = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    %cst_0 = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 3, 4]> : tensor<5xsi32>
    %cst_1 = const.Declare tensor<5xui8> = dense<[1, 5, 10, 8, 10]> : tensor<5xui8>
    %0 = IE.EmbeddingSegmentsSum(%arg0, %cst, %cst_0, %cst_1) {default_index_value = 0 : i32, num_segments_value = 7 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<5x6x4xui8>, tensor<5xsi32>, tensor<5xsi32>, tensor<5xui8> -> tensor<7x6x4xui8>
    return %0 : tensor<7x6x4xui8>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    // CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 3, 4]> : tensor<5xsi32>
    // CHECK-DAG: [[CST1:%.+]] = const.Declare tensor<5xui8> = dense<[1, 5, 10, 8, 10]> : tensor<5xui8>
    // CHECK: [[VAR0:%.+]] = IE.EmbeddingSegmentsSum([[ARG_0]], [[CST]], [[CST0]], [[CST1]]) {default_index_value = 0 : i32, num_segments_value = 7 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<5x6x4xui8>, tensor<5xsi32>, tensor<5xsi32>, tensor<5xui8> -> tensor<7x6x4xui8>
    // CHECK: return [[VAR0]] : tensor<7x6x4xui8>
}
