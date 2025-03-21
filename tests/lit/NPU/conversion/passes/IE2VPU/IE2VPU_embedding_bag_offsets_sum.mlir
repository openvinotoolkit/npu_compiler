//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-layers-to-VPU %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @EmbeddingBagOffsetsSumWithWeights
func.func @EmbeddingBagOffsetsSumWithWeights(%arg0: tensor<5x6xf16>) -> tensor<2x6xf16> {
    // CHECK:  ([[ARG0:[^:]+]]: tensor<5x6xf16>)
    %cst = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    %cst_0 = const.Declare tensor<2xsi32> = dense<[0, 2]> : tensor<2xsi32>
    %cst_1 = const.Declare tensor<5xf16> = dense<[1.000000e+00, 4.753910e+00, 9.976560e+00, 7.484380e+00, 1.000000e+01]> : tensor<5xf16>
    %0 = IE.EmbeddingBagOffsetsSum(%arg0, %cst, %cst_0, %cst_1) {default_index_value = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>} : tensor<5x6xf16>, tensor<5xsi32>, tensor<2xsi32>, tensor<5xf16> -> tensor<2x6xf16>
    return %0 : tensor<2x6xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    // CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<2xsi32> = dense<[0, 2]> : tensor<2xsi32>
    // CHECK-DAG: [[CST1:%.+]] = const.Declare tensor<5xf16> = dense<[1.000000e+00, 4.753910e+00, 9.976560e+00, 7.484380e+00, 1.000000e+01]> : tensor<5xf16>
    // CHECK: [[VAR0:%.+]] = VPU.EmbeddingBagOffsetsSum([[ARG0]], [[CST]], [[CST0]], [[CST1]]) {default_index_value = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<5x6xf16>, tensor<5xsi32>, tensor<2xsi32>, tensor<5xf16> -> tensor<2x6xf16>
    // CHECK: return [[VAR0]] : tensor<2x6xf16>
}

// -----

// CHECK-LABEL: @EmbeddingBagOffsetsSumNoWeights
func.func @EmbeddingBagOffsetsSumNoWeights(%arg0: tensor<5x6xf16>) -> tensor<2x6xf16> {
    // CHECK:  ([[ARG0:[^:]+]]: tensor<5x6xf16>)
    %cst = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    %cst_0 = const.Declare tensor<2xsi32> = dense<[0, 2]> : tensor<2xsi32>
    %0 = IE.EmbeddingBagOffsetsSum(%arg0, %cst, %cst_0) {default_index_value = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>} : tensor<5x6xf16>, tensor<5xsi32>, tensor<2xsi32> -> tensor<2x6xf16>
    return %0 : tensor<2x6xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    // CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<2xsi32> = dense<[0, 2]> : tensor<2xsi32>
    // CHECK: [[VAR0:%.+]] = VPU.EmbeddingBagOffsetsSum([[ARG0]], [[CST]], [[CST0]]) {default_index_value = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0>} : tensor<5x6xf16>, tensor<5xsi32>, tensor<2xsi32> -> tensor<2x6xf16>
    // CHECK: return [[VAR0]] : tensor<2x6xf16>
}
