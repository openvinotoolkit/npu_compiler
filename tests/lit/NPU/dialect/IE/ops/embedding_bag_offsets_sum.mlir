//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertConstToAttr
func.func @ConvertConstToAttr(%arg0: tensor<5x6xf16>) -> tensor<2x6xf16> {
    %cst = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    %cst_0 = const.Declare tensor<2xsi32> = dense<[0, 2]> : tensor<2xsi32>
    %cst_1 = const.Declare tensor<1xsi32> = dense<0> : tensor<si32>, [#const.Reshape<[1]>]
    %cst_2 = const.Declare tensor<5xf16> = dense<[1.000000e+00, 4.753910e+00, 9.976560e+00, 7.484380e+00, 1.000000e+01]> : tensor<5xf16>
    %0 = IE.EmbeddingBagOffsetsSum(%arg0, %cst, %cst_0, %cst_1, %cst_2) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : tensor<5x6xf16>, tensor<5xsi32>, tensor<2xsi32>, tensor<1xsi32>, tensor<5xf16> -> tensor<2x6xf16>
    return %0 : tensor<2x6xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    // CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<2xsi32> = dense<[0, 2]> : tensor<2xsi32>
    // CHECK-DAG: [[CST1:%.+]] = dense<[1.000000e+00, 4.753910e+00, 9.976560e+00, 7.484380e+00, 1.000000e+01]> : tensor<5xf16>
    // CHECK: [[VAR0:%.+]] = IE.EmbeddingBagOffsetsSum(%arg0, %cst, %cst_0, %cst_1) {default_index_value = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>} : tensor<5x6xf16>, tensor<5xsi32>, tensor<2xsi32>, tensor<5xf16> -> tensor<2x6xf16>
    // CHECK: return [[VAR0]] : tensor<2x6xf16>
}
