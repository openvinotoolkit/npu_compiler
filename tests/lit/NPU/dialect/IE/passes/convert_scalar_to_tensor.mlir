//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-scalar-to-tensor %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @Gather
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<18x8x72x64xf16>
func.func @Gather(%arg0: tensor<18x8x72x64xf16>) -> tensor<8x72x64xf16> {
    %cst = const.Declare tensor<si32> = dense<1> : tensor<si32>

    %0 = IE.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64}
            : tensor<18x8x72x64xf16>, tensor<si32> -> tensor<8x72x64xf16>

    return %0 : tensor<8x72x64xf16>

    // CHECK-DAG:       [[VAL0:%.+]] = const.Declare tensor<1xsi32> = dense<1> : tensor<si32>, [#const.Reshape<[1]>]
    // CHECK:       [[VAL1:%.+]] = IE.Gather([[ARG_0]], [[VAL0]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<18x8x72x64xf16>, tensor<1xsi32> -> tensor<1x8x72x64xf16>
    // CHECK:       [[VAL2:%.+]] = IE.Reshape([[VAL1]]) {shape_value = [8, 72, 64]} : tensor<1x8x72x64xf16> -> tensor<8x72x64xf16>
    // CHECK:       return [[VAL2]]
}

// CHECK-LABEL: @TopK
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<6x12x10x24xf16>
func.func @TopK(%arg0: tensor<6x12x10x24xf16>) -> (tensor<6x3x10x24xf16>, tensor<6x3x10x24xi64>) {
    %cst = const.Declare tensor<si32> = dense<3> : tensor<si32>

    %0:2 = IE.TopK(%arg0, %cst) {axis = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_VALUES>, element_type = i64}
            : tensor<6x12x10x24xf16>, tensor<si32> -> tensor<6x3x10x24xf16>, tensor<6x3x10x24xi64>

    return %0#0, %0#1 : tensor<6x3x10x24xf16>, tensor<6x3x10x24xi64>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1xsi32> = dense<3> : tensor<si32>, [#const.Reshape<[1]>]
    // CHECK:       [[VAL1:%.+]], [[VAL2:%.+]] = IE.TopK([[ARG_0]], [[CST0]])
    // CHECK-SAME:      {axis = 1 : i64, element_type = i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_VALUES>}
    // CHECK-SAME:      : tensor<6x12x10x24xf16>, tensor<1xsi32> -> tensor<6x3x10x24xf16>, tensor<6x3x10x24xi64>
    // CHECK:       return [[VAL1]], [[VAL2]]
}

// CHECK-LABEL: @Multiply
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<f16>, [[ARG_1:%[^:]+]]: tensor<1x16x32xf16>
func.func @Multiply(%arg0: tensor<f16>, %arg1: tensor<1x16x32xf16>) -> tensor<1x16x32xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<f16>, tensor<1x16x32xf16> -> tensor<1x16x32xf16>

    return %0 : tensor<1x16x32xf16>

    // CHECK:       [[VAL0:%.+]] = IE.Reshape([[ARG_0]]) {shape_value = [1]} : tensor<f16> -> tensor<1xf16>
    // CHECK:       [[VAL1:%.+]] = IE.Multiply([[VAL0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<1x16x32xf16> -> tensor<1x16x32xf16>
    // CHECK:       return [[VAL1]]
}

// CHECK-LABEL: @AddResultRank0
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<f16>, [[ARG_1:%[^:]+]]: tensor<f16>
func.func @AddResultRank0(%arg0: tensor<f16>, %arg1: tensor<f16>) -> tensor<f16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<f16>, tensor<f16> -> tensor<f16>

    return %0 : tensor<f16>

    // CHECK:       [[VAL1:%.+]] = IE.Reshape([[ARG_1]]) {shape_value = [1]} : tensor<f16> -> tensor<1xf16>
    // CHECK:       [[VAL0:%.+]] = IE.Reshape([[ARG_0]]) {shape_value = [1]} : tensor<f16> -> tensor<1xf16>
    // CHECK:       [[VAL2:%.+]] = IE.Add([[VAL0]], [[VAL1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<1xf16> -> tensor<1xf16>
    // CHECK:       [[VAL3:%.+]] = IE.Reshape([[VAL2]]) {shape_value = []} : tensor<1xf16> -> tensor<f16>
    // CHECK:       return [[VAL3]]
}
