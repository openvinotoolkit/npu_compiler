//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --debatcher %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @HeterogenousInputsBatchInConstant
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<3x320xf32>,
// CHECK-SAME:     [[ARG1:%.+]]: tensor<1xf64>,
// CHECK-SAME:     [[ARG2:%.+]]: tensor<1x160xf32>)
func.func @HeterogenousInputsBatchInConstant(%arg0: tensor<3x320xf32>, %arg1: tensor<1xf64>, %arg2: tensor<1x160xf32>) -> tensor<3x320xf32> {
    %batch_cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi32>
    %1 = IE.Broadcast(%arg1, %batch_cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1xf64>, tensor<1xsi32> -> tensor<3xf64>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1]], shape_value = [3, 1]} : tensor<3xf64> -> tensor<3x1xf64>
    %3 = IE.Convert(%2) {dstElemType = f32} : tensor<3x1xf64> -> tensor<3x1xf32>
    %4 = IE.Multiply(%3, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x1xf32>, tensor<1x160xf32> -> tensor<3x160xf32>
    %5 = IE.Sin(%4) : tensor<3x160xf32> -> tensor<3x160xf32>
    %6 = IE.Cos(%5) : tensor<3x160xf32> -> tensor<3x160xf32>
    %7 = IE.Concat(%5, %6) {static_offsets = [[0, 0], [0, 160]]} : tensor<3x160xf32>, tensor<3x160xf32> -> tensor<3x320xf32>
    %RET = IE.Add(%arg0, %7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x320xf32>, tensor<3x320xf32> -> tensor<3x320xf32>
    return %RET : tensor<3x320xf32>

    // CHECK-DAG: [[VAL0:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : tensor<3x320xf32> to tensor<1x320xf32>
    // CHECK: [[CONST:%.+]] = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    // CHECK: [[VAL1:%.+]] = IE.Broadcast([[ARG1]], [[CONST]]) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1xf64>, tensor<1xsi32> -> tensor<1xf64>
    // CHECK: [[VAL2:%.+]] = IE.AffineReshape([[VAL1]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0, 1]], shape_value = [1, 1]} :
    // CHECK-SAME:    tensor<1xf64> -> tensor<1x1xf64>
    // CHECK: [[VAL3:%.+]] = IE.Convert([[VAL2]]) {dstElemType = f32} : tensor<1x1xf64> -> tensor<1x1xf32>
    // CHECK: [[VAL4:%.+]] = IE.Multiply([[VAL3]], [[ARG2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xf32>, tensor<1x160xf32> -> tensor<1x160xf32>
    // CHECK: [[VAL5:%.+]] = IE.Sin([[VAL4]]) : tensor<1x160xf32> -> tensor<1x160xf32>
    // CHECK: [[VAL6:%.+]] = IE.Cos([[VAL5]]) : tensor<1x160xf32> -> tensor<1x160xf32>
    // CHECK: [[VAL7:%.+]] = IE.Concat([[VAL5]], [[VAL6]])
    // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0], [0, 160]]} :
    // CHECK-SAME:   tensor<1x160xf32>, tensor<1x160xf32> -> tensor<1x320xf32>
    // CHECK: [[VAL8:%.+]] = IE.Add([[VAL0]], [[VAL7]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x320xf32>, tensor<1x320xf32> -> tensor<1x320xf32>
    // CHECK: [[RET:%.+]] = builtin.unrealized_conversion_cast [[VAL8]] : tensor<1x320xf32> to tensor<3x320xf32>
    // CHECK: return [[RET]] : tensor<3x320xf32>
}


// -----


// CHECK-LABEL: @HeterogenousInputsBatchInConstantBroadcastNonArgConst
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<3x320xf32>,
// CHECK-SAME:     [[ARG1:%.+]]: tensor<1xf64>,
// CHECK-SAME:     [[ARG2:%.+]]: tensor<1x160xf32>)
func.func @HeterogenousInputsBatchInConstantBroadcastNonArgConst(%arg0: tensor<3x320xf32>, %arg1: tensor<1xf64>, %arg2: tensor<1x160xf32>) -> tensor<3x320xf32> {
    %batch_cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi32>
    %NonArg1 = const.Declare tensor<3xf64> = dense<2.000> : tensor<3xf64>
    %1 = IE.Broadcast(%NonArg1, %batch_cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<3xf64>, tensor<1xsi32> -> tensor<3xf64>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1]], shape_value = [3, 1]} : tensor<3xf64> -> tensor<3x1xf64>
    %3 = IE.Convert(%2) {dstElemType = f32} : tensor<3x1xf64> -> tensor<3x1xf32>
    %4 = IE.Multiply(%3, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x1xf32>, tensor<1x160xf32> -> tensor<3x160xf32>
    %5 = IE.Sin(%4) : tensor<3x160xf32> -> tensor<3x160xf32>
    %6 = IE.Cos(%5) : tensor<3x160xf32> -> tensor<3x160xf32>
    %7 = IE.Concat(%5, %6) {static_offsets = [[0, 0], [0, 160]]} : tensor<3x160xf32>, tensor<3x160xf32> -> tensor<3x320xf32>
    %RET = IE.Add(%arg0, %7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x320xf32>, tensor<3x320xf32> -> tensor<3x320xf32>
    return %RET : tensor<3x320xf32>

    // CHECK-DAG: [[VAL0:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : tensor<3x320xf32> to tensor<1x320xf32>
    // CHECK: [[CONST1:%.+]] = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    // CHECK: [[CONST1_ORIG:%.+]] = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi32>
    // CHECK: [[CONST2:%.+]] = const.Declare tensor<1xf64> = dense<2.000000e+00> : tensor<1xf64>
    // CHECK: [[CONST2_ORIG:%.+]] = const.Declare tensor<3xf64> = dense<2.000000e+00> : tensor<3xf64>
    // CHECK: [[VAL1:%.+]] = IE.Broadcast([[CONST2]], [[CONST1]]) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1xf64>, tensor<1xsi32> -> tensor<1xf64>
    // CHECK: [[VAL2:%.+]] = IE.AffineReshape([[VAL1]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0, 1]], shape_value = [1, 1]} :
    // CHECK-SAME:    tensor<1xf64> -> tensor<1x1xf64>
    // CHECK: [[VAL3:%.+]] = IE.Convert([[VAL2]]) {dstElemType = f32} : tensor<1x1xf64> -> tensor<1x1xf32>
    // CHECK: [[VAL4:%.+]] = IE.Multiply([[VAL3]], [[ARG2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xf32>, tensor<1x160xf32> -> tensor<1x160xf32>
    // CHECK: [[VAL5:%.+]] = IE.Sin([[VAL4]]) : tensor<1x160xf32> -> tensor<1x160xf32>
    // CHECK: [[VAL6:%.+]] = IE.Cos([[VAL5]]) : tensor<1x160xf32> -> tensor<1x160xf32>
    // CHECK: [[VAL7:%.+]] = IE.Concat([[VAL5]], [[VAL6]])
    // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0], [0, 160]]} :
    // CHECK-SAME:   tensor<1x160xf32>, tensor<1x160xf32> -> tensor<1x320xf32>
    // CHECK: [[VAL8:%.+]] = IE.Add([[VAL0]], [[VAL7]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x320xf32>, tensor<1x320xf32> -> tensor<1x320xf32>
    // CHECK: [[RET:%.+]] = builtin.unrealized_conversion_cast [[VAL8]] : tensor<1x320xf32> to tensor<3x320xf32>
    // CHECK: return [[RET]] : tensor<3x320xf32>
}
