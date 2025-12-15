//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @FoldReduceL1
func.func @FoldReduceL1(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceL1(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceL1
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @FoldReduceL2
func.func @FoldReduceL2(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceL2(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceL2
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @FoldReduceLogicalAnd
func.func @FoldReduceLogicalAnd(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceLogicalAnd(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceLogicalAnd
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @FoldReduceLogicalOr
func.func @FoldReduceLogicalOr(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceLogicalOr(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceLogicalOr
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @FoldReduceMax
func.func @FoldReduceMax(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceMax(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceMax
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @FoldReduceMean
func.func @FoldReduceMean(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceMean
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @DoNotFoldPaddedReduceMean
func.func @DoNotFoldPaddedReduceMean(%arg0: tensor<1x16x4x2xf16>) -> tensor<1x16x4x2xf16> {
    %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims, input_padding = [0, 4, 0, 0], output_padding = [0, 15, 0, 0]} : tensor<1x16x4x2xf16> -> tensor<1x16x4x2xf16>
    return %0 : tensor<1x16x4x2xf16>

    // CHECK:   [[REDUCE:%.+]] = IE.ReduceMean
    // CHECK:   return [[REDUCE]]
}

// -----

// CHECK-LABEL: @FoldReduceMin
func.func @FoldReduceMin(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceMin(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceMin
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @FoldReduceProd
func.func @FoldReduceProd(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceProd(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceProd
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @FoldReduceSum
func.func @FoldReduceSum(%arg0: tensor<1x1x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x1x4x2xf16> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK-NOT:   IE.ReduceSum
    // CHECK:       return %arg0
}

// -----

// CHECK-LABEL: @DoNotFoldPaddedReduceSum
func.func @DoNotFoldPaddedReduceSum(%arg0: tensor<1x16x4x2xf16>) -> tensor<1x16x4x2xf16> {
    %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims, input_padding = [0, 4, 0, 0], output_padding = [0, 15, 0, 0]} : tensor<1x16x4x2xf16> -> tensor<1x16x4x2xf16>
    return %0 : tensor<1x16x4x2xf16>

    // CHECK:   [[REDUCE:%.+]] = IE.ReduceSum
    // CHECK:   return [[REDUCE]]
}

// -----

// CHECK-LABEL: @ConvertToAttrReduceL1
func.func @ConvertToAttrReduceL1(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceL1(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_L1:%.*]] = IE.ReduceL1(%arg0) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_L1]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceL2
func.func @ConvertToAttrReduceL2(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceL2(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_L2:%.*]] = IE.ReduceL2(%arg0) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_L2]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceLogicalAnd
func.func @ConvertToAttrReduceLogicalAnd(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceLogicalAnd(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_LOGICAL_AND:%.*]] = IE.ReduceLogicalAnd(%arg0) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_LOGICAL_AND]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceLogicalOr
func.func @ConvertToAttrReduceLogicalOr(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceLogicalOr(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_LOGICAL_OR:%.*]] = IE.ReduceLogicalOr(%arg0) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_LOGICAL_OR]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceMax
func.func @ConvertToAttrReduceMax(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceMax(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_MAX:%.*]] = IE.ReduceMax(%arg0) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_MAX]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceMean
func.func @ConvertToAttrReduceMean(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceMean(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_MEAN:%.*]] = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_MEAN]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceMin
func.func @ConvertToAttrReduceMin(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceMin(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_MIN:%.*]] = IE.ReduceMin(%arg0) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_MIN]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceProd
func.func @ConvertToAttrReduceProd(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceProd(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_PROD:%.*]] = IE.ReduceProd(%arg0) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_PROD]] : tensor<1x1x4x2xf16>

}

// -----

// CHECK-LABEL: @ConvertToAttrReduceSum
func.func @ConvertToAttrReduceSum(%arg0: tensor<1x3x4x2xf16>) -> tensor<1x1x4x2xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ReduceSum(%arg0, %cst) {keep_dims} : tensor<1x3x4x2xf16>, tensor<1xsi32> -> tensor<1x1x4x2xf16>
    return %0 : tensor<1x1x4x2xf16>

    // CHECK: [[REDUCE_SUM:%.*]] = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x3x4x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK: return [[REDUCE_SUM]] : tensor<1x1x4x2xf16>

}
