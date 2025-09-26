//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @ReverseSequenceConvertU8ToFP16
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x128xui8>, [[ARG1:%.+]]: tensor<1xsi32>)
func.func @ReverseSequenceConvertU8ToFP16(%arg0: tensor<1x128xui8>, %arg1: tensor<1xsi32>) -> tensor<1x128xui8> {
    %0 = IE.ReverseSequence(%arg0, %arg1) {batch_axis = 0 : i64, seq_axis = 1 : i64} : tensor<1x128xui8>, tensor<1xsi32> -> tensor<1x128xui8>
    return %0 : tensor<1x128xui8>

    // CHECK-DAG: [[CONVERT_IN:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16}
    // CHECK-DAG: [[REVERSE_SEQ:%.+]] = IE.ReverseSequence([[CONVERT_IN]], [[ARG1]]) {batch_axis = 0 : i64, seq_axis = 1 : i64}
    // CHECK-DAG: [[CONVERT_OUT:%.+]] = IE.Convert([[REVERSE_SEQ]]) {dstElemType = ui8}
    // CHECK: return [[CONVERT_OUT]]
}

// CHECK-LABEL: @ReverseSequenceConvertI8ToFP16
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x128xi8>, [[ARG1:%.+]]: tensor<1xsi32>)
func.func @ReverseSequenceConvertI8ToFP16(%arg0: tensor<1x128xi8>, %arg1: tensor<1xsi32>) -> tensor<1x128xi8> {
    %0 = IE.ReverseSequence(%arg0, %arg1) {batch_axis = 0 : i64, seq_axis = 1 : i64} : tensor<1x128xi8>, tensor<1xsi32> -> tensor<1x128xi8>
    return %0 : tensor<1x128xi8>

    // CHECK-DAG: [[CONVERT_IN:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16}
    // CHECK-DAG: [[REVERSE_SEQ:%.+]] = IE.ReverseSequence([[CONVERT_IN]], [[ARG1]]) {batch_axis = 0 : i64, seq_axis = 1 : i64}
    // CHECK-DAG: [[CONVERT_OUT:%.+]] = IE.Convert([[REVERSE_SEQ]]) {dstElemType = i8}
    // CHECK: return [[CONVERT_OUT]]
}

// CHECK-LABEL: @ReverseSequenceConvertSI8ToFP16
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x128xsi8>, [[ARG1:%.+]]: tensor<1xsi32>)
func.func @ReverseSequenceConvertSI8ToFP16(%arg0: tensor<1x128xsi8>, %arg1: tensor<1xsi32>) -> tensor<1x128xsi8> {
    %0 = IE.ReverseSequence(%arg0, %arg1) {batch_axis = 0 : i64, seq_axis = 1 : i64} : tensor<1x128xsi8>, tensor<1xsi32> -> tensor<1x128xsi8>
    return %0 : tensor<1x128xsi8>

    // CHECK-DAG: [[CONVERT_IN:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16}
    // CHECK-DAG: [[REVERSE_SEQ:%.+]] = IE.ReverseSequence([[CONVERT_IN]], [[ARG1]]) {batch_axis = 0 : i64, seq_axis = 1 : i64}
    // CHECK-DAG: [[CONVERT_OUT:%.+]] = IE.Convert([[REVERSE_SEQ]]) {dstElemType = si8}
    // CHECK: return [[CONVERT_OUT]]
}
