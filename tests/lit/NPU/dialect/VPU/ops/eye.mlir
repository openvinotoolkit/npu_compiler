//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#C = affine_map<(d0) -> (d0)>
#NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL:  func.func @InferTypeDifferentInputOutputRanks
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1xsi32, {order = #C}>)
func.func @InferTypeDifferentInputOutputRanks(%input: tensor<1xsi32, {order = #C}>) -> tensor<128x128xf16, {order = #NC}> {
    // The operation has different ranks for the input and output and the tensors have encoding to represent the order
    // The purpose of the test is to validate that the inferReturnTypes is able to correctly infer the result type (i.e. should not propagate the input order)
    %eye = VPU.Eye(%input) {batch_shape_value = [0], num_columns_value = 128 : i64, num_rows_value = 128 : i64, outputType = f16}
        : tensor<1xsi32, {order = #C}>
        -> tensor<128x128xf16, {order = #NC}>
    return %eye : tensor<128x128xf16, {order = #NC}>

    // CHECK:       [[EYE:%.+]] = VPU.Eye([[INPUT]])
    // CHECK-SAME:    -> tensor<128x128xf16, {order = #NC}>
    // CHECK:       return [[EYE]]
}
