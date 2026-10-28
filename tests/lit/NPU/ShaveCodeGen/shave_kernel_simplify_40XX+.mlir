//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --shave-kernel-simplify %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010


func.func @PaddedInputSoftmax(%arg0: tensor<1x1x1x1008xf16>) -> tensor<1x1x1x1000xf16> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1008xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %extracted_slice = tensor.extract_slice %arg1[0, 0, 0, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : tensor<1x1x1x1008xf16> to tensor<1000xf16>
    %expanded = tensor.expand_shape %extracted_slice [[0, 1, 2, 3]] output_shape [1, 1, 1, 1000] : tensor<1000xf16> into tensor<1x1x1x1000xf16>
    %1 = tensor.empty() : tensor<1x1x1x1008xf16>
    %2 = linalg.fill ins(%cst : f16) outs(%1 : tensor<1x1x1x1008xf16>) -> tensor<1x1x1x1008xf16>
    %extracted_slice_0 = tensor.extract_slice %2[0, 0, 0, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : tensor<1x1x1x1008xf16> to tensor<1000xf16>
    %expanded_1 = tensor.expand_shape %extracted_slice_0 [[0, 1, 2, 3]] output_shape [1, 1, 1, 1000] : tensor<1000xf16> into tensor<1x1x1x1000xf16>
    %3 = linalg.softmax dimension(3) ins(%expanded : tensor<1x1x1x1000xf16>) outs(%expanded_1 : tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16>
    IE.CGCYield %3 : tensor<1x1x1x1000xf16>
  } -> tensor<1x1x1x1000xf16>
  return %0 : tensor<1x1x1x1000xf16>

// CHECK:    func.func @PaddedInputSoftmax([[ARG0:%.+]]: tensor<1x1x1x1008xf16>)
// CHECK:         [[CAP:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[ARG1:%.+]]: tensor<1x1x1x1008xf16>) {
// CHECK-DAG:       [[SLICE:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : tensor<1x1x1x1008xf16> to tensor<1000xf16>
// CHECK-DAG:       [[RESHAPED_SLICE:%.+]] = tensor.expand_shape [[SLICE]] {{.+}} : tensor<1000xf16> into tensor<1x1x1x1000xf16>
// CHECK-DAG:       [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-DAG:       [[OUT_INIT:%.+]] = linalg.fill ins({{%.+}} : f16) outs([[EMPTY]] : tensor<1x1x1x1000xf16>)
// CHECK-NEXT:      [[SM:%.+]] = linalg.softmax dimension(3) ins([[RESHAPED_SLICE]] : tensor<1x1x1x1000xf16>) outs([[OUT_INIT]] : tensor<1x1x1x1000xf16>)
// CHECK-NEXT:      IE.CGCYield [[SM]] : tensor<1x1x1x1000xf16>
// CHECK-NEXT: } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:     return [[CAP]] : tensor<1x1x1x1000xf16>
}
