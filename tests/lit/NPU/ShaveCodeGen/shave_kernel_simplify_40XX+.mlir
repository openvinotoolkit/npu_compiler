//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --shave-kernel-simplify %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010


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

// -----

func.func @SwapExtractSliceOfHighPad(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x16xf16> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %padded = tensor.pad %arg1 low[0, 0, 0, 0] high[0, 0, 0, 8] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      tensor.yield %cst : f16
    } : tensor<1x1x1x1000xf16> to tensor<1x1x1x1008xf16>
    %extracted_slice = tensor.extract_slice %padded[0, 0, 0, 992] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x1008xf16> to tensor<1x1x1x16xf16>
    IE.CGCYield %extracted_slice : tensor<1x1x1x16xf16>
  } -> tensor<1x1x1x16xf16>
  return %0 : tensor<1x1x1x16xf16>

// CHECK:    func.func @SwapExtractSliceOfHighPad([[ARG0:%.+]]: tensor<1x1x1x1000xf16>)
// CHECK:         [[CAP:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 992] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<1x1x1x1000xf16> to tensor<1x1x1x8xf16>
// CHECK-NEXT:      [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, 0, 0] high[0, 0, 0, 8] {
// CHECK-NEXT:      ^bb0({{%.+}}: index, {{%.+}}: index, {{%.+}}: index, {{%.+}}: index):
// CHECK-NEXT:        tensor.yield [[CST]] : f16
// CHECK-NEXT:      } : tensor<1x1x1x8xf16> to tensor<1x1x1x16xf16>
// CHECK-NEXT:      IE.CGCYield [[PAD]] : tensor<1x1x1x16xf16>
// CHECK-NEXT: } -> tensor<1x1x1x16xf16>
// CHECK-NEXT:     return [[CAP]] : tensor<1x1x1x16xf16>
}

// -----

func.func @SwapExtractSliceOfLowPad(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x16xf16> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %padded = tensor.pad %arg1 low[0, 0, 0, 8] high[0, 0, 0, 0] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      tensor.yield %cst : f16
    } : tensor<1x1x1x1000xf16> to tensor<1x1x1x1008xf16>
    %extracted_slice = tensor.extract_slice %padded[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x1008xf16> to tensor<1x1x1x16xf16>
    %empty = tensor.empty() : tensor<1x1x1x16xf16>
    %abs = linalg.abs ins(%extracted_slice : tensor<1x1x1x16xf16>) outs(%empty : tensor<1x1x1x16xf16>) -> tensor<1x1x1x16xf16>
    IE.CGCYield %abs : tensor<1x1x1x16xf16>
  } -> tensor<1x1x1x16xf16>
  return %0 : tensor<1x1x1x16xf16>

// CHECK:    func.func @SwapExtractSliceOfLowPad([[ARG0:%.+]]: tensor<1x1x1x1000xf16>)
// CHECK:         [[CAP:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 1, 1, 8] [1, 1, 1, 1] : tensor<1x1x1x1000xf16> to tensor<1x1x1x8xf16>
// CHECK-NEXT:      [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, 0, 8] high[0, 0, 0, 0] {
// CHECK-NEXT:      ^bb0({{%.+}}: index, {{%.+}}: index, {{%.+}}: index, {{%.+}}: index):
// CHECK-NEXT:        tensor.yield [[CST]] : f16
// CHECK-NEXT:      } : tensor<1x1x1x8xf16> to tensor<1x1x1x16xf16>
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x16xf16>
// CHECK-NEXT:      [[ABS:%.+]] = linalg.abs ins([[PAD]] : tensor<1x1x1x16xf16>) outs([[EMPTY]] : tensor<1x1x1x16xf16>)
// CHECK-NEXT:      IE.CGCYield [[ABS]] : tensor<1x1x1x16xf16>
// CHECK-NEXT: } -> tensor<1x1x1x16xf16>
// CHECK-NEXT:     return [[CAP]] : tensor<1x1x1x16xf16>
}

// -----

func.func @ExtractSliceOfPadInsideSource(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x16xf16> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %padded = tensor.pad %arg1 low[0, 0, 0, 0] high[0, 0, 0, 8] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      tensor.yield %cst : f16
    } : tensor<1x1x1x1000xf16> to tensor<1x1x1x1008xf16>
    %extracted_slice = tensor.extract_slice %padded[0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x1008xf16> to tensor<1x1x1x16xf16>
    IE.CGCYield %extracted_slice : tensor<1x1x1x16xf16>
  } -> tensor<1x1x1x16xf16>
  return %0 : tensor<1x1x1x16xf16>

// CHECK:    func.func @ExtractSliceOfPadInsideSource([[ARG0:%.+]]: tensor<1x1x1x1000xf16>)
// CHECK:         [[CAP:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : tensor<1x1x1x1000xf16> to tensor<1x1x1x16xf16>
// CHECK-NEXT:      IE.CGCYield [[SLICE]] : tensor<1x1x1x16xf16>
// CHECK-NEXT: } -> tensor<1x1x1x16xf16>
// CHECK-NEXT:     return [[CAP]] : tensor<1x1x1x16xf16>
// CHECK-NOT:  tensor.pad
}

// -----

func.func @ExtractSliceOfPadInsidePadding(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x4xf16> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %padded = tensor.pad %arg1 low[0, 0, 0, 0] high[0, 0, 0, 8] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      tensor.yield %cst : f16
    } : tensor<1x1x1x1000xf16> to tensor<1x1x1x1008xf16>
    %extracted_slice = tensor.extract_slice %padded[0, 0, 0, 1002] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<1x1x1x1008xf16> to tensor<1x1x1x4xf16>
    IE.CGCYield %extracted_slice : tensor<1x1x1x4xf16>
  } -> tensor<1x1x1x4xf16>
  return %0 : tensor<1x1x1x4xf16>

// CHECK:    func.func @ExtractSliceOfPadInsidePadding([[ARG0:%.+]]: tensor<1x1x1x1000xf16>)
// CHECK:         [[CAP:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as {{%.+}}: tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[GEN:%.+]] = tensor.generate {
// CHECK-NEXT:      ^bb0({{%.+}}: index, {{%.+}}: index, {{%.+}}: index, {{%.+}}: index):
// CHECK-NEXT:        tensor.yield [[CST]] : f16
// CHECK-NEXT:      } : tensor<1x1x1x4xf16>
// CHECK-NEXT:      IE.CGCYield [[GEN]] : tensor<1x1x1x4xf16>
// CHECK-NEXT: } -> tensor<1x1x1x4xf16>
// CHECK-NEXT:     return [[CAP]] : tensor<1x1x1x4xf16>
// CHECK-NOT:  scf.if
}

// -----

// A rank-reducing extract_slice of a tensor.pad is left alone.

func.func @RankReducingExtractSliceOfPadIsNotSwapped(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1000xf16> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %padded = tensor.pad %arg1 low[0, 0, 0, 0] high[0, 0, 0, 8] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      tensor.yield %cst : f16
    } : tensor<1x1x1x1000xf16> to tensor<1x1x1x1008xf16>
    %extracted_slice = tensor.extract_slice %padded[0, 0, 0, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : tensor<1x1x1x1008xf16> to tensor<1000xf16>
    IE.CGCYield %extracted_slice : tensor<1000xf16>
  } -> tensor<1000xf16>
  return %0 : tensor<1000xf16>

// CHECK:    func.func @RankReducingExtractSliceOfPadIsNotSwapped([[ARG0:%.+]]: tensor<1x1x1x1000xf16>)
// CHECK:         [[CAP:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[PAD:%.+]] = tensor.pad [[ARG1]] low[0, 0, 0, 0] high[0, 0, 0, 8] {
// CHECK-NEXT:      ^bb0({{%.+}}: index, {{%.+}}: index, {{%.+}}: index, {{%.+}}: index):
// CHECK-NEXT:        tensor.yield [[CST]] : f16
// CHECK-NEXT:      } : tensor<1x1x1x1000xf16> to tensor<1x1x1x1008xf16>
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[PAD]][0, 0, 0, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : tensor<1x1x1x1008xf16> to tensor<1000xf16>
// CHECK-NEXT:      IE.CGCYield [[SLICE]] : tensor<1000xf16>
// CHECK-NEXT: } -> tensor<1000xf16>
// CHECK-NEXT:     return [[CAP]] : tensor<1000xf16>
}
