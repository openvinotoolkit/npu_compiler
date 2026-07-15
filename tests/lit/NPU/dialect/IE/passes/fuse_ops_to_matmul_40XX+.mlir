//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --fuse-ops-to-matmul="enable-grouped-matmul=true" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ConvertBroadcastMultiplyReduceSumToMatMul
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<6x256x1x8192xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<6x1x256x8192xf16>
func.func @ConvertBroadcastMultiplyReduceSumToMatMul(%arg0: tensor<6x256x1x8192xf16>, %arg1: tensor<6x1x256x8192xf16>) -> tensor<6x256x256x1xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x256x1x8192xf16>, tensor<6x1x256x8192xf16> -> tensor<6x256x256x8192xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [3], keep_dims} : tensor<6x256x256x8192xf16> -> tensor<6x256x256x1xf16>
  return %red : tensor<6x256x256x1xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK-NOT:   IE.ReduceSum
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT_A]]) {shape_value = [6, 256, 8192]}
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT_B]]) {shape_value = [6, 256, 8192]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b} : tensor<6x256x8192xf16>, tensor<6x256x8192xf16> -> tensor<6x256x256xf16>
  // CHECK:       [[OUT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [6, 256, 256, 1]}
  // CHECK:       return [[OUT]] : tensor<6x256x256x1xf16>
}

// -----

// CHECK-LABEL: @ConvertBroadcastMultiplyReduceSumToMatMulSingleAxis
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<6x256x1x128xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<6x1x64x128xf16>
func.func @ConvertBroadcastMultiplyReduceSumToMatMulSingleAxis(%arg0: tensor<6x256x1x128xf16>, %arg1: tensor<6x1x64x128xf16>) -> tensor<6x256x64x1xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x256x1x128xf16>, tensor<6x1x64x128xf16> -> tensor<6x256x64x128xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [3], keep_dims} : tensor<6x256x64x128xf16> -> tensor<6x256x64x1xf16>
  return %red : tensor<6x256x64x1xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK-NOT:   IE.ReduceSum
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT_A]]) {shape_value = [6, 256, 128]}
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT_B]]) {shape_value = [6, 64, 128]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b}
  // CHECK:       [[OUT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [6, 256, 64, 1]}
  // CHECK:       return [[OUT]]
}

// -----

// CHECK-LABEL: @NotConvertSingleAxisBroadcastMultiplyReduceSum
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x256x8192xf16>
func.func @NotConvertSingleAxisBroadcastMultiplyReduceSum(%arg0: tensor<1x256x8192xf16>, %arg1: tensor<1x1x8192xf16>) -> tensor<1x256x1xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x8192xf16>, tensor<1x1x8192xf16> -> tensor<1x256x8192xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [2], keep_dims} : tensor<1x256x8192xf16> -> tensor<1x256x1xf16>
  return %red : tensor<1x256x1xf16>

  // CHECK-NOT:   IE.MatMul
  // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT]], %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[REDUCESUM:%.+]] = IE.ReduceSum([[MULTIPLY]]) {axes_value = [2], keep_dims}
  // CHECK:       return [[REDUCESUM]] : tensor<1x256x1xf16>
}

// -----

// CHECK-LABEL: @ConvertBroadcastMultiplyReduceSumToMatMulKeepDimsFalse
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<6x256x1x8192xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<6x1x256x8192xf16>
func.func @ConvertBroadcastMultiplyReduceSumToMatMulKeepDimsFalse(%arg0: tensor<6x256x1x8192xf16>, %arg1: tensor<6x1x256x8192xf16>) -> tensor<6x256x256xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x256x1x8192xf16>, tensor<6x1x256x8192xf16> -> tensor<6x256x256x8192xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [3]} : tensor<6x256x256x8192xf16> -> tensor<6x256x256xf16>
  return %red : tensor<6x256x256xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK-NOT:   IE.ReduceSum
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT_A]]) {shape_value = [6, 256, 8192]}
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT_B]]) {shape_value = [6, 256, 8192]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b}
  // CHECK:       [[OUT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [6, 256, 256]}
  // CHECK:       return [[OUT]] : tensor<6x256x256xf16>
}

// -----

// CHECK-LABEL: @ConvertBroadcastMultiplyReduceSumToMatMulRank6
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<1x4x256x1x64x128xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<1x4x1x256x64x128xf16>
func.func @ConvertBroadcastMultiplyReduceSumToMatMulRank6(%arg0: tensor<1x4x256x1x64x128xf16>, %arg1: tensor<1x4x1x256x64x128xf16>) -> tensor<1x4x256x256x64x1xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x256x1x64x128xf16>, tensor<1x4x1x256x64x128xf16> -> tensor<1x4x256x256x64x128xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [5], keep_dims} : tensor<1x4x256x256x64x128xf16> -> tensor<1x4x256x256x64x1xf16>
  return %red : tensor<1x4x256x256x64x1xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK-NOT:   IE.ReduceSum
  // CHECK:       [[LHS_T:%.+]] = IE.Transpose([[INPUT_A]]) {{.+}} : tensor<1x4x256x1x64x128xf16> -> tensor<1x4x64x256x1x128xf16>
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[LHS_T]]) {shape_value = [256, 256, 128]}
  // CHECK:       [[RHS_T:%.+]] = IE.Transpose([[INPUT_B]]) {{.+}} : tensor<1x4x1x256x64x128xf16> -> tensor<1x4x64x256x1x128xf16>
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[RHS_T]]) {shape_value = [256, 256, 128]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b}
  // CHECK:       [[EXP:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 4, 64, 256, 256]}
  // CHECK:       [[INV:%.+]] = IE.Transpose([[EXP]]) {{.+}} : tensor<1x4x64x256x256xf16> -> tensor<1x4x256x256x64xf16>
  // CHECK:       [[OUT:%.+]] = IE.Reshape([[INV]]) {shape_value = [1, 4, 256, 256, 64, 1]}
  // CHECK:       return [[OUT]] : tensor<1x4x256x256x64x1xf16>
}

// -----

// CHECK-LABEL: @ConvertBroadcastMultiplyReduceSumNonLastAxis
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<1x4x64x256x128x1xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<1x4x64x256x1x64xf16>
func.func @ConvertBroadcastMultiplyReduceSumNonLastAxis(%arg0: tensor<1x4x64x256x128x1xf16>, %arg1: tensor<1x4x64x256x1x64xf16>) -> tensor<1x4x64x1x128x64xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x64x256x128x1xf16>, tensor<1x4x64x256x1x64xf16> -> tensor<1x4x64x256x128x64xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [3], keep_dims} : tensor<1x4x64x256x128x64xf16> -> tensor<1x4x64x1x128x64xf16>
  return %red : tensor<1x4x64x1x128x64xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK-NOT:   IE.ReduceSum
  // CHECK:       [[LHS_K:%.+]] = IE.Transpose([[INPUT_A]]) {{.+}} : tensor<1x4x64x256x128x1xf16> -> tensor<1x4x64x128x1x256xf16>
  // CHECK:       [[RHS_K:%.+]] = IE.Transpose([[INPUT_B]]) {{.+}} : tensor<1x4x64x256x1x64xf16> -> tensor<1x4x64x1x64x256xf16>
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[LHS_K]]) {shape_value = [256, 128, 256]}
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[RHS_K]]) {shape_value = [256, 64, 256]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b}
  // CHECK:       [[OUT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 4, 64, 1, 128, 64]}
  // CHECK:       return [[OUT]] : tensor<1x4x64x1x128x64xf16>
}

// -----

// CHECK-LABEL: @NotConvertMultiplyWithReduceSumUserInterleavedDims
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<4x256x1x128xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<4x1x256x128xf16>
func.func @NotConvertMultiplyWithReduceSumUserInterleavedDims(%arg0: tensor<4x256x1x128xf16>, %arg1: tensor<4x1x256x128xf16>) -> (tensor<4x256x256x1xf16>, tensor<4x256x256x128xf16>) {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x256x1x128xf16>, tensor<4x1x256x128xf16> -> tensor<4x256x256x128xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [3], keep_dims} : tensor<4x256x256x128xf16> -> tensor<4x256x256x1xf16>
  return %red, %mul : tensor<4x256x256x1xf16>, tensor<4x256x256x128xf16>

  // CHECK-NOT:   IE.MatMul
  // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_A]], [[INPUT_B]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[REDUCESUM:%.+]] = IE.ReduceSum([[MULTIPLY]]) {axes_value = [3], keep_dims}
  // CHECK:       return [[REDUCESUM]], [[MULTIPLY]]
}

// -----

// Outer product Multiply [1,16,128,1] * [1,16,1,128] -> MatMul on DPU (Qwen3.5 Mamba2 pattern)

// CHECK-LABEL: @ConvertOuterProductMultiplyToMatMul
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<1x16x128x1xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<1x16x1x128xf16>
func.func @ConvertOuterProductMultiplyToMatMul(%arg0: tensor<1x16x128x1xf16>, %arg1: tensor<1x16x1x128xf16>) -> tensor<1x16x128x128xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x128x1xf16>, tensor<1x16x1x128xf16> -> tensor<1x16x128x128xf16>
  return %mul : tensor<1x16x128x128xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT_A]]) {shape_value = [16, 128, 1]}
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT_B]]) {shape_value = [16, 128, 1]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b} : tensor<16x128x1xf16>, tensor<16x128x1xf16> -> tensor<16x128x128xf16>
  // CHECK:       [[OUT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 16, 128, 128]}
  // CHECK:       return [[OUT]] : tensor<1x16x128x128xf16>
}

// -----

// 3D outer product

// CHECK-LABEL: @ConvertOuterProductMultiplyToMatMul3D
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<8x64x1xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<8x1x64xf16>
func.func @ConvertOuterProductMultiplyToMatMul3D(%arg0: tensor<8x64x1xf16>, %arg1: tensor<8x1x64xf16>) -> tensor<8x64x64xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x64x1xf16>, tensor<8x1x64xf16> -> tensor<8x64x64xf16>
  return %mul : tensor<8x64x64xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT_A]]) {shape_value = [8, 64, 1]}
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT_B]]) {shape_value = [8, 64, 1]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b} : tensor<8x64x1xf16>, tensor<8x64x1xf16> -> tensor<8x64x64xf16>
  // CHECK:       [[OUT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [8, 64, 64]}
  // CHECK:       return [[OUT]] : tensor<8x64x64xf16>
}

// -----

// Outer product with ReduceSum user is handled by BroadcastMultiplyReduceSumToMatMulRewriter, not OuterProduct rewriter

// CHECK-LABEL: @OuterProductWithReduceSumHandledByReduceSumRewriter
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<6x256x1x8192xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<6x1x256x8192xf16>
func.func @OuterProductWithReduceSumHandledByReduceSumRewriter(%arg0: tensor<6x256x1x8192xf16>, %arg1: tensor<6x1x256x8192xf16>) -> tensor<6x256x256x1xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x256x1x8192xf16>, tensor<6x1x256x8192xf16> -> tensor<6x256x256x8192xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [3], keep_dims} : tensor<6x256x256x8192xf16> -> tensor<6x256x256x1xf16>
  return %red : tensor<6x256x256x1xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK-NOT:   IE.ReduceSum
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT_A]]) {shape_value = [6, 256, 8192]}
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT_B]]) {shape_value = [6, 256, 8192]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b}
  // CHECK:       [[OUT:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [6, 256, 256, 1]}
  // CHECK:       return [[OUT]] : tensor<6x256x256x1xf16>
}

// -----

// Do not convert when M*N*K < 4096

// CHECK-LABEL: @NotConvertOuterProductTooSmall
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<1x4x8x1xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<1x4x1x8xf16>
func.func @NotConvertOuterProductTooSmall(%arg0: tensor<1x4x8x1xf16>, %arg1: tensor<1x4x1x8xf16>) -> tensor<1x4x8x8xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8x1xf16>, tensor<1x4x1x8xf16> -> tensor<1x4x8x8xf16>
  return %mul : tensor<1x4x8x8xf16>

  // CHECK-NOT:   IE.MatMul
  // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_A]], [[INPUT_B]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       return [[MULTIPLY]] : tensor<1x4x8x8xf16>
}

// -----

// CHECK-LABEL: @NotConvertOuterProductInterleavedDims
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<4x256x1x128xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<4x1x256x128xf16>
func.func @NotConvertOuterProductInterleavedDims(%arg0: tensor<4x256x1x128xf16>, %arg1: tensor<4x1x256x128xf16>) -> tensor<4x256x256x128xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x256x1x128xf16>, tensor<4x1x256x128xf16> -> tensor<4x256x256x128xf16>
  return %mul : tensor<4x256x256x128xf16>

  // CHECK-NOT:   IE.MatMul
  // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_A]], [[INPUT_B]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       return [[MULTIPLY]] : tensor<4x256x256x128xf16>
}

// -----

// Multi-use Multiply with ReduceSum user but non-interleaved dims: outer product rewriter converts.
// The ReduceSum remains but the Multiply is replaced by a cheaper Reshape+MatMul+Reshape.

// CHECK-LABEL: @ConvertOuterProductMultiplyWithReduceSumMultiUser
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<1x16x128x1xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<1x16x1x128xf16>
func.func @ConvertOuterProductMultiplyWithReduceSumMultiUser(%arg0: tensor<1x16x128x1xf16>, %arg1: tensor<1x16x1x128xf16>) -> (tensor<1x16x128x1xf16>, tensor<1x16x128x128xf16>) {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x128x1xf16>, tensor<1x16x1x128xf16> -> tensor<1x16x128x128xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [3], keep_dims} : tensor<1x16x128x128xf16> -> tensor<1x16x128x1xf16>
  return %red, %mul : tensor<1x16x128x1xf16>, tensor<1x16x128x128xf16>

  // CHECK-NOT:   IE.Multiply
  // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT_A]]) {shape_value = [16, 128, 1]}
  // CHECK:       [[RHS:%.+]] = IE.Reshape([[INPUT_B]]) {shape_value = [16, 128, 1]}
  // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[LHS]], [[RHS]]) {transpose_b}
  // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 16, 128, 128]}
  // CHECK:       [[REDUCESUM:%.+]] = IE.ReduceSum([[RESHAPE]]) {axes_value = [3], keep_dims}
  // CHECK:       return [[REDUCESUM]], [[RESHAPE]]
}

// -----

// Do not convert single-axis broadcast (not an outer product)

// CHECK-LABEL: @NotConvertSingleAxisBroadcastMultiply
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<1x16x128x128xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<1x16x1x128xf16>
func.func @NotConvertSingleAxisBroadcastMultiply(%arg0: tensor<1x16x128x128xf16>, %arg1: tensor<1x16x1x128xf16>) -> tensor<1x16x128x128xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x128x128xf16>, tensor<1x16x1x128xf16> -> tensor<1x16x128x128xf16>
  return %mul : tensor<1x16x128x128xf16>

  // CHECK-NOT:   IE.MatMul
  // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_A]], [[INPUT_B]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       return [[MULTIPLY]] : tensor<1x16x128x128xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d5, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>

// CHECK-LABEL: @PropagateTransposeThroughMulAndReduce
// CHECK-SAME:  [[INPUT_A:%.+]]: tensor<1x4x256x64x64xf16>, [[INPUT_B:%.+]]: tensor<1x4x256x64x128xf16>
func.func @PropagateTransposeThroughMulAndReduce(%arg0: tensor<1x4x256x64x64xf16>, %arg1: tensor<1x4x256x64x128xf16>) -> tensor<1x4x64x128x64xf16> {
  %t0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>} : tensor<1x4x256x64x64xf16> -> tensor<1x4x64x256x64xf16>
  %r0 = IE.AffineReshape(%t0) {dim_mapping = [[0], [1], [2], [3, 4], [5]], shape_value = [1, 4, 64, 256, 1, 64]} : tensor<1x4x64x256x64xf16> -> tensor<1x4x64x256x1x64xf16>

  %t1 = IE.Transpose(%arg1) {order_value = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>} : tensor<1x4x256x64x128xf16> -> tensor<1x4x64x256x128xf16>
  %r1 = IE.AffineReshape(%t1) {dim_mapping = [[0], [1], [2], [3], [4, 5]], shape_value = [1, 4, 64, 256, 128, 1]} : tensor<1x4x64x256x128xf16> -> tensor<1x4x64x256x128x1xf16>

  %mul = IE.Multiply(%r0, %r1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x64x256x1x64xf16>, tensor<1x4x64x256x128x1xf16> -> tensor<1x4x64x256x128x64xf16>
  %reduce = IE.ReduceSum(%mul) {axes_value = [3]} : tensor<1x4x64x256x128x64xf16> -> tensor<1x4x64x128x64xf16>
  return %reduce : tensor<1x4x64x128x64xf16>

  // CHECK: [[RESH0:%.+]] = IE.AffineReshape([[INPUT_A]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3, 4], [5]], shape_value = [1, 4, 256, 64, 1, 64]} : tensor<1x4x256x64x64xf16> -> tensor<1x4x256x64x1x64xf16>
  // CHECK: [[RESH1:%.+]] = IE.AffineReshape([[INPUT_B]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3], [4, 5]], shape_value = [1, 4, 256, 64, 128, 1]} : tensor<1x4x256x64x128xf16> -> tensor<1x4x256x64x128x1xf16>

  // CHECK: [[TRANS0:%.+]] = IE.Transpose([[RESH0]]) {order_value = #map} : tensor<1x4x256x64x1x64xf16> -> tensor<1x4x64x1x64x256xf16>
  // CHECK: [[TRANS1:%.+]] = IE.Transpose([[RESH1]]) {order_value = #map} : tensor<1x4x256x64x128x1xf16> -> tensor<1x4x64x128x1x256xf16>
  // CHECK: [[IN_RESH0:%.+]] = IE.Reshape([[TRANS0]]) {shape_value = [256, 64, 256]} : tensor<1x4x64x1x64x256xf16> -> tensor<256x64x256xf16>
  // CHECK: [[IN_RESH1:%.+]] = IE.Reshape([[TRANS1]]) {shape_value = [256, 128, 256]} : tensor<1x4x64x128x1x256xf16> -> tensor<256x128x256xf16>

  // CHECK: [[MATMUL:%.+]] = IE.MatMul([[IN_RESH0]], [[IN_RESH1]]) {transpose_b} : tensor<256x64x256xf16>, tensor<256x128x256xf16> -> tensor<256x64x128xf16>

  // CHECK: [[OUT_RESHAPE:%.+]] = IE.Reshape([[MATMUL]]) {shape_value = [1, 4, 64, 64, 128]} : tensor<256x64x128xf16> -> tensor<1x4x64x64x128xf16>
  // CHECK: [[OUT_TRANS:%.+]] = IE.Transpose([[OUT_RESHAPE]]) {order_value = #map1} : tensor<1x4x64x64x128xf16> -> tensor<1x4x64x128x64xf16>
  // CHECK: [[RES_RESHAPE:%.+]] = IE.Reshape([[OUT_TRANS]]) {shape_value = [1, 4, 64, 128, 64]} : tensor<1x4x64x128x64xf16> -> tensor<1x4x64x128x64xf16>
  // CHECK: return [[RES_RESHAPE]] : tensor<1x4x64x128x64xf16>
}
