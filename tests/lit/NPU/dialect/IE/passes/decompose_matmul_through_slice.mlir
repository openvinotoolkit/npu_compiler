//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --decompose-matmul-through-slice %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// -----

//
// Positive: fused QKV projection decomposed into 3 independent MatMuls (transposeB=false).
// Pattern: MatMul(act, W[32x48]) -> Add(bias) -> Reshape([4,3,16]) -> VariadicSplit(axis=1)
//

// CHECK-LABEL: @DecomposeBasicQKV
// CHECK-SAME:  ([[ACT:%.+]]: tensor<4x32xf16>, [[W:%.+]]: tensor<32x48xf16>, [[B:%.+]]: tensor<1x48xf16>)
func.func @DecomposeBasicQKV(%act: tensor<4x32xf16>, %w: tensor<32x48xf16>, %b: tensor<1x48xf16>)
        -> (tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>) {
    %mm  = IE.MatMul(%act, %w) : tensor<4x32xf16>, tensor<32x48xf16> -> tensor<4x48xf16>
    %add = IE.Add(%mm, %b) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x48xf16>, tensor<1x48xf16> -> tensor<4x48xf16>
    %rs  = IE.Reshape(%add) {shape_value = [4, 3, 16]} : tensor<4x48xf16> -> tensor<4x3x16xf16>
    %q:3 = IE.VariadicSplit(%rs) {axis=1, split_lengths=[1, 1, 1]} : tensor<4x3x16xf16> -> tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>
    return %q#0, %q#1, %q#2 : tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>

    // CHECK-NOT: IE.VariadicSplit
    // CHECK:     [[W0:%.+]] = IE.Slice [[W]] [0, 0] [32, 16]  : tensor<32x48xf16> to tensor<32x16xf16>
    // CHECK:     [[MM0:%.+]] = IE.MatMul([[ACT]], [[W0]])      : tensor<4x32xf16>, tensor<32x16xf16> -> tensor<4x16xf16>
    // CHECK:     [[B0:%.+]] = IE.Slice [[B]] [0, 0] [1, 16]   : tensor<1x48xf16> to tensor<1x16xf16>
    // CHECK:     [[ADD0:%.+]] = IE.Add([[MM0]], [[B0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x16xf16>, tensor<1x16xf16> -> tensor<4x16xf16>
    // CHECK:     [[RS0:%.+]] = IE.Reshape([[ADD0]]) {shape_value = [4, 1, 16]} : tensor<4x16xf16> -> tensor<4x1x16xf16>
    // CHECK:     [[W1:%.+]] = IE.Slice [[W]] [0, 16] [32, 16] : tensor<32x48xf16> to tensor<32x16xf16>
    // CHECK:     [[MM1:%.+]] = IE.MatMul([[ACT]], [[W1]])      : tensor<4x32xf16>, tensor<32x16xf16> -> tensor<4x16xf16>
    // CHECK:     [[B1:%.+]] = IE.Slice [[B]] [0, 16] [1, 16]  : tensor<1x48xf16> to tensor<1x16xf16>
    // CHECK:     [[ADD1:%.+]] = IE.Add([[MM1]], [[B1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x16xf16>, tensor<1x16xf16> -> tensor<4x16xf16>
    // CHECK:     [[RS1:%.+]] = IE.Reshape([[ADD1]]) {shape_value = [4, 1, 16]} : tensor<4x16xf16> -> tensor<4x1x16xf16>
    // CHECK:     [[W2:%.+]] = IE.Slice [[W]] [0, 32] [32, 16] : tensor<32x48xf16> to tensor<32x16xf16>
    // CHECK:     [[MM2:%.+]] = IE.MatMul([[ACT]], [[W2]])      : tensor<4x32xf16>, tensor<32x16xf16> -> tensor<4x16xf16>
    // CHECK:     [[B2:%.+]] = IE.Slice [[B]] [0, 32] [1, 16]  : tensor<1x48xf16> to tensor<1x16xf16>
    // CHECK:     [[ADD2:%.+]] = IE.Add([[MM2]], [[B2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x16xf16>, tensor<1x16xf16> -> tensor<4x16xf16>
    // CHECK:     [[RS2:%.+]] = IE.Reshape([[ADD2]]) {shape_value = [4, 1, 16]} : tensor<4x16xf16> -> tensor<4x1x16xf16>
    // CHECK:     return [[RS0]], [[RS1]], [[RS2]]
    // CHECK-NOT: IE.VariadicSplit
}

// -----

//
// Positive: transposeB=true — weight sliced along axis=0 (rows).
//

// CHECK-LABEL: @DecomposeTransposedWeights
// CHECK-SAME:  ([[ACT:%.+]]: tensor<4x32xf16>, [[W:%.+]]: tensor<48x32xf16>, [[B:%.+]]: tensor<1x48xf16>)
func.func @DecomposeTransposedWeights(%act: tensor<4x32xf16>, %w: tensor<48x32xf16>, %b: tensor<1x48xf16>)
        -> (tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>) {
    %mm  = IE.MatMul(%act, %w) {transpose_b} : tensor<4x32xf16>, tensor<48x32xf16> -> tensor<4x48xf16>
    %add = IE.Add(%mm, %b) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x48xf16>, tensor<1x48xf16> -> tensor<4x48xf16>
    %rs  = IE.Reshape(%add) {shape_value = [4, 3, 16]} : tensor<4x48xf16> -> tensor<4x3x16xf16>
    %q:3 = IE.VariadicSplit(%rs) {axis=1, split_lengths=[1, 1, 1]} : tensor<4x3x16xf16> -> tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>
    return %q#0, %q#1, %q#2 : tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>

    // CHECK-NOT: IE.VariadicSplit
    // CHECK:     [[W0:%.+]] = IE.Slice [[W]] [0, 0] [16, 32]  : tensor<48x32xf16> to tensor<16x32xf16>
    // CHECK:     [[MM0:%.+]] = IE.MatMul([[ACT]], [[W0]]) {transpose_b} : tensor<4x32xf16>, tensor<16x32xf16> -> tensor<4x16xf16>
    // CHECK:     [[B0:%.+]] = IE.Slice [[B]] [0, 0] [1, 16]   : tensor<1x48xf16> to tensor<1x16xf16>
    // CHECK:     [[RS0:%.+]] = IE.Reshape({{%.+}}) {shape_value = [4, 1, 16]} : tensor<4x16xf16> -> tensor<4x1x16xf16>
    // CHECK:     [[W1:%.+]] = IE.Slice [[W]] [16, 0] [16, 32] : tensor<48x32xf16> to tensor<16x32xf16>
    // CHECK:     [[MM1:%.+]] = IE.MatMul([[ACT]], [[W1]]) {transpose_b} : tensor<4x32xf16>, tensor<16x32xf16> -> tensor<4x16xf16>
    // CHECK:     [[B1:%.+]] = IE.Slice [[B]] [0, 16] [1, 16]  : tensor<1x48xf16> to tensor<1x16xf16>
    // CHECK:     [[RS1:%.+]] = IE.Reshape({{%.+}}) {shape_value = [4, 1, 16]} : tensor<4x16xf16> -> tensor<4x1x16xf16>
    // CHECK:     [[W2:%.+]] = IE.Slice [[W]] [32, 0] [16, 32] : tensor<48x32xf16> to tensor<16x32xf16>
    // CHECK:     [[MM2:%.+]] = IE.MatMul([[ACT]], [[W2]]) {transpose_b} : tensor<4x32xf16>, tensor<16x32xf16> -> tensor<4x16xf16>
    // CHECK:     [[B2:%.+]] = IE.Slice [[B]] [0, 32] [1, 16]  : tensor<1x48xf16> to tensor<1x16xf16>
    // CHECK:     [[RS2:%.+]] = IE.Reshape({{%.+}}) {shape_value = [4, 1, 16]} : tensor<4x16xf16> -> tensor<4x1x16xf16>
    // CHECK:     return [[RS0]], [[RS1]], [[RS2]]
    // CHECK-NOT: IE.VariadicSplit
}

// -----

//
// Negative: VariadicSplit on the first axis (axis=0) — interior-axis guard rejects.
//

// CHECK-LABEL: @NoTransformFirstAxis
func.func @NoTransformFirstAxis(%act: tensor<4x32xf16>, %w: tensor<32x12xf16>, %b: tensor<1x12xf16>)
        -> (tensor<2x3x4xf16>, tensor<2x3x4xf16>) {
    %mm  = IE.MatMul(%act, %w) : tensor<4x32xf16>, tensor<32x12xf16> -> tensor<4x12xf16>
    %add = IE.Add(%mm, %b) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x12xf16>, tensor<1x12xf16> -> tensor<4x12xf16>
    %rs  = IE.Reshape(%add) {shape_value = [4, 3, 4]} : tensor<4x12xf16> -> tensor<4x3x4xf16>
    %q:2 = IE.VariadicSplit(%rs) {axis=0, split_lengths=[2, 2]} : tensor<4x3x4xf16> -> tensor<2x3x4xf16>, tensor<2x3x4xf16>
    return %q#0, %q#1 : tensor<2x3x4xf16>, tensor<2x3x4xf16>

    // CHECK: IE.VariadicSplit
}

// -----

//
// Negative: VariadicSplit on the last axis (axis=rank-1) — interior-axis guard rejects.
//

// CHECK-LABEL: @NoTransformLastAxis
func.func @NoTransformLastAxis(%act: tensor<4x32xf16>, %w: tensor<32x48xf16>, %b: tensor<1x48xf16>)
        -> (tensor<4x3x8xf16>, tensor<4x3x8xf16>) {
    %mm  = IE.MatMul(%act, %w) : tensor<4x32xf16>, tensor<32x48xf16> -> tensor<4x48xf16>
    %add = IE.Add(%mm, %b) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x48xf16>, tensor<1x48xf16> -> tensor<4x48xf16>
    %rs  = IE.Reshape(%add) {shape_value = [4, 3, 16]} : tensor<4x48xf16> -> tensor<4x3x16xf16>
    %q:2 = IE.VariadicSplit(%rs) {axis=2, split_lengths=[8, 8]} : tensor<4x3x16xf16> -> tensor<4x3x8xf16>, tensor<4x3x8xf16>
    return %q#0, %q#1 : tensor<4x3x8xf16>, tensor<4x3x8xf16>

    // CHECK: IE.VariadicSplit
}

// -----

//
// Positive: AffineReshape bridging flat MatMul output to split shape.
// Pattern: MatMul(act, W[32x48]) -> Add(bias) -> AffineReshape([4,3,16]) -> VariadicSplit(axis=1)
//

// CHECK-LABEL: @DecomposeWithAffineReshape
// CHECK-SAME:  ([[ACT:%.+]]: tensor<4x32xf16>, [[W:%.+]]: tensor<32x48xf16>, [[B:%.+]]: tensor<1x48xf16>)
func.func @DecomposeWithAffineReshape(%act: tensor<4x32xf16>, %w: tensor<32x48xf16>, %b: tensor<1x48xf16>)
        -> (tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>) {
    %mm  = IE.MatMul(%act, %w) : tensor<4x32xf16>, tensor<32x48xf16> -> tensor<4x48xf16>
    %add = IE.Add(%mm, %b) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x48xf16>, tensor<1x48xf16> -> tensor<4x48xf16>
    %rs  = IE.AffineReshape(%add) {
      dim_mapping = [[0], [1, 2]],
      shape_value = [4, 3, 16]
    } : tensor<4x48xf16> -> tensor<4x3x16xf16>
    %q:3 = IE.VariadicSplit(%rs) {axis=1, split_lengths=[1, 1, 1]} : tensor<4x3x16xf16> -> tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>
    return %q#0, %q#1, %q#2 : tensor<4x1x16xf16>, tensor<4x1x16xf16>, tensor<4x1x16xf16>

    // CHECK-NOT: IE.VariadicSplit
    // CHECK:     [[W0:%.+]] = IE.Slice [[W]] [0, 0] [32, 16]  : tensor<32x48xf16> to tensor<32x16xf16>
    // CHECK:     [[MM0:%.+]] = IE.MatMul([[ACT]], [[W0]])      : tensor<4x32xf16>, tensor<32x16xf16> -> tensor<4x16xf16>
    // CHECK:     [[B0:%.+]] = IE.Slice [[B]] [0, 0] [1, 16]   : tensor<1x48xf16> to tensor<1x16xf16>
    // CHECK:     [[ADD0:%.+]] = IE.Add([[MM0]], [[B0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x16xf16>, tensor<1x16xf16> -> tensor<4x16xf16>
    // CHECK:     [[RS0:%.+]] = IE.AffineReshape([[ADD0]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0], [1, 2]],
    // CHECK-SAME:           shape_value = [4, 1, 16]
    // CHECK-SAME: } : tensor<4x16xf16> -> tensor<4x1x16xf16>
    // CHECK:     [[W1:%.+]] = IE.Slice [[W]] [0, 16] [32, 16] : tensor<32x48xf16> to tensor<32x16xf16>
    // CHECK:     [[MM1:%.+]] = IE.MatMul([[ACT]], [[W1]])      : tensor<4x32xf16>, tensor<32x16xf16> -> tensor<4x16xf16>
    // CHECK:     [[B1:%.+]] = IE.Slice [[B]] [0, 16] [1, 16]  : tensor<1x48xf16> to tensor<1x16xf16>
    // CHECK:     [[ADD1:%.+]] = IE.Add([[MM1]], [[B1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x16xf16>, tensor<1x16xf16> -> tensor<4x16xf16>
    // CHECK:     [[RS1:%.+]] = IE.AffineReshape([[ADD1]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0], [1, 2]],
    // CHECK-SAME:           shape_value = [4, 1, 16]
    // CHECK-SAME: } : tensor<4x16xf16> -> tensor<4x1x16xf16>
    // CHECK:     [[W2:%.+]] = IE.Slice [[W]] [0, 32] [32, 16] : tensor<32x48xf16> to tensor<32x16xf16>
    // CHECK:     [[MM2:%.+]] = IE.MatMul([[ACT]], [[W2]])      : tensor<4x32xf16>, tensor<32x16xf16> -> tensor<4x16xf16>
    // CHECK:     [[B2:%.+]] = IE.Slice [[B]] [0, 32] [1, 16]  : tensor<1x48xf16> to tensor<1x16xf16>
    // CHECK:     [[ADD2:%.+]] = IE.Add([[MM2]], [[B2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x16xf16>, tensor<1x16xf16> -> tensor<4x16xf16>
    // CHECK:     [[RS2:%.+]] = IE.AffineReshape([[ADD2]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0], [1, 2]],
    // CHECK-SAME:           shape_value = [4, 1, 16]
    // CHECK-SAME: } : tensor<4x16xf16> -> tensor<4x1x16xf16>
    // CHECK:     return [[RS0]], [[RS1]], [[RS2]]
    // CHECK-NOT: IE.VariadicSplit
}
