// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --run-batch-op-processing-rewriters="rewriter=eliminate-reshape-roundtrip-in-sdpa-set" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// No Add on the SoftMax path, just Concat -> Reshape -> SoftMax -> Reshape -> Slice.

// CHECK-LABEL: @PropagateHeadShrinkNoAdd
func.func @PropagateHeadShrinkNoAdd(%arg0: tensor<1x8x32x4352xf32>, %arg1: tensor<1x8x32x4352xf32>) -> tensor<1x8x32x8704xf32> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 4352]]} : tensor<1x8x32x4352xf32>, tensor<1x8x32x4352xf32> -> tensor<1x8x32x8704xf32>
    %1 = IE.Reshape(%0) {shape_value = [1, 32, 8, 8704]} : tensor<1x8x32x8704xf32> -> tensor<1x32x8x8704xf32>
    %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<1x32x8x8704xf32> -> tensor<1x32x8x8704xf32>
    %3 = IE.Reshape(%2) {shape_value = [1, 8, 32, 8704]} : tensor<1x32x8x8704xf32> -> tensor<1x8x32x8704xf32>
    %4 = IE.Slice %3 [0, 0, 0, 0] [1, 8, 32, 4352] : tensor<1x8x32x8704xf32> to tensor<1x8x32x4352xf32>
    %5 = IE.Slice %3 [0, 0, 0, 4352] [1, 8, 32, 4352] : tensor<1x8x32x8704xf32> to tensor<1x8x32x4352xf32>
    %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 4352]]} : tensor<1x8x32x4352xf32>, tensor<1x8x32x4352xf32> -> tensor<1x8x32x8704xf32>

    return %6 : tensor<1x8x32x8704xf32>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT0:%.+]], [[INPUT1:%.+]]) {static_offsets = {{\[\[0, 0, 0, 0\], \[0, 0, 0, 4352\]\]}}} : tensor<1x8x32x4352xf32>, tensor<1x8x32x4352xf32> -> tensor<1x8x32x8704xf32>
    // CHECK-NOT:   IE.Reshape
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[CONCAT]]) {axisInd = 3 : i64} : tensor<1x8x32x8704xf32> -> tensor<1x8x32x8704xf32>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[SOFTMAX]] [0, 0, 0, 0] [1, 8, 32, 4352] : tensor<1x8x32x8704xf32> to tensor<1x8x32x4352xf32>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[SOFTMAX]] [0, 0, 0, 4352] [1, 8, 32, 4352] : tensor<1x8x32x8704xf32> to tensor<1x8x32x4352xf32>
    // CHECK:       [[OUT:%.+]] = IE.Concat([[SLICE0]], [[SLICE1]]) {static_offsets = {{\[\[0, 0, 0, 0\], \[0, 0, 0, 4352\]\]}}} : tensor<1x8x32x4352xf32>, tensor<1x8x32x4352xf32> -> tensor<1x8x32x8704xf32>
    // CHECK:       return [[OUT]] : tensor<1x8x32x8704xf32>
    // CHECK-NOT:   IE.Reshape
}

// -----

// Per-token causal mask case: mask is 1x1xTxK (here T=8, K=8704).
// It broadcasts in expanded shape 1x32x8x8704, but not in shrunk shape
// 1x8x32x8704 because dim2 changes from T to H_per_G*T (=32).
// Tile the mask by H_per_G on dim2 (IE.Tile repeats [1, 1, 4, 1]) to build an
// equivalent shrunk-domain mask, then fold away both Reshapes.

// CHECK-LABEL: @PropagateHeadShrinkPerTokenCausalMask
func.func @PropagateHeadShrinkPerTokenCausalMask(%arg0: tensor<1x8x32x4352xf32>, %arg1: tensor<1x8x32x4352xf32>,
                                                 %arg2: tensor<1x1x8x8704xf32>) -> tensor<1x8x32x8704xf32> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 4352]]} : tensor<1x8x32x4352xf32>, tensor<1x8x32x4352xf32> -> tensor<1x8x32x8704xf32>
    %1 = IE.Reshape(%0) {shape_value = [1, 32, 8, 8704]} : tensor<1x8x32x8704xf32> -> tensor<1x32x8x8704xf32>
    %2 = IE.Add(%1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x8x8704xf32>, tensor<1x1x8x8704xf32> -> tensor<1x32x8x8704xf32>
    %3 = IE.SoftMax(%2) {axisInd = 3 : i64} : tensor<1x32x8x8704xf32> -> tensor<1x32x8x8704xf32>
    %4 = IE.Reshape(%3) {shape_value = [1, 8, 32, 8704]} : tensor<1x32x8x8704xf32> -> tensor<1x8x32x8704xf32>
    %5 = IE.Slice %4 [0, 0, 0, 0] [1, 8, 32, 4352] : tensor<1x8x32x8704xf32> to tensor<1x8x32x4352xf32>
    %6 = IE.Slice %4 [0, 0, 0, 4352] [1, 8, 32, 4352] : tensor<1x8x32x8704xf32> to tensor<1x8x32x4352xf32>
    %7 = IE.Concat(%5, %6) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 4352]]} : tensor<1x8x32x4352xf32>, tensor<1x8x32x4352xf32> -> tensor<1x8x32x8704xf32>

    return %7 : tensor<1x8x32x8704xf32>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT0:%.+]], [[INPUT1:%.+]]) {static_offsets = {{\[\[0, 0, 0, 0\], \[0, 0, 0, 4352\]\]}}} : tensor<1x8x32x4352xf32>, tensor<1x8x32x4352xf32> -> tensor<1x8x32x8704xf32>
    // CHECK:       [[TILED_MASK:%.+]] = IE.Tile([[MASK:%.+]]) {repeats_values = [1, 1, 4, 1]} : tensor<1x1x8x8704xf32> -> tensor<1x1x32x8704xf32>
    // CHECK:       [[ADD:%.+]] = IE.Add([[CONCAT]], [[TILED_MASK]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x32x8704xf32>, tensor<1x1x32x8704xf32> -> tensor<1x8x32x8704xf32>
    // CHECK-NOT:   IE.Reshape
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 3 : i64} : tensor<1x8x32x8704xf32> -> tensor<1x8x32x8704xf32>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[SOFTMAX]] [0, 0, 0, 0] [1, 8, 32, 4352] : tensor<1x8x32x8704xf32> to tensor<1x8x32x4352xf32>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[SOFTMAX]] [0, 0, 0, 4352] [1, 8, 32, 4352] : tensor<1x8x32x8704xf32> to tensor<1x8x32x4352xf32>
    // CHECK:       [[OUT:%.+]] = IE.Concat([[SLICE0]], [[SLICE1]]) {static_offsets = {{\[\[0, 0, 0, 0\], \[0, 0, 0, 4352\]\]}}} : tensor<1x8x32x4352xf32>, tensor<1x8x32x4352xf32> -> tensor<1x8x32x8704xf32>
    // CHECK:       return [[OUT]] : tensor<1x8x32x8704xf32>
    // CHECK-NOT:   IE.Reshape
}
