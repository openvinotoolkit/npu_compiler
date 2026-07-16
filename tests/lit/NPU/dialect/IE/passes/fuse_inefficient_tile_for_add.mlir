//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-inefficient-tile-for-add %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// -----

// Tile broadcasts the innermost memory dimension (W) — should be fused.
//
// CHECK-LABEL: @FuseInnermostDimBroadcast
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x8x1x1xsi32>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x8x1x2048xsi32>
func.func @FuseInnermostDimBroadcast(%arg0: tensor<1x8x1x1xsi32>, %arg1: tensor<1x8x1x2048xsi32>) -> tensor<1x8x1x2048xsi32> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 2048]} : tensor<1x8x1x1xsi32> -> tensor<1x8x1x2048xsi32>
    %1 = IE.Add(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x1x2048xsi32>, tensor<1x8x1x2048xsi32> -> tensor<1x8x1x2048xsi32>
    return %1 : tensor<1x8x1x2048xsi32>

    // Tile should be removed; Add uses the original small input directly.
    // CHECK-NOT: IE.Tile
    // CHECK:     [[ADD:%.+]] = IE.Add([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:              : tensor<1x8x1x1xsi32>, tensor<1x8x1x2048xsi32> -> tensor<1x8x1x2048xsi32>
    // CHECK:     return [[ADD]]
}

// -----

// Tile broadcasts the innermost memory dimension on input2 — should be fused.
//
// CHECK-LABEL: @FuseInnermostDimBroadcastInput2
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x8x1x2048xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x8x1x1xf16>
func.func @FuseInnermostDimBroadcastInput2(%arg0: tensor<1x8x1x2048xf16>, %arg1: tensor<1x8x1x1xf16>) -> tensor<1x8x1x2048xf16> {
    %0 = IE.Tile(%arg1) {repeats_values = [1, 1, 1, 2048]} : tensor<1x8x1x1xf16> -> tensor<1x8x1x2048xf16>
    %1 = IE.Add(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x1x2048xf16>, tensor<1x8x1x2048xf16> -> tensor<1x8x1x2048xf16>
    return %1 : tensor<1x8x1x2048xf16>

    // CHECK-NOT: IE.Tile
    // CHECK:     [[ADD:%.+]] = IE.Add([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:              : tensor<1x8x1x2048xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x1x2048xf16>
    // CHECK:     return [[ADD]]
}

// -----

// Tile broadcasts a non-innermost dimension (C) — should NOT be fused.
//
// CHECK-LABEL: @NoFuseNonInnermostDimBroadcast
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x1x1x16xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x32x1x16xf16>
func.func @NoFuseNonInnermostDimBroadcast(%arg0: tensor<1x1x1x16xf16>, %arg1: tensor<1x32x1x16xf16>) -> tensor<1x32x1x16xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 32, 1, 1]} : tensor<1x1x1x16xf16> -> tensor<1x32x1x16xf16>
    %1 = IE.Add(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x16xf16>, tensor<1x32x1x16xf16> -> tensor<1x32x1x16xf16>
    return %1 : tensor<1x32x1x16xf16>

    // Innermost non-trivial output dim is W=16; input has W=16 (not 1) — no fusion.
    // CHECK: [[TILE:%.+]] = IE.Tile([[ARG_0]]) {repeats_values = [1, 32, 1, 1]}
    // CHECK: [[ADD:%.+]]  = IE.Add([[TILE]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK: return [[ADD]]
}

// -----

// Splat input (1x1x1x1) — DMA handles this efficiently; should NOT be fused.
//
// CHECK-LABEL: @NoFuseSplatInput
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x1x1x1xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x8x1x2048xf16>
func.func @NoFuseSplatInput(%arg0: tensor<1x1x1x1xf16>, %arg1: tensor<1x8x1x2048xf16>) -> tensor<1x8x1x2048xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 8, 1, 2048]} : tensor<1x1x1x1xf16> -> tensor<1x8x1x2048xf16>
    %1 = IE.Add(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x1x2048xf16>, tensor<1x8x1x2048xf16> -> tensor<1x8x1x2048xf16>
    return %1 : tensor<1x8x1x2048xf16>

    // CHECK: [[TILE:%.+]] = IE.Tile([[ARG_0]]) {repeats_values = [1, 8, 1, 2048]}
    // CHECK-SAME:           : tensor<1x1x1x1xf16> -> tensor<1x8x1x2048xf16>
    // CHECK: [[ADD:%.+]]  = IE.Add([[TILE]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:           : tensor<1x8x1x2048xf16>, tensor<1x8x1x2048xf16> -> tensor<1x8x1x2048xf16>
    // CHECK: return [[ADD]]
}

// -----

// Tile output is used by multiple consumers — should NOT be fused (no hasOneUse).
//
// CHECK-LABEL: @NoFuseMultiUse
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x8x1x1xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: tensor<1x8x1x2048xf16>
func.func @NoFuseMultiUse(%arg0: tensor<1x8x1x1xf16>, %arg1: tensor<1x8x1x2048xf16>) -> (tensor<1x8x1x2048xf16>, tensor<1x8x1x2048xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 2048]} : tensor<1x8x1x1xf16> -> tensor<1x8x1x2048xf16>
    %1 = IE.Add(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x1x2048xf16>, tensor<1x8x1x2048xf16> -> tensor<1x8x1x2048xf16>
    return %1, %0 : tensor<1x8x1x2048xf16>, tensor<1x8x1x2048xf16>

    // CHECK: [[TILE:%.+]] = IE.Tile([[ARG_0]]) {repeats_values = [1, 1, 1, 2048]}
    // CHECK-SAME:           : tensor<1x8x1x1xf16> -> tensor<1x8x1x2048xf16>
    // CHECK: [[ADD:%.+]]  = IE.Add([[TILE]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:           : tensor<1x8x1x2048xf16>, tensor<1x8x1x2048xf16> -> tensor<1x8x1x2048xf16>
    // CHECK: return [[ADD]], [[TILE]]
}
