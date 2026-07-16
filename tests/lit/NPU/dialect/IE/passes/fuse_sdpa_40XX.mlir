//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-sdpa --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000

// CHECK-LABEL: @FuseSDPA_FullyConnectedWithMultiply
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x1x64xf32>, [[ARG1:%.+]]: tensor<1x1x64x64xf32>, [[ARG2:%.+]]: tensor<1xf32>, [[ARG3:%.+]]: tensor<1x1x1x64xf32>, [[ARG4:%.+]]: tensor<1x1x64x64xf32>)
func.func @FuseSDPA_FullyConnectedWithMultiply(%arg0: tensor<1x1x1x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1x1x1x64xf32>, %arg4: tensor<1x1x64x64xf32>) -> tensor<1x1x1x64xf32> {
  %0 = IE.Reshape(%arg0) {shape_value = [1, 64]} : tensor<1x1x1x64xf32> -> tensor<1x64xf32>
  %1 = IE.Multiply(%arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x64xf32>, tensor<1xf32> -> tensor<1x1x64x64xf32>
  %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x64xf32> -> tensor<1x1x64x64xf32>
  %3 = IE.Reshape(%2) {shape_value = [64, 64]} : tensor<1x1x64x64xf32> -> tensor<64x64xf32>
  %4 = IE.FullyConnected(%0, %3) : tensor<1x64xf32>, tensor<64x64xf32> -> tensor<1x64xf32>
  %5 = IE.Reshape(%4) {shape_value = [1, 1, 1, 64]} : tensor<1x64xf32> -> tensor<1x1x1x64xf32>
  %6 = IE.Add(%5, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32> -> tensor<1x1x1x64xf32>
  %7 = IE.SoftMax(%6) {axisInd = 3 : i64} : tensor<1x1x1x64xf32> -> tensor<1x1x1x64xf32>
  %8 = IE.Reshape(%7) {shape_value = [1, 64]} : tensor<1x1x1x64xf32> -> tensor<1x64xf32>
  %9 = IE.Reshape(%arg4) {shape_value = [64, 64]} : tensor<1x1x64x64xf32> -> tensor<64x64xf32>
  %10 = IE.FullyConnected(%8, %9) : tensor<1x64xf32>, tensor<64x64xf32> -> tensor<1x64xf32>
  %11 = IE.Reshape(%10) {shape_value = [1, 1, 1, 64]} : tensor<1x64xf32> -> tensor<1x1x1x64xf32>

  return %11 : tensor<1x1x1x64xf32>

    // CHECK: [[TRANSPOSEK:%.+]] = IE.Transpose([[ARG1]]) {order_value = #NCWH} : tensor<1x1x64x64xf32> -> tensor<1x1x64x64xf32>
    // CHECK: [[SDPA:%.+]] = IE.SDPA([[ARG0]], [[TRANSPOSEK]], [[ARG4]], [[ARG3]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>} : tensor<1x1x1x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x1x64xf32> -> tensor<1x1x1x64xf32>
    // CHECK: return [[SDPA]] : tensor<1x1x1x64xf32>
}

// -----

// CHECK-LABEL: @FuseSDPA_FullyConnectedWithDivide
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x1x64xf32>, [[ARG1:%.+]]: tensor<1x1x64x64xf32>, [[ARG2:%.+]]: tensor<1xf32>, [[ARG3:%.+]]: tensor<1x1x1x64xf32>, [[ARG4:%.+]]: tensor<1x1x64x64xf32>)
func.func @FuseSDPA_FullyConnectedWithDivide(%arg0: tensor<1x1x1x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1x1x1x64xf32>, %arg4: tensor<1x1x64x64xf32>) -> tensor<1x1x1x64xf32> {
  %0 = IE.Reshape(%arg0) {shape_value = [1, 64]} : tensor<1x1x1x64xf32> -> tensor<1x64xf32>
  %1 = IE.Reshape(%arg1) {shape_value = [64, 64]} : tensor<1x1x64x64xf32> -> tensor<64x64xf32>
  %2 = IE.FullyConnected(%0, %1) : tensor<1x64xf32>, tensor<64x64xf32> -> tensor<1x64xf32>
  %3 = IE.Reshape(%2) {shape_value = [1, 1, 1, 64]} : tensor<1x64xf32> -> tensor<1x1x1x64xf32>
  %4 = IE.Divide(%3, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf32>, tensor<1xf32> -> tensor<1x1x1x64xf32>
  %5 = IE.Add(%4, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32> -> tensor<1x1x1x64xf32>
  %6 = IE.SoftMax(%5) {axisInd = 3 : i64} : tensor<1x1x1x64xf32> -> tensor<1x1x1x64xf32>
  %7 = IE.Reshape(%6) {shape_value = [1, 64]} : tensor<1x1x1x64xf32> -> tensor<1x64xf32>
  %8 = IE.Reshape(%arg4) {shape_value = [64, 64]} : tensor<1x1x64x64xf32> -> tensor<64x64xf32>
  %9 = IE.FullyConnected(%7, %8) : tensor<1x64xf32>, tensor<64x64xf32> -> tensor<1x64xf32>
  %10 = IE.Reshape(%9) {shape_value = [1, 1, 1, 64]} : tensor<1x64xf32> -> tensor<1x1x1x64xf32>

  return %10 : tensor<1x1x1x64xf32>

    // CHECK: [[SDPA:%.+]] = IE.SDPA([[ARG0]], [[ARG1]], [[ARG4]], [[ARG3]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>} : tensor<1x1x1x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x1x64xf32> -> tensor<1x1x1x64xf32>
    // CHECK: return [[SDPA]] : tensor<1x1x1x64xf32>
}

// -----

// CHECK-LABEL: @FuseSDPA_FullyConnectedWithMultiplyInsteadOfDivide
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x1x64xf32>, [[ARG1:%.+]]: tensor<1x1x64x64xf32>, [[ARG2:%.+]]: tensor<1xf32>, [[ARG3:%.+]]: tensor<1x1x1x64xf32>, [[ARG4:%.+]]: tensor<1x1x64x64xf32>)
func.func @FuseSDPA_FullyConnectedWithMultiplyInsteadOfDivide(%arg0: tensor<1x1x1x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1x1x1x64xf32>, %arg4: tensor<1x1x64x64xf32>) -> tensor<1x1x1x64xf32> {
  %0 = IE.Reshape(%arg0) {shape_value = [1, 64]} : tensor<1x1x1x64xf32> -> tensor<1x64xf32>
  %1 = IE.Reshape(%arg1) {shape_value = [64, 64]} : tensor<1x1x64x64xf32> -> tensor<64x64xf32>
  %2 = IE.FullyConnected(%0, %1) : tensor<1x64xf32>, tensor<64x64xf32> -> tensor<1x64xf32>
  %3 = IE.Reshape(%2) {shape_value = [1, 1, 1, 64]} : tensor<1x64xf32> -> tensor<1x1x1x64xf32>
  %4 = IE.Multiply(%3, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf32>, tensor<1xf32> -> tensor<1x1x1x64xf32>
  %5 = IE.Add(%4, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32> -> tensor<1x1x1x64xf32>
  %6 = IE.SoftMax(%5) {axisInd = 3 : i64} : tensor<1x1x1x64xf32> -> tensor<1x1x1x64xf32>
  %7 = IE.Reshape(%6) {shape_value = [1, 64]} : tensor<1x1x1x64xf32> -> tensor<1x64xf32>
  %8 = IE.Reshape(%arg4) {shape_value = [64, 64]} : tensor<1x1x64x64xf32> -> tensor<64x64xf32>
  %9 = IE.FullyConnected(%7, %8) : tensor<1x64xf32>, tensor<64x64xf32> -> tensor<1x64xf32>
  %10 = IE.Reshape(%9) {shape_value = [1, 1, 1, 64]} : tensor<1x64xf32> -> tensor<1x1x1x64xf32>

  return %10 : tensor<1x1x1x64xf32>

    // CHECK: [[SDPA:%.+]] = IE.SDPA([[ARG0]], [[ARG1]], [[ARG4]], [[ARG3]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>} : tensor<1x1x1x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x1x64xf32> -> tensor<1x1x1x64xf32>
    // CHECK: return [[SDPA]] : tensor<1x1x1x64xf32>
}

// -----

// CHECK-LABEL: @FuseSDPA_FullyConnectedAdaptiveStripping
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x1x64xf32>, [[ARG1:%.+]]: tensor<1x1x64x64xf32>, [[ARG2:%.+]]: tensor<1xf32>, [[ARG3:%.+]]: tensor<1x1x1x64xf32>, [[ARG4:%.+]]: tensor<1x1x64x64xf32>)
func.func @FuseSDPA_FullyConnectedAdaptiveStripping(%arg0: tensor<1x1x1x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1x1x1x64xf32>, %arg4: tensor<1x1x64x64xf32>) -> tensor<1x1x1x64xf32> {
  %0 = IE.Reshape(%arg0) {shape_value = [1, 64]} : tensor<1x1x1x64xf32> -> tensor<1x64xf32>
  %1 = IE.Reshape(%arg1) {shape_value = [64, 64]} : tensor<1x1x64x64xf32> -> tensor<64x64xf32>
  %2 = IE.FullyConnected(%0, %1) : tensor<1x64xf32>, tensor<64x64xf32> -> tensor<1x64xf32>
  %3 = IE.Reshape(%2) {shape_value = [1, 1, 1, 64]} : tensor<1x64xf32> -> tensor<1x1x1x64xf32>
  %4 = IE.Multiply(%3, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf32>, tensor<1xf32> -> tensor<1x1x1x64xf32>
  %5 = IE.Add(%4, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32> -> tensor<1x1x1x64xf32>
  %6 = IE.SoftMax(%5) {axisInd = 3 : i64} : tensor<1x1x1x64xf32> -> tensor<1x1x1x64xf32>
  %7 = IE.ReLU(%6) : tensor<1x1x1x64xf32> -> tensor<1x1x1x64xf32>
  %8 = IE.Reshape(%7) {shape_value = [1, 64]} : tensor<1x1x1x64xf32> -> tensor<1x64xf32>
  %9 = IE.Reshape(%arg4) {shape_value = [64, 64]} : tensor<1x1x64x64xf32> -> tensor<64x64xf32>
  %10 = IE.FullyConnected(%8, %9) : tensor<1x64xf32>, tensor<64x64xf32> -> tensor<1x64xf32>
  %11 = IE.Reshape(%10) {shape_value = [1, 1, 1, 64]} : tensor<1x64xf32> -> tensor<1x1x1x64xf32>

  return %11 : tensor<1x1x1x64xf32>

    // CHECK: [[SDPA:%.+]] = IE.SDPA([[ARG0]], [[ARG1]], [[ARG4]], [[ARG3]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>} : tensor<1x1x1x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x1x64xf32> -> tensor<1x1x1x64xf32>
    // CHECK: return [[SDPA]] : tensor<1x1x1x64xf32>
}

// -----

// CHECK-LABEL: @FuseSDPA_AdjustSDPA
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x18x12x8xf32>, [[ARG1:%.+]]: tensor<1x18x16x8xf32>, [[ARG2:%.+]]: tensor<1x18x16x4xf32>, [[ARG3:%.+]]: tensor<1x1x1x1xf32>, [[ARG4:%.+]]: tensor<1xf32>)
func.func @FuseSDPA_AdjustSDPA(%arg0: tensor<1x18x12x8xf32>, %arg1: tensor<1x18x16x8xf32>, %arg2: tensor<1x18x16x4xf32>, %arg3: tensor<1x1x1x1xf32>, %arg4: tensor<1xf32>) -> tensor<1x18x12x4xf32> {
  %0 = IE.SDPA(%arg0, %arg1, %arg2, %arg3, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : tensor<1x18x12x8xf32>, tensor<1x18x16x8xf32>, tensor<1x18x16x4xf32>, tensor<1x1x1x1xf32>, tensor<1xf32> -> tensor<1x18x12x4xf32>
  return %0 : tensor<1x18x12x4xf32>

  // CHECK: [[TRANSPOSEV:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH} : tensor<1x18x16x4xf32> -> tensor<1x18x4x16xf32>
  // CHECK: [[SDPA:%.+]] = IE.SDPA([[ARG0]], [[ARG1]], [[TRANSPOSEV]], [[ARG3]], [[ARG4]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : tensor<1x18x12x8xf32>, tensor<1x18x16x8xf32>, tensor<1x18x4x16xf32>, tensor<1x1x1x1xf32>, tensor<1xf32> -> tensor<1x18x12x4xf32>
}

// -----

// CHECK-LABEL: @NoFuseSDPA_IllegalConfig_tSL64_sSL256_e64
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x64x64xf32>, [[ARG1:%.+]]: tensor<1x1x256x64xf32>, [[ARG2:%.+]]: tensor<1xf32>, [[ARG3:%.+]]: tensor<1x1x64x256xf32>, [[ARG4:%.+]]: tensor<1x1x256x256xf32>)
func.func @NoFuseSDPA_IllegalConfig_tSL64_sSL256_e64(%arg0: tensor<1x1x64x64xf32>, %arg1: tensor<1x1x256x64xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1x1x64x256xf32>, %arg4: tensor<1x1x256x256xf32>) -> tensor<1x1x64x256xf32> {
  %0 = IE.Reshape(%arg0) {shape_value = [64, 64]} : tensor<1x1x64x64xf32> -> tensor<64x64xf32>
  %1 = IE.Reshape(%arg1) {shape_value = [256, 64]} : tensor<1x1x256x64xf32> -> tensor<256x64xf32>
  %2 = IE.FullyConnected(%0, %1) : tensor<64x64xf32>, tensor<256x64xf32> -> tensor<64x256xf32>
  %3 = IE.Reshape(%2) {shape_value = [1, 1, 64, 256]} : tensor<64x256xf32> -> tensor<1x1x64x256xf32>
  %4 = IE.Multiply(%3, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x256xf32>, tensor<1xf32> -> tensor<1x1x64x256xf32>
  %5 = IE.Add(%4, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x256xf32>, tensor<1x1x64x256xf32> -> tensor<1x1x64x256xf32>
  %6 = IE.SoftMax(%5) {axisInd = 3 : i64} : tensor<1x1x64x256xf32> -> tensor<1x1x64x256xf32>
  %7 = IE.Reshape(%6) {shape_value = [64, 256]} : tensor<1x1x64x256xf32> -> tensor<64x256xf32>
  %8 = IE.Reshape(%arg4) {shape_value = [256, 256]} : tensor<1x1x256x256xf32> -> tensor<256x256xf32>
  %9 = IE.FullyConnected(%7, %8) : tensor<64x256xf32>, tensor<256x256xf32> -> tensor<64x256xf32>
  %10 = IE.Reshape(%9) {shape_value = [1, 1, 64, 256]} : tensor<64x256xf32> -> tensor<1x1x64x256xf32>

  return %10 : tensor<1x1x64x256xf32>

    // CHECK-NOT: IE.SDPA
}
