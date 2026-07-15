//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --fuse-ops-to-matmul="enable-grouped-matmul=false" %s | FileCheck %s
// REQUIRES: platform-NPU3720

// CHECK-LABEL: @NotConvertBroadcastMultiplyReduceSumToMatMulOnNPU37XX
// CHECK-SAME:      [[INPUT_A:%.+]]: tensor<6x256x1x8192xf16>
// CHECK-SAME:      [[INPUT_B:%.+]]: tensor<6x1x256x8192xf16>
func.func @NotConvertBroadcastMultiplyReduceSumToMatMulOnNPU37XX(%arg0: tensor<6x256x1x8192xf16>, %arg1: tensor<6x1x256x8192xf16>) -> tensor<6x256x256x1xf16> {
  %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x256x1x8192xf16>, tensor<6x1x256x8192xf16> -> tensor<6x256x256x8192xf16>
  %red = IE.ReduceSum(%mul) {axes_value = [3], keep_dims} : tensor<6x256x256x8192xf16> -> tensor<6x256x256x1xf16>
  return %red : tensor<6x256x256x1xf16>

  // CHECK-NOT:   IE.MatMul
  // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_A]], [[INPUT_B]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:       [[REDUCESUM:%.+]] = IE.ReduceSum([[MULTIPLY]]) {axes_value = [3], keep_dims}
  // CHECK:       return [[REDUCESUM]] : tensor<6x256x256x1xf16>
}
