//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --split-real-dft-ops %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @RDFT
// CHECK-SAME:  (%arg0: tensor<10x4x2xf32>) -> tensor<10x3x2x2xf32>
func.func @RDFT(%arg0: tensor<10x4x2xf32>) -> tensor<10x3x2x2xf32> {
  %0 = VPU.RDFT(%arg0) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x3x2x2xf32>
  return %0 : tensor<10x3x2x2xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<232xf32> = dense
  // CHECK: [[RDFTUncut:%.+]] = VPU.RDFTUncut(%arg0, [[CST]]) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32>, tensor<232xf32> -> tensor<10x4x2x2xf32>
  // CHECK: [[OUTPUT:%.+]] = VPU.Slice [[RDFTUncut]] [0, 0, 0, 0] [10, 3, 2, 2] : tensor<10x4x2x2xf32> to tensor<10x3x2x2xf32>
  // CHECK: return [[OUTPUT]] : tensor<10x3x2x2xf32>
}

// -----

// CHECK-LABEL: @IRDFT
// CHECK-SAME: (%arg0: tensor<10x4x2xf32>) -> tensor<10x6xf32>
func.func @IRDFT(%arg0: tensor<10x4x2xf32>) -> tensor<10x6xf32> {
  %0 = VPU.IRDFT(%arg0) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x6xf32>
  return %0 : tensor<10x6xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<200xf32> = dense
  // CHECK: [[IDFT:%.+]] = VPU.IDFT(%arg0, [[CST]]) {axes_attr = [0], signal_size_attr = [-1]} : tensor<10x4x2xf32>, tensor<200xf32> -> tensor<10x4x2xf32>
  // CHECK: [[CST_0:%.+]] = const.Declare tensor<72xf32> = dense
  // CHECK: [[OUTPUT:%.+]] = VPU.IRDFTLastAxis([[IDFT]], [[CST_0]]) {axes_attr = [1], signal_size_attr = [-1]} : tensor<10x4x2xf32>, tensor<72xf32> -> tensor<10x6xf32>
  // CHECK: return [[OUTPUT]] : tensor<10x6xf32>
}

// -----

// CHECK-LABEL: @IRDFTOneAxis
// CHECK-SAME: (%arg0: tensor<10x4x2xf32>) -> tensor<10x6xf32>
func.func @IRDFTOneAxis(%arg0: tensor<10x4x2xf32>) -> tensor<10x6xf32> {
  %0 = VPU.IRDFT(%arg0) {axes_attr = [1], signal_size_attr = [-1]} : tensor<10x4x2xf32> -> tensor<10x6xf32>
  return %0 : tensor<10x6xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<72xf32> = dense
  // CHECK: [[OUTPUT:%.+]] = VPU.IRDFTLastAxis(%arg0, [[CST]]) {axes_attr = [1], signal_size_attr = [-1]} : tensor<10x4x2xf32>, tensor<72xf32> -> tensor<10x6xf32>
  // CHECK: return [[OUTPUT]] : tensor<10x6xf32>
}

// -----

// CHECK-LABEL: @DFT
// CHECK-SAME:  (%arg0: tensor<10x4x2xf32>) -> tensor<10x4x2xf32>
func.func @DFT(%arg0: tensor<10x4x2xf32>) -> tensor<10x4x2xf32> {
  %0 = VPU.DFT(%arg0) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x4x2xf32>
  return %0 : tensor<10x4x2xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<232xf32> = dense
  // CHECK: [[OUTPUT:%.+]] = VPU.DFT(%arg0, [[CST]]) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32>, tensor<232xf32> -> tensor<10x4x2xf32>
  // CHECK: return [[OUTPUT]] : tensor<10x4x2xf32>
}

// -----

// CHECK-LABEL: @IDFT
// CHECK-SAME:  (%arg0: tensor<10x4x2xf32>) -> tensor<10x4x2xf32>
func.func @IDFT(%arg0: tensor<10x4x2xf32>) -> tensor<10x4x2xf32> {
  %0 = VPU.IDFT(%arg0) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x4x2xf32>
  return %0 : tensor<10x4x2xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<232xf32> = dense
  // CHECK: [[OUTPUT:%.+]] = VPU.IDFT(%arg0, [[CST]]) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32>, tensor<232xf32> -> tensor<10x4x2xf32>
  // CHECK: return [[OUTPUT]] : tensor<10x4x2xf32>
}
