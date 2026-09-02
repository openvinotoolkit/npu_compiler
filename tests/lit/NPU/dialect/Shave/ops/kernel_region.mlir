//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: func.func @KernelRegionSingleInputOutput
// CHECK-SAME:      ([[ARG:%.+]]: f32)
func.func @KernelRegionSingleInputOutput(%arg: f32) -> f32 {
  // CHECK: [[OUT:%.+]] = Shave.KernelRegion([[ARG]] : f32) -> f32 {
  // CHECK-NEXT: ^bb0([[BARG:%.+]]: f32):
  // CHECK-NEXT:   Shave.KernelRegion.Yield [[BARG]] : f32
  // CHECK-NEXT: }
  %out = Shave.KernelRegion(%arg : f32) -> f32 {
  ^bb0(%barg: f32):
    Shave.KernelRegion.Yield %barg : f32
  }
  return %out : f32
}

// -----

// CHECK-LABEL: func.func @KernelRegionMultipleInputsOutputs
// CHECK-SAME:      ([[A:%.+]]: i32, [[B:%.+]]: f32)
func.func @KernelRegionMultipleInputsOutputs(%a: i32, %b: f32) -> (i32, f32) {
  // CHECK: [[R:%.+]]:2 = Shave.KernelRegion([[A]], [[B]] : i32, f32) -> i32, f32 {
  // CHECK-NEXT: ^bb0([[BA:%.+]]: i32, [[BB:%.+]]: f32):
  // CHECK-NEXT:   Shave.KernelRegion.Yield [[BA]], [[BB]] : i32, f32
  // CHECK-NEXT: }
  %r0, %r1 = Shave.KernelRegion(%a, %b : i32, f32) -> i32, f32 {
  ^bb0(%ba: i32, %bb: f32):
    Shave.KernelRegion.Yield %ba, %bb : i32, f32
  }
  return %r0, %r1 : i32, f32
}

// -----

// CHECK-LABEL: func.func @KernelRegionNoInputsNoOutputs
func.func @KernelRegionNoInputsNoOutputs() {
  // CHECK: Shave.KernelRegion {
  // CHECK-NEXT:   Shave.KernelRegion.Yield
  // CHECK-NEXT: }
  Shave.KernelRegion {
  ^bb0:
    Shave.KernelRegion.Yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @KernelRegionTensorTypes
// CHECK-SAME:      ([[T:%.+]]: tensor<4xf32>)
func.func @KernelRegionTensorTypes(%t: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: [[RES:%.+]] = Shave.KernelRegion([[T]] : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: ^bb0([[BT:%.+]]: tensor<4xf32>):
  // CHECK-NEXT:   Shave.KernelRegion.Yield [[BT]] : tensor<4xf32>
  // CHECK-NEXT: }
  %res = Shave.KernelRegion(%t : tensor<4xf32>) -> tensor<4xf32> {
  ^bb0(%bt: tensor<4xf32>):
    Shave.KernelRegion.Yield %bt : tensor<4xf32>
  }
  return %res : tensor<4xf32>
}
