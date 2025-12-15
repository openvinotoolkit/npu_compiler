//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --shave-stack-allocation %s -o - | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

module @PromoteToAlloca {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x1x1xf16>) -> memref<1x512x1x1xf16> {
      %scalar = memref.alloc() {alignment = 64 : i64} : memref<1x1xf32>
      %large_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xf32>
      %large_alloc_f16 = memref.alloc() {alignment = 64 : i64} : memref<1x32xf16>
      return %arg1 : memref<1x512x1x1xf16>

// CHECK: module @PromoteToAlloca
// CHECK: memref.alloca() {alignment = 64 : i64} : memref<1x1xf32>
// CHECK: memref.alloca() {alignment = 64 : i64} : memref<1x16xf32>
// CHECK: memref.alloca() {alignment = 64 : i64} : memref<1x32xf16>
    }
  }
}
