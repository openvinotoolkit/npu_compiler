//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --shave-stack-allocation %s -o - | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// Large allocations should have been removed before the shave stack allocation pass.
// CHECK-NOT: memref.alloca()

module @InvalidPromoteToAlloca_0 {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x1x1xf16>) -> memref<1x512x1x1xf16> {
      %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<1x17xf32>
      return %arg1 : memref<1x512x1x1xf16>
    }
  }
}

// -----

module @InvalidPromoteToAlloca_1 {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x1x1xf16>) -> memref<1x512x1x1xf16> {
      %alloc2 = memref.alloc() {alignment = 64 : i64} : memref<1x32xf32>
      return %arg1 : memref<1x512x1x1xf16>
    }
  }
}

// -----

module @InvalidPromoteToAlloca_2 {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x1x1xf16>) -> memref<1x512x1x1xf16> {
      %alloc2 = memref.alloc() {alignment = 64 : i64} : memref<1x33xf16>
      return %arg1 : memref<1x512x1x1xf16>
    }
  }
}

// -----

module @InvalidPromoteToAlloca_3 {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x1x1xf16>) -> memref<1x512x1x1xf16> {
      %alloc2 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf16>
      return %arg1 : memref<1x512x1x1xf16>
    }
  }
}
