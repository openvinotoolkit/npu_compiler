//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
// RUN: vpux-opt %s --split-input-file --init-compiler="vpu-arch=%arch%" \
// RUN:     --lower-math-to-shave-intrinsics | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

module @Foo {
  module @VPU.SW {
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK: func.func @generated_0(
// CHECK: func.call @llvm.shave.sau.tanh.f16.l.r({{.*}}) : (f16) -> f16
// CHECK: func.func private @llvm.shave.sau.tanh.f16.l.r(f16) -> f16 attributes {ShaveCodeGenIntrinsic}
    func.func @generated_0(%arg0: memref<1x1x128x32xf16>, %arg1: memref<1x1x128x32xf16>) -> memref<1x1x128x32xf16> {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 128 {
            affine.for %arg5 = 0 to 32 {
              %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x1x128x32xf16>
              %1 = math.tanh %0 fastmath<afn> : f16
              affine.store %1, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x128x32xf16>
            }
          }
        }
      }
      return %arg1 : memref<1x1x128x32xf16>
    }
  }
}

// -----

module @Bar {
  module @VPU.SW {
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK: func.func @generated_0(
// CHECK: func.call @tanhf({{.*}}) : (f32) -> f32
// CHECK: func.func private @tanhf(f32) -> f32 attributes {ShaveCodeGenIntrinsic}
    func.func @generated_0(%arg0: memref<1x1x128x32xf32>, %arg1: memref<1x1x128x32xf32>) -> memref<1x1x128x32xf32> {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 128 {
            affine.for %arg5 = 0 to 32 {
              %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x1x128x32xf32>
              %1 = math.tanh %0 fastmath<afn> : f32
              affine.store %1, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x128x32xf32>
            }
          }
        }
      }
      return %arg1 : memref<1x1x128x32xf32>
    }
  }
}

// -----

module @atanf16 {
  module @VPU.SW {
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK: func.func @generated_0(
// CHECK: func.call @llvm.shave.sau.atn.f16.l.r({{.*}}) : (f16) -> f16
// CHECK: arith.select
// CHECK: func.func private @llvm.shave.sau.atn.f16.l.r(f16) -> f16 attributes {ShaveCodeGenIntrinsic}
    func.func @generated_0(%arg0: memref<1x1x128x32xf16>, %arg1: memref<1x1x128x32xf16>) -> memref<1x1x128x32xf16> {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 128 {
            affine.for %arg5 = 0 to 32 {
              %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x1x128x32xf16>
              %1 = math.atan %0 fastmath<afn> : f16
              affine.store %1, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x128x32xf16>
            }
          }
        }
      }
      return %arg1 : memref<1x1x128x32xf16>
    }
  }
}

// -----

module @atanf32 {
  module @VPU.SW {
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK: func.func @generated_0(
// CHECK: func.call @atanf({{.*}}) : (f32) -> f32
// CHECK-NEXT: affine.store
// CHECK: func.func private @atanf(f32) -> f32 attributes {ShaveCodeGenIntrinsic}
    func.func @generated_0(%arg0: memref<1x1x128x32xf32>, %arg1: memref<1x1x128x32xf32>) -> memref<1x1x128x32xf32> {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 128 {
            affine.for %arg5 = 0 to 32 {
              %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x1x128x32xf32>
              %1 = math.atan %0 fastmath<afn> : f32
              affine.store %1, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x128x32xf32>
            }
          }
        }
      }
      return %arg1 : memref<1x1x128x32xf32>
    }
  }
}
