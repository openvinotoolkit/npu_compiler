//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-Affine-to-LLVM --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// Since we don't have callers of generated_0 we can safely append llvm.noalias attributes.
module @SingleCosLayer {
  module @VPU.SW {
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}

    func.func @generated_0(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 1 {
            affine.for %arg5 = 0 to 1000 {
              %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
              %1 = math.cos %0 : f16
              affine.store %1, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
            }
          }
        }
      }
      return %arg1 : memref<1x1x1x1000xf16>
    }
    // CHECK: module @SingleCosLayer
    // CHECK: module @VPU.SW
    // CHECK: llvm.func @generated_0({{.+}}: !llvm.ptr, [[IN_PTR:%.+]]: !llvm.ptr {llvm.noalias}, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: !llvm.ptr, [[OUT_PTR:%.+]]: !llvm.ptr {llvm.noalias}, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32)
    // CHECK: [[IN_ELEMENT_ADDR:%.+]] = llvm.getelementptr inbounds|nuw [[IN_PTR]][{{.+}}] : (!llvm.ptr, i32) -> !llvm.ptr, f16
    // CHECK: [[INPUT:%.+]] = llvm.load [[IN_ELEMENT_ADDR]] : !llvm.ptr -> f16
    // CHECK: [[COS_RES:%.+]] = llvm.intr.cos([[INPUT]])
    // CHECK: [[OUT_ELEMENT_ADDR:%.+]] = llvm.getelementptr inbounds|nuw [[OUT_PTR]][
    // CHECK: llvm.store [[COS_RES]], [[OUT_ELEMENT_ADDR]]
  }
}

// -----
// The source and destination are separate for the SW kernel invocation so the lowering
// to LLVM should append llvm.noalias on the source and destination pointers.

module @SingleCosLayerNoAlias {
  module @VPU.SW {
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}

    func.func @generated_0(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 1 {
            affine.for %arg5 = 0 to 1000 {
              %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
              %1 = math.cos %0 : f16
              affine.store %1, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
            }
          }
        }
      }
      return %arg1 : memref<1x1x1x1000xf16>
    }
    // CHECK: module @SingleCosLayerNoAlias
    // CHECK: module @VPU.SW
    // CHECK: llvm.func @generated_0({{.+}}: !llvm.ptr, [[IN_PTR:%.+]]: !llvm.ptr {llvm.noalias}, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: !llvm.ptr, [[OUT_PTR:%.+]]: !llvm.ptr {llvm.noalias}, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32)
    // CHECK: [[IN_ELEMENT_ADDR:%.+]] = llvm.getelementptr inbounds|nuw [[IN_PTR]][{{.+}}] : (!llvm.ptr, i32) -> !llvm.ptr, f16
    // CHECK: [[INPUT:%.+]] = llvm.load [[IN_ELEMENT_ADDR]] : !llvm.ptr -> f16
    // CHECK: [[COS_RES:%.+]] = llvm.intr.cos([[INPUT]])
    // CHECK: [[OUT_ELEMENT_ADDR:%.+]] = llvm.getelementptr inbounds|nuw [[OUT_PTR]][
    // CHECK: llvm.store [[COS_RES]], [[OUT_ELEMENT_ADDR]]
  }
  func.func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %alloc = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%alloc : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {isJitCompiled, resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0 inputs(%0 as %arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%alloc_0 as %arg3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    %alloc_1 = memref.alloc() : memref<1x1x1x1000xf16>
    %1 = VPUIP.Copy inputs(%results : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%alloc_1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    %2 = VPUIP.Copy inputs(%1 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    return %2 : memref<1x1x1x1000xf16>
  }
}

// -----
// The call to generated_0 has the same source as the destination so the llvm dialect should not
// have llvm.noalias attributes.

module @SingleCosLayerInPlace {
  module @VPU.SW {
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}

    func.func @generated_0(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 1 {
            affine.for %arg5 = 0 to 1000 {
              %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
              %1 = math.cos %0 : f16
              affine.store %1, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
            }
          }
        }
      }
      return %arg1 : memref<1x1x1x1000xf16>
    }
    // CHECK: module @SingleCosLayerInPlace
    // CHECK: module @VPU.SW
    // CHECK: llvm.func @generated_0({{.+}}: !llvm.ptr, [[IN_PTR:%.+]]: !llvm.ptr, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: !llvm.ptr, [[OUT_PTR:%.+]]: !llvm.ptr, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32)
// CHECK: [[IN_ELEMENT_ADDR:%.+]] = llvm.getelementptr inbounds|nuw [[IN_PTR]][{{.+}}] : (!llvm.ptr, i32) -> !llvm.ptr, f16
    // CHECK: [[INPUT:%.+]] = llvm.load [[IN_ELEMENT_ADDR]] : !llvm.ptr -> f16
    // CHECK: [[COS_RES:%.+]] = llvm.intr.cos([[INPUT]])
    // CHECK: [[OUT_ELEMENT_ADDR:%.+]] = llvm.getelementptr inbounds|nuw [[OUT_PTR]][
    // CHECK: llvm.store [[COS_RES]], [[OUT_ELEMENT_ADDR]]
  }
  func.func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %alloc = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%alloc : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {isJitCompiled, resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0 inputs(%0 as %arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%alloc as %arg3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    %alloc_1 = memref.alloc() : memref<1x1x1x1000xf16>
    %1 = VPUIP.Copy inputs(%results : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%alloc_1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    %2 = VPUIP.Copy inputs(%1 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    return %2 : memref<1x1x1x1000xf16>
  }
}

// -----
// Two calls to generated_0, first having the same source as the destination and
// the second doesn't. We are not allowed to add noalias here.

module @SingleCosLayerTwoCallsWithInPlace {
  module @VPU.SW {
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}

    func.func @generated_0(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 1 {
            affine.for %arg5 = 0 to 1000 {
              %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
              %1 = math.cos %0 : f16
              affine.store %1, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
            }
          }
        }
      }
      return %arg1 : memref<1x1x1x1000xf16>
    }
    // CHECK: module @SingleCosLayerTwoCallsWithInPlace
    // CHECK: module @VPU.SW
    // CHECK: llvm.func @generated_0({{.+}}: !llvm.ptr, [[IN_PTR:%.+]]: !llvm.ptr, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: !llvm.ptr, [[OUT_PTR:%.+]]: !llvm.ptr, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32)
    // CHECK: [[IN_ELEMENT_ADDR:%.+]] = llvm.getelementptr inbounds|nuw [[IN_PTR]][{{.+}}] : (!llvm.ptr, i32) -> !llvm.ptr, f16
    // CHECK: [[INPUT:%.+]] = llvm.load [[IN_ELEMENT_ADDR]] : !llvm.ptr -> f16
    // CHECK: [[COS_RES:%.+]] = llvm.intr.cos([[INPUT]])
    // CHECK: [[OUT_ELEMENT_ADDR:%.+]] = llvm.getelementptr inbounds|nuw [[OUT_PTR]][
    // CHECK: llvm.store [[COS_RES]], [[OUT_ELEMENT_ADDR]]
  }
  func.func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %alloc = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%alloc : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {isJitCompiled, resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0 inputs(%0 as %arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%alloc as %arg3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    %results1 = VPUIP.SW.Kernel {isJitCompiled, resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0 inputs(%results as %arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%alloc_0 as %arg3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    %alloc_1 = memref.alloc() : memref<1x1x1x1000xf16>
    %1 = VPUIP.Copy inputs(%results1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%alloc_1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    %2 = VPUIP.Copy inputs(%1 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    return %2 : memref<1x1x1x1000xf16>
  }
}

// -----

// Two calls to generated_0, both with non-overlapping inputs and outputs.
// We should be adding the llvm.noalias in generated_0.

module @SingleCosLayerTwoCalls {
  module @VPU.SW {
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}

    func.func @generated_0(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 1 {
            affine.for %arg5 = 0 to 1000 {
              %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
              %1 = math.cos %0 : f16
              affine.store %1, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x1000xf16>
            }
          }
        }
      }
      return %arg1 : memref<1x1x1x1000xf16>
    }
    // CHECK: module @SingleCosLayerTwoCalls
    // CHECK: module @VPU.SW
    // CHECK: llvm.func @generated_0({{.+}}: !llvm.ptr, [[IN_PTR:%.+]]: !llvm.ptr {llvm.noalias}, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: !llvm.ptr, [[OUT_PTR:%.+]]: !llvm.ptr {llvm.noalias}, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32, {{.+}}: i32)
    // CHECK: [[IN_ELEMENT_ADDR:%.+]] = llvm.getelementptr inbounds|nuw [[IN_PTR]][{{.+}}] : (!llvm.ptr, i32) -> !llvm.ptr, f16
    // CHECK: [[INPUT:%.+]] = llvm.load [[IN_ELEMENT_ADDR]] : !llvm.ptr -> f16
    // CHECK: [[COS_RES:%.+]] = llvm.intr.cos([[INPUT]])
    // CHECK: [[OUT_ELEMENT_ADDR:%.+]] = llvm.getelementptr inbounds|nuw [[OUT_PTR]][
    // CHECK: llvm.store [[COS_RES]], [[OUT_ELEMENT_ADDR]]
  }
  func.func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %alloc = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%alloc : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {isJitCompiled, resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0 inputs(%0 as %arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%alloc_0 as %arg3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    %results1 = VPUIP.SW.Kernel {isJitCompiled, resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0 inputs(%results as %arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%alloc as %arg3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    %alloc_1 = memref.alloc() : memref<1x1x1x1000xf16>
    %1 = VPUIP.Copy inputs(%results1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%alloc_1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    %2 = VPUIP.Copy inputs(%1 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    return %2 : memref<1x1x1x1000xf16>
  }
}

// -----

// Constant values should be propagated in the IR even though they need to be
// encoded as function arguments. In this case, both memref offsets are known to be
// zero.

// CHECK: module @MemrefStaticVals
module @MemrefStaticVals {
  module @VPU.SW {
    // CHECK: llvm.func @generated_0(
    // CHECK-SAME: {{%.+}}: !llvm.ptr, [[ARG1:%.+]]: !llvm.ptr {llvm.noalias}, {{%.+}}: i32, {{%.+}}: i32, {{%.+}}: i32, {{%.+}}: i32, {{%.+}}: i32, {{.*}}: !llvm.ptr, [[ARG8:%.+]]: !llvm.ptr {llvm.noalias}, {{%.+}}: i32, {{%.+}}: i32, {{%.+}}: i32, {{%.+}}: i32, {{%.+}}: i32)
    func.func @generated_0(%arg0: memref<10x?xf32>, %arg1: memref<10x?xf32>) {
      memref.copy %arg0, %arg1 : memref<10x?xf32> to memref<10x?xf32>
      return
      // CHECK: "llvm.intr.memcpy"([[ARG8]], [[ARG1]], {{%.+}})
      // CHECK-NEXT: return
    }
  }
}

// -----

// Index arguments are passed in memory as i64.

// CHECK: module @IndexArgs
module @IndexArgs {
  module @VPU.SW {
    // CHECK: llvm.func @generated_0(
    // CHECK-SAME: {{%.+}}: !llvm.ptr, [[ARG1:%.+]]: !llvm.ptr {llvm.noalias}, {{%.+}}: i32, {{%.+}}: i32, {{%.+}}: i32, {{%.+}}: i32, {{%.+}}: i32, [[ARG7:%.+]]: i64)
    func.func @generated_0(%arg0: memref<10x?xi32>, %arg1 : index) {
      %c1 = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      affine.store %c1, %arg0[%c0, %arg1] : memref<10x?xi32>
      return

// CHECK:      [[TRUNC:%.+]] = llvm.trunc [[ARG7]] : i64 to i32
// CHECK:      [[ADD:%.+]] = llvm.add {{%.+}}, [[TRUNC]] overflow<nsw, nuw> : i32
// CHECK:      [[GEP:%.+]] = llvm.getelementptr inbounds|nuw [[ARG1]]{{\[}}[[ADD]]{{\]}} : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:      llvm.store {{%.+}}, [[GEP]] : i32, !llvm.ptr
    }
  }
}
