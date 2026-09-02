//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --shave-stack-allocation %s -o - | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

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

// -----

// The large memref<1x1x1x1000xf16> should be promoted to a scratch buffer and removed,
// while the small one (memref<1x1x1x1xf16>) gets promoted to an alloca.

// CHECK: module @ScratchBuffer
module @ScratchBuffer {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<1x1x16x1000xf16>, %arg1: memref<1x1x16x1000xf16>) {
      %c1000 = arith.constant 1000 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      scf.for %arg2 = %c0 to %c16 step %c1 {
          %subview_1 = memref.subview %arg0[0, 0, %arg2, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : memref<1x1x16x1000xf16> to memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
          %subview_2 = memref.subview %arg1[0, 0, %arg2, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : memref<1x1x16x1000xf16> to memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
          %subview_3 = memref.subview %arg0[0, 0, %arg2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x16x1000xf16> to memref<1x1x1x1xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
          %subview_4 = memref.subview %arg1[0, 0, %arg2, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x16x1000xf16> to memref<1x1x1x1xf16, strided<[16000, 16000, 1000, 1], offset: ?>>

          %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1xf16>
          memref.copy %subview_3, %alloc : memref<1x1x1x1xf16, strided<[16000, 16000, 1000, 1], offset: ?>> to memref<1x1x1x1xf16>
          memref.copy %alloc, %subview_4: memref<1x1x1x1xf16> to memref<1x1x1x1xf16, strided<[16000, 16000, 1000, 1], offset: ?>>

          %alloc2 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1000xf16>
          memref.copy %subview_1, %alloc2 : memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>> to memref<1x1x1x1000xf16>
          memref.copy %alloc2, %subview_2: memref<1x1x1x1000xf16> to memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
      }

// CHECK: func.func @generated_0([[ARG0:%.+]]: memref<1x1x16x1000xf16>, [[ARGSCRATCH:%.+]]: memref<1x1x1x1000xf16>, [[ARG1:%.+]]: memref<1x1x16x1000xf16>, {{%.+}}: memref<1x1x1x1000xf16>)
// CHECK-NOT: memref.alloc() {{.+}} : memref<1x1x1x1000xf16>
// CHECK-DAG: [[C1000:%.+]] = arith.constant 1000 : index
// CHECK-DAG: [[C16:%.+]] = arith.constant 16 : index
// CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG: [[ALLOCA:%.+]] = memref.alloca() {alignment = 64 : i64} : memref<1x1x1x1xf16>
// CHECK: scf.for [[ARG3:%.+]] = [[C0]] to [[C16]] step [[C1]] {
// CHECK-NEXT:  [[S0:%.+]] = memref.subview [[ARG0]]
// CHECK-SAME:     to memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>
// CHECK-NEXT:  [[S1:%.+]] = memref.subview [[ARG1]]
// CHECK-SAME:     to memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>
// CHECK-NEXT:  [[S2:%.+]] = memref.subview [[ARG0]]
// CHECK-SAME:     to memref<1x1x1x1xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
// CHECK-NEXT:  [[S3:%.+]] = memref.subview [[ARG1]]
// CHECK-SAME:     to memref<1x1x1x1xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
// CHECK-NEXT:  memref.copy [[S2]], [[ALLOCA]]
// CHECK-NEXT:  memref.copy [[ALLOCA]], [[S3]]
// CHECK-NEXT:  memref.copy [[S0]], [[ARGSCRATCH]]
// CHECK-NEXT:  memref.copy [[ARGSCRATCH]], [[S1]]
      return
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x1000xf16>
  }
  func.func @main(%arg0: tensor<1x1x16x1000xf16>) -> tensor<1x1x16x1000xf16> {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x1x16x1000xf16>) @VPU.SW::@generated_0 -> tensor<1x1x16x1000xf16>
    return %0 : tensor<1x1x16x1000xf16>

// CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x1x16x1000xf16>) -> tensor<1x1x16x1000xf16>
// CHECK-NEXT:    [[SCRATCH:%.+]] = VPU.Empty : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[RES:%.+]] = VPU.GenericSwLayer([[ARG0]] : tensor<1x1x16x1000xf16>) scratch([[SCRATCH]] : tensor<1x1x1x1000xf16>) @VPU.SW::@generated_0 -> tensor<1x1x16x1000xf16>
// CHECK-NEXT:    return [[RES]] : tensor<1x1x16x1000xf16>
  }
}

// -----

// If we need a scratch buffer for a SW kernel then all callsites need to be updated.

// CHECK: module @MultipleCallsiteWithScratchBuffer
module @MultipleCallsiteWithScratchBuffer {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<1x1x16x1000xf16>, %arg1: memref<1x1x16x1000xf16>) {
      %c1000 = arith.constant 1000 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      scf.for %arg2 = %c0 to %c16 step %c1 {
          %subview_1 = memref.subview %arg0[0, 0, %arg2, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : memref<1x1x16x1000xf16> to memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
          %subview_2 = memref.subview %arg1[0, 0, %arg2, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : memref<1x1x16x1000xf16> to memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1000xf16>
          memref.copy %subview_1, %alloc : memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>> to memref<1x1x1x1000xf16>
          memref.copy %alloc, %subview_2: memref<1x1x1x1000xf16> to memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
      }

// CHECK: func.func @generated_0([[ARG0:%.+]]: memref<1x1x16x1000xf16>, [[ARGSCRATCH:%.+]]: memref<1x1x1x1000xf16>, [[ARG1:%.+]]: memref<1x1x16x1000xf16>, {{%.+}}: memref<1x1x1x1000xf16>)
// CHECK-NOT: memref.alloc() {{.+}} : memref<1x1x1x1000xf16>
// CHECK-DAG: [[C1000:%.+]] = arith.constant 1000 : index
// CHECK-DAG: [[C16:%.+]] = arith.constant 16 : index
// CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK: scf.for [[ARG3:%.+]] = [[C0]] to [[C16]] step [[C1]] {
// CHECK-NEXT:  [[S0:%.+]] = memref.subview [[ARG0]]
// CHECK-SAME:     to memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>
// CHECK-NEXT:  [[S1:%.+]] = memref.subview [[ARG1]]
// CHECK-SAME:     to memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>
// CHECK-NEXT:  memref.copy [[S0]], [[ARGSCRATCH]]
// CHECK-NEXT:  memref.copy [[ARGSCRATCH]], [[S1]]
      return
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x1000xf16>
  }
  func.func @main(%arg0: tensor<1x1x16x1000xf16>) -> tensor<1x1x16x1000xf16> {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x1x16x1000xf16>) @VPU.SW::@generated_0 -> tensor<1x1x16x1000xf16>
    %1 = VPU.GenericSwLayer(%0 : tensor<1x1x16x1000xf16>) @VPU.SW::@generated_0 -> tensor<1x1x16x1000xf16>
    return %1 : tensor<1x1x16x1000xf16>

// CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x1x16x1000xf16>) -> tensor<1x1x16x1000xf16>
// CHECK-NEXT:    [[SCRATCH0:%.+]] = VPU.Empty : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[RES0:%.+]] = VPU.GenericSwLayer([[ARG0]] : tensor<1x1x16x1000xf16>) scratch([[SCRATCH0]] : tensor<1x1x1x1000xf16>) @VPU.SW::@generated_0 -> tensor<1x1x16x1000xf16>
// CHECK-NEXT:    [[SCRATCH1:%.+]] = VPU.Empty : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[RES1:%.+]] = VPU.GenericSwLayer([[RES0]] : tensor<1x1x16x1000xf16>) scratch([[SCRATCH1]] : tensor<1x1x1x1000xf16>) @VPU.SW::@generated_0 -> tensor<1x1x16x1000xf16>
// CHECK-NEXT:    return [[RES1]] : tensor<1x1x16x1000xf16>
  }
}

// -----

// CHECK: module @ScratchBufferWithIndexArg
module @ScratchBufferWithIndexArg {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<1x1x16x?xf16>, %arg1: memref<1x1x16x?xf16>, %argidx0: index, %argidx1: index) attributes {
     kernelInfo = #VPU.KernelInfo<
       tilingInfoFunc = @generated_info_0,
       tilingAxes = [3],
       numSlicedInputs = 1 : i64
     >
   } {
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      scf.for %arg3 = %c0 to %c16 step %c1 {
          %subview_1 = memref.subview %arg0[0, 0, 0, %argidx1] [1, 1, 1, %argidx0] [1, 1, 1, 1] : memref<1x1x16x?xf16> to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>>
          %subview_2 = memref.subview %arg1[0, 0, 0, %argidx1] [1, 1, 1, %argidx0] [1, 1, 1, 1] : memref<1x1x16x?xf16> to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>>

          %alloc = memref.alloc(%argidx0) {alignment = 64 : i64} : memref<1x1x1x?xf16>
          memref.copy %subview_1, %alloc : memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x1x1x?xf16>
          memref.copy %alloc, %subview_2: memref<1x1x1x?xf16> to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>>
      }

// CHECK: func.func @generated_0([[ARG0:%.+]]: memref<1x1x16x?xf16>, [[ARGSCRATCH:%.+]]: memref<1x1x1x?xf16>, [[ARG1:%.+]]: memref<1x1x16x?xf16>, {{%.+}}: memref<1x1x1x?xf16>, [[SZ:%.+]]: index, [[IDX:%.+]]: index)
// CHECK-SAME: kernelInfo = #VPU.KernelInfo<
// CHECK-SAME:   tilingInfoFunc = @generated_info_0,
// CHECK-SAME:   tilingAxes = [3],
// CHECK-SAME:   numSlicedInputs = 1 : i64
// CHECK-NOT: memref.alloc() {{.+}} : memref<1x1x1x?xf16>
// CHECK-DAG: [[C16:%.+]] = arith.constant 16 : index
// CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK: scf.for [[ARG3:%.+]] = [[C0]] to [[C16]] step [[C1]] {
// CHECK-NEXT:  [[S0:%.+]] = memref.subview [[ARG0]]
// CHECK-SAME:     to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>
// CHECK-NEXT:  [[S1:%.+]] = memref.subview [[ARG1]]
// CHECK-SAME:     to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>
// CHECK-NEXT:  memref.copy [[S0]], [[ARGSCRATCH]]
// CHECK-NEXT:  memref.copy [[ARGSCRATCH]], [[S1]]
      return
    }
    func.func @generated_info_0(%arg0: index, %arg1: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      return
          %c1, %c1, %c16, %arg0,
          %c0, %c0, %c0, %arg1 : index, index, index, index, index, index, index, index
    }

// CHECK: func.func @generated_info_0([[ARG0:%.+]]: index, [[ARG1:%.+]]: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
// CHECK:      [[C1:%.+]] = arith.constant 1 : index
// CHECK:      [[C16:%.+]] = arith.constant 16 : index
// CHECK:      [[C0:%.+]] = arith.constant 0 : index
// CHECK:      [[C00:%.+]] = arith.constant 0 : index
// CHECK:      [[C11:%.+]] = arith.constant 1 : index
// CHECK:      [[C12:%.+]] = arith.constant 1 : index
// CHECK:      [[C13:%.+]] = arith.constant 1 : index
// CHECK:      return [[C1]], [[C1]], [[C16]], [[ARG0]], [[C0]], [[C0]], [[C0]], [[ARG1]], [[C11]], [[C12]], [[C13]], [[ARG0]], [[C00]], [[C00]], [[C00]], [[ARG1]]
// CHECK-SAME:   : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x16x1000xf16>) -> tensor<1x1x16x1000xf16> {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x1x16x1000xf16>) @VPU.SW::@generated_0
        tiling(sizes = [1000], offsets = [0]) -> tensor<1x1x16x1000xf16>
    return %0 : tensor<1x1x16x1000xf16>


// CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x1x16x1000xf16>) -> tensor<1x1x16x1000xf16>
// CHECK:    [[EMPTY:%.+]] = VPU.Empty : tensor<1x1x1x1000xf16>
// CHECK:    [[LAYER:%.+]] = VPU.GenericSwLayer([[ARG0]] : tensor<1x1x16x1000xf16>)
// CHECK-SAME: scratch([[EMPTY]] : tensor<1x1x1x1000xf16>)
// CHECK-SAME: @VPU.SW::@generated_0
// CHECK-SAME: tiling(sizes = [1000], offsets = [0])
// CHECK-SAME: -> tensor<1x1x16x1000xf16>
// CHECK:    return [[LAYER]] : tensor<1x1x16x1000xf16>
  }
}

// -----

// Two distinct kernels share a single tiling info function and both need a scratch buffer.
// The info function must be cloned so each kernel gets its own modified copy.

// CHECK: module @SharedTilingInfoFuncWithScratch
module @SharedTilingInfoFuncWithScratch {
  module @VPU.SW {
// CHECK:      func.func @generated_0
// CHECK-SAME:     ({{%.+}}: memref<1x1x16x?xf16>, {{%.+}}: memref<1x1x1x?xf16>, {{%.+}}: memref<1x1x16x?xf16>, {{%.+}}: memref<1x1x1x?xf16>, {{%.+}}: index, {{%.+}}: index)
// CHECK-SAME:     tilingInfoFunc = [[INFO_CLONE:@generated_info_0_[0-9]+]]
    func.func @generated_0(%arg0: memref<1x1x16x?xf16>, %arg1: memref<1x1x16x?xf16>, %argidx0: index, %argidx1: index) attributes {
     kernelInfo = #VPU.KernelInfo<
       tilingInfoFunc = @generated_info_0,
       tilingAxes = [3],
       numSlicedInputs = 1 : i64
     >
   } {
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      scf.for %arg3 = %c0 to %c16 step %c1 {
          %subview_1 = memref.subview %arg0[0, 0, 0, %argidx1] [1, 1, 1, %argidx0] [1, 1, 1, 1] : memref<1x1x16x?xf16> to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>>
          %subview_2 = memref.subview %arg1[0, 0, 0, %argidx1] [1, 1, 1, %argidx0] [1, 1, 1, 1] : memref<1x1x16x?xf16> to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>>

          %alloc = memref.alloc(%argidx0) {alignment = 64 : i64} : memref<1x1x1x?xf16>
          memref.copy %subview_1, %alloc : memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x1x1x?xf16>
          memref.copy %alloc, %subview_2: memref<1x1x1x?xf16> to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>>
      }
      return
    }

// CHECK:      func.func @generated_1
// CHECK-SAME:     tilingInfoFunc = @generated_info_0,
    func.func @generated_1(%arg0: memref<1x1x16x?xf16>, %arg1: memref<1x1x16x?xf16>, %argidx0: index, %argidx1: index) attributes {
     kernelInfo = #VPU.KernelInfo<
       tilingInfoFunc = @generated_info_0,
       tilingAxes = [3],
       numSlicedInputs = 1 : i64
     >
   } {
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      scf.for %arg3 = %c0 to %c16 step %c1 {
          %subview_1 = memref.subview %arg0[0, 0, 0, %argidx1] [1, 1, 1, %argidx0] [1, 1, 1, 1] : memref<1x1x16x?xf16> to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>>
          %subview_2 = memref.subview %arg1[0, 0, 0, %argidx1] [1, 1, 1, %argidx0] [1, 1, 1, 1] : memref<1x1x16x?xf16> to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>>

          %alloc = memref.alloc(%argidx0) {alignment = 64 : i64} : memref<1x1x1x?xf16>
          memref.copy %subview_1, %alloc : memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x1x1x?xf16>
          memref.copy %alloc, %subview_2: memref<1x1x1x?xf16> to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>>
      }
      return
    }

// CHECK:      func.func @generated_info_0([[INFO_SZ:%.+]]: index, [[INFO_OFF:%.+]]: index)
// CHECK-SAME:     -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index)
// CHECK:        return [[C1:%.+]], [[C1]], [[C16:%.+]], [[INFO_SZ]], [[C0:%.+]], [[C0]], [[C0]], [[INFO_OFF]], {{%.+}}, {{%.+}}, {{%.+}}, [[INFO_SZ]], {{%.+}}, {{%.+}}, {{%.+}}, [[INFO_OFF]]
    func.func @generated_info_0(%arg0: index, %arg1: index) -> (index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      return
          %c1, %c1, %c16, %arg0,
          %c0, %c0, %c0, %arg1 : index, index, index, index, index, index, index, index
    }

// CHECK:      func.func [[INFO_CLONE]]({{%.+}}: index, {{%.+}}: index)
// CHECK-SAME:     -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index)
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x16x1000xf16>) -> tensor<1x1x16x1000xf16> {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x1x16x1000xf16>) @VPU.SW::@generated_0
        tiling(sizes = [1000], offsets = [0]) -> tensor<1x1x16x1000xf16>
    %1 = VPU.GenericSwLayer(%0 : tensor<1x1x16x1000xf16>) @VPU.SW::@generated_1
        tiling(sizes = [1000], offsets = [0]) -> tensor<1x1x16x1000xf16>
    return %1 : tensor<1x1x16x1000xf16>

// CHECK:      func.func @main([[MAIN_ARG0:%.+]]: tensor<1x1x16x1000xf16>)
// CHECK:        [[EMPTY0:%.+]] = VPU.Empty : tensor<1x1x1x1000xf16>
// CHECK:        [[RES0:%.+]] = VPU.GenericSwLayer([[MAIN_ARG0]] : tensor<1x1x16x1000xf16>) scratch([[EMPTY0]] : tensor<1x1x1x1000xf16>) @VPU.SW::@generated_0 tiling(sizes = [1000], offsets = [0]) -> tensor<1x1x16x1000xf16>
// CHECK:        [[EMPTY1:%.+]] = VPU.Empty : tensor<1x1x1x1000xf16>
// CHECK:        [[RES1:%.+]] = VPU.GenericSwLayer([[RES0]] : tensor<1x1x16x1000xf16>) scratch([[EMPTY1]] : tensor<1x1x1x1000xf16>) @VPU.SW::@generated_1 tiling(sizes = [1000], offsets = [0]) -> tensor<1x1x16x1000xf16>
// CHECK:        return [[RES1]]
  }
}
