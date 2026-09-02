//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --one-shot-bufferize-sw-kernels --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @SingleCosLayer {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) {
      %arg = bufferization.to_tensor %arg1 restrict writable : memref<1x1x1x1000xf16> to tensor<1x1x1x1000xf16>
      %0 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x1x1000xf16>) outs(%arg : tensor<1x1x1x1000xf16>) {
      ^bb0(%in: f16, %out: f16):
        %1 = math.cos %in : f16
        linalg.yield %1 : f16
      } -> tensor<1x1x1x1000xf16>
      bufferization.materialize_in_destination %0 in writable %arg1 : (tensor<1x1x1x1000xf16>, memref<1x1x1x1000xf16>) -> ()
      return
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x1x1x1000xf16>) @VPU.SW::@generated_0 -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>
  }
}

  // CHECK: module @VPU.SW
  // CHECK: func.func @generated
  // CHECK-SAME: [[VAR0:%.+]]: memref<1x1x1x1000xf16>,
  // CHECK-SAME: [[VAR1:%.+]]: memref<1x1x1x1000xf16>

  // CHECK: linalg.generic
  // CHECK-SAME: ins([[VAR0]]
  // CHECK-SAME: outs([[VAR1]]

  // CHECK: linalg.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: memref.copy [[VAR1]], [[VAR1]] : memref<1x1x1x1000xf16> to memref<1x1x1x1000xf16>
  // CHECK-NEXT: return

  // CHECK: func.func @main
  // CHECK-SAME: [[ARG0:%.+]]: memref<1x1x1x1000xf16>
  // CHECK-SAME: -> memref<1x1x1x1000xf16>

  // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1000xf16>

  // CHECK: [[SHAVE_RES:%.+]] = VPUIP.SW.Kernel
  // CHECK-SAME: @VPU.SW::@generated_0
  // CHECK-SAME: inputs([[ARG0]] as [[KERNEL_INPUT:%.+]]: memref<1x1x1x1000xf16>) outputs([[ALLOC]] as [[KERNEL_OUTPUT:%.+]]: memref<1x1x1x1000xf16>)
  // CHECK-SAME: -> memref<1x1x1x1000xf16>
  // CHECK-NEXT: VPUIP.SW.Kernel.run([[KERNEL_INPUT]], [[KERNEL_OUTPUT]])

  // CHECK: return [[SHAVE_RES]] : memref<1x1x1x1000xf16>

// -----

// CHECK: module @BufferizeSizesAndOffsets
module @BufferizeSizesAndOffsets {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x8x32xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x8x32xf16>
  }

  func.func private @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index)
  func.func private @generated_0(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<
      tilingInfoFunc = @generated_info_0,
      tilingAxes = [1, 2, 3],
      numSlicedInputs = 1 : i64
    >
  }

  // CHECK: func.func @main([[ARG0:%.+]]: memref<1x3x8x32xf16>) -> memref<1x3x8x32xf16>
  func.func @main(%arg0: tensor<1x3x8x32xf16>) -> tensor<1x3x8x32xf16> {
    %kern = VPU.GenericSwLayer(%arg0 : tensor<1x3x8x32xf16>) @generated_0 tiling(sizes = [3, 8, 32], offsets = [0, 0, 0]) -> tensor<1x3x8x32xf16>
    return %kern : tensor<1x3x8x32xf16>

    // CHECK:  [[ALLOC:%.+]] = memref.alloc() : memref<1x3x8x32xf16>
    // CHECK:  [[OP:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @generated_0 inputs([[ARG0]] as [[ARG1:%.+]]: memref<1x3x8x32xf16>) outputs([[ALLOC]] as [[ARG2:%.+]]: memref<1x3x8x32xf16>) on tile 0 -> memref<1x3x8x32xf16>{
    // CHECK-NEXT:  VPUIP.SW.Kernel.run {attrs = [3, 8, 32, 0, 0, 0]}([[ARG1]], [[ARG2]]) : memref<1x3x8x32xf16>, memref<1x3x8x32xf16>
    // CHECK:  return [[OP]] : memref<1x3x8x32xf16>
  }
}

// -----

// CHECK: module @BufferizeScratchBuffer
module @BufferizeScratchBuffer {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<1x1x16x1000xf16>, %argscratch: memref<1x1x1x1000xf16>, %arg1: memref<1x1x16x1000xf16>, %arg2: memref<1x1x1x1000xf16>) {
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %alloca = memref.alloca() {alignment = 64 : i64} : memref<1x1x1x1xf16>
      scf.for %arg3 = %c0 to %c16 step %c1 {
          %subview_1 = memref.subview %arg0[0, 0, %arg3, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : memref<1x1x16x1000xf16> to memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
          %subview_2 = memref.subview %arg1[0, 0, %arg3, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : memref<1x1x16x1000xf16> to memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
          %subview_3 = memref.subview %arg0[0, 0, %arg3, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x16x1000xf16> to memref<1x1x1x1xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
          %subview_4 = memref.subview %arg1[0, 0, %arg3, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x16x1000xf16> to memref<1x1x1x1xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
          memref.copy %subview_3, %alloca : memref<1x1x1x1xf16, strided<[16000, 16000, 1000, 1], offset: ?>> to memref<1x1x1x1xf16>
          memref.copy %alloca, %subview_4 : memref<1x1x1x1xf16> to memref<1x1x1x1xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
          memref.copy %subview_1, %argscratch : memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>> to memref<1x1x1x1000xf16>
          memref.copy %argscratch, %subview_2 : memref<1x1x1x1000xf16> to memref<1x1x1x1000xf16, strided<[16000, 16000, 1000, 1], offset: ?>>
      }
      return
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x1000xf16>
  }
  func.func @main(%arg0: tensor<1x1x16x1000xf16>) -> tensor<1x1x16x1000xf16> {
    %scratch = VPU.Empty : tensor<1x1x1x1000xf16>
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x1x16x1000xf16>) scratch(%scratch : tensor<1x1x1x1000xf16>) @VPU.SW::@generated_0 -> tensor<1x1x16x1000xf16>
    return %0 : tensor<1x1x16x1000xf16>

    // CHECK:  func.func @main([[ARG0:%[^:]+]]: memref<1x1x16x1000xf16>) -> memref<1x1x16x1000xf16>
    // CHECK:    [[SCRATCH:%.+]] = memref.alloc() : memref<1x1x1x1000xf16>
    // CHECK:    [[OUT:%.+]] = memref.alloc() : memref<1x1x16x1000xf16>
    // CHECK:    [[RES:%[^:]+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@generated_0
    // CHECK-SAME:  inputs([[ARG0]] as [[IN:%[^:]+]]: memref<1x1x16x1000xf16>, [[SCRATCH]] as [[IN_SCRATCH:%[^:]+]]: memref<1x1x1x1000xf16>)
    // CHECK-SAME:  outputs([[OUT]] as [[OUT_ARG:%[^:]+]]: memref<1x1x16x1000xf16>, [[SCRATCH]] as [[OUT_SCRATCH:%[^:]+]]: memref<1x1x1x1000xf16>)
    // CHECK-SAME:  on tile 0 -> (memref<1x1x16x1000xf16>, memref<1x1x1x1000xf16>)
    // CHECK-NEXT:  VPUIP.SW.Kernel.run([[IN]], [[IN_SCRATCH]], [[OUT_ARG]], [[OUT_SCRATCH]])
    // CHECK-SAME:  : memref<1x1x16x1000xf16>, memref<1x1x1x1000xf16>, memref<1x1x16x1000xf16>, memref<1x1x1x1000xf16>
    // CHECK:    return [[RES]]#0 : memref<1x1x16x1000xf16>
  }
}

// -----

// CHECK: module @BufferizeScratchBufferWithTiling
module @BufferizeScratchBufferWithTiling {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<1x1x16x?xf16>, %argscratch: memref<1x1x1x?xf16>, %arg1: memref<1x1x16x?xf16>, %arg2: memref<1x1x1x?xf16>, %argidx0: index, %argidx1: index) attributes {
     kernelInfo = #VPU.KernelInfo<
       tilingInfoFunc = @generated_info_0,
       tilingAxes = [3],
       numSlicedInputs = 1 : i64
     >}
    {
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      scf.for %arg3 = %c0 to %c16 step %c1 {
          %subview_1 = memref.subview %arg0[0, 0, 0, %argidx1] [1, 1, 1, %argidx0] [1, 1, 1, 1] : memref<1x1x16x?xf16> to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>>
          %subview_2 = memref.subview %arg1[0, 0, 0, %argidx1] [1, 1, 1, %argidx0] [1, 1, 1, 1] : memref<1x1x16x?xf16> to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>>
          memref.copy %subview_1, %argscratch : memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x1x1x?xf16>
          memref.copy %argscratch, %subview_2 : memref<1x1x1x?xf16> to memref<1x1x1x?xf16, strided<[?, ?, ?, 1], offset: ?>>
      }
      return
    }
    func.func @generated_info_0(%arg0: index, %arg1: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      return
          %c1, %c1, %c16, %arg0, %c0, %c0, %c0, %arg1,
          %c1, %c1, %c1, %arg0, %c0, %c0, %c0, %arg1 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x16x1000xf16>) -> tensor<1x1x16x1000xf16> {
    %scratch = VPU.Empty : tensor<1x1x1x1000xf16>
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x1x16x1000xf16>) scratch(%scratch : tensor<1x1x1x1000xf16>) @VPU.SW::@generated_0 tiling(sizes = [1000], offsets = [0]) -> tensor<1x1x16x1000xf16>
    return %0 : tensor<1x1x16x1000xf16>

    // CHECK:  func.func @main([[ARG0:%[^:]+]]: memref<1x1x16x1000xf16>) -> memref<1x1x16x1000xf16>
    // CHECK:    [[SCRATCH:%.+]] = memref.alloc() : memref<1x1x1x1000xf16>
    // CHECK:    [[OUT:%.+]] = memref.alloc() : memref<1x1x16x1000xf16>
    // CHECK:    [[RES:%[^:]+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@generated_0
    // CHECK-SAME:  inputs([[ARG0]] as [[IN:%[^:]+]]: memref<1x1x16x1000xf16>, [[SCRATCH]] as [[IN_SCRATCH:%[^:]+]]: memref<1x1x1x1000xf16>)
    // CHECK-SAME:  outputs([[OUT]] as [[OUT_ARG:%[^:]+]]: memref<1x1x16x1000xf16>, [[SCRATCH]] as [[OUT_SCRATCH:%[^:]+]]: memref<1x1x1x1000xf16>)
    // CHECK-SAME:  on tile 0 -> (memref<1x1x16x1000xf16>, memref<1x1x1x1000xf16>)
    // CHECK-NEXT:  VPUIP.SW.Kernel.run {attrs = [1000, 0]}([[IN]], [[IN_SCRATCH]], [[OUT_ARG]], [[OUT_SCRATCH]])
    // CHECK-SAME:  : memref<1x1x16x1000xf16>, memref<1x1x1x1000xf16>, memref<1x1x16x1000xf16>, memref<1x1x1x1000xf16>
    // CHECK:    return [[RES]]#0 : memref<1x1x16x1000xf16>
  }
}
