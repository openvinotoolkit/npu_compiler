//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --specialize-sw-kernels="heuristic=always" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// Specialization should skip SwKernel ops with multiple SwKernelRun ops.

module @VPU.SW {
  func.func @generated_0(%in0: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %in0Flat = memref.collapse_shape %in0 [[0, 1, 2, 3]] : memref<1x?x?x?xf16> into memref<?xf16>
    %outFlat = memref.collapse_shape %out [[0, 1, 2, 3]] : memref<1x?x?x?xf16> into memref<?xf16>
    %numElems = memref.dim %in0Flat, %c0 : memref<?xf16>
    scf.for %i = %c0 to %numElems step %c1 {
      %inElem = memref.subview %in0Flat[%i] [1] [1] : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      %outElem = memref.subview %outFlat[%i] [1] [1] : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      memref.copy %inElem, %outElem
          : memref<1xf16, strided<[1], offset: ?>> to memref<1xf16, strided<[1], offset: ?>>
    }
    return
  }
}

func.func @foo(%p0: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %p1: memref<1x3x103x64xf16, [@CMX_NN, 0]>,
               %r0: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %r1: memref<1x3x103x64xf16, [@CMX_NN, 0]>)
        -> (memref<1x3x103x512xf16, [@CMX_NN, 0]>, memref<1x3x103x64xf16, [@CMX_NN, 0]>) {
  VPURT.Task {
    %swKernel:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@generated_0
        inputs(%p0 as %in_0: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %p1 as %in_1: memref<1x3x103x64xf16, [@CMX_NN, 0]>)
        outputs(%r0 as %out_0: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
                %r1 as %out_1: memref<1x3x103x64xf16, [@CMX_NN, 0]>)
        on tile 0 list 0
        -> (memref<1x3x103x512xf16, [@CMX_NN, 0]>, memref<1x3x103x64xf16, [@CMX_NN, 0]>) {
      VPUIP.SW.Kernel.run(%in_0, %out_0)
          : memref<1x3x103x512xf16, [@CMX_NN, 0]>, memref<1x3x103x512xf16, [@CMX_NN, 0]>
      VPUIP.SW.Kernel.run(%in_1, %out_1)
          : memref<1x3x103x64xf16, [@CMX_NN, 0]>, memref<1x3x103x64xf16, [@CMX_NN, 0]>
    }
  }
  return %r0, %r1 : memref<1x3x103x512xf16, [@CMX_NN, 0]>, memref<1x3x103x64xf16, [@CMX_NN, 0]>
}

// CHECK:       module @VPU.SW {
// CHECK:         func.func @generated_0(%{{.+}}: memref<1x?x?x?xf16>, %{{.+}}: memref<1x?x?x?xf16>)
// CHECK:           memref.collapse_shape %{{.+}} {{\[}}[0, 1, 2, 3]] : memref<1x?x?x?xf16> into memref<?xf16>
// CHECK:           memref.dim
// CHECK-NOT:     func.func @generated_0_spec
// CHECK:       func.func @foo

// CHECK:         %{{.+}}:2 = VPUIP.SW.Kernel {{.*}} @VPU.SW::@generated_0 inputs(
// CHECK-NEXT:      VPUIP.SW.Kernel.run({{[^)]+}}) : memref<1x3x103x512xf16, [@CMX_NN, 0]>, memref<1x3x103x512xf16, [@CMX_NN, 0]>
// CHECK-NEXT:      VPUIP.SW.Kernel.run({{[^)]+}}) : memref<1x3x103x64xf16, [@CMX_NN, 0]>, memref<1x3x103x64xf16, [@CMX_NN, 0]>

// -----

module @VPU.SW {
  func.func @generated_0(%in0: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %in0Flat = memref.collapse_shape %in0 [[0, 1, 2, 3]] : memref<1x?x?x?xf16> into memref<?xf16>
    %outFlat = memref.collapse_shape %out [[0, 1, 2, 3]] : memref<1x?x?x?xf16> into memref<?xf16>
    %numElems = memref.dim %in0Flat, %c0 : memref<?xf16>
    scf.for %i = %c0 to %numElems step %c1 {
      %inElem = memref.subview %in0Flat[%i] [1] [1] : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      %outElem = memref.subview %outFlat[%i] [1] [1] : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      memref.copy %inElem, %outElem
          : memref<1xf16, strided<[1], offset: ?>> to memref<1xf16, strided<[1], offset: ?>>
    }
    return
  }
}

func.func @foo(%p0: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %p1: memref<1x3x103x64xf16, [@CMX_NN, 0]>,
               %r0: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %r1: memref<1x3x103x64xf16, [@CMX_NN, 0]>)
        -> (memref<1x3x103x512xf16, [@CMX_NN, 0]>, memref<1x3x103x64xf16, [@CMX_NN, 0]>) {
  VPURT.Task {
    %multiRun:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@generated_0
        inputs(%p0 as %in_0: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %p1 as %in_1: memref<1x3x103x64xf16, [@CMX_NN, 0]>)
        outputs(%r0 as %out_0: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
                %r1 as %out_1: memref<1x3x103x64xf16, [@CMX_NN, 0]>)
        on tile 0 list 0
        -> (memref<1x3x103x512xf16, [@CMX_NN, 0]>, memref<1x3x103x64xf16, [@CMX_NN, 0]>) {
      VPUIP.SW.Kernel.run(%in_0, %out_0)
          : memref<1x3x103x512xf16, [@CMX_NN, 0]>, memref<1x3x103x512xf16, [@CMX_NN, 0]>
      VPUIP.SW.Kernel.run(%in_1, %out_1)
          : memref<1x3x103x64xf16, [@CMX_NN, 0]>, memref<1x3x103x64xf16, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %singleRun = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0
        inputs(%p0 as %in_2: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
        outputs(%r0 as %out_2: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
        on tile 0 list 0
        -> memref<1x3x103x512xf16, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run(%in_2, %out_2)
          : memref<1x3x103x512xf16, [@CMX_NN, 0]>, memref<1x3x103x512xf16, [@CMX_NN, 0]>
    }
  }
  return %r0, %r1 : memref<1x3x103x512xf16, [@CMX_NN, 0]>, memref<1x3x103x64xf16, [@CMX_NN, 0]>
}

// CHECK:       module @VPU.SW {
// CHECK:         func.func @generated_0(%{{.+}}: memref<1x?x?x?xf16>, %{{.+}}: memref<1x?x?x?xf16>)
// CHECK:         func.func @generated_0_spec_0(%{{.+}}: memref<1x3x103x512xf16>, %{{.+}}: memref<1x3x103x512xf16>)
// CHECK:           arith.constant 158208 : index
// CHECK-NOT:     func.func @generated_0_spec_1
// CHECK:       func.func @foo

// CHECK:         %{{.+}}:2 = VPUIP.SW.Kernel {{.*}} @VPU.SW::@generated_0 inputs(
// CHECK-NEXT:      VPUIP.SW.Kernel.run({{[^)]+}}) : memref<1x3x103x512xf16, [@CMX_NN, 0]>, memref<1x3x103x512xf16, [@CMX_NN, 0]>
// CHECK-NEXT:      VPUIP.SW.Kernel.run({{[^)]+}}) : memref<1x3x103x64xf16, [@CMX_NN, 0]>, memref<1x3x103x64xf16, [@CMX_NN, 0]>

// CHECK:         VPUIP.SW.Kernel {{.*}} @VPU.SW::@generated_0_spec_0 inputs(
