//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --specialize-sw-kernels %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

module @VPU.SW {
  func.func private @generated_0_spec_0(memref<1xf16>)
  func.func @generated_0(%in: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %inFlat = memref.collapse_shape %in [[0, 1, 2, 3]] : memref<1x?x?x?xf16> into memref<?xf16>
    %outFlat = memref.collapse_shape %out [[0, 1, 2, 3]] : memref<1x?x?x?xf16> into memref<?xf16>
    %numElems = memref.dim %inFlat, %c0 : memref<?xf16>
    scf.for %i = %c0 to %numElems step %c1 {
      %inElem = memref.subview %inFlat[%i] [1] [1] : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      %outElem = memref.subview %outFlat[%i] [1] [1] : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      memref.copy %inElem, %outElem
          : memref<1xf16, strided<[1], offset: ?>> to memref<1xf16, strided<[1], offset: ?>>
    }
    return
  }
}

func.func @clash(%param: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
                 %ret: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
        -> memref<1x3x103x512xf16, [@CMX_NN, 0]> {
  VPURT.Task {
    %swKernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0
        inputs(%param as %in: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
        outputs(%ret as %out: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
        on tile 0 list 0
        -> memref<1x3x103x512xf16, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run(%in, %out)
          : memref<1x3x103x512xf16, [@CMX_NN, 0]>, memref<1x3x103x512xf16, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<1x3x103x512xf16, [@CMX_NN, 0]>
}

// CHECK:       module @VPU.SW
// CHECK:         func.func private @generated_0_spec_0(memref<1xf16>)
// CHECK-NOT:     func.func @generated_0(
// CHECK:         func.func @generated_0_spec_0_0([[IN:%.+]]: memref<1x3x103x512xf16>, [[OUT:%.+]]: memref<1x3x103x512xf16>)
// CHECK:           arith.constant 158208 : index

// CHECK-LABEL: func.func @clash
// CHECK:         VPUIP.SW.Kernel {{.*}}@VPU.SW::@generated_0_spec_0_0 inputs(
