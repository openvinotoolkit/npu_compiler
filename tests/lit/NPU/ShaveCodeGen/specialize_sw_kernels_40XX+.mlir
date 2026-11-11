//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --specialize-sw-kernels %s | FileCheck %s --check-prefixes=CHECK,UNIFORM
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --specialize-sw-kernels="heuristic=never" %s | FileCheck %s --check-prefixes=CHECK,UNIFORM
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --specialize-sw-kernels="heuristic=always" %s | FileCheck %s --check-prefixes=CHECK,ALWAYS
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// End-to-end specialization cases.

#C = affine_map<(d0) -> (d0)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @generated_info_0(%outSizeC: index, %outSizeH: index, %outSizeW: index,
                              %outOffC: index, %outOffH: index, %outOffW: index)
      -> (index, index, index, index, index, index, index, index,
          index, index, index, index, index, index, index, index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    return %c1, %outSizeC, %outSizeH, %outSizeW, %c0, %outOffC, %outOffH, %outOffW,
           %c1, %outSizeC, %outSizeH, %outSizeW, %c0, %outOffC, %outOffH, %outOffW
        : index, index, index, index, index, index, index, index,
          index, index, index, index, index, index, index, index
  }
  func.func @generated_0(%in0: memref<1x?x?x?xf16>, %in1: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>,
                         %sizeC: index, %sizeH: index, %sizeW: index,
                         %offC: index, %offH: index, %offW: index) attributes {
    kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [1, 2, 3],
        numSlicedInputs = 2 : i64,
        numTensorInputs = 2 : i64,
        numAuxBuffers = 0 : i64,
        inputSegmentationAxes = []>
  } {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %in0Flat = memref.collapse_shape %in0 [[0, 1, 2, 3]] : memref<1x?x?x?xf16> into memref<?xf16>
    %in1Flat = memref.collapse_shape %in1 [[0, 1, 2, 3]] : memref<1x?x?x?xf16> into memref<?xf16>
    %outFlat = memref.collapse_shape %out [[0, 1, 2, 3]] : memref<1x?x?x?xf16> into memref<?xf16>
    %numElems = memref.dim %in0Flat, %c0 : memref<?xf16>
    scf.for %i = %c0 to %numElems step %c1 {
      %in0Elem = memref.subview %in0Flat[%i] [1] [1]
          : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      %in1Elem = memref.subview %in1Flat[%i] [1] [1]
          : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      %resElem = memref.subview %outFlat[%i] [1] [1]
          : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#C, #C, #C], iterator_types = ["parallel"]}
          ins(%in0Elem, %in1Elem : memref<1xf16, strided<[1], offset: ?>>,
                                   memref<1xf16, strided<[1], offset: ?>>)
          outs(%resElem : memref<1xf16, strided<[1], offset: ?>>) {
      ^bb0(%lhs: f16, %rhs: f16, %outInit: f16):
        %diff = arith.subf %lhs, %rhs : f16
        %sqDiff = arith.mulf %diff, %diff : f16
        linalg.yield %sqDiff : f16
      }
      %outElem = memref.subview %outFlat[%i] [1] [1]
          : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      memref.copy %resElem, %outElem
          : memref<1xf16, strided<[1], offset: ?>> to memref<1xf16, strided<[1], offset: ?>>
    }
    %dimC = memref.dim %in0, %c1 : memref<1x?x?x?xf16>
    %dimH = memref.dim %in0, %c2 : memref<1x?x?x?xf16>
    %dimW = memref.dim %in0, %c3 : memref<1x?x?x?xf16>
    %outExpanded = memref.expand_shape %outFlat [[0, 1, 2, 3]] output_shape [1, %dimC, %dimH, %dimW]
        : memref<?xf16> into memref<1x?x?x?xf16>
    memref.copy %outExpanded, %out : memref<1x?x?x?xf16> to memref<1x?x?x?xf16>
    return
  }
}

func.func @foo(%param0: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %param1: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %ret: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
        -> memref<1x3x103x512xf16, [@CMX_NN, 0]> {

  VPURT.Task {
    %swKernel0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0
        inputs(%param0 as %in0_0: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %param1 as %in1_0: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
        outputs(%ret as %out_0: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
        on tile 0 list 0
        -> memref<1x3x103x512xf16, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run {attrs = [3, 103, 512, 4, 0, 0]}(%in0_0, %in1_0, %out_0)
          : memref<1x3x103x512xf16, [@CMX_NN, 0]>,
            memref<1x3x103x512xf16, [@CMX_NN, 0]>,
            memref<1x3x103x512xf16, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %swKernel1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0
        inputs(%param0 as %in0_1: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %param1 as %in1_1: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
        outputs(%ret as %out_1: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
        on tile 0 list 0
        -> memref<1x3x103x512xf16, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run {attrs = [3, 103, 512, 8, 0, 0]}(%in0_1, %in1_1, %out_1)
          : memref<1x3x103x512xf16, [@CMX_NN, 0]>,
            memref<1x3x103x512xf16, [@CMX_NN, 0]>,
            memref<1x3x103x512xf16, [@CMX_NN, 0]>
    }
  }

  return %ret : memref<1x3x103x512xf16, [@CMX_NN, 0]>
}

// CHECK:       module @VPU.SW
// CHECK-NOT:     func.func @generated_0(
// CHECK:         func.func @generated_info_0

// CHECK-LABEL: func.func @generated_0_spec_0
// CHECK-SAME:      ([[IN0:%.+]]: memref<1x3x103x512xf16>, [[IN1:%.+]]: memref<1x3x103x512xf16>, [[OUT:%.+]]: memref<1x3x103x512xf16>,
// CHECK-SAME:      kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_0

// CHECK:           [[NUM_ELEMS:%.+]] = arith.constant 158208 : index
// CHECK:           [[C1:%.+]] = arith.constant 1 : index
// CHECK:           [[C0:%.+]] = arith.constant 0 : index

// CHECK-NOT:       llvm.intr.assume
// CHECK:           [[IN0_FLAT:%.+]] = memref.collapse_shape [[IN0]] {{\[}}[0, 1, 2, 3]] : memref<1x3x103x512xf16> into memref<158208xf16>
// CHECK:           [[IN1_FLAT:%.+]] = memref.collapse_shape [[IN1]] {{\[}}[0, 1, 2, 3]] : memref<1x3x103x512xf16> into memref<158208xf16>
// CHECK:           [[OUT_FLAT:%.+]] = memref.collapse_shape [[OUT]] {{\[}}[0, 1, 2, 3]] : memref<1x3x103x512xf16> into memref<158208xf16>
// CHECK-NOT:       memref.dim
// CHECK:           scf.for [[I:%.+]] = [[C0]] to [[NUM_ELEMS]] step [[C1]]
// CHECK:             [[IN0_ELEM:%.+]] = memref.subview [[IN0_FLAT]]{{\[}}[[I]]] [1] [1] : memref<158208xf16> to memref<1xf16, strided<[1], offset: ?>>
// CHECK:             [[IN1_ELEM:%.+]] = memref.subview [[IN1_FLAT]]{{\[}}[[I]]] [1] [1] : memref<158208xf16> to memref<1xf16, strided<[1], offset: ?>>
// CHECK:             [[RES_ELEM:%.+]] = memref.subview [[OUT_FLAT]]{{\[}}[[I]]] [1] [1] : memref<158208xf16> to memref<1xf16, strided<[1], offset: ?>>
// CHECK:             linalg.generic
// CHECK-SAME:            ins([[IN0_ELEM]], [[IN1_ELEM]]
// CHECK-SAME:            outs([[RES_ELEM]]
// CHECK:               arith.subf
// CHECK:               arith.mulf
// CHECK:             [[OUT_ELEM:%.+]] = memref.subview [[OUT_FLAT]]{{\[}}[[I]]] [1] [1] : memref<158208xf16> to memref<1xf16, strided<[1], offset: ?>>
// CHECK:             memref.copy [[RES_ELEM]], [[OUT_ELEM]]
// CHECK:           }
// CHECK-NEXT:      return

// CHECK-LABEL: func.func @foo
// CHECK-SAME:      ([[PARAM0:%.+]]: memref<1x3x103x512xf16, [@CMX_NN, 0]>, [[PARAM1:%.+]]: memref<1x3x103x512xf16, [@CMX_NN, 0]>, [[RET:%.+]]: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
// CHECK:           VPURT.Task
// CHECK:             VPUIP.SW.Kernel {{.*}}@VPU.SW::@generated_0_spec_0 inputs([[PARAM0]] as {{[^:]+}}: memref<1x3x103x512xf16, [@CMX_NN, 0]>, [[PARAM1]] as {{[^:]+}}: memref<1x3x103x512xf16, [@CMX_NN, 0]>) outputs([[RET]] as {{[^:]+}}: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
// CHECK:               VPUIP.SW.Kernel.run {attrs = [3, 103, 512, 4, 0, 0]}
// CHECK:           VPURT.Task
// CHECK:             VPUIP.SW.Kernel {{.*}}@VPU.SW::@generated_0_spec_0 inputs([[PARAM0]] as {{[^:]+}}: memref<1x3x103x512xf16, [@CMX_NN, 0]>, [[PARAM1]] as {{[^:]+}}: memref<1x3x103x512xf16, [@CMX_NN, 0]>) outputs([[RET]] as {{[^:]+}}: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
// CHECK:               VPUIP.SW.Kernel.run {attrs = [3, 103, 512, 8, 0, 0]}

// -----

#C = affine_map<(d0) -> (d0)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @generated_info_0(%outSizeC: index, %outSizeH: index, %outSizeW: index,
                              %outOffC: index, %outOffH: index, %outOffW: index)
      -> (index, index, index, index, index, index, index, index,
          index, index, index, index, index, index, index, index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    return %c1, %outSizeC, %outSizeH, %outSizeW, %c0, %outOffC, %outOffH, %outOffW,
           %c1, %outSizeC, %outSizeH, %outSizeW, %c0, %outOffC, %outOffH, %outOffW
        : index, index, index, index, index, index, index, index,
          index, index, index, index, index, index, index, index
  }
  func.func @generated_0(%in0: memref<1x?x?x?xf16>, %in1: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>,
                         %sizeC: index, %sizeH: index, %sizeW: index,
                         %offC: index, %offH: index, %offW: index) attributes {
    kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [1, 2, 3],
        numSlicedInputs = 2 : i64,
        numTensorInputs = 2 : i64,
        numAuxBuffers = 0 : i64,
        inputSegmentationAxes = []>
  } {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %in0Flat = memref.collapse_shape %in0 [[0, 1, 2, 3]] : memref<1x?x?x?xf16> into memref<?xf16>
    %in1Flat = memref.collapse_shape %in1 [[0, 1, 2, 3]] : memref<1x?x?x?xf16> into memref<?xf16>
    %outFlat = memref.collapse_shape %out [[0, 1, 2, 3]] : memref<1x?x?x?xf16> into memref<?xf16>
    %numElems = memref.dim %in0Flat, %c0 : memref<?xf16>
    scf.for %i = %c0 to %numElems step %c1 {
      %in0Elem = memref.subview %in0Flat[%i] [1] [1]
          : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      %in1Elem = memref.subview %in1Flat[%i] [1] [1]
          : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      %resElem = memref.subview %outFlat[%i] [1] [1]
          : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#C, #C, #C], iterator_types = ["parallel"]}
          ins(%in0Elem, %in1Elem : memref<1xf16, strided<[1], offset: ?>>,
                                   memref<1xf16, strided<[1], offset: ?>>)
          outs(%resElem : memref<1xf16, strided<[1], offset: ?>>) {
      ^bb0(%lhs: f16, %rhs: f16, %outInit: f16):
        %diff = arith.subf %lhs, %rhs : f16
        %sqDiff = arith.mulf %diff, %diff : f16
        linalg.yield %sqDiff : f16
      }
      %outElem = memref.subview %outFlat[%i] [1] [1]
          : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
      memref.copy %resElem, %outElem
          : memref<1xf16, strided<[1], offset: ?>> to memref<1xf16, strided<[1], offset: ?>>
    }
    %dimC = memref.dim %in0, %c1 : memref<1x?x?x?xf16>
    %dimH = memref.dim %in0, %c2 : memref<1x?x?x?xf16>
    %dimW = memref.dim %in0, %c3 : memref<1x?x?x?xf16>
    %outExpanded = memref.expand_shape %outFlat [[0, 1, 2, 3]] output_shape [1, %dimC, %dimH, %dimW]
        : memref<?xf16> into memref<1x?x?x?xf16>
    memref.copy %outExpanded, %out : memref<1x?x?x?xf16> to memref<1x?x?x?xf16>
    return
  }
}

func.func @bar(%param0: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %param1: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %ret: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %param00: memref<1x4x104x512xf16, [@CMX_NN, 0]>,
               %param01: memref<1x4x104x512xf16, [@CMX_NN, 0]>,
               %ret0: memref<1x4x104x512xf16, [@CMX_NN, 0]>)
        -> memref<1x3x103x512xf16, [@CMX_NN, 0]> {

  VPURT.Task {
    %swKernel0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0
        inputs(%param0 as %in0_0: memref<1x3x103x512xf16, [@CMX_NN, 0]>,
               %param1 as %in1_0: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
        outputs(%ret as %out_0: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
        on tile 0 list 0
        -> memref<1x3x103x512xf16, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run {attrs = [3, 103, 512, 4, 0, 0]}(%in0_0, %in1_0, %out_0)
          : memref<1x3x103x512xf16, [@CMX_NN, 0]>,
            memref<1x3x103x512xf16, [@CMX_NN, 0]>,
            memref<1x3x103x512xf16, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %swKernel1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0
        inputs(%param00 as %in0_1: memref<1x4x104x512xf16, [@CMX_NN, 0]>,
               %param01 as %in1_1: memref<1x4x104x512xf16, [@CMX_NN, 0]>)
        outputs(%ret0 as %out_1: memref<1x4x104x512xf16, [@CMX_NN, 0]>)
        on tile 0 list 0
        -> memref<1x4x104x512xf16, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run {attrs = [4, 104, 512, 8, 0, 0]}(%in0_1, %in1_1, %out_1)
          : memref<1x4x104x512xf16, [@CMX_NN, 0]>,
            memref<1x4x104x512xf16, [@CMX_NN, 0]>,
            memref<1x4x104x512xf16, [@CMX_NN, 0]>
    }
  }

  return %ret : memref<1x3x103x512xf16, [@CMX_NN, 0]>
}

// CHECK:       module @VPU.SW
// CHECK-NOT:     func.func @generated_0(

// UNIFORM-LABEL: func.func @generated_0_spec_0
// UNIFORM-SAME:      ([[IN0:%.+]]: memref<1x?x?x512xf16>, [[IN1:%.+]]: memref<1x?x?x512xf16>, [[OUT:%.+]]: memref<1x?x?x512xf16>,
// UNIFORM-DAG:       [[C0:%.+]] = arith.constant 0 : index
// UNIFORM-DAG:       [[C4:%.+]] = arith.constant 4 : index
// UNIFORM-DAG:       [[C3:%.+]] = arith.constant 3 : index
// UNIFORM-DAG:       [[C1:%.+]] = arith.constant 1 : index
// UNIFORM-DAG:       [[C104:%.+]] = arith.constant 104 : index
// UNIFORM-DAG:       [[C103:%.+]] = arith.constant 103 : index
// UNIFORM-DAG:       [[C2:%.+]] = arith.constant 2 : index

// UNIFORM:           [[IN0_H:%.+]] = memref.dim [[IN0]], [[C2]] : memref<1x?x?x512xf16>
// UNIFORM:           [[IN0_H_103:%.+]] = arith.cmpi eq, [[IN0_H]], [[C103]]
// UNIFORM:           [[IN0_H_104:%.+]] = arith.cmpi eq, [[IN0_H]], [[C104]]
// UNIFORM:           [[IN0_H_OK:%.+]] = arith.ori [[IN0_H_103]], [[IN0_H_104]]
// UNIFORM:           llvm.intr.assume [[IN0_H_OK]]
// UNIFORM:           [[IN0_C:%.+]] = memref.dim [[IN0]], [[C1]] : memref<1x?x?x512xf16>
// UNIFORM:           [[IN0_C_3:%.+]] = arith.cmpi eq, [[IN0_C]], [[C3]]
// UNIFORM:           [[IN0_C_4:%.+]] = arith.cmpi eq, [[IN0_C]], [[C4]]
// UNIFORM:           [[IN0_C_OK:%.+]] = arith.ori [[IN0_C_3]], [[IN0_C_4]]
// UNIFORM:           llvm.intr.assume [[IN0_C_OK]]

// UNIFORM:           [[IN1_H:%.+]] = memref.dim [[IN1]], [[C2]] : memref<1x?x?x512xf16>
// UNIFORM:           [[IN1_H_103:%.+]] = arith.cmpi eq, [[IN1_H]], [[C103]]
// UNIFORM:           [[IN1_H_104:%.+]] = arith.cmpi eq, [[IN1_H]], [[C104]]
// UNIFORM:           [[IN1_H_OK:%.+]] = arith.ori [[IN1_H_103]], [[IN1_H_104]]
// UNIFORM:           llvm.intr.assume [[IN1_H_OK]]
// UNIFORM:           [[IN1_C:%.+]] = memref.dim [[IN1]], [[C1]] : memref<1x?x?x512xf16>
// UNIFORM:           [[IN1_C_3:%.+]] = arith.cmpi eq, [[IN1_C]], [[C3]]
// UNIFORM:           [[IN1_C_4:%.+]] = arith.cmpi eq, [[IN1_C]], [[C4]]
// UNIFORM:           [[IN1_C_OK:%.+]] = arith.ori [[IN1_C_3]], [[IN1_C_4]]
// UNIFORM:           llvm.intr.assume [[IN1_C_OK]]

// UNIFORM:           [[OUT_H:%.+]] = memref.dim [[OUT]], [[C2]] : memref<1x?x?x512xf16>
// UNIFORM:           [[OUT_H_103:%.+]] = arith.cmpi eq, [[OUT_H]], [[C103]]
// UNIFORM:           [[OUT_H_104:%.+]] = arith.cmpi eq, [[OUT_H]], [[C104]]
// UNIFORM:           [[OUT_H_OK:%.+]] = arith.ori [[OUT_H_103]], [[OUT_H_104]]
// UNIFORM:           llvm.intr.assume [[OUT_H_OK]]
// UNIFORM:           [[OUT_C:%.+]] = memref.dim [[OUT]], [[C1]] : memref<1x?x?x512xf16>
// UNIFORM:           [[OUT_C_3:%.+]] = arith.cmpi eq, [[OUT_C]], [[C3]]
// UNIFORM:           [[OUT_C_4:%.+]] = arith.cmpi eq, [[OUT_C]], [[C4]]
// UNIFORM:           [[OUT_C_OK:%.+]] = arith.ori [[OUT_C_3]], [[OUT_C_4]]
// UNIFORM:           llvm.intr.assume [[OUT_C_OK]]

// UNIFORM:           [[IN0_FLAT:%.+]] = memref.collapse_shape [[IN0]] {{\[}}[0, 1, 2, 3]] : memref<1x?x?x512xf16> into memref<?xf16>
// UNIFORM:           [[IN1_FLAT:%.+]] = memref.collapse_shape [[IN1]] {{\[}}[0, 1, 2, 3]] : memref<1x?x?x512xf16> into memref<?xf16>
// UNIFORM:           [[OUT_FLAT:%.+]] = memref.collapse_shape [[OUT]] {{\[}}[0, 1, 2, 3]] : memref<1x?x?x512xf16> into memref<?xf16>
// UNIFORM:           [[NUM_ELEMS:%.+]] = memref.dim [[IN0_FLAT]], [[C0]] : memref<?xf16>
// UNIFORM:           scf.for [[I:%.+]] = [[C0]] to [[NUM_ELEMS]] step [[C1]]
// UNIFORM:             [[IN0_ELEM:%.+]] = memref.subview [[IN0_FLAT]]{{\[}}[[I]]] [1] [1] : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
// UNIFORM:             [[IN1_ELEM:%.+]] = memref.subview [[IN1_FLAT]]{{\[}}[[I]]] [1] [1] : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
// UNIFORM:             [[RES_ELEM:%.+]] = memref.subview [[OUT_FLAT]]{{\[}}[[I]]] [1] [1] : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
// UNIFORM:             linalg.generic
// UNIFORM-SAME:            ins([[IN0_ELEM]], [[IN1_ELEM]]
// UNIFORM-SAME:            outs([[RES_ELEM]]
// UNIFORM:             [[OUT_ELEM:%.+]] = memref.subview [[OUT_FLAT]]{{\[}}[[I]]] [1] [1] : memref<?xf16> to memref<1xf16, strided<[1], offset: ?>>
// UNIFORM:             memref.copy [[RES_ELEM]], [[OUT_ELEM]]
// UNIFORM:           [[EXP_C:%.+]] = memref.dim [[IN0]], [[C1]] : memref<1x?x?x512xf16>
// UNIFORM:           [[EXP_H:%.+]] = memref.dim [[IN0]], [[C2]] : memref<1x?x?x512xf16>
// UNIFORM:           [[OUT_EXPANDED:%.+]] = memref.expand_shape [[OUT_FLAT]] {{\[}}[0, 1, 2, 3]] output_shape [1, [[EXP_C]], [[EXP_H]], 512] : memref<?xf16> into memref<1x?x?x512xf16>
// UNIFORM:           memref.copy [[OUT_EXPANDED]], [[OUT]] : memref<1x?x?x512xf16> to memref<1x?x?x512xf16>

// ALWAYS-LABEL: func.func @generated_0_spec_1
// ALWAYS-SAME:      ([[IN0_0:%.+]]: memref<1x3x103x512xf16>, [[IN1_0:%.+]]: memref<1x3x103x512xf16>, [[OUT_0:%.+]]: memref<1x3x103x512xf16>,
// ALWAYS:           [[NUM_ELEMS_0:%.+]] = arith.constant 158208 : index
// ALWAYS:           [[C1_0:%.+]] = arith.constant 1 : index
// ALWAYS:           [[C0_0:%.+]] = arith.constant 0 : index
// ALWAYS-NOT:       llvm.intr.assume
// ALWAYS:           [[IN0_FLAT_0:%.+]] = memref.collapse_shape [[IN0_0]] {{\[}}[0, 1, 2, 3]] : memref<1x3x103x512xf16> into memref<158208xf16>
// ALWAYS:           [[IN1_FLAT_0:%.+]] = memref.collapse_shape [[IN1_0]] {{\[}}[0, 1, 2, 3]] : memref<1x3x103x512xf16> into memref<158208xf16>
// ALWAYS:           [[OUT_FLAT_0:%.+]] = memref.collapse_shape [[OUT_0]] {{\[}}[0, 1, 2, 3]] : memref<1x3x103x512xf16> into memref<158208xf16>
// ALWAYS-NOT:       memref.dim
// ALWAYS:           scf.for [[I_0:%.+]] = [[C0_0]] to [[NUM_ELEMS_0]] step [[C1_0]]
// ALWAYS:             linalg.generic
// ALWAYS:             memref.copy
// ALWAYS:           }
// ALWAYS-NEXT:      return

// ALWAYS-LABEL: func.func @generated_0_spec_0
// ALWAYS-SAME:      ([[IN0_1:%.+]]: memref<1x4x104x512xf16>, [[IN1_1:%.+]]: memref<1x4x104x512xf16>, [[OUT_1:%.+]]: memref<1x4x104x512xf16>,
// ALWAYS:           [[NUM_ELEMS_1:%.+]] = arith.constant 212992 : index
// ALWAYS:           [[C1_1:%.+]] = arith.constant 1 : index
// ALWAYS:           [[C0_1:%.+]] = arith.constant 0 : index
// ALWAYS-NOT:       llvm.intr.assume
// ALWAYS:           [[IN0_FLAT_1:%.+]] = memref.collapse_shape [[IN0_1]] {{\[}}[0, 1, 2, 3]] : memref<1x4x104x512xf16> into memref<212992xf16>
// ALWAYS:           [[IN1_FLAT_1:%.+]] = memref.collapse_shape [[IN1_1]] {{\[}}[0, 1, 2, 3]] : memref<1x4x104x512xf16> into memref<212992xf16>
// ALWAYS:           [[OUT_FLAT_1:%.+]] = memref.collapse_shape [[OUT_1]] {{\[}}[0, 1, 2, 3]] : memref<1x4x104x512xf16> into memref<212992xf16>
// ALWAYS-NOT:       memref.dim
// ALWAYS:           scf.for [[I_1:%.+]] = [[C0_1]] to [[NUM_ELEMS_1]] step [[C1_1]]
// ALWAYS:             linalg.generic
// ALWAYS:             memref.copy
// ALWAYS:           }
// ALWAYS-NEXT:      return

// CHECK-LABEL: func.func @bar
// CHECK-SAME:      ([[PARAM0:%.+]]: memref<1x3x103x512xf16, [@CMX_NN, 0]>, [[PARAM1:%.+]]: memref<1x3x103x512xf16, [@CMX_NN, 0]>, [[RET:%.+]]: memref<1x3x103x512xf16, [@CMX_NN, 0]>, [[PARAM00:%.+]]: memref<1x4x104x512xf16, [@CMX_NN, 0]>, [[PARAM01:%.+]]: memref<1x4x104x512xf16, [@CMX_NN, 0]>, [[RET0:%.+]]: memref<1x4x104x512xf16, [@CMX_NN, 0]>)

// UNIFORM:         VPUIP.SW.Kernel {{.*}}@VPU.SW::@generated_0_spec_0 inputs([[PARAM0]] as {{[^:]+}}: memref<1x3x103x512xf16, [@CMX_NN, 0]>, [[PARAM1]] as {{[^:]+}}: memref<1x3x103x512xf16, [@CMX_NN, 0]>) outputs([[RET]] as {{[^:]+}}: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
// UNIFORM:           VPUIP.SW.Kernel.run {attrs = [3, 103, 512, 4, 0, 0]}
// UNIFORM:         VPUIP.SW.Kernel {{.*}}@VPU.SW::@generated_0_spec_0 inputs([[PARAM00]] as {{[^:]+}}: memref<1x4x104x512xf16, [@CMX_NN, 0]>, [[PARAM01]] as {{[^:]+}}: memref<1x4x104x512xf16, [@CMX_NN, 0]>) outputs([[RET0]] as {{[^:]+}}: memref<1x4x104x512xf16, [@CMX_NN, 0]>)
// UNIFORM:           VPUIP.SW.Kernel.run {attrs = [4, 104, 512, 8, 0, 0]}

// ALWAYS:          VPUIP.SW.Kernel {{.*}}@VPU.SW::@generated_0_spec_1 inputs([[PARAM0]] as {{[^:]+}}: memref<1x3x103x512xf16, [@CMX_NN, 0]>, [[PARAM1]] as {{[^:]+}}: memref<1x3x103x512xf16, [@CMX_NN, 0]>) outputs([[RET]] as {{[^:]+}}: memref<1x3x103x512xf16, [@CMX_NN, 0]>)
// ALWAYS:            VPUIP.SW.Kernel.run {attrs = [3, 103, 512, 4, 0, 0]}
// ALWAYS:          VPUIP.SW.Kernel {{.*}}@VPU.SW::@generated_0_spec_0 inputs([[PARAM00]] as {{[^:]+}}: memref<1x4x104x512xf16, [@CMX_NN, 0]>, [[PARAM01]] as {{[^:]+}}: memref<1x4x104x512xf16, [@CMX_NN, 0]>) outputs([[RET0]] as {{[^:]+}}: memref<1x4x104x512xf16, [@CMX_NN, 0]>)
// ALWAYS:            VPUIP.SW.Kernel.run {attrs = [4, 104, 512, 8, 0, 0]}

// -----

// A kernel where the output is unranked.

#C = affine_map<(d0) -> (d0)>
#SCALAR = affine_map<(d0) -> ()>

module @VPU.SW {
  func.func @generated_0(%in: memref<?xf16>, %out: memref<f16>) {
    linalg.generic {indexing_maps = [#C, #SCALAR], iterator_types = ["reduction"]}
        ins(%in : memref<?xf16>)
        outs(%out : memref<f16>) {
    ^bb0(%elem: f16, %acc: f16):
      %sum = arith.addf %acc, %elem : f16
      linalg.yield %sum : f16
    }
    return
  }
}

func.func @nonUniformInput(%param0: memref<128xf16, [@CMX_NN, 0]>,
                           %param1: memref<256xf16, [@CMX_NN, 0]>,
                           %ret0: memref<f16, [@CMX_NN, 0]>,
                           %ret1: memref<f16, [@CMX_NN, 0]>) -> memref<f16, [@CMX_NN, 0]> {
  VPURT.Task {
    %swKernel0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0
        inputs(%param0 as %in_0: memref<128xf16, [@CMX_NN, 0]>)
        outputs(%ret0 as %out_0: memref<f16, [@CMX_NN, 0]>)
        on tile 0 list 0
        -> memref<f16, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run(%in_0, %out_0)
          : memref<128xf16, [@CMX_NN, 0]>, memref<f16, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %swKernel1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_0
        inputs(%param1 as %in_1: memref<256xf16, [@CMX_NN, 0]>)
        outputs(%ret1 as %out_1: memref<f16, [@CMX_NN, 0]>)
        on tile 0 list 0
        -> memref<f16, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run(%in_1, %out_1)
          : memref<256xf16, [@CMX_NN, 0]>, memref<f16, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<f16, [@CMX_NN, 0]>
}

// CHECK:       module @VPU.SW
// CHECK-NOT:     func.func @generated_0(

// CHECK-LABEL: func.func @generated_0_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<?xf16>, [[OUT:%.+]]: memref<f16>)
// CHECK-DAG:       [[C256:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[C0:%.+]] = arith.constant 0 : index
// CHECK:           [[DIM:%.+]] = memref.dim [[IN]], [[C0]] : memref<?xf16>
// CHECK:           [[EQ128:%.+]] = arith.cmpi eq, [[DIM]], [[C128]]
// CHECK:           [[EQ256:%.+]] = arith.cmpi eq, [[DIM]], [[C256]]
// CHECK:           [[ANY:%.+]] = arith.ori [[EQ128]], [[EQ256]]
// CHECK:           llvm.intr.assume [[ANY]]
// CHECK:           linalg.generic
// CHECK-SAME:          ins([[IN]] : memref<?xf16>)
// CHECK-SAME:          outs([[OUT]] : memref<f16>)
// CHECK-NOT:   func.func @generated_0_spec_1

// CHECK-LABEL: func.func @nonUniformInput
// CHECK-SAME:      ([[PARAM0:%.+]]: memref<128xf16, [@CMX_NN, 0]>, [[PARAM1:%.+]]: memref<256xf16, [@CMX_NN, 0]>, [[RET0:%.+]]: memref<f16, [@CMX_NN, 0]>, [[RET1:%.+]]: memref<f16, [@CMX_NN, 0]>)
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@generated_0_spec_0 inputs([[PARAM0]] as {{[^:]+}}: memref<128xf16, [@CMX_NN, 0]>) outputs([[RET0]] as {{[^:]+}}: memref<f16, [@CMX_NN, 0]>)
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@generated_0_spec_0 inputs([[PARAM1]] as {{[^:]+}}: memref<256xf16, [@CMX_NN, 0]>) outputs([[RET1]] as {{[^:]+}}: memref<f16, [@CMX_NN, 0]>)
