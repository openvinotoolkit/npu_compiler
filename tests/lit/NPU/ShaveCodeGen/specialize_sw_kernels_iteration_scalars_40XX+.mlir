//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --specialize-sw-kernels="heuristic=never" %s | FileCheck %s --check-prefixes=CHECK,NEVER
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --specialize-sw-kernels="heuristic=always" %s | FileCheck %s --check-prefixes=CHECK,SPLIT
// REQUIRES: platform-NPU4000 || platform-NPU5010

module @VPU.SW {
  func.func private @info(index, index) -> (index, index)

  func.func @uniform_fill(%in: memref<1x64xf16>, %out: memref<1x64xf16>, %size: index, %off: index)
      attributes {kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @info, tilingAxes = [1],
          numSlicedInputs = 1 : i64,
          numTensorInputs = 1 : i64,
          numAuxBuffers = 0 : i64,
          inputSegmentationAxes = []>} {
    %c0 = arith.constant 0 : index
    %v = memref.load %in[%c0, %c0] : memref<1x64xf16>
    %tile = memref.subview %out[0, %off] [1, %size] [1, 1]
        : memref<1x64xf16> to memref<1x?xf16, strided<[64, 1], offset: ?>>
    linalg.fill ins(%v : f16) outs(%tile : memref<1x?xf16, strided<[64, 1], offset: ?>>)
    return
  }
}

func.func @UniformScalarsFold(%param0: memref<1x64xf16, [@CMX_NN, 0]>, %ret0: memref<1x64xf16, [@CMX_NN, 0]>, %param1: memref<1x64xf16, [@CMX_NN, 0]>, %ret1: memref<1x64xf16, [@CMX_NN, 0]>) -> memref<1x64xf16, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@uniform_fill inputs(%param0 as %arg0: memref<1x64xf16, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x64xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x64xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [5, 9]}(%arg0, %arg1) : memref<1x64xf16, [@CMX_NN, 0]>, memref<1x64xf16, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@uniform_fill inputs(%param1 as %arg0: memref<1x64xf16, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x64xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x64xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [5, 9]}(%arg0, %arg1) : memref<1x64xf16, [@CMX_NN, 0]>, memref<1x64xf16, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x64xf16, [@CMX_NN, 0]>
}

// CHECK:       module @VPU.SW
// CHECK:         func.func private @info
// CHECK-NOT:     func.func @uniform_fill(

// CHECK-LABEL: func.func @uniform_fill_spec_0
// CHECK-SAME:      ([[IN:%[^:]+]]: memref<1x64xf16>, [[OUT:%[^:]+]]: memref<1x64xf16>, %{{[^:]+}}: index, %{{[^:]+}}: index)
// CHECK-SAME:      kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @info
// CHECK-NEXT:      [[C0:%.+]] = arith.constant 0 : index
// CHECK-NEXT:      [[VAL:%.+]] = memref.load [[IN]]{{\[}}[[C0]], [[C0]]] : memref<1x64xf16>
// CHECK-NEXT:      [[TILE:%.+]] = memref.subview [[OUT]]{{\[}}0, 9] [1, 5] [1, 1] : memref<1x64xf16> to memref<1x5xf16, strided<[64, 1], offset: 9>>
// CHECK-NEXT:      linalg.fill ins([[VAL]] : f16) outs([[TILE]] : memref<1x5xf16, strided<[64, 1], offset: 9>>)
// CHECK-NEXT:      return

// CHECK:       func.func @UniformScalarsFold
// CHECK-SAME:      ([[PARAM0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[RET0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[PARAM1:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[RET1:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@uniform_fill_spec_0 inputs([[PARAM0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>) outputs([[RET0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:             VPUIP.SW.Kernel.run {attrs = [5, 9]}
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@uniform_fill_spec_0 inputs([[PARAM1]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>) outputs([[RET1]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:             VPUIP.SW.Kernel.run {attrs = [5, 9]}

// -----

module @VPU.SW {
  func.func private @info(index, index) -> (index, index)

  func.func @assume_fill(%in: memref<1x64xf16>, %out: memref<1x64xf16>, %size: index, %off: index)
      attributes {kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @info, tilingAxes = [1],
          numSlicedInputs = 1 : i64,
          numTensorInputs = 1 : i64,
          numAuxBuffers = 0 : i64,
          inputSegmentationAxes = []>} {
    %c0 = arith.constant 0 : index
    %v = memref.load %in[%c0, %c0] : memref<1x64xf16>
    %tile = memref.subview %out[0, %off] [1, %size] [1, 1]
        : memref<1x64xf16> to memref<1x?xf16, strided<[64, 1], offset: ?>>
    linalg.fill ins(%v : f16) outs(%tile : memref<1x?xf16, strided<[64, 1], offset: ?>>)
    return
  }
}

func.func @BoundedScalarGetsAssume(%param0: memref<1x64xf16, [@CMX_NN, 0]>, %ret0: memref<1x64xf16, [@CMX_NN, 0]>, %param1: memref<1x64xf16, [@CMX_NN, 0]>, %ret1: memref<1x64xf16, [@CMX_NN, 0]>) -> memref<1x64xf16, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@assume_fill inputs(%param0 as %arg0: memref<1x64xf16, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x64xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x64xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [5, 9]}(%arg0, %arg1) : memref<1x64xf16, [@CMX_NN, 0]>, memref<1x64xf16, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@assume_fill inputs(%param1 as %arg0: memref<1x64xf16, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x64xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x64xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [5, 13]}(%arg0, %arg1) : memref<1x64xf16, [@CMX_NN, 0]>, memref<1x64xf16, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x64xf16, [@CMX_NN, 0]>
}

// CHECK:       module @VPU.SW
// CHECK:         func.func private @info
// CHECK-NOT:     func.func @assume_fill(

// CHECK-LABEL: func.func @assume_fill_spec_0
// CHECK-SAME:      ([[IN:%[^:]+]]: memref<1x64xf16>, [[OUT:%[^:]+]]: memref<1x64xf16>, %{{[^:]+}}: index, [[OFF:%[^:]+]]: index)
// CHECK-DAG:       [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[C13:%.+]] = arith.constant 13 : index
// CHECK-DAG:       [[C9:%.+]] = arith.constant 9 : index
// CHECK-NOT:       memref.dim
// CHECK:           [[OFF_IS_9:%.+]] = arith.cmpi eq, [[OFF]], [[C9]] : index
// CHECK-NEXT:      [[OFF_IS_13:%.+]] = arith.cmpi eq, [[OFF]], [[C13]] : index
// CHECK-NEXT:      [[OFF_OK:%.+]] = arith.ori [[OFF_IS_9]], [[OFF_IS_13]] : i1
// CHECK-NEXT:      llvm.intr.assume [[OFF_OK]] : i1
// CHECK-NEXT:      [[VAL:%.+]] = memref.load [[IN]]{{\[}}[[C0]], [[C0]]] : memref<1x64xf16>
// CHECK-NEXT:      [[TILE:%.+]] = memref.subview [[OUT]]{{\[}}0, [[OFF]]] [1, 5] [1, 1] : memref<1x64xf16> to memref<1x5xf16, strided<[64, 1], offset: ?>>
// CHECK-NEXT:      linalg.fill ins([[VAL]] : f16) outs([[TILE]] : memref<1x5xf16, strided<[64, 1], offset: ?>>)
// CHECK-NEXT:      return
// CHECK:       func.func @BoundedScalarGetsAssume
// CHECK-SAME:      ([[PARAM0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[RET0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[PARAM1:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[RET1:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@assume_fill_spec_0 inputs([[PARAM0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>) outputs([[RET0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:             VPUIP.SW.Kernel.run {attrs = [5, 9]}
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@assume_fill_spec_0 inputs([[PARAM1]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>) outputs([[RET1]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:             VPUIP.SW.Kernel.run {attrs = [5, 13]}

// -----

module @VPU.SW {
  func.func private @info(index, index) -> (index, index)

  func.func @dead_fill(%in: memref<1x64xf16>, %out: memref<1x64xf16>, %size: index, %off: index)
      attributes {kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @info, tilingAxes = [1],
          numSlicedInputs = 1 : i64,
          numTensorInputs = 1 : i64,
          numAuxBuffers = 0 : i64,
          inputSegmentationAxes = []>} {
    %c0 = arith.constant 0 : index
    %v = memref.load %in[%c0, %c0] : memref<1x64xf16>
    linalg.fill ins(%v : f16) outs(%out : memref<1x64xf16>)
    return
  }
}

func.func @DeadScalarsProduceNoClone(%param0: memref<1x64xf16, [@CMX_NN, 0]>, %ret0: memref<1x64xf16, [@CMX_NN, 0]>, %param1: memref<1x64xf16, [@CMX_NN, 0]>, %ret1: memref<1x64xf16, [@CMX_NN, 0]>) -> memref<1x64xf16, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@dead_fill inputs(%param0 as %arg0: memref<1x64xf16, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x64xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x64xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [5, 9]}(%arg0, %arg1) : memref<1x64xf16, [@CMX_NN, 0]>, memref<1x64xf16, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@dead_fill inputs(%param1 as %arg0: memref<1x64xf16, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x64xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x64xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [5, 13]}(%arg0, %arg1) : memref<1x64xf16, [@CMX_NN, 0]>, memref<1x64xf16, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x64xf16, [@CMX_NN, 0]>
}

// CHECK:       module @VPU.SW
// CHECK:         func.func private @info
// CHECK-LABEL: func.func @dead_fill(
// CHECK-SAME:      [[IN:%[^:]+]]: memref<1x64xf16>, [[OUT:%[^:]+]]: memref<1x64xf16>, %{{[^:]+}}: index, %{{[^:]+}}: index)
// CHECK-NEXT:      [[C0:%.+]] = arith.constant 0 : index
// CHECK-NEXT:      [[VAL:%.+]] = memref.load [[IN]]{{\[}}[[C0]], [[C0]]] : memref<1x64xf16>
// CHECK-NEXT:      linalg.fill ins([[VAL]] : f16) outs([[OUT]] : memref<1x64xf16>)
// CHECK-NEXT:      return
// CHECK-NOT:   _spec_
// CHECK:       func.func @DeadScalarsProduceNoClone
// CHECK-SAME:      ([[PARAM0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[RET0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[PARAM1:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[RET1:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@dead_fill inputs([[PARAM0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>) outputs([[RET0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@dead_fill inputs([[PARAM1]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>) outputs([[RET1]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>)

// -----

module @VPU.SW {
  func.func private @info(index, index) -> (index, index)

  func.func @noattrs_fill(%in: memref<1x64xf16>, %out: memref<1x64xf16>, %size: index, %off: index)
      attributes {kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @info, tilingAxes = [1],
          numSlicedInputs = 1 : i64,
          numTensorInputs = 1 : i64,
          numAuxBuffers = 0 : i64,
          inputSegmentationAxes = []>} {
    %c0 = arith.constant 0 : index
    %v = memref.load %in[%c0, %c0] : memref<1x64xf16>
    %tile = memref.subview %out[0, %off] [1, %size] [1, 1]
        : memref<1x64xf16> to memref<1x?xf16, strided<[64, 1], offset: ?>>
    linalg.fill ins(%v : f16) outs(%tile : memref<1x?xf16, strided<[64, 1], offset: ?>>)
    return
  }
}

func.func @MissingAttrsAreUnknown(%param0: memref<1x64xf16, [@CMX_NN, 0]>, %ret0: memref<1x64xf16, [@CMX_NN, 0]>, %param1: memref<1x64xf16, [@CMX_NN, 0]>, %ret1: memref<1x64xf16, [@CMX_NN, 0]>) -> memref<1x64xf16, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@noattrs_fill inputs(%param0 as %arg0: memref<1x64xf16, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x64xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x64xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x64xf16, [@CMX_NN, 0]>, memref<1x64xf16, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@noattrs_fill inputs(%param1 as %arg0: memref<1x64xf16, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x64xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x64xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x64xf16, [@CMX_NN, 0]>, memref<1x64xf16, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x64xf16, [@CMX_NN, 0]>
}

// CHECK:       module @VPU.SW
// CHECK:         func.func private @info
// CHECK-LABEL: func.func @noattrs_fill(
// CHECK-SAME:      [[IN:%[^:]+]]: memref<1x64xf16>, [[OUT:%[^:]+]]: memref<1x64xf16>, [[SIZE:%[^:]+]]: index, [[OFF:%[^:]+]]: index)
// CHECK-NEXT:      [[C0:%.+]] = arith.constant 0 : index
// CHECK-NEXT:      [[VAL:%.+]] = memref.load [[IN]]{{\[}}[[C0]], [[C0]]] : memref<1x64xf16>
// CHECK-NEXT:      [[TILE:%.+]] = memref.subview [[OUT]]{{\[}}0, [[OFF]]] [1, [[SIZE]]] [1, 1] : memref<1x64xf16> to memref<1x?xf16, strided<[64, 1], offset: ?>>
// CHECK-NEXT:      linalg.fill ins([[VAL]] : f16) outs([[TILE]] : memref<1x?xf16, strided<[64, 1], offset: ?>>)
// CHECK-NEXT:      return
// CHECK-NOT:   _spec_
// CHECK:       func.func @MissingAttrsAreUnknown
// CHECK-SAME:      ([[PARAM0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[RET0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[PARAM1:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[RET1:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@noattrs_fill inputs([[PARAM0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>) outputs([[RET0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@noattrs_fill inputs([[PARAM1]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>) outputs([[RET1]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>)

// -----

module @VPU.SW {
  func.func private @info(index, index) -> (index, index)

  func.func @nonint_fill(%in: memref<1x64xf16>, %out: memref<1x64xf16>, %size: index, %off: index)
      attributes {kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @info, tilingAxes = [1],
          numSlicedInputs = 1 : i64,
          numTensorInputs = 1 : i64,
          numAuxBuffers = 0 : i64,
          inputSegmentationAxes = []>} {
    %c0 = arith.constant 0 : index
    %v = memref.load %in[%c0, %c0] : memref<1x64xf16>
    %tile = memref.subview %out[0, %off] [1, %size] [1, 1]
        : memref<1x64xf16> to memref<1x?xf16, strided<[64, 1], offset: ?>>
    linalg.fill ins(%v : f16) outs(%tile : memref<1x?xf16, strided<[64, 1], offset: ?>>)
    return
  }
}

func.func @NonIntegerAttrIsUnknown(%param0: memref<1x64xf16, [@CMX_NN, 0]>, %ret0: memref<1x64xf16, [@CMX_NN, 0]>) -> memref<1x64xf16, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@nonint_fill inputs(%param0 as %arg0: memref<1x64xf16, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x64xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x64xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [5, 1.300000e+01]}(%arg0, %arg1) : memref<1x64xf16, [@CMX_NN, 0]>, memref<1x64xf16, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x64xf16, [@CMX_NN, 0]>
}

// CHECK:       module @VPU.SW
// CHECK:         func.func private @info
// CHECK-NOT:     func.func @nonint_fill(
// CHECK-LABEL: func.func @nonint_fill_spec_0
// CHECK-SAME:      ([[IN:%[^:]+]]: memref<1x64xf16>, [[OUT:%[^:]+]]: memref<1x64xf16>, %{{[^:]+}}: index, [[OFF:%[^:]+]]: index)
// CHECK-NEXT:      [[C0:%.+]] = arith.constant 0 : index
// CHECK-NEXT:      [[VAL:%.+]] = memref.load [[IN]]{{\[}}[[C0]], [[C0]]] : memref<1x64xf16>
// CHECK-NEXT:      [[TILE:%.+]] = memref.subview [[OUT]]{{\[}}0, [[OFF]]] [1, 5] [1, 1] : memref<1x64xf16> to memref<1x5xf16, strided<[64, 1], offset: ?>>
// CHECK-NEXT:      linalg.fill ins([[VAL]] : f16) outs([[TILE]] : memref<1x5xf16, strided<[64, 1], offset: ?>>)
// CHECK-NEXT:      return

// CHECK:       func.func @NonIntegerAttrIsUnknown
// CHECK-SAME:      ([[PARAM0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[RET0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@nonint_fill_spec_0 inputs([[PARAM0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>) outputs([[RET0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:             VPUIP.SW.Kernel.run {attrs = [5, 1.300000e+01]}

// -----

module @VPU.SW {
  func.func private @info(index, index) -> (index, index)

  func.func @alpha_fill(%in: memref<1x64xf16>, %out: memref<1x64xf16>, %alpha: f16, %size: index, %off: index)
      attributes {kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @info, tilingAxes = [1],
          numSlicedInputs = 1 : i64,
          numTensorInputs = 1 : i64,
          numAuxBuffers = 0 : i64,
          inputSegmentationAxes = []>} {
    %tile = memref.subview %out[0, %off] [1, %size] [1, 1]
        : memref<1x64xf16> to memref<1x?xf16, strided<[64, 1], offset: ?>>
    linalg.fill ins(%alpha : f16) outs(%tile : memref<1x?xf16, strided<[64, 1], offset: ?>>)
    return
  }
}

func.func @AttrIndexSkipsPlainScalarArgs(%param0: memref<1x64xf16, [@CMX_NN, 0]>, %ret0: memref<1x64xf16, [@CMX_NN, 0]>) -> memref<1x64xf16, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@alpha_fill inputs(%param0 as %arg0: memref<1x64xf16, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x64xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x64xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [7.000000e+00, 5, 9]}(%arg0, %arg1) : memref<1x64xf16, [@CMX_NN, 0]>, memref<1x64xf16, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x64xf16, [@CMX_NN, 0]>
}

// CHECK:       module @VPU.SW
// CHECK:         func.func private @info
// CHECK-NOT:     func.func @alpha_fill(

// CHECK-LABEL: func.func @alpha_fill_spec_0
// CHECK-SAME:      (%{{[^:]+}}: memref<1x64xf16>, [[OUT:%[^:]+]]: memref<1x64xf16>, [[ALPHA:%[^:]+]]: f16, %{{[^:]+}}: index, %{{[^:]+}}: index)
// CHECK-NEXT:      [[TILE:%.+]] = memref.subview [[OUT]]{{\[}}0, 9] [1, 5] [1, 1] : memref<1x64xf16> to memref<1x5xf16, strided<[64, 1], offset: 9>>
// CHECK-NEXT:      linalg.fill ins([[ALPHA]] : f16) outs([[TILE]] : memref<1x5xf16, strided<[64, 1], offset: 9>>)
// CHECK-NEXT:      return

// CHECK:       func.func @AttrIndexSkipsPlainScalarArgs
// CHECK-SAME:      ([[PARAM0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[RET0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@alpha_fill_spec_0 inputs([[PARAM0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>) outputs([[RET0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>)
// CHECK:             VPUIP.SW.Kernel.run {attrs = [7.000000e+00, 5, 9]}

// -----

module @VPU.SW {
  func.func private @info(index, index) -> (index, index)

  func.func @split_fill(%in: memref<1x?xf16>, %out: memref<1x?xf16>, %size: index, %off: index)
      attributes {kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @info, tilingAxes = [1],
          numSlicedInputs = 1 : i64,
          numTensorInputs = 1 : i64,
          numAuxBuffers = 0 : i64,
          inputSegmentationAxes = []>} {
    %c0 = arith.constant 0 : index
    %v = memref.load %in[%c0, %c0] : memref<1x?xf16>
    %tile = memref.subview %out[0, %off] [1, %size] [1, 1]
        : memref<1x?xf16> to memref<1x?xf16, strided<[?, 1], offset: ?>>
    linalg.fill ins(%v : f16) outs(%tile : memref<1x?xf16, strided<[?, 1], offset: ?>>)
    return
  }
}

func.func @SplitMakesScalarUniform(%param0: memref<1x64xf16, [@CMX_NN, 0]>, %ret0: memref<1x64xf16, [@CMX_NN, 0]>, %param1: memref<1x128xf16, [@CMX_NN, 0]>, %ret1: memref<1x128xf16, [@CMX_NN, 0]>) -> memref<1x64xf16, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@split_fill inputs(%param0 as %arg0: memref<1x64xf16, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x64xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x64xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [24, 7]}(%arg0, %arg1) : memref<1x64xf16, [@CMX_NN, 0]>, memref<1x64xf16, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@split_fill inputs(%param1 as %arg0: memref<1x128xf16, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x128xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x128xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [48, 7]}(%arg0, %arg1) : memref<1x128xf16, [@CMX_NN, 0]>, memref<1x128xf16, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x64xf16, [@CMX_NN, 0]>
}

// CHECK:       module @VPU.SW
// CHECK:         func.func private @info
// CHECK-NOT:     func.func @split_fill(

// NEVER-LABEL: func.func @split_fill_spec_0
// NEVER-SAME:      ([[IN:%[^:]+]]: memref<1x?xf16>, [[OUT:%[^:]+]]: memref<1x?xf16>, [[SIZE:%[^:]+]]: index, %{{[^:]+}}: index)
// NEVER-DAG:       [[C0:%.+]] = arith.constant 0 : index
// NEVER-DAG:       [[C48:%.+]] = arith.constant 48 : index
// NEVER-DAG:       [[C24:%.+]] = arith.constant 24 : index
// NEVER-DAG:       [[C128:%.+]] = arith.constant 128 : index
// NEVER-DAG:       [[C64:%.+]] = arith.constant 64 : index
// NEVER-DAG:       [[C1:%.+]] = arith.constant 1 : index

// NEVER:           [[IN_DIM:%.+]] = memref.dim [[IN]], [[C1]] : memref<1x?xf16>
// NEVER-NEXT:      [[IN_IS_64:%.+]] = arith.cmpi eq, [[IN_DIM]], [[C64]] : index
// NEVER-NEXT:      [[IN_IS_128:%.+]] = arith.cmpi eq, [[IN_DIM]], [[C128]] : index
// NEVER-NEXT:      [[IN_OK:%.+]] = arith.ori [[IN_IS_64]], [[IN_IS_128]] : i1
// NEVER-NEXT:      llvm.intr.assume [[IN_OK]] : i1
// NEVER-NEXT:      [[OUT_DIM:%.+]] = memref.dim [[OUT]], [[C1]] : memref<1x?xf16>
// NEVER-NEXT:      [[OUT_IS_64:%.+]] = arith.cmpi eq, [[OUT_DIM]], [[C64]] : index
// NEVER-NEXT:      [[OUT_IS_128:%.+]] = arith.cmpi eq, [[OUT_DIM]], [[C128]] : index
// NEVER-NEXT:      [[OUT_OK:%.+]] = arith.ori [[OUT_IS_64]], [[OUT_IS_128]] : i1
// NEVER-NEXT:      llvm.intr.assume [[OUT_OK]] : i1

// NEVER-NEXT:      [[SIZE_IS_24:%.+]] = arith.cmpi eq, [[SIZE]], [[C24]] : index
// NEVER-NEXT:      [[SIZE_IS_48:%.+]] = arith.cmpi eq, [[SIZE]], [[C48]] : index
// NEVER-NEXT:      [[SIZE_OK:%.+]] = arith.ori [[SIZE_IS_24]], [[SIZE_IS_48]] : i1
// NEVER-NEXT:      llvm.intr.assume [[SIZE_OK]] : i1
// NEVER-NEXT:      [[VAL:%.+]] = memref.load [[IN]]{{\[}}[[C0]], [[C0]]] : memref<1x?xf16>
// NEVER-NEXT:      [[TILE:%.+]] = memref.subview [[OUT]]{{\[}}0, 7] [1, [[SIZE]]] [1, 1] : memref<1x?xf16> to memref<1x?xf16, strided<[?, 1], offset: 7>>
// NEVER-NEXT:      linalg.fill ins([[VAL]] : f16) outs([[TILE]] : memref<1x?xf16, strided<[?, 1], offset: 7>>)
// NEVER-NEXT:      return

// SPLIT-LABEL: func.func @split_fill_spec_1
// SPLIT-SAME:      ([[IN_64:%[^:]+]]: memref<1x64xf16>, [[OUT_64:%[^:]+]]: memref<1x64xf16>, %{{[^:]+}}: index, %{{[^:]+}}: index)
// SPLIT-NEXT:      [[C0_64:%.+]] = arith.constant 0 : index
// SPLIT-NEXT:      [[VAL_64:%.+]] = memref.load [[IN_64]]{{\[}}[[C0_64]], [[C0_64]]] : memref<1x64xf16>
// SPLIT-NEXT:      [[TILE_64:%.+]] = memref.subview [[OUT_64]]{{\[}}0, 7] [1, 24] [1, 1] : memref<1x64xf16> to memref<1x24xf16, strided<[64, 1], offset: 7>>
// SPLIT-NEXT:      linalg.fill ins([[VAL_64]] : f16) outs([[TILE_64]] : memref<1x24xf16, strided<[64, 1], offset: 7>>)
// SPLIT-NEXT:      return

// SPLIT-LABEL: func.func @split_fill_spec_0
// SPLIT-SAME:      ([[IN_128:%[^:]+]]: memref<1x128xf16>, [[OUT_128:%[^:]+]]: memref<1x128xf16>, %{{[^:]+}}: index, %{{[^:]+}}: index)
// SPLIT-NEXT:      [[C0_128:%.+]] = arith.constant 0 : index
// SPLIT-NEXT:      [[VAL_128:%.+]] = memref.load [[IN_128]]{{\[}}[[C0_128]], [[C0_128]]] : memref<1x128xf16>
// SPLIT-NEXT:      [[TILE_128:%.+]] = memref.subview [[OUT_128]]{{\[}}0, 7] [1, 48] [1, 1] : memref<1x128xf16> to memref<1x48xf16, strided<[128, 1], offset: 7>>
// SPLIT-NEXT:      linalg.fill ins([[VAL_128]] : f16) outs([[TILE_128]] : memref<1x48xf16, strided<[128, 1], offset: 7>>)
// SPLIT-NEXT:      return

// CHECK:       func.func @SplitMakesScalarUniform
// CHECK-SAME:      ([[PARAM0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[RET0:%[^:]+]]: memref<1x64xf16, [@CMX_NN, 0]>, [[PARAM1:%[^:]+]]: memref<1x128xf16, [@CMX_NN, 0]>, [[RET1:%[^:]+]]: memref<1x128xf16, [@CMX_NN, 0]>)

// NEVER:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@split_fill_spec_0 inputs([[PARAM0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>) outputs([[RET0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>)
// NEVER:             VPUIP.SW.Kernel.run {attrs = [24, 7]}
// NEVER:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@split_fill_spec_0 inputs([[PARAM1]] as {{[^:]+}}: memref<1x128xf16, [@CMX_NN, 0]>) outputs([[RET1]] as {{[^:]+}}: memref<1x128xf16, [@CMX_NN, 0]>)
// NEVER:             VPUIP.SW.Kernel.run {attrs = [48, 7]}

// SPLIT:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@split_fill_spec_1 inputs([[PARAM0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>) outputs([[RET0]] as {{[^:]+}}: memref<1x64xf16, [@CMX_NN, 0]>)
// SPLIT:             VPUIP.SW.Kernel.run {attrs = [24, 7]}
// SPLIT:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@split_fill_spec_0 inputs([[PARAM1]] as {{[^:]+}}: memref<1x128xf16, [@CMX_NN, 0]>) outputs([[RET1]] as {{[^:]+}}: memref<1x128xf16, [@CMX_NN, 0]>)
// SPLIT:             VPUIP.SW.Kernel.run {attrs = [48, 7]}
