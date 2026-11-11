//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --specialize-sw-kernels="heuristic=never" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// Ensure callsite types (order, strides, shapes) are properly read and propagated
// to the specialized function.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func @strided_sqrt(%in: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %out: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%out : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @StridedNHWCOperands(%param0: memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>, %ret: memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>) -> memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@strided_sqrt inputs(%param0 as %arg0: memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>, memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @strided_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x8x4x16xf16, strided<[1024, 128, 32, 1]>>, [[OUT:%.+]]: memref<1x8x4x16xf16, strided<[1024, 128, 32, 1]>>)
// CHECK-NEXT:      linalg.generic
// CHECK-SAME:          ins([[IN]] : memref<1x8x4x16xf16, strided<[1024, 128, 32, 1]>>)
// CHECK-SAME:          outs([[OUT]] : memref<1x8x4x16xf16, strided<[1024, 128, 32, 1]>>)
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@strided_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func @compact_sqrt(%in: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %out: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%out : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @CompactNHWCOperands(%param0: memref<1x16x8x4xf16, {order = #NHWC}, [@CMX_NN, 0]>, %ret: memref<1x16x8x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x8x4xf16, {order = #NHWC}, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@compact_sqrt inputs(%param0 as %arg0: memref<1x16x8x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<1x16x8x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x16x8x4xf16, {order = #NHWC}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x16x8x4xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x16x8x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<1x16x8x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @compact_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x8x4x16xf16, strided<[512, 64, 16, 1]>>, [[OUT:%.+]]: memref<1x8x4x16xf16, strided<[512, 64, 16, 1]>>)
// CHECK-NOT:   llvm.intr.assume
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@compact_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @mixed_sqrt(%in: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %out: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%out : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @NonUniformStrides(%param0: memref<1x16x8x4xf16, {order = #NCHW, strides = [2048, 128, 8, 1]}, [@CMX_NN, 0]>, %ret0: memref<1x16x8x4xf16, {order = #NCHW, strides = [2048, 128, 8, 1]}, [@CMX_NN, 0]>, %param1: memref<1x16x8x4xf16, {order = #NCHW, strides = [4096, 256, 8, 1]}, [@CMX_NN, 0]>, %ret1: memref<1x16x8x4xf16, {order = #NCHW, strides = [4096, 256, 8, 1]}, [@CMX_NN, 0]>) -> memref<1x16x8x4xf16, {order = #NCHW, strides = [2048, 128, 8, 1]}, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@mixed_sqrt inputs(%param0 as %arg0: memref<1x16x8x4xf16, {order = #NCHW, strides = [2048, 128, 8, 1]}, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x16x8x4xf16, {order = #NCHW, strides = [2048, 128, 8, 1]}, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x16x8x4xf16, {order = #NCHW, strides = [2048, 128, 8, 1]}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x16x8x4xf16, {order = #NCHW, strides = [2048, 128, 8, 1]}, [@CMX_NN, 0]>, memref<1x16x8x4xf16, {order = #NCHW, strides = [2048, 128, 8, 1]}, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@mixed_sqrt inputs(%param1 as %arg0: memref<1x16x8x4xf16, {order = #NCHW, strides = [4096, 256, 8, 1]}, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x16x8x4xf16, {order = #NCHW, strides = [4096, 256, 8, 1]}, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x16x8x4xf16, {order = #NCHW, strides = [4096, 256, 8, 1]}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x16x8x4xf16, {order = #NCHW, strides = [4096, 256, 8, 1]}, [@CMX_NN, 0]>, memref<1x16x8x4xf16, {order = #NCHW, strides = [4096, 256, 8, 1]}, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x16x8x4xf16, {order = #NCHW, strides = [2048, 128, 8, 1]}, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @mixed_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x16x8x4xf16, strided<[?, ?, 8, 1]>>, [[OUT:%.+]]: memref<1x16x8x4xf16, strided<[?, ?, 8, 1]>>)
// CHECK-NOT:   llvm.intr.assume

// CHECK:       func.func @NonUniformStrides
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@mixed_sqrt_spec_0
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@mixed_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func @packed_sqrt(%in: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16>) outs(%out : memref<1x?x?x?xf16>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @PackedCalleeStridedCallsite(%param0: memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>, %ret: memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>) -> memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@packed_sqrt inputs(%param0 as %arg0: memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>, memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32]}, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @packed_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x8x4x16xf16>, [[OUT:%.+]]: memref<1x8x4x16xf16>)
// CHECK-NOT:   strided<
// CHECK-NOT:   llvm.intr.assume

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @dyn_sqrt(%in: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %out: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%out : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @DynamicOuterDim(%param0: memref<1x?x8x4xf16, [@CMX_NN, 0]>, %ret: memref<1x?x8x4xf16, [@CMX_NN, 0]>) -> memref<1x?x8x4xf16, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@dyn_sqrt inputs(%param0 as %arg0: memref<1x?x8x4xf16, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<1x?x8x4xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x?x8x4xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x?x8x4xf16, [@CMX_NN, 0]>, memref<1x?x8x4xf16, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<1x?x8x4xf16, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @dyn_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x?x8x4xf16, strided<[?, 32, 4, 1]>>, [[OUT:%.+]]: memref<1x?x8x4xf16, strided<[?, 32, 4, 1]>>)
// CHECK-NOT:   llvm.intr.assume

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func @perarg_sqrt(%in: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %out: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%out : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @PerArgOrders(%param0: memref<1x16x8x4xf16, {order = #NHWC}, [@CMX_NN, 0]>, %ret: memref<1x8x4x16xf16, [@CMX_NN, 0]>) -> memref<1x8x4x16xf16, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@perarg_sqrt inputs(%param0 as %arg0: memref<1x16x8x4xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<1x8x4x16xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x8x4x16xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x16x8x4xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x8x4x16xf16, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<1x8x4x16xf16, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @perarg_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x8x4x16xf16, strided<[512, 64, 16, 1]>>, [[OUT:%.+]]: memref<1x8x4x16xf16, strided<[512, 64, 16, 1]>>)
// CHECK-NOT:   llvm.intr.assume

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @assume_sqrt(%in: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %out: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%out : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @BoundedInnermostDim(%param0: memref<1x16x8x4xf16, [@CMX_NN, 0]>, %ret0: memref<1x16x8x4xf16, [@CMX_NN, 0]>, %param1: memref<1x16x8x32xf16, [@CMX_NN, 0]>, %ret1: memref<1x16x8x32xf16, [@CMX_NN, 0]>) -> memref<1x16x8x4xf16, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@assume_sqrt inputs(%param0 as %arg0: memref<1x16x8x4xf16, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x16x8x4xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x16x8x4xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x16x8x4xf16, [@CMX_NN, 0]>, memref<1x16x8x4xf16, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@assume_sqrt inputs(%param1 as %arg0: memref<1x16x8x32xf16, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x16x8x32xf16, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x16x8x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x16x8x32xf16, [@CMX_NN, 0]>, memref<1x16x8x32xf16, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x16x8x4xf16, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @assume_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x16x8x?xf16, strided<[?, ?, ?, 1]>>, [[OUT:%.+]]: memref<1x16x8x?xf16, strided<[?, ?, ?, 1]>>)
// CHECK-DAG:       [[C32:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[C4:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[C3:%.+]] = arith.constant 3 : index
// CHECK:           [[IN_DIM:%.+]] = memref.dim [[IN]], [[C3]]
// CHECK:           [[IN_IS_4:%.+]] = arith.cmpi eq, [[IN_DIM]], [[C4]]
// CHECK:           [[IN_IS_32:%.+]] = arith.cmpi eq, [[IN_DIM]], [[C32]]
// CHECK:           [[IN_EITHER:%.+]] = arith.ori [[IN_IS_4]], [[IN_IS_32]]
// CHECK:           llvm.intr.assume [[IN_EITHER]]
// CHECK:           [[OUT_DIM:%.+]] = memref.dim [[OUT]], [[C3]]
// CHECK:           [[OUT_IS_4:%.+]] = arith.cmpi eq, [[OUT_DIM]], [[C4]]
// CHECK:           [[OUT_IS_32:%.+]] = arith.cmpi eq, [[OUT_DIM]], [[C32]]
// CHECK:           [[OUT_EITHER:%.+]] = arith.ori [[OUT_IS_4]], [[OUT_IS_32]]
// CHECK:           llvm.intr.assume [[OUT_EITHER]]

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// A callsite whose layout is a StridedLayoutAttr with everything dynamic.
module @VPU.SW {
  func.func @dyn_strided_sqrt(%in: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %out: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%out : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @DynamicStridedCallsite(%param0: memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>, %ret: memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>> {
  %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@dyn_strided_sqrt inputs(%param0 as %arg0: memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>) outputs(%ret as %arg1: memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>) on tile 0 list 0 -> memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>{
    VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>
  }
  return %res : memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>
}

// CHECK-LABEL: func.func @dyn_strided_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>, [[OUT:%.+]]: memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>)
// CHECK-NEXT:      linalg.generic
// CHECK-SAME:          ins([[IN]] : memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>)
// CHECK-SAME:          outs([[OUT]] : memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>)
// CHECK-NOT:   llvm.intr.assume

// CHECK:       func.func @DynamicStridedCallsite
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@dyn_strided_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// A fully static StridedLayoutAttr callsite.

module @VPU.SW {
  func.func @static_strided_sqrt(%in: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %out: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%out : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @StaticStridedCallsite(%param0: memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: 57>>, %ret: memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: 57>>) -> memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: 57>> {
  %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@static_strided_sqrt inputs(%param0 as %arg0: memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: 57>>) outputs(%ret as %arg1: memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: 57>>) on tile 0 list 0 -> memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: 57>>{
    VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: 57>>, memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: 57>>
  }
  return %res : memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: 57>>
}

// CHECK-LABEL: func.func @static_strided_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: 57>>, [[OUT:%.+]]: memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: 57>>)
// CHECK-NEXT:      linalg.generic
// CHECK-SAME:          ins([[IN]] : memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: 57>>)
// CHECK-SAME:          outs([[OUT]] : memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: 57>>)
// CHECK-NOT:   llvm.intr.assume

// CHECK:       func.func @StaticStridedCallsite
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@static_strided_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Two StridedLayoutAttr callsites that agree on the two innermost strides and disagree on
// the two outermost ones.

module @VPU.SW {
  func.func @mixed_strided_sqrt(%in: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %out: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%out : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @NonUniformStridedCallsites(%param0: memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: ?>>, %ret0: memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: ?>>, %param1: memref<1x8x4x16xf16, strided<[7168, 896, 96, 3], offset: ?>>, %ret1: memref<1x8x4x16xf16, strided<[7168, 896, 96, 3], offset: ?>>) -> memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: ?>> {
  %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@mixed_strided_sqrt inputs(%param0 as %arg0: memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: ?>>) outputs(%ret0 as %arg1: memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: ?>>) on tile 0 list 0 -> memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: ?>>{
    VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: ?>>, memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: ?>>
  }
  %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@mixed_strided_sqrt inputs(%param1 as %arg0: memref<1x8x4x16xf16, strided<[7168, 896, 96, 3], offset: ?>>) outputs(%ret1 as %arg1: memref<1x8x4x16xf16, strided<[7168, 896, 96, 3], offset: ?>>) on tile 0 list 0 -> memref<1x8x4x16xf16, strided<[7168, 896, 96, 3], offset: ?>>{
    VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x8x4x16xf16, strided<[7168, 896, 96, 3], offset: ?>>, memref<1x8x4x16xf16, strided<[7168, 896, 96, 3], offset: ?>>
  }
  return %res0 : memref<1x8x4x16xf16, strided<[9216, 1152, 96, 3], offset: ?>>
}

// CHECK-LABEL: func.func @mixed_strided_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x8x4x16xf16, strided<[?, ?, 96, 3], offset: ?>>, [[OUT:%.+]]: memref<1x8x4x16xf16, strided<[?, ?, 96, 3], offset: ?>>)
// CHECK-NEXT:      linalg.generic
// CHECK-SAME:          ins([[IN]] : memref<1x8x4x16xf16, strided<[?, ?, 96, 3], offset: ?>>)
// CHECK-SAME:          outs([[OUT]] : memref<1x8x4x16xf16, strided<[?, ?, 96, 3], offset: ?>>)
// CHECK-NOT:   llvm.intr.assume

// CHECK:       func.func @NonUniformStridedCallsites
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@mixed_strided_sqrt_spec_0
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@mixed_strided_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// A MemRefAttr callsite carrying both explicit strides and a base offset. The offset reaches the
// specialized type unchanged, while the strides are permuted from logical into memory order.

module @VPU.SW {
  func.func @offset_nhwc_sqrt(%in: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %out: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%out : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @OffsetNHWCOperands(%param0: memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32], offset = 57}, [@CMX_NN, 0]>, %ret: memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32], offset = 57}, [@CMX_NN, 0]>) -> memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32], offset = 57}, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@offset_nhwc_sqrt inputs(%param0 as %arg0: memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32], offset = 57}, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32], offset = 57}, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32], offset = 57}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32], offset = 57}, [@CMX_NN, 0]>, memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32], offset = 57}, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<1x16x8x4xf16, {order = #NHWC, strides = [1024, 1, 128, 32], offset = 57}, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @offset_nhwc_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x8x4x16xf16, strided<[1024, 128, 32, 1], offset: 57>>, [[OUT:%.+]]: memref<1x8x4x16xf16, strided<[1024, 128, 32, 1], offset: 57>>)
// CHECK-NEXT:      linalg.generic
// CHECK-SAME:          ins([[IN]] : memref<1x8x4x16xf16, strided<[1024, 128, 32, 1], offset: 57>>)
// CHECK-SAME:          outs([[OUT]] : memref<1x8x4x16xf16, strided<[1024, 128, 32, 1], offset: 57>>)
// CHECK-NOT:   llvm.intr.assume

// CHECK:       func.func @OffsetNHWCOperands
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@offset_nhwc_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// A MemRefAttr callsite that carries a base offset but no explicit strides: the offset has to be
// picked up on the compact path too, where the strides are derived from the shape.

module @VPU.SW {
  func.func @offset_compact_sqrt(%in: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %out: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%out : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @OffsetCompactOperands(%param0: memref<1x16x8x4xf16, {order = #NCHW, offset = 57}, [@CMX_NN, 0]>, %ret: memref<1x16x8x4xf16, {order = #NCHW, offset = 57}, [@CMX_NN, 0]>) -> memref<1x16x8x4xf16, {order = #NCHW, offset = 57}, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@offset_compact_sqrt inputs(%param0 as %arg0: memref<1x16x8x4xf16, {order = #NCHW, offset = 57}, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<1x16x8x4xf16, {order = #NCHW, offset = 57}, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x16x8x4xf16, {order = #NCHW, offset = 57}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x16x8x4xf16, {order = #NCHW, offset = 57}, [@CMX_NN, 0]>, memref<1x16x8x4xf16, {order = #NCHW, offset = 57}, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<1x16x8x4xf16, {order = #NCHW, offset = 57}, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @offset_compact_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x16x8x4xf16, strided<[512, 32, 4, 1], offset: 57>>, [[OUT:%.+]]: memref<1x16x8x4xf16, strided<[512, 32, 4, 1], offset: 57>>)
// CHECK-NOT:   llvm.intr.assume

// CHECK:       func.func @OffsetCompactOperands
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@offset_compact_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// A dynamic MemRefAttr callsite with bounds. Bounds are what a dynamic buffer's compact strides
// are allocated against, so every stride is statically known even though the dim is not.

module @VPU.SW {
  func.func @bounded_sqrt(%in: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %out: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%out : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @BoundedDynamicDim(%param0: memref<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 32]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>, %ret: memref<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 32]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>) -> memref<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 32]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@bounded_sqrt inputs(%param0 as %arg0: memref<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 32]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 32]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 32]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 32]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>, memref<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 32]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 32]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @bounded_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x16x8x?xf16, strided<[4096, 256, 32, 1]>>, [[OUT:%.+]]: memref<1x16x8x?xf16, strided<[4096, 256, 32, 1]>>)
// CHECK-NOT:   llvm.intr.assume

// CHECK:       func.func @BoundedDynamicDim
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@bounded_sqrt_spec_0
