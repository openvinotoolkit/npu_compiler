//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --specialize-sw-kernels %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --specialize-sw-kernels="heuristic=always" %s | FileCheck %s --check-prefix=ALWAYS
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @ideal_sqrt(%in: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16>) outs(%out : memref<1x?x?x?xf16>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @IdealParentIsALeaf(%param0: memref<1x2x4x256xf16, #NCHW, [@CMX_NN, 0]>, %ret0: memref<1x2x4x256xf16, #NCHW, [@CMX_NN, 0]>, %param1: memref<1x3x4x256xf16, #NCHW, [@CMX_NN, 0]>, %ret1: memref<1x3x4x256xf16, #NCHW, [@CMX_NN, 0]>) -> memref<1x2x4x256xf16, #NCHW, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@ideal_sqrt inputs(%param0 as %arg0: memref<1x2x4x256xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x2x4x256xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x2x4x256xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x2x4x256xf16, #NCHW, [@CMX_NN, 0]>, memref<1x2x4x256xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@ideal_sqrt inputs(%param1 as %arg0: memref<1x3x4x256xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x3x4x256xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x3x4x256xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x3x4x256xf16, #NCHW, [@CMX_NN, 0]>, memref<1x3x4x256xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x2x4x256xf16, #NCHW, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @ideal_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x?x4x256xf16>, [[OUT:%.+]]: memref<1x?x4x256xf16>)
// CHECK-DAG:       [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[C2:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[C1:%.+]] = arith.constant 1 : index
// CHECK:           [[IN_DIM:%.+]] = memref.dim [[IN]], [[C1]]
// CHECK:           [[IN_IS_2:%.+]] = arith.cmpi eq, [[IN_DIM]], [[C2]]
// CHECK:           [[IN_IS_3:%.+]] = arith.cmpi eq, [[IN_DIM]], [[C3]]
// CHECK:           [[IN_EITHER:%.+]] = arith.ori [[IN_IS_2]], [[IN_IS_3]]
// CHECK:           llvm.intr.assume [[IN_EITHER]]
// CHECK:           [[OUT_DIM:%.+]] = memref.dim [[OUT]], [[C1]]
// CHECK:           [[OUT_IS_2:%.+]] = arith.cmpi eq, [[OUT_DIM]], [[C2]]
// CHECK:           [[OUT_IS_3:%.+]] = arith.cmpi eq, [[OUT_DIM]], [[C3]]
// CHECK:           [[OUT_EITHER:%.+]] = arith.ori [[OUT_IS_2]], [[OUT_IS_3]]
// CHECK:           llvm.intr.assume [[OUT_EITHER]]
// CHECK:           linalg.generic
// CHECK-SAME:          ins([[IN]] : memref<1x?x4x256xf16>)
// CHECK-SAME:          outs([[OUT]] : memref<1x?x4x256xf16>)

// CHECK:       func.func @IdealParentIsALeaf
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@ideal_sqrt_spec_0
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@ideal_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @peel_sqrt(%in: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16>) outs(%out : memref<1x?x?x?xf16>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @SubGranularParentSplits(%param0: memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>, %ret0: memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>, %param1: memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>, %ret1: memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>) -> memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@peel_sqrt inputs(%param0 as %arg0: memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@peel_sqrt inputs(%param1 as %arg0: memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @peel_sqrt_spec_1
// CHECK-SAME:      (%{{.+}}: memref<1x1x8x256xf16>, %{{.+}}: memref<1x1x8x256xf16>)
// CHECK-NOT:       llvm.intr.assume
// CHECK-LABEL: func.func @peel_sqrt_spec_0
// CHECK-SAME:      (%{{.+}}: memref<1x1x8x128xf16>, %{{.+}}: memref<1x1x8x128xf16>)
// CHECK-NOT:       llvm.intr.assume

// CHECK:       func.func @SubGranularParentSplits
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@peel_sqrt_spec_1
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@peel_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @recurse_sqrt(%in: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16>) outs(%out : memref<1x?x?x?xf16>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @RecursiveSplitThreeValues(%param0: memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>, %ret0: memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>, %param1: memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>, %ret1: memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>, %param2: memref<1x1x8x64xf16, #NCHW, [@CMX_NN, 0]>, %ret2: memref<1x1x8x64xf16, #NCHW, [@CMX_NN, 0]>) -> memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@recurse_sqrt inputs(%param0 as %arg0: memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@recurse_sqrt inputs(%param1 as %arg0: memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@recurse_sqrt inputs(%param2 as %arg0: memref<1x1x8x64xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret2 as %arg1: memref<1x1x8x64xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x8x64xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x8x64xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x8x64xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @recurse_sqrt_spec_2
// CHECK-SAME:      (%{{.+}}: memref<1x1x8x256xf16>, %{{.+}}: memref<1x1x8x256xf16>)
// CHECK-NOT:       llvm.intr.assume
// CHECK-LABEL: func.func @recurse_sqrt_spec_1
// CHECK-SAME:      (%{{.+}}: memref<1x1x8x128xf16>, %{{.+}}: memref<1x1x8x128xf16>)
// CHECK-NOT:       llvm.intr.assume
// CHECK-LABEL: func.func @recurse_sqrt_spec_0
// CHECK-SAME:      (%{{.+}}: memref<1x1x8x64xf16>, %{{.+}}: memref<1x1x8x64xf16>)
// CHECK-NOT:       llvm.intr.assume
// CHECK:       func.func @RecursiveSplitThreeValues
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@recurse_sqrt_spec_2
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@recurse_sqrt_spec_1
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@recurse_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @below_sqrt(%in: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16>) outs(%out : memref<1x?x?x?xf16>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @SavingBelowExtraKernelCost(%param0: memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, %ret0: memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, %param1: memref<1x1x2x224xf16, #NCHW, [@CMX_NN, 0]>, %ret1: memref<1x1x2x224xf16, #NCHW, [@CMX_NN, 0]>) -> memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@below_sqrt inputs(%param0 as %arg0: memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@below_sqrt inputs(%param1 as %arg0: memref<1x1x2x224xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x1x2x224xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x2x224xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x2x224xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x2x224xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @below_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x1x?x224xf16>, [[OUT:%.+]]: memref<1x1x?x224xf16>)
// CHECK-DAG:       [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[C2:%.+]] = arith.constant 2 : index
// CHECK:           [[IN_DIM:%.+]] = memref.dim [[IN]], [[C2]]
// CHECK:           [[IN_IS_1:%.+]] = arith.cmpi eq, [[IN_DIM]], [[C1]]
// CHECK:           [[IN_IS_2:%.+]] = arith.cmpi eq, [[IN_DIM]], [[C2]]
// CHECK:           [[IN_EITHER:%.+]] = arith.ori [[IN_IS_1]], [[IN_IS_2]]
// CHECK:           llvm.intr.assume [[IN_EITHER]]
// CHECK:           [[OUT_DIM:%.+]] = memref.dim [[OUT]], [[C2]]
// CHECK:           [[OUT_IS_1:%.+]] = arith.cmpi eq, [[OUT_DIM]], [[C1]]
// CHECK:           [[OUT_IS_2:%.+]] = arith.cmpi eq, [[OUT_DIM]], [[C2]]
// CHECK:           [[OUT_EITHER:%.+]] = arith.ori [[OUT_IS_1]], [[OUT_IS_2]]
// CHECK:           llvm.intr.assume [[OUT_EITHER]]
// CHECK:       func.func @SavingBelowExtraKernelCost
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@below_sqrt_spec_0
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@below_sqrt_spec_0

// ALWAYS-LABEL: func.func @below_sqrt_spec_1
// ALWAYS-SAME:      (%{{.+}}: memref<1x1x1x224xf16>, %{{.+}}: memref<1x1x1x224xf16>)
// ALWAYS-NOT:       llvm.intr.assume
// ALWAYS-LABEL: func.func @below_sqrt_spec_0
// ALWAYS-SAME:      (%{{.+}}: memref<1x1x2x224xf16>, %{{.+}}: memref<1x1x2x224xf16>)
// ALWAYS-NOT:       llvm.intr.assume
// ALWAYS:       func.func @SavingBelowExtraKernelCost
// ALWAYS:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@below_sqrt_spec_1
// ALWAYS:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@below_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @boundary_sqrt(%in: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16>) outs(%out : memref<1x?x?x?xf16>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @SavingAtExtraKernelCost(%param0: memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>, %ret0: memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>, %param1: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>, %ret1: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>) -> memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@boundary_sqrt inputs(%param0 as %arg0: memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@boundary_sqrt inputs(%param1 as %arg0: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @boundary_sqrt_spec_0
// CHECK-SAME:      ([[IN:%.+]]: memref<1x1x?x32xf16>, [[OUT:%.+]]: memref<1x1x?x32xf16>)
// CHECK-DAG:       [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[C2:%.+]] = arith.constant 2 : index
// CHECK:           [[IN_DIM:%.+]] = memref.dim [[IN]], [[C2]]
// CHECK:           [[IN_IS_1:%.+]] = arith.cmpi eq, [[IN_DIM]], [[C1]]
// CHECK:           [[IN_IS_2:%.+]] = arith.cmpi eq, [[IN_DIM]], [[C2]]
// CHECK:           [[IN_EITHER:%.+]] = arith.ori [[IN_IS_1]], [[IN_IS_2]]
// CHECK:           llvm.intr.assume [[IN_EITHER]]
// CHECK:           [[OUT_DIM:%.+]] = memref.dim [[OUT]], [[C2]]
// CHECK:           [[OUT_IS_1:%.+]] = arith.cmpi eq, [[OUT_DIM]], [[C1]]
// CHECK:           [[OUT_IS_2:%.+]] = arith.cmpi eq, [[OUT_DIM]], [[C2]]
// CHECK:           [[OUT_EITHER:%.+]] = arith.ori [[OUT_IS_1]], [[OUT_IS_2]]
// CHECK:           llvm.intr.assume [[OUT_EITHER]]
// CHECK:       func.func @SavingAtExtraKernelCost
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@boundary_sqrt_spec_0
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@boundary_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @volume_sqrt(%in: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16>) outs(%out : memref<1x?x?x?xf16>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @VolumeAccumulatesOverCallsites(%param0: memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>, %ret0: memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>, %param1: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>, %ret1: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>, %param2: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>, %ret2: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>) -> memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@volume_sqrt inputs(%param0 as %arg0: memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@volume_sqrt inputs(%param1 as %arg0: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@volume_sqrt inputs(%param2 as %arg0: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret2 as %arg1: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @volume_sqrt_spec_1
// CHECK-SAME:      (%{{.+}}: memref<1x1x2x32xf16>, %{{.+}}: memref<1x1x2x32xf16>)
// CHECK-NOT:       llvm.intr.assume
// CHECK-LABEL: func.func @volume_sqrt_spec_0
// CHECK-SAME:      (%{{.+}}: memref<1x1x1x32xf16>, %{{.+}}: memref<1x1x1x32xf16>)
// CHECK-NOT:       llvm.intr.assume

// CHECK:       func.func @VolumeAccumulatesOverCallsites
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@volume_sqrt_spec_0
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@volume_sqrt_spec_1
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@volume_sqrt_spec_1

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @unaligned_sqrt(%in: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16>) outs(%out : memref<1x?x?x?xf16>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @UnalignedParentFlipsDecision(%param0: memref<1x1x1x232xf16, #NCHW, [@CMX_NN, 0]>, %ret0: memref<1x1x1x232xf16, #NCHW, [@CMX_NN, 0]>, %param1: memref<1x1x2x232xf16, #NCHW, [@CMX_NN, 0]>, %ret1: memref<1x1x2x232xf16, #NCHW, [@CMX_NN, 0]>) -> memref<1x1x1x232xf16, #NCHW, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@unaligned_sqrt inputs(%param0 as %arg0: memref<1x1x1x232xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x1x1x232xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x1x232xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x1x232xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x232xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@unaligned_sqrt inputs(%param1 as %arg0: memref<1x1x2x232xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x1x2x232xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x2x232xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x2x232xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x2x232xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x1x1x232xf16, #NCHW, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @unaligned_sqrt_spec_1
// CHECK-SAME:      (%{{.+}}: memref<1x1x2x232xf16>, %{{.+}}: memref<1x1x2x232xf16>)
// CHECK-NOT:       llvm.intr.assume
// CHECK-LABEL: func.func @unaligned_sqrt_spec_0
// CHECK-SAME:      (%{{.+}}: memref<1x1x1x232xf16>, %{{.+}}: memref<1x1x1x232xf16>)
// CHECK-NOT:       llvm.intr.assume
// CHECK:       func.func @UnalignedParentFlipsDecision
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@unaligned_sqrt_spec_0
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@unaligned_sqrt_spec_1

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @order_sqrt(%in: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16>) outs(%out : memref<1x?x?x?xf16>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @InnermostCandidateFirst(%param0: memref<1x2x8x256xf16, #NCHW, [@CMX_NN, 0]>, %ret0: memref<1x2x8x256xf16, #NCHW, [@CMX_NN, 0]>, %param1: memref<1x3x8x256xf16, #NCHW, [@CMX_NN, 0]>, %ret1: memref<1x3x8x256xf16, #NCHW, [@CMX_NN, 0]>, %param2: memref<1x2x8x128xf16, #NCHW, [@CMX_NN, 0]>, %ret2: memref<1x2x8x128xf16, #NCHW, [@CMX_NN, 0]>, %param3: memref<1x3x8x128xf16, #NCHW, [@CMX_NN, 0]>, %ret3: memref<1x3x8x128xf16, #NCHW, [@CMX_NN, 0]>) -> memref<1x2x8x256xf16, #NCHW, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@order_sqrt inputs(%param0 as %arg0: memref<1x2x8x256xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x2x8x256xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x2x8x256xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x2x8x256xf16, #NCHW, [@CMX_NN, 0]>, memref<1x2x8x256xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@order_sqrt inputs(%param1 as %arg0: memref<1x3x8x256xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x3x8x256xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x3x8x256xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x3x8x256xf16, #NCHW, [@CMX_NN, 0]>, memref<1x3x8x256xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@order_sqrt inputs(%param2 as %arg0: memref<1x2x8x128xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret2 as %arg1: memref<1x2x8x128xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x2x8x128xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x2x8x128xf16, #NCHW, [@CMX_NN, 0]>, memref<1x2x8x128xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@order_sqrt inputs(%param3 as %arg0: memref<1x3x8x128xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret3 as %arg1: memref<1x3x8x128xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x3x8x128xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x3x8x128xf16, #NCHW, [@CMX_NN, 0]>, memref<1x3x8x128xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x2x8x256xf16, #NCHW, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @order_sqrt_spec_1
// CHECK-SAME:      ([[IN1:%.+]]: memref<1x?x8x256xf16>, [[OUT1:%.+]]: memref<1x?x8x256xf16>)
// CHECK-DAG:       [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[C2:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[C1:%.+]] = arith.constant 1 : index
// CHECK:           [[IN1_DIM:%.+]] = memref.dim [[IN1]], [[C1]]
// CHECK:           [[IN1_IS_2:%.+]] = arith.cmpi eq, [[IN1_DIM]], [[C2]]
// CHECK:           [[IN1_IS_3:%.+]] = arith.cmpi eq, [[IN1_DIM]], [[C3]]
// CHECK:           [[IN1_EITHER:%.+]] = arith.ori [[IN1_IS_2]], [[IN1_IS_3]]
// CHECK:           llvm.intr.assume [[IN1_EITHER]]
// CHECK:           [[OUT1_DIM:%.+]] = memref.dim [[OUT1]], [[C1]]
// CHECK:           [[OUT1_IS_2:%.+]] = arith.cmpi eq, [[OUT1_DIM]], [[C2]]
// CHECK:           [[OUT1_IS_3:%.+]] = arith.cmpi eq, [[OUT1_DIM]], [[C3]]
// CHECK:           [[OUT1_EITHER:%.+]] = arith.ori [[OUT1_IS_2]], [[OUT1_IS_3]]
// CHECK:           llvm.intr.assume [[OUT1_EITHER]]
// CHECK-LABEL: func.func @order_sqrt_spec_0
// CHECK-SAME:      ([[IN0:%.+]]: memref<1x?x8x128xf16>, [[OUT0:%.+]]: memref<1x?x8x128xf16>)
// CHECK-DAG:       [[D3:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[D2:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[D1:%.+]] = arith.constant 1 : index
// CHECK:           [[IN0_DIM:%.+]] = memref.dim [[IN0]], [[D1]]
// CHECK:           [[IN0_IS_2:%.+]] = arith.cmpi eq, [[IN0_DIM]], [[D2]]
// CHECK:           [[IN0_IS_3:%.+]] = arith.cmpi eq, [[IN0_DIM]], [[D3]]
// CHECK:           [[IN0_EITHER:%.+]] = arith.ori [[IN0_IS_2]], [[IN0_IS_3]]
// CHECK:           llvm.intr.assume [[IN0_EITHER]]
// CHECK:           [[OUT0_DIM:%.+]] = memref.dim [[OUT0]], [[D1]]
// CHECK:           [[OUT0_IS_2:%.+]] = arith.cmpi eq, [[OUT0_DIM]], [[D2]]
// CHECK:           [[OUT0_IS_3:%.+]] = arith.cmpi eq, [[OUT0_DIM]], [[D3]]
// CHECK:           [[OUT0_EITHER:%.+]] = arith.ori [[OUT0_IS_2]], [[OUT0_IS_3]]
// CHECK:           llvm.intr.assume [[OUT0_EITHER]]

// CHECK:       func.func @InnermostCandidateFirst
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@order_sqrt_spec_1
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@order_sqrt_spec_1
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@order_sqrt_spec_0
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@order_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @unknown_sqrt(%in: memref<1x?x?x?xf16>, %out: memref<1x?x?x?xf16>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16>) outs(%out : memref<1x?x?x?xf16>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @UnknownValueNeverSplit(%param0: memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>, %ret0: memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>, %param1: memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>, %ret1: memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>, %param2: memref<1x1x8x?xf16, #NCHW, [@CMX_NN, 0]>, %ret2: memref<1x1x8x?xf16, #NCHW, [@CMX_NN, 0]>) -> memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@unknown_sqrt inputs(%param0 as %arg0: memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@unknown_sqrt inputs(%param1 as %arg0: memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x8x128xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@unknown_sqrt inputs(%param2 as %arg0: memref<1x1x8x?xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret2 as %arg1: memref<1x1x8x?xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x8x?xf16, #NCHW, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x8x?xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x8x?xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x1x8x256xf16, #NCHW, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @unknown_sqrt_spec_2
// CHECK-SAME:      (%{{.+}}: memref<1x1x8x256xf16>, %{{.+}}: memref<1x1x8x256xf16>)
// CHECK-NOT:       llvm.intr.assume
// CHECK-LABEL: func.func @unknown_sqrt_spec_1
// CHECK-SAME:      (%{{.+}}: memref<1x1x8x128xf16>, %{{.+}}: memref<1x1x8x128xf16>)
// CHECK-NOT:       llvm.intr.assume

// CHECK-LABEL: func.func @unknown_sqrt_spec_0
// CHECK-SAME:      (%{{.+}}: memref<1x1x8x?xf16>, %{{.+}}: memref<1x1x8x?xf16>)
// CHECK-NOT:       llvm.intr.assume
// CHECK:       func.func @UnknownValueNeverSplit
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@unknown_sqrt_spec_2
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@unknown_sqrt_spec_1
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@unknown_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @auxin_sqrt(%in: memref<1x?x?x?xf16>, %aux: memref<1x1x1x?xf16>, %out: memref<1x?x?x?xf16>, %auxOut: memref<1x1x1x?xf16>) {
    %c0 = arith.constant 0 : index
    %bias = memref.load %aux[%c0, %c0, %c0, %c0] : memref<1x1x1x?xf16>
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16>) outs(%out : memref<1x?x?x?xf16>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      %1 = arith.addf %0, %bias : f16
      linalg.yield %1 : f16
    }
    return
  }
}

func.func @OnlyFirstOutputDrivesSplit(%param0: memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, %aux0: memref<1x1x1x8xf16, #NCHW, [@CMX_NN, 0]>, %ret0: memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, %param1: memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, %aux1: memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>, %ret1: memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>) -> memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@auxin_sqrt inputs(%param0 as %arg0: memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, %aux0 as %arg1: memref<1x1x1x8xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret0 as %arg2: memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, %aux0 as %arg3: memref<1x1x1x8xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> (memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x8xf16, #NCHW, [@CMX_NN, 0]>){
      VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2, %arg3) : memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x8xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x8xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@auxin_sqrt inputs(%param1 as %arg0: memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, %aux1 as %arg1: memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret1 as %arg2: memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, %aux1 as %arg3: memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> (memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>){
      VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2, %arg3) : memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x1x1x224xf16, #NCHW, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @auxin_sqrt_spec_0
// CHECK-SAME:      (%{{.+}}: memref<1x1x1x224xf16>, [[AUX:%.+]]: memref<1x1x1x?xf16>,
// CHECK-SAME:       %{{.+}}: memref<1x1x1x224xf16>, %{{.+}}: memref<1x1x1x?xf16>)
// CHECK-DAG:       [[C16:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[C8:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[C3:%.+]] = arith.constant 3 : index
// CHECK:           [[AUX_DIM:%.+]] = memref.dim [[AUX]], [[C3]]
// CHECK:           [[IS_8:%.+]] = arith.cmpi eq, [[AUX_DIM]], [[C8]]
// CHECK:           [[IS_16:%.+]] = arith.cmpi eq, [[AUX_DIM]], [[C16]]
// CHECK:           [[EITHER:%.+]] = arith.ori [[IS_8]], [[IS_16]]
// CHECK:           llvm.intr.assume [[EITHER]]
// CHECK-NOT:       llvm.intr.assume
// CHECK:           memref.load [[AUX]]
// CHECK:       func.func @OnlyFirstOutputDrivesSplit
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@auxin_sqrt_spec_0
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@auxin_sqrt_spec_0

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @auxvol_sqrt(%in: memref<1x?x?x?xf16>, %aux: memref<1x1x1x?xf16>, %out: memref<1x?x?x?xf16>, %auxOut: memref<1x1x1x?xf16>) {
    %c0 = arith.constant 0 : index
    %bias = memref.load %aux[%c0, %c0, %c0, %c0] : memref<1x1x1x?xf16>
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16>) outs(%out : memref<1x?x?x?xf16>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      %1 = arith.addf %0, %bias : f16
      linalg.yield %1 : f16
    }
    return
  }
}

func.func @AuxBufferWeighsTheSaving(%param0: memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>, %aux0: memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>, %ret0: memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>, %param1: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>, %aux1: memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>, %ret1: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>) -> memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@auxvol_sqrt inputs(%param0 as %arg0: memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>, %aux0 as %arg1: memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret0 as %arg2: memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>, %aux0 as %arg3: memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> (memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>){
      VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2, %arg3) : memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@auxvol_sqrt inputs(%param1 as %arg0: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>, %aux1 as %arg1: memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>) outputs(%ret1 as %arg2: memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>, %aux1 as %arg3: memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 list 0 -> (memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>){
      VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2, %arg3) : memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x2x32xf16, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x16xf16, #NCHW, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x1x1x32xf16, #NCHW, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @auxvol_sqrt_spec_1
// CHECK-SAME:      (%{{.+}}: memref<1x1x2x32xf16>, [[AUX1:%.+]]: memref<1x1x1x16xf16>,
// CHECK-SAME:       %{{.+}}: memref<1x1x2x32xf16>, %{{.+}}: memref<1x1x1x?xf16>)
// CHECK-NOT:       llvm.intr.assume
// CHECK:           memref.load [[AUX1]]
// CHECK-LABEL: func.func @auxvol_sqrt_spec_0
// CHECK-SAME:      (%{{.+}}: memref<1x1x1x32xf16>, [[AUX0:%.+]]: memref<1x1x1x16xf16>,
// CHECK-SAME:       %{{.+}}: memref<1x1x1x32xf16>, %{{.+}}: memref<1x1x1x?xf16>)
// CHECK-NOT:       llvm.intr.assume
// CHECK:           memref.load [[AUX0]]
// CHECK:       func.func @AuxBufferWeighsTheSaving
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@auxvol_sqrt_spec_0
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@auxvol_sqrt_spec_1

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func @stride_sqrt(%in: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, %out: memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%out : memref<1x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) {
    ^bb0(%arg0: f16, %arg1: f16):
      %0 = math.sqrt %arg0 : f16
      linalg.yield %0 : f16
    }
    return
  }
}

func.func @StrideDrivesSplit(%param0: memref<1x1x8x32xf16, {order = #NCHW, strides = [256, 256, 32, 1]}, [@CMX_NN, 0]>, %ret0: memref<1x1x8x32xf16, {order = #NCHW, strides = [256, 256, 32, 1]}, [@CMX_NN, 0]>, %param1: memref<1x1x8x32xf16, {order = #NCHW, strides = [512, 512, 64, 1]}, [@CMX_NN, 0]>, %ret1: memref<1x1x8x32xf16, {order = #NCHW, strides = [512, 512, 64, 1]}, [@CMX_NN, 0]>) -> memref<1x1x8x32xf16, {order = #NCHW, strides = [256, 256, 32, 1]}, [@CMX_NN, 0]> {
  VPURT.Task {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@stride_sqrt inputs(%param0 as %arg0: memref<1x1x8x32xf16, {order = #NCHW, strides = [256, 256, 32, 1]}, [@CMX_NN, 0]>) outputs(%ret0 as %arg1: memref<1x1x8x32xf16, {order = #NCHW, strides = [256, 256, 32, 1]}, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x8x32xf16, {order = #NCHW, strides = [256, 256, 32, 1]}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x8x32xf16, {order = #NCHW, strides = [256, 256, 32, 1]}, [@CMX_NN, 0]>, memref<1x1x8x32xf16, {order = #NCHW, strides = [256, 256, 32, 1]}, [@CMX_NN, 0]>
    }
  }
  VPURT.Task {
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@stride_sqrt inputs(%param1 as %arg0: memref<1x1x8x32xf16, {order = #NCHW, strides = [512, 512, 64, 1]}, [@CMX_NN, 0]>) outputs(%ret1 as %arg1: memref<1x1x8x32xf16, {order = #NCHW, strides = [512, 512, 64, 1]}, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<1x1x8x32xf16, {order = #NCHW, strides = [512, 512, 64, 1]}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<1x1x8x32xf16, {order = #NCHW, strides = [512, 512, 64, 1]}, [@CMX_NN, 0]>, memref<1x1x8x32xf16, {order = #NCHW, strides = [512, 512, 64, 1]}, [@CMX_NN, 0]>
    }
  }
  return %ret0 : memref<1x1x8x32xf16, {order = #NCHW, strides = [256, 256, 32, 1]}, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @stride_sqrt_spec_1
// CHECK-SAME:      (%{{.+}}: memref<1x1x8x32xf16, strided<[256, 256, 32, 1]>>, %{{.+}}: memref<1x1x8x32xf16, strided<[256, 256, 32, 1]>>)
// CHECK-NOT:       llvm.intr.assume
// CHECK-LABEL: func.func @stride_sqrt_spec_0
// CHECK-SAME:      (%{{.+}}: memref<1x1x8x32xf16, strided<[512, 512, 64, 1]>>, %{{.+}}: memref<1x1x8x32xf16, strided<[512, 512, 64, 1]>>)
// CHECK-NOT:       llvm.intr.assume
// CHECK:       func.func @StrideDrivesSplit
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@stride_sqrt_spec_1
// CHECK:           VPUIP.SW.Kernel {{.*}}@VPU.SW::@stride_sqrt_spec_0
