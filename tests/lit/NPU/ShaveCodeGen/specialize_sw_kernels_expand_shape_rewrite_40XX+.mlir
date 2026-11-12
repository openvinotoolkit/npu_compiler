//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --specialize-sw-kernels %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// Both source dims static, one group resolves and the other does not: [0, 1] = [3, ?]
// resolves to 12 / 3 = 4, while [2, 3] = [?, ?] has no unique answer, so source dim 1
// is relaxed back to dynamic through a memref.cast.

module @VPU.SW {
  func.func @expand_resolve_and_relax(%src: memref<?x?xf32>, %out: memref<3x?x?x?xf32>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %d1 = memref.dim %out, %c1 : memref<3x?x?x?xf32>
    %d2 = memref.dim %out, %c2 : memref<3x?x?x?xf32>
    %d3 = memref.dim %out, %c3 : memref<3x?x?x?xf32>
    %expanded = memref.expand_shape %src [[0, 1], [2, 3]] output_shape [3, %d1, %d2, %d3] : memref<?x?xf32> into memref<3x?x?x?xf32>
    memref.copy %expanded, %out : memref<3x?x?x?xf32> to memref<3x?x?x?xf32>
    return
  }
}

func.func @ResolveOneGroupRelaxOther(%param0: memref<12x20xf32, [@CMX_NN, 0]>, %ret: memref<3x?x?x?xf32, [@CMX_NN, 0]>) -> memref<3x?x?x?xf32, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@expand_resolve_and_relax inputs(%param0 as %arg0: memref<12x20xf32, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<3x?x?x?xf32, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<3x?x?x?xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<12x20xf32, [@CMX_NN, 0]>, memref<3x?x?x?xf32, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<3x?x?x?xf32, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @expand_resolve_and_relax_spec_0
// CHECK-SAME:      ([[SRC:%.+]]: memref<12x20xf32>, [[OUT:%.+]]: memref<3x?x?x?xf32>)
// CHECK:           [[DIM2:%.+]] = memref.dim [[OUT]], {{%.+}} : memref<3x?x?x?xf32>
// CHECK-NEXT:      [[DIM3:%.+]] = memref.dim [[OUT]], {{%.+}} : memref<3x?x?x?xf32>
// CHECK-NEXT:      [[CAST:%.+]] = memref.cast [[SRC]] : memref<12x20xf32> to memref<12x?xf32>
// CHECK-NEXT:      [[EXPAND:%.+]] = memref.expand_shape [[CAST]] {{\[}}[0, 1], [2, 3]] output_shape [3, 4, [[DIM2]], [[DIM3]]] : memref<12x?xf32> into memref<3x4x?x?xf32>
// CHECK-NEXT:      memref.copy [[EXPAND]], [[OUT]] : memref<3x4x?x?xf32> to memref<3x?x?x?xf32>
// CHECK:       @VPU.SW::@expand_resolve_and_relax_spec_0

// -----

// A group that resolves next to a group that is already fully static: [0, 1] = [3, ?]
// resolves to 4, while [2] = [20] holds no unknown and is skipped. No group is left
// unresolved, so no source relaxation and no memref.cast is needed.

module @VPU.SW {
  func.func @expand_static_sibling_group(%src: memref<?x20xf32>, %out: memref<3x?x20xf32>) {
    %c1 = arith.constant 1 : index
    %d1 = memref.dim %out, %c1 : memref<3x?x20xf32>
    %expanded = memref.expand_shape %src [[0, 1], [2]] output_shape [3, %d1, 20] : memref<?x20xf32> into memref<3x?x20xf32>
    memref.copy %expanded, %out : memref<3x?x20xf32> to memref<3x?x20xf32>
    return
  }
}

func.func @ResolveWithStaticSiblingGroup(%param0: memref<12x20xf32, [@CMX_NN, 0]>, %ret: memref<3x?x20xf32, [@CMX_NN, 0]>) -> memref<3x?x20xf32, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@expand_static_sibling_group inputs(%param0 as %arg0: memref<12x20xf32, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<3x?x20xf32, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<3x?x20xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<12x20xf32, [@CMX_NN, 0]>, memref<3x?x20xf32, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<3x?x20xf32, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @expand_static_sibling_group_spec_0
// CHECK-SAME:      ([[SRC:%.+]]: memref<12x20xf32>, [[OUT:%.+]]: memref<3x?x20xf32>)
// CHECK-NEXT:      [[EXPAND:%.+]] = memref.expand_shape [[SRC]] {{\[}}[0, 1], [2]] output_shape [3, 4, 20] : memref<12x20xf32> into memref<3x4x20xf32>
// CHECK-NEXT:      memref.copy [[EXPAND]], [[OUT]] : memref<3x4x20xf32> to memref<3x?x20xf32>
// CHECK-NEXT:      return
// CHECK:       @VPU.SW::@expand_static_sibling_group_spec_0

// -----

// The unknown sits in the middle of a three-dim group with two known siblings, so the
// resolution has to divide by the product of both: [0, 1, 2] = [2, ?, 3] resolves to
// 24 / (2 * 3) = 4.

module @VPU.SW {
  func.func @expand_unknown_inside_group(%src: memref<?x20xf32>, %out: memref<2x?x3x20xf32>) {
    %c1 = arith.constant 1 : index
    %d1 = memref.dim %out, %c1 : memref<2x?x3x20xf32>
    %expanded = memref.expand_shape %src [[0, 1, 2], [3]] output_shape [2, %d1, 3, 20] : memref<?x20xf32> into memref<2x?x3x20xf32>
    memref.copy %expanded, %out : memref<2x?x3x20xf32> to memref<2x?x3x20xf32>
    return
  }
}

func.func @ResolveUnknownInsideGroup(%param0: memref<24x20xf32, [@CMX_NN, 0]>, %ret: memref<2x?x3x20xf32, [@CMX_NN, 0]>) -> memref<2x?x3x20xf32, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@expand_unknown_inside_group inputs(%param0 as %arg0: memref<24x20xf32, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<2x?x3x20xf32, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<2x?x3x20xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<24x20xf32, [@CMX_NN, 0]>, memref<2x?x3x20xf32, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<2x?x3x20xf32, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @expand_unknown_inside_group_spec_0
// CHECK-SAME:      ([[SRC:%.+]]: memref<24x20xf32>, [[OUT:%.+]]: memref<2x?x3x20xf32>)
// CHECK-NEXT:      [[EXPAND:%.+]] = memref.expand_shape [[SRC]] {{\[}}[0, 1, 2], [3]] output_shape [2, 4, 3, 20] : memref<24x20xf32> into memref<2x4x3x20xf32>
// CHECK-NEXT:      memref.copy [[EXPAND]], [[OUT]] : memref<2x4x3x20xf32> to memref<2x?x3x20xf32>
// CHECK-NEXT:      return
// CHECK:       @VPU.SW::@expand_unknown_inside_group_spec_0

// -----

// The expand_shape source is hidden behind a memref.cast that drops static dim 1. The
// cast only removes dynamism, so it is foldable and the folder must look past it to the
// memref<12x20xf32> underneath.

module @VPU.SW {
  func.func @expand_through_source_cast(%src: memref<?x20xf32>, %out: memref<3x?x?x?xf32>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %d1 = memref.dim %out, %c1 : memref<3x?x?x?xf32>
    %d2 = memref.dim %out, %c2 : memref<3x?x?x?xf32>
    %d3 = memref.dim %out, %c3 : memref<3x?x?x?xf32>
    %relaxed = memref.cast %src : memref<?x20xf32> to memref<?x?xf32>
    %expanded = memref.expand_shape %relaxed [[0, 1], [2, 3]] output_shape [3, %d1, %d2, %d3] : memref<?x?xf32> into memref<3x?x?x?xf32>
    memref.copy %expanded, %out : memref<3x?x?x?xf32> to memref<3x?x?x?xf32>
    return
  }
}

func.func @ResolveThroughSourceCast(%param0: memref<12x20xf32, [@CMX_NN, 0]>, %ret: memref<3x?x?x?xf32, [@CMX_NN, 0]>) -> memref<3x?x?x?xf32, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@expand_through_source_cast inputs(%param0 as %arg0: memref<12x20xf32, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<3x?x?x?xf32, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<3x?x?x?xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<12x20xf32, [@CMX_NN, 0]>, memref<3x?x?x?xf32, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<3x?x?x?xf32, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @expand_through_source_cast_spec_0
// CHECK-SAME:      ([[SRC:%.+]]: memref<12x20xf32>, [[OUT:%.+]]: memref<3x?x?x?xf32>)
// CHECK:           [[DIM2:%.+]] = memref.dim [[OUT]], {{%.+}} : memref<3x?x?x?xf32>
// CHECK-NEXT:      [[DIM3:%.+]] = memref.dim [[OUT]], {{%.+}} : memref<3x?x?x?xf32>
// CHECK-NEXT:      [[CAST:%.+]] = memref.cast [[SRC]] : memref<12x20xf32> to memref<12x?xf32>
// CHECK-NEXT:      [[EXPAND:%.+]] = memref.expand_shape [[CAST]] {{\[}}[0, 1], [2, 3]] output_shape [3, 4, [[DIM2]], [[DIM3]]] : memref<12x?xf32> into memref<3x4x?x?xf32>
// CHECK-NEXT:      memref.copy [[EXPAND]], [[OUT]] : memref<3x4x?x?xf32> to memref<3x?x?x?xf32>
// CHECK-NEXT:      return
// CHECK:       @VPU.SW::@expand_through_source_cast_spec_0

// -----

// Source dim 1 is dynamic at the callsite and therefore stays dynamic after
// specialization: group [2, 3] is skipped because its source dim is already dynamic
// so it needs no relaxation and no memref.cast, while group [0, 1] = [3, ?] still
// resolves to 4.

module @VPU.SW {
  func.func @expand_dynamic_source_dim(%src: memref<?x?xf32>, %out: memref<3x?x?x?xf32>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %d1 = memref.dim %out, %c1 : memref<3x?x?x?xf32>
    %d2 = memref.dim %out, %c2 : memref<3x?x?x?xf32>
    %d3 = memref.dim %out, %c3 : memref<3x?x?x?xf32>
    %expanded = memref.expand_shape %src [[0, 1], [2, 3]] output_shape [3, %d1, %d2, %d3] : memref<?x?xf32> into memref<3x?x?x?xf32>
    memref.copy %expanded, %out : memref<3x?x?x?xf32> to memref<3x?x?x?xf32>
    return
  }
}

func.func @ResolveWithDynamicSourceDim(%param0: memref<12x?xf32, [@CMX_NN, 0]>, %ret: memref<3x?x?x?xf32, [@CMX_NN, 0]>) -> memref<3x?x?x?xf32, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@expand_dynamic_source_dim inputs(%param0 as %arg0: memref<12x?xf32, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<3x?x?x?xf32, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<3x?x?x?xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<12x?xf32, [@CMX_NN, 0]>, memref<3x?x?x?xf32, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<3x?x?x?xf32, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @expand_dynamic_source_dim_spec_0
// CHECK-SAME:      ([[SRC:%.+]]: memref<12x?xf32>, [[OUT:%.+]]: memref<3x?x?x?xf32>)
// CHECK:           [[DIM2:%.+]] = memref.dim [[OUT]], {{%.+}} : memref<3x?x?x?xf32>
// CHECK-NEXT:      [[DIM3:%.+]] = memref.dim [[OUT]], {{%.+}} : memref<3x?x?x?xf32>
// CHECK-NEXT:      [[EXPAND:%.+]] = memref.expand_shape [[SRC]] {{\[}}[0, 1], [2, 3]] output_shape [3, 4, [[DIM2]], [[DIM3]]] : memref<12x?xf32> into memref<3x4x?x?xf32>
// CHECK-NEXT:      memref.copy [[EXPAND]], [[OUT]] : memref<3x4x?x?xf32> to memref<3x?x?x?xf32>
// CHECK-NEXT:      return
// CHECK:       @VPU.SW::@expand_dynamic_source_dim_spec_0

// -----

// The expand_shape is already in its most static form, so the folder must bail out.

module @VPU.SW {
  func.func @expand_already_static(%src: memref<12x20xf32>, %out: memref<?x?x?x?xf32>) {
    %expanded = memref.expand_shape %src [[0, 1], [2, 3]] output_shape [3, 4, 5, 4] : memref<12x20xf32> into memref<3x4x5x4xf32>
    memref.copy %expanded, %out : memref<3x4x5x4xf32> to memref<?x?x?x?xf32>
    return
  }
}

func.func @ExpandShapeAlreadyStatic(%param0: memref<12x20xf32, [@CMX_NN, 0]>, %ret: memref<3x4x5x4xf32, [@CMX_NN, 0]>) -> memref<3x4x5x4xf32, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@expand_already_static inputs(%param0 as %arg0: memref<12x20xf32, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<3x4x5x4xf32, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<3x4x5x4xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<12x20xf32, [@CMX_NN, 0]>, memref<3x4x5x4xf32, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<3x4x5x4xf32, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @expand_already_static_spec_0
// CHECK-SAME:      ([[SRC:%.+]]: memref<12x20xf32>, [[OUT:%.+]]: memref<3x4x5x4xf32>)
// CHECK-NEXT:      [[EXPAND:%.+]] = memref.expand_shape [[SRC]] {{\[}}[0, 1], [2, 3]] output_shape [3, 4, 5, 4] : memref<12x20xf32> into memref<3x4x5x4xf32>
// CHECK-NEXT:      memref.copy [[EXPAND]], [[OUT]] : memref<3x4x5x4xf32> to memref<3x4x5x4xf32>
// CHECK-NEXT:      return
// CHECK:       @VPU.SW::@expand_already_static_spec_0

// -----

module @VPU.SW {
  func.func @expand_empty_source_dim(%src: memref<?xf32>, %out: memref<0x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f32
    %d1 = memref.dim %out, %c1 : memref<0x?xf32>
    %expanded = memref.expand_shape %src [[0, 1]] output_shape [0, %d1] : memref<?xf32> into memref<0x?xf32>
    memref.store %cst, %expanded[%c0, %c0] : memref<0x?xf32>
    return
  }
}

func.func @EmptySourceDimStaysDynamic(%param0: memref<0xf32, [@CMX_NN, 0]>, %ret: memref<0x?xf32, [@CMX_NN, 0]>) -> memref<0x?xf32, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@expand_empty_source_dim inputs(%param0 as %arg0: memref<0xf32, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<0x?xf32, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<0x?xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<0xf32, [@CMX_NN, 0]>, memref<0x?xf32, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<0x?xf32, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @expand_empty_source_dim_spec_0
// CHECK-SAME:      ([[SRC:%.+]]: memref<0xf32>, [[OUT:%.+]]: memref<0x?xf32>)
// CHECK:           [[CAST:%.+]] = memref.cast [[SRC]] : memref<0xf32> to memref<?xf32>
// CHECK:           [[DIM:%.+]] = memref.dim [[OUT]], {{%.+}} : memref<0x?xf32>
// CHECK-NEXT:      [[EXPAND:%.+]] = memref.expand_shape [[CAST]] {{\[}}[0, 1]] output_shape [0, [[DIM]]] : memref<?xf32> into memref<0x?xf32>
// CHECK-NEXT:      memref.store {{%.+}}, [[EXPAND]]
// CHECK:       @VPU.SW::@expand_empty_source_dim_spec_0

// -----

module @VPU.SW {
  func.func @expand_empty_source_dim_resolvable(%src: memref<?xf32>, %out: memref<2x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f32
    %d1 = memref.dim %out, %c1 : memref<2x?xf32>
    %expanded = memref.expand_shape %src [[0, 1]] output_shape [2, %d1] : memref<?xf32> into memref<2x?xf32>
    memref.store %cst, %expanded[%c0, %c0] : memref<2x?xf32>
    return
  }
}

func.func @EmptySourceDimResolvesToZero(%param0: memref<0xf32, [@CMX_NN, 0]>, %ret: memref<2x?xf32, [@CMX_NN, 0]>) -> memref<2x?xf32, [@CMX_NN, 0]> {
  VPURT.Task {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@expand_empty_source_dim_resolvable inputs(%param0 as %arg0: memref<0xf32, [@CMX_NN, 0]>) outputs(%ret as %arg1: memref<2x?xf32, [@CMX_NN, 0]>) on tile 0 list 0 -> memref<2x?xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg0, %arg1) : memref<0xf32, [@CMX_NN, 0]>, memref<2x?xf32, [@CMX_NN, 0]>
    }
  }
  return %ret : memref<2x?xf32, [@CMX_NN, 0]>
}

// CHECK-LABEL: func.func @expand_empty_source_dim_resolvable_spec_0
// CHECK-SAME:      ([[SRC:%.+]]: memref<0xf32>, [[OUT:%.+]]: memref<2x?xf32>)
// CHECK-NOT:       memref.cast
// CHECK:           [[EXPAND:%.+]] = memref.expand_shape [[SRC]] {{\[}}[0, 1]] output_shape [2, 0] : memref<0xf32> into memref<2x0xf32>
// CHECK-NEXT:      memref.store {{%.+}}, [[EXPAND]]
// CHECK-NEXT:      return
// CHECK:       @VPU.SW::@expand_empty_source_dim_resolvable_spec_0
