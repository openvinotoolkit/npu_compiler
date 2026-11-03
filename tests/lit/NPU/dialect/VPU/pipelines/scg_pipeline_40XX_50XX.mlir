//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --shavecodegen-vpu %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NC = affine_map<(d0, d1) -> (d0, d1)>
#map = affine_map<(d0, d1) -> ()>

module @EltwiseFlatten {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<16x16xf32>, %arg1: tensor<f32>) -> tensor<16x16xf32> {
      %empt = tensor.empty() : tensor<16x16xf32>
      %1 = linalg.generic {indexing_maps = [#NC, #map, #NC], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<16x16xf32>, tensor<f32>) outs(%empt : tensor<16x16xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %2 = arith.mulf %in, %in_0 : f32
        linalg.yield %2 : f32
      } -> tensor<16x16xf32>
      return %1 : tensor<16x16xf32>
    }
  }

// CHECK-LABEL: @EltwiseFlatten
// CHECK: func.func @generated_0(
// CHECK-SAME: [[ARG0:%.+]]: memref<16x16xf32>, [[ARG1:%.+]]: memref<f32>, [[RET:%.+]]: memref<16x16xf32>) {
// CHECK-DAG:   [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:   [[C256:%.+]] = arith.constant 256 : index
// CHECK-DAG:   [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:   [[CARG0:%.+]] = memref.collapse_shape [[ARG0]]
// CHECK-SAME:      : memref<16x16xf32> into memref<256xf32>
// CHECK-DAG:   [[CRET:%.+]] = memref.collapse_shape [[RET]]
// CHECK-SAME:      : memref<16x16xf32> into memref<256xf32>
// CHECK:       scf.for [[COUNTER:%.+]] = [[C0]] to [[C256]] step [[C1]] {
// CHECK-NEXT:    [[ELEM_LHS:%.+]] = memref.subview [[CARG0]][[[COUNTER]]] [1] [1]
// CHECK-SAME:        : memref<256xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK-NEXT:    [[ELEM_OUT:%.+]] = memref.subview [[CRET]][[[COUNTER]]] [1] [1]
// CHECK-SAME:        : memref<256xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK-NEXT:    linalg.generic
// CHECK-SAME:         ins([[ELEM_LHS]], [[ARG1]] :
// CHECK-SAME:         outs([[ELEM_OUT]] :
// CHECK-NEXT:    ^bb0([[IN:%.+]]: f32, [[SCALAR:%.+]]: f32, {{%.*}}: f32):
// CHECK-NEXT:      [[MUL:%.+]] = arith.mulf [[IN]], [[SCALAR]] : f32
// CHECK-NEXT:      linalg.yield [[MUL]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NOT:   memref.copy
// CHECK-NOT:   memref.expand_shape
// CHECK:       return
}
