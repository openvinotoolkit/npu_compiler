//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --shavecodegen-vpu %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NC = affine_map<(d0, d1) -> (d0, d1)>
#map = affine_map<(d0, d1) -> ()>

module @EltwiseFlatten {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<16x16xf32>, %arg1: tensor<f32>, %arg2: tensor<16x16xf32>) -> tensor<16x16xf32> {
      %1 = linalg.generic {indexing_maps = [#NC, #map, #NC], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<16x16xf32>, tensor<f32>) outs(%arg2 : tensor<16x16xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %2 = arith.mulf %in, %in_0 : f32
        linalg.yield %2 : f32
      } -> tensor<16x16xf32>
      return %1 : tensor<16x16xf32>
    }
  }

// CHECK-LABEL: @EltwiseFlatten
// CHECK:    func.func @generated_0([[ARG0:%.+]]: tensor<16x16xf32>, [[ARG1:%.+]]: tensor<f32>, [[ARG2:%.+]]: tensor<16x16xf32>) -> tensor<16x16xf32> {
// CHECK-DAG:      [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:      [[C256:%.+]] = arith.constant 256 : index
// CHECK-DAG:      [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:      [[CARG0:%.+]] = tensor.collapse_shape [[ARG0]] {{\[\[}}0, 1{{\]\]}} : tensor<16x16xf32> into tensor<256xf32>
// CHECK-DAG:      [[CARG2:%.+]] = tensor.collapse_shape [[ARG2]] {{\[\[}}0, 1{{\]\]}} : tensor<16x16xf32> into tensor<256xf32>
// CHECK:          [[RET_COLLAPSED:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C256]] step [[C1]] iter_args([[ARG4:%.+]] = [[CARG2]]) -> (tensor<256xf32>) {
// CHECK-DAG:        [[SLICE_IN:%.+]] = tensor.extract_slice [[CARG0]][[[IDX]]] [1] [1] : tensor<256xf32> to tensor<1xf32>
// CHECK-DAG:        [[SLICE_OUT:%.+]] = tensor.extract_slice [[ARG4]][[[IDX]]] [1] [1] : tensor<256xf32> to tensor<1xf32>
// CHECK:            [[OP:%.+]] = linalg.generic
// CHECK-SAME:            ins([[SLICE_IN]], [[ARG1]] : tensor<1xf32>, tensor<f32>)
// CHECK-SAME:            outs([[SLICE_OUT]] : tensor<1xf32>)
// CHECK:            } -> tensor<1xf32>
// CHECK-NEXT:       [[UPDATED:%.+]] = tensor.insert_slice [[OP]] into [[ARG4]][[[IDX]]] [1] [1] : tensor<1xf32> into tensor<256xf32>
// CHECK-NEXT:       scf.yield [[UPDATED]] : tensor<256xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     [[RET:%.+]] = tensor.expand_shape [[RET_COLLAPSED]] {{\[\[}}0, 1{{\]\]}} output_shape [16, 16] : tensor<256xf32> into tensor<16x16xf32>
// CHECK-NEXT:     return [[RET]] : tensor<16x16xf32>
}
