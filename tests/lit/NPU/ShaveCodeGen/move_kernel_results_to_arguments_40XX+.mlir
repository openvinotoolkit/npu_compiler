//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="platform=%platform%" --move-kernel-results-to-arguments --canonicalize | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, 0)>

module @Simple {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x3x16x16xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x3x16x16xf32> {
      %0 = tensor.empty() : tensor<1x3x16x16xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x3x16x16xf32>, tensor<1x1x1x1xf32>) outs(%0 : tensor<1x3x16x16xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %2 = arith.mulf %in, %in_0 : f32
        linalg.yield %2 : f32
      } -> tensor<1x3x16x16xf32>
      return %1 : tensor<1x3x16x16xf32>
    }
    // CHECK: func.func @generated_0(
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x16x16xf32>, [[ARG1:%.+]]: tensor<1x1x1x1xf32>, [[RET:%.+]]: memref<1x3x16x16xf32>)
    // CHECK-NOT: tensor.empty()
    // CHECK-NEXT: [[RET_TENSOR:%.+]] = bufferization.to_tensor [[RET]] restrict writable : memref<1x3x16x16xf32>
    // CHECK-NEXT: [[OP:%.+]] = linalg.generic
    // CHECK-SAME: ins([[ARG0]], [[ARG1]] : tensor<1x3x16x16xf32>, tensor<1x1x1x1xf32>)
    // CHECK-SAME: outs([[RET_TENSOR]] : tensor<1x3x16x16xf32>)
    // CHECK: bufferization.materialize_in_destination [[OP]] in writable [[RET]] : (tensor<1x3x16x16xf32>, memref<1x3x16x16xf32>)
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>

module @PaddedReduce {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x16x4000x200xf32>) -> tensor<16x4000x200xf32> {
      %c0 = arith.constant 0.000000e+00 : f32
      %empt = tensor.empty() : tensor<16x4000x200xf32>
      %fill = linalg.fill ins(%c0 : f32) outs(%empt : tensor<16x4000x200xf32>) -> tensor<16x4000x200xf32>
      %extract_pad = tensor.extract_slice %fill[0, 0, 0] [1, 4000, 200] [1, 1, 1] : tensor<16x4000x200xf32> to tensor<1x4000x200xf32>
      %out_slice = linalg.fill ins(%c0 : f32) outs(%extract_pad : tensor<1x4000x200xf32>) -> tensor<1x4000x200xf32>
      %in_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 12, 4000, 200] [1, 1, 1, 1] : tensor<1x16x4000x200xf32> to tensor<1x12x4000x200xf32>
      %reduce = linalg.generic {indexing_maps = [#NCHW, #map], iterator_types = ["parallel", "reduction", "parallel", "parallel"]} ins(%in_slice : tensor<1x12x4000x200xf32>) outs(%out_slice : tensor<1x4000x200xf32>) {
      ^bb0(%in: f32, %out: f32):
        %add = arith.addf %out, %in fastmath<reassoc> : f32
        linalg.yield %add : f32
      } -> tensor<1x4000x200xf32>
      %out = tensor.insert_slice %reduce into %fill[0, 0, 0] [1, 4000, 200] [1, 1, 1] : tensor<1x4000x200xf32> into tensor<16x4000x200xf32>
      return %out : tensor<16x4000x200xf32>
    }
    // CHECK: func.func @generated_0(
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x4000x200xf32>, [[RET:%.+]]: memref<16x4000x200xf32>)
    // CHECK-NOT: tensor.empty()
    // CHECK: [[RET_TENSOR:%.+]] = bufferization.to_tensor [[ARG1]] restrict writable : memref<16x4000x200xf32>
    // CHECK-NEXT: [[PADDED_OUTPUT:%.+]] = linalg.fill
    // CHECK-SAME:     outs([[RET_TENSOR]] : tensor<16x4000x200xf32>) -> tensor<16x4000x200xf32>
    // CHECK-NEXT: [[REDUCE_OUT_SLICE:%.+]] = tensor.extract_slice [[PADDED_OUTPUT]][0, 0, 0] [1, 4000, 200] [1, 1, 1] : tensor<16x4000x200xf32> to tensor<1x4000x200xf32>
    // CHECK-NEXT: [[REDUCE_OUT_SLICE_INIT:%.+]] = linalg.fill
    // CHECK-SAME:     outs([[REDUCE_OUT_SLICE]] : tensor<1x4000x200xf32>) -> tensor<1x4000x200xf32>
    // CHECK-NEXT: [[IN_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, 0] [1, 12, 4000, 200] [1, 1, 1, 1] : tensor<1x16x4000x200xf32> to tensor<1x12x4000x200xf32>
    // CHECK-NEXT: [[REDUCED:%.+]] = linalg.generic
    // CHECK-SAME:   ins([[IN_SLICE]] : tensor<1x12x4000x200xf32>)
    // CHECK-SAME:   outs([[REDUCE_OUT_SLICE_INIT]] : tensor<1x4000x200xf32>) {
    // CHECK:  [[OUT:%.+]] = tensor.insert_slice [[REDUCED]] into [[PADDED_OUTPUT]][0, 0, 0] [1, 4000, 200] [1, 1, 1] : tensor<1x4000x200xf32> into tensor<16x4000x200xf32>
    // CHECK-NEXT:  bufferization.materialize_in_destination [[OUT]] in writable [[RET]] : (tensor<16x4000x200xf32>, memref<16x4000x200xf32>) -> ()
    // CHECK: return
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, 0)>

module @WithScalarIndexArg {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x3x16x16xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: index) -> tensor<1x3x16x16xf32> {
      %0 = tensor.empty() : tensor<1x3x16x16xf32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x3x16x16xf32>, tensor<1x1x1x1xf32>) outs(%0 : tensor<1x3x16x16xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %2 = arith.mulf %in, %in_0 : f32
        linalg.yield %2 : f32
      } -> tensor<1x3x16x16xf32>
      return %1 : tensor<1x3x16x16xf32>
    }
    // CHECK: func.func @generated_0(
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x16x16xf32>, [[ARG1:%.+]]: tensor<1x1x1x1xf32>, [[RET:%.+]]: memref<1x3x16x16xf32>, {{%.+}}: index)
    // CHECK-NOT: tensor.empty()
    // CHECK-NEXT: [[RET_TENSOR:%.+]] = bufferization.to_tensor [[RET]] restrict writable : memref<1x3x16x16xf32>
    // CHECK-NEXT: [[OP:%.+]] = linalg.generic
    // CHECK-SAME: ins([[ARG0]], [[ARG1]] : tensor<1x3x16x16xf32>, tensor<1x1x1x1xf32>)
    // CHECK-SAME: outs([[RET_TENSOR]] : tensor<1x3x16x16xf32>)
    // CHECK: bufferization.materialize_in_destination [[OP]] in writable [[RET]] : (tensor<1x3x16x16xf32>, memref<1x3x16x16xf32>)
  }
}

// -----

module @WithTilingInfoFunc {
  module @VPU.SW {
    func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index, index) {
      %c0 = arith.constant 0 : index
      return %arg0, %arg1, %arg2, %c0 : index, index, index, index
    }
    // CHECK:      func.func @generated_info_0(
    // CHECK-SAME:   [[SIZE_0:%[^:]+]]: index, [[SIZE_1:%[^:]+]]: index, [[OFFSET_0:%[^:]+]]: index) -> (index, index, index, index)
    // CHECK:        [[C0:%.+]] = arith.constant 0 : index
    // CHECK:        return [[SIZE_0]], [[SIZE_1]], [[OFFSET_0]], [[C0]] : index, index, index, index
  }
}
