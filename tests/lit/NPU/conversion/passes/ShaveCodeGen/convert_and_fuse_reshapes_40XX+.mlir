//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-DAG: [[PERMUTE_CAST_MAP:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: [[RESHAPE_MAP:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK-DAG: [[IDENTITY:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: func.func @convertAndFuse(
func.func @convertAndFuse(%arg0: tensor<1x1008x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x1000xf32> {
  %caps = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1008x1x1xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg1 [0, 0, 0, 0] [1, 1000, 1, 1] : tensor<1x1008x1x1xf16, {order = #NHWC}> to tensor<1x1000x1x1xf16, {order = #NHWC}>
    %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1000x1x1xf16, {order = #NHWC}> -> tensor<1x1000x1x1xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000x1x1xf16> -> tensor<1x1x1x1000xf16>
    %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf32>
    IE.CGCYield %3 : tensor<1x1x1x1000xf32>
  } -> tensor<1x1x1x1000xf32>

  return %caps : tensor<1x1x1x1000xf32>

// CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x1x1x1008xf16>) {
// CHECK-NEXT:      [[EXTRACT_SLICE:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 1, 1, 1000] [1, 1, 1, 1] : tensor<1x1x1x1008xf16> to tensor<1x1x1x1000xf16>

// CHECK-NEXT:      [[EMPTY0:%.+]] = tensor.empty() : tensor<1x1000x1x1xf16>
// CHECK-NEXT:      [[PERMUTE_CAST:%.+]] = linalg.generic
// CHECK-SAME:          indexing_maps = [[[PERMUTE_CAST_MAP]], [[IDENTITY]]]
// CHECK-SAME:          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:          ins([[EXTRACT_SLICE]] : tensor<1x1x1x1000xf16>)
// CHECK-SAME:          outs([[EMPTY0]] : tensor<1x1000x1x1xf16>) {
// CHECK-NEXT:        ^bb0([[IN0:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:          linalg.yield [[IN0]] : f16
// CHECK-NEXT:      } -> tensor<1x1000x1x1xf16>

// CHECK-NEXT:      [[EMPTY1:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:      [[RESHAPE:%.+]] = linalg.generic
// CHECK-SAME:          indexing_maps = [[[RESHAPE_MAP]], [[IDENTITY]]]
// CHECK-SAME:          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:          ins([[PERMUTE_CAST]] : tensor<1x1000x1x1xf16>)
// CHECK-SAME:          outs([[EMPTY1]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:        ^bb0([[IN1:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:          linalg.yield [[IN1]] : f16
// CHECK-NEXT:      } -> tensor<1x1x1x1000xf16>

// CHECK-NEXT:      [[EMPTY2:%.+]] = tensor.empty() : tensor<1x1x1x1000xf32>
// CHECK-NEXT:      [[CONVERT:%.+]] = linalg.generic
// CHECK-SAME:          indexing_maps = [[[IDENTITY]], [[IDENTITY]]]
// CHECK-SAME:          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:          ins([[RESHAPE]] : tensor<1x1x1x1000xf16>)
// CHECK-SAME:          outs([[EMPTY2]] : tensor<1x1x1x1000xf32>) {
// CHECK-NEXT:        ^bb0([[IN2:%.+]]: f16, {{%.+}}: f32):
// CHECK-NEXT:          [[EXTF:%.+]] = arith.extf [[IN2]] : f16 to f32
// CHECK-NEXT:          linalg.yield [[EXTF]] : f32
// CHECK-NEXT:      } -> tensor<1x1x1x1000xf32>

// CHECK-NEXT:      IE.CGCYield [[CONVERT]] : tensor<1x1x1x1000xf32>
}
