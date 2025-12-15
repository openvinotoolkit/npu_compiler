//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --flatten-eltwise-kernel -canonicalize %s -o - | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[C:#.+]] = affine_map<(d0) -> (d0)>
// CHECK: module @CollapseClamp
module @CollapseClamp {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x50x20x2000xi32>, %arg1: tensor<1x50x20x2000xi32>) -> tensor<1x50x20x2000xi32> {
      %c1_i32 = arith.constant 1 : i32
      %c-1_i32 = arith.constant -1 : i32
      %0 = tensor.empty() : tensor<1x50x20x2000xi32>
      %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x50x20x2000xi32>) outs(%arg1 : tensor<1x50x20x2000xi32>) {
      ^bb0(%in: i32, %out: i32):
        %2 = arith.cmpi sle, %in, %c-1_i32 : i32
        %3 = arith.select %2, %c-1_i32, %in : i32
        %4 = arith.cmpi sge, %3, %c1_i32 : i32
        %5 = arith.select %4, %c1_i32, %3 : i32
        linalg.yield %5 : i32
      } -> tensor<1x50x20x2000xi32>
      return %1 : tensor<1x50x20x2000xi32>
// CHECK: func.func @generated_0([[ARG0:%.+]]: tensor<1x50x20x2000xi32>, [[ARG1:%.+]]: tensor<1x50x20x2000xi32>) -> tensor<1x50x20x2000xi32> {
// CHECK-NEXT:      [[ONE:%.+]] = arith.constant 1 : i32
// CHECK-NEXT:      [[NEGONE:%.+]] = arith.constant -1 : i32
// CHECK-NEXT:      [[COLLAPSED_IN:%.+]] = tensor.collapse_shape [[ARG0]] {{\[\[}}0, 1, 2, 3{{\]\]}} : tensor<1x50x20x2000xi32> into tensor<2000000xi32>
// CHECK-NEXT:      [[COLLAPSED_OUT:%.+]] = tensor.collapse_shape [[ARG1]] {{\[\[}}0, 1, 2, 3{{\]\]}} : tensor<1x50x20x2000xi32> into tensor<2000000xi32>
// CHECK-NEXT:      [[OP:%.+]] = linalg.generic {indexing_maps = [[[C]], [[C]]], iterator_types = ["parallel"]} ins([[COLLAPSED_IN]] : tensor<2000000xi32>) outs([[COLLAPSED_OUT]] : tensor<2000000xi32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: i32, [[OUT:%.+]]: i32):
// CHECK-NEXT:        [[CMP1:%.+]] = arith.cmpi sle, [[IN]], [[NEGONE]] : i32
// CHECK-NEXT:        [[SEL1:%.+]] = arith.select [[CMP1]], [[NEGONE]], [[IN]] : i32
// CHECK-NEXT:        [[CMP2:%.+]] = arith.cmpi sge, [[SEL1]], [[ONE]] : i32
// CHECK-NEXT:        [[SEL2:%.+]] = arith.select [[CMP2]], [[ONE]], [[SEL1]] : i32
// CHECK-NEXT:        linalg.yield [[SEL2]] : i32
// CHECK-NEXT:      } -> tensor<2000000xi32>
// CHECK-NEXT:      [[EXPAND:%.+]] = tensor.expand_shape [[OP]] {{\[\[}}0, 1, 2, 3{{\]\]}} output_shape [1, 50, 20, 2000] : tensor<2000000xi32> into tensor<1x50x20x2000xi32>
// CHECK-NEXT:      return [[EXPAND]] : tensor<1x50x20x2000xi32>
    }
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>

// CHECK: [[C:#.+]] = affine_map<(d0) -> (d0)>
// CHECK: [[map:#.+]] = affine_map<(d0) -> ()>
// CHECK: module @BroadcastingDivide
module @BroadcastingDivide {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x1x128x32xf16>, %arg1: tensor<1x1x1x1xf16>, %arg2: tensor<1x1x128x32xf16>) -> tensor<1x1x128x32xf16> {
      %0 = linalg.generic {indexing_maps = [#NCHW, #map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x1x128x32xf16>, tensor<1x1x1x1xf16>) outs(%arg2 : tensor<1x1x128x32xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %1 = arith.divf %in, %in_0 fastmath<arcp> : f16
        linalg.yield %1 : f16
      } -> tensor<1x1x128x32xf16>
      return %0 : tensor<1x1x128x32xf16>

// CHECK: func.func @generated_0([[ARG0:%.+]]: tensor<1x1x128x32xf16>, [[ARG1:%.+]]: tensor<1x1x1x1xf16>, [[ARG2:%.+]]: tensor<1x1x128x32xf16>) -> tensor<1x1x128x32xf16>
// CHECK-NEXT:      [[IN2_COLLAPSED:%.+]] = tensor.collapse_shape [[ARG1]] [] : tensor<1x1x1x1xf16> into tensor<f16>
// CHECK-NEXT:      [[IN1_COLLAPSED:%.+]] = tensor.collapse_shape [[ARG0]] {{\[\[}}0, 1, 2, 3{{\]\]}} : tensor<1x1x128x32xf16> into tensor<4096xf16>
// CHECK-NEXT:      [[OUT_COLLAPSED:%.+]] = tensor.collapse_shape [[ARG2]] {{\[\[}}0, 1, 2, 3{{\]\]}} : tensor<1x1x128x32xf16> into tensor<4096xf16>
// CHECK-NEXT:      [[OP:%.+]] = linalg.generic {indexing_maps = [[[C]], [[map]], [[C]]], iterator_types = ["parallel"]} ins([[IN1_COLLAPSED]], [[IN2_COLLAPSED]] : tensor<4096xf16>, tensor<f16>) outs([[OUT_COLLAPSED]] : tensor<4096xf16>) {
// CHECK-NEXT:      ^bb0([[IN1:%.+]]: f16, [[IN2:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[RES:%.+]] = arith.divf [[IN1]], [[IN2]] fastmath<arcp> : f16
// CHECK-NEXT:        linalg.yield [[RES]] : f16
// CHECK-NEXT:      } -> tensor<4096xf16>
// CHECK-NEXT:      [[EXPAND:%.+]] = tensor.expand_shape [[OP]] {{\[\[}}0, 1, 2, 3{{\]\]}} output_shape [1, 1, 128, 32] : tensor<4096xf16> into tensor<1x1x128x32xf16>
// CHECK-NEXT:      return [[EXPAND]] : tensor<1x1x128x32xf16>
    }
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>

// CHECK: module @PaddedBroadcastingDivide
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape
module @PaddedBroadcastingDivide {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x1x128x32xf16>, %arg1: tensor<1x1x1x1xf16>, %arg2: tensor<1x1x128x32xf16>) -> tensor<1x1x128x48xf16> {
      %c_zero = arith.constant 0. : f16
      %0 = linalg.generic {indexing_maps = [#NCHW, #map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x1x128x32xf16>, tensor<1x1x1x1xf16>) outs(%arg2 : tensor<1x1x128x32xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %1 = arith.divf %in, %in_0 fastmath<arcp> : f16
        linalg.yield %1 : f16
      } -> tensor<1x1x128x32xf16>
      %1 = tensor.pad %0 low[0, 0, 0, 8] high[0, 0, 0, 8] {
      ^bb0(%in0 : index, %in1 : index, %in2 : index, %in3 : index):
        tensor.yield %c_zero : f16
      } : tensor<1x1x128x32xf16> to tensor<1x1x128x48xf16>

      return %1 : tensor<1x1x128x48xf16>
    }
  }
}

// -----

// CHECK: [[NC:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: [[map:#.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK: module @PartialFlattenInnerBroadcast
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
module @PartialFlattenInnerBroadcast {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x2441x20x7xf16>, %arg1: tensor<1x2441x1x1xf16>, %arg2: tensor<1x2441x20x7xf16>) -> tensor<1x2441x20x7xf16> {
      %0 = linalg.generic {indexing_maps = [#NCHW, #map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x2441x20x7xf16>, tensor<1x2441x1x1xf16>) outs(%arg2 : tensor<1x2441x20x7xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %1 = arith.subf %in, %in_0 : f16
        linalg.yield %1 : f16
      } -> tensor<1x2441x20x7xf16>
      return %0 : tensor<1x2441x20x7xf16>

// CHECK:    func.func @generated_0([[ARG0:%.+]]: tensor<1x2441x20x7xf16>, [[ARG1:%.+]]: tensor<1x2441x1x1xf16>, [[ARG2:%.+]]: tensor<1x2441x20x7xf16>) -> tensor<1x2441x20x7xf16> {
// CHECK-NEXT:      [[COLLAPSED_IN1:%.+]] = tensor.collapse_shape [[ARG1]] {{\[\[}}0, 1, 2, 3{{\]\]}} : tensor<1x2441x1x1xf16> into tensor<2441xf16>
// CHECK-NEXT:      [[COLLAPSED_IN0:%.+]] = tensor.collapse_shape [[ARG0]] {{\[\[}}0, 1], [2, 3{{\]\]}} : tensor<1x2441x20x7xf16> into tensor<2441x140xf16>
// CHECK-NEXT:      [[COLLAPSED_OUT:%.+]] = tensor.collapse_shape [[ARG2]] {{\[\[}}0, 1], [2, 3{{\]\]}} : tensor<1x2441x20x7xf16> into tensor<2441x140xf16>
// CHECK-NEXT:      [[OP:%.+]] = linalg.generic {indexing_maps = [[[NC]], [[map]], [[NC]]], iterator_types = ["parallel", "parallel"]} ins([[COLLAPSED_IN0]], [[COLLAPSED_IN1]] : tensor<2441x140xf16>, tensor<2441xf16>) outs([[COLLAPSED_OUT]] : tensor<2441x140xf16>) {
// CHECK-NEXT:      ^bb0([[IN0:%.+]]: f16, [[IN1:%.+]]: f16, {{.+}}: f16):
// CHECK-NEXT:        [[SUB:%.+]] = arith.subf [[IN0]], [[IN1]] : f16
// CHECK-NEXT:        linalg.yield [[SUB]] : f16
// CHECK-NEXT:      } -> tensor<2441x140xf16>
// CHECK-NEXT:      [[EXPAND:%.+]] = tensor.expand_shape [[OP]] {{\[\[}}0, 1], [2, 3{{\]\]}} output_shape [1, 2441, 20, 7] : tensor<2441x140xf16> into tensor<1x2441x20x7xf16>
// CHECK-NEXT:      return [[EXPAND]] : tensor<1x2441x20x7xf16>
    }
  }
}

// -----

// CHECK: [[NC:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: [[map:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK: module @PartialFlattenOuterBroadcast
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
module @PartialFlattenOuterBroadcast {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x2441x20x7xf16>, %arg1: tensor<1x1x20x7xf16>, %arg2: tensor<1x2441x20x7xf16>) -> tensor<1x2441x20x7xf16> {
      %0 = linalg.generic {indexing_maps = [#NCHW, #map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x2441x20x7xf16>, tensor<1x1x20x7xf16>) outs(%arg2 : tensor<1x2441x20x7xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %1 = arith.subf %in, %in_0 : f16
        linalg.yield %1 : f16
      } -> tensor<1x2441x20x7xf16>
      return %0 : tensor<1x2441x20x7xf16>

// CHECK:    func.func @generated_0([[ARG0:%.+]]: tensor<1x2441x20x7xf16>, [[ARG1:%.+]]: tensor<1x1x20x7xf16>, [[ARG2:%.+]]: tensor<1x2441x20x7xf16>) -> tensor<1x2441x20x7xf16> {
// CHECK-NEXT:      [[COLLAPSED_IN0:%.+]] = tensor.collapse_shape [[ARG0]] {{\[\[}}0, 1], [2, 3{{\]\]}} : tensor<1x2441x20x7xf16> into tensor<2441x140xf16>
// CHECK-NEXT:      [[COLLAPSED_IN1:%.+]] = tensor.collapse_shape [[ARG1]] {{\[\[}}0, 1, 2, 3{{\]\]}} : tensor<1x1x20x7xf16> into tensor<140xf16>
// CHECK-NEXT:      [[COLLAPSED_OUT:%.+]] = tensor.collapse_shape [[ARG2]] {{\[\[}}0, 1], [2, 3{{\]\]}} : tensor<1x2441x20x7xf16> into tensor<2441x140xf16>
// CHECK-NEXT:      [[OP:%.+]] = linalg.generic {indexing_maps = [[[NC]], [[map]], [[NC]]], iterator_types = ["parallel", "parallel"]} ins([[COLLAPSED_IN0]], [[COLLAPSED_IN1]] : tensor<2441x140xf16>, tensor<140xf16>) outs([[COLLAPSED_OUT]] : tensor<2441x140xf16>) {
// CHECK-NEXT:      ^bb0([[IN0:%.+]]: f16, [[IN1:%.+]]: f16, %out: f16):
// CHECK-NEXT:        [[SUB:%.+]] = arith.subf [[IN0]], [[IN1]] : f16
// CHECK-NEXT:        linalg.yield [[SUB]] : f16
// CHECK-NEXT:      } -> tensor<2441x140xf16>
// CHECK-NEXT:      [[EXPAND:%.+]] = tensor.expand_shape [[OP]] {{\[\[}}0, 1], [2, 3{{\]\]}} output_shape [1, 2441, 20, 7] : tensor<2441x140xf16> into tensor<1x2441x20x7xf16>
// CHECK-NEXT:      return [[EXPAND]] : tensor<1x2441x20x7xf16>
    }
  }
}
