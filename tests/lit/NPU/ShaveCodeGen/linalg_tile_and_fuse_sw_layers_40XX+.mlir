//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --linalg-tile-and-fuse-sw-layers %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module @Reduce {
  module @VPU.SW {
    func.func @reduce(%arg0: tensor<1x128x64x64xf16>, %arg1: tensor<1x128x1x1xf16>) -> tensor<1x128x1x1xf16> {
        %cst = arith.constant 0.000000e+00 : f32
        %1 = tensor.empty() : tensor<1x128xf32>
        %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x128xf32>) -> tensor<1x128xf32>
        %3 = linalg.generic {indexing_maps = [#NCHW, #map], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%arg0 : tensor<1x128x64x64xf16>) outs(%2 : tensor<1x128xf32>) {
        ^bb0(%in: f16, %out: f32):
          %6 = arith.extf %in : f16 to f32
          %7 = arith.addf %out, %6 fastmath<reassoc> : f32
          linalg.yield %7 : f32
        } -> tensor<1x128xf32>
        %4 = tensor.empty() : tensor<1x128x1x1xf16>
        %5 = linalg.generic {indexing_maps = [#map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x128xf32>) outs(%arg1 : tensor<1x128x1x1xf16>) {
        ^bb0(%in: f32, %out: f16):
          %7 = arith.truncf %in : f32 to f16
          linalg.yield %7 : f16
        } -> tensor<1x128x1x1xf16>
        return %5 : tensor<1x128x1x1xf16>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK: func.func @reduce([[ARG0:%.+]]: tensor<1x128x64x64xf16>, [[ARG1:%.+]]: tensor<1x128x1x1xf16>) -> tensor<1x128x1x1xf16> {
// CHECK-NEXT:     [[C64:%.+]] = arith.constant 64 : index
// CHECK-NEXT:     [[C128:%.+]] = arith.constant 128 : index
// CHECK-NEXT:     [[C1:%.+]] = arith.constant 1 : index
// CHECK-NEXT:     [[C0:%.+]] = arith.constant 0 : index
// CHECK-NEXT:     [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     [[RET:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[C128]] step [[C1]] iter_args([[ARG3:%.+]] = [[ARG1]]) -> (tensor<1x128x1x1xf16>) {
// CHECK-NEXT:       [[EXT_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, [[ARG2]], 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : tensor<1x128x64x64xf16> to tensor<1x1x64x64xf16>
// CHECK-NEXT:       [[EMPTY:%.+]] = tensor.empty() : tensor<1x1xf32>
// CHECK-NEXT:       [[FILL:%.+]] = linalg.fill ins([[CST]] : f32) outs([[EMPTY]] : tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT:       [[REDUCE_OUTER:%.+]] = scf.for [[ARG4:%.+]] = [[C0]] to [[C64]] step [[C1]] iter_args([[ARG5:%.+]] = [[FILL]]) -> (tensor<1x1xf32>) {
// CHECK-NEXT:         [[REDUCE_INNER:%.+]] = scf.for [[ARG6:%.+]] = [[C0]] to [[C64]] step [[C1]] iter_args([[ARG7:%.+]] = [[ARG5]]) -> (tensor<1x1xf32>) {
// CHECK-NEXT:           [[EXT_SLICE_REDUCE:%.+]] = tensor.extract_slice [[EXT_SLICE]][0, 0, [[ARG4]], [[ARG6]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x64x64xf16> to tensor<1x1x1x1xf16>
// CHECK-NEXT:           [[REDUCE_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins([[EXT_SLICE_REDUCE]] : tensor<1x1x1x1xf16>) outs([[ARG7]] : tensor<1x1xf32>) {
// CHECK-NEXT:           ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f32):
// CHECK-NEXT:             [[IN_EXT:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK-NEXT:             [[ADD:%.+]] = arith.addf [[OUT]], [[IN_EXT]] fastmath<reassoc> : f32
// CHECK-NEXT:             linalg.yield [[ADD]] : f32
// CHECK-NEXT:           } -> tensor<1x1xf32>
// CHECK-NEXT:           scf.yield [[REDUCE_OP]] : tensor<1x1xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield [[REDUCE_INNER]] : tensor<1x1xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       [[EXT_SLICE_NORM:%.+]] = tensor.extract_slice [[ARG3]][0, [[ARG2]], 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x128x1x1xf16> to tensor<1x1x1x1xf16>
// CHECK-NEXT:       [[NORM:%.+]] = linalg.generic {indexing_maps = [[[map]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[REDUCE_OUTER]] : tensor<1x1xf32>) outs([[EXT_SLICE_NORM]] : tensor<1x1x1x1xf16>) {
// CHECK-NEXT:       ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f16):
// CHECK-NEXT:         [[TRUNC:%.+]] = arith.truncf [[IN]] : f32 to f16
// CHECK-NEXT:         linalg.yield [[TRUNC]] : f16
// CHECK-NEXT:       } -> tensor<1x1x1x1xf16>
// CHECK-NEXT:       [[INS_SLICE_NORM:%.+]] = tensor.insert_slice [[NORM]] into [[ARG3]][0, [[ARG2]], 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf16> into tensor<1x128x1x1xf16>
// CHECK-NEXT:       scf.yield [[INS_SLICE_NORM]] : tensor<1x128x1x1xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return [[RET]] : tensor<1x128x1x1xf16>
    }
  }
}
