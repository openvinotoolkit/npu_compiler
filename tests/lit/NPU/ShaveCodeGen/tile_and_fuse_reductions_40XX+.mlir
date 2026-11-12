//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --linalg-tile-and-fuse-sw-layers %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>

module @StackedReduce {
  module @VPU.SW {
    func.func @StackedReduce(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x1x1xf32> {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = tensor.empty() : tensor<1x16x4000xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x16x4000xf32>) -> tensor<1x16x4000xf32>
      %2 = linalg.generic {indexing_maps = [#NCHW, #map], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x16x4000x200xf32>) outs(%1 : tensor<1x16x4000xf32>) {
      ^bb0(%in: f32, %out: f32):
        %12 = arith.addf %out, %in fastmath<reassoc> : f32
        linalg.yield %12 : f32
      } -> tensor<1x16x4000xf32>
      %3 = tensor.empty() : tensor<1x16x4000x1xf32>
      %4 = linalg.generic {indexing_maps = [#map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x16x4000xf32>) outs(%3 : tensor<1x16x4000x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x16x4000x1xf32>
      %5 = tensor.empty() : tensor<1x16x4000x1xf32>
      %6 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<1x16x4000x1xf32>) outs(%5 : tensor<1x16x4000x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %12 = math.log %in fastmath<afn> : f32
        linalg.yield %12 : f32
      } -> tensor<1x16x4000x1xf32>
      %7 = tensor.empty() : tensor<1x16x1xf32>
      %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
      %9 = linalg.generic {indexing_maps = [#NCHW, #map1], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%6 : tensor<1x16x4000x1xf32>) outs(%8 : tensor<1x16x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %12 = arith.addf %out, %in fastmath<reassoc> : f32
        linalg.yield %12 : f32
      } -> tensor<1x16x1xf32>
      %10 = tensor.empty() : tensor<1x16x1x1xf32>
      %11 = linalg.generic {indexing_maps = [#map1, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9 : tensor<1x16x1xf32>) outs(%10 : tensor<1x16x1x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x16x1x1xf32>
      return %11 : tensor<1x16x1x1xf32>

// CHECK: func.func @StackedReduce([[ARG0:%.+]]: tensor<1x16x4000x200xf32>) -> tensor<1x16x1x1xf32> {
// CHECK-DAG:     [[C200:%.+]] = arith.constant 200 : index
// CHECK-DAG:     [[C4000:%.+]] = arith.constant 4000 : index
// CHECK-DAG:     [[C16:%.+]] = arith.constant 16 : index
// CHECK-DAG:     [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     [[EMPT_RET:%.+]] = tensor.empty() : tensor<1x16x1x1xf32>
// CHECK-DAG:     [[EMPTY3:%.+]] = tensor.empty() : tensor<1x1x1xf32>
// CHECK-DAG:     [[ZERO3:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPTY3]] : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
// CHECK-DAG:     [[EMPTY4:%.+]] = tensor.empty() : tensor<1x1x1x1xf32>
// CHECK-NEXT:    [[FOR_C:%.+]] = scf.for [[C_IDX:%.+]] = [[C0]] to [[C16]] step [[C1]] iter_args([[ACCUM_C:%.+]] = [[EMPT_RET]]) -> (tensor<1x16x1x1xf32>) {
// CHECK-NEXT:      [[FOR_H:%.+]] = scf.for [[H_IDX:%.+]] = [[C0]] to [[C4000]] step [[C1]] iter_args([[ACCUM_H:%.+]] = [[ZERO3]]) -> (tensor<1x1x1xf32>) {
// CHECK-NEXT:        [[FOR_W:%.+]] = scf.for [[W_IDX:%.+]] = [[C0]] to [[C200]] step [[C1]] iter_args([[ACCUM_W:%.+]] = [[ZERO3]]) -> (tensor<1x1x1xf32>) {
// CHECK-NEXT:          [[SCALAR:%.+]] = tensor.extract_slice [[ARG0]][0, [[C_IDX]], [[H_IDX]], [[W_IDX]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x16x4000x200xf32> to tensor<1x1x1x1xf32>
// CHECK-NEXT:          [[L0:%.+]] = linalg.generic
// CHECK-SAME:                ins([[SCALAR]] : tensor<1x1x1x1xf32>)
// CHECK-SAME:                outs([[ACCUM_W]] : tensor<1x1x1xf32>)
// CHECK:               scf.yield [[L0]] : tensor<1x1x1xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        [[L1:%.+]] = linalg.generic
// CHECK-SAME:             ins([[FOR_W]] : tensor<1x1x1xf32>)
// CHECK-SAME:             outs([[EMPTY4]] : tensor<1x1x1x1xf32>)
// CHECK:             [[L2:%.+]] = linalg.generic
// CHECK-SAME:             ins([[L1]] : tensor<1x1x1x1xf32>)
// CHECK-SAME:             outs([[EMPTY4]] : tensor<1x1x1x1xf32>)
// CHECK:             [[L3:%.+]] = linalg.generic
// CHECK-SAME:             ins([[L2]] : tensor<1x1x1x1xf32>)
// CHECK-SAME:             outs([[ACCUM_H]] : tensor<1x1x1xf32>)
// CHECK:             scf.yield [[L3]] : tensor<1x1x1xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      [[SLICE_C:%.+]] = tensor.extract_slice [[ACCUM_C]][0, [[C_IDX]], 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x16x1x1xf32> to tensor<1x1x1x1xf32>
// CHECK-NEXT:      [[L4:%.+]] = linalg.generic
// CHECK-SAME:           ins([[FOR_H]] : tensor<1x1x1xf32>)
// CHECK-SAME:           outs([[SLICE_C]] : tensor<1x1x1x1xf32>)
// CHECK:           [[UPDATED:%.+]] = tensor.insert_slice [[L4]] into [[ACCUM_C]][0, [[C_IDX]], 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x16x1x1xf32>
// CHECK-NEXT:      scf.yield [[UPDATED]] : tensor<1x16x1x1xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return [[FOR_C]] : tensor<1x16x1x1xf32>
    }
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module @SoftmaxF16 {
  module @VPU.SW {
    func.func @SoftmaxF16(%arg0: tensor<1x16x4000x200xf16>) -> tensor<1x16x4000x200xf16> {
      %0 = tensor.empty() : tensor<1x16x4000x200xf16>
      %cst = arith.constant -6.550400e+04 : f16
      %1 = tensor.empty() : tensor<1x16x4000xf16>
      %2 = linalg.fill ins(%cst : f16) outs(%1 : tensor<1x16x4000xf16>) -> tensor<1x16x4000xf16>
      %3 = linalg.generic {indexing_maps = [#NCHW, #map], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x16x4000x200xf16>) outs(%2 : tensor<1x16x4000xf16>) {
      ^bb0(%in: f16, %out: f16):
        %9 = arith.maximumf %in, %out fastmath<nnan,nsz> : f16
        linalg.yield %9 : f16
      } -> tensor<1x16x4000xf16>
      %4 = linalg.generic {indexing_maps = [#NCHW, #map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %3 : tensor<1x16x4000x200xf16>, tensor<1x16x4000xf16>) outs(%0 : tensor<1x16x4000x200xf16>) {
      ^bb0(%in: f16, %in_1: f16, %out: f16):
        %9 = arith.subf %in, %in_1 : f16
        %10 = math.exp %9 fastmath<afn> : f16
        linalg.yield %10 : f16
      } -> tensor<1x16x4000x200xf16>
      %5 = tensor.empty() : tensor<1x16x4000xf32>
      %cst_0 = arith.constant 0.000000e+00 : f32
      %6 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<1x16x4000xf32>) -> tensor<1x16x4000xf32>
      %7 = linalg.generic {indexing_maps = [#NCHW, #map], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4 : tensor<1x16x4000x200xf16>) outs(%6 : tensor<1x16x4000xf32>) {
      ^bb0(%in: f16, %out: f32):
        %9 = arith.extf %in : f16 to f32
        %10 = arith.addf %out, %9 fastmath<reassoc> : f32
        linalg.yield %10 : f32
      } -> tensor<1x16x4000xf32>
      %8 = linalg.generic {indexing_maps = [#NCHW, #map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %7 : tensor<1x16x4000x200xf16>, tensor<1x16x4000xf32>) outs(%0 : tensor<1x16x4000x200xf16>) {
      ^bb0(%in: f16, %in_1: f32, %out: f16):
        %9 = arith.extf %in : f16 to f32
        %10 = arith.divf %9, %in_1 fastmath<arcp> : f32
        %11 = arith.truncf %10 : f32 to f16
        linalg.yield %11 : f16
      } -> tensor<1x16x4000x200xf16>
      return %8 : tensor<1x16x4000x200xf16>

// CHECK:    func.func @SoftmaxF16([[ARG0:%.+]]: tensor<1x16x4000x200xf16>) -> tensor<1x16x4000x200xf16> {
// CHECK-DAG:    [[C200:%.+]] = arith.constant 200 : index
// CHECK-DAG:    [[C4000:%.+]] = arith.constant 4000 : index
// CHECK-DAG:    [[C16:%.+]] = arith.constant 16 : index
// CHECK-DAG:    [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:    [[MAX_NEG:%.+]] = arith.constant -6.550400e+04 : f16
// CHECK-DAG:    [[EMPT_RET:%.+]] = tensor.empty() : tensor<1x16x4000x200xf16>
// CHECK-DAG:    [[EMPTY_F16_3:%.+]] = tensor.empty() : tensor<1x1x1xf16>
// CHECK-DAG:    [[INIT_MAX_ACCUM:%.+]] = linalg.fill ins([[MAX_NEG]] : f16) outs([[EMPTY_F16_3]] : tensor<1x1x1xf16>) -> tensor<1x1x1xf16>
// CHECK-DAG:    [[EMPTY_F16_4:%.+]] = tensor.empty() : tensor<1x1x1x1xf16>
// CHECK-DAG:    [[EMPTY_F32_3:%.+]] = tensor.empty() : tensor<1x1x1xf32>
// CHECK-DAG:    [[SUM_ACCUM_INIT:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPTY_F32_3]] : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
// CHECK-NEXT:   [[FOR_C:%.+]] = scf.for [[IDX_C:%.+]] = [[C0]] to [[C16]] step [[C1]] iter_args([[ACCUM_C:%.+]] = [[EMPT_RET]]) -> (tensor<1x16x4000x200xf16>) {
// CHECK-NEXT:     [[FOR_H:%.+]] = scf.for [[IDX_H:%.+]] = [[C0]] to [[C4000]] step [[C1]] iter_args([[ACCUM_H:%.+]] = [[ACCUM_C]]) -> (tensor<1x16x4000x200xf16>) {
// CHECK-NEXT:       [[MAX:%.+]] = scf.for [[IDX_W:%.+]] = [[C0]] to [[C200]] step [[C1]] iter_args([[MAX_ACCUM:%.+]] = [[INIT_MAX_ACCUM]]) -> (tensor<1x1x1xf16>) {
// CHECK-NEXT:         [[MAX_IN_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, [[IDX_C]], [[IDX_H]], [[IDX_W]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x16x4000x200xf16> to tensor<1x1x1x1xf16>
// CHECK-NEXT:         [[L0:%.+]] = linalg.generic
// CHECK-SAME:               ins([[MAX_IN_SLICE]] : tensor<1x1x1x1xf16>)
// CHECK-SAME:               outs([[MAX_ACCUM]] : tensor<1x1x1xf16>)
// CHECK:              scf.yield [[L0]] : tensor<1x1x1xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:       [[SUM:%.+]] = scf.for [[IDX_W:.+]] = [[C0]] to [[C200]] step [[C1]] iter_args([[SUM_ACCUM:%.+]] = [[SUM_ACCUM_INIT]]) -> (tensor<1x1x1xf32>) {
// CHECK-NEXT:         [[REDUCE_SUM_SLICE_W:%.+]] = tensor.extract_slice [[ARG0]][0, [[IDX_C]], [[IDX_H]], [[IDX_W]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x16x4000x200xf16> to tensor<1x1x1x1xf16>
// CHECK-NEXT:         [[L1:%.+]] = linalg.generic
// CHECK-SAME:               ins([[REDUCE_SUM_SLICE_W]], [[MAX]] : tensor<1x1x1x1xf16>, tensor<1x1x1xf16>)
// CHECK-SAME:               outs([[EMPTY_F16_4]] : tensor<1x1x1x1xf16>)
// CHECK:              [[L2:%.+]] = linalg.generic
// CHECK-SAME:               ins([[L1]] : tensor<1x1x1x1xf16>)
// CHECK-SAME:               outs([[SUM_ACCUM]] : tensor<1x1x1xf32>)
// CHECK:              scf.yield [[L2]] : tensor<1x1x1xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       [[NORM:%.+]] = scf.for [[IDX_W:%.+]] = [[C0]] to [[C200]] step [[C1]] iter_args([[NORM_ACCUM:%.+]] = [[ACCUM_H]]) -> (tensor<1x16x4000x200xf16>) {
// CHECK-NEXT:         [[NORM_SLICE_W:%.+]] = tensor.extract_slice [[ARG0]][0, [[IDX_C]], [[IDX_H]], [[IDX_W]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x16x4000x200xf16> to tensor<1x1x1x1xf16>
// CHECK-NEXT:         [[L3:%.+]] = linalg.generic
// CHECK-SAME:               ins([[NORM_SLICE_W]], [[MAX]] : tensor<1x1x1x1xf16>, tensor<1x1x1xf16>)
// CHECK-SAME:               outs([[EMPTY_F16_4]] : tensor<1x1x1x1xf16>)
// CHECK:              [[NORM_OUT_SLICE_W:%.+]] = tensor.extract_slice [[NORM_ACCUM]][0, [[IDX_C]], [[IDX_H]], [[IDX_W]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x16x4000x200xf16> to tensor<1x1x1x1xf16>
// CHECK-NEXT:         [[L4:%.+]] = linalg.generic
// CHECK-SAME:               ins([[L3]], [[SUM]] : tensor<1x1x1x1xf16>, tensor<1x1x1xf32>)
// CHECK-SAME:               outs([[NORM_OUT_SLICE_W]] : tensor<1x1x1x1xf16>) {
// CHECK:              [[UPDATED:%.+]] = tensor.insert_slice [[L4]] into [[NORM_ACCUM]][0, [[IDX_C]], [[IDX_H]], [[IDX_W]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf16> into tensor<1x16x4000x200xf16>
// CHECK-NEXT:         scf.yield [[UPDATED]] : tensor<1x16x4000x200xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield [[NORM]] : tensor<1x16x4000x200xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield [[FOR_H]] : tensor<1x16x4000x200xf16>
// CHECK-NEXT:   }
// CHECK-NEXT:   return [[FOR_C]] : tensor<1x16x4000x200xf16>
    }
  }
}
