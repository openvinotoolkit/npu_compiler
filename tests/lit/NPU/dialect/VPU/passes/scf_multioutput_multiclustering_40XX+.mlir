//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --scf-multiclustering --canonicalize --cse %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// Multiclustering: LSTMGates already tiled into scf.for, with multiClusterStrategy.
// LSTMGates has two outputs (hiddenState, cellState), both inserted back.
// MC pass should handle multi-output insert_slice/yield correctly.

// CHECK-LABEL: @MCLSTMGatesSplitOverHeight
func.func @MCLSTMGatesSplitOverHeight(
    %arg0: tensor<1x1x128x512xf16>,
    %arg1: tensor<1x1x128x128xf16>
) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %out0 = tensor.empty() : tensor<1x1x128x128xf16>
    %out1 = tensor.empty() : tensor<1x1x128x128xf16>

    %result:2 = scf.for %iv = %c0 to %c128 step %c32
        iter_args(%arg2 = %out0, %arg3 = %out1) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) {

      %slice_gates = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 1, 32, 512] [1, 1, 1, 1]
          : tensor<1x1x128x512xf16> to tensor<1x1x32x512xf16>
      %slice_cell = tensor.extract_slice %arg1[0, 0, %iv, 0] [1, 1, 32, 128] [1, 1, 1, 1]
          : tensor<1x1x128x128xf16> to tensor<1x1x32x128xf16>

      %h, %c = VPU.LSTMGates(%slice_gates, %slice_cell) {
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
      } : tensor<1x1x32x512xf16>, tensor<1x1x32x128xf16>
        -> tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>

      %ins_h = tensor.insert_slice %h into %arg2[0, 0, %iv, 0] [1, 1, 32, 128] [1, 1, 1, 1]
          : tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>
      %ins_c = tensor.insert_slice %c into %arg3[0, 0, %iv, 0] [1, 1, 32, 128] [1, 1, 1, 1]
          : tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>

      scf.yield %ins_h, %ins_c : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>
    }

    return %result#0, %result#1 : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>

//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
//CHECK-DAG:    [[C32:%.+]] = arith.constant 32 : index
//CHECK-DAG:    [[EMPTY0:%.+]] = tensor.empty() : tensor<1x1x128x128xf16>
//CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX0:%.+]] = [[C0]] to [[C128]] step [[C32]] iter_args([[ACC0:%.+]] = [[EMPTY0]], [[ACC1:%.+]] = [[EMPTY0]]) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>)
//CHECK-NEXT:      [[EXTRACT0:%.+]] = tensor.extract_slice [[ARG0:%.+]][0, 0, [[IDX0]], 0] [1, 1, 32, 512] [1, 1, 1, 1] : tensor<1x1x128x512xf16> to tensor<1x1x32x512xf16>
//CHECK-NEXT:      [[EXTRACT1:%.+]] = tensor.extract_slice [[ARG1:%.+]][0, 0, [[IDX0]], 0] [1, 1, 32, 128] [1, 1, 1, 1] : tensor<1x1x128x128xf16> to tensor<1x1x32x128xf16>
//CHECK-NEXT:      [[EMPTY1:%.+]] = tensor.empty() : tensor<1x1x32x128xf16>
//CHECK-NEXT:      [[FOR1:%.+]]:2 = scf.forall ([[IDX1:%.+]]) = (0) to (32) step (6) shared_outs([[ACC2:%.+]] = [[EMPTY1]], [[ACC3:%.+]] = [[EMPTY1]]) -> (tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
//CHECK-NEXT:           [[MIN:%.+]] = affine.min #map([[IDX1]])
//CHECK-NEXT:           [[EXTRACT2:%.+]] = tensor.extract_slice [[EXTRACT0]][0, 0, [[IDX1]], 0] [1, 1, [[MIN]], 512] [1, 1, 1, 1] : tensor<1x1x32x512xf16> to tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 32, 512]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:           [[EXTRACT3:%.+]] = tensor.extract_slice [[EXTRACT1]][0, 0, [[IDX1]], 0] [1, 1, [[MIN]], 128] [1, 1, 1, 1] : tensor<1x1x32x128xf16> to tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 32, 128]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:           [[H:%.+]], [[C:%.+]] = VPU.LSTMGates([[EXTRACT2]], [[EXTRACT3]]) : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 32, 512]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 32, 128]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-SAME:               -> tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 32, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 32, 128]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:           scf.forall.in_parallel {
//CHECK-NEXT:                   tensor.parallel_insert_slice [[H]] into [[ACC2]][0, 0, [[IDX1]], 0] [1, 1, [[MIN]], 128] [1, 1, 1, 1]
//CHECK-SAME:                       : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 32, 128]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x32x128xf16>
//CHECK-NEXT:                   tensor.parallel_insert_slice [[C]] into [[ACC3]][0, 0, [[IDX1]], 0] [1, 1, [[MIN]], 128] [1, 1, 1, 1]
//CHECK-SAME:                       : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 32, 128]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x32x128xf16>
//CHECK-NEXT:           }
//CHECK-NEXT:      }
//CHECK-NEXT:      [[INSERT0:%.+]] = tensor.insert_slice [[FOR1]]#0 into [[ACC0]][0, 0, [[IDX0]], 0] [1, 1, 32, 128] [1, 1, 1, 1] : tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>
//CHECK-NEXT:      [[INSERT1:%.+]] = tensor.insert_slice [[FOR1]]#1 into [[ACC1]][0, 0, [[IDX0]], 0] [1, 1, 32, 128] [1, 1, 1, 1] : tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>
//CHECK-NEXT:      scf.yield [[INSERT0]], [[INSERT1]] : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>
//CHECK:        return [[SCF]]#0, [[SCF]]#1 : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map_tile = affine_map<(d0) -> (-d0 + 1024, 205)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// Multiclustering: LSTMGates with larger shape and boundary tile.
// Tiled over dim 2 with step 205 (1024/205 non-divisible, last tile is 4).
// Input IR matches apply-tiling output: affine.min for tile size, dynamic types with bounds.

// CHECK-LABEL: @MCLSTMGatesLargeShape
func.func @MCLSTMGatesLargeShape(
    %arg0: tensor<1x1x1024x2048xf16>,
    %arg1: tensor<1x1x1024x512xf16>
) -> (tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>) {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c205 = arith.constant 205 : index
    %out0 = tensor.empty() : tensor<1x1x1024x512xf16>
    %out1 = tensor.empty() : tensor<1x1x1024x512xf16>

    %result:2 = scf.for %iv = %c0 to %c1024 step %c205
        iter_args(%arg2 = %out0, %arg3 = %out1) -> (tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>) {

      %tile_sz = affine.min #map_tile(%iv)

      %slice_gates = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 1, %tile_sz, 2048] [1, 1, 1, 1]
          : tensor<1x1x1024x2048xf16> to tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>, order = #NCHW}>
      %slice_cell = tensor.extract_slice %arg1[0, 0, %iv, 0] [1, 1, %tile_sz, 512] [1, 1, 1, 1]
          : tensor<1x1x1024x512xf16> to tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>

      %h, %c = VPU.LSTMGates(%slice_gates, %slice_cell) {
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
      } : tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>, order = #NCHW}>,
          tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>,
           tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>

      %ins_h = tensor.insert_slice %h into %arg2[0, 0, %iv, 0] [1, 1, %tile_sz, 512] [1, 1, 1, 1]
          : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x1024x512xf16>
      %ins_c = tensor.insert_slice %c into %arg3[0, 0, %iv, 0] [1, 1, %tile_sz, 512] [1, 1, 1, 1]
          : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x1024x512xf16>

      scf.yield %ins_h, %ins_c : tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>
    }

    return %result#0, %result#1 : tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>

//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[C1024:%.+]] = arith.constant 1024 : index
//CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
//CHECK-DAG:    [[EMPTY0:%.+]] = tensor.empty() : tensor<1x1x1024x512xf16>
//CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX0:%.+]] = [[C0]] to [[C1024]] step [[C205]] iter_args([[ACC0:%.+]] = [[EMPTY0]], [[ACC1:%.+]] = [[EMPTY0]]) -> (tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>)
//CHECK:            [[TILE_SZ:%.+]] = affine.min
//CHECK:            [[EXTRACT0:%.+]] = tensor.extract_slice {{%.+}}[0, 0, [[IDX0]], 0] [1, 1, [[TILE_SZ]], 2048]
//CHECK:            [[EXTRACT1:%.+]] = tensor.extract_slice {{%.+}}[0, 0, [[IDX0]], 0] [1, 1, [[TILE_SZ]], 512]
//CHECK:            [[FOR1:%.+]]:2 = scf.forall ([[IDX1:%.+]]) = (0) to ([[TILE_SZ]]) step ({{%.+}}) shared_outs([[ACC2:%.+]] = {{%.+}}, [[ACC3:%.+]] = {{%.+}})
//CHECK:                 [[MIN:%.+]] = affine.min
//CHECK:                 [[EXTRACT2:%.+]] = tensor.extract_slice [[EXTRACT0]][0, 0, [[IDX1]], 0] [1, 1, [[MIN]], 2048]
//CHECK:                 [[EXTRACT3:%.+]] = tensor.extract_slice [[EXTRACT1]][0, 0, [[IDX1]], 0] [1, 1, [[MIN]], 512]
//CHECK:                 [[H:%.+]], [[C:%.+]] = VPU.LSTMGates([[EXTRACT2]], [[EXTRACT3]])
//CHECK:                 scf.forall.in_parallel {
//CHECK:                         tensor.parallel_insert_slice [[H]] into [[ACC2]][0, 0, [[IDX1]], 0] [1, 1, [[MIN]], 512]
//CHECK:                         tensor.parallel_insert_slice [[C]] into [[ACC3]][0, 0, [[IDX1]], 0] [1, 1, [[MIN]], 512]
//CHECK:                 }
//CHECK:            }
//CHECK:            [[INSERT0:%.+]] = tensor.insert_slice [[FOR1]]#0 into [[ACC0]][0, 0, [[IDX0]], 0] [1, 1, [[TILE_SZ]], 512]
//CHECK:            [[INSERT1:%.+]] = tensor.insert_slice [[FOR1]]#1 into [[ACC1]][0, 0, [[IDX0]], 0] [1, 1, [[TILE_SZ]], 512]
//CHECK:            scf.yield [[INSERT0]], [[INSERT1]] : tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>
//CHECK:        return [[SCF]]#0, [[SCF]]#1 : tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>
}
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// Multiclustering: TopK already tiled into scf.for, with multiClusterStrategy.
// TopK has two outputs (values f32, indices si32), both inserted back.
// MC pass should handle mixed-type multi-output insert_slice/yield.

// CHECK-LABEL: @MCTopKSplitOverHeight
func.func @MCTopKSplitOverHeight(
    %arg0: tensor<1x64x128x128xf32>
) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>) {
    %k_buf = VPU.Empty : tensor<1x1x1x1024xui8>
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %out_vals = tensor.empty() : tensor<1x8x128x128xf32>
    %out_inds = tensor.empty() : tensor<1x8x128x128xsi32>

    %result:2 = scf.for %iv = %c0 to %c128 step %c32
        iter_args(%arg1 = %out_vals, %arg2 = %out_inds) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>) {

      %slice_in = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 64, 32, 128] [1, 1, 1, 1]
          : tensor<1x64x128x128xf32> to tensor<1x64x32x128xf32>

      %vals, %inds = VPU.TopK(%slice_in, %k_buf) {
          axis = 1 : i64,
          element_type = si32,
          k_value = 8 : i64,
          mode = #IE.topk_mode<MAX>,
          sort = #IE.topk_sort_type<SORT_INDICES>,
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
      } : tensor<1x64x32x128xf32>, tensor<1x1x1x1024xui8>
        -> tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>

      %ins_vals = tensor.insert_slice %vals into %arg1[0, 0, %iv, 0] [1, 8, 32, 128] [1, 1, 1, 1]
          : tensor<1x8x32x128xf32> into tensor<1x8x128x128xf32>
      %ins_inds = tensor.insert_slice %inds into %arg2[0, 0, %iv, 0] [1, 8, 32, 128] [1, 1, 1, 1]
          : tensor<1x8x32x128xsi32> into tensor<1x8x128x128xsi32>

      scf.yield %ins_vals, %ins_inds : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>
    }

    return %result#0, %result#1 : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>

//CHECK-DAG:    [[C32:%.+]] = arith.constant 32 : index
//CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[EMPTY0:%.+]] = VPU.Empty : tensor<1x1x1x1024xui8>
//CHECK-DAG:    [[EMPTY1:%.+]] = tensor.empty() : tensor<1x8x128x128xf32>
//CHECK-DAG:    [[EMPTY2:%.+]] = tensor.empty() : tensor<1x8x128x128xsi32>
//CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX0:%.+]] = [[C0]] to [[C128]] step [[C32]] iter_args([[ACC0:%.+]] = [[EMPTY1]], [[ACC1:%.+]] = [[EMPTY2]]) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>)
//CHECK-NEXT:       [[EXTRACT0:%.+]] = tensor.extract_slice [[ARG0:%.+]][0, 0, [[IDX0]], 0] [1, 64, 32, 128] [1, 1, 1, 1] : tensor<1x64x128x128xf32> to tensor<1x64x32x128xf32>
//CHECK-NEXT:       [[EMPTY3:%.+]] = tensor.empty() : tensor<1x8x32x128xf32>
//CHECK-NEXT:       [[EMPTY4:%.+]] = tensor.empty() : tensor<1x8x32x128xsi32>
//CHECK-NEXT:       [[FOR1:%.+]]:2 = scf.forall ([[IDX1:%.+]]) = (0) to (32) step (6) shared_outs([[ACC2:%.+]] = [[EMPTY3]], [[ACC3:%.+]] = [[EMPTY4]]) -> (tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>) {
//CHECK-NEXT:           [[MIN:%.+]] = affine.min #map([[IDX1]])
//CHECK-NEXT:           [[EXTRACT1:%.+]] = tensor.extract_slice [[EXTRACT0]][0, 0, [[IDX1]], 0] [1, 64, [[MIN]], 128] [1, 1, 1, 1] : tensor<1x64x32x128xf32> to tensor<1x64x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 32, 128]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:           [[V:%.+]], [[S:%.+]] = VPU.TopK([[EXTRACT1]], [[EMPTY0]]) {axis = 1 : i64, element_type = si32, k_value = 8 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x64x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 32, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x1024xui8>
//CHECK-SAME:               -> tensor<1x8x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 8, 32, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x8x?x128xsi32, {bounds = #const.OpaqueI64Elements<[1, 8, 32, 128]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:           scf.forall.in_parallel {
//CHECK-NEXT:                   tensor.parallel_insert_slice [[V]] into [[ACC2]][0, 0, [[IDX1]], 0] [1, 8, [[MIN]], 128] [1, 1, 1, 1]
//CHECK-SAME:                       : tensor<1x8x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 8, 32, 128]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x8x32x128xf32>
//CHECK-NEXT:                   tensor.parallel_insert_slice [[S]] into [[ACC3]][0, 0, [[IDX1]], 0] [1, 8, [[MIN]], 128] [1, 1, 1, 1]
//CHECK-SAME:                       : tensor<1x8x?x128xsi32, {bounds = #const.OpaqueI64Elements<[1, 8, 32, 128]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x8x32x128xsi32>
//CHECK-NEXT:           }
//CHECK-NEXT:      }
//CHECK-NEXT:      [[INSERT0:%.+]] = tensor.insert_slice [[FOR1]]#0 into [[ACC0]][0, 0, [[IDX0]], 0] [1, 8, 32, 128] [1, 1, 1, 1] : tensor<1x8x32x128xf32> into tensor<1x8x128x128xf32>
//CHECK-NEXT:      [[INSERT1:%.+]] = tensor.insert_slice [[FOR1]]#1 into [[ACC1]][0, 0, [[IDX0]], 0] [1, 8, 32, 128] [1, 1, 1, 1] : tensor<1x8x32x128xsi32> into tensor<1x8x128x128xsi32>
//CHECK-NEXT:      scf.yield [[INSERT0]], [[INSERT1]] : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>
//CHECK:        return [[SCF]]#0, [[SCF]]#1 : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>


}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// Multiclustering: TopK with large shape (matching apply-tiling test dimensions).
// Tiled over dim 2 with step 1024 (4096/4).

// CHECK-LABEL: @MCTopKLargeShape
func.func @MCTopKLargeShape(
    %arg0: tensor<1x512x4096x4096xf32>
) -> (tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>) {
    %k_buf = VPU.Empty : tensor<1x1x1x8192xui8>
    %c0 = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    %c1024 = arith.constant 1024 : index
    %out_vals = tensor.empty() : tensor<1x1x4096x4096xf32>
    %out_inds = tensor.empty() : tensor<1x1x4096x4096xsi32>

    %result:2 = scf.for %iv = %c0 to %c4096 step %c1024
        iter_args(%arg1 = %out_vals, %arg2 = %out_inds) -> (tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>) {

      %slice_in = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 512, 1024, 4096] [1, 1, 1, 1]
          : tensor<1x512x4096x4096xf32> to tensor<1x512x1024x4096xf32>

      %vals, %inds = VPU.TopK(%slice_in, %k_buf) {
          axis = 1 : i64,
          element_type = si32,
          k_value = 1 : i64,
          mode = #IE.topk_mode<MAX>,
          sort = #IE.topk_sort_type<SORT_INDICES>,
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
      } : tensor<1x512x1024x4096xf32>, tensor<1x1x1x8192xui8>
        -> tensor<1x1x1024x4096xf32>, tensor<1x1x1024x4096xsi32>

      %ins_vals = tensor.insert_slice %vals into %arg1[0, 0, %iv, 0] [1, 1, 1024, 4096] [1, 1, 1, 1]
          : tensor<1x1x1024x4096xf32> into tensor<1x1x4096x4096xf32>
      %ins_inds = tensor.insert_slice %inds into %arg2[0, 0, %iv, 0] [1, 1, 1024, 4096] [1, 1, 1, 1]
          : tensor<1x1x1024x4096xsi32> into tensor<1x1x4096x4096xsi32>

      scf.yield %ins_vals, %ins_inds : tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>
    }

    return %result#0, %result#1 : tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>
//CHECK-DAG:    [[C1024:%.+]] = arith.constant 1024 : index
//CHECK-DAG:    [[C4096:%.+]] = arith.constant 4096 : index
//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[EMPTY0:%.+]] = VPU.Empty : tensor<1x1x1x8192xui8>
//CHECK-DAG:    [[EMPTY1:%.+]] = tensor.empty() : tensor<1x1x4096x4096xf32>
//CHECK-DAG:    [[EMPTY2:%.+]] = tensor.empty() : tensor<1x1x4096x4096xsi32>
//CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX0:%.+]] = [[C0]] to [[C4096]] step [[C1024]] iter_args([[ACC0:%.+]] = [[EMPTY1]], [[ACC1:%.+]] = [[EMPTY2]]) -> (tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>)
//CHECK-NEXT:       [[EXTRACT0:%.+]] = tensor.extract_slice [[ARG0:%.+]][0, 0, [[IDX0]], 0] [1, 512, 1024, 4096] [1, 1, 1, 1] : tensor<1x512x4096x4096xf32> to tensor<1x512x1024x4096xf32>
//CHECK-NEXT:       [[EMPTY3:%.+]] = tensor.empty() : tensor<1x1x1024x4096xf32>
//CHECK-NEXT:       [[EMPTY4:%.+]] = tensor.empty() : tensor<1x1x1024x4096xsi32>
//CHECK-NEXT:       [[FOR1:%.+]]:2 = scf.forall ([[IDX1:%.+]]) = (0) to (1024) step (171) shared_outs([[ACC2:%.+]] = [[EMPTY3]], [[ACC3:%.+]] = [[EMPTY4]]) -> (tensor<1x1x1024x4096xf32>, tensor<1x1x1024x4096xsi32>) {
//CHECK-NEXT:           [[MIN:%.+]] = affine.min #map([[IDX1]])
//CHECK-NEXT:           [[EXTRACT1:%.+]] = tensor.extract_slice [[EXTRACT0]][0, 0, [[IDX1]], 0] [1, 512, [[MIN]], 4096] [1, 1, 1, 1] : tensor<1x512x1024x4096xf32> to tensor<1x512x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:           [[V:%.+]], [[S:%.+]] = VPU.TopK([[EXTRACT1]], [[EMPTY0]]) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x512x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x8192xui8>
//CHECK-SAME:               -> tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:           scf.forall.in_parallel {
//CHECK-NEXT:                   tensor.parallel_insert_slice [[V]] into [[ACC2]][0, 0, [[IDX1]], 0] [1, 1, [[MIN]], 4096] [1, 1, 1, 1]
//CHECK-SAME:                       : tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x1024x4096xf32>
//CHECK-NEXT:                   tensor.parallel_insert_slice [[S]] into [[ACC3]][0, 0, [[IDX1]], 0] [1, 1, [[MIN]], 4096] [1, 1, 1, 1]
//CHECK-SAME:                       : tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x1024x4096xsi32>
//CHECK-NEXT:           }
//CHECK-NEXT:      }
//CHECK-NEXT:      [[INSERT0:%.+]] = tensor.insert_slice [[FOR1]]#0 into [[ACC0]][0, 0, [[IDX0]], 0] [1, 1, 1024, 4096] [1, 1, 1, 1] : tensor<1x1x1024x4096xf32> into tensor<1x1x4096x4096xf32>
//CHECK-NEXT:      [[INSERT1:%.+]] = tensor.insert_slice [[FOR1]]#1 into [[ACC1]][0, 0, [[IDX0]], 0] [1, 1, 1024, 4096] [1, 1, 1, 1] : tensor<1x1x1024x4096xsi32> into tensor<1x1x4096x4096xsi32>
//CHECK-NEXT:      scf.yield [[INSERT0]], [[INSERT1]] : tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>
//CHECK:        return [[SCF]]#0, [[SCF]]#1 : tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>

}
}

// -----

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// TopK with Clustering strategy — multi-result op (2 outputs)

// CHECK-LABEL: @TopKClustering
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x64x32x128xf32>
func.func @TopKClustering(%arg0: tensor<1x64x32x128xf32>) -> (tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>) {
  %k_buf = VPU.Empty : tensor<1x1x1x1024xui8>
  %vals, %inds = VPU.TopK(%arg0, %k_buf) {
      axis = 1 : i64,
      element_type = si32,
      k_value = 8 : i64,
      mode = #IE.topk_mode<MAX>,
      sort = #IE.topk_sort_type<SORT_INDICES>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
  } : tensor<1x64x32x128xf32>, tensor<1x1x1x1024xui8>
    -> tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>
  return %vals, %inds : tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>

// CHECK-DAG: [[K_BUF:%.+]] = VPU.Empty : tensor<1x1x1x1024xui8>
// CHECK-DAG: [[EMPTY_VALS:%.+]] = tensor.empty() : tensor<6x8x32x128xf32>
// CHECK-DAG: [[EMPTY_INDS:%.+]] = tensor.empty() : tensor<6x8x32x128xsi32>
// CHECK:     [[FORALL:%.+]]:2 = scf.forall ([[IV:%.+]]) in (6) shared_outs([[OUT_VALS:%.+]] = [[EMPTY_VALS]], [[OUT_INDS:%.+]] = [[EMPTY_INDS]])
// CHECK:         [[VALS:%.+]], [[INDS:%.+]] = VPU.TopK([[INPUT]], [[K_BUF]]) {axis = 1 : i64, element_type = si32, k_value = 8 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
// CHECK:         scf.forall.in_parallel {
// CHECK:             tensor.parallel_insert_slice [[VALS]] into [[OUT_VALS]][[[IV]], 0, 0, 0] [1, 8, 32, 128] [1, 1, 1, 1]
// CHECK:             tensor.parallel_insert_slice [[INDS]] into [[OUT_INDS]][[[IV]], 0, 0, 0] [1, 8, 32, 128] [1, 1, 1, 1]
// CHECK: [[EXT_VALS:%.+]] = tensor.extract_slice [[FORALL]]#0[0, 0, 0, 0] [1, 8, 32, 128] [1, 1, 1, 1]
// CHECK: [[EXT_INDS:%.+]] = tensor.extract_slice [[FORALL]]#1[0, 0, 0, 0] [1, 8, 32, 128] [1, 1, 1, 1]
// CHECK: return [[EXT_VALS]], [[EXT_INDS]]
}
}

// -----

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// LSTMGates with Clustering strategy — multi-result op (2 outputs)
// Small H dimension (1) forces Clustering instead of SplitOverHeight

// CHECK-LABEL: @LSTMGatesClustering
// CHECK-SAME:       [[GATES:%[^:]+]]: tensor<1x1x1x2048xf16>
// CHECK-SAME:       [[CELL:%[^:]+]]: tensor<1x1x1x512xf16>
func.func @LSTMGatesClustering(%arg0: tensor<1x1x1x2048xf16>, %arg1: tensor<1x1x1x512xf16>) -> (tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>) {
  %h, %c = VPU.LSTMGates(%arg0, %arg1) {
      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
  } : tensor<1x1x1x2048xf16>, tensor<1x1x1x512xf16>
    -> tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>
  return %h, %c : tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>

// CHECK:      [[EMPTY:%.+]] = tensor.empty() : tensor<6x1x1x512xf16>
// CHECK:      [[FORALL:%.+]]:2 = scf.forall ([[IV:%.+]]) in (6) shared_outs([[OUT_H:%.+]] = [[EMPTY]], [[OUT_C:%.+]] = [[EMPTY]])
// CHECK:          [[H:%.+]], [[C:%.+]] = VPU.LSTMGates([[GATES]], [[CELL]])
// CHECK:          scf.forall.in_parallel {
// CHECK:              tensor.parallel_insert_slice [[H]] into [[OUT_H]][[[IV]], 0, 0, 0] [1, 1, 1, 512] [1, 1, 1, 1]
// CHECK:              tensor.parallel_insert_slice [[C]] into [[OUT_C]][[[IV]], 0, 0, 0] [1, 1, 1, 512] [1, 1, 1, 1]
// CHECK: [[EXT_H:%.+]] = tensor.extract_slice [[FORALL]]#0[0, 0, 0, 0] [1, 1, 1, 512] [1, 1, 1, 1]
// CHECK: [[EXT_C:%.+]] = tensor.extract_slice [[FORALL]]#1[0, 0, 0, 0] [1, 1, 1, 512] [1, 1, 1, 1]
// CHECK: return [[EXT_H]], [[EXT_C]]
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// Multiclustering: standalone LSTMGates with SplitOverHeight (not pre-tiled).
// Validates multi-result op is correctly handled by the multiclustering algorithm:
// one parallel_insert_slice per result, with tensor.empty reuse allowed after canonicalization/CSE.

// CHECK-LABEL: @StandaloneLSTMGatesSOH
// CHECK-SAME:       [[GATES:%[^:]+]]: tensor<1x1x128x512xf16>
// CHECK-SAME:       [[CELL:%[^:]+]]: tensor<1x1x128x128xf16>
func.func @StandaloneLSTMGatesSOH(
    %arg0: tensor<1x1x128x512xf16>,
    %arg1: tensor<1x1x128x128xf16>
) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) {
  %h, %c = VPU.LSTMGates(%arg0, %arg1) {
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
  } : tensor<1x1x128x512xf16>, tensor<1x1x128x128xf16>
    -> tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>
  return %h, %c : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>
}

// CHECK-DAG:  [[EMPTY_H:%.+]] = tensor.empty() : tensor<1x1x128x128xf16>
// CHECK:      [[FORALL:%.+]]:2 = scf.forall ([[IV:%.+]]) = (0) to (128) step ({{[0-9]+}}) shared_outs([[OUT_H:%.+]] = [[EMPTY_H]], [[OUT_C:%.+]] = [[EMPTY_H]]) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) {
// CHECK:          [[MIN:%.+]] = affine.min
// CHECK:          [[SLICE_GATES:%.+]] = tensor.extract_slice [[GATES]][0, 0, [[IV]], 0] [1, 1, [[MIN]], 512]
// CHECK:          [[SLICE_CELL:%.+]] = tensor.extract_slice [[CELL]][0, 0, [[IV]], 0] [1, 1, [[MIN]], 128]
// CHECK:          [[H:%.+]], [[C:%.+]] = VPU.LSTMGates([[SLICE_GATES]], [[SLICE_CELL]])
// CHECK:          scf.forall.in_parallel {
// CHECK:              tensor.parallel_insert_slice [[H]] into [[OUT_H]][0, 0, [[IV]], 0] [1, 1, [[MIN]], 128]
// CHECK:              tensor.parallel_insert_slice [[C]] into [[OUT_C]][0, 0, [[IV]], 0] [1, 1, [[MIN]], 128]
// CHECK:          }
// CHECK:      }
// CHECK:      return [[FORALL]]#0, [[FORALL]]#1 : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// Multiclustering: standalone TopK with SplitOverHeight (not pre-tiled).
// TopK has mixed-type multi-result outputs (f32 values, si32 indices).
// Validates that the multi-result tiling code creates correct typed
// tensor.empty, parallel_insert_slice, and forall results.

// CHECK-LABEL: @StandaloneTopKSOH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x64x128x128xf32>
func.func @StandaloneTopKSOH(
    %arg0: tensor<1x64x128x128xf32>
) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>) {
  %k_buf = VPU.Empty : tensor<1x1x1x1024xui8>
  %vals, %inds = VPU.TopK(%arg0, %k_buf) {
      axis = 1 : i64,
      element_type = si32,
      k_value = 8 : i64,
      mode = #IE.topk_mode<MAX>,
      sort = #IE.topk_sort_type<SORT_INDICES>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
  } : tensor<1x64x128x128xf32>, tensor<1x1x1x1024xui8>
    -> tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>
  return %vals, %inds : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>
}

// CHECK-DAG:  [[K_BUF:%.+]] = VPU.Empty : tensor<1x1x1x1024xui8>
// CHECK-DAG:  [[EMPTY_VALS:%.+]] = tensor.empty() : tensor<1x8x128x128xf32>
// CHECK-DAG:  [[EMPTY_INDS:%.+]] = tensor.empty() : tensor<1x8x128x128xsi32>
// CHECK:      [[FORALL:%.+]]:2 = scf.forall ([[IV:%.+]]) = (0) to (128) step ({{[0-9]+}}) shared_outs([[OUT_VALS:%.+]] = [[EMPTY_VALS]], [[OUT_INDS:%.+]] = [[EMPTY_INDS]]) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>) {
// CHECK:          [[MIN:%.+]] = affine.min
// CHECK:          [[SLICE_IN:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IV]], 0] [1, 64, [[MIN]], 128]
// CHECK:          [[V:%.+]], [[S:%.+]] = VPU.TopK([[SLICE_IN]], [[K_BUF]])
// CHECK:          scf.forall.in_parallel {
// CHECK:              tensor.parallel_insert_slice [[V]] into [[OUT_VALS]][0, 0, [[IV]], 0] [1, 8, [[MIN]], 128]
// CHECK:              tensor.parallel_insert_slice [[S]] into [[OUT_INDS]][0, 0, [[IV]], 0] [1, 8, [[MIN]], 128]
// CHECK:          }
// CHECK:      }
// CHECK:      return [[FORALL]]#0, [[FORALL]]#1 : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>
}
