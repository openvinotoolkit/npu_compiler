//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --scf-multiclustering --canonicalize --cse %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 30)>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0) -> (0, d0 - 30)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0) -> (d0 ceildiv 6)>
// CHECK-DAG: #[[$MAP6:.+]] = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 6)>
// CHECK-DAG: #[[$MAP7:.+]] = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 2)>
// CHECK-DAG: #[[$MAP8:.+]] = affine_map<(d0, d1) -> (0, d0 - d1)>
// CHECK-DAG: #[[$MAP9:.+]] = affine_map<(d0, d1, d2) -> (0, d0 - d1 + d2 - 31)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @SOHConvTileOverH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SOHConvTileOverH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %cst = arith.constant 0.000000e+00 : f16
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %out = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    %tiling_loop = scf.for %out_offset_h = %c0 to %c64 step %c32 iter_args(%arg2 = %out) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
      %in_offset_h = affine.max #map(%out_offset_h)
      %temp = affine.max #map1(%out_offset_h)
      %pad_top = affine.min #map2()[%temp]
      %temp0 = affine.max #map3(%in_offset_h)
      %pad_bottom = affine.min #map2()[%temp0]

      %extracted_slice = tensor.extract_slice %arg0[0, 0, %in_offset_h, 0] [1, 32, 33, 64] [1, 1, 1, 1]
          : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

      %padded = tensor.pad %extracted_slice low[0, 0, %pad_top, 1] high[0, 0, %pad_bottom, 1] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x33x64xf16, {order = #NHWC}> to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>

      %conv = VPU.NCE.Convolution(%padded, %weights) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,  strides = [1, 1]
      } : tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<256x32x3x3xf16, {order = #NHWC}>
        -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

      %cast = tensor.cast %conv : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
                                to tensor<1x256x32x64xf16, {order = #NHWC}>

      %inserted_slice = tensor.insert_slice %cast into %arg2[0, 0, %out_offset_h, 0] [1, 256, 32, 64] [1, 1, 1, 1]
        : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
    }
    return %tiling_loop : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG: [[CST_31:%.+]] = arith.constant 31 : index
    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 64 : index
    // CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 32 : index
    // CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK-SAME:     [[TILE_LOOP_ITER:%[^:]+]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    // CHECK-SAME:     iter_args([[LOOP_OUT:%[^:]+]]  = [[LOOP_OUTPUT]]) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {

    // CHECK:         [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP]]([[TILE_LOOP_ITER]])
    // CHECK:         [[DIFF1:%.+]] = affine.max #[[$MAP1]]([[TILE_LOOP_ITER]])
    // CHECK:         [[PAD_LOW:%.+]] = affine.min #[[$MAP2]]()[[[DIFF1]]]
    // CHECK:         [[DIFF2:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET]])
    // CHECK:         [[PAD_HIGH:%.+]] = affine.min #[[$MAP2]]()[[[DIFF2]]]

    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [1, 32, 33, 64] [1, 1, 1, 1]
    // CHECK-SAME:         tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:         [[TOTAL_PAD:%.+]] = affine.apply #[[$MAP4]]([[PAD_LOW]], [[PAD_HIGH]])
    // CHECK:         [[OUT_DIM_H_SZ:%.+]] = arith.addi [[TOTAL_PAD]], [[CST_31]] : index
    // CHECK:         [[MC_OUT:%.+]] = tensor.empty([[OUT_DIM_H_SZ]])
    // CHECK-SAME:        : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:         [[MC_STEP:%.+]] = affine.apply #[[$MAP5]]([[OUT_DIM_H_SZ]])

    // CHECK:         [[MC_LOOP:%.+]] = scf.forall ([[MC_LOOP_ITER:%.+]]) = (0) to ([[OUT_DIM_H_SZ]]) step ([[MC_STEP]])
    // CHECK-SAME:        shared_outs([[MC_LOOP_OUT:%.+]] = [[MC_OUT]])
    // CHECK-SAME:          -> (tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:             [[MC_TILE_SZ:%.+]] = affine.min #[[$MAP6]]([[MC_LOOP_ITER]], [[OUT_DIM_H_SZ]])[[[OUT_DIM_H_SZ]]]
    // CHECK:             [[MC_IN_TILE_OFF:%.+]] = affine.max #[[$MAP8]]([[MC_LOOP_ITER]], [[PAD_LOW]])

    // CHECK:             [[MC_PAD_LOW:%.+]] = affine.max #[[$MAP8]]([[PAD_LOW]], [[MC_IN_TILE_OFF]])
    // CHECK:             [[MC_PAD_HIGH:%.+]] = affine.max #[[$MAP9]]([[MC_LOOP_ITER]], [[PAD_LOW]], [[MC_TILE_SZ]])

    // CHECK:             [[MC_IN_TILE_SZ:%.+]] = affine.apply #[[$MAP7]]([[MC_PAD_LOW]], [[MC_PAD_HIGH]], [[MC_TILE_SZ]])

    // CHECK:             [[IN_TILE:%.+]] = tensor.extract_slice [[SLICE]][0, 0, [[MC_IN_TILE_OFF]], 0] [1, 32, [[MC_IN_TILE_SZ]], 64] [1, 1, 1, 1]
    // CHECK-SAME:             tensor<1x32x33x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {order = #NHWC}>

    // CHECK:             [[PAD:%.+]] = tensor.pad [[IN_TILE]] low[0, 0, [[MC_PAD_LOW]], 1] high[0, 0, [[MC_PAD_HIGH]], 1] {
    // CHECK:                tensor.yield [[PAD_VALUE]] : f16
    // CHECK:                : tensor<1x32x?x64xf16, {order = #NHWC}>
    // CHECK-SAME:           to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 13, 66]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:             [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[WEIGHTS]])
    // CHECK-SAME:            {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:            : tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 13, 66]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME:              tensor<256x32x3x3xf16, {order = #NHWC}>
    // CHECK-SAME:            -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 11, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:             scf.forall.in_parallel {
    // CHECK:                 tensor.parallel_insert_slice [[CONV]] into [[MC_LOOP_OUT]][0, 0, [[MC_LOOP_ITER]], 0] [1, 256, [[MC_TILE_SZ]], 64] [1, 1, 1, 1]
    // CHECK-SAME:              : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 11, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:              into tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:         }

    // CHECK:         [[CAST:%.+]] = tensor.cast [[MC_LOOP]]
    // CHECK-SAME:       : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:       to tensor<1x256x32x64xf16, {order = #NHWC}>

    // CHECK:         [[INSERT:%.+]] = tensor.insert_slice [[CAST]] into [[LOOP_OUT]][0, 0, [[TILE_LOOP_ITER]], 0] [1, 256, 32, 64] [1, 1, 1, 1]
    // CHECK-SAME:       : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK: scf.yield [[INSERT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
    // CHECK: return [[LOOP]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0) -> (0, d0 * 2 - 1)>
#map1 = affine_map<(d0) -> (d0 * -2 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 47)>
#map4 = affine_map<(d0, d1) -> (-d0 - d1 + 17)>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (0, d0 * 2 - 1)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * -2 + 1, 0)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0) -> (0, d0 - 47)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1) -> (-d0 - d1 + 17)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG: #[[$MAP6:.+]] = affine_map<(d0) -> (d0 ceildiv 6)>
// CHECK-DAG: #[[$MAP7:.+]] = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 6)>
// CHECK-DAG: #[[$MAP8:.+]] = affine_map<(d0) -> (0, d0 * 2)>
// CHECK-DAG: #[[$MAP9:.+]] = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 * 2 + 1)>
// CHECK-DAG: #[[$MAP10:.+]] = affine_map<(d0, d1) -> (0, d0 - d1)>
// CHECK-DAG: #[[$MAP11:.+]] = affine_map<(d0, d1, d2) -> (0, d0 + d1 * 2 + d2 - 16)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @SOHConvWithStrideTileOverH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SOHConvWithStrideTileOverH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x32x32xf16, {order = #NHWC}> {
    %cst = arith.constant 0.000000e+00 : f16
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %out = tensor.empty() : tensor<1x256x32x32xf16, {order = #NHWC}>
    %tiling_loop = scf.for %out_offset_h = %c0 to %c32 step %c8 iter_args(%arg2 = %out) -> (tensor<1x256x32x32xf16, {order = #NHWC}>) {
      %in_offset_h = affine.max #map(%out_offset_h)
      %temp = affine.max #map1(%out_offset_h)
      %pad_top = affine.min #map2()[%temp]
      %temp0 = affine.max #map3(%in_offset_h)
      %pad_bottom = affine.min #map2()[%temp0]
      %in_size_h = affine.apply #map4(%pad_top, %pad_bottom)

      %extracted_slice = tensor.extract_slice %arg0[0, 0, %in_offset_h, 0] [1, 32, %in_size_h, 64] [1, 1, 1, 1]
          : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

      %padded = tensor.pad %extracted_slice low[0, 0, %pad_top, 1] high[0, 0, %pad_bottom, 1] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
        to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>

      %conv = VPU.NCE.Convolution(%padded, %weights) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,  strides = [2, 2], tiling_index = 0 : i64
      } : tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<256x32x3x3xf16, {order = #NHWC}>
        -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 32, 32]> : tensor<4xsi64>, order = #NHWC}>

      %cast = tensor.cast %conv : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 32, 32]> : tensor<4xsi64>, order = #NHWC}>
                                to tensor<1x256x8x32xf16, {order = #NHWC}>

      %inserted_slice = tensor.insert_slice %cast into %arg2[0, 0, %out_offset_h, 0] [1, 256, 8, 32] [1, 1, 1, 1]
        : tensor<1x256x8x32xf16, {order = #NHWC}> into tensor<1x256x32x32xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x32x32xf16, {order = #NHWC}>
    }
    return %tiling_loop : tensor<1x256x32x32xf16, {order = #NHWC}>

    // CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK-DAG: [[CST_MIN_1:%.+]] = arith.constant -1 : index
    // CHECK-DAG: [[CST_2:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[CST_8:%.+]] = arith.constant 8 : index
    // CHECK-DAG: [[CST_32:%.+]] = arith.constant 32 : index
    // CHECK-DAG: [[CST_0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x256x32x32xf16, {order = #NHWC}>
    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK-SAME:     [[TILE_LOOP_ITER:%[^:]+]] = [[CST_0]] to [[CST_32]] step [[CST_8]]
    // CHECK-SAME:     iter_args([[LOOP_OUT:%[^:]+]]  = [[LOOP_OUTPUT]]) -> (tensor<1x256x32x32xf16, {order = #NHWC}>) {

    // CHECK:         [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP]]([[TILE_LOOP_ITER]])
    // CHECK:         [[DIFF1:%.+]] = affine.max #[[$MAP1]]([[TILE_LOOP_ITER]])
    // CHECK:         [[PAD_LOW:%.+]] = affine.min #[[$MAP2]]()[[[DIFF1]]]
    // CHECK:         [[DIFF2:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET]])
    // CHECK:         [[PAD_HIGH:%.+]] = affine.min #[[$MAP2]]()[[[DIFF2]]]

    // CHECK:         [[IN_SIZE:%.+]] = affine.apply #[[$MAP4]]([[PAD_LOW]], [[PAD_HIGH]])

    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [1, 32, [[IN_SIZE]], 64] [1, 1, 1, 1]
    // CHECK-SAME:         tensor<1x32x64x64xf16, {order = #NHWC}>
    // CHECK-SAME:         to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:         [[TOTAL_PAD:%.+]] = affine.apply #[[$MAP5]]([[PAD_LOW]], [[PAD_HIGH]])
    // CHECK:         [[PADDED_IN:%.+]] = arith.addi [[IN_SIZE]], [[TOTAL_PAD]] : index
    // CHECK:         [[ADJUST_PADDED_IN:%.+]] = arith.addi [[PADDED_IN]], [[CST_MIN_1]] : index
    // CHECK:         [[OUT_DIM_H_SZ:%.+]] = arith.divsi [[ADJUST_PADDED_IN]], [[CST_2]] : index
    // CHECK:         [[MC_OUT:%.+]] = tensor.empty([[OUT_DIM_H_SZ]])
    // CHECK-SAME:        : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 32, 32]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:         [[MC_STEP:%.+]] = affine.apply #[[$MAP6]]([[OUT_DIM_H_SZ]])

    // CHECK:         [[MC_LOOP:%.+]] = scf.forall ([[MC_LOOP_ITER:%.+]]) = (0) to ([[OUT_DIM_H_SZ]]) step ([[MC_STEP]])
    // CHECK-SAME:        shared_outs([[MC_LOOP_OUT:%.+]] = [[MC_OUT]])
    // CHECK-SAME:          -> (tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 32, 32]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:             [[MC_TILE_SZ:%.+]] = affine.min #[[$MAP7]]([[MC_LOOP_ITER]], [[OUT_DIM_H_SZ]])[[[OUT_DIM_H_SZ]]]
    // CHECK:             [[STRIDED_TEMP:%.+]] = affine.max #[[$MAP8]]([[MC_LOOP_ITER]])
    // CHECK:             [[MC_IN_TILE_OFF:%.+]] = affine.max #[[$MAP10]]([[STRIDED_TEMP]], [[PAD_LOW]])

    // CHECK:             [[MC_PAD_LOW:%.+]] = affine.max #[[$MAP10]]([[PAD_LOW]], [[MC_IN_TILE_OFF]])
    // CHECK:             [[MC_PAD_HIGH:%.+]] = affine.max #[[$MAP11]]([[STRIDED_TEMP]], [[MC_TILE_SZ]], [[PAD_HIGH]])

    // CHECK:             [[MC_IN_TILE_SZ:%.+]] = affine.apply #[[$MAP9]]([[MC_PAD_LOW]], [[MC_PAD_HIGH]], [[MC_TILE_SZ]])

    // CHECK:             [[IN_TILE:%.+]] = tensor.extract_slice [[SLICE]][0, 0, [[MC_IN_TILE_OFF]], 0] [1, 32, [[MC_IN_TILE_SZ]], 64] [1, 1, 1, 1]
    // CHECK-SAME:             tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:             to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:             [[PAD:%.+]] = tensor.pad [[IN_TILE]] low[0, 0, [[MC_PAD_LOW]], 1] high[0, 0, [[MC_PAD_HIGH]], 1] {
    // CHECK:                tensor.yield [[PAD_VALUE]] : f16
    // CHECK:                : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:           to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 13, 65]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:             [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[WEIGHTS]])
    // CHECK-SAME:            {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:            : tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 13, 65]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME:              tensor<256x32x3x3xf16, {order = #NHWC}>
    // CHECK-SAME:            -> tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 6, 32]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:             scf.forall.in_parallel {
    // CHECK:                 tensor.parallel_insert_slice [[CONV]] into [[MC_LOOP_OUT]][0, 0, [[MC_LOOP_ITER]], 0] [1, 256, [[MC_TILE_SZ]], 32] [1, 1, 1, 1]
    // CHECK-SAME:              : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 6, 32]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:              into tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 32, 32]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:         }

    // CHECK:         [[CAST:%.+]] = tensor.cast [[MC_LOOP]]
    // CHECK-SAME:       : tensor<1x256x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 32, 32]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:       to tensor<1x256x8x32xf16, {order = #NHWC}>

    // CHECK:         [[INSERT:%.+]] = tensor.insert_slice [[CAST]] into [[LOOP_OUT]][0, 0, [[TILE_LOOP_ITER]], 0] [1, 256, 8, 32] [1, 1, 1, 1]
    // CHECK-SAME:       : tensor<1x256x8x32xf16, {order = #NHWC}> into tensor<1x256x32x32xf16, {order = #NHWC}>

    // CHECK: scf.yield [[INSERT]] : tensor<1x256x32x32xf16, {order = #NHWC}>
    // CHECK: return [[LOOP]] : tensor<1x256x32x32xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 30)>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0) -> (0, d0 - 30)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0) -> (-d0 + 256, 96)>

module {
config.Resources 3 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @SOKConvTileOverH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SOKConvTileOverH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %cst = arith.constant 0.000000e+00 : f16
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %out = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    %tiling_loop = scf.for %out_offset_h = %c0 to %c64 step %c32 iter_args(%arg2 = %out) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
      %in_offset_h = affine.max #map(%out_offset_h)
      %temp = affine.max #map1(%out_offset_h)
      %pad_top = affine.min #map2()[%temp]
      %temp0 = affine.max #map3(%in_offset_h)
      %pad_bottom = affine.min #map2()[%temp0]

      %extracted_slice = tensor.extract_slice %arg0[0, 0, %in_offset_h, 0] [1, 32, 33, 64] [1, 1, 1, 1]
          : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

      %padded = tensor.pad %extracted_slice low[0, 0, %pad_top, 1] high[0, 0, %pad_bottom, 1] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x33x64xf16, {order = #NHWC}> to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>

      %conv = VPU.NCE.Convolution(%padded, %weights) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,  strides = [1, 1]
      } : tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<256x32x3x3xf16, {order = #NHWC}>
        -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

      %cast = tensor.cast %conv : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
                                to tensor<1x256x32x64xf16, {order = #NHWC}>

      %inserted_slice = tensor.insert_slice %cast into %arg2[0, 0, %out_offset_h, 0] [1, 256, 32, 64] [1, 1, 1, 1]
        : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
    }
    return %tiling_loop : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 64 : index
    // CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 32 : index
    // CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK-DAG: [[CST_31:%.+]] = arith.constant 31 : index

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK-SAME:     [[TILE_LOOP_ITER:%[^:]+]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    // CHECK-SAME:     iter_args([[LOOP_OUT:%[^:]+]]  = [[LOOP_OUTPUT]]) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {

    // CHECK:         [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP]]([[TILE_LOOP_ITER]])
    // CHECK:         [[DIFF1:%.+]] = affine.max #[[$MAP1]]([[TILE_LOOP_ITER]])
    // CHECK:         [[PAD_LOW:%.+]] = affine.min #[[$MAP2]]()[[[DIFF1]]]
    // CHECK:         [[DIFF2:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET]])
    // CHECK:         [[PAD_HIGH:%.+]] = affine.min #[[$MAP2]]()[[[DIFF2]]]

    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [1, 32, 33, 64] [1, 1, 1, 1]
    // CHECK-SAME:         tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:         [[TOTAL_PAD:%.+]] = affine.apply #[[$MAP4]]([[PAD_LOW]], [[PAD_HIGH]])
    // CHECK:         [[OUT_DIM_H_SZ:%.+]] = arith.addi [[TOTAL_PAD]], [[CST_31]] : index

    // CHECK:         [[MC_OUT:%.+]] = tensor.empty([[OUT_DIM_H_SZ]])
    // CHECK-SAME:        : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:         [[MC_LOOP:%.+]] = scf.forall ([[MC_LOOP_ITER:%.+]]) = (0) to (256) step (96)
    // CHECK-SAME:        shared_outs([[MC_LOOP_OUT:%.+]] = [[MC_OUT]])
    // CHECK-SAME:          -> (tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:             [[MC_TILE_SZ:%.+]] = affine.min #[[$MAP5]]([[MC_LOOP_ITER]])

    // CHECK:             [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    // CHECK:                tensor.yield [[PAD_VALUE]] : f16
    // CHECK:                : tensor<1x32x33x64xf16, {order = #NHWC}>
    // CHECK-SAME:           to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:             [[WEIGHTS_TILE:%.+]] = tensor.extract_slice [[WEIGHTS]][[[MC_LOOP_ITER]], 0, 0, 0] [[[MC_TILE_SZ]], 32, 3, 3] [1, 1, 1, 1]
    // CHECK-SAME:             : tensor<256x32x3x3xf16, {order = #NHWC}>
    // CHECK-SAME:             to tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:             [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[WEIGHTS_TILE]])
    // CHECK-SAME:            {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:            : tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME:              tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:            -> tensor<1x?x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:             [[OUT_CAST:%.+]] = tensor.cast [[CONV]]
    // CHECK-SAME:            : tensor<1x?x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:            to tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:             scf.forall.in_parallel {
    // CHECK:                 tensor.parallel_insert_slice [[OUT_CAST]] into [[MC_LOOP_OUT]][0, [[MC_LOOP_ITER]], 0, 0] [1, [[MC_TILE_SZ]], 64, 64] [1, 1, 1, 1]
    // CHECK-SAME:              : tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:              into tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:         }

    // CHECK:         [[CAST:%.+]] = tensor.cast [[MC_LOOP]]
    // CHECK-SAME:       : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:       to tensor<1x256x32x64xf16, {order = #NHWC}>

    // CHECK:         [[INSERT:%.+]] = tensor.insert_slice [[CAST]] into [[LOOP_OUT]][0, 0, [[TILE_LOOP_ITER]], 0] [1, 256, 32, 64] [1, 1, 1, 1]
    // CHECK-SAME:       : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK: scf.yield [[INSERT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
    // CHECK: return [[LOOP]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$TILE_SIZE_EXPR:.+]] = affine_map<(d0) -> (-d0 + 256, 48)>

// CHECK-LABEL:   @NCEConvSOK
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @NCEConvSOK(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = VPU.NCE.Convolution(%arg0, %cst_0) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]
  } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x256x64x64xf16, {order = #NHWC}>
  return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}>
// CHECK: [[OUTPUT:%.+]] = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>

// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (256) step (48) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:      -> (tensor<1x256x64x64xf16, {order = #NHWC}>)
// CHECK:       [[TILE_SZ:%.+]] = affine.min #[[$TILE_SIZE_EXPR]]([[LOOP_ITER]])

// CHECK:       [[WEIGHTS_SLICE:%.+]] = tensor.extract_slice [[WEIGHTS]][[[LOOP_ITER]], 0, 0, 0] [[[TILE_SZ]], 32, 3, 3] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<256x32x3x3xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE]]) rawFilterShape [[[TILE_SZ]], 32, 3, 3] {
// CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
// CHECK-SAME:      ppe = #VPU.PPEStub<>, 
// CHECK-SAME:      strides = [1, 1]}
// CHECK-SAME:    : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:    -> tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[CONV]] into [[LOOP_OUT]][0, [[LOOP_ITER]], 0, 0] [1, [[TILE_SZ]], 64, 64] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:          into tensor<1x256x64x64xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 3 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$TILE_SIZE_EXPR:.+]] = affine_map<(d0) -> (-d0 + 128, 48)>

// CHECK-LABEL:   @NCEDWConvSOK
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x128x64x64xf16, {order = #NHWC}>
func.func @NCEDWConvSOK(%arg0: tensor<1x128x64x64xf16, {order = #NHWC}>) -> tensor<1x128x64x64xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = VPU.NCE.DepthConvolution(%arg0, %weights) rawFilterShape [128, 1, 3, 3] {
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
       strides = [1, 1]
  } : tensor<1x128x64x64xf16, {order = #NHWC}>, tensor<128x16x1x1xf16, {order = #NHWC}>
      -> tensor<1x128x64x64xf16, {order = #NHWC}>
  return %0 : tensor<1x128x64x64xf16, {order = #NHWC}>
}

// CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
// CHECK: [[OUTPUT:%.+]] = tensor.empty() : tensor<1x128x64x64xf16, {order = #NHWC}>

// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (128) step (48) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:      -> (tensor<1x128x64x64xf16, {order = #NHWC}>)
// CHECK:       [[TILE_SZ:%.+]] = affine.min #[[$TILE_SIZE_EXPR]]([[LOOP_ITER]])

// CHECK:       [[IN_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, [[LOOP_ITER]], 0, 0] [1, [[TILE_SZ]], 64, 64] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<1x128x64x64xf16, {order = #NHWC}>
// CHECK-SAME:      tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[WEIGHTS_SLICE:%.+]] = tensor.extract_slice [[WEIGHTS]][[[LOOP_ITER]], 0, 0, 0] [[[TILE_SZ]], 16, 1, 1] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<128x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<?x16x1x1xf16, {bounds = #const.OpaqueI64Elements<[128, 16, 1, 1]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[IN_SLICE]], [[WEIGHTS_SLICE]]) rawFilterShape [[[TILE_SZ]], 1, 3, 3] {
// CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
// CHECK-SAME:      ppe = #VPU.PPEStub<>, 
// CHECK-SAME:      strides = [1, 1]}
// CHECK-SAME:    -> tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[DWCONV]] into [[LOOP_OUT]][0, [[LOOP_ITER]], 0, 0] [1, [[TILE_SZ]], 64, 64] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:          into tensor<1x128x64x64xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

module {
config.Resources 5 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @NCEPoolSOK
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x64x12x12xf16, {order = #NHWC}>
func.func @NCEPoolSOK(%arg0: tensor<1x64x12x12xf16, {order = #NHWC}>) -> tensor<1x64x12x12xf16, {order = #NHCW}> {
  %0 = VPU.NCE.AveragePool(%arg0) {
      kernel_size = [2, 2],
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
      pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>, strides = [1, 1]
  } : tensor<1x64x12x12xf16, {order = #NHWC}> -> tensor<1x64x12x12xf16, {order = #NHCW}>

  return %0 : tensor<1x64x12x12xf16, {order = #NHCW}>
}

// CHECK: [[OUTPUT:%.+]] = tensor.empty() : tensor<1x64x12x12xf16, {order = #NHCW}>

// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (64) step (16) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:      -> (tensor<1x64x12x12xf16, {order = #NHCW}>)

// CHECK:       [[IN_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, [[LOOP_ITER]], 0, 0] [1, 16, 12, 12] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<1x64x12x12xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x16x12x12xf16, {order = #NHWC}>

// CHECK:       [[POOL:%.+]] = VPU.NCE.AveragePool([[IN_SLICE]]) {
// CHECK-SAME:      kernel_size = [2, 2], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
// CHECK-SAME:      ppe = #VPU.PPEStub<>, strides = [1, 1]}
// CHECK-SAME:    -> tensor<1x16x12x12xf16, {order = #NHCW}>

// CHECK:       scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[POOL]] into [[LOOP_OUT]][0, [[LOOP_ITER]], 0, 0] [1, 16, 12, 12] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x16x12x12xf16, {order = #NHCW}>
// CHECK-SAME:          into tensor<1x64x12x12xf16, {order = #NHCW}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

module {
config.Resources 4 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$TILE_SIZE_EXPR:.+]] = affine_map<(d0) -> (-d0 + 112, 32)>

// CHECK-LABEL:   @NCEEltwiseSOK
// CHECK-SAME:       [[INPUT0:%[^:]+]]: tensor<1x112x12x12xf16, {order = #NHWC}>
// CHECK-SAME:       [[INPUT1:%[^:]+]]: tensor<1x112x12x12xf16, {order = #NHWC}>
func.func @NCEEltwiseSOK(%arg0: tensor<1x112x12x12xf16, {order = #NHWC}>, %arg1: tensor<1x112x12x12xf16, {order = #NHWC}>)
    -> tensor<1x112x12x12xf16, {order = #NHCW}> {
  %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
      is_inplace = true,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
      op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>
  } -> tensor<1x112x12x12xf16, {order = #NHCW}>

  return %0 : tensor<1x112x12x12xf16, {order = #NHCW}>
}

// CHECK: [[OUTPUT:%.+]] = tensor.empty() : tensor<1x112x12x12xf16, {order = #NHCW}>

// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (112) step (32) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:      -> (tensor<1x112x12x12xf16, {order = #NHCW}>)
// CHECK:       [[TILE_SZ:%.+]] = affine.min #[[$TILE_SIZE_EXPR]]([[LOOP_ITER]])
// CHECK:       [[ALIGNED_TILE_SZ:%.+]] = affine.apply #{{.+}}([[TILE_SZ]])

// CHECK:       [[IN_SLICE0:%.+]] = tensor.extract_slice [[INPUT0]][0, [[LOOP_ITER]], 0, 0] [1, [[ALIGNED_TILE_SZ]], 12, 12] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<1x112x12x12xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x?x12x12xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 12, 12]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[IN_SLICE1:%.+]] = tensor.extract_slice [[INPUT1]][0, [[LOOP_ITER]], 0, 0] [1, [[ALIGNED_TILE_SZ]], 12, 12] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<1x112x12x12xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x?x12x12xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 12, 12]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[IN_SLICE0]], [[IN_SLICE1]])
// CHECK-SAME:    -> tensor<1x?x12x12xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 12, 12]> : tensor<4xsi64>, order = #NHCW}>

// CHECK:       scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[ELTWISE]] into [[LOOP_OUT]][0, [[LOOP_ITER]], 0, 0] [1, [[TILE_SZ]], 12, 12] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x?x12x12xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 12, 12]> : tensor<4xsi64>, order = #NHCW}>
// CHECK-SAME:          into tensor<1x112x12x12xf16, {order = #NHCW}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
config.Resources 3 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$TILE_SIZE_EXPR:.+]] = affine_map<(d0) -> (-d0 + 47, 16)>

// CHECK-LABEL:   @NCEPermuteSOKWithChannelAlignment
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<1x47x16x16xf16>
func.func @NCEPermuteSOKWithChannelAlignment(%arg0: tensor<1x47x16x16xf16>) -> tensor<1x48x16x16xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = f16, dstOrder = #NHWC, expandedChannels = 48 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x48x16x16xf16, {order = #NHWC}>
  return %0 : tensor<1x48x16x16xf16, {order = #NHWC}>
}

// CHECK: [[OUTPUT:%.+]] = tensor.empty() : tensor<1x48x16x16xf16, {order = #NHWC}>

// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (48) step (16) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:      -> (tensor<1x48x16x16xf16, {order = #NHWC}>)
// CHECK:       [[TILE_SZ:%.+]] = affine.min #[[$TILE_SIZE_EXPR]]([[LOOP_ITER]])

// CHECK:       [[IN_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, [[LOOP_ITER]], 0, 0] [1, [[TILE_SZ]], 16, 16] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<1x47x16x16xf16> to tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 47, 16, 16]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:       [[PERMUTE:%.+]] = VPU.NCE.Permute([[IN_SLICE]])
// CHECK-SAME:       expandedChannels = 16 : i64
// CHECK-SAME:    -> tensor<1x16x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[PERMUTE]] into [[LOOP_OUT]][0, [[LOOP_ITER]], 0, 0] [1, 16, 16, 16] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x16x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:          into tensor<1x48x16x16xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
config.Resources 3 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @NCEPermuteSOKNoChannelAlignment
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<1x60x16x16xf16>
func.func @NCEPermuteSOKNoChannelAlignment(%arg0: tensor<1x60x16x16xf16>) -> tensor<1x60x16x16xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = f16, dstOrder = #NHWC, expandedChannels = 60 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x60x16x16xf16, {order = #NHWC}>
  return %0 : tensor<1x60x16x16xf16, {order = #NHWC}>
}

// CHECK: [[OUTPUT:%.+]] = tensor.empty() : tensor<1x60x16x16xf16, {order = #NHWC}>

// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (60) step (20) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:      -> (tensor<1x60x16x16xf16, {order = #NHWC}>)

// CHECK:       [[IN_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, [[LOOP_ITER]], 0, 0] [1, 20, 16, 16] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<1x60x16x16xf16> to tensor<1x20x16x16xf16>

// CHECK:       [[PERMUTE:%.+]] = VPU.NCE.Permute([[IN_SLICE]])
// CHECK-SAME:       expandedChannels = 20 : i64
// CHECK-SAME:    -> tensor<1x20x16x16xf16, {order = #NHWC}>

// CHECK:       scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[PERMUTE]] into [[LOOP_OUT]][0, [[LOOP_ITER]], 0, 0] [1, 20, 16, 16] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x20x16x16xf16, {order = #NHWC}>
// CHECK-SAME:          into tensor<1x60x16x16xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 5 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (-d0 + 31, 7)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (0, d0 * 2 - 1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * -2 + 1, 0)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1) -> (0, d0 * 2 + d1 - 61)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 - d1 - d2 + 3)>

// CHECK-LABEL:   @NCEConvSOH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @NCEConvSOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x31x31xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<256x32x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x5x5xf16>, [#const.Reorder<#NHWC>]
  %0 = VPU.NCE.Convolution(%arg0, %cst_0) rawFilterShape [256, 32, 5, 5] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [2, 2]
  } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<256x32x5x5xf16, {order = #NHWC}>
      -> tensor<1x256x31x31xf16, {order = #NHWC}>
  return %0 : tensor<1x256x31x31xf16, {order = #NHWC}>

// CHECK-DAG: [[ZERO_CST:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x5x5xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT:%.+]] = tensor.empty() : tensor<1x256x31x31xf16, {order = #NHWC}>
// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (31) step (7) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:      -> (tensor<1x256x31x31xf16, {order = #NHWC}>)

// CHECK:       [[OUT_TILE_SZ:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])
// CHECK:       [[IN_TILE_OFFSET:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
// CHECK:       [[TEMP0:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER]])
// CHECK:       [[PAD_TOP:%.+]] = affine.min #[[$MAP3]]()[[[TEMP0]]]
// CHECK:       [[TEMP1:%.+]] = affine.max #[[$MAP4]]([[OUT_TILE_SZ]], [[IN_TILE_OFFSET]])
// CHECK:       [[PAD_BOTTOM:%.+]] = affine.min #[[$MAP3]]()[[[TEMP1]]]
// CHECK:       [[IN_TILE_SZ:%.+]] = affine.apply #[[$MAP5]]([[OUT_TILE_SZ]], [[PAD_TOP]], [[PAD_BOTTOM]])

// CHECK:       [[INPUT_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IN_TILE_OFFSET]], 0] [1, 32, [[IN_TILE_SZ]], 64] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:       [[PADDED_INPUT:%.+]] = tensor.pad [[INPUT_SLICE]] low[0, 0, [[PAD_TOP]], 1] high[0, 0, [[PAD_BOTTOM]], 1] {
// CHECK:       ^bb0({{[^:]+}}: index, {{[^:]+}}: index, {{[^:]+}}: index, {{[^:]+}}: index):
// CHECK:         tensor.yield [[ZERO_CST]] : f16
// CHECK:       } : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:    to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[PADDED_INPUT]], [[WEIGHTS]])
// CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
// CHECK-SAME:    : tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x5x5xf16, {order = #NHWC}>
// CHECK-SAME:    -> tensor<1x256x?x31xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 31, 31]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[CONV]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 256, [[OUT_TILE_SZ]], 31] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x256x?x31xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 31, 31]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:          into tensor<1x256x31x31xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 3 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$OUTPUT_SZ_EXPR:.+]] = affine_map<(d0) -> (-d0 + 32, 11)>
// CHECK: #[[$INPUT_OFFSET_EXPR:.+]] = affine_map<(d0) -> (0, d0 * 2 - 1)>
// CHECK: #[[$PAD_TEMP_EXPR:.+]] = affine_map<(d0) -> (d0 * -2 + 1, 0)>
// CHECK: #[[$PAD_LOW_EXPR:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$INPUT_SZ_EXPR:.+]] = affine_map<(d0, d1) -> (d0 * 2 - d1)>

// CHECK-LABEL:   @NCEDWConvSOH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x128x64x64xf16, {order = #NHWC}>
func.func @NCEDWConvSOH(%arg0: tensor<1x128x64x64xf16, {order = #NHWC}>) -> tensor<1x128x32x32xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = VPU.NCE.DepthConvolution(%arg0, %weights) rawFilterShape [128, 1, 2, 2] {
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
       strides = [2, 2]
  } : tensor<1x128x64x64xf16, {order = #NHWC}>, tensor<128x16x1x1xf16, {order = #NHWC}>
      -> tensor<1x128x32x32xf16, {order = #NHWC}>
  return %0 : tensor<1x128x32x32xf16, {order = #NHWC}>
}

// CHECK-DAG: [[ZERO_CST:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
// CHECK-DAG: [[OUTPUT:%.+]] = tensor.empty() : tensor<1x128x32x32xf16, {order = #NHWC}>

// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (32) step (11) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:      -> (tensor<1x128x32x32xf16, {order = #NHWC}>)

// CHECK:       [[OUT_SIZE:%.+]] = affine.min #[[$OUTPUT_SZ_EXPR]]([[LOOP_ITER]])
// CHECK:       [[IN_OFFSET:%.+]] = affine.max #[[$INPUT_OFFSET_EXPR]]([[LOOP_ITER]])
// CHECK:       [[PAD_TEMP:%.+]] = affine.max #[[$PAD_TEMP_EXPR]]([[LOOP_ITER]])
// CHECK:       [[PAD_LOW:%.+]] = affine.min #[[$PAD_LOW_EXPR]]()[[[PAD_TEMP]]]
// CHECK:       [[IN_SIZE:%.+]] = affine.apply #[[$INPUT_SZ_EXPR]]([[OUT_SIZE]], [[PAD_LOW]])

// CHECK:       [[IN_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IN_OFFSET]], 0] [1, 128, [[IN_SIZE]], 64] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<1x128x64x64xf16, {order = #NHWC}>
// CHECK-SAME:      tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[PAD:%.+]] = tensor.pad [[IN_SLICE]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, 0, 0] {
// CHECK:          tensor.yield [[ZERO_CST]] : f16
// CHECK:          : tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:     to tensor<1x128x?x65xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 65, 65]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[PAD]], [[WEIGHTS]]) rawFilterShape [128, 1, 2, 2] {
// CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:      ppe = #VPU.PPEStub<>, 
// CHECK-SAME:      strides = [2, 2]}
// CHECK-SAME:    -> tensor<1x128x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 32, 32]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[DWCONV]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 128, [[OUT_SIZE]], 32] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x128x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 32, 32]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:          into tensor<1x128x32x32xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 4 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @NCEPoolSOH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x64x12x12xf16, {order = #NHWC}>
func.func @NCEPoolSOH(%arg0: tensor<1x64x12x12xf16, {order = #NHWC}>) -> tensor<1x64x12x12xf16, {order = #NHWC}> {
  %0 = VPU.NCE.MaxPool(%arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      kernel_size = [1, 1],
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>, strides = [1, 1]
  } : tensor<1x64x12x12xf16, {order = #NHWC}> -> tensor<1x64x12x12xf16, {order = #NHWC}>

  return %0 : tensor<1x64x12x12xf16, {order = #NHWC}>
}

// CHECK: [[OUTPUT:%.+]] = tensor.empty() : tensor<1x64x12x12xf16, {order = #NHWC}>

// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (12) step (3) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:      -> (tensor<1x64x12x12xf16, {order = #NHWC}>)

// CHECK:       [[IN_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[LOOP_ITER]], 0] [1, 64, 3, 12] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<1x64x12x12xf16, {order = #NHWC}> to tensor<1x64x3x12xf16, {order = #NHWC}>

// CHECK:       [[POOL:%.+]] = VPU.NCE.MaxPool([[IN_SLICE]]) {
// CHECK-SAME:      kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:      ppe = #VPU.PPEStub<>,
// CHECK-SAME:      strides = [1, 1]}
// CHECK-SAME:    -> tensor<1x64x3x12xf16, {order = #NHWC}>

// CHECK:       scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[POOL]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 64, 3, 12] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x64x3x12xf16, {order = #NHWC}>
// CHECK-SAME:          into tensor<1x64x12x12xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 5 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// E#195056: test should use 5 clusters, but it ends up using just 4 due to uneven tiles not
//           being enabled, apart from reminder

// CHECK-LABEL:   @NCEEltwiseSOHUsing4Clusters
// CHECK-SAME:       [[INPUT0:%[^:]+]]: tensor<1x112x12x12xf16, {order = #NHWC}>
// CHECK-SAME:       [[INPUT1:%[^:]+]]: tensor<1x112x12x12xf16, {order = #NHWC}>
func.func @NCEEltwiseSOHUsing4Clusters(%arg0: tensor<1x112x12x12xf16, {order = #NHWC}>, %arg1: tensor<1x112x12x12xf16, {order = #NHWC}>)
    -> tensor<1x112x12x12xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
      is_inplace = true,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>
  } -> tensor<1x112x12x12xf16, {order = #NHWC}>

  return %0 : tensor<1x112x12x12xf16, {order = #NHWC}>
}

// CHECK: [[OUTPUT:%.+]] = tensor.empty() : tensor<1x112x12x12xf16, {order = #NHWC}>

// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (12) step (3) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:      -> (tensor<1x112x12x12xf16, {order = #NHWC}>)

// CHECK:       [[IN_SLICE0:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, [[LOOP_ITER]], 0] [1, 112, 3, 12] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<1x112x12x12xf16, {order = #NHWC}> to tensor<1x112x3x12xf16, {order = #NHWC}>

// CHECK:       [[IN_SLICE1:%.+]] = tensor.extract_slice [[INPUT1]][0, 0, [[LOOP_ITER]], 0] [1, 112, 3, 12] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<1x112x12x12xf16, {order = #NHWC}> to tensor<1x112x3x12xf16, {order = #NHWC}>

// CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[IN_SLICE0]], [[IN_SLICE1]])
// CHECK-SAME:    -> tensor<1x112x3x12xf16, {order = #NHWC}>

// CHECK:       scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[ELTWISE]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 112, 3, 12] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x112x3x12xf16, {order = #NHWC}> into tensor<1x112x12x12xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
config.Resources 3 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$TILE_SIZE_EXPR:.+]] = affine_map<(d0) -> (-d0 + 16, 6)>

// CHECK-LABEL:   @NCEPermuteSOH
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x43x16x16xf16>
func.func @NCEPermuteSOH(%arg0: tensor<1x43x16x16xf16>) -> tensor<1x48x16x16xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = f16, dstOrder = #NHWC, expandedChannels = 48 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x48x16x16xf16, {order = #NHWC}>
  return %0 : tensor<1x48x16x16xf16, {order = #NHWC}>
}

// CHECK: [[OUTPUT:%.+]] = tensor.empty() : tensor<1x48x16x16xf16, {order = #NHWC}>

// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (16) step (6) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:      -> (tensor<1x48x16x16xf16, {order = #NHWC}>)
// CHECK:       [[TILE_SZ:%.+]] = affine.min #[[$TILE_SIZE_EXPR]]([[LOOP_ITER]])

// CHECK:       [[IN_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[LOOP_ITER]], 0] [1, 43, [[TILE_SZ]], 16] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<1x43x16x16xf16>
// CHECK-SAME:      to tensor<1x43x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 43, 16, 16]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:       [[PERMUTE:%.+]] = VPU.NCE.Permute([[IN_SLICE]])
// CHECK-SAME:    -> tensor<1x48x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 16, 16]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[PERMUTE]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 48, [[TILE_SZ]], 16] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x48x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 16, 16]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:          into tensor<1x48x16x16xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
config.Resources 4 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$INPUT_OFFSET_EXPR:.+]] = affine_map<(d0) -> (d0 floordiv 4)>

// CHECK-LABEL: @DepthToSpaceSOH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x128x12x270xf16, {order = #NHWC}>
func.func @DepthToSpaceSOH(%arg0: tensor<1x128x12x270xf16, {order = #NHWC}>) -> tensor<1x8x48x1080xf16, {order = #NHWC}> {
  %0 = VPU.DepthToSpace(%arg0) {
    block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
  } : tensor<1x128x12x270xf16, {order = #NHWC}> -> tensor<1x8x48x1080xf16, {order = #NHWC}>

  return %0 : tensor<1x8x48x1080xf16, {order = #NHWC}>

// CHECK: [[OUTPUT:%.+]] = tensor.empty() : tensor<1x8x48x1080xf16, {order = #NHWC}>

// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (48) step (12) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:     -> (tensor<1x8x48x1080xf16, {order = #NHWC}>)

// CHECK:          [[INPUT_OFFSET:%.+]] = affine.apply #[[$INPUT_OFFSET_EXPR]]([[LOOP_ITER]])
// CHECK:          [[INPUT_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[INPUT_OFFSET]], 0] [1, 128, 3, 270] [1, 1, 1, 1]
// CHECK-SAME:         : tensor<1x128x12x270xf16, {order = #NHWC}> to tensor<1x128x3x270xf16, {order = #NHWC}>

// CHECK:          [[D2S:%.+]] = VPU.DepthToSpace([[INPUT_SLICE]])
// CHECK-SAME:         {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
// CHECK-SAME:       : tensor<1x128x3x270xf16, {order = #NHWC}> -> tensor<1x8x12x1080xf16, {order = #NHWC}>

// CHECK:          scf.forall.in_parallel
// CHECK:            tensor.parallel_insert_slice [[D2S]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 8, 12, 1080] [1, 1, 1, 1]
// CHECK-SAME:         : tensor<1x8x12x1080xf16, {order = #NHWC}> into tensor<1x8x48x1080xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
config.Resources 5 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$INPUT_OFFSET_EXPR:.+]] = affine_map<(d0) -> (d0 floordiv 4)>

// CHECK-LABEL: @DepthToSpaceSOW
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x128x12x270xf16, {order = #NHWC}>
func.func @DepthToSpaceSOW(%arg0: tensor<1x128x12x270xf16, {order = #NHWC}>) -> tensor<1x8x48x1080xf16, {order = #NHWC}> {
  %0 = VPU.DepthToSpace(%arg0) {
    block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>
  } : tensor<1x128x12x270xf16, {order = #NHWC}> -> tensor<1x8x48x1080xf16, {order = #NHWC}>

  return %0 : tensor<1x8x48x1080xf16, {order = #NHWC}>

// CHECK: [[OUTPUT:%.+]] = tensor.empty() : tensor<1x8x48x1080xf16, {order = #NHWC}>

// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (1080) step (216) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:     -> (tensor<1x8x48x1080xf16, {order = #NHWC}>)

// CHECK:          [[INPUT_OFFSET:%.+]] = affine.apply #[[$INPUT_OFFSET_EXPR]]([[LOOP_ITER]])
// CHECK:          [[INPUT_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[INPUT_OFFSET]]] [1, 128, 12, 54] [1, 1, 1, 1]
// CHECK-SAME:         : tensor<1x128x12x270xf16, {order = #NHWC}> to tensor<1x128x12x54xf16, {order = #NHWC}>

// CHECK:          [[D2S:%.+]] = VPU.DepthToSpace([[INPUT_SLICE]])
// CHECK-SAME:         {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
// CHECK-SAME:       : tensor<1x128x12x54xf16, {order = #NHWC}> -> tensor<1x8x48x216xf16, {order = #NHWC}>

// CHECK:          scf.forall.in_parallel
// CHECK:            tensor.parallel_insert_slice [[D2S]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 8, 48, 216] [1, 1, 1, 1]
// CHECK-SAME:         : tensor<1x8x48x216xf16, {order = #NHWC}> into tensor<1x8x48x1080xf16, {order = #NHWC}>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 200)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 640)>

!dynamicInType = tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
!dynamicOutType = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

!dynamicTiledInType = tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 200, 640]> : tensor<4xsi64>, order = #NCHW}>
!dynamicTiledOutType = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 200, 640]> : tensor<4xsi64>, order = #NCHW}>

module {
  config.Resources 4 of @NCE at 1.850000e+03 MHz {
    config.ExecutorResource 1 of @DPU
  }

// CHECK-DAG: #[[$MAP_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 200)>
// CHECK-DAG: #[[$MAP_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 640)>
// CHECK-DAG: #[[$MAP_MC_STEP_SZ:.+]] = affine_map<(d0) -> (d0 ceildiv 4)>
// CHECK-DAG: #[[$MAP_MC_SZ:.+]] = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 4)>

// CHECK-LABEL: @DynamicConvertMC
// CHECK-SAME:      [[INPUT:%[^:]+]]: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
func.func @DynamicConvertMC(%arg0: !dynamicInType) -> !dynamicOutType {
  %c640 = arith.constant 640 : index
  %c200 = arith.constant 200 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %dim = tensor.dim %arg0, %c2 : !dynamicInType
  %dim_0 = tensor.dim %arg0, %c3 : !dynamicInType
  %0 = tensor.empty(%dim, %dim_0) : !dynamicOutType
  %1 = scf.for %arg1 = %c0 to %dim step %c200 iter_args(%arg2 = %0)
      -> (!dynamicOutType) {
    %2 = scf.for %arg3 = %c0 to %dim_0 step %c640 iter_args(%arg4 = %arg2) -> (!dynamicOutType) {
      %3 = affine.min #map(%arg1)[%dim]
      %4 = affine.min #map1(%arg3)[%dim_0]

      %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg1, %arg3] [1, 3, %3, %4] [1, 1, 1, 1]
        : !dynamicInType to !dynamicTiledInType

      %5 = VPU.Convert(%extracted_slice) {
        dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
      } : !dynamicTiledInType -> !dynamicTiledOutType

      %inserted_slice = tensor.insert_slice %5 into %arg4[0, 0, %arg1, %arg3] [1, 3, %3, %4] [1, 1, 1, 1]
        : !dynamicTiledOutType into !dynamicOutType
      scf.yield %inserted_slice : !dynamicOutType
    }
    scf.yield %2 : !dynamicOutType
  }
  return %1 : !dynamicOutType
}

// CHECK-DAG: [[LOOP_H_STEP:%.+]] = arith.constant 200 : index
// CHECK-DAG: [[LOOP_W_STEP:%.+]] = arith.constant 640 : index
// CHECK-DAG: [[CST_2:%.+]] = arith.constant 2 : index
// CHECK-DAG: [[CST_3:%.+]] = arith.constant 3 : index
// CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

// CHECK:       [[DIM_H_END:%.+]] = tensor.dim [[INPUT]], [[CST_2]] :
// CHECK-SAME:    tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:       [[DIM_W_END:%.+]] = tensor.dim [[INPUT]], [[CST_3]] :
// CHECK-SAME:    tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:       [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_H_END]], [[DIM_W_END]])
// CHECK-SAME:    : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:       [[LOOP_H:%.+]] = scf.for [[LOOP_H_ITER:%[^:]+]] = [[LOOP_BEGIN]] to [[DIM_H_END]] step [[LOOP_H_STEP]] iter_args([[LOOP_OUT:%[^:]+]]  = [[LOOP_OUTPUT]])
// CHECK-SAME:     -> (tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>) {

// CHECK:          [[LOOP_W:%.+]] = scf.for [[LOOP_W_ITER:%[^:]+]] = [[LOOP_BEGIN]] to [[DIM_W_END]] step [[LOOP_W_STEP]]
// CHECK-SAME:        iter_args([[LOOP_OUT_H:%[^:]+]] = [[LOOP_OUT]])
// CHECK-SAME:        -> (tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>) {

// CHECK:                [[SIZE_H:%.+]] = affine.min #[[$MAP_H]]([[LOOP_H_ITER]])[[[DIM_H_END]]]
// CHECK:                [[SIZE_W:%.+]] = affine.min #[[$MAP_W]]([[LOOP_W_ITER]])[[[DIM_W_END]]]

// CHECK:                [[IN_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[LOOP_H_ITER]], [[LOOP_W_ITER]]] [1, 3, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]

// CHECK:                [[MC_OUTPUT:%.+]] = tensor.empty([[SIZE_H]], [[SIZE_W]])
// CHECK-SAME:             : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 200, 640]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:                [[MC_STEP:%.+]] = affine.apply #[[$MAP_MC_STEP_SZ]]([[SIZE_H]])

// CHECK:                [[MC_LOOP:%.+]] = scf.forall ([[MC_LOOP_ITER:%.+]]) = (0) to ([[SIZE_H]]) step ([[MC_STEP]]) shared_outs([[MC_LOOP_OUT:%.+]] = [[MC_OUTPUT]])
// CHECK-SAME:             -> (tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 200, 640]> : tensor<4xsi64>, order = #NCHW}>)

// CHECK:                    [[MC_TILE_SZ:%.+]] = affine.min #[[$MAP_MC_SZ]]([[MC_LOOP_ITER]], [[SIZE_H]])[[[SIZE_H]]]

// CHECK:                    [[MC_IN_SLICE:%.+]] = tensor.extract_slice [[IN_SLICE]][0, 0, [[MC_LOOP_ITER]], 0] [1, 3, [[MC_TILE_SZ]], 640] [1, 1, 1, 1]
// CHECK-SAME:                 : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 200, 640]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-SAME:                 to tensor<1x3x?x640xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 640]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:                    [[CONVERT:%.+]] = VPU.Convert([[MC_IN_SLICE]])

// CHECK:                    scf.forall.in_parallel
// CHECK:                      tensor.parallel_insert_slice [[CONVERT]] into [[MC_LOOP_OUT]][0, 0, [[MC_LOOP_ITER]], 0] [1, 3, [[MC_TILE_SZ]], 640] [1, 1, 1, 1]
// CHECK-SAME:                   : tensor<1x3x?x640xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 640]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-SAME:                   into tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 200, 640]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:                [[INSERT_SLICE:%.+]] = tensor.insert_slice [[MC_LOOP]] into [[LOOP_OUT_H]][0, 0, [[LOOP_H_ITER]], [[LOOP_W_ITER]]] [1, 3, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
// CHECK:                  scf.yield [[INSERT_SLICE]] : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
// CHECK: scf.yield [[LOOP_W]]
// CHECK: return [[LOOP_H]] : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 96)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 732)>
#map2 = affine_map<(d0) -> (d0 floordiv 2 - 1, 0)>
#map3 = affine_map<(d0) -> (-(d0 floordiv 2) + 1, 0)>
#map4 = affine_map<()[s0] -> (1, s0)>
#map5 = affine_map<(d0, d1)[s0] -> (d0 - s0 + d1 floordiv 2 + 2, 0)>
#map6 = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 floordiv 2 + 2)>

!inputConvDynamicType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
!outputD2SDynamicType = tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>

!inputConvDynamicTiledType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 366]> : tensor<4xsi64>, order = #NHWC}>
!inputConvDynamicTiledPaddedType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 368]> : tensor<4xsi64>, order = #NHWC}>

!outConvDynamicTiled = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 366]> : tensor<4xsi64>, order = #NHWC}>
!outD2SDynamicTiled = tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 96, 732]> : tensor<4xsi64>, order = #NHWC}>

module @test {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 96)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 732)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 2 - 1, 0)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0) -> (-(d0 floordiv 2) + 1, 0)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0, d1)[s0] -> (d0 - s0 + d1 floordiv 2 + 2, 0)>
// CHECK-DAG: #[[$MAP6:.+]] = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 floordiv 2 + 2)>
// CHECK-DAG: #[[$MAP7:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG: #[[$MAP8:.+]] = affine_map<(d0) -> (d0 ceildiv 3)>
// CHECK-DAG: #[[$MAP9:.+]] = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 3)>
// CHECK-DAG: #[[$MAP10:.+]] = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 2)>
// CHECK-DAG: #[[$MAP11:.+]] = affine_map<(d0, d1) -> (0, d0 - d1)>
// CHECK-DAG: #[[$MAP12:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 - d3 floordiv 2, 0)>
// CHECK-DAG: #[[$MAP13:.+]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2)>
// CHECK-DAG: #[[$MAP14:.+]] = affine_map<(d0) -> (-d0, 0)>
// CHECK-DAG: #[[$MAP15:.+]] = affine_map<(d0) -> (d0 floordiv 2)>

// CHECK: @NotFusedMCFor2DVFChainConvAddD2S
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>)
func.func @NotFusedMCFor2DVFChainConvAddD2S(%arg0: !inputConvDynamicType) -> !outputD2SDynamicType {
  %cst = arith.constant 0.000000e+00 : f16
  %c732 = arith.constant 732 : index
  %c96 = arith.constant 96 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %cst_0 = const.Declare tensor<16x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    : tensor<16x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

  %dim = tensor.dim %arg0, %c2 : !inputConvDynamicType
  %0 = arith.muli %dim, %c2 : index
  %dim_1 = tensor.dim %arg0, %c3 : !inputConvDynamicType
  %1 = arith.muli %dim_1, %c2 : index
  %2 = tensor.empty(%0, %1) : !outputD2SDynamicType
  %dim_2 = tensor.dim %arg0, %c2 : !inputConvDynamicType
  %3 = arith.muli %dim_2, %c2 : index
  %dim_3 = tensor.dim %arg0, %c3 : !inputConvDynamicType
  %4 = arith.muli %dim_3, %c2 : index
  %5 = scf.for %arg1 = %c0 to %3 step %c96 iter_args(%arg2 = %2) -> (!outputD2SDynamicType) {
    %6 = scf.for %arg3 = %c0 to %4 step %c732 iter_args(%arg4 = %arg2) -> (!outputD2SDynamicType) {
      %7 = affine.min #map(%arg1)[%3]
      %8 = affine.min #map1(%arg3)[%4]
      %dim_4 = tensor.dim %arg0, %c2 : !inputConvDynamicType
      %dim_5 = tensor.dim %arg0, %c3 : !inputConvDynamicType
      %9 = affine.max #map2(%arg1)
      %10 = affine.max #map3(%arg1)
      %11 = affine.min #map4()[%10]
      %12 = affine.max #map5(%9, %7)[%dim_4]
      %13 = affine.min #map4()[%12]
      %14 = affine.apply #map6(%11, %13, %7)
      %15 = affine.max #map2(%arg3)
      %16 = affine.max #map3(%arg3)
      %17 = affine.min #map4()[%16]
      %18 = affine.max #map5(%15, %8)[%dim_5]
      %19 = affine.min #map4()[%18]
      %20 = affine.apply #map6(%17, %19, %8)

      %extracted_slice = tensor.extract_slice %arg0[0, 0, %9, %15] [1, 32, %14, %20] [1, 1, 1, 1]
        : !inputConvDynamicType to !inputConvDynamicTiledType
      %padded = tensor.pad %extracted_slice low[0, 0, %11, %17] high[0, 0, %13, %19] {
      ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
        tensor.yield %cst : f16
      } : !inputConvDynamicTiledType to !inputConvDynamicTiledPaddedType

      %21 = VPU.NCE.Convolution(%padded, %cst_0) rawFilterShape [16, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,  strides = [1, 1]}
      : !inputConvDynamicTiledPaddedType, tensor<16x32x3x3xf16, {order = #NHWC}>
      -> !outConvDynamicTiled

      %22 = VPU.DepthToSpace(%21) {
        block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
      } : !outConvDynamicTiled -> !outD2SDynamicTiled

      %inserted_slice = tensor.insert_slice %22 into %arg4[0, 0, %arg1, %arg3] [1, 4, %7, %8] [1, 1, 1, 1]
        : !outD2SDynamicTiled into !outputD2SDynamicType
      scf.yield %inserted_slice : !outputD2SDynamicType
    }
    scf.yield %6 : !outputD2SDynamicType
  }
  return %5 : !outputD2SDynamicType
}

// CHECK-DAG:   [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
// CHECK-DAG:   [[LOOP_STEP_H:%.+]] = arith.constant 96 : index
// CHECK-DAG:   [[LOOP_STEP_W:%.+]] = arith.constant 732 : index
// CHECK-DAG:   [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
// CHECK-DAG:   [[CST_VAL_2:%.+]] = arith.constant 2 : index
// CHECK-DAG:   [[CST_MIN_2:%.+]] = arith.constant -2 : index
// CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x32x3x3xf16, {order = #NHWC}>

// CHECK:   [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[CST_VAL_2]]
// CHECK:   [[OUT_DIM_H:%.+]] = arith.muli [[DIM_H]], [[CST_VAL_2]]

// CHECK:   [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]]
// CHECK:   [[OUT_DIM_W:%.+]] = arith.muli [[DIM_W]], [[CST_VAL_2]]

// CHECK:   [[EMPTY:%.+]] = tensor.empty([[OUT_DIM_H]], [[OUT_DIM_W]])
// CHECK-SAME:    : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:   [[RESULT:%.+]] = scf.for [[SLICE_OFFSET_H:%.+]] = [[LOOP_BEGIN]] to [[OUT_DIM_H]] step [[LOOP_STEP_H]] iter_args([[OUTER_OUTPUT:%.+]] = [[EMPTY]])
// CHECK:       [[RESULT_W:%.+]] = scf.for [[SLICE_OFFSET_W:%.+]] = [[LOOP_BEGIN]] to [[OUT_DIM_W]] step [[LOOP_STEP_W]] iter_args([[INNER_OUTPUT:%.+]] = [[OUTER_OUTPUT]])

// CHECK:       [[SLICE_SIZE_H:%.+]] = affine.min #[[$MAP]]([[SLICE_OFFSET_H]])[[[OUT_DIM_H]]]
// CHECK:       [[SLICE_SIZE_W:%.+]] = affine.min #[[$MAP1]]([[SLICE_OFFSET_W]])[[[OUT_DIM_W]]]

// CHECK:       [[IN_SLICE_OFFSET_H:%.+]] = affine.max #[[$MAP2]]([[SLICE_OFFSET_H]])
// CHECK:       [[TEMP_VAL0:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET_H]])
// CHECK:       [[PAD_BOTTOM:%.+]] = affine.min #[[$MAP4]]()[[[TEMP_VAL0]]]
// CHECK:       [[TEMP_VAL1:%.+]] = affine.max #[[$MAP5]]([[IN_SLICE_OFFSET_H]], [[SLICE_SIZE_H]])[[[DIM_H]]]
// CHECK:       [[PAD_TOP:%.+]] = affine.min #[[$MAP4]]()[[[TEMP_VAL1]]]
// CHECK:       [[IN_SLICE_SIZE_H:%.+]] = affine.apply #[[$MAP6]]([[PAD_BOTTOM]], [[PAD_TOP]], [[SLICE_SIZE_H]])

// CHECK:       [[IN_SLICE_OFFSET_W:%.+]] = affine.max #[[$MAP2]]([[SLICE_OFFSET_W]])
// CHECK:       [[TEMP_VAL2:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET_W]])
// CHECK:       [[PAD_LEFT:%.+]] = affine.min #[[$MAP4]]()[[[TEMP_VAL2]]]
// CHECK:       [[TEMP_VAL3:%.+]] = affine.max #[[$MAP5]]([[IN_SLICE_OFFSET_W]], [[SLICE_SIZE_W]])[[[DIM_W]]]
// CHECK:       [[PAD_RIGHT:%.+]] = affine.min #[[$MAP4]]()[[[TEMP_VAL3]]]
// CHECK:       [[IN_SLICE_SIZE_W:%.+]] = affine.apply #[[$MAP6]]([[PAD_LEFT]], [[PAD_RIGHT]], [[SLICE_SIZE_W]])

// CHECK:       [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IN_SLICE_OFFSET_H]], [[IN_SLICE_OFFSET_W]]] [1, 32, [[IN_SLICE_SIZE_H]], [[IN_SLICE_SIZE_W]]]

// CHECK:       [[TOTAL_PAD_H:%.+]] = affine.apply #[[$MAP7]]([[PAD_BOTTOM]], [[PAD_TOP]])
// CHECK:       [[PADDED_IN_H:%.+]] = arith.addi [[IN_SLICE_SIZE_H]], [[TOTAL_PAD_H]] : index
// CHECK:       [[MC_CONV_OUT_H:%.+]] = arith.addi [[PADDED_IN_H]], [[CST_MIN_2]] : index
// CHECK:       [[TOTAL_PAD_W:%.+]] = affine.apply #[[$MAP7]]([[PAD_LEFT]], [[PAD_RIGHT]])
// CHECK:       [[PADDED_IN_W:%.+]] = arith.addi [[IN_SLICE_SIZE_W]], [[TOTAL_PAD_W]] : index
// CHECK:       [[MC_CONV_OUT_W:%.+]] = arith.addi [[PADDED_IN_W]], [[CST_MIN_2]] : index

// CHECK:       [[MC_CONV_OUT:%.+]] = tensor.empty([[MC_CONV_OUT_H]], [[MC_CONV_OUT_W]])
// CHECK-SAME:    : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 366]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:       [[MC_CONV_STEP_H:%.+]] = affine.apply #[[$MAP8]]([[MC_CONV_OUT_H]])

// CHECK:       [[MC_CONV_LOOP:%.+]] = scf.forall ([[MC_CONV_LOOP_ITER:%.+]]) = (0) to ([[MC_CONV_OUT_H]]) step ([[MC_CONV_STEP_H]])
// CHECK-SAME:    shared_outs([[MC_CONV_LOOP_OUT:%.+]] = [[MC_CONV_OUT]])
// CHECK-SAME:    -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 366]> : tensor<4xsi64>, order = #NHWC}>)

// CHECK:         [[OUT_CONV_H_SIZE:%.+]] = affine.min #[[$MAP9]]([[MC_CONV_LOOP_ITER]], [[MC_CONV_OUT_H]])[[[MC_CONV_OUT_H]]]
// CHECK:         [[MC_OFFSET_H:%.+]] = affine.max #[[$MAP11]]([[MC_CONV_LOOP_ITER]], [[PAD_BOTTOM]])

// CHECK:         [[MC_PAD_BOTTOM:%.+]] = affine.max #[[$MAP11]]([[PAD_BOTTOM]], [[MC_OFFSET_H]])
// CHECK:         [[MC_PAD_TOP:%.+]] = affine.max #[[$MAP12]]([[MC_CONV_LOOP_ITER]], [[OUT_CONV_H_SIZE]], [[PAD_TOP]], [[SLICE_SIZE_H]])

// CHECK:         [[IN_CONV_H_SIZE:%.+]] = affine.apply #[[$MAP10]]([[MC_PAD_BOTTOM]], [[MC_PAD_TOP]], [[OUT_CONV_H_SIZE]])

// CHECK:         [[MC_OFFSET_W:%.+]] = affine.max #[[$MAP14]]([[PAD_LEFT]])
// CHECK:         [[IN_CONV_W_SIZE:%.+]] = affine.apply #[[$MAP13]]([[PADDED_IN_W]], [[PAD_LEFT]], [[PAD_RIGHT]])

// CHECK:         [[IN_CONV_MC_SLICE:%.+]] = tensor.extract_slice [[SLICE]][0, 0, [[MC_OFFSET_H]], [[MC_OFFSET_W]]] [1, 32, [[IN_CONV_H_SIZE]], [[IN_CONV_W_SIZE]]]
// CHECK-SAME:      : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 366]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:      to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 366]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:         [[PAD:%.+]] = tensor.pad [[IN_CONV_MC_SLICE]] low[0, 0, [[MC_PAD_BOTTOM]], [[PAD_LEFT]]] high[0, 0, [[MC_PAD_TOP]], [[PAD_RIGHT]]]
// CHECK:           : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 366]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:      to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 18, 368]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:         [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[WEIGHTS]]) rawFilterShape [16, 32, 3, 3] {
// CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:        {{.*}}: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 18, 368]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x32x3x3xf16, {order = #NHWC}>
// CHECK-SAME:        -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 366]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:         [[CAST:%.+]] = tensor.cast [[CONV]]
// CHECK-SAME:      : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 366]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:      to tensor<1x16x?x366xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 366]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[CAST]] into [[MC_CONV_LOOP_OUT]][0, 0, [[MC_CONV_LOOP_ITER]], 0] [1, 16, [[OUT_CONV_H_SIZE]], 366] [1, 1, 1, 1]
// CHECK-SAME:        : tensor<1x16x?x366xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 366]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:        into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 366]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[MC_D2S_OUT_H:%.+]] = arith.muli [[MC_CONV_OUT_H]], [[CST_VAL_2]] : index
// CHECK:       [[MC_D2S_OUT_W:%.+]] = arith.muli [[MC_CONV_OUT_W]], [[CST_VAL_2]] : index
// CHECK:       [[MC_D2S_OUT:%.+]] = tensor.empty([[MC_D2S_OUT_H]], [[MC_D2S_OUT_W]])
// CHECK-SAME:      : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 96, 732]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:       [[MC_D2S_STEP:%.+]] = affine.apply #[[$MAP8]]([[MC_D2S_OUT_H]])

// CHECK:       [[MC_D2S_LOOP:%.+]] = scf.forall ([[MC_D2S_LOOP_ITER:%.+]]) = (0) to ([[MC_D2S_OUT_H]]) step ([[MC_D2S_STEP]])
// CHECK-SAME:      shared_outs([[MC_D2S_LOOP_OUT:%.+]] = [[MC_D2S_OUT]])
// CHECK-SAME:      -> (tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 96, 732]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:           [[OUT_D2S_H_SIZE:%.+]] = affine.min #[[$MAP9]]([[MC_D2S_LOOP_ITER]], [[MC_D2S_OUT_H]])[[[MC_D2S_OUT_H]]]
// CHECK:           [[IN_D2S_H_OFFSET:%.+]] = affine.apply #[[$MAP15]]([[MC_D2S_LOOP_ITER]])
// CHECK:           [[IN_D2S_H_SIZE:%.+]] = affine.apply #[[$MAP15]]([[OUT_D2S_H_SIZE]])

// CHECK:           [[IN_D2S_MC_SLICE:%.+]] = tensor.extract_slice [[MC_CONV_LOOP]][0, 0, [[IN_D2S_H_OFFSET]], 0] [1, 16, [[IN_D2S_H_SIZE]], 366] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 366]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:          to tensor<1x16x?x366xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 366]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:           [[D2S:%.+]] = VPU.DepthToSpace([[IN_D2S_MC_SLICE]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
// CHECK-SAME:          : tensor<1x16x?x366xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 366]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:          -> tensor<1x4x?x732xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 32, 732]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:           scf.forall.in_parallel
// CHECK:             tensor.parallel_insert_slice [[D2S]] into [[MC_D2S_LOOP_OUT]][0, 0, [[MC_D2S_LOOP_ITER]], 0] [1, 4, [[OUT_D2S_H_SIZE]], 732] [1, 1, 1, 1]
// CHECK-SAME:            : tensor<1x4x?x732xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 32, 732]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:            into tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 96, 732]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[INSERT:%.+]] = tensor.insert_slice [[MC_D2S_LOOP]] into [[INNER_OUTPUT]]
// CHECK:       scf.yield [[INSERT]]
// CHECK:   scf.yield [[RESULT_W]]
// CHECK:   return [[RESULT]] : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 103)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 512)>

!dynamicConvInput = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>
!dynamicConvOutput = tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>

!dynamicConvTiledInput = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 103, 512]> : tensor<4xsi64>, order = #NHWC}>
!dynamicConvTiledOutput = tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 103, 512]> : tensor<4xsi64>, order = #NHWC}>

module @test {
config.Resources 4 of @NCE at 6.000000e+02 MHz {
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$HEIGHT_SZ_EXPR:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 103)>
// CHECK: #[[$WIDTH_SZ_EXPR:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
// CHECK: #[[$MC_STEP_EXPR:.+]] = affine_map<(d0) -> (d0 ceildiv 4)>
// CHECK: #[[$MC_IN_TILE_SZ_EXPR:.+]] = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 4)>

// CHECK-LABEL: @DynamicMatMulMC
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>)
func.func @DynamicMatMulMC(%arg0: !dynamicConvInput) -> !dynamicConvOutput {
  %c512 = arith.constant 512 : index
  %c103 = arith.constant 103 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %cst = const.Declare tensor<256x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x1x1xf16>, [#const.Reorder<#NHWC>]
  %c2 = arith.constant 2 : index
  %dim = tensor.dim %arg0, %c2 : !dynamicConvInput
  %dim_0 = tensor.dim %arg0, %c3 : !dynamicConvInput
  %0 = tensor.empty(%dim, %dim_0) : !dynamicConvOutput
  %1 = scf.for %arg1 = %c0 to %dim step %c103 iter_args(%arg2 = %0) -> (!dynamicConvOutput) {
    %2 = scf.for %arg3 = %c0 to %dim_0 step %c512 iter_args(%arg4 = %arg2) -> (!dynamicConvOutput) {
      %3 = affine.min #map(%arg1)[%dim]
      %4 = affine.min #map1(%arg3)[%dim_0]
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg1, %arg3] [1, 32, %3, %4] [1, 1, 1, 1]
        : !dynamicConvInput to !dynamicConvTiledInput

      %5 = VPU.NCE.Convolution(%extracted_slice, %cst) rawFilterShape [256, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,  strides = [1, 1]
      } : !dynamicConvTiledInput, tensor<256x32x1x1xf16, {order = #NHWC}>
        -> !dynamicConvTiledOutput

      %inserted_slice = tensor.insert_slice %5 into %arg4[0, 0, %arg1, %arg3] [1, 256, %3, %4] [1, 1, 1, 1]
        : !dynamicConvTiledOutput into !dynamicConvOutput
      scf.yield %inserted_slice : !dynamicConvOutput
    }
    scf.yield %2 : !dynamicConvOutput
  }
  return %1 : !dynamicConvOutput

// CHECK-DAG: [[WIDTH_STEP:%.+]] = arith.constant 512 : index
// CHECK-DAG: [[HEIGHT_STEP:%.+]] = arith.constant 103 : index
// CHECK-DAG: [[ZERO_CST:%.+]] = arith.constant 0 : index
// CHECK-DAG: [[TWO_CST:%.+]] = arith.constant 2 : index
// CHECK-DAG: [[THREE_CST:%.+]] = arith.constant 3 : index
// CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x1x1xf16, {order = #NHWC}>

// CHECK:       [[HEIGHT_DIM:%.+]] = tensor.dim [[INPUT]], [[TWO_CST]]
// CHECK-SAME:    : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:       [[WIDTH_DIM:%.+]] = tensor.dim [[INPUT]], [[THREE_CST]]
// CHECK-SAME:    : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:       [[OUTPUT:%.+]] = tensor.empty([[HEIGHT_DIM]], [[WIDTH_DIM]])
// CHECK-SAME:    : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[LOOP_H:%.+]] = scf.for [[LOOP_ITER_H:%.+]] = [[ZERO_CST]] to [[HEIGHT_DIM]] step [[HEIGHT_STEP]]
// CHECK-SAME:    iter_args([[LOOP_OUTPUT_H:%.+]] = [[OUTPUT]])
// CHECK-SAME:      -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>)

// CHECK:         [[LOOP_W:%.+]] = scf.for [[LOOP_ITER_W:%.+]] = [[ZERO_CST]] to [[WIDTH_DIM]] step [[WIDTH_STEP]]
// CHECK-SAME:      iter_args([[LOOP_OUTPUT_W:%.+]] = [[LOOP_OUTPUT_H]])
// CHECK-SAME:        -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>) {

// CHECK:         [[HEIGHT_SIZE:%.+]] = affine.min #{{.+}}([[LOOP_ITER_H]])[[[HEIGHT_DIM]]]
// CHECK:         [[WIDTH_SIZE:%.+]] = affine.min #{{.+}}([[LOOP_ITER_W]])[[[WIDTH_DIM]]]
// CHECK:         [[INPUT_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 32, [[HEIGHT_SIZE]], [[WIDTH_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:      to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 103, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:         [[MC_OUT:%.+]] = tensor.empty([[HEIGHT_SIZE]], [[WIDTH_SIZE]])
// CHECK-SAME:      : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 103, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         [[MC_STEP:%.+]] = affine.apply #{{.+}}([[HEIGHT_SIZE]])

// CHECK:           [[MC_LOOP:%.+]] = scf.forall ([[MC_LOOP_ITER:%.+]]) = (0) to ([[HEIGHT_SIZE]]) step ([[MC_STEP]])
// CHECK-SAME:        shared_outs([[MC_LOOP_OUT:%.+]] = [[MC_OUT]])
// CHECK-SAME:        -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 103, 512]> : tensor<4xsi64>, order = #NHWC}>) {

// CHECK:             [[MC_IN_H:%.+]] = affine.min #{{.+}}([[MC_LOOP_ITER]], [[HEIGHT_SIZE]])[[[HEIGHT_SIZE]]]
// CHECK:             [[MC_IN_SLICE:%.+]] = tensor.extract_slice [[INPUT_SLICE]][0, 0, [[MC_LOOP_ITER]], 0] [1, 32, [[MC_IN_H]], [[WIDTH_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 103, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:          to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 26, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:             [[CONV:%.+]] = VPU.NCE.Convolution([[MC_IN_SLICE]], [[WEIGHTS]])
// CHECK-SAME:          : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 26, 512]> : tensor<4xsi64>, order = #NHWC}>,
// CHECK-SAME:          -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 26, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:             [[CAST:%.+]] = tensor.cast [[CONV]]
// CHECK-SAME:          : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 26, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:          to tensor<1x256x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 26, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:             scf.forall.in_parallel
// CHECK:               tensor.parallel_insert_slice [[CAST]] into [[MC_LOOP_OUT]][0, 0, [[MC_LOOP_ITER]], 0] [1, 256, [[MC_IN_H]], 512] [1, 1, 1, 1]
// CHECK-SAME:            : tensor<1x256x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 26, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:            into tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 103, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:           [[SCF_OUT:%.+]] = tensor.insert_slice [[MC_LOOP]]
// CHECK-SAME:        into [[LOOP_OUTPUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 256, [[HEIGHT_SIZE]], [[WIDTH_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:        : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 103, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:        into tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:           scf.yield [[SCF_OUT]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 103)>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 103)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 175, 44)>

module @test {
config.Resources 4 of @NCE at 6.000000e+02 MHz {
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @DynamicNCEReduce
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
func.func @DynamicNCEReduce(%arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> {
  %c103 = arith.constant 103 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %dim = tensor.dim %arg0, %c3 : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %0 = tensor.empty(%dim) : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %dim step %c103 iter_args(%arg2 = %0) -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
    %2 = affine.min #map(%arg1)[%dim]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 32, 175, %2] [1, 1, 1, 1] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.NCE.Reduce(%extracted_slice) {
        axes = [1],
        input_padding = [0, 12, 0, 0],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        op_type = #VPU.reduce_type<SUM>,
        ppe = #VPU.PPEStub<>,
        tiling_loop_index = 0 : i64
    } -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %inserted_slice = tensor.insert_slice %3 into %arg2[0, 0, 0, %arg1] [1, 1, 175, %2] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  }
  return %1 : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-DAG:    [[C103:%.+]] = arith.constant 103 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]])
// CHECK-SAME:      : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[LOOP:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C103]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:        -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-NEXT:      [[TILING_SIZE:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
// CHECK-DAG:       [[TILING_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[IDX]]] [1, 32, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:       [[MC_EMPTY:%.+]] = tensor.empty([[TILING_SIZE]])
// CHECK-SAME:            : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[MC_LOOP:%.+]] = scf.forall ([[MC_IDX:%.+]]) = (0) to (175) step (44) shared_outs([[MC_OUT:%.+]] = [[MC_EMPTY]])
// CHECK-SAME:            -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-DAG:          [[MC_H_SIZE:%.+]] = affine.min #[[$MAP1]]([[MC_IDX]])
// CHECK-DAG:          [[MC_SLICE:%.+]] = tensor.extract_slice [[TILING_SLICE]][0, 0, [[MC_IDX]], 0] [1, 32, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:              [[REDUCED:%.+]] = VPU.NCE.Reduce([[MC_SLICE]])
// CHECK-SAME:            -> tensor<1x1x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:         scf.forall.in_parallel {
// CHECK-NEXT:           tensor.parallel_insert_slice [[REDUCED]] into [[MC_OUT]][0, 0, [[MC_IDX]], 0] [1, 1, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:                into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[TILING_INSERT_SLICE:%.+]] = tensor.insert_slice [[MC_LOOP]] into [[OUT]][0, 0, 0, [[IDX]]] [1, 1, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:           into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      scf.yield [[TILING_INSERT_SLICE]]
// CHECK:        return [[LOOP]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 103)>

// CHECK: #[[MAP:.+]]= affine_map<(d0)[s0] -> (-d0 + s0, 103)>
// CHECK: #[[MAP1:.+]]= affine_map<(d0) -> (-d0 + 175, 44)>

module @test {
config.Resources 4 of @NCE at 6.000000e+02 MHz {
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @DynamicReduceL1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
func.func @DynamicReduceL1(%arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> {
  %c103 = arith.constant 103 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %dim = tensor.dim %arg0, %c3 : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %0 = tensor.empty(%dim) : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %dim step %c103 iter_args(%arg2 = %0) -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
    %2 = affine.min #map(%arg1)[%dim]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 32, 175, %2] [1, 1, 1, 1] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.ReduceL1(%extracted_slice) {
        axes_value = [1],
        keep_dims,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %inserted_slice = tensor.insert_slice %3 into %arg2[0, 0, 0, %arg1] [1, 1, 175, %2] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  }
  return %1 : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-DAG:    [[C103:%.+]] = arith.constant 103 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]])
// CHECK-SAME:      : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[LOOP:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C103]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:        -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-NEXT:      [[TILING_SIZE:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
// CHECK-DAG:       [[TILING_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[IDX]]] [1, 32, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:       [[MC_EMPTY:%.+]] = tensor.empty([[TILING_SIZE]])
// CHECK-SAME:            : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[MC_LOOP:%.+]] = scf.forall ([[MC_IDX:%.+]]) = (0) to (175) step (44) shared_outs([[MC_OUT:%.+]] = [[MC_EMPTY]])
// CHECK-SAME:            -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-DAG:          [[MC_H_SIZE:%.+]] = affine.min #[[$MAP1]]([[MC_IDX]])
// CHECK-DAG:          [[MC_SLICE:%.+]] = tensor.extract_slice [[TILING_SLICE]][0, 0, [[MC_IDX]], 0] [1, 32, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:              [[REDUCED:%.+]] = VPU.ReduceL1([[MC_SLICE]])
// CHECK-SAME:            -> tensor<1x1x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:         scf.forall.in_parallel {
// CHECK-NEXT:           tensor.parallel_insert_slice [[REDUCED]] into [[MC_OUT]][0, 0, [[MC_IDX]], 0] [1, 1, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:                into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[TILING_INSERT_SLICE:%.+]] = tensor.insert_slice [[MC_LOOP]] into [[OUT]][0, 0, 0, [[IDX]]] [1, 1, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:           into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      scf.yield [[TILING_INSERT_SLICE]]
// CHECK:        return [[LOOP]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 103)>

// CHECK: #[[MAP:.+]]= affine_map<(d0)[s0] -> (-d0 + s0, 103)>
// CHECK: #[[MAP1:.+]]= affine_map<(d0) -> (-d0 + 175, 44)>

module @test {
config.Resources 4 of @NCE at 6.000000e+02 MHz {
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @DynamicReduceL2
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
func.func @DynamicReduceL2(%arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> {
  %c103 = arith.constant 103 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %dim = tensor.dim %arg0, %c3 : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %0 = tensor.empty(%dim) : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %dim step %c103 iter_args(%arg2 = %0) -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
    %2 = affine.min #map(%arg1)[%dim]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 32, 175, %2] [1, 1, 1, 1] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.ReduceL2(%extracted_slice) {
        axes_value = [1],
        keep_dims,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %inserted_slice = tensor.insert_slice %3 into %arg2[0, 0, 0, %arg1] [1, 1, 175, %2] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  }
  return %1 : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-DAG:    [[C103:%.+]] = arith.constant 103 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]])
// CHECK-SAME:      : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[LOOP:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C103]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:        -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-NEXT:      [[TILING_SIZE:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
// CHECK-DAG:       [[TILING_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[IDX]]] [1, 32, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:       [[MC_EMPTY:%.+]] = tensor.empty([[TILING_SIZE]])
// CHECK-SAME:            : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[MC_LOOP:%.+]] = scf.forall ([[MC_IDX:%.+]]) = (0) to (175) step (44) shared_outs([[MC_OUT:%.+]] = [[MC_EMPTY]])
// CHECK-SAME:            -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-DAG:          [[MC_H_SIZE:%.+]] = affine.min #[[$MAP1]]([[MC_IDX]])
// CHECK-DAG:          [[MC_SLICE:%.+]] = tensor.extract_slice [[TILING_SLICE]][0, 0, [[MC_IDX]], 0] [1, 32, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:              [[REDUCED:%.+]] = VPU.ReduceL2([[MC_SLICE]])
// CHECK-SAME:            -> tensor<1x1x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:         scf.forall.in_parallel {
// CHECK-NEXT:           tensor.parallel_insert_slice [[REDUCED]] into [[MC_OUT]][0, 0, [[MC_IDX]], 0] [1, 1, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:                into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[TILING_INSERT_SLICE:%.+]] = tensor.insert_slice [[MC_LOOP]] into [[OUT]][0, 0, 0, [[IDX]]] [1, 1, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:           into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      scf.yield [[TILING_INSERT_SLICE]]
// CHECK:        return [[LOOP]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 103)>

// CHECK: #[[MAP:.+]]= affine_map<(d0)[s0] -> (-d0 + s0, 103)>
// CHECK: #[[MAP1:.+]]= affine_map<(d0) -> (-d0 + 175, 44)>

module @test {
config.Resources 4 of @NCE at 6.000000e+02 MHz {
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @DynamicReduceMax
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
func.func @DynamicReduceMax(%arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> {
  %c103 = arith.constant 103 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %dim = tensor.dim %arg0, %c3 : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %0 = tensor.empty(%dim) : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %dim step %c103 iter_args(%arg2 = %0) -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
    %2 = affine.min #map(%arg1)[%dim]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 32, 175, %2] [1, 1, 1, 1] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.ReduceMax(%extracted_slice) {
        axes_value = [1],
        keep_dims,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %inserted_slice = tensor.insert_slice %3 into %arg2[0, 0, 0, %arg1] [1, 1, 175, %2] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  }
  return %1 : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-DAG:    [[C103:%.+]] = arith.constant 103 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]])
// CHECK-SAME:      : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[LOOP:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C103]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:        -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-NEXT:      [[TILING_SIZE:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
// CHECK-DAG:       [[TILING_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[IDX]]] [1, 32, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:       [[MC_EMPTY:%.+]] = tensor.empty([[TILING_SIZE]])
// CHECK-SAME:            : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[MC_LOOP:%.+]] = scf.forall ([[MC_IDX:%.+]]) = (0) to (175) step (44) shared_outs([[MC_OUT:%.+]] = [[MC_EMPTY]])
// CHECK-SAME:            -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-DAG:          [[MC_H_SIZE:%.+]] = affine.min #[[$MAP1]]([[MC_IDX]])
// CHECK-DAG:          [[MC_SLICE:%.+]] = tensor.extract_slice [[TILING_SLICE]][0, 0, [[MC_IDX]], 0] [1, 32, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:              [[REDUCED:%.+]] = VPU.ReduceMax([[MC_SLICE]])
// CHECK-SAME:            -> tensor<1x1x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:         scf.forall.in_parallel {
// CHECK-NEXT:           tensor.parallel_insert_slice [[REDUCED]] into [[MC_OUT]][0, 0, [[MC_IDX]], 0] [1, 1, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:                into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[TILING_INSERT_SLICE:%.+]] = tensor.insert_slice [[MC_LOOP]] into [[OUT]][0, 0, 0, [[IDX]]] [1, 1, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:           into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      scf.yield [[TILING_INSERT_SLICE]]
// CHECK:        return [[LOOP]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 103)>

// CHECK: #[[MAP:.+]]= affine_map<(d0)[s0] -> (-d0 + s0, 103)>
// CHECK: #[[MAP1:.+]]= affine_map<(d0) -> (-d0 + 175, 44)>

module @test {
config.Resources 4 of @NCE at 6.000000e+02 MHz {
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @DynamicReduceMean
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
func.func @DynamicReduceMean(%arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> {
  %c103 = arith.constant 103 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %dim = tensor.dim %arg0, %c3 : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %0 = tensor.empty(%dim) : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %dim step %c103 iter_args(%arg2 = %0) -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
    %2 = affine.min #map(%arg1)[%dim]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 32, 175, %2] [1, 1, 1, 1] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.ReduceMean(%extracted_slice) {
        axes_value = [1],
        keep_dims,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %inserted_slice = tensor.insert_slice %3 into %arg2[0, 0, 0, %arg1] [1, 1, 175, %2] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  }
  return %1 : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-DAG:    [[C103:%.+]] = arith.constant 103 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]])
// CHECK-SAME:      : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[LOOP:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C103]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:        -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-NEXT:      [[TILING_SIZE:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
// CHECK-DAG:       [[TILING_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[IDX]]] [1, 32, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:       [[MC_EMPTY:%.+]] = tensor.empty([[TILING_SIZE]])
// CHECK-SAME:            : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[MC_LOOP:%.+]] = scf.forall ([[MC_IDX:%.+]]) = (0) to (175) step (44) shared_outs([[MC_OUT:%.+]] = [[MC_EMPTY]])
// CHECK-SAME:            -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-DAG:          [[MC_H_SIZE:%.+]] = affine.min #[[$MAP1]]([[MC_IDX]])
// CHECK-DAG:          [[MC_SLICE:%.+]] = tensor.extract_slice [[TILING_SLICE]][0, 0, [[MC_IDX]], 0] [1, 32, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:              [[REDUCED:%.+]] = VPU.ReduceMean([[MC_SLICE]])
// CHECK-SAME:            -> tensor<1x1x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:         scf.forall.in_parallel {
// CHECK-NEXT:           tensor.parallel_insert_slice [[REDUCED]] into [[MC_OUT]][0, 0, [[MC_IDX]], 0] [1, 1, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:                into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[TILING_INSERT_SLICE:%.+]] = tensor.insert_slice [[MC_LOOP]] into [[OUT]][0, 0, 0, [[IDX]]] [1, 1, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:           into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      scf.yield [[TILING_INSERT_SLICE]]
// CHECK:        return [[LOOP]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 103)>

// CHECK: #[[MAP:.+]]= affine_map<(d0)[s0] -> (-d0 + s0, 103)>
// CHECK: #[[MAP1:.+]]= affine_map<(d0) -> (-d0 + 175, 44)>

module @test {
config.Resources 4 of @NCE at 6.000000e+02 MHz {
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @DynamicReduceSum
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
func.func @DynamicReduceSum(%arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> {
  %c103 = arith.constant 103 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %dim = tensor.dim %arg0, %c3 : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %0 = tensor.empty(%dim) : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %dim step %c103 iter_args(%arg2 = %0) -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
    %2 = affine.min #map(%arg1)[%dim]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 32, 175, %2] [1, 1, 1, 1] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.ReduceSum(%extracted_slice) {
        axes_value = [1],
        keep_dims,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %inserted_slice = tensor.insert_slice %3 into %arg2[0, 0, 0, %arg1] [1, 1, 175, %2] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  }
  return %1 : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-DAG:    [[C103:%.+]] = arith.constant 103 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]])
// CHECK-SAME:      : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[LOOP:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C103]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:        -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-NEXT:      [[TILING_SIZE:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
// CHECK-DAG:       [[TILING_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[IDX]]] [1, 32, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:       [[MC_EMPTY:%.+]] = tensor.empty([[TILING_SIZE]])
// CHECK-SAME:            : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[MC_LOOP:%.+]] = scf.forall ([[MC_IDX:%.+]]) = (0) to (175) step (44) shared_outs([[MC_OUT:%.+]] = [[MC_EMPTY]])
// CHECK-SAME:            -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-DAG:          [[MC_H_SIZE:%.+]] = affine.min #[[$MAP1]]([[MC_IDX]])
// CHECK-DAG:          [[MC_SLICE:%.+]] = tensor.extract_slice [[TILING_SLICE]][0, 0, [[MC_IDX]], 0] [1, 32, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:              [[REDUCED:%.+]] = VPU.ReduceSum([[MC_SLICE]])
// CHECK-SAME:            -> tensor<1x1x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:         scf.forall.in_parallel {
// CHECK-NEXT:           tensor.parallel_insert_slice [[REDUCED]] into [[MC_OUT]][0, 0, [[MC_IDX]], 0] [1, 1, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:                into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[TILING_INSERT_SLICE:%.+]] = tensor.insert_slice [[MC_LOOP]] into [[OUT]][0, 0, 0, [[IDX]]] [1, 1, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:           into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      scf.yield [[TILING_INSERT_SLICE]]
// CHECK:        return [[LOOP]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 103)>

// CHECK: #[[MAP:.+]]= affine_map<(d0)[s0] -> (-d0 + s0, 103)>
// CHECK: #[[MAP1:.+]]= affine_map<(d0) -> (-d0 + 175, 44)>

module @test {
config.Resources 4 of @NCE at 6.000000e+02 MHz {
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @DynamicReduceProd
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
func.func @DynamicReduceProd(%arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> {
  %c103 = arith.constant 103 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %dim = tensor.dim %arg0, %c3 : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %0 = tensor.empty(%dim) : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %dim step %c103 iter_args(%arg2 = %0) -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
    %2 = affine.min #map(%arg1)[%dim]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 32, 175, %2] [1, 1, 1, 1] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.ReduceProd(%extracted_slice) {
        axes_value = [1],
        keep_dims,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %inserted_slice = tensor.insert_slice %3 into %arg2[0, 0, 0, %arg1] [1, 1, 175, %2] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  }
  return %1 : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-DAG:    [[C103:%.+]] = arith.constant 103 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]])
// CHECK-SAME:      : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[LOOP:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C103]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:        -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-NEXT:      [[TILING_SIZE:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
// CHECK-DAG:       [[TILING_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[IDX]]] [1, 32, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:       [[MC_EMPTY:%.+]] = tensor.empty([[TILING_SIZE]])
// CHECK-SAME:            : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[MC_LOOP:%.+]] = scf.forall ([[MC_IDX:%.+]]) = (0) to (175) step (44) shared_outs([[MC_OUT:%.+]] = [[MC_EMPTY]])
// CHECK-SAME:            -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-DAG:          [[MC_H_SIZE:%.+]] = affine.min #[[$MAP1]]([[MC_IDX]])
// CHECK-DAG:          [[MC_SLICE:%.+]] = tensor.extract_slice [[TILING_SLICE]][0, 0, [[MC_IDX]], 0] [1, 32, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:              [[REDUCED:%.+]] = VPU.ReduceProd([[MC_SLICE]])
// CHECK-SAME:            -> tensor<1x1x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:         scf.forall.in_parallel {
// CHECK-NEXT:           tensor.parallel_insert_slice [[REDUCED]] into [[MC_OUT]][0, 0, [[MC_IDX]], 0] [1, 1, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:                into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[TILING_INSERT_SLICE:%.+]] = tensor.insert_slice [[MC_LOOP]] into [[OUT]][0, 0, 0, [[IDX]]] [1, 1, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:           into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      scf.yield [[TILING_INSERT_SLICE]]
// CHECK:        return [[LOOP]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 103)>

// CHECK: #[[MAP:.+]]= affine_map<(d0)[s0] -> (-d0 + s0, 103)>
// CHECK: #[[MAP1:.+]]= affine_map<(d0) -> (-d0 + 175, 44)>

module @test {
config.Resources 4 of @NCE at 6.000000e+02 MHz {
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @DynamicReduceLogicalOr
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
func.func @DynamicReduceLogicalOr(%arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> {
  %c103 = arith.constant 103 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %dim = tensor.dim %arg0, %c3 : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %0 = tensor.empty(%dim) : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %dim step %c103 iter_args(%arg2 = %0) -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
    %2 = affine.min #map(%arg1)[%dim]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 32, 175, %2] [1, 1, 1, 1] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.ReduceLogicalOr(%extracted_slice) {
        axes_value = [1],
        keep_dims,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %inserted_slice = tensor.insert_slice %3 into %arg2[0, 0, 0, %arg1] [1, 1, 175, %2] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  }
  return %1 : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-DAG:    [[C103:%.+]] = arith.constant 103 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]])
// CHECK-SAME:      : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[LOOP:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C103]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:        -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-NEXT:      [[TILING_SIZE:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
// CHECK-DAG:       [[TILING_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[IDX]]] [1, 32, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:       [[MC_EMPTY:%.+]] = tensor.empty([[TILING_SIZE]])
// CHECK-SAME:            : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[MC_LOOP:%.+]] = scf.forall ([[MC_IDX:%.+]]) = (0) to (175) step (44) shared_outs([[MC_OUT:%.+]] = [[MC_EMPTY]])
// CHECK-SAME:            -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-DAG:          [[MC_H_SIZE:%.+]] = affine.min #[[$MAP1]]([[MC_IDX]])
// CHECK-DAG:          [[MC_SLICE:%.+]] = tensor.extract_slice [[TILING_SLICE]][0, 0, [[MC_IDX]], 0] [1, 32, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:              [[REDUCED:%.+]] = VPU.ReduceLogicalOr([[MC_SLICE]])
// CHECK-SAME:            -> tensor<1x1x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:         scf.forall.in_parallel {
// CHECK-NEXT:           tensor.parallel_insert_slice [[REDUCED]] into [[MC_OUT]][0, 0, [[MC_IDX]], 0] [1, 1, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:                into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[TILING_INSERT_SLICE:%.+]] = tensor.insert_slice [[MC_LOOP]] into [[OUT]][0, 0, 0, [[IDX]]] [1, 1, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:           into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      scf.yield [[TILING_INSERT_SLICE]]
// CHECK:        return [[LOOP]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 103)>

// CHECK: #[[MAP:.+]]= affine_map<(d0)[s0] -> (-d0 + s0, 103)>
// CHECK: #[[MAP1:.+]]= affine_map<(d0) -> (-d0 + 175, 44)>

module @test {
config.Resources 4 of @NCE at 6.000000e+02 MHz {
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @DynamicReduceLogicalAnd
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
func.func @DynamicReduceLogicalAnd(%arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> {
  %c103 = arith.constant 103 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %dim = tensor.dim %arg0, %c3 : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %0 = tensor.empty(%dim) : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %dim step %c103 iter_args(%arg2 = %0) -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
    %2 = affine.min #map(%arg1)[%dim]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 32, 175, %2] [1, 1, 1, 1] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.ReduceLogicalAnd(%extracted_slice) {
        axes_value = [1],
        keep_dims,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
    %inserted_slice = tensor.insert_slice %3 into %arg2[0, 0, 0, %arg1] [1, 1, 175, %2] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
  }
  return %1 : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-DAG:    [[C103:%.+]] = arith.constant 103 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]])
// CHECK-SAME:      : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[LOOP:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C103]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:        -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-NEXT:      [[TILING_SIZE:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
// CHECK-DAG:       [[TILING_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[IDX]]] [1, 32, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:       [[MC_EMPTY:%.+]] = tensor.empty([[TILING_SIZE]])
// CHECK-SAME:            : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[MC_LOOP:%.+]] = scf.forall ([[MC_IDX:%.+]]) = (0) to (175) step (44) shared_outs([[MC_OUT:%.+]] = [[MC_EMPTY]])
// CHECK-SAME:            -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-DAG:          [[MC_H_SIZE:%.+]] = affine.min #[[$MAP1]]([[MC_IDX]])
// CHECK-DAG:          [[MC_SLICE:%.+]] = tensor.extract_slice [[TILING_SLICE]][0, 0, [[MC_IDX]], 0] [1, 32, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:              [[REDUCED:%.+]] = VPU.ReduceLogicalAnd([[MC_SLICE]])
// CHECK-SAME:            -> tensor<1x1x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 44, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:         scf.forall.in_parallel {
// CHECK-NEXT:           tensor.parallel_insert_slice [[REDUCED]] into [[MC_OUT]][0, 0, [[MC_IDX]], 0] [1, 1, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:                into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[TILING_INSERT_SLICE:%.+]] = tensor.insert_slice [[MC_LOOP]] into [[OUT]][0, 0, 0, [[IDX]]] [1, 1, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:           into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      scf.yield [[TILING_INSERT_SLICE]]
// CHECK:        return [[LOOP]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 103)>

// CHECK: #[[MAP:.+]]= affine_map<(d0)[s0] -> (-d0 + s0, 103)>
// CHECK: #[[MAP1:.+]]= affine_map<(d0) -> (-d0 + 175, 44)>

module @test {
config.Resources 4 of @NCE at 6.000000e+02 MHz {
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @DynamicReduceSquare
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>}>)
func.func @DynamicReduceSquare(%arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>}>) -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>}> {
  %c103 = arith.constant 103 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %dim = tensor.dim %arg0, %c3 : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>}>
  %0 = tensor.empty(%dim) : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>}>
  %1 = scf.for %arg1 = %c0 to %dim step %c103 iter_args(%arg2 = %0) -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>}>) {
    %2 = affine.min #map(%arg1)[%dim]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 32, 175, %2] [1, 1, 1, 1] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>}>
    %3 = VPU.ReduceSquare(%extracted_slice) {
        axes_value = [1],
        keep_dims,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>}> -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>}>
    %inserted_slice = tensor.insert_slice %3 into %arg2[0, 0, 0, %arg1] [1, 1, 175, %2] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>}> into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>}>
    scf.yield %inserted_slice : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>}>
  }
  return %1 : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>}>

// CHECK-DAG:    [[C103:%.+]] = arith.constant 103 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>}>
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]])
// CHECK-SAME:      : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>}>
// CHECK:        [[LOOP:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C103]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:        -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>}>) {
// CHECK-NEXT:      [[TILING_SIZE:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
// CHECK-DAG:       [[TILING_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[IDX]]] [1, 32, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>}>
// CHECK-DAG:       [[MC_EMPTY:%.+]] = tensor.empty([[TILING_SIZE]])
// CHECK-SAME:            : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>}>
// CHECK:           [[MC_LOOP:%.+]] = scf.forall ([[MC_IDX:%.+]]) = (0) to (175) step (44) shared_outs([[MC_OUT:%.+]] = [[MC_EMPTY]])
// CHECK-SAME:            -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>}>) {
// CHECK-DAG:          [[MC_H_SIZE:%.+]] = affine.min #[[$MAP1]]([[MC_IDX]])
// CHECK-DAG:          [[MC_SLICE:%.+]] = tensor.extract_slice [[TILING_SLICE]][0, 0, [[MC_IDX]], 0] [1, 32, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:            to tensor<1x32x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 44, 103]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:              [[REDUCED:%.+]] = VPU.ReduceSquare([[MC_SLICE]])
// CHECK-SAME:            -> tensor<1x1x?x103xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 44, 103]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-NEXT:         scf.forall.in_parallel {
// CHECK-NEXT:           tensor.parallel_insert_slice [[REDUCED]] into [[MC_OUT]][0, 0, [[MC_IDX]], 0] [1, 1, [[MC_H_SIZE]], 103] [1, 1, 1, 1]
// CHECK-SAME:                into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>}>
// CHECK:           [[TILING_INSERT_SLICE:%.+]] = tensor.insert_slice [[MC_LOOP]] into [[OUT]][0, 0, 0, [[IDX]]] [1, 1, 175, [[TILING_SIZE]]] [1, 1, 1, 1]
// CHECK-SAME:           into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>}>
// CHECK-NEXT:      scf.yield [[TILING_INSERT_SLICE]]
// CHECK:        return [[LOOP]]
}
}

// -----

// Uniform SOK split: 64 OC / 2 clusters = [32, 32]. No balanced arithmetic needed.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 2 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @NCEConvSOK_2Clusters_Uniform
func.func @NCEConvSOK_2Clusters_Uniform(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<64x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = VPU.NCE.Convolution(%arg0, %cst_0) rawFilterShape [64, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      strides = [1, 1]
  } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<64x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x64x64x64xf16, {order = #NHWC}>
  return %0 : tensor<1x64x64x64xf16, {order = #NHWC}>
}

// CHECK:       [[EMPTY:%.+]] = tensor.empty() : tensor<1x64x64x64xf16, {order = #NHWC}>
// CHECK:       [[FORALL:%.+]] = scf.forall ([[IV:%.+]]) = (0) to (64) step (32)
// CHECK-SAME:    shared_outs([[OUT:%.+]] = [[EMPTY]])
// CHECK:         [[W_SLICE:%.+]] = tensor.extract_slice {{%.+}}[[[IV]], 0, 0, 0] [32, 32, 1, 1]
// CHECK:         [[CONV:%.+]] = VPU.NCE.Convolution({{%.+}}, [[W_SLICE]])
// CHECK-NOT:         multiClusterStrategy
// CHECK-SAME:        rawFilterShape [32, 32, 1, 1]
// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[CONV]] into [[OUT]][0, [[IV]], 0, 0] [1, 32, 64, 64]
// CHECK:       return [[FORALL]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 3 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @NCEConvSOK_3Clusters
func.func @NCEConvSOK_3Clusters(%arg0: tensor<1x16x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = VPU.NCE.Convolution(%arg0, %cst_0) rawFilterShape [64, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
      ppe = #VPU.PPEStub<>,
      strides = [1, 1]
  } : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}>
      -> tensor<1x64x64x64xf16, {order = #NHWC}>
  return %0 : tensor<1x64x64x64xf16, {order = #NHWC}>
}

// CHECK-DAG:   [[C32:%.+]] = arith.constant 32 : index
// CHECK-DAG:   [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:   [[C16:%.+]] = arith.constant 16 : index
// CHECK:       [[EMPTY:%.+]] = tensor.empty() : tensor<1x64x64x64xf16, {order = #NHWC}>
// CHECK:       [[FORALL:%.+]] = scf.forall ([[IV:%.+]]) in (3)
// CHECK-SAME:    shared_outs([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:    -> (tensor<1x64x64x64xf16, {order = #NHWC}>)
// CHECK:         [[EXTRA:%.+]] = arith.minui [[IV]], [[C1]] : index
// CHECK:         [[BASE_OFF:%.+]] = arith.muli [[IV]], [[C16]] : index
// CHECK:         [[EXTRA_OFF:%.+]] = arith.muli [[EXTRA]], [[C16]] : index
// CHECK:         [[REAL_OFF:%.+]] = arith.addi [[BASE_OFF]], [[EXTRA_OFF]] : index
// CHECK:         [[IS_LARGE:%.+]] = arith.cmpi ult, [[IV]], [[C1]] : index
// CHECK:         [[REAL_SZ:%.+]] = arith.select [[IS_LARGE]], [[C32]], [[C16]] : index
// CHECK:         tensor.extract_slice {{%.+}}[[[REAL_OFF]], 0, 0, 0] [[[REAL_SZ]], 16, 3, 3]
// CHECK:         VPU.NCE.Convolution
// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice {{%.+}} into [[OUT]][0, [[REAL_OFF]], 0, 0] [1, [[REAL_SZ]], 64, 64]
// CHECK:       return [[FORALL]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 4 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @NCEConvSOK_4Clusters
func.func @NCEConvSOK_4Clusters(%arg0: tensor<1x16x64x64xf16, {order = #NHWC}>) -> tensor<1x96x64x64xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<96x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x16x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = VPU.NCE.Convolution(%arg0, %cst_0) rawFilterShape [96, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
      ppe = #VPU.PPEStub<>,
      strides = [1, 1]
  } : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<96x16x3x3xf16, {order = #NHWC}>
      -> tensor<1x96x64x64xf16, {order = #NHWC}>
  return %0 : tensor<1x96x64x64xf16, {order = #NHWC}>
}

// CHECK-DAG:   [[C32:%.+]] = arith.constant 32 : index
// CHECK-DAG:   [[C2:%.+]] = arith.constant 2 : index
// CHECK-DAG:   [[C16:%.+]] = arith.constant 16 : index
// CHECK:       [[EMPTY:%.+]] = tensor.empty() : tensor<1x96x64x64xf16, {order = #NHWC}>
// CHECK:       [[FORALL:%.+]] = scf.forall ([[IV:%.+]]) in (4)
// CHECK-SAME:    shared_outs([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:    -> (tensor<1x96x64x64xf16, {order = #NHWC}>)
// CHECK:         [[EXTRA:%.+]] = arith.minui [[IV]], [[C2]] : index
// CHECK:         [[BASE_OFF:%.+]] = arith.muli [[IV]], [[C16]] : index
// CHECK:         [[EXTRA_OFF:%.+]] = arith.muli [[EXTRA]], [[C16]] : index
// CHECK:         [[REAL_OFF:%.+]] = arith.addi [[BASE_OFF]], [[EXTRA_OFF]] : index
// CHECK:         [[IS_LARGE:%.+]] = arith.cmpi ult, [[IV]], [[C2]] : index
// CHECK:         [[REAL_SZ:%.+]] = arith.select [[IS_LARGE]], [[C32]], [[C16]] : index
// CHECK:         tensor.extract_slice {{%.+}}[[[REAL_OFF]], 0, 0, 0] [[[REAL_SZ]], 16, 3, 3]
// CHECK:         VPU.NCE.Convolution
// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice {{%.+}} into [[OUT]][0, [[REAL_OFF]], 0, 0] [1, [[REAL_SZ]], 64, 64]
// CHECK:       return [[FORALL]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 6)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1 ceildiv 6)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> ((d0 * 7) floordiv 8, 0)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<()[s0] -> (13, s0)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1) -> ((d0 * 7 + d1 * 7 - 7) ceildiv 8, 0)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0, d1) -> (-d0 + d1 + 1)>

// CHECK-LABEL:   @InterpolateSOH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x21x?x14xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 14, 14]> : tensor<4xsi64>, order = #NHWC}>
func.func @InterpolateSOH(%arg0: tensor<1x21x?x14xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 14, 14]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x21x?x10xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 16, 10]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>,
            nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0],
            pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3],
        initial_input_dims_attr = [1, 21, 14, 14],
        initial_output_dims_attr = [1, 21, 16, 10],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        scales_attr = [2.3571428571428572, 2.3571428571428572],
        sizes_attr = [16, 10],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
    } : tensor<1x21x?x14xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 14, 14]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x21x?x10xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 16, 10]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x21x?x10xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 16, 10]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-DAG:   [[CST_SCALE:%.+]] = arith.constant 2.3571428571428572 : f64
    // CHECK-DAG:   [[C2:%.+]] = arith.constant 2 : index
    // CHECK:       [[DIM:%.+]] = tensor.dim [[INPUT]], [[C2]]
    // CHECK:       [[DIM_I64:%.+]] = arith.index_cast [[DIM]] : index to i64
    // CHECK:       [[DIM_F64:%.+]] = arith.sitofp [[DIM_I64]] : i64 to f64
    // CHECK:       [[SCALED:%.+]] = arith.mulf [[DIM_F64]], [[CST_SCALE]] : f64
    // CHECK:       [[OUT_H_I64:%.+]] = arith.fptosi [[SCALED]] : f64 to i64
    // CHECK:       [[OUT_H:%.+]] = arith.index_cast [[OUT_H_I64]] : i64 to index

    // CHECK:       [[OUTPUT:%.+]] = tensor.empty([[OUT_H]])
    // CHECK-SAME:      : tensor<1x21x?x10xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 16, 10]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[MC_STEP:%.+]] = affine.apply #[[$MAP]]()[[[OUT_H]]]
    // CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to ([[OUT_H]]) step ([[MC_STEP]])
    // CHECK-SAME:      shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
    // CHECK-SAME:      -> (tensor<1x21x?x10xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 16, 10]> : tensor<4xsi64>, order = #NHWC}>)

    // CHECK:           [[OUT_TILE_SZ:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER]])[[[OUT_H]], [[OUT_H]]]
    // CHECK:           [[IN_OFFSET:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER]])
    // CHECK:           [[IN_OFFSET_CLAMPED:%.+]] = affine.min #[[$MAP3]]()[[[IN_OFFSET]]]
    // CHECK:           [[IN_END:%.+]] = affine.max #[[$MAP4]]([[LOOP_ITER]], [[OUT_TILE_SZ]])
    // CHECK:           [[IN_END_CLAMPED:%.+]] = affine.min #[[$MAP3]]()[[[IN_END]]]
    // CHECK:           [[IN_TILE_SZ:%.+]] = affine.apply #[[$MAP5]]([[IN_OFFSET_CLAMPED]], [[IN_END_CLAMPED]])

    // CHECK:           [[INPUT_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IN_OFFSET_CLAMPED]], 0] [1, 21, [[IN_TILE_SZ]], 14] [1, 1, 1, 1]
    // CHECK-SAME:          : tensor<1x21x?x14xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 14, 14]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:          to tensor<1x21x?x14xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 6, 14]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:           [[INTERP:%.+]] = VPU.Interpolate([[INPUT_SLICE]])
    // CHECK-SAME:          attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>
    // CHECK-SAME:          initial_input_dims_attr = [1, 21, 14, 14]
    // CHECK-SAME:          initial_input_offset_attr = [0, 0, 0, 0]
    // CHECK-SAME:          initial_output_dims_attr = [1, 21, 16, 10]
    // CHECK-SAME:          initial_output_offset_attr = [0, 0, 0, 0]
    // CHECK-SAME:          scales_attr = [1.1428571428571428, 0.7142857142857143]
    // CHECK-SAME:          : tensor<1x21x?x14xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 6, 14]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x21x?x10xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 6, 10]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:           scf.forall.in_parallel
    // CHECK:               tensor.parallel_insert_slice [[INTERP]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 21, [[OUT_TILE_SZ]], 10] [1, 1, 1, 1]
    // CHECK-SAME:              : tensor<1x21x?x10xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 6, 10]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:              into tensor<1x21x?x10xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 16, 10]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       return [[LOOP]] : tensor<1x21x?x10xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 16, 10]> : tensor<4xsi64>, order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @NCEConvSOK_6Clusters
func.func @NCEConvSOK_6Clusters(%arg0: tensor<1x16x64x64xf16, {order = #NHWC}>) -> tensor<1x128x64x64xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<128x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = VPU.NCE.Convolution(%arg0, %cst_0) rawFilterShape [128, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
      ppe = #VPU.PPEStub<>,
      strides = [1, 1]
  } : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<128x16x3x3xf16, {order = #NHWC}>
      -> tensor<1x128x64x64xf16, {order = #NHWC}>
  return %0 : tensor<1x128x64x64xf16, {order = #NHWC}>
}

// CHECK-DAG:   [[C32:%.+]] = arith.constant 32 : index
// CHECK-DAG:   [[C2:%.+]] = arith.constant 2 : index
// CHECK-DAG:   [[C16:%.+]] = arith.constant 16 : index
// CHECK:       [[EMPTY:%.+]] = tensor.empty() : tensor<1x128x64x64xf16, {order = #NHWC}>
// CHECK:       [[FORALL:%.+]] = scf.forall ([[IV:%.+]]) in (6)
// CHECK-SAME:    shared_outs([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:    -> (tensor<1x128x64x64xf16, {order = #NHWC}>)
// CHECK:         [[EXTRA:%.+]] = arith.minui [[IV]], [[C2]] : index
// CHECK:         [[BASE_OFF:%.+]] = arith.muli [[IV]], [[C16]] : index
// CHECK:         [[EXTRA_OFF:%.+]] = arith.muli [[EXTRA]], [[C16]] : index
// CHECK:         [[REAL_OFF:%.+]] = arith.addi [[BASE_OFF]], [[EXTRA_OFF]] : index
// CHECK:         [[IS_LARGE:%.+]] = arith.cmpi ult, [[IV]], [[C2]] : index
// CHECK:         [[REAL_SZ:%.+]] = arith.select [[IS_LARGE]], [[C32]], [[C16]] : index
// CHECK:         tensor.extract_slice
// CHECK:         VPU.NCE.Convolution
// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice
// CHECK:       return [[FORALL]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @NCEConvSOH_6Clusters
func.func @NCEConvSOH_6Clusters(%arg0: tensor<1x32x25x64xf16, {order = #NHWC}>) -> tensor<1x64x25x64xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = VPU.NCE.Convolution(%arg0, %cst_0) rawFilterShape [64, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
      ppe = #VPU.PPEStub<>,
      strides = [1, 1]
  } : tensor<1x32x25x64xf16, {order = #NHWC}>, tensor<64x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x64x25x64xf16, {order = #NHWC}>
  return %0 : tensor<1x64x25x64xf16, {order = #NHWC}>
}

// Height=25 across 6 clusters: even split with step=5
// forall (0) to (25) step (5), SOH tiling with pad computation
// CHECK:       [[EMPTY:%.+]] = tensor.empty() : tensor<1x64x25x64xf16, {order = #NHWC}>
// CHECK:       [[FORALL:%.+]] = scf.forall ([[IV:%.+]]) = (0) to (25) step (5)
// CHECK-SAME:    shared_outs([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:    -> (tensor<1x64x25x64xf16, {order = #NHWC}>)
// CHECK:         tensor.extract_slice
// CHECK:         tensor.pad
// CHECK:         VPU.NCE.Convolution
// CHECK:         tensor.cast
// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice
// CHECK:       return [[FORALL]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @NCEConvSplitOverBatch
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<6x32x16x16xf16, {order = #NHWC}>
func.func @NCEConvSplitOverBatch(%arg0: tensor<6x32x16x16xf16, {order = #NHWC}>) -> tensor<6x64x16x16xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [64, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
      ppe = #VPU.PPEStub<>,  strides = [1, 1]
  } : tensor<6x32x16x16xf16, {order = #NHWC}>, tensor<64x32x3x3xf16, {order = #NHWC}>
      -> tensor<6x64x16x16xf16, {order = #NHWC}>
  return %0 : tensor<6x64x16x16xf16, {order = #NHWC}>

// CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}>
// CHECK: [[OUTPUT:%.+]] = tensor.empty() : tensor<6x64x16x16xf16, {order = #NHWC}>
// CHECK: [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) in (6)
// CHECK-SAME:     shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:     -> (tensor<6x64x16x16xf16, {order = #NHWC}>)

// CHECK:       [[INPUT_SLICE:%.+]] = tensor.extract_slice [[INPUT]][[[LOOP_ITER]], 0, 0, 0] [1, 32, 16, 16] [1, 1, 1, 1]
// CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE]], [[WEIGHTS]])
// CHECK:       scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[CONV]] into [[LOOP_OUT]][[[LOOP_ITER]], 0, 0, 0] [1, 64, 16, 16] [1, 1, 1, 1]
// CHECK: return [[LOOP]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 30)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @SOBConvTileOverH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<6x32x64x64xf16, {order = #NHWC}>
func.func @SOBConvTileOverH(%arg0: tensor<6x32x64x64xf16, {order = #NHWC}>) -> tensor<6x256x64x64xf16, {order = #NHWC}> {
    %cst = arith.constant 0.000000e+00 : f16
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %out = tensor.empty() : tensor<6x256x64x64xf16, {order = #NHWC}>
    %tiling_loop = scf.for %out_offset_h = %c0 to %c64 step %c32 iter_args(%arg2 = %out) -> (tensor<6x256x64x64xf16, {order = #NHWC}>) {
      %in_offset_h = affine.max #map(%out_offset_h)
      %temp = affine.max #map1(%out_offset_h)
      %pad_top = affine.min #map2()[%temp]
      %temp0 = affine.max #map3(%in_offset_h)
      %pad_bottom = affine.min #map2()[%temp0]

      %extracted_slice = tensor.extract_slice %arg0[0, 0, %in_offset_h, 0] [6, 32, 33, 64] [1, 1, 1, 1]
          : tensor<6x32x64x64xf16, {order = #NHWC}> to tensor<6x32x33x64xf16, {order = #NHWC}>

      %padded = tensor.pad %extracted_slice low[0, 0, %pad_top, 1] high[0, 0, %pad_bottom, 1] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<6x32x33x64xf16, {order = #NHWC}> to tensor<6x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[6, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>

      %conv = VPU.NCE.Convolution(%padded, %weights) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,  strides = [1, 1]
      } : tensor<6x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[6, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<256x32x3x3xf16, {order = #NHWC}>
        -> tensor<6x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[6, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

      %cast = tensor.cast %conv : tensor<6x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[6, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
                                to tensor<6x256x32x64xf16, {order = #NHWC}>

      %inserted_slice = tensor.insert_slice %cast into %arg2[0, 0, %out_offset_h, 0] [6, 256, 32, 64] [1, 1, 1, 1]
        : tensor<6x256x32x64xf16, {order = #NHWC}> into tensor<6x256x64x64xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<6x256x64x64xf16, {order = #NHWC}>
    }
    return %tiling_loop : tensor<6x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG: [[CST_31:%.+]] = arith.constant 31 : index
    // CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<6x256x64x64xf16, {order = #NHWC}>
    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK-SAME:     [[TILE_LOOP_ITER:%[^:]+]] = {{.*}} to {{.*}} step
    // CHECK-SAME:     iter_args([[LOOP_OUT:%[^:]+]] = [[LOOP_OUTPUT]]) -> (tensor<6x256x64x64xf16, {order = #NHWC}>) {

    // CHECK:         [[SLICE_OFFSET:%.+]] = affine.max
    // CHECK:         [[DIFF1:%.+]] = affine.max
    // CHECK:         [[PAD_LOW:%.+]] = affine.min
    // CHECK:         [[DIFF2:%.+]] = affine.max
    // CHECK:         [[PAD_HIGH:%.+]] = affine.min

    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [6, 32, 33, 64] [1, 1, 1, 1]
    // CHECK-SAME:         tensor<6x32x64x64xf16, {order = #NHWC}> to tensor<6x32x33x64xf16, {order = #NHWC}>

    // CHECK:         [[TOTAL_PAD:%.+]] = affine.apply {{.+}}([[PAD_LOW]], [[PAD_HIGH]])
    // CHECK:         [[OUT_DIM_H_SZ:%.+]] = arith.addi [[TOTAL_PAD]], [[CST_31]] : index
    // CHECK:         [[MC_OUT:%.+]] = tensor.empty([[OUT_DIM_H_SZ]])
    // CHECK-SAME:        : tensor<6x256x?x64xf16

    // CHECK:         [[MC_LOOP:%.+]] = scf.forall ([[MC_LOOP_ITER:%.+]]) in (6)
    // CHECK-SAME:        shared_outs([[MC_LOOP_OUT:%.+]] = [[MC_OUT]])

    // Pad moved inside forall: extract batch=1 from unpadded H-slice, then pad
    // CHECK:             [[BATCH_SLICE:%.+]] = tensor.extract_slice [[SLICE]][[[MC_LOOP_ITER]],
    // CHECK-SAME:             tensor<6x32x33x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16

    // CHECK:             [[PAD:%.+]] = tensor.pad [[BATCH_SLICE]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    // CHECK:                tensor.yield [[PAD_VALUE]] : f16

    // CHECK:             [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[WEIGHTS]])
    // CHECK-SAME:            {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // CHECK:             scf.forall.in_parallel {
    // CHECK:                 tensor.parallel_insert_slice {{%.+}} into [[MC_LOOP_OUT]][[[MC_LOOP_ITER]], 0, 0, 0]

    // CHECK:         [[CAST:%.+]] = tensor.cast [[MC_LOOP]]
    // CHECK-SAME:       to tensor<6x256x32x64xf16, {order = #NHWC}>

    // CHECK:         [[INSERT:%.+]] = tensor.insert_slice [[CAST]] into [[LOOP_OUT]][0, 0, [[TILE_LOOP_ITER]], 0] [6, 256, 32, 64] [1, 1, 1, 1]
    // CHECK-SAME:       : tensor<6x256x32x64xf16, {order = #NHWC}> into tensor<6x256x64x64xf16, {order = #NHWC}>

    // CHECK: scf.yield [[INSERT]] : tensor<6x256x64x64xf16, {order = #NHWC}>
    // CHECK: return [[LOOP]] : tensor<6x256x64x64xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @NCEConvClustering
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x16x16xf16, {order = #NHWC}>
func.func @NCEConvClustering(%arg0: tensor<1x32x16x16xf16, {order = #NHWC}>) -> tensor<1x64x16x16xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [64, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
      ppe = #VPU.PPEStub<>,  strides = [1, 1]
  } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<64x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x64x16x16xf16, {order = #NHWC}>
  return %0 : tensor<1x64x16x16xf16, {order = #NHWC}>

// CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}>
// CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<6x64x16x16xf16, {order = #NHWC}>
// CHECK: [[FORALL:%.+]] = scf.forall ([[IV:%.+]]) in (6) shared_outs([[OUT:%.+]] = [[EMPTY]])
// CHECK:     [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])
// CHECK-NOT:     multiClusterStrategy
// CHECK-SAME:    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
// CHECK:     scf.forall.in_parallel {
// CHECK:         tensor.parallel_insert_slice [[CONV]] into [[OUT]][[[IV]], 0, 0, 0] [1, 64, 16, 16] [1, 1, 1, 1]
// CHECK: [[RESULT:%.+]] = tensor.extract_slice [[FORALL]][0, 0, 0, 0] [1, 64, 16, 16] [1, 1, 1, 1]
// CHECK: return [[RESULT]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 30)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @ClusteringConvTileOverH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @ClusteringConvTileOverH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %cst = arith.constant 0.000000e+00 : f16
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %out = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    %tiling_loop = scf.for %out_offset_h = %c0 to %c64 step %c32 iter_args(%arg2 = %out) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
      %in_offset_h = affine.max #map(%out_offset_h)
      %temp = affine.max #map1(%out_offset_h)
      %pad_top = affine.min #map2()[%temp]
      %temp0 = affine.max #map3(%in_offset_h)
      %pad_bottom = affine.min #map2()[%temp0]

      %extracted_slice = tensor.extract_slice %arg0[0, 0, %in_offset_h, 0] [1, 32, 33, 64] [1, 1, 1, 1]
          : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

      %padded = tensor.pad %extracted_slice low[0, 0, %pad_top, 1] high[0, 0, %pad_bottom, 1] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x33x64xf16, {order = #NHWC}> to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>

      %conv = VPU.NCE.Convolution(%padded, %weights) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,  strides = [1, 1]
      } : tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<256x32x3x3xf16, {order = #NHWC}>
        -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

      %cast = tensor.cast %conv : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
                                to tensor<1x256x32x64xf16, {order = #NHWC}>

      %inserted_slice = tensor.insert_slice %cast into %arg2[0, 0, %out_offset_h, 0] [1, 256, 32, 64] [1, 1, 1, 1]
        : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
    }
    return %tiling_loop : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}>

    // CHECK: [[LOOP:%.+]] = scf.for {{.*}} {
    // CHECK:     [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
    // CHECK:     [[PAD:%.+]] = tensor.pad [[SLICE]]
    // CHECK:         tensor.yield [[PAD_VALUE]] : f16

    // CHECK:     [[MC_OUT:%.+]] = tensor.empty
    // CHECK:     [[MC_LOOP:%.+]] = scf.forall ([[MC_IV:%.+]]) in (6) shared_outs([[MC_LOOP_OUT:%.+]] = [[MC_OUT]])
    // CHECK:         [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[WEIGHTS]])
    // CHECK-NOT:         multiClusterStrategy
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:         scf.forall.in_parallel {
    // CHECK:             tensor.parallel_insert_slice [[CONV]] into [[MC_LOOP_OUT]]{{.*}}[[[MC_IV]], 0, 0, 0]

    // CHECK:     [[MC_EXTRACT:%.+]] = tensor.extract_slice [[MC_LOOP]]
    // CHECK:     [[CAST:%.+]] = tensor.cast [[MC_EXTRACT]]
    // CHECK:     tensor.insert_slice [[CAST]]
    // CHECK: return [[LOOP]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// Balanced multiclustering: 128 OC across 6 clusters with alignment 16.
// 128 / 6 = 21.33, can't tile uniformly with alignment 16.
// Balanced algorithm distributes as [32, 32, 16, 16, 16, 16] (max=32)
// Verifies the divui+minui+muli+addi balanced pattern is emitted.

// CHECK-LABEL: @BalancedSOKConv128OC6Clusters
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x128x1x1xf16, {order = #NHWC}>
// CHECK-SAME:       [[WEIGHTS:%[^:]+]]: tensor<128x128x1x1xf16, {order = #NHWC}>
func.func @BalancedSOKConv128OC6Clusters(
    %arg0: tensor<1x128x1x1xf16, {order = #NHWC}>,
    %arg1: tensor<128x128x1x1xf16, {order = #NHWC}>
) -> tensor<1x128x1x1xf16, {order = #NHWC}> {
    %conv = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [128, 128, 1, 1] {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } : tensor<1x128x1x1xf16, {order = #NHWC}>,
        tensor<128x128x1x1xf16, {order = #NHWC}>
      -> tensor<1x128x1x1xf16, {order = #NHWC}>

    return %conv : tensor<1x128x1x1xf16, {order = #NHWC}>

// Balanced distribution: normalized loop (0 to 6 step 1), 6 iterations.
// numLarge=2 iterations get largeTile=32, remaining 4 get smallTile=16.
// CHECK-DAG:  [[C32:%.+]] = arith.constant 32 : index
// CHECK-DAG:  [[C16:%.+]] = arith.constant 16 : index
// CHECK-DAG:  [[C2:%.+]] = arith.constant 2 : index
// CHECK:      [[FORALL:%.+]] = scf.forall ([[IV:%.+]]) in (6)
// CHECK:          [[EXTRA:%.+]] = arith.minui [[IV]], [[C2]]
// CHECK:          [[BASE_OFF:%.+]] = arith.muli [[IV]], [[C16]]
// CHECK:          [[EXTRA_OFF:%.+]] = arith.muli [[EXTRA]], [[C16]]
// CHECK:          [[REAL_OFF:%.+]] = arith.addi [[BASE_OFF]], [[EXTRA_OFF]]
// CHECK:          [[IS_LARGE:%.+]] = arith.cmpi ult, [[IV]], [[C2]]
// CHECK:          [[REAL_SZ:%.+]] = arith.select [[IS_LARGE]], [[C32]], [[C16]]
// CHECK:          VPU.NCE.Convolution
// CHECK-NOT:          multiClusterStrategy
// CHECK:          scf.forall.in_parallel
// CHECK:      return [[FORALL]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map_tile = affine_map<(d0) -> (-d0 + 64, 32)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// Dynamic balanced multiclustering: 128 OC across 6 clusters with alignment 16.
// The op is inside scf.for (tiling over H), making H dynamic.
// The MC axis (C=128) is static but H is dynamic → shape is dynamic → dynamic fallback.
// Balanced distribution: numClusters=6, normalized loop (0 to 6 step 1).
// After canonicalization, balanced arithmetic folds to constants.

// CHECK-LABEL: @DynamicBalancedSOKConvInsideForLoop
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x128x64x64xf16, {order = #NHWC}>
// CHECK-SAME:       [[WEIGHTS:%[^:]+]]: tensor<128x128x1x1xf16, {order = #NHWC}>
func.func @DynamicBalancedSOKConvInsideForLoop(
    %arg0: tensor<1x128x64x64xf16, {order = #NHWC}>,
    %arg1: tensor<128x128x1x1xf16, {order = #NHWC}>
) -> tensor<1x128x64x64xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %out = tensor.empty() : tensor<1x128x64x64xf16, {order = #NHWC}>

    %tiling_loop = scf.for %iv = %c0 to %c64 step %c32 iter_args(%arg2 = %out) -> (tensor<1x128x64x64xf16, {order = #NHWC}>) {
      %tile_sz = affine.min #map_tile(%iv)

      %extracted_slice = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 128, %tile_sz, 64] [1, 1, 1, 1]
          : tensor<1x128x64x64xf16, {order = #NHWC}>
          to tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

      %conv = VPU.NCE.Convolution(%extracted_slice, %arg1) rawFilterShape [128, 128, 1, 1] {
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
          resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } : tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<128x128x1x1xf16, {order = #NHWC}>
        -> tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

      %cast = tensor.cast %conv
          : tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
          to tensor<1x128x32x64xf16, {order = #NHWC}>

      %inserted_slice = tensor.insert_slice %cast into %arg2[0, 0, %iv, 0] [1, 128, 32, 64] [1, 1, 1, 1]
          : tensor<1x128x32x64xf16, {order = #NHWC}> into tensor<1x128x64x64xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x128x64x64xf16, {order = #NHWC}>
    }
    return %tiling_loop : tensor<1x128x64x64xf16, {order = #NHWC}>

// Dynamic balanced: normalized loop `in (6)` form.
// With static C=128, balanced arithmetic folds to constants.
// CHECK-DAG:  [[C16:%.+]] = arith.constant 16 : index
// CHECK-DAG:  [[C2:%.+]] = arith.constant 2 : index
// CHECK-DAG:  [[C32:%.+]] = arith.constant 32 : index
// CHECK:      scf.for %
// CHECK:          [[FORALL:%.+]] = scf.forall ([[IV:%.+]]) in (6)
// CHECK:              [[EXTRA:%.+]] = arith.minui [[IV]], [[C2]]
// CHECK:              [[BASE_OFF:%.+]] = arith.muli [[IV]], [[C16]]
// CHECK:              [[EXTRA_OFF:%.+]] = arith.muli [[EXTRA]], [[C16]]
// CHECK:              [[REAL_OFF:%.+]] = arith.addi [[BASE_OFF]], [[EXTRA_OFF]]
// CHECK:              [[IS_LARGE:%.+]] = arith.cmpi ult, [[IV]], [[C2]]
// CHECK:              [[REAL_SZ:%.+]] = arith.select [[IS_LARGE]], [[C32]], [[C16]]
// CHECK:              VPU.NCE.Convolution
// CHECK-NOT:              multiClusterStrategy
// CHECK:              scf.forall.in_parallel
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map_tile = affine_map<(d0) -> (-d0 + 64, 32)>

// CHECK-DAG: #[[$TILE_MAP:.+]] = affine_map<(d0) -> (-d0 + 64, 32)>
// CHECK-DAG: #[[$MC_STEP:.+]] = affine_map<(d0) -> (d0 ceildiv 3)>
// CHECK-DAG: #[[$MC_TILE_SZ:.+]] = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 3)>

module {
config.Resources 3 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// Dynamic SOH: conv with 1x1 kernel inside a for loop that produces dynamic height tiles.
// The forall distributes the dynamic H uniformly across 3 clusters: (0) to (H) step (ceil(H/3)).
// CHECK-LABEL: @DynamicSOHConvInsideForLoop
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
// CHECK-SAME:       [[WEIGHTS:%[^:]+]]: tensor<256x32x1x1xf16, {order = #NHWC}>
func.func @DynamicSOHConvInsideForLoop(
    %arg0: tensor<1x32x64x64xf16, {order = #NHWC}>,
    %arg1: tensor<256x32x1x1xf16, {order = #NHWC}>
) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %out = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>

    %tiling_loop = scf.for %iv = %c0 to %c64 step %c32 iter_args(%arg2 = %out) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
      %tile_sz = affine.min #map_tile(%iv)

      %extracted_slice = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 32, %tile_sz, 64] [1, 1, 1, 1]
          : tensor<1x32x64x64xf16, {order = #NHWC}>
          to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

      %conv = VPU.NCE.Convolution(%extracted_slice, %arg1) rawFilterShape [256, 32, 1, 1] {
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
          resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<256x32x1x1xf16, {order = #NHWC}>
        -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

      %cast = tensor.cast %conv
          : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
          to tensor<1x256x32x64xf16, {order = #NHWC}>

      %inserted_slice = tensor.insert_slice %cast into %arg2[0, 0, %iv, 0] [1, 256, 32, 64] [1, 1, 1, 1]
          : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
    }
    return %tiling_loop : tensor<1x256x64x64xf16, {order = #NHWC}>

// CHECK:      scf.for
// CHECK:          [[TILE_SZ:%.+]] = affine.min #[[$TILE_MAP]]
// CHECK:          [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, {{%.+}}, 0] [1, 32, [[TILE_SZ]], 64]
// CHECK:          [[MC_OUT:%.+]] = tensor.empty([[TILE_SZ]])
// CHECK-SAME:         : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:          [[STEP:%.+]] = affine.apply #[[$MC_STEP]]([[TILE_SZ]])
// CHECK:          [[FORALL:%.+]] = scf.forall ([[IV:%.+]]) = (0) to ([[TILE_SZ]]) step ([[STEP]])
// CHECK-SAME:         shared_outs({{%.+}} = [[MC_OUT]])
// CHECK:              [[MC_TILE:%.+]] = affine.min #[[$MC_TILE_SZ]]([[IV]], [[TILE_SZ]]){{\[}}[[TILE_SZ]]]
// CHECK:              [[IN_TILE:%.+]] = tensor.extract_slice [[SLICE]][0, 0, [[IV]], 0] [1, 32, [[MC_TILE]], 64]
// CHECK:              [[CONV:%.+]] = VPU.NCE.Convolution([[IN_TILE]], [[WEIGHTS]])
// CHECK-NOT:              multiClusterStrategy
// CHECK-SAME:             : tensor<1x32x?x64xf16
// CHECK-SAME:             -> tensor<1x256x?x64xf16
// CHECK:              scf.forall.in_parallel
// CHECK:                  tensor.parallel_insert_slice [[CONV]] into {{%.+}}[0, 0, [[IV]], 0] [1, 256, [[MC_TILE]], 64]
// CHECK:          [[CAST:%.+]] = tensor.cast [[FORALL]]
// CHECK:          tensor.insert_slice [[CAST]] into
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// SOK tiling (scf.for over OC=256, step=128) + SOK multiclustering on the same axis.
// Each tile has 128 OC distributed across 6 clusters with alignment 16.
// Balanced: smallTile=16, numLarge=2, first 2 clusters get 32 OC, remaining 4 get 16 OC.
// This covers the same tiling-multiclustering axis scenario.

// CHECK-LABEL: @DynamicSOKConvSOKTilingInsideForLoop
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x32x32xf16, {order = #NHWC}>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<256x64x1x1xf16, {order = #NHWC}>)
func.func @DynamicSOKConvSOKTilingInsideForLoop(
    %arg0: tensor<1x64x32x32xf16, {order = #NHWC}>,
    %arg1: tensor<256x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x256x32x32xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %out = tensor.empty() : tensor<1x256x32x32xf16, {order = #NHWC}>

    %tiling_loop = scf.for %iv = %c0 to %c256 step %c128 iter_args(%arg2 = %out) -> (tensor<1x256x32x32xf16, {order = #NHWC}>) {
      %w_slice = tensor.extract_slice %arg1[%iv, 0, 0, 0] [128, 64, 1, 1] [1, 1, 1, 1]
          : tensor<256x64x1x1xf16, {order = #NHWC}>
          to tensor<128x64x1x1xf16, {order = #NHWC}>

      %conv = VPU.NCE.Convolution(%arg0, %w_slice) rawFilterShape [128, 64, 1, 1] {
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
          resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } : tensor<1x64x32x32xf16, {order = #NHWC}>,
          tensor<128x64x1x1xf16, {order = #NHWC}>
        -> tensor<1x128x32x32xf16, {order = #NHWC}>

      %inserted_slice = tensor.insert_slice %conv into %arg2[0, %iv, 0, 0] [1, 128, 32, 32] [1, 1, 1, 1]
          : tensor<1x128x32x32xf16, {order = #NHWC}> into tensor<1x256x32x32xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x32x32xf16, {order = #NHWC}>
    }
    return %tiling_loop : tensor<1x256x32x32xf16, {order = #NHWC}>
}

// CHECK:       [[FOR:%.+]] = scf.for [[IV:%.+]] = {{.+}} to {{.+}} step {{.+}} iter_args({{.+}})
// CHECK:         [[W_SLICE:%.+]] = tensor.extract_slice [[WEIGHTS]]{{\[}}[[IV]], 0, 0, 0] [128, 64, 1, 1]
// CHECK:         [[FORALL:%.+]] = scf.forall ([[MC_IV:%.+]]) in (6)
// CHECK:           [[NUM_LARGE:%.+]] = arith.minui [[MC_IV]], {{%.+}} : index
// CHECK:           [[BASE_OFF:%.+]] = arith.muli [[MC_IV]], {{%.+}} : index
// CHECK:           [[EXTRA_OFF:%.+]] = arith.muli [[NUM_LARGE]], {{%.+}} : index
// CHECK:           [[OFFSET:%.+]] = arith.addi [[BASE_OFF]], [[EXTRA_OFF]] : index
// CHECK:           [[IS_LARGE:%.+]] = arith.cmpi ult, [[MC_IV]], {{%.+}} : index
// CHECK:           [[TILE_SZ:%.+]] = arith.select [[IS_LARGE]], {{%.+}}, {{%.+}} : index
// CHECK:           [[FILTER_SLICE:%.+]] = tensor.extract_slice [[W_SLICE]]{{\[}}[[OFFSET]], 0, 0, 0] [[[TILE_SZ]], 64, 1, 1]
// CHECK:           [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_SLICE]])
// CHECK-NOT:           multiClusterStrategy
// CHECK-SAME:          rawFilterShape [[[TILE_SZ]], 64, 1, 1]
// CHECK-SAME:          -> tensor<1x?x32x32xf16
// CHECK:           scf.forall.in_parallel
// CHECK:             tensor.parallel_insert_slice [[CONV]] into {{%.+}}[0, [[OFFSET]], 0, 0] [1, [[TILE_SZ]], 32, 32]
// CHECK:         tensor.insert_slice [[FORALL]] into {{%.+}}[0, [[IV]], 0, 0] [1, 128, 32, 32]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: #[[$TILE_SZ_MAP:.+]] = affine_map<(d0) -> (-d0 + 128, 48)>

module {
config.Resources 3 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @NCEConvSOKStaticRawFilter
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @NCEConvSOKStaticRawFilter(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x128x64x64xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<128x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = VPU.NCE.Convolution(%arg0, %cst_0) rawFilterShape [128, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]
  } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<128x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x128x64x64xf16, {order = #NHWC}>
  return %0 : tensor<1x128x64x64xf16, {order = #NHWC}>
}
}

// CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<128x32x1x1xf16, {order = #NHWC}>
// CHECK: [[OUTPUT:%.+]] = tensor.empty() : tensor<1x128x64x64xf16, {order = #NHWC}>

// CHECK:       [[LOOP:%.+]] = scf.forall ([[LOOP_ITER:%.+]]) = (0) to (128) step (48) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])
// CHECK-SAME:      -> (tensor<1x128x64x64xf16, {order = #NHWC}>)
// CHECK:       [[TILE_SZ:%.+]] = affine.min #[[$TILE_SZ_MAP]]([[LOOP_ITER]])
// CHECK:       [[WEIGHTS_SLICE:%.+]] = tensor.extract_slice [[WEIGHTS]][[[LOOP_ITER]], 0, 0, 0] [[[TILE_SZ]], 32, 1, 1] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<128x32x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[128, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE]]) rawFilterShape [[[TILE_SZ]], 32, 1, 1] {
// CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:      ppe = #VPU.PPEStub<>,
// CHECK-SAME:      strides = [1, 1]}
// CHECK-SAME:    : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[128, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:    -> tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice [[CONV]] into [[LOOP_OUT]][0, [[LOOP_ITER]], 0, 0] [1, [[TILE_SZ]], 64, 64] [1, 1, 1, 1]
// CHECK-SAME:          : tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:          into tensor<1x128x64x64xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0) -> (-d0 + 256, 96)>

// CHECK-DAG: #[[$OUTER_MAP:.+]] = affine_map<(d0) -> (-d0 + 256, 96)>
// CHECK-DAG: #[[$INNER_STEP_MAP:.+]] = affine_map<(d0) -> ((d0 ceildiv 3 + 15) floordiv 16)>
// CHECK-DAG: #[[$INNER_TILE_MAP:.+]] = affine_map<(d0, d1)[s0] -> (-d0 + s0, (d1 ceildiv 3 + 15) floordiv 16)>

module {
config.Resources 3 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @SOKConvSOKTilingDynamicRawFilter
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SOKConvSOKTilingDynamicRawFilter(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<256x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x1x1xf16>, [#const.Reorder<#NHWC>]
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c96 = arith.constant 96 : index

  %out = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
  %result = scf.for %k = %c0 to %c256 step %c96 iter_args(%arg2 = %out) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
    %outer_tile_sz = affine.min #map(%k)

    %weights_tile = tensor.extract_slice %cst_0[%k, 0, 0, 0] [%outer_tile_sz, 32, 1, 1] [1, 1, 1, 1]
      : tensor<256x32x1x1xf16, {order = #NHWC}>
      to tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>

    // rawFilterShape is already dynamic from the outer SOK tiling.
    %conv = VPU.NCE.Convolution(%arg0, %weights_tile) rawFilterShape [%outer_tile_sz, 32, 1, 1] {
      resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>, strides = [1, 1]
    } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>, index
      -> tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    %insert = tensor.insert_slice %conv into %arg2[0, %k, 0, 0] [1, %outer_tile_sz, 64, 64] [1, 1, 1, 1]
      : tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
      into tensor<1x256x64x64xf16, {order = #NHWC}>
    scf.yield %insert : tensor<1x256x64x64xf16, {order = #NHWC}>
  }
  return %result : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x1x1xf16, {order = #NHWC}>
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[C256:%.+]] = arith.constant 256 : index
    // CHECK-DAG: [[C96:%.+]] = arith.constant 96 : index

    // CHECK: [[LOOP:%.+]] = scf.for [[K:%.+]] = [[C0]] to [[C256]] step [[C96]]
    // CHECK-SAME:     iter_args([[LOOP_OUT:%.+]] = {{.+}}) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {

    // CHECK:     [[OUTER_TILE:%.+]] = affine.min #[[$OUTER_MAP]]([[K]])

    // CHECK:     [[WEIGHTS_TILE:%.+]] = tensor.extract_slice [[WEIGHTS]][[[K]], 0, 0, 0] [[[OUTER_TILE]], 32, 1, 1] [1, 1, 1, 1]
    // CHECK-SAME:     : tensor<256x32x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:     to tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:     [[INNER_STEP:%.+]] = affine.apply #[[$INNER_STEP_MAP]]([[OUTER_TILE]])

    // CHECK:     [[MC_LOOP:%.+]] = scf.forall ([[MC_ITER:%.+]]) = (0) to ([[OUTER_TILE]]) step ([[INNER_STEP]])
    // CHECK-SAME:     shared_outs([[MC_OUT:%.+]] = {{[^)]+}}) -> (tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:         [[INNER_TILE:%.+]] = affine.min #[[$INNER_TILE_MAP]]([[MC_ITER]], [[OUTER_TILE]])[[[OUTER_TILE]]]

    // CHECK:         [[WEIGHTS_INNER:%.+]] = tensor.extract_slice [[WEIGHTS_TILE]][[[MC_ITER]], 0, 0, 0] [[[INNER_TILE]], 32, 1, 1] [1, 1, 1, 1]
    // CHECK-SAME:         : tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:         to tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:         [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_INNER]]) rawFilterShape [[[INNER_TILE]], 32, 1, 1] {
    // CHECK-SAME:         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:         ppe = #VPU.PPEStub<>,
    // CHECK-SAME:         strides = [1, 1]}
    // CHECK-SAME:       : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:       -> tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:         scf.forall.in_parallel {
    // CHECK:             tensor.parallel_insert_slice [[CONV]] into [[MC_OUT]][0, [[MC_ITER]], 0, 0] [1, [[INNER_TILE]], 64, 64] [1, 1, 1, 1]
    // CHECK-SAME:             : tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:             into tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:     [[INSERT:%.+]] = tensor.insert_slice [[MC_LOOP]] into [[LOOP_OUT]][0, [[K]], 0, 0] [1, [[OUTER_TILE]], 64, 64] [1, 1, 1, 1]
    // CHECK-SAME:     : tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:     into tensor<1x256x64x64xf16, {order = #NHWC}>
    // CHECK: scf.yield [[INSERT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
    // CHECK: return [[LOOP]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map_h = affine_map<(d0) -> (-d0 + 64, 32)>

// CHECK: #[[$MAP_H:.+]] = affine_map<(d0) -> (-d0 + 64, 32)>
// CHECK: #[[$MAP_K:.+]] = affine_map<(d0) -> (-d0 + 256, 96)>

module {
config.Resources 3 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @SOKConvNonSOKTilingStaticRawFilter
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SOKConvNonSOKTilingStaticRawFilter(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<256x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x1x1xf16>, [#const.Reorder<#NHWC>]
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index

  %out = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
  %result = scf.for %h = %c0 to %c64 step %c32 iter_args(%arg2 = %out) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
    %tile_h = affine.min #map_h(%h)

    %input_slice = tensor.extract_slice %arg0[0, 0, %h, 0] [1, 32, %tile_h, 64] [1, 1, 1, 1]
      : tensor<1x32x64x64xf16, {order = #NHWC}>
      to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // rawFilterShape is static: the outer tiling is over H, not C.
    %conv = VPU.NCE.Convolution(%input_slice, %cst_0) rawFilterShape [256, 32, 1, 1] {
      resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>, strides = [1, 1]
    } : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    %cast = tensor.cast %conv
      : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
      to tensor<1x256x32x64xf16, {order = #NHWC}>

    %insert = tensor.insert_slice %cast into %arg2[0, 0, %h, 0] [1, 256, 32, 64] [1, 1, 1, 1]
      : tensor<1x256x32x64xf16, {order = #NHWC}>
      into tensor<1x256x64x64xf16, {order = #NHWC}>
    scf.yield %insert : tensor<1x256x64x64xf16, {order = #NHWC}>
  }
  return %result : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x1x1xf16, {order = #NHWC}>
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[C64:%.+]] = arith.constant 64 : index
    // CHECK-DAG: [[C32:%.+]] = arith.constant 32 : index

    // CHECK: [[LOOP:%.+]] = scf.for [[H:%.+]] = [[C0]] to [[C64]] step [[C32]]
    // CHECK-SAME:     iter_args([[LOOP_OUT:%.+]] = {{.+}}) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {

    // CHECK:     [[TILE_H:%.+]] = affine.min #[[$MAP_H]]([[H]])
    // CHECK:     [[INPUT_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[H]], 0] [1, 32, [[TILE_H]], 64] [1, 1, 1, 1]
    // CHECK-SAME:     : tensor<1x32x64x64xf16, {order = #NHWC}>
    // CHECK-SAME:     to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:     [[MC_OUT:%.+]] = tensor.empty([[TILE_H]])
    // CHECK-SAME:     : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:     [[MC_LOOP:%.+]] = scf.forall ([[MC_ITER:%.+]]) = (0) to (256) step (96)
    // CHECK-SAME:     shared_outs([[MC_OUT_ARG:%.+]] = [[MC_OUT]])
    // CHECK-SAME:     -> (tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:         [[TILE_K:%.+]] = affine.min #[[$MAP_K]]([[MC_ITER]])

    // CHECK:         [[INNER_INPUT:%.+]] = tensor.extract_slice [[INPUT_SLICE]][0, 0, 0, 0] [1, 32, [[TILE_H]], 64] [1, 1, 1, 1]
    // CHECK-SAME:         : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:         to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:         [[WEIGHTS_TILE:%.+]] = tensor.extract_slice [[WEIGHTS]][[[MC_ITER]], 0, 0, 0] [[[TILE_K]], 32, 1, 1] [1, 1, 1, 1]
    // CHECK-SAME:         : tensor<256x32x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:         to tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:         [[CONV:%.+]] = VPU.NCE.Convolution([[INNER_INPUT]], [[WEIGHTS_TILE]]) rawFilterShape [[[TILE_K]], 32, 1, 1] {
    // CHECK-SAME:         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:         ppe = #VPU.PPEStub<>,
    // CHECK-SAME:         strides = [1, 1]}
    // CHECK-SAME:       : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME:         tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:       -> tensor<1x?x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:         scf.forall.in_parallel {
    // CHECK:             tensor.parallel_insert_slice {{.+}} into [[MC_OUT_ARG]][0, [[MC_ITER]], 0, 0] [1, [[TILE_K]], 64, 64] [1, 1, 1, 1]

    // CHECK:     [[CAST:%.+]] = tensor.cast [[MC_LOOP]]
    // CHECK-SAME:     : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:     to tensor<1x256x32x64xf16, {order = #NHWC}>

    // CHECK:     [[INSERT:%.+]] = tensor.insert_slice [[CAST]] into [[LOOP_OUT]][0, 0, [[H]], 0] [1, 256, 32, 64] [1, 1, 1, 1]
    // CHECK-SAME:     : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
    // CHECK: scf.yield [[INSERT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
    // CHECK: return [[LOOP]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0) -> (-d0 + 256, 96)>

// CHECK: #[[$OUTER_SOK_MAP:.+]] = affine_map<(d0) -> (-d0 + 256, 96)>
// CHECK: #[[$INNER_SOH_MAP:.+]] = affine_map<(d0) -> (-d0 + 64, 22)>

module {
config.Resources 3 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @NonSOKConvSOKTilingDynamicRawFilter
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @NonSOKConvSOKTilingDynamicRawFilter(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<256x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x1x1xf16>, [#const.Reorder<#NHWC>]
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c96 = arith.constant 96 : index

  %out = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
  %result = scf.for %k = %c0 to %c256 step %c96 iter_args(%arg2 = %out) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
    %outer_tile_sz = affine.min #map(%k)

    %weights_tile = tensor.extract_slice %cst_0[%k, 0, 0, 0] [%outer_tile_sz, 32, 1, 1] [1, 1, 1, 1]
      : tensor<256x32x1x1xf16, {order = #NHWC}>
      to tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>

    // rawFilterShape is already dynamic from the outer SOK tiling.
    // The MC strategy is SplitOverHeight, so the OC dimension is NOT re-tiled.
    %conv = VPU.NCE.Convolution(%arg0, %weights_tile) rawFilterShape [%outer_tile_sz, 32, 1, 1] {
      resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>, strides = [1, 1]
    } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>, index
      -> tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    %insert = tensor.insert_slice %conv into %arg2[0, %k, 0, 0] [1, %outer_tile_sz, 64, 64] [1, 1, 1, 1]
      : tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
      into tensor<1x256x64x64xf16, {order = #NHWC}>
    scf.yield %insert : tensor<1x256x64x64xf16, {order = #NHWC}>
  }
  return %result : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x1x1xf16, {order = #NHWC}>
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[C256:%.+]] = arith.constant 256 : index
    // CHECK-DAG: [[C96:%.+]] = arith.constant 96 : index

    // CHECK: [[LOOP:%.+]] = scf.for [[K:%.+]] = [[C0]] to [[C256]] step [[C96]]
    // CHECK-SAME:     iter_args([[LOOP_OUT:%.+]] = {{.+}}) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {

    // CHECK:     [[OUTER_TILE:%.+]] = affine.min #[[$OUTER_SOK_MAP]]([[K]])

    // CHECK:     [[WEIGHTS_TILE:%.+]] = tensor.extract_slice [[WEIGHTS]][[[K]], 0, 0, 0] [[[OUTER_TILE]], 32, 1, 1] [1, 1, 1, 1]
    // CHECK-SAME:     : tensor<256x32x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:     to tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:     [[MC_LOOP:%.+]] = scf.forall ([[MC_ITER:%.+]]) = (0) to (64) step (22)
    // CHECK-SAME:     shared_outs([[MC_OUT:%.+]] = {{[^)]+}}) -> (tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:         [[TILE_H:%.+]] = affine.min #[[$INNER_SOH_MAP]]([[MC_ITER]])
    // CHECK:         [[INPUT_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[MC_ITER]], 0] [1, 32, [[TILE_H]], 64] [1, 1, 1, 1]

    // CHECK:         [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE]], [[WEIGHTS_TILE]]) rawFilterShape [[[OUTER_TILE]], 32, 1, 1] {
    // CHECK-SAME:         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:         ppe = #VPU.PPEStub<>,
    // CHECK-SAME:         strides = [1, 1]}
    // CHECK-SAME:       : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME:         tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:       -> tensor<1x?x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:         scf.forall.in_parallel {
    // CHECK:             tensor.parallel_insert_slice {{.+}} into [[MC_OUT]][0, 0, [[MC_ITER]], 0] [1, 256, [[TILE_H]], 64] [1, 1, 1, 1]

    // CHECK:     [[INSERT:%.+]] = tensor.insert_slice [[MC_LOOP]] into [[LOOP_OUT]][0, [[K]], 0, 0] [1, [[OUTER_TILE]], 64, 64] [1, 1, 1, 1]
    // CHECK-SAME:     : tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:     into tensor<1x256x64x64xf16, {order = #NHWC}>
    // CHECK: scf.yield [[INSERT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
    // CHECK: return [[LOOP]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}
}