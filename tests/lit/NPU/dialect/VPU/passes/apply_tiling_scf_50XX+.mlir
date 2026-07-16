//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true enable-is-reduce-supported" --apply-tiling="enable-scf-tiling=true" --cse --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$OUT_OFFSET_AND_SIZE_MAP:.+]] = affine_map<(d0) -> ((d0 floordiv 103) * 103, (d0 floordiv 103) * 102 + 2)>
// CHECK: #[[$TILE_INDEX_CAP_AND_REMAINDER_MAP:.+]] = affine_map<(d0) -> (d0 floordiv 103, 2)>
// CHECK: #[[$SIZE_BY_CAPPED_TILE_INDEX_MAP:.+]] = affine_map<(d0) -> (-d0 + 104, 103)>

// CHECK-LABEL: @ApplyTilingNCEReduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x512xf16, {order = #NHWC}>
func.func @ApplyTilingNCEReduce(%arg0 : tensor<1x32x175x512xf16, {order = #NHWC}>) -> tensor<1x1x175x512xf16, {order = #NHWC}> {
   %0 = VPU.NCE.Reduce(%arg0) {
     axes = [1], input_padding = [0, 12, 0, 0],
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
     op_type = #VPU.reduce_type<SUM>, ppe = #VPU.PPEFp<mode = <NOOP>,
     clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 5]
   } -> tensor<1x1x175x512xf16, {order = #NHWC}>
   return %0 : tensor<1x1x175x512xf16, {order = #NHWC}>

   // CHECK-DAG:  [[EMPT:%.+]] = tensor.empty() : tensor<1x1x175x512xf16, {order = #NHWC}>
   // CHECK-DAG:  [[C0:%.+]] = arith.constant 0 : index
   // CHECK-DAG:  [[C512:%.+]] = arith.constant 512 : index
   // CHECK-DAG:  [[C103:%.+]] = arith.constant 103 : index
   // CHECK:      [[REDUCED:%.+]] = scf.for [[ITER:%.+]] = [[C0]] to [[C512]] step [[C103]] iter_args([[ARG2:%.+]] = [[EMPT]]) -> (tensor<1x1x175x512xf16, {order = #NHWC}>) {
  // CHECK:         [[OFFSET:%.+]] = affine.min #[[$OUT_OFFSET_AND_SIZE_MAP]]([[ITER]])
  // CHECK:         [[TILE_INDEX_CAP:%.+]] = affine.min #[[$TILE_INDEX_CAP_AND_REMAINDER_MAP]]([[ITER]])
  // CHECK:         [[SIZE:%.+]] = affine.min #[[$SIZE_BY_CAPPED_TILE_INDEX_MAP]]([[TILE_INDEX_CAP]])
   // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[OFFSET]]] [1, 32, 175, [[SIZE]]] [1, 1, 1, 1] : tensor<1x32x175x512xf16, {order = #NHWC}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
   // CHECK:         [[REDUCED_SLICE:%.+]] = VPU.NCE.Reduce([[SLICE]])
   // CHECK:         [[INSERT_SLICE:%.+]] = tensor.insert_slice [[REDUCED_SLICE]] into [[ARG2]][0, 0, 0, [[OFFSET]]] [1, 1, 175, [[SIZE]]] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x512xf16, {order = #NHWC}>
   // CHECK:         scf.yield [[INSERT_SLICE]] : tensor<1x1x175x512xf16, {order = #NHWC}>
   // CHECK-NEXT: }
   // CHECK:      return [[REDUCED]] : tensor<1x1x175x512xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$MAP_DYN:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 103)>

// CHECK-LABEL: @ApplyTilingDynamicNCEReduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
func.func @ApplyTilingDynamicNCEReduce(%arg0 : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}> {
   %0 = VPU.NCE.Reduce(%arg0) {
     axes = [1], input_padding = [0, 12, 0, 0],
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
     op_type = #VPU.reduce_type<SUM>, ppe = #VPU.PPEFp<mode = <NOOP>,
     clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 5]
   } -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
   return %0 : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-DAG:    [[C103:%.+]] = arith.constant 103 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C103]] iter_args([[ARG2:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK-NEXT:      [[SIZE:%.+]] = affine.min #[[$MAP_DYN]]([[IDX]])[[[DIM]]]
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[IDX]]] [1, 32, 175, [[SIZE]]] [1, 1, 1, 1] : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      [[REDUCE:%.+]] = VPU.NCE.Reduce([[SLICE]])
// CHECK-SAME:          -> tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into [[ARG2]][0, 0, 0, [[IDX]]] [1, 1, 175, [[SIZE]]] [1, 1, 1, 1] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 103]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:       scf.yield [[INSERT]] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        return [[SCF]] : tensor<1x1x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-DAG: #[[$MAP_TILE_H:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 360)>
// CHECK-DAG: #[[$MAP_TILE_W:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 428)>
// CHECK-DAG: #[[$MAP_DIV2:.*]] = affine_map<(d0) -> (d0 floordiv 2)>

// CHECK-LABEL: @ApplyTilingDynamicYuvToRgbI420
// CHECK-SAME:  [[ARG0:%arg[0-9]]]: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 1]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-SAME:  [[ARG1:%arg[0-9]]]: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 1]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-SAME:  [[ARG2:%arg[0-9]]]: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 1]> : tensor<4xsi64>, order = #NCHW}>
func.func @ApplyTilingDynamicYuvToRgbI420(
    %arg0: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg1: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg2: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 1]> : tensor<4xsi64>, order = #NCHW}>)
        -> tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = VPU.YuvToRgb(%arg0, %arg1, %arg2) {
        inFmt = #IE.color_fmt<I420>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
        operandSegmentSizes = array<i32: 1, 1, 1>,
        outFmt = #IE.color_fmt<BGR>,
        tilingStrategy = [1, 4, 6, 1]
    } : tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 1]> : tensor<4xsi64>, order = #NCHW}>,
        tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 1]> : tensor<4xsi64>, order = #NCHW}>,
        tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 1]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-DAG:    [[C360:%.+]] = arith.constant 360 : index
// CHECK-DAG:    [[C428:%.+]] = arith.constant 428 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
// CHECK-DAG:    [[DIM_H:%.+]] = tensor.dim [[ARG0]], [[C1]]
// CHECK-DAG:    [[DIM_W:%.+]] = tensor.dim [[ARG0]], [[C2]]
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:        [[OUTER:%.+]] = scf.for [[IDX_H:%.+]] = [[C0]] to [[DIM_H]] step [[C360]] iter_args([[OUT_H:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}>) {
// CHECK:            [[INNER:%.+]] = scf.for [[IDX_W:%.+]] = [[C0]] to [[DIM_W]] step [[C428]] iter_args([[OUT_W:%.+]] = [[OUT_H]])
// CHECK-SAME:              -> (tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}>) {
// CHECK:                [[TILE_H:%.+]] = affine.min #[[$MAP_TILE_H]]([[IDX_H]])[[[DIM_H]]]
// CHECK:                [[TILE_W:%.+]] = affine.min #[[$MAP_TILE_W]]([[IDX_W]])[[[DIM_W]]]
// CHECK:                [[UV_OFF_H:%.+]] = affine.apply #[[$MAP_DIV2]]([[IDX_H]])
// CHECK:                [[UV_SIZE_H:%.+]] = affine.apply #[[$MAP_DIV2]]([[TILE_H]])
// CHECK:                [[UV_OFF_W:%.+]] = affine.apply #[[$MAP_DIV2]]([[IDX_W]])
// CHECK:                [[UV_SIZE_W:%.+]] = affine.apply #[[$MAP_DIV2]]([[TILE_W]])
// CHECK:                [[SLICE_Y:%.+]] = tensor.extract_slice [[ARG0]][0, [[IDX_H]], [[IDX_W]], 0] [1, [[TILE_H]], [[TILE_W]], 1] [1, 1, 1, 1]
// CHECK-SAME:              to tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 360, 428, 1]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:                [[SLICE_U:%.+]] = tensor.extract_slice [[ARG1]][0, [[UV_OFF_H]], [[UV_OFF_W]], 0] [1, [[UV_SIZE_H]], [[UV_SIZE_W]], 1] [1, 1, 1, 1]
// CHECK-SAME:              to tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 180, 214, 1]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:                [[SLICE_V:%.+]] = tensor.extract_slice [[ARG2]][0, [[UV_OFF_H]], [[UV_OFF_W]], 0] [1, [[UV_SIZE_H]], [[UV_SIZE_W]], 1] [1, 1, 1, 1]
// CHECK-SAME:              to tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 180, 214, 1]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:                [[YUV:%.+]] = VPU.YuvToRgb([[SLICE_Y]], [[SLICE_U]], [[SLICE_V]])
// CHECK-SAME:              {inFmt = #IE.color_fmt<I420>
// CHECK-SAME:              outFmt = #IE.color_fmt<BGR>
// CHECK-SAME:              -> tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 360, 428, 3]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[YUV]] into [[OUT_W]][0, [[IDX_H]], [[IDX_W]], 0] [1, [[TILE_H]], [[TILE_W]], 3] [1, 1, 1, 1]
// CHECK-SAME:              : tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 360, 428, 3]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:                scf.yield [[INSERT]]
// CHECK:            scf.yield [[INNER]]
// CHECK:        return [[OUTER]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-DAG: #[[$MAP_TILE_H:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 360)>
// CHECK-DAG: #[[$MAP_TILE_W:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 428)>
// CHECK-DAG: #[[$MAP_DIV2:.*]] = affine_map<(d0) -> (d0 floordiv 2)>

// CHECK-LABEL: @ApplyTilingDynamicYuvToRgbNV12
// CHECK-SAME:  [[ARG0:%arg[0-9]]]: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 1]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-SAME:  [[ARG1:%arg[0-9]]]: tensor<1x?x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 2]> : tensor<4xsi64>, order = #NCHW}>
func.func @ApplyTilingDynamicYuvToRgbNV12(
    %arg0: tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg1: tensor<1x?x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 2]> : tensor<4xsi64>, order = #NCHW}>)
        -> tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = VPU.YuvToRgb(%arg0, %arg1) {
        inFmt = #IE.color_fmt<NV12>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
        operandSegmentSizes = array<i32: 1, 1, 0>,
        outFmt = #IE.color_fmt<RGB>,
        tilingStrategy = [1, 4, 6, 1]
    } : tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 1]> : tensor<4xsi64>, order = #NCHW}>,
        tensor<1x?x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 720, 1280, 2]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-DAG:    [[C360:%.+]] = arith.constant 360 : index
// CHECK-DAG:    [[C428:%.+]] = arith.constant 428 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
// CHECK-DAG:    [[DIM_H:%.+]] = tensor.dim [[ARG0]], [[C1]]
// CHECK-DAG:    [[DIM_W:%.+]] = tensor.dim [[ARG0]], [[C2]]
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:        [[OUTER:%.+]] = scf.for [[IDX_H:%.+]] = [[C0]] to [[DIM_H]] step [[C360]] iter_args([[OUT_H:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}>) {
// CHECK:            [[INNER:%.+]] = scf.for [[IDX_W:%.+]] = [[C0]] to [[DIM_W]] step [[C428]] iter_args([[OUT_W:%.+]] = [[OUT_H]])
// CHECK-SAME:              -> (tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}>) {
// CHECK:                [[TILE_H:%.+]] = affine.min #[[$MAP_TILE_H]]([[IDX_H]])[[[DIM_H]]]
// CHECK:                [[TILE_W:%.+]] = affine.min #[[$MAP_TILE_W]]([[IDX_W]])[[[DIM_W]]]
// CHECK:                [[UV_OFF_H:%.+]] = affine.apply #[[$MAP_DIV2]]([[IDX_H]])
// CHECK:                [[UV_SIZE_H:%.+]] = affine.apply #[[$MAP_DIV2]]([[TILE_H]])
// CHECK:                [[UV_OFF_W:%.+]] = affine.apply #[[$MAP_DIV2]]([[IDX_W]])
// CHECK:                [[UV_SIZE_W:%.+]] = affine.apply #[[$MAP_DIV2]]([[TILE_W]])
// CHECK:                [[SLICE_Y:%.+]] = tensor.extract_slice [[ARG0]][0, [[IDX_H]], [[IDX_W]], 0] [1, [[TILE_H]], [[TILE_W]], 1] [1, 1, 1, 1]
// CHECK-SAME:              to tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 360, 428, 1]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:                [[SLICE_UV:%.+]] = tensor.extract_slice [[ARG1]][0, [[UV_OFF_H]], [[UV_OFF_W]], 0] [1, [[UV_SIZE_H]], [[UV_SIZE_W]], 2] [1, 1, 1, 1]
// CHECK-SAME:              to tensor<1x?x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 180, 214, 2]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:                [[YUV:%.+]] = VPU.YuvToRgb([[SLICE_Y]], [[SLICE_UV]])
// CHECK-SAME:              {inFmt = #IE.color_fmt<NV12>
// CHECK-SAME:              outFmt = #IE.color_fmt<RGB>
// CHECK-SAME:              -> tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 360, 428, 3]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[YUV]] into [[OUT_W]][0, [[IDX_H]], [[IDX_W]], 0] [1, [[TILE_H]], [[TILE_W]], 3] [1, 1, 1, 1]
// CHECK-SAME:              : tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 360, 428, 3]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 1440, 2560, 3]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:                scf.yield [[INSERT]]
// CHECK:            scf.yield [[INNER]]
// CHECK:        return [[OUTER]]
}
