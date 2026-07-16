//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --apply-tiling="enable-scf-tiling=true" --cse --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (0, d0 - 30)>

// CHECK-LABEL:   @ApplyTilingNCEConv
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @ApplyTilingNCEConv(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    //CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 64 : index
    //CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 32 : index
    //CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {

    //CHECK:                [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP]]([[LOOP_ITER]])
    //CHECK:                [[DIFF1:%.+]] = affine.max #[[$MAP1]]([[ARG_1:%[^)]+]])
    //CHECK:                [[PAD_LOW:%.+]] = affine.min #[[$MAP2]]()[[[DIFF1]]]
    //CHECK:                [[DIFF2:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET]])
    //CHECK:                [[PAD_HIGH:%.+]] = affine.min #[[$MAP2]]()[[[DIFF2]]]

    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
    //CHECK:                [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    //CHECK:                   tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                   tensor<1x32x33x64xf16, {order = #NHWC}> to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[WEIGHTS]]) rawFilterShape [256, 32, 3, 3] 
    //CHECK-SAME:           {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK-SAME:           -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:               [[CAST:%.+]] = tensor.cast [[CONV]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x256x32x64xf16, {order = #NHWC}>

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[CAST]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK: scf.yield [[INSERT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!SparseType = !VPU.SparseTensor<data=tensor<1280x16x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1280x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<1280xi64>, alignment = 16 : i64>>

// CHECK-LABEL:   @NoApplyTilingSparseNCEConv
func.func @NoApplyTilingSparseNCEConv(%arg0: tensor<1x16x64x16xf16, {order = #NHWC}>, %arg1: !SparseType) -> tensor<1x1280x64x16xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [1280, 16, 3, 3]  {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
        adder = 0.000000e+00 : f64>,strides = [1, 1],
        tilingStrategy = [1, 8, 1, 1]} : tensor<1x16x64x16xf16, {order = #NHWC}>,
        !SparseType ->
        tensor<1x1280x64x16xf16, {order = #NHWC}>

    return %0 : tensor<1x1280x64x16xf16, {order = #NHWC}>

    //CHECK-NOT: scf.for
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NoApplyTilingNotTiledOp
func.func @NoApplyTilingNotTiledOp(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %arg1: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {

    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
      tilingStrategy = [1, 1, 1, 1]} -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    //CHECK-NOT: scf.for
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NoApplyTilingSliceOp
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x64x100x100xf16, {order = #NHWC}>
func.func @NoApplyTilingSliceOp(%arg0: tensor<1x64x100x100xf16, {order = #NHWC}>)
    -> tensor<1x64x50x100xf16, {order = #NHWC}> {
    %0 = VPU.Slice %arg0 [0, 0, 25, 0] [1, 64, 50, 100] : tensor<1x64x100x100xf16, {order = #NHWC}> to tensor<1x64x50x100xf16, {order = #NHWC}>
    return %0 : tensor<1x64x50x100xf16, {order = #NHWC}>

    // CHECK-NOT: scf.for
    // CHECK: [[SLICE:%.+]] = VPU.Slice [[ARG0]] [0, 0, 25, 0] [1, 64, 50, 100]
    // CHECK: return [[SLICE]] : tensor<1x64x50x100xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK: #[[$W_OFFSET_AND_SIZE_MAP:.+]] = affine_map<(d0) -> ((d0 floordiv 5) * 5, (d0 floordiv 5) * 4 + 4)>
//CHECK: #[[$W_TILE_INDEX_CAP_AND_REMAINDER_MAP:.+]] = affine_map<(d0) -> (d0 floordiv 5, 4)>
//CHECK: #[[$W_SIZE_BY_CAPPED_TILE_INDEX_MAP:.+]] = affine_map<(d0) -> (-d0 + 8, 5)>
//CHECK: #[[$C_SIZE_BY_ITER_MAP:.+]] = affine_map<(d0) -> (-d0 + 640, 96)>
//CHECK: #[[$OFFSET_TO_SLICE_OFFSET_MAP:.+]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$OFFSET_TO_PAD_CANDIDATE_MAP:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$PAD_CLAMP_MAP:.+]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$PAD_HIGH_CANDIDATE_MAP:.+]] = affine_map<(d0, d1) -> (0, d0 + d1 - 30)>
//CHECK: #[[$SLICE_SIZE_BY_OUT_AND_PAD_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>

// CHECK-LABEL:   @ApplyChannelUnevenTiling
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x640x32x32xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<640x640x3x3xf16, {order = #NHWC}>)
func.func @ApplyChannelUnevenTiling(%arg0: tensor<1x640x32x32xf16, {order = #NHWC}>,
%arg1: tensor<640x640x3x3xf16, {order = #NHWC}>)
        -> tensor<1x640x32x32xf16, {order = #NHWC}> {

    %0 = VPU.NCE.Convolution(%arg0, %arg1)  rawFilterShape [640, 640, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
        prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
        strides = [1, 1], tilingStrategy = [1, 7, 1, 7]
        }
        : tensor<1x640x32x32xf16, {order = #NHWC}>,
        tensor<640x640x3x3xf16, {order = #NHWC}>
        -> tensor<1x640x32x32xf16, {order = #NHWC}>

    return %0: tensor<1x640x32x32xf16, {order = #NHWC}>

    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_END_C:%.+]] = arith.constant 640 : index
    //CHECK-DAG: [[LOOP_END_W:%.+]] = arith.constant 32 : index
    //CHECK-DAG: [[LOOP_STEP_C:%.+]]  = arith.constant 96 : index
    //CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant 5 : index
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x640x32x32xf16, {order = #NHWC}>

    //CHECK: [[LOOP_C:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER_C:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_C]] step [[LOOP_STEP_C]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x640x32x32xf16, {order = #NHWC}>)

    //CHECK: [[LOOP_W:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    //CHECK-SAME:           iter_args([[LOOP_OUT_W:%arg[0-9]]]  = [[LOOP_OUT]]) -> (tensor<1x640x32x32xf16, {order = #NHWC}>)

    //CHECK:                [[CORRECTED_OFFSET_W:%.+]] = affine.min #[[$W_OFFSET_AND_SIZE_MAP]]([[LOOP_ITER_W]])
    //CHECK:                [[TILE_INDEX_W_CAP:%.+]] = affine.min #[[$W_TILE_INDEX_CAP_AND_REMAINDER_MAP]]([[LOOP_ITER_W]])
    //CHECK:                [[CORRECTED_SIZE_W:%.+]] = affine.min #[[$W_SIZE_BY_CAPPED_TILE_INDEX_MAP]]([[TILE_INDEX_W_CAP]])

    //CHECK:                [[SIZE_C:%.+]] = affine.min #[[$C_SIZE_BY_ITER_MAP]]([[LOOP_ITER_C]])
    //CHECK:                [[OFFSET_W:%.+]] = affine.max #[[$OFFSET_TO_SLICE_OFFSET_MAP]]([[CORRECTED_OFFSET_W]])
    //CHECK:                [[VALUE1:%.+]] = affine.max #[[$OFFSET_TO_PAD_CANDIDATE_MAP]]([[CORRECTED_OFFSET_W]])
    //CHECK:                [[PAD_LOW:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[VALUE1]]]
    //CHECK:                [[VALUE2:%.+]] = affine.max #[[$PAD_HIGH_CANDIDATE_MAP]]([[CORRECTED_SIZE_W]], [[OFFSET_W]])
    //CHECK:                [[PAD_HIGH:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[VALUE2]]]
    //CHECK:                [[SIZE_W:%.+]] = affine.apply #[[$SLICE_SIZE_BY_OUT_AND_PAD_MAP]]([[CORRECTED_SIZE_W]], [[PAD_LOW]], [[PAD_HIGH]])
    //CHECK:                [[SLICE_INPUT:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, 0, [[OFFSET_W]]] [1, 640, 32, [[SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:           tensor<1x640x32x32xf16, {order = #NHWC}> to tensor<1x640x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 32, 32]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[SLICE_WEIGHTS:%.+]] = tensor.extract_slice [[INPUT1]][[[LOOP_ITER_C]], 0, 0, 0] [[[SIZE_C]], 640, 3, 3] [1, 1, 1, 1]
    //CHECK-SAME:           tensor<640x640x3x3xf16, {order = #NHWC}> to tensor<?x640x3x3xf16, {bounds = #const.OpaqueI64Elements<[640, 640, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[PAD:%.+]] = tensor.pad [[SLICE_INPUT]] low[0, 0, 1, [[PAD_LOW]]] high[0, 0, 1, [[PAD_HIGH]]]

    //CHECK:                [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[SLICE_WEIGHTS]])

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[CONV]] into [[LOOP_OUT_W]][0, [[LOOP_ITER_C]], 0, [[CORRECTED_OFFSET_W]]] [1, [[SIZE_C]], 32, [[CORRECTED_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:           tensor<1x?x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 32, 32]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x640x32x32xf16, {order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x640x32x32xf16, {order = #NHWC}>

    //CHECK:   scf.yield [[LOOP_W]] : tensor<1x640x32x32xf16, {order = #NHWC}>
    //CHECK: return [[LOOP_C]] : tensor<1x640x32x32xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (0, d0 - 98)>

// CHECK-LABEL: @ApplyTilingMaxPool
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16, {order = #NHWC}>)
func.func @ApplyTilingMaxPool(%arg0: tensor<1x16x200x200xf16, {order = #NHWC}>) -> tensor<1x16x200x200xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]
    } -> tensor<1x16x200x200xf16, {order = #NHWC}>

    return %0 : tensor<1x16x200x200xf16, {order = #NHWC}>

    //CHECK-DAG: [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 200 : index
    //CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 100 : index
    //CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x200x200xf16, {order = #NHWC}>) {

    //CHECK:                [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP]]([[LOOP_ITER]])
    //CHECK:                [[DIFF1:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:                [[PAD_LOW:%.+]] = affine.min #[[$MAP2]]()[[[DIFF1]]]
    //CHECK:                [[DIFF2:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET]])
    //CHECK:                [[PAD_HIGH:%.+]] = affine.min #[[$MAP2]]()[[[DIFF2]]]
    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [1, 16, 101, 200] [1, 1, 1, 1] : tensor<1x16x200x200xf16, {order = #NHWC}> to tensor<1x16x101x200xf16, {order = #NHWC}>
    //CHECK:                [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    //CHECK:                   tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                   tensor<1x16x101x200xf16, {order = #NHWC}> to tensor<1x16x?x202xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 202, 202]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[PAD]], [[WEIGHTS_TABLE]] )
    //CHECK-SAME:           pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK-SAME:           tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 200, 200]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:               [[CAST:%.+]] = tensor.cast [[MAXPOOL]] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 200, 200]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x100x200xf16, {order = #NHWC}>

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[CAST]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 16, 100, 200] [1, 1, 1, 1] : tensor<1x16x100x200xf16, {order = #NHWC}> into tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x16x200x200xf16, {order = #NHWC}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (0, d0 - 148)>
//CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1) -> (-d0 - d1 + 52)>

// CHECK-LABEL: @ApplyTilingMaxPool4Tiles
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16, {order = #NHWC}>)
func.func @ApplyTilingMaxPool4Tiles(%arg0: tensor<1x16x200x200xf16, {order = #NHWC}>) -> tensor<1x16x200x200xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 4, 1]
    } -> tensor<1x16x200x200xf16, {order = #NHWC}>

    return %0 : tensor<1x16x200x200xf16, {order = #NHWC}>

    //CHECK-DAG: [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 200 : index
    //CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 50 : index
    //CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x200x200xf16, {order = #NHWC}>) {

    //CHECK:                [[OFFSET:%.+]] = affine.max #[[$MAP]]([[LOOP_ITER]])
    //CHECK:                [[DIFF1:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:                [[PAD_LOW:%.+]] = affine.min #[[$MAP2]]()[[[DIFF1]]]
    //CHECK:                [[DIFF2:%.+]] = affine.max #[[$MAP3]]([[OFFSET]])
    //CHECK:                [[PAD_HIGH:%.+]] = affine.min #[[$MAP2]]()[[[DIFF2]]]
    //CHECK:                [[INPUT_SIZE:%.+]] = affine.apply #[[$MAP4]]([[PAD_LOW]], [[PAD_HIGH]])

    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[OFFSET]], 0] [1, 16, [[INPUT_SIZE]], 200] [1, 1, 1, 1] : tensor<1x16x200x200xf16, {order = #NHWC}> to tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 200, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    //CHECK:                   tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                   tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 200, 200]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x202xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 202, 202]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[PAD]], [[WEIGHTS_TABLE]] )
    //CHECK-SAME:           pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK-SAME:           tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 200, 200]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:               [[CAST:%.+]] = tensor.cast [[MAXPOOL]] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 200, 200]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x50x200xf16, {order = #NHWC}>

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[CAST]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 16, 50, 200] [1, 1, 1, 1] : tensor<1x16x50x200xf16, {order = #NHWC}> into tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x16x200x200xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x16x200x200xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ApplyTilingAvgPool
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x12960xf16, {order = #NHWC}>
func.func @ApplyTilingAvgPool(%arg0: tensor<1x16x7x12960xf16, {order = #NHWC}>) -> tensor<1x16x1x12960xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [7, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 3]
        } -> tensor<1x16x1x12960xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x12960xf16, {order = #NHWC}>

    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 12960 : index
    //CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 4320 : index

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x1x12960xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x1x12960xf16, {order = #NHWC}>) {

    //CHECK:       [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 7, 4320] [1, 1, 1, 1] : tensor<1x16x7x12960xf16, {order = #NHWC}> to tensor<1x16x7x4320xf16, {order = #NHWC}>
    //CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[SLICE]])

    //CHECK: [[INSERT:%.+]] = tensor.insert_slice [[AVGPOOL]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 1, 4320] [1, 1, 1, 1] : tensor<1x16x1x4320xf16, {order = #NHWC}> into tensor<1x16x1x12960xf16, {order = #NHWC}>
    //CHECK: scf.yield [[INSERT]] : tensor<1x16x1x12960xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x16x1x12960xf16, {order = #NHWC}>

}


// -----

 #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

 // CHECK-LABEL: @NoPaddingDWCONV
 // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x32x200x200xf16, {order = #NHWC}>,
 // CHECK-SAME:      [[WEIGHTS:%arg[0-9]]]: tensor<32x16x1x1xf16, {order = #NHWC}>
 func.func @NoPaddingDWCONV(
         %arg0: tensor<1x32x200x200xf16, {order = #NHWC}>,
         %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>
 ) -> tensor<1x32x200x200xf16, {order = #NHWC}> {
     %1 = VPU.NCE.DepthConvolution(%arg0, %arg1) rawFilterShape [32, 1, 1, 1] {
         pad = #VPU.Padding<
             left = 0 : i64,
             right = 0 : i64,
             top = 0 : i64,
             bottom = 0 : i64
         >,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         strides = [1, 1],
         tilingStrategy = [1, 1, 1, 4]
     } -> tensor<1x32x200x200xf16, {order = #NHWC}>

     return %1 : tensor<1x32x200x200xf16, {order = #NHWC}>

    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 200 : index
    //CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 50 : index

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x200x200xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x32x200x200xf16, {order = #NHWC}>) {

    //CHECK:      [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 200, 50] [1, 1, 1, 1] : tensor<1x32x200x200xf16, {order = #NHWC}> to tensor<1x32x200x50xf16, {order = #NHWC}>
    //CHECK:      [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[SLICE]], [[WEIGHTS]])

    //CHECK: [[INSERT:%.+]] = tensor.insert_slice [[DWCONV]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 200, 50] [1, 1, 1, 1] : tensor<1x32x200x50xf16, {order = #NHWC}> into tensor<1x32x200x200xf16, {order = #NHWC}>
    //CHECK: scf.yield [[INSERT]] : tensor<1x32x200x200xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x32x200x200xf16, {order = #NHWC}>
}

// -----

 #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

 // CHECK-LABEL: @NotPaddedMaxPool
 // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x256x480xf16, {order = #NHWC}>
 func.func @NotPaddedMaxPool(
         %arg0: tensor<1x16x256x480xf16, {order = #NHWC}>
 ) -> tensor<1x16x127x480xf16, {order = #NHWC}> {
     %1 = VPU.NCE.MaxPool(%arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
         kernel_size = [3, 1],
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<
             left = 0 : i64,
             right = 0 : i64,
             top = 0 : i64,
             bottom = 0 : i64>,
             ppe = #VPU.PPEInt<mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         strides = [2, 1],
         tilingStrategy = [1, 1, 1, 4]
     } -> tensor<1x16x127x480xf16, {order = #NHWC}>

     return %1 : tensor<1x16x127x480xf16, {order = #NHWC}>

    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 480 : index
    //CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 120 : index

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x127x480xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x127x480xf16, {order = #NHWC}>) {

    //CHECK:      [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, 120] [1, 1, 1, 1] : tensor<1x16x256x480xf16, {order = #NHWC}> to tensor<1x16x256x120xf16, {order = #NHWC}>
    //CHECK:      [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[SLICE]])
    //CHECK-SAME: pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    //CHECK: [[INSERT:%.+]]  = tensor.insert_slice [[MAXPOOL]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 127, 120] [1, 1, 1, 1] : tensor<1x16x127x120xf16, {order = #NHWC}> into tensor<1x16x127x480xf16, {order = #NHWC}>
    //CHECK:  scf.yield [[INSERT]] : tensor<1x16x127x480xf16, {order = #NHWC}>
    //CHECK:  return [[LOOP]] : tensor<1x16x127x480xf16, {order = #NHWC}>
 }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

 // CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (-d0 + 140, 47)>
 // CHECK-LABEL: @UnEvenEltwiseTiling
 // CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x16x256x140xf16, {order = #NHWC}>,
 // CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x16x256x140xf16, {order = #NHWC}>)
 func.func @UnEvenEltwiseTiling(
         %arg0: tensor<1x16x256x140xf16, {order = #NHWC}>,
         %arg1: tensor<1x16x256x140xf16, {order = #NHWC}>
 ) -> tensor<1x16x256x140xf16, {order = #NHWC}> {
     %1 = VPU.NCE.Eltwise(%arg0, %arg1) {
         is_inplace = true,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             quant_scale = [1.000000e+00],
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         tilingStrategy = [1, 1, 1, 3]
     } -> tensor<1x16x256x140xf16, {order = #NHWC}>

     return %1 : tensor<1x16x256x140xf16, {order = #NHWC}>

    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 140 : index
    //CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 47 : index

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x256x140xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x256x140xf16, {order = #NHWC}>) {

    //CHECK:      [[UNEVEN_SIZE:%.+]]  = affine.min #[[$MAP]]([[LOOP_ITER]])
    //CHECK:      [[SLICE0:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[UNEVEN_SIZE]]] [1, 1, 1, 1] : tensor<1x16x256x140xf16, {order = #NHWC}> to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:      [[SLICE1:%.+]] = tensor.extract_slice [[INPUT1]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[UNEVEN_SIZE]]] [1, 1, 1, 1] : tensor<1x16x256x140xf16, {order = #NHWC}> to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE0]], [[SLICE1]])

    //CHECK: [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[UNEVEN_SIZE]]] [1, 1, 1, 1] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x256x140xf16, {order = #NHWC}>
    //CHECK:   scf.yield [[INSERT]] : tensor<1x16x256x140xf16, {order = #NHWC}>

    //CHECK: return [[LOOP]] : tensor<1x16x256x140xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$MAP_MIN:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 240)>

// CHECK-LABEL: @DynamicEltwiseTiling
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
func.func @DynamicEltwiseTiling(
         %arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
         %arg1: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
     %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
         is_inplace = true,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             quant_scale = [1.000000e+00],
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         tilingStrategy = [1, 1, 1, 2]
     } -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

     return %0 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK-DAG: [[DIM_VALUE:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 240 : index

    //CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT0]], [[DIM_VALUE]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[LOOP_END]]) : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:      [[UNEVEN_SIZE:%.+]] = affine.min #[[$MAP_MIN]]([[LOOP_ITER]])[[[LOOP_END]]]
    //CHECK:      [[SLICE0:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[UNEVEN_SIZE]]] [1, 1, 1, 1] :
    //CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:           to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 240]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:      [[SLICE1:%.+]] = tensor.extract_slice [[INPUT1]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[UNEVEN_SIZE]]] [1, 1, 1, 1] :
    //CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:           to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 240]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE0]], [[SLICE1]])
    //CHECK:      [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[UNEVEN_SIZE]]] [1, 1, 1, 1] :
    //CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 240]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:           into tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:      scf.yield [[INSERT]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: return [[LOOP]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

 #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
 // CHECK: #[[$OUT_OFFSET_AND_SIZE_MAP:.+]] = affine_map<(d0) -> ((d0 floordiv 69) * 69, (d0 floordiv 69) * 68 + 4)>
 // CHECK: #[[$TILE_INDEX_CAP_AND_REMAINDER_MAP:.+]] = affine_map<(d0) -> (d0 floordiv 69, 4)>
 // CHECK: #[[$OUT_SIZE_BY_CAPPED_TILE_INDEX_MAP:.+]] = affine_map<(d0) -> (-d0 + 72, 69)>


 // CHECK-LABEL: @NotPaddedUnevenMaxPool
 // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x256x480xf16, {order = #NHWC}>
 func.func @NotPaddedUnevenMaxPool(
         %arg0: tensor<1x16x256x480xf16, {order = #NHWC}>
 ) -> tensor<1x16x127x480xf16, {order = #NHWC}> {
     %1 = VPU.NCE.MaxPool(%arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
         kernel_size = [3, 1],
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<
             left = 0 : i64,
             right = 0 : i64,
             top = 0 : i64,
             bottom = 0 : i64>,
             ppe = #VPU.PPEInt<mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         strides = [2, 1],
         tilingStrategy = [1, 1, 1, 7]
     } -> tensor<1x16x127x480xf16, {order = #NHWC}>

     return %1 : tensor<1x16x127x480xf16, {order = #NHWC}>

    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 480 : index
    //CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 69 : index

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x127x480xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x127x480xf16, {order = #NHWC}>) {

    //CHECK:      [[CORRECTED_OFFSET:%.+]] = affine.min #[[$OUT_OFFSET_AND_SIZE_MAP]]([[LOOP_ITER]])
    //CHECK:      [[TILE_INDEX_CAP:%.+]] = affine.min #[[$TILE_INDEX_CAP_AND_REMAINDER_MAP]]([[LOOP_ITER]])
    //CHECK:      [[CORRECTED_SIZE:%.+]] = affine.min #[[$OUT_SIZE_BY_CAPPED_TILE_INDEX_MAP]]([[TILE_INDEX_CAP]])


    //CHECK:      [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[CORRECTED_OFFSET]]] [1, 16, 256, [[CORRECTED_SIZE]]] [1, 1, 1, 1] : tensor<1x16x256x480xf16, {order = #NHWC}> to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:      [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[SLICE]])
    //CHECK-SAME: pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    //CHECK: [[INSERT:%.+]]  = tensor.insert_slice [[MAXPOOL]] into [[LOOP_OUT]][0, 0, 0, [[CORRECTED_OFFSET]]] [1, 16, 127, [[CORRECTED_SIZE]]] [1, 1, 1, 1] : tensor<1x16x127x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 127, 480]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x127x480xf16, {order = #NHWC}>
    //CHECK:  scf.yield [[INSERT]] : tensor<1x16x127x480xf16, {order = #NHWC}>
    //CHECK:  return [[LOOP]] : tensor<1x16x127x480xf16, {order = #NHWC}>
 }

// -----

 #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

 func.func @NotTileLayoutCast(
         %arg0: tensor<1x16x127x480xf16>
 ) -> tensor<1x16x127x480xf16, {order = #NHWC}> {
     %0 = VPU.LayoutCast(%arg0) {dst_order = #NHWC, tilingStrategy = [1, 1, 1, 2]} : tensor<1x16x127x480xf16> -> tensor<1x16x127x480xf16, {order = #NHWC}>

     return %0 : tensor<1x16x127x480xf16, {order = #NHWC}>

    //CHECK-NOT: scf.for
 }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (0, d0 - 254)>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0) -> (0, d0 - 358)>
// CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1) -> (-d0 - d1 + 122)>

// CHECK-LABEL:   @Tiling2DNotPaddedMaxPool
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x512x480xf16, {order = #NHWC}>
 func.func @Tiling2DNotPaddedMaxPool(
         %arg0: tensor<1x16x512x480xf16, {order = #NHWC}>
 ) -> tensor<1x16x512x480xf16, {order = #NHWC}> {
     %1 = VPU.NCE.MaxPool(%arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
         kernel_size = [3, 3],
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<
             left = 1 : i64,
             right = 1 : i64,
             top = 1 : i64,
             bottom = 1 : i64>,
             ppe = #VPU.PPEInt<mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         strides = [1, 1],
         tilingStrategy = [1, 1, 2, 4]
     } -> tensor<1x16x512x480xf16, {order = #NHWC}>

     return %1 : tensor<1x16x512x480xf16, {order = #NHWC}>

    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_END_H:%.+]]  = arith.constant 512 : index
    //CHECK-DAG: [[LOOP_STEP_H:%.+]] = arith.constant 256 : index

    //CHECK-DAG: [[LOOP_END_W:%.+]] = arith.constant 480 : index
    //CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant 120 : index

    //CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x512x480xf16, {order = #NHWC}>
    //CHECK: [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x512x480xf16, {order = #NHWC}>)

    //CHECK:                [[LOOP_W:%.+]] = scf.for
    //CHECK-SAME:                            [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    //CHECK-SAME:                            iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x16x512x480xf16, {order = #NHWC}>)

    //CHECK:                                 [[SLICE_OFFSET_H:%.+]] = affine.max #[[$MAP0]]([[LOOP_ITER_H]])
    //CHECK:                                 [[TEMP_VALUE0:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER_H]])
    //CHECK:                                 [[PAD_LOW_H:%.+]] = affine.min #[[$MAP2]]()[[[TEMP_VALUE0]]]
    //CHECK:                                 [[TEMP_VALUE1:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET_H]])
    //CHECK:                                 [[PAD_HIGH_H:%.+]] = affine.min #[[$MAP2]]()[[[TEMP_VALUE1]]]
    //CHECK:                                 [[SLICE_OFFSET_W:%.+]] = affine.max #[[$MAP0]]([[LOOP_ITER_W]])
    //CHECK:                                 [[TEMP_VALUE2:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER_W]])
    //CHECK:                                 [[PAD_LOW_W:%.+]] = affine.min #[[$MAP2]]()[[[TEMP_VALUE2]]]
    //CHECK:                                 [[TEMP_VALUE3:%.+]] = affine.max #[[$MAP4]]([[SLICE_OFFSET_W]])
    //CHECK:                                 [[PAD_HIGH_W:%.+]] = affine.min #[[$MAP2]]()[[[TEMP_VALUE3]]]
    //CHECK:                                 [[W_SIZE:%.+]] = affine.apply #map5([[PAD_LOW_W]], [[PAD_HIGH_W]])

    //CHECK:                                 [[SLICE:%.+]]  = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET_H]], [[SLICE_OFFSET_W]]] [1, 16, 257, [[W_SIZE]]] [1, 1, 1, 1] : tensor<1x16x512x480xf16, {order = #NHWC}> to tensor<1x16x257x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                                 [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW_H]], [[PAD_LOW_W]]] high[0, 0, [[PAD_HIGH_H]], [[PAD_HIGH_W]]] {
    //CHECK:                                 tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                                 tensor<1x16x257x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 259, 482]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                                 [[POOL:%.+]] = VPU.NCE.MaxPool([[PAD]])
    //CHECK-SAME:                                           pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    //CHECK:                                 [[CAST:%.+]] = tensor.cast [[POOL]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x120xf16, {order = #NHWC}>
    //CHECK:                                 [[INSERT:%.+]] = tensor.insert_slice [[CAST]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, 256, 120] [1, 1, 1, 1]
    //CHECK-SAME:                            tensor<1x16x256x120xf16, {order = #NHWC}> into tensor<1x16x512x480xf16, {order = #NHWC}>

    //CHECK:  scf.yield [[INSERT]] : tensor<1x16x512x480xf16, {order = #NHWC}>
    //CHECK:  scf.yield [[LOOP_W]] : tensor<1x16x512x480xf16, {order = #NHWC}>
    //CHECK:  return [[LOOP_H]] : tensor<1x16x512x480xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$MAP_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 128)>
// CHECK: #[[$MAP_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 240)>

// CHECK-LABEL: @Dynamic2DEltwiseTiling
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
func.func @Dynamic2DEltwiseTiling(
         %arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
         %arg1: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
     %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
         is_inplace = true,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             quant_scale = [1.000000e+00],
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         tilingStrategy = [1, 1, 2, 2]
     } -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

     return %0 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant 240 : index
    //CHECK-DAG: [[LOOP_STEP_H:%.+]] = arith.constant 128 : index
    //CHECK-DAG: [[CST_3:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[CST_2:%.+]] = arith.constant 2 : index
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    //CHECK: [[LOOP_END_H:%.+]] = tensor.dim [[INPUT0]], [[CST_2]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_END_W:%.+]] = tensor.dim [[INPUT0]], [[CST_3]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[LOOP_END_H]], [[LOOP_END_W]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:             [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    //CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)

    //CHECK:                  [[LOOP_W:%.+]] = scf.for
    //CHECK-SAME:                              [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    //CHECK-SAME:                              iter_args([[LOOP_OUT_W:%arg[0-9]]]  = [[LOOP_OUT]])

    //CHECK:                                   [[SLICE_SIZE_H:%.+]] = affine.min #[[$MAP_MIN_H]]([[LOOP_ITER_H]])[[[LOOP_END_H]]]
    //CHECK:                                   [[SLICE_SIZE_W:%.+]] = affine.min #[[$MAP_MIN_W]]([[LOOP_ITER_W]])[[[LOOP_END_W]]]
    //CHECK:                                   [[SLICE_INPUT0:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    //CHECK:                                   [[SLICE_INPUT1:%.+]] = tensor.extract_slice [[INPUT1]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    //CHECK:                                   [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE_INPUT0]], [[SLICE_INPUT1]])
    //CHECK:                                   [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:                              tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 240]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:  scf.yield [[INSERT]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:  scf.yield [[LOOP_W]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:  return [[LOOP_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP3:.+]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
//CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>

// CHECK-LABEL:   @ApplyTilingNCEConvDyn
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
func.func @ApplyTilingNCEConvDyn(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 512 : index
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[DIM_INDEX:%.+]] = arith.constant 2 : index
    //CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16

    //CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[LOOP_END]]) : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                [[RESULT_SIZE:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[LOOP_END]]]
    //CHECK:                [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:                [[TEMP_VALUE0:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER]])
    //CHECK:                [[PAD_LOW:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE0]]]
    //CHECK:                [[TEMP:%.+]] = affine.max #[[$MAP4]]([[RESULT_SIZE]], [[SLICE_OFFSET]])[[[LOOP_END]]]
    //CHECK:                [[PAD_HIGH:%.+]] = affine.min #[[$MAP3]]()[[[TEMP]]]
    //CHECK:                [[STRIDE_OFFSET:%.+]] = affine.apply #[[$MAP5]]([[RESULT_SIZE]], [[PAD_LOW]], [[PAD_HIGH]])

    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [1, 32, [[STRIDE_OFFSET]], 64] [1, 1, 1, 1]
    //CHECK-SAME:           : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    //CHECK:                   tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                   tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 514, 66]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[WEIGHTS]])
    //CHECK-SAME:           {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK-SAME:           -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 64]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[CONV]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 256, [[RESULT_SIZE]], 64] [1, 1, 1, 1] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 64]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP3:.+]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
//CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
// CHECK-LABEL: @ApplyTilingMaxPool4Tiles
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>)
func.func @ApplyTilingMaxPool4Tiles(%arg0: tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 4, 1]
    } -> tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK-DAG: [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK-DAG: [[CST_2:%.+]] = arith.constant 2 : index
    //CHECK-DAG: [[STEP:%.+]] = arith.constant 100 : index
    //CHECK-DAG: [[LOOP_START:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16

    //CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT]], [[CST_2]] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[OUTPUT:%.+]] = tensor.empty([[LOOP_END]]) : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[RESULT:%.+]] = scf.for [[LOOP_ITER:%.+]] = [[LOOP_START]] to [[LOOP_END]] step [[STEP]] iter_args([[LOOP_OUT:%.+]] = [[OUTPUT]]) -> (tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>) {
    //CHECK:                [[MIN_OFFSET:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[LOOP_END]]]
    //CHECK:                [[OFFSET:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:                [[TEMP_VALUE0:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER]])
    //CHECK:                [[PAD_LOW:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE0]]]
    //CHECK:                [[TEMP_VALUE1:%.+]] = affine.max #[[$MAP4]]([[MIN_OFFSET]], [[OFFSET]])[[[LOOP_END]]]
    //CHECK:                [[PAD_HIGH:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE1]]]
    //CHECK:                [[SIZE:%.+]] = affine.apply #[[$MAP5]]([[MIN_OFFSET]], [[PAD_LOW]], [[PAD_HIGH]])
    //CHECK:                [[SLICE0:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[OFFSET]], 0] [1, 16, [[SIZE]], 200] [1, 1, 1, 1] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 200]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[PAD:%.+]] = tensor.pad [[SLICE0]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    //CHECK:                    ^bb0([[ARG3:%.+]]: index, [[ARG4:%.+]]: index, [[ARG5:%.+]]: index, [[ARG6:%.+]]: index):
    //CHECK:                    tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                } : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 200]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x202xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 102, 202]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[POOL_RESULT:%.+]] = VPU.NCE.MaxPool([[PAD]], [[WEIGHTS_TABLE]] ) {kernel_size = [3, 3],
    //CHECK-SAME:                                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          strides = [1, 1], tiling_loop_index = 0 : i64}
    //CHECK-SAME:                                 -> tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[SLICE1:%.+]] = tensor.insert_slice [[POOL_RESULT]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 16, [[MIN_OFFSET]], 200] [1, 1, 1, 1] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 200]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                scf.yield [[SLICE1]] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: }
    //CHECK: return [[RESULT]] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK: #[[$MAP_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 200)>
//CHECK: #[[$MAP_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 640)>

// CHECK-LABEL: @DynamicConvertTiling
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
func.func @DynamicConvertTiling(
         %arg0: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}> {
     %0 = VPU.Convert(%arg0) {dstElemType = f16,
                              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                              tilingStrategy = [1, 1, 8, 4]}
        : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

     return %0 : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

    //CHECK-DAG: [[LOOP_H_STEP:%.+]] = arith.constant 200 : index
    //CHECK-DAG: [[LOOP_W_STEP:%.+]] = arith.constant 640 : index
    //CHECK-DAG: [[CST_2:%.+]] = arith.constant 2 : index
    //CHECK-DAG: [[CST_3:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    //CHECK: [[DIM_H_END:%.+]] = tensor.dim [[INPUT]], [[CST_2]] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK: [[DIM_W_END:%.+]] = tensor.dim [[INPUT]], [[CST_3]] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_H_END]], [[DIM_W_END]]) : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

    //CHECK: [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:             [[LOOP_H_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[DIM_H_END]] step [[LOOP_H_STEP]]
    //CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>) {

    //CHECK:                  [[LOOP_W:%.+]] = scf.for
    //CHECK-SAME:                             [[LOOP_W_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[DIM_W_END]] step [[LOOP_W_STEP]]
    //CHECK-SAME:                             iter_args([[LOOP_OUT_H:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>) {
    //CHECK:                                  [[SIZE_H:%.+]] = affine.min #[[$MAP_H]]([[LOOP_H_ITER]])[[[DIM_H_END]]]
    //CHECK:                                  [[SIZE_W:%.+]] = affine.min #[[$MAP_W]]([[LOOP_W_ITER]])[[[DIM_W_END]]]

    //CHECK:                                  [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[LOOP_H_ITER]], [[LOOP_W_ITER]]] [1, 3, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
    //CHECK:                                  [[CONVERT:%.+]] = VPU.Convert([[SLICE]])
    //CHECK:                                  [[INSERT_SLICE:%.+]] = tensor.insert_slice [[CONVERT]] into [[LOOP_OUT_H]][0, 0, [[LOOP_H_ITER]], [[LOOP_W_ITER]]] [1, 3, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
    //CHECK:                  scf.yield [[INSERT_SLICE]] : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK: scf.yield [[LOOP_W]]
    //CHECK: return [[LOOP_H]] : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

//CHECK: #[[$MAP_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 80)>
//CHECK: #[[$MAP_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 640)>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DynamicNCEPermuteTiling
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
func.func @DynamicNCEPermuteTiling(
         %arg0: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> {
     %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64,
                                  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
                                  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                  tilingStrategy = [1, 1, 20, 4]}
    -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>

     return %0 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[CST_2:%.+]] = arith.constant 2 : index
    //CHECK-DAG: [[CST_3:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[LOOP_H_STEP:%.+]] = arith.constant 80 : index
    //CHECK-DAG: [[LOOP_W_STEP:%.+]] = arith.constant 640 : index

    //CHECK: [[LOOP_H_END:%.+]] = tensor.dim [[INPUT]], [[CST_2]] : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK: [[LOOP_W_END:%.+]] = tensor.dim [[INPUT]], [[CST_3]] : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[LOOP_H_END]], [[LOOP_W_END]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:             [[LOOP_H_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_H_END]] step [[LOOP_H_STEP]]
    //CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>)
    //CHECK:                  [[LOOP_W:%.+]] = scf.for
    //CHECK-SAME:                             [[LOOP_W_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_W_END]] step [[LOOP_W_STEP]]
    //CHECK-SAME:                             iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>) {
    //CHECK:                                  [[SIZE_H:%.+]] = affine.min #[[$MAP_H]]([[LOOP_H_ITER]])[[[LOOP_H_END]]]
    //CHECK:                                  [[SIZE_W:%.+]] = affine.min #[[$MAP_W]]([[LOOP_W_ITER]])[[[LOOP_W_END]]]
    //CHECK:                                  [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[LOOP_H_ITER]], [[LOOP_W_ITER]]] [1, 3, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
    //CHECK:                                  [[PERMUTE:%.+]] = VPU.NCE.Permute([[SLICE]])
    //CHECK:                                  [[INSERT_SLICE:%.+]] = tensor.insert_slice [[PERMUTE]] into [[LOOP_OUT_W]][0, 0, [[LOOP_H_ITER]], [[LOOP_W_ITER]]] [1, 16, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
    //CHECK: scf.yield [[INSERT_SLICE]]

    //CHECK: scf.yield [[LOOP_W]]
    //CHECK: return [[LOOP_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 11)>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!dynInputType = tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
!dynOutputType = tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @NoPaddingDWCONV_W_DynamicInput
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:      [[WEIGHTS:%arg[0-9]]]: tensor<32x16x1x1xf16, {order = #NHWC}>
func.func @NoPaddingDWCONV_W_DynamicInput(
         %arg0: !dynInputType,
         %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>
 ) -> !dynOutputType {
     %1 = VPU.NCE.DepthConvolution(%arg0, %arg1) rawFilterShape [32, 1, 1, 1] {
         pad = #VPU.Padding<
             left = 0 : i64,
             right = 0 : i64,
             top = 0 : i64,
             bottom = 0 : i64
         >,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         strides = [1, 1],
         tilingStrategy = [1, 1, 1, 117]
     } -> !dynOutputType

    //CHECK-DAG: [[CST_3:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 11 : index

    //CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT]], [[CST_3]] : tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[LOOP_END]]) : tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                [[SIZE:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[LOOP_END]]]
    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 800, [[SIZE]]] [1, 1, 1, 1]
    //CHECK-SAME:           : tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 11]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[DEPTH_CONV:%.+]] = VPU.NCE.DepthConvolution([[SLICE]], [[WEIGHTS]])  rawFilterShape [32, 1, 1, 1]
    //CHECK-SAME:           {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK-SAME:           , ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:          strides = [1, 1], tiling_loop_index = 0 : i64} -> tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 11]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[DEPTH_CONV]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 800, [[SIZE]]] [1, 1, 1, 1] : tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 11]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

     return %1 : !dynOutputType

    //CHECK: return [[LOOP]] : tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 400)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 11)>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!dynInputType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
!dynOutputType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @NoPaddingDWCONV_HW_DynamicInput
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:      [[WEIGHTS:%arg[0-9]]]: tensor<32x16x1x1xf16, {order = #NHWC}>
func.func @NoPaddingDWCONV_HW_DynamicInput(
         %arg0: !dynInputType,
         %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>
 ) -> !dynOutputType {
     %1 = VPU.NCE.DepthConvolution(%arg0, %arg1) rawFilterShape [32, 1, 1, 1] {
         pad = #VPU.Padding<
             left = 0 : i64,
             right = 0 : i64,
             top = 0 : i64,
             bottom = 0 : i64
         >,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         strides = [1, 1],
         tilingStrategy = [1, 1, 2, 117]
     } -> !dynOutputType

    //CHECK-DAG: [[LOOP_STEP_H:%.+]] = arith.constant 400 : index
    //CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant 11 : index
    //CHECK-DAG: [[CST_2:%.+]] = arith.constant 2 : index
    //CHECK-DAG: [[CST_3:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    //CHECK: [[LOOP_END_H:%.+]] = tensor.dim [[INPUT]], [[CST_2]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_END_W:%.+]] = tensor.dim [[INPUT]], [[CST_3]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[LOOP_END_H]], [[LOOP_END_W]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    //CHECK-SAME:           iter_args([[LOOP_OUT_H:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK: [[LOOP_W:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUT_H]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                [[SIZE_H:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER_H]])[[[LOOP_END_H]]]
    //CHECK:                [[SIZE_W:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_W]])[[[LOOP_END_W]]]

    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 32, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:           : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 11]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[DEPTH_CONV:%.+]] = VPU.NCE.DepthConvolution([[SLICE]], [[WEIGHTS]])  rawFilterShape [32, 1, 1, 1]
    //CHECK-SAME:           {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK-SAME:           , ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:          strides = [1, 1], tiling_loop_index = 0 : i64} -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 11]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[DEPTH_CONV]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 32, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 11]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

     return %1 : !dynOutputType
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (0, d0 - 2)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 2, 0)>
//CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (2, s0)>
//CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (0, d0 - 24)>

// CHECK-LABEL: @SCFTilingWithChannelPaddedWeights
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x32x56x56xf16, {order = #NHWC}>
func.func @SCFTilingWithChannelPaddedWeights(%arg0: tensor<1x32x56x56xf16, {order = #NHWC}>) -> tensor<1x64x56x56xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x32x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x32x5x5xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [16, 0, 0, 0]>]

    %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [64, 32, 5, 5] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x32x56x56xf16, {order = #NHWC}>, tensor<64x32x5x5xf16, {order = #NHWC}> -> tensor<1x64x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x64x56x56xf16, {order = #NHWC}>

    //CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<64x32x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x32x5x5xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [16, 0, 0, 0]>]
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 56 : index
    //CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 28 : index
    //CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x64x56x56xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x64x56x56xf16, {order = #NHWC}>) {

    //CHECK:                [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP]]([[LOOP_ITER]])
    //CHECK:                [[DIFF1:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:                [[PAD_LOW:%.+]] = affine.min #[[$MAP2]]()[[[DIFF1]]]
    //CHECK:                [[DIFF2:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET]])
    //CHECK:                [[PAD_HIGH:%.+]] = affine.min #[[$MAP2]]()[[[DIFF2]]]

    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [1, 32, 30, 56] [1, 1, 1, 1] : tensor<1x32x56x56xf16, {order = #NHWC}> to tensor<1x32x30x56xf16, {order = #NHWC}>
    //CHECK:                [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW]], 2] high[0, 0, [[PAD_HIGH]], 2] {
    //CHECK:                   tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                   tensor<1x32x30x56xf16, {order = #NHWC}> to tensor<1x32x?x60xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 60, 60]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[WEIGHTS]])
    //CHECK-SAME:           {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK-SAME:           -> tensor<1x64x?x56xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 56, 56]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:               [[CAST:%.+]] = tensor.cast [[CONV]] : tensor<1x64x?x56xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 56, 56]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x64x28x56xf16, {order = #NHWC}>

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[CAST]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 64, 28, 56] [1, 1, 1, 1] : tensor<1x64x28x56xf16, {order = #NHWC}> into tensor<1x64x56x56xf16, {order = #NHWC}>
    //CHECK: scf.yield [[INSERT]] : tensor<1x64x56x56xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x64x56x56xf16, {order = #NHWC}>
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 108)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 52)>

module @test {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
  }

  // CHECK-LABEL: @EltwiseAutoPadded
  // CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>,
  // CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>)
  func.func @EltwiseAutoPadded(
          %arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>,
          %arg1: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
  ) -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}> {
      %21 = VPU.NCE.Eltwise(%arg0, %arg1) {
          input_padding = [0, 13, 0, 0],
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
          op_type = #VPU.eltwise_type<ADD>,
          ppe = #VPU.PPEFp<mode = <NOOP>,
          clamp_low = -3.4028234663852886E+38 : f64,
          clamp_high = 3.4028234663852886E+38 : f64,
          scale = 1.000000e+00 : f64,
          prelu_alpha = [1.000000e+00],
          bias = 0.000000e+00 : f64,
          adder = 0.000000e+00 : f64>,
          tilingStrategy = [1, 1, 10, 37]
      } -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>

      return %21 : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[H_LOOP_STEP:%.+]] = arith.constant 108 : index
    //CHECK-DAG: [[W_LOOP_STEP:%.+]] = arith.constant 52 : index
    //CHECK-DAG: [[CST_2:%.+]] = arith.constant 2 : index
    //CHECK-DAG: [[CST_3:%.+]] = arith.constant 3 : index

    //CHECK: [[H_DIM:%.+]] = tensor.dim [[INPUT0]], [[CST_2]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[W_DIM:%.+]] = tensor.dim [[INPUT0]], [[CST_3]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[H_DIM]], [[W_DIM]]) : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>

    //CHECK: [[H_LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[H_LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[H_DIM]] step [[H_LOOP_STEP]]
    //CHECK-SAME:           iter_args([[H_LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>) {

    //CHECK: [[W_LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[W_LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[W_DIM]] step [[W_LOOP_STEP]]
    //CHECK-SAME:           iter_args([[W_LOOP_OUT:%arg[0-9]]]  = [[H_LOOP_OUT]]) -> (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>) {

    //CHECK:      [[UNEVEN_SIZE_0:%.+]]  = affine.min #[[$MAP]]([[H_LOOP_ITER]])[[[H_DIM]]]
    //CHECK:      [[UNEVEN_SIZE_1:%.+]]  = affine.min #[[$MAP1]]([[W_LOOP_ITER]])[[[W_DIM]]]
    //CHECK:      [[SLICE0:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, [[H_LOOP_ITER]], [[W_LOOP_ITER]]] [1, 16, [[UNEVEN_SIZE_0]], [[UNEVEN_SIZE_1]]] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 108, 52]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:      [[SLICE1:%.+]] = tensor.extract_slice [[INPUT1]][0, 0, [[H_LOOP_ITER]], [[W_LOOP_ITER]]] [1, 16, [[UNEVEN_SIZE_0]], [[UNEVEN_SIZE_1]]] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 108, 52]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE0]], [[SLICE1]])

    //CHECK: [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[W_LOOP_OUT]][0, 0, [[H_LOOP_ITER]], [[W_LOOP_ITER]]] [1, 3, [[UNEVEN_SIZE_0]], [[UNEVEN_SIZE_1]]] [1, 1, 1, 1] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 108, 52]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK:   scf.yield [[INSERT]] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK:   scf.yield [[W_LOOP]] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>

    //CHECK: return [[H_LOOP]] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 30)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 192)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 2)>

// CHECK-LABEL: @ApplyTilingD2SPadded
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
func.func @ApplyTilingD2SPadded(
          %arg0: tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}> {
    %20 = VPU.DepthToSpace(%arg0) {
        block_size = 2 : i64,
        mode = #IE.depth_to_space_mode<DEPTH_FIRST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        padded_channels = #IE.ChannelPadding<input = 0 : i64, output = 13 : i64>,
        tilingStrategy = [1, 1, 36, 10]
    } : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    return %20 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[H_LOOP_STEP:%.+]] = arith.constant 30 : index
    //CHECK-DAG: [[W_LOOP_STEP:%.+]] = arith.constant 192 : index

    //CHECK-DAG: [[THREE:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[TWO:%.+]] = arith.constant 2 : index

    //CHECK: [[H_RAW:%.+]] = tensor.dim [[INPUT]], [[TWO]] : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[H_DIM:%.+]] = arith.muli [[H_RAW]], [[TWO]] : index
    //CHECK: [[W_RAW:%.+]] = tensor.dim [[INPUT]], [[THREE]] : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[W_DIM:%.+]] = arith.muli [[W_RAW]], [[TWO]] : index
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[H_DIM]], [[W_DIM]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[H_LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[H_LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[H_DIM]] step [[H_LOOP_STEP]]
    //CHECK-SAME:           iter_args([[H_LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK: [[W_LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[W_LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[W_DIM]] step [[W_LOOP_STEP]]
    //CHECK-SAME:           iter_args([[W_LOOP_OUT:%arg[0-9]]]  = [[H_LOOP_OUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:      [[H_BOUND:%.+]]  = affine.min #[[$MAP]]([[H_LOOP_ITER]])[[[H_DIM]]]
    //CHECK:      [[W_BOUND:%.+]]  = affine.min #[[$MAP1]]([[W_LOOP_ITER]])[[[W_DIM]]]
    //CHECK:      [[H_OFFST:%.+]]  = affine.apply #[[$MAP2]]([[H_LOOP_ITER]])
    //CHECK:      [[H_SHAPE:%.+]]  = affine.apply #[[$MAP2]]([[H_BOUND]])
    //CHECK:      [[W_OFFST:%.+]]  = affine.apply #[[$MAP2]]([[W_LOOP_ITER]])
    //CHECK:      [[W_SHAPE:%.+]]  = affine.apply #[[$MAP2]]([[W_BOUND]])

    //CHECK:      [[SLICE0:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[H_OFFST]], [[W_OFFST]]] [1, 12, [[H_SHAPE]], [[W_SHAPE]]] [1, 1, 1, 1] : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 15, 96]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:      [[D2S:%.+]] = VPU.DepthToSpace([[SLICE0]])
    //CHECK:      [[INSERT:%.+]] = tensor.insert_slice [[D2S]] into [[W_LOOP_OUT]][0, 0, [[H_LOOP_ITER]], [[W_LOOP_ITER]]] [1, 16, [[H_BOUND]], [[W_BOUND]]] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 30, 192]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:   scf.yield [[INSERT]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:   scf.yield [[W_LOOP]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: return [[H_LOOP]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 11)>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!dynInputType = tensor<1x4x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
!dynOutputType = tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @NoPaddingCompressCONV_W_DynamicInput
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x4x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:      [[WEIGHTS:%arg[0-9]]]: tensor<32x4x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[WEIGHTS_TABLE:%arg[0-9]]]: tensor<32x1x1x4xsi32>
func.func @NoPaddingCompressCONV_W_DynamicInput(
         %arg0: !dynInputType,
         %arg1: tensor<32x4x1x1xf16, {order = #NHWC}>,
         %arg2: tensor<32x1x1x4xsi32>
 ) -> !dynOutputType {
     %1 = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) rawFilterShape [32, 4, 1, 1] {
         pad = #VPU.Padding<
             left = 0 : i64,
             right = 0 : i64,
             top = 0 : i64,
             bottom = 0 : i64
         >,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         strides = [1, 1],
         tilingStrategy = [1, 1, 1, 117],
         cm_sp_pattern = 0
     } : !dynInputType, tensor<32x4x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> !dynOutputType

    //CHECK-DAG: [[DIM_VALUE_0:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[DIM_0:%.+]] = tensor.dim [[INPUT]], [[DIM_VALUE_0]] : tensor<1x4x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[C11:%.+]] = arith.constant 11 : index
    //CHECK-DAG: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_0]]) : tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[C0]] to [[DIM_0]] step [[C11]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                [[SIZE:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[DIM_0]]]
    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 4, 800, [[SIZE]]] [1, 1, 1, 1]
    //CHECK-SAME:           : tensor<1x4x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x4x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 800, 11]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[COMPRESS_CONV:%.+]] = VPU.NCE.CompressConvolution([[SLICE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])  rawFilterShape [32, 4, 1, 1]
    //CHECK-SAME:           {cm_sp_pattern = 0 : i64, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK-SAME:           , ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:          strides = [1, 1], tiling_loop_index = 0 : i64} -> tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 11]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[COMPRESS_CONV]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 800, [[SIZE]]] [1, 1, 1, 1] : tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 11]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

     return %1 : !dynOutputType

    //CHECK: return [[LOOP]] : tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 400)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 11)>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!dynInputType = tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
!dynOutputType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @NoPaddingCompressCONV_HW_DynamicInput
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:      [[WEIGHTS:%arg[0-9]]]: tensor<32x4x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[WEIGHTS_TABLE:%arg[0-9]]]: tensor<32x1x1x4xsi32>
func.func @NoPaddingCompressCONV_HW_DynamicInput(
         %arg0: !dynInputType,
         %arg1: tensor<32x4x1x1xf16, {order = #NHWC}>,
         %arg2: tensor<32x1x1x4xsi32>
 ) -> !dynOutputType {
     %1 = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) rawFilterShape [32, 4, 1, 1] {
         pad = #VPU.Padding<
             left = 0 : i64,
             right = 0 : i64,
             top = 0 : i64,
             bottom = 0 : i64
         >,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         strides = [1, 1],
         tilingStrategy = [1, 1, 2, 117],
         cm_sp_pattern = 0
     } : !dynInputType, tensor<32x4x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> !dynOutputType

    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    //CHECK-DAG: [[DIM_VALUE_H_1:%.+]] = arith.constant 2 : index
    //CHECK-DAG: [[LOOP_END_H:%.+]] = tensor.dim [[INPUT]], [[DIM_VALUE_H_1]] : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK-DAG: [[DIM_VALUE_W_1:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[LOOP_END_W:%.+]] = tensor.dim [[INPUT]], [[DIM_VALUE_W_1]] : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK-DAG: [[LOOP_STEP_H:%.+]] = arith.constant 400 : index

    //CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant 11 : index

	//CHECK-DAG: [[LOOP_OUTPUT:%.+]] = tensor.empty([[LOOP_END_H]], [[LOOP_END_W]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

	//CHECK: [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    //CHECK-SAME:           iter_args([[LOOP_OUT_H:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>) {


    //CHECK: [[LOOP_W:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUT_H]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                [[SIZE_H:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER_H]])[[[LOOP_END_H]]]
    //CHECK:                [[SIZE_W:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_W]])[[[LOOP_END_W]]]

    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 4, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:           : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 400, 11]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[COMPRESS_CONV:%.+]] = VPU.NCE.CompressConvolution([[SLICE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])  rawFilterShape [32, 4, 1, 1]
    //CHECK-SAME:           {cm_sp_pattern = 0 : i64, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK-SAME:           , ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:          strides = [1, 1], tiling_loop_index = 0 : i64} -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 11]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[COMPRESS_CONV]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 32, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 11]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

     return %1 : !dynOutputType
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP3:.+]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
//CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
// CHECK-LABEL: @ApplyTilingAveragePool4Tiles
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>)
func.func @ApplyTilingAveragePool4Tiles(%arg0: tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 4, 1]
    } -> tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK-DAG: [[CST_2:%.+]] = arith.constant 2 : index
    //CHECK-DAG: [[STEP:%.+]] = arith.constant 100 : index
    //CHECK-DAG: [[LOOP_START:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16

    //CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT]], [[CST_2]] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[OUTPUT:%.+]] = tensor.empty([[LOOP_END]]) : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[RESULT:%.+]] = scf.for [[LOOP_ITER:%.+]] = [[LOOP_START]] to [[LOOP_END]] step [[STEP]] iter_args([[LOOP_OUT:%.+]] = [[OUTPUT]]) -> (tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>) {
    //CHECK:                [[MIN_OFFSET:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[LOOP_END]]]
    //CHECK:                [[OFFSET:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:                [[TEMP_VALUE0:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER]])
    //CHECK:                [[PAD_LOW:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE0]]]
    //CHECK:                [[TEMP_VALUE1:%.+]] = affine.max #[[$MAP4]]([[MIN_OFFSET]], [[OFFSET]])[[[LOOP_END]]]
    //CHECK:                [[PAD_HIGH:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE1]]]
    //CHECK:                [[SIZE:%.+]] = affine.apply #[[$MAP5]]([[MIN_OFFSET]], [[PAD_LOW]], [[PAD_HIGH]])
    //CHECK:                [[SLICE0:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[OFFSET]], 0] [1, 16, [[SIZE]], 200] [1, 1, 1, 1] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 200]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[PAD:%.+]] = tensor.pad [[SLICE0]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    //CHECK:                    ^bb0([[ARG3:%.+]]: index, [[ARG4:%.+]]: index, [[ARG5:%.+]]: index, [[ARG6:%.+]]: index):
    //CHECK:                    tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                } : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 200]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x202xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 102, 202]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[POOL_RESULT:%.+]] = VPU.NCE.AveragePool([[PAD]]) {kernel_size = [3, 3],
    //CHECK-SAME:                                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], tiling_loop_index = 0 : i64}
    //CHECK-SAME:                                 -> tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[SLICE1:%.+]] = tensor.insert_slice [[POOL_RESULT]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 16, [[MIN_OFFSET]], 200] [1, 1, 1, 1] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 200]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                scf.yield [[SLICE1]] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: }
    //CHECK: return [[RESULT]] : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x16x?x200xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 400, 200]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 205)>

//CHECK-LABEL: @TileReduceL1
//CHECK-SAME:  [[ARG0:%.+]]: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
func.func @TileReduceL1(%arg0: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>) -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}> {
     %0 = VPU.ReduceL1(%arg0) {
         axes_value = [1, 3],
         keep_dims,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         tilingStrategy = [1, 1, 5, 1]
     } : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}> -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
     return %0 : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>

//CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
//CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
//CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C205]] iter_args([[OUT:%.+]] = [[EMPTY]])
//CHECK-SAME:          -> (tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>)
//CHECK-NEXT:      [[COUNT:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
//CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 20, [[COUNT]], 1024] [1, 1, 1, 1]
//CHECK-SAME:          to tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 205, 1024]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceL1([[SLICE]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tiling_loop_index = 0 : i64}
//CHECK-SAME:          -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 1]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, [[IDX]], 0] [1, 1, [[COUNT]], 1] [1, 1, 1, 1]
//CHECK-SAME:          into tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK-NEXT:      scf.yield [[INSERT]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        return [[SCF]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 205)>

//CHECK-LABEL: @TileReduceL2
//CHECK-SAME:  [[ARG0:%.+]]: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
func.func @TileReduceL2(%arg0: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>) -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}> {
     %0 = VPU.ReduceL2(%arg0) {
         axes_value = [1, 3],
         keep_dims,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         tilingStrategy = [1, 1, 5, 1]
     } : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}> -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
     return %0 : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>

//CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
//CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
//CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C205]] iter_args([[OUT:%.+]] = [[EMPTY]])
//CHECK-SAME:          -> (tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>)
//CHECK-NEXT:      [[COUNT:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
//CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 20, [[COUNT]], 1024] [1, 1, 1, 1]
//CHECK-SAME:          to tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 205, 1024]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceL2([[SLICE]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tiling_loop_index = 0 : i64}
//CHECK-SAME:          -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 1]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, [[IDX]], 0] [1, 1, [[COUNT]], 1] [1, 1, 1, 1]
//CHECK-SAME:          into tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK-NEXT:      scf.yield [[INSERT]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        return [[SCF]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 205)>

//CHECK-LABEL: @TileReduceMin
//CHECK-SAME:  [[ARG0:%.+]]: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
func.func @TileReduceMin(%arg0: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>) -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}> {
     %0 = VPU.ReduceMin(%arg0) {
         axes_value = [1, 3],
         keep_dims,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         tilingStrategy = [1, 1, 5, 1]
     } : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}> -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
     return %0 : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>

//CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
//CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
//CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C205]] iter_args([[OUT:%.+]] = [[EMPTY]])
//CHECK-SAME:          -> (tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>)
//CHECK-NEXT:      [[COUNT:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
//CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 20, [[COUNT]], 1024] [1, 1, 1, 1]
//CHECK-SAME:          to tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 205, 1024]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceMin([[SLICE]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tiling_loop_index = 0 : i64}
//CHECK-SAME:          -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 1]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, [[IDX]], 0] [1, 1, [[COUNT]], 1] [1, 1, 1, 1]
//CHECK-SAME:          into tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK-NEXT:      scf.yield [[INSERT]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        return [[SCF]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 205)>

//CHECK-LABEL: @TileReduceMax
//CHECK-SAME:  [[ARG0:%.+]]: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
func.func @TileReduceMax(%arg0: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>) -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}> {
     %0 = VPU.ReduceMax(%arg0) {
         axes_value = [1, 3],
         keep_dims,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         tilingStrategy = [1, 1, 5, 1]
     } : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}> -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
     return %0 : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>

//CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
//CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
//CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C205]] iter_args([[OUT:%.+]] = [[EMPTY]])
//CHECK-SAME:          -> (tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>)
//CHECK-NEXT:      [[COUNT:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
//CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 20, [[COUNT]], 1024] [1, 1, 1, 1]
//CHECK-SAME:          to tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 205, 1024]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceMax([[SLICE]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tiling_loop_index = 0 : i64}
//CHECK-SAME:          -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 1]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, [[IDX]], 0] [1, 1, [[COUNT]], 1] [1, 1, 1, 1]
//CHECK-SAME:          into tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK-NEXT:      scf.yield [[INSERT]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        return [[SCF]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 205)>

//CHECK-LABEL: @TileReduceMean
//CHECK-SAME:  [[ARG0:%.+]]: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
func.func @TileReduceMean(%arg0: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>) -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}> {
     %0 = VPU.ReduceMean(%arg0) {
         axes_value = [1, 3],
         keep_dims,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         tilingStrategy = [1, 1, 5, 1]
     } : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}> -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
     return %0 : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>

//CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
//CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
//CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C205]] iter_args([[OUT:%.+]] = [[EMPTY]])
//CHECK-SAME:          -> (tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>)
//CHECK-NEXT:      [[COUNT:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
//CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 20, [[COUNT]], 1024] [1, 1, 1, 1]
//CHECK-SAME:          to tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 205, 1024]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceMean([[SLICE]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tiling_loop_index = 0 : i64}
//CHECK-SAME:          -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 1]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, [[IDX]], 0] [1, 1, [[COUNT]], 1] [1, 1, 1, 1]
//CHECK-SAME:          into tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK-NEXT:      scf.yield [[INSERT]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        return [[SCF]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 205)>

//CHECK-LABEL: @TileReduceSum
//CHECK-SAME:  [[ARG0:%.+]]: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
func.func @TileReduceSum(%arg0: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>) -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}> {
     %0 = VPU.ReduceSum(%arg0) {
         axes_value = [1, 3],
         keep_dims,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         tilingStrategy = [1, 1, 5, 1]
     } : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}> -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
     return %0 : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>

//CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
//CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
//CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C205]] iter_args([[OUT:%.+]] = [[EMPTY]])
//CHECK-SAME:          -> (tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>)
//CHECK-NEXT:      [[COUNT:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
//CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 20, [[COUNT]], 1024] [1, 1, 1, 1]
//CHECK-SAME:          to tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 205, 1024]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceSum([[SLICE]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tiling_loop_index = 0 : i64}
//CHECK-SAME:          -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 1]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, [[IDX]], 0] [1, 1, [[COUNT]], 1] [1, 1, 1, 1]
//CHECK-SAME:          into tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK-NEXT:      scf.yield [[INSERT]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        return [[SCF]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 205)>

//CHECK-LABEL: @TileReduceProd
//CHECK-SAME:  [[ARG0:%.+]]: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
func.func @TileReduceProd(%arg0: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>) -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}> {
     %0 = VPU.ReduceProd(%arg0) {
         axes_value = [1, 3],
         keep_dims,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         tilingStrategy = [1, 1, 5, 1]
     } : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}> -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
     return %0 : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>

//CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
//CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
//CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C205]] iter_args([[OUT:%.+]] = [[EMPTY]])
//CHECK-SAME:          -> (tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>)
//CHECK-NEXT:      [[COUNT:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
//CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 20, [[COUNT]], 1024] [1, 1, 1, 1]
//CHECK-SAME:          to tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 205, 1024]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceProd([[SLICE]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tiling_loop_index = 0 : i64}
//CHECK-SAME:          -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 1]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, [[IDX]], 0] [1, 1, [[COUNT]], 1] [1, 1, 1, 1]
//CHECK-SAME:          into tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK-NEXT:      scf.yield [[INSERT]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        return [[SCF]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 205)>

//CHECK-LABEL: @TileReduceSquare
//CHECK-SAME:  [[ARG0:%.+]]: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
func.func @TileReduceSquare(%arg0: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>) -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}> {
     %0 = VPU.ReduceSquare(%arg0) {
         axes_value = [1, 3],
         keep_dims,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         tilingStrategy = [1, 1, 5, 1]
     } : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}> -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
     return %0 : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>

//CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
//CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
//CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C205]] iter_args([[OUT:%.+]] = [[EMPTY]])
//CHECK-SAME:          -> (tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>)
//CHECK-NEXT:      [[COUNT:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
//CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 20, [[COUNT]], 1024] [1, 1, 1, 1]
//CHECK-SAME:          to tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 205, 1024]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceSquare([[SLICE]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tiling_loop_index = 0 : i64}
//CHECK-SAME:          -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 1]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, [[IDX]], 0] [1, 1, [[COUNT]], 1] [1, 1, 1, 1]
//CHECK-SAME:          into tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK-NEXT:      scf.yield [[INSERT]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        return [[SCF]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 205)>

//CHECK-LABEL: @TileReduceLogicalOr
//CHECK-SAME:  [[ARG0:%.+]]: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
func.func @TileReduceLogicalOr(%arg0: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>) -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}> {
     %0 = VPU.ReduceLogicalOr(%arg0) {
         axes_value = [1, 3],
         keep_dims,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         tilingStrategy = [1, 1, 5, 1]
     } : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}> -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
     return %0 : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>

//CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
//CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
//CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C205]] iter_args([[OUT:%.+]] = [[EMPTY]])
//CHECK-SAME:          -> (tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>)
//CHECK-NEXT:      [[COUNT:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
//CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 20, [[COUNT]], 1024] [1, 1, 1, 1]
//CHECK-SAME:          to tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 205, 1024]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceLogicalOr([[SLICE]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tiling_loop_index = 0 : i64}
//CHECK-SAME:          -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 1]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, [[IDX]], 0] [1, 1, [[COUNT]], 1] [1, 1, 1, 1]
//CHECK-SAME:          into tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK-NEXT:      scf.yield [[INSERT]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        return [[SCF]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 205)>

//CHECK-LABEL: @TileReduceLogicalAnd
//CHECK-SAME:  [[ARG0:%.+]]: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
func.func @TileReduceLogicalAnd(%arg0: tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>) -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}> {
     %0 = VPU.ReduceLogicalAnd(%arg0) {
         axes_value = [1, 3],
         keep_dims,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         tilingStrategy = [1, 1, 5, 1]
     } : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}> -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
     return %0 : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>

//CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
//CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
//CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
//CHECK-DAG:    [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 1024, 1024]> : tensor<4xsi64>}>
//CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C205]] iter_args([[OUT:%.+]] = [[EMPTY]])
//CHECK-SAME:          -> (tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>)
//CHECK-NEXT:      [[COUNT:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[DIM]]]
//CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 20, [[COUNT]], 1024] [1, 1, 1, 1]
//CHECK-SAME:          to tensor<1x20x?x1024xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 205, 1024]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceLogicalAnd([[SLICE]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tiling_loop_index = 0 : i64}
//CHECK-SAME:          -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 1]> : tensor<4xsi64>, order = #NCHW}>
//CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, [[IDX]], 0] [1, 1, [[COUNT]], 1] [1, 1, 1, 1]
//CHECK-SAME:          into tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK-NEXT:      scf.yield [[INSERT]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
//CHECK:        return [[SCF]] : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 1]> : tensor<4xsi64>}>
}

// -----

// Dynamic H and W dimensions with scaling using scales_attr (4x upscale)
// For ASYMMETRIC coord_mode with NEAREST mode and ROUND_PREFER_FLOOR:
// CHECK: #[[$MAP_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 200)>
// CHECK: #[[$MAP_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 160)>
// CHECK: #[[$MAP_CLAMP_START_H:.+]] = affine_map<(d0) -> ((d0 * 200 + 399) floordiv 800, 0)>
// CHECK: #[[$MAP_CLAMP_MAX_H:.+]] = affine_map<()[s0] -> (99, s0)>
// CHECK: #[[$MAP_CLAMP_START_W:.+]] = affine_map<(d0) -> ((d0 * 160 + 319) floordiv 640, 0)>
// CHECK: #[[$MAP_CLAMP_MAX_W:.+]] = affine_map<()[s0] -> (79, s0)>
// CHECK: #[[$MAP_END_COORD:.+]] = affine_map<(d0, d1) -> ((d0 + d1) floordiv 4, 0)>
// CHECK: #[[$MAP_SIZE:.+]] = affine_map<(d0, d1) -> (-d0 + d1 + 1)>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ApplyTilingInterpDynamic2DWithScale
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
func.func @ApplyTilingInterpDynamic2DWithScale(
        %input1: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3], scales_attr = [4.0, 4.0], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            tilingStrategy = [1, 1, 2, 2]} :
        tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant 160 : index
    //CHECK-DAG: [[LOOP_STEP_H:%.+]] = arith.constant 200 : index
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[SCALE:%.+]] = arith.constant 4.000000e+00 : f64
    //CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index

    //CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[C2]]
    //CHECK: [[DIM_H_I64:%.+]] = arith.index_cast [[DIM_H]] : index to i64
    //CHECK: [[DIM_H_F64:%.+]] = arith.sitofp [[DIM_H_I64]] : i64 to f64
    //CHECK: [[OUT_H_F64:%.+]] = arith.mulf [[DIM_H_F64]], [[SCALE]] : f64
    //CHECK: [[OUT_H_I64:%.+]] = arith.fptosi [[OUT_H_F64]] : f64 to i64
    //CHECK: [[OUT_H:%.+]] = arith.index_cast [[OUT_H_I64]] : i64 to index

    //CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[C3]]
    //CHECK: [[DIM_W_I64:%.+]] = arith.index_cast [[DIM_W]] : index to i64
    //CHECK: [[DIM_W_F64:%.+]] = arith.sitofp [[DIM_W_I64]] : i64 to f64
    //CHECK: [[OUT_W_F64:%.+]] = arith.mulf [[DIM_W_F64]], [[SCALE]] : f64
    //CHECK: [[OUT_W_I64:%.+]] = arith.fptosi [[OUT_W_F64]] : f64 to i64
    //CHECK: [[OUT_W:%.+]] = arith.index_cast [[OUT_W_I64]] : i64 to index

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[OUT_H]], [[OUT_W]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[H_LOOP:%.+]] = scf.for
    //CHECK-SAME:             [[H_LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[OUT_H]] step [[LOOP_STEP_H]]
    //CHECK-SAME:             iter_args([[H_LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>)

    //CHECK:                  [[W_LOOP:%.+]] = scf.for
    //CHECK-SAME:                              [[W_LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[OUT_W]] step [[LOOP_STEP_W]]
    //CHECK-SAME:                              iter_args([[W_LOOP_OUT:%arg[0-9]]]  = [[H_LOOP_OUT]])

    //CHECK:                                   [[OUT_TILE_SIZE_H:%.+]] = affine.min #[[$MAP_MIN_H]]([[H_LOOP_ITER]])[[[OUT_H]]]
    //CHECK:                                   [[OUT_TILE_SIZE_W:%.+]] = affine.min #[[$MAP_MIN_W]]([[W_LOOP_ITER]])[[[OUT_W]]]

    //CHECK:                                   [[START_H_UNCLAMPED:%.+]] = affine.max #[[$MAP_CLAMP_START_H]]([[H_LOOP_ITER]])
    //CHECK:                                   [[START_H:%.+]] = affine.min #[[$MAP_CLAMP_MAX_H]]()[[[START_H_UNCLAMPED]]]
    //CHECK:                                   [[START_W_UNCLAMPED:%.+]] = affine.max #[[$MAP_CLAMP_START_W]]([[W_LOOP_ITER]])
    //CHECK:                                   [[START_W:%.+]] = affine.min #[[$MAP_CLAMP_MAX_W]]()[[[START_W_UNCLAMPED]]]

    //CHECK:                                   [[END_H_UNCLAMPED:%.+]] = affine.max #[[$MAP_END_COORD]]([[H_LOOP_ITER]], [[OUT_TILE_SIZE_H]])
    //CHECK:                                   [[END_H:%.+]] = affine.min #[[$MAP_CLAMP_MAX_H]]()[[[END_H_UNCLAMPED]]]
    //CHECK:                                   [[END_W_UNCLAMPED:%.+]] = affine.max #[[$MAP_END_COORD]]([[W_LOOP_ITER]], [[OUT_TILE_SIZE_W]])
    //CHECK:                                   [[END_W:%.+]] = affine.min #[[$MAP_CLAMP_MAX_W]]()[[[END_W_UNCLAMPED]]]

    //CHECK:                                   [[SLICE_SIZE_H:%.+]] = affine.apply #[[$MAP_SIZE]]([[START_H]], [[END_H]])
    //CHECK:                                   [[SLICE_SIZE_W:%.+]] = affine.apply #[[$MAP_SIZE]]([[START_W]], [[END_W]])

    //CHECK:                                   [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[START_H]], [[START_W]]] [1, 32, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:                              : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:                              to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 40]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                                   [[INTERP:%.+]] = VPU.Interpolate([[SLICE]])
    //CHECK-SAME:                              attr = #IE.Interpolate<mode = <NEAREST>
    //CHECK-SAME:                              coord_mode = <ASYMMETRIC>
    //CHECK-SAME:                              nearest_mode = <ROUND_PREFER_FLOOR>
    //CHECK-SAME:                              scales_attr = [4.000000e+00, 4.000000e+00]
    //CHECK-SAME:                              -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 200, 160]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                                   [[INSERT:%.+]] = tensor.insert_slice [[INTERP]] into [[W_LOOP_OUT]][0, 0, [[H_LOOP_ITER]], [[W_LOOP_ITER]]] [1, 32, [[OUT_TILE_SIZE_H]], [[OUT_TILE_SIZE_W]]]
    //CHECK-SAME:                              : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 200, 160]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:                              into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:  scf.yield [[INSERT]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:  scf.yield [[W_LOOP]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:  return [[H_LOOP]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

// Dynamic H and W dimensions with HALF_PIXEL coord_mode and LINEAR interpolation (4x upscale)
// For HALF_PIXEL coord_mode with LINEAR mode:
// - inCoord = (outCoord + 0.5) / scale - 0.5 = (outCoord - 1.5) / 4 for scale=4
// - Start uses floor: floor((2*outCoord - 3) / 8)
// - End uses ceil: ceil((2*(outCoord + tileSize) - 5) / 8) = ceil((2*outCoord + 2*tileSize - 5) / 8)
// CHECK: #[[$MAP2_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 200)>
// CHECK: #[[$MAP2_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 160)>
// CHECK: #[[$MAP2_FLOOR_COORD:.+]] = affine_map<(d0) -> ((d0 * 2 - 3) floordiv 8, 0)>
// CHECK: #[[$MAP2_CLAMP_MAX_H:.+]] = affine_map<()[s0] -> (99, s0)>
// CHECK: #[[$MAP2_CLAMP_MAX_W:.+]] = affine_map<()[s0] -> (79, s0)>
// CHECK: #[[$MAP2_CEIL_COORD:.+]] = affine_map<(d0, d1) -> ((d0 * 2 + d1 * 2 - 5) ceildiv 8, 0)>
// CHECK: #[[$MAP2_SIZE:.+]] = affine_map<(d0, d1) -> (-d0 + d1 + 1)>

#NHWC2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ApplyTilingInterpHalfPixelLinear
// CHECK-SAME:      [[INPUT2:%arg[0-9]]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
func.func @ApplyTilingInterpHalfPixelLinear(
        %input1: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC2}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC2}> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3], scales_attr = [4.0, 4.0], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            tilingStrategy = [1, 1, 2, 2]} :
        tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC2}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC2}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC2}>

    //CHECK-DAG: [[LOOP2_STEP_W:%.+]] = arith.constant 160 : index
    //CHECK-DAG: [[LOOP2_STEP_H:%.+]] = arith.constant 200 : index
    //CHECK-DAG: [[LOOP2_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[C3_2:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[SCALE2:%.+]] = arith.constant 4.000000e+00 : f64
    //CHECK-DAG: [[C2_2:%.+]] = arith.constant 2 : index

    //CHECK: [[DIM2_H:%.+]] = tensor.dim [[INPUT2]], [[C2_2]]
    //CHECK: [[DIM2_H_I64:%.+]] = arith.index_cast [[DIM2_H]] : index to i64
    //CHECK: [[DIM2_H_F64:%.+]] = arith.sitofp [[DIM2_H_I64]] : i64 to f64
    //CHECK: [[OUT2_H_F64:%.+]] = arith.mulf [[DIM2_H_F64]], [[SCALE2]] : f64
    //CHECK: [[OUT2_H_I64:%.+]] = arith.fptosi [[OUT2_H_F64]] : f64 to i64
    //CHECK: [[OUT2_H:%.+]] = arith.index_cast [[OUT2_H_I64]] : i64 to index

    //CHECK: [[DIM2_W:%.+]] = tensor.dim [[INPUT2]], [[C3_2]]
    //CHECK: [[DIM2_W_I64:%.+]] = arith.index_cast [[DIM2_W]] : index to i64
    //CHECK: [[DIM2_W_F64:%.+]] = arith.sitofp [[DIM2_W_I64]] : i64 to f64
    //CHECK: [[OUT2_W_F64:%.+]] = arith.mulf [[DIM2_W_F64]], [[SCALE2]] : f64
    //CHECK: [[OUT2_W_I64:%.+]] = arith.fptosi [[OUT2_W_F64]] : f64 to i64
    //CHECK: [[OUT2_W:%.+]] = arith.index_cast [[OUT2_W_I64]] : i64 to index

    //CHECK: [[LOOP2_OUTPUT:%.+]] = tensor.empty([[OUT2_H]], [[OUT2_W]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[H_LOOP2:%.+]] = scf.for
    //CHECK-SAME:             [[H_LOOP2_ITER:%arg[0-9]]] = [[LOOP2_BEGIN]] to [[OUT2_H]] step [[LOOP2_STEP_H]]
    //CHECK-SAME:             iter_args([[H_LOOP2_OUT:%arg[0-9]]]  = [[LOOP2_OUTPUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>)

    //CHECK:                  [[W_LOOP2:%.+]] = scf.for
    //CHECK-SAME:                              [[W_LOOP2_ITER:%arg[0-9]]] = [[LOOP2_BEGIN]] to [[OUT2_W]] step [[LOOP2_STEP_W]]
    //CHECK-SAME:                              iter_args([[W_LOOP2_OUT:%arg[0-9]]]  = [[H_LOOP2_OUT]])

    //CHECK:                                   [[OUT2_TILE_SIZE_H:%.+]] = affine.min #[[$MAP2_MIN_H]]([[H_LOOP2_ITER]])[[[OUT2_H]]]
    //CHECK:                                   [[OUT2_TILE_SIZE_W:%.+]] = affine.min #[[$MAP2_MIN_W]]([[W_LOOP2_ITER]])[[[OUT2_W]]]

    //CHECK:                                   [[START2_H_UNCLAMPED:%.+]] = affine.max #[[$MAP2_FLOOR_COORD]]([[H_LOOP2_ITER]])
    //CHECK:                                   [[START2_H:%.+]] = affine.min #[[$MAP2_CLAMP_MAX_H]]()[[[START2_H_UNCLAMPED]]]
    //CHECK:                                   [[START2_W_UNCLAMPED:%.+]] = affine.max #[[$MAP2_FLOOR_COORD]]([[W_LOOP2_ITER]])
    //CHECK:                                   [[START2_W:%.+]] = affine.min #[[$MAP2_CLAMP_MAX_W]]()[[[START2_W_UNCLAMPED]]]

    //CHECK:                                   [[END2_H_UNCLAMPED:%.+]] = affine.max #[[$MAP2_CEIL_COORD]]([[H_LOOP2_ITER]], [[OUT2_TILE_SIZE_H]])
    //CHECK:                                   [[END2_H:%.+]] = affine.min #[[$MAP2_CLAMP_MAX_H]]()[[[END2_H_UNCLAMPED]]]
    //CHECK:                                   [[END2_W_UNCLAMPED:%.+]] = affine.max #[[$MAP2_CEIL_COORD]]([[W_LOOP2_ITER]], [[OUT2_TILE_SIZE_W]])
    //CHECK:                                   [[END2_W:%.+]] = affine.min #[[$MAP2_CLAMP_MAX_W]]()[[[END2_W_UNCLAMPED]]]

    //CHECK:                                   [[SLICE2_SIZE_H:%.+]] = affine.apply #[[$MAP2_SIZE]]([[START2_H]], [[END2_H]])
    //CHECK:                                   [[SLICE2_SIZE_W:%.+]] = affine.apply #[[$MAP2_SIZE]]([[START2_W]], [[END2_W]])

    //CHECK:                                   [[SLICE2:%.+]] = tensor.extract_slice [[INPUT2]][0, 0, [[START2_H]], [[START2_W]]] [1, 32, [[SLICE2_SIZE_H]], [[SLICE2_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:                              : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:                              to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 40]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                                   [[INTERP2:%.+]] = VPU.Interpolate([[SLICE2]])
    //CHECK-SAME:                              attr = #IE.Interpolate<mode = <LINEAR>
    //CHECK-SAME:                              coord_mode = <HALF_PIXEL>
    //CHECK-SAME:                              nearest_mode = <FLOOR>
    //CHECK-SAME:                              scales_attr = [4.000000e+00, 4.000000e+00]
    //CHECK-SAME:                              -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 200, 160]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                                   [[INSERT2:%.+]] = tensor.insert_slice [[INTERP2]] into [[W_LOOP2_OUT]][0, 0, [[H_LOOP2_ITER]], [[W_LOOP2_ITER]]] [1, 32, [[OUT2_TILE_SIZE_H]], [[OUT2_TILE_SIZE_W]]]
    //CHECK-SAME:                              : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 200, 160]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:                              into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:  scf.yield [[INSERT2]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:  scf.yield [[W_LOOP2]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:  return [[H_LOOP2]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

// Dynamic H and W dimensions with PYTORCH_HALF_PIXEL coord_mode and CUBIC interpolation (4x upscale)
// For PYTORCH_HALF_PIXEL coord_mode (same as HALF_PIXEL when outSize > 1):
// - inCoord = (outCoord + 0.5) / scale - 0.5 = (outCoord - 1.5) / 4 for scale=4
// For CUBIC mode:
// - Start uses floor(inCoord) - 1: floor((2*outCoord - 3) / 8) - 1
// - End uses floor(inCoord) + 2: floor((2*outCoord - 3) / 8) + 2
// CHECK: #[[$MAP3_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 200)>
// CHECK: #[[$MAP3_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 160)>
// CHECK: #[[$MAP3_CUBIC_START:.+]] = affine_map<(d0) -> ((d0 * 2 - 3) floordiv 8 - 1, 0)>
// CHECK: #[[$MAP3_CLAMP_MAX_H:.+]] = affine_map<()[s0] -> (99, s0)>
// CHECK: #[[$MAP3_CLAMP_MAX_W:.+]] = affine_map<()[s0] -> (79, s0)>
// CHECK: #[[$MAP3_CUBIC_END:.+]] = affine_map<(d0, d1) -> ((d0 * 2 + d1 * 2 - 5) floordiv 8 + 2, 0)>
// CHECK: #[[$MAP3_SIZE:.+]] = affine_map<(d0, d1) -> (-d0 + d1 + 1)>

#NHWC3 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ApplyTilingInterpPytorchHalfPixelCubic
// CHECK-SAME:      [[INPUT3:%arg[0-9]]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
func.func @ApplyTilingInterpPytorchHalfPixelCubic(
        %input1: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC3}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC3}> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <CUBIC>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3], scales_attr = [4.0, 4.0], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            tilingStrategy = [1, 1, 2, 2]} :
        tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC3}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC3}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC3}>

    //CHECK-DAG: [[LOOP3_STEP_W:%.+]] = arith.constant 160 : index
    //CHECK-DAG: [[LOOP3_STEP_H:%.+]] = arith.constant 200 : index
    //CHECK-DAG: [[LOOP3_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[C3_3:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[SCALE3:%.+]] = arith.constant 4.000000e+00 : f64
    //CHECK-DAG: [[C2_3:%.+]] = arith.constant 2 : index

    //CHECK: [[DIM3_H:%.+]] = tensor.dim [[INPUT3]], [[C2_3]]
    //CHECK: [[DIM3_H_I64:%.+]] = arith.index_cast [[DIM3_H]] : index to i64
    //CHECK: [[DIM3_H_F64:%.+]] = arith.sitofp [[DIM3_H_I64]] : i64 to f64
    //CHECK: [[OUT3_H_F64:%.+]] = arith.mulf [[DIM3_H_F64]], [[SCALE3]] : f64
    //CHECK: [[OUT3_H_I64:%.+]] = arith.fptosi [[OUT3_H_F64]] : f64 to i64
    //CHECK: [[OUT3_H:%.+]] = arith.index_cast [[OUT3_H_I64]] : i64 to index

    //CHECK: [[DIM3_W:%.+]] = tensor.dim [[INPUT3]], [[C3_3]]
    //CHECK: [[DIM3_W_I64:%.+]] = arith.index_cast [[DIM3_W]] : index to i64
    //CHECK: [[DIM3_W_F64:%.+]] = arith.sitofp [[DIM3_W_I64]] : i64 to f64
    //CHECK: [[OUT3_W_F64:%.+]] = arith.mulf [[DIM3_W_F64]], [[SCALE3]] : f64
    //CHECK: [[OUT3_W_I64:%.+]] = arith.fptosi [[OUT3_W_F64]] : f64 to i64
    //CHECK: [[OUT3_W:%.+]] = arith.index_cast [[OUT3_W_I64]] : i64 to index

    //CHECK: [[LOOP3_OUTPUT:%.+]] = tensor.empty([[OUT3_H]], [[OUT3_W]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[H_LOOP3:%.+]] = scf.for
    //CHECK-SAME:             [[H_LOOP3_ITER:%arg[0-9]]] = [[LOOP3_BEGIN]] to [[OUT3_H]] step [[LOOP3_STEP_H]]
    //CHECK-SAME:             iter_args([[H_LOOP3_OUT:%arg[0-9]]]  = [[LOOP3_OUTPUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>)

    //CHECK:                  [[W_LOOP3:%.+]] = scf.for
    //CHECK-SAME:                              [[W_LOOP3_ITER:%arg[0-9]]] = [[LOOP3_BEGIN]] to [[OUT3_W]] step [[LOOP3_STEP_W]]
    //CHECK-SAME:                              iter_args([[W_LOOP3_OUT:%arg[0-9]]]  = [[H_LOOP3_OUT]])

    //CHECK:                                   [[OUT3_TILE_SIZE_H:%.+]] = affine.min #[[$MAP3_MIN_H]]([[H_LOOP3_ITER]])[[[OUT3_H]]]
    //CHECK:                                   [[OUT3_TILE_SIZE_W:%.+]] = affine.min #[[$MAP3_MIN_W]]([[W_LOOP3_ITER]])[[[OUT3_W]]]

    //CHECK:                                   [[START3_H_UNCLAMPED:%.+]] = affine.max #[[$MAP3_CUBIC_START]]([[H_LOOP3_ITER]])
    //CHECK:                                   [[START3_H:%.+]] = affine.min #[[$MAP3_CLAMP_MAX_H]]()[[[START3_H_UNCLAMPED]]]
    //CHECK:                                   [[START3_W_UNCLAMPED:%.+]] = affine.max #[[$MAP3_CUBIC_START]]([[W_LOOP3_ITER]])
    //CHECK:                                   [[START3_W:%.+]] = affine.min #[[$MAP3_CLAMP_MAX_W]]()[[[START3_W_UNCLAMPED]]]

    //CHECK:                                   [[END3_H_UNCLAMPED:%.+]] = affine.max #[[$MAP3_CUBIC_END]]([[H_LOOP3_ITER]], [[OUT3_TILE_SIZE_H]])
    //CHECK:                                   [[END3_H:%.+]] = affine.min #[[$MAP3_CLAMP_MAX_H]]()[[[END3_H_UNCLAMPED]]]
    //CHECK:                                   [[END3_W_UNCLAMPED:%.+]] = affine.max #[[$MAP3_CUBIC_END]]([[W_LOOP3_ITER]], [[OUT3_TILE_SIZE_W]])
    //CHECK:                                   [[END3_W:%.+]] = affine.min #[[$MAP3_CLAMP_MAX_W]]()[[[END3_W_UNCLAMPED]]]

    //CHECK:                                   [[SLICE3_SIZE_H:%.+]] = affine.apply #[[$MAP3_SIZE]]([[START3_H]], [[END3_H]])
    //CHECK:                                   [[SLICE3_SIZE_W:%.+]] = affine.apply #[[$MAP3_SIZE]]([[START3_W]], [[END3_W]])

    //CHECK:                                   [[SLICE3:%.+]] = tensor.extract_slice [[INPUT3]][0, 0, [[START3_H]], [[START3_W]]] [1, 32, [[SLICE3_SIZE_H]], [[SLICE3_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:                              : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:                              to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 40]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                                   [[INTERP3:%.+]] = VPU.Interpolate([[SLICE3]])
    //CHECK-SAME:                              attr = #IE.Interpolate<mode = <CUBIC>
    //CHECK-SAME:                              coord_mode = <PYTORCH_HALF_PIXEL>
    //CHECK-SAME:                              cube_coeff = -7.500000e-01
    //CHECK-SAME:                              scales_attr = [4.000000e+00, 4.000000e+00]
    //CHECK-SAME:                              -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 200, 160]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                                   [[INSERT3:%.+]] = tensor.insert_slice [[INTERP3]] into [[W_LOOP3_OUT]][0, 0, [[H_LOOP3_ITER]], [[W_LOOP3_ITER]]] [1, 32, [[OUT3_TILE_SIZE_H]], [[OUT3_TILE_SIZE_W]]]
    //CHECK-SAME:                              : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 200, 160]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:                              into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:  scf.yield [[INSERT3]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:  scf.yield [[W_LOOP3]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:  return [[H_LOOP3]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

// Dynamic H and W dimensions with TF_HALF_PIXEL_FOR_NN coord_mode and NEAREST/CEIL (4x upscale)
// For TF_HALF_PIXEL_FOR_NN coord_mode:
// - inCoord = (outCoord + 0.5) * scale = (outCoord + 0.5) / 4 for scale=1/4 (but we have 4x upscale so scale=0.25)
// - Actually for upscale: inCoord = (outCoord + 0.5) * (inSize/outSize) = (outCoord + 0.5) / 4
// For NEAREST with CEIL:
// - nearestDim = ceil(inCoord) = ceil((2*outCoord + 1) / 8)

// CHECK: #[[$MAP4_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 200)>
// CHECK: #[[$MAP4_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 160)>
// CHECK: #[[$MAP4_CEIL_COORD:.+]] = affine_map<(d0) -> ((d0 * 2 + 1) ceildiv 8, 0)>
// CHECK: #[[$MAP4_CLAMP_MAX_H:.+]] = affine_map<()[s0] -> (99, s0)>
// CHECK: #[[$MAP4_CLAMP_MAX_W:.+]] = affine_map<()[s0] -> (79, s0)>
// CHECK: #[[$MAP4_END_COORD:.+]] = affine_map<(d0, d1) -> ((d0 * 2 + d1 * 2 - 1) ceildiv 8, 0)>
// CHECK: #[[$MAP4_SIZE:.+]] = affine_map<(d0, d1) -> (-d0 + d1 + 1)>

#NHWC4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ApplyTilingInterpTfHalfPixelCeil
// CHECK-SAME:      [[INPUT4:%arg[0-9]]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
func.func @ApplyTilingInterpTfHalfPixelCeil(
        %input1: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC4}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC4}> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <TF_HALF_PIXEL_FOR_NN>, cube_coeff = -7.500000e-01, mode = <NEAREST>, nearest_mode = <CEIL>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3], scales_attr = [4.0, 4.0], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            tilingStrategy = [1, 1, 2, 2]} :
        tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC4}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC4}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC4}>

    //CHECK-DAG: [[LOOP4_STEP_W:%.+]] = arith.constant 160 : index
    //CHECK-DAG: [[LOOP4_STEP_H:%.+]] = arith.constant 200 : index
    //CHECK-DAG: [[LOOP4_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[C3_4:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[SCALE4:%.+]] = arith.constant 4.000000e+00 : f64
    //CHECK-DAG: [[C2_4:%.+]] = arith.constant 2 : index

    //CHECK: [[DIM4_H:%.+]] = tensor.dim [[INPUT4]], [[C2_4]]
    //CHECK: [[DIM4_H_I64:%.+]] = arith.index_cast [[DIM4_H]] : index to i64
    //CHECK: [[DIM4_H_F64:%.+]] = arith.sitofp [[DIM4_H_I64]] : i64 to f64
    //CHECK: [[OUT4_H_F64:%.+]] = arith.mulf [[DIM4_H_F64]], [[SCALE4]] : f64
    //CHECK: [[OUT4_H_I64:%.+]] = arith.fptosi [[OUT4_H_F64]] : f64 to i64
    //CHECK: [[OUT4_H:%.+]] = arith.index_cast [[OUT4_H_I64]] : i64 to index

    //CHECK: [[DIM4_W:%.+]] = tensor.dim [[INPUT4]], [[C3_4]]
    //CHECK: [[DIM4_W_I64:%.+]] = arith.index_cast [[DIM4_W]] : index to i64
    //CHECK: [[DIM4_W_F64:%.+]] = arith.sitofp [[DIM4_W_I64]] : i64 to f64
    //CHECK: [[OUT4_W_F64:%.+]] = arith.mulf [[DIM4_W_F64]], [[SCALE4]] : f64
    //CHECK: [[OUT4_W_I64:%.+]] = arith.fptosi [[OUT4_W_F64]] : f64 to i64
    //CHECK: [[OUT4_W:%.+]] = arith.index_cast [[OUT4_W_I64]] : i64 to index

    //CHECK: [[LOOP4_OUTPUT:%.+]] = tensor.empty([[OUT4_H]], [[OUT4_W]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[H_LOOP4:%.+]] = scf.for
    //CHECK-SAME:             [[H_LOOP4_ITER:%arg[0-9]]] = [[LOOP4_BEGIN]] to [[OUT4_H]] step [[LOOP4_STEP_H]]
    //CHECK-SAME:             iter_args([[H_LOOP4_OUT:%arg[0-9]]]  = [[LOOP4_OUTPUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>)

    //CHECK:                  [[W_LOOP4:%.+]] = scf.for
    //CHECK-SAME:                              [[W_LOOP4_ITER:%arg[0-9]]] = [[LOOP4_BEGIN]] to [[OUT4_W]] step [[LOOP4_STEP_W]]
    //CHECK-SAME:                              iter_args([[W_LOOP4_OUT:%arg[0-9]]]  = [[H_LOOP4_OUT]])

    //CHECK:                                   [[OUT4_TILE_SIZE_H:%.+]] = affine.min #[[$MAP4_MIN_H]]([[H_LOOP4_ITER]])[[[OUT4_H]]]
    //CHECK:                                   [[OUT4_TILE_SIZE_W:%.+]] = affine.min #[[$MAP4_MIN_W]]([[W_LOOP4_ITER]])[[[OUT4_W]]]

    //CHECK:                                   [[START4_H_UNCLAMPED:%.+]] = affine.max #[[$MAP4_CEIL_COORD]]([[H_LOOP4_ITER]])
    //CHECK:                                   [[START4_H:%.+]] = affine.min #[[$MAP4_CLAMP_MAX_H]]()[[[START4_H_UNCLAMPED]]]
    //CHECK:                                   [[START4_W_UNCLAMPED:%.+]] = affine.max #[[$MAP4_CEIL_COORD]]([[W_LOOP4_ITER]])
    //CHECK:                                   [[START4_W:%.+]] = affine.min #[[$MAP4_CLAMP_MAX_W]]()[[[START4_W_UNCLAMPED]]]

    //CHECK:                                   [[END4_H_UNCLAMPED:%.+]] = affine.max #[[$MAP4_END_COORD]]([[H_LOOP4_ITER]], [[OUT4_TILE_SIZE_H]])
    //CHECK:                                   [[END4_H:%.+]] = affine.min #[[$MAP4_CLAMP_MAX_H]]()[[[END4_H_UNCLAMPED]]]
    //CHECK:                                   [[END4_W_UNCLAMPED:%.+]] = affine.max #[[$MAP4_END_COORD]]([[W_LOOP4_ITER]], [[OUT4_TILE_SIZE_W]])
    //CHECK:                                   [[END4_W:%.+]] = affine.min #[[$MAP4_CLAMP_MAX_W]]()[[[END4_W_UNCLAMPED]]]

    //CHECK:                                   [[SLICE4_SIZE_H:%.+]] = affine.apply #[[$MAP4_SIZE]]([[START4_H]], [[END4_H]])
    //CHECK:                                   [[SLICE4_SIZE_W:%.+]] = affine.apply #[[$MAP4_SIZE]]([[START4_W]], [[END4_W]])

    //CHECK:                                   [[SLICE4:%.+]] = tensor.extract_slice [[INPUT4]][0, 0, [[START4_H]], [[START4_W]]] [1, 32, [[SLICE4_SIZE_H]], [[SLICE4_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:                              : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:                              to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 40]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                                   [[INTERP4:%.+]] = VPU.Interpolate([[SLICE4]])
    //CHECK-SAME:                              attr = #IE.Interpolate<mode = <NEAREST>
    //CHECK-SAME:                              coord_mode = <TF_HALF_PIXEL_FOR_NN>
    //CHECK-SAME:                              nearest_mode = <CEIL>
    //CHECK-SAME:                              scales_attr = [4.000000e+00, 4.000000e+00]
    //CHECK-SAME:                              -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 200, 160]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                                   [[INSERT4:%.+]] = tensor.insert_slice [[INTERP4]] into [[W_LOOP4_OUT]][0, 0, [[H_LOOP4_ITER]], [[W_LOOP4_ITER]]] [1, 32, [[OUT4_TILE_SIZE_H]], [[OUT4_TILE_SIZE_W]]]
    //CHECK-SAME:                              : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 200, 160]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:                              into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:  scf.yield [[INSERT4]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:  scf.yield [[W_LOOP4]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:  return [[H_LOOP4]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

// Dynamic H and W dimensions with ALIGN_CORNERS coord_mode and LINEAR (4x upscale)
// For ALIGN_CORNERS coord_mode:
// - inCoord = outCoord * (inSize - 1) / (outSize - 1)
// - For inSize=100, outSize=400: inCoord = outCoord * 99 / 399
// For LINEAR mode:
// - Start uses floor: floor(outCoord * (inSize-1) / (outSize-1))
// - End uses ceil: ceil((outCoord + tileSize - 1) * (inSize-1) / (outSize-1))

#NHWC5 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// ALIGN_CORNERS uses (inSize-1)/(outSize-1) ratios, simplified to coprime fractions
// CHECK: #[[$MAP5_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 200)>
// CHECK: #[[$MAP5_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 160)>
// CHECK: #[[$MAP5_FLOOR_H:.+]] = affine_map<(d0) -> ((d0 * 33) floordiv 133, 0)>
// CHECK: #[[$MAP5_CLAMP_MAX_H:.+]] = affine_map<()[s0] -> (99, s0)>
// CHECK: #[[$MAP5_FLOOR_W:.+]] = affine_map<(d0) -> ((d0 * 79) floordiv 319, 0)>
// CHECK: #[[$MAP5_CLAMP_MAX_W:.+]] = affine_map<()[s0] -> (79, s0)>
// CHECK: #[[$MAP5_CEIL_H:.+]] = affine_map<(d0, d1) -> ((d0 * 33 + d1 * 33 - 33) ceildiv 133, 0)>
// CHECK: #[[$MAP5_CEIL_W:.+]] = affine_map<(d0, d1) -> ((d0 * 79 + d1 * 79 - 79) ceildiv 319, 0)>
// CHECK: #[[$MAP5_SIZE:.+]] = affine_map<(d0, d1) -> (-d0 + d1 + 1)>

// CHECK-LABEL: @ApplyTilingInterpAlignCorners
// CHECK-SAME:      [[INPUT5:%arg[0-9]]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
func.func @ApplyTilingInterpAlignCorners(
        %input1: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC5}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC5}> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3], scales_attr = [4.0, 4.0], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            tilingStrategy = [1, 1, 2, 2]} :
        tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC5}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC5}>

    return %0 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC5}>

    //CHECK-DAG: [[LOOP5_STEP_W:%.+]] = arith.constant 160 : index
    //CHECK-DAG: [[LOOP5_STEP_H:%.+]] = arith.constant 200 : index
    //CHECK-DAG: [[LOOP5_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[C3_5:%.+]] = arith.constant 3 : index
    //CHECK-DAG: [[SCALE5:%.+]] = arith.constant 4.000000e+00 : f64
    //CHECK-DAG: [[C2_5:%.+]] = arith.constant 2 : index

    //CHECK: [[DIM5_H:%.+]] = tensor.dim [[INPUT5]], [[C2_5]]
    //CHECK: [[DIM5_H_I64:%.+]] = arith.index_cast [[DIM5_H]] : index to i64
    //CHECK: [[DIM5_H_F64:%.+]] = arith.sitofp [[DIM5_H_I64]] : i64 to f64
    //CHECK: [[OUT5_H_F64:%.+]] = arith.mulf [[DIM5_H_F64]], [[SCALE5]] : f64
    //CHECK: [[OUT5_H_I64:%.+]] = arith.fptosi [[OUT5_H_F64]] : f64 to i64
    //CHECK: [[OUT5_H:%.+]] = arith.index_cast [[OUT5_H_I64]] : i64 to index

    //CHECK: [[DIM5_W:%.+]] = tensor.dim [[INPUT5]], [[C3_5]]
    //CHECK: [[DIM5_W_I64:%.+]] = arith.index_cast [[DIM5_W]] : index to i64
    //CHECK: [[DIM5_W_F64:%.+]] = arith.sitofp [[DIM5_W_I64]] : i64 to f64
    //CHECK: [[OUT5_W_F64:%.+]] = arith.mulf [[DIM5_W_F64]], [[SCALE5]] : f64
    //CHECK: [[OUT5_W_I64:%.+]] = arith.fptosi [[OUT5_W_F64]] : f64 to i64
    //CHECK: [[OUT5_W:%.+]] = arith.index_cast [[OUT5_W_I64]] : i64 to index

    //CHECK: [[LOOP5_OUTPUT:%.+]] = tensor.empty([[OUT5_H]], [[OUT5_W]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[H_LOOP5:%.+]] = scf.for
    //CHECK-SAME:             [[H_LOOP5_ITER:%arg[0-9]]] = [[LOOP5_BEGIN]] to [[OUT5_H]] step [[LOOP5_STEP_H]]
    //CHECK-SAME:             iter_args([[H_LOOP5_OUT:%arg[0-9]]]  = [[LOOP5_OUTPUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>)

    //CHECK:                  [[W_LOOP5:%.+]] = scf.for
    //CHECK-SAME:                              [[W_LOOP5_ITER:%arg[0-9]]] = [[LOOP5_BEGIN]] to [[OUT5_W]] step [[LOOP5_STEP_W]]
    //CHECK-SAME:                              iter_args([[W_LOOP5_OUT:%arg[0-9]]]  = [[H_LOOP5_OUT]])

    //CHECK:                                   [[OUT5_TILE_SIZE_H:%.+]] = affine.min #[[$MAP5_MIN_H]]([[H_LOOP5_ITER]])[[[OUT5_H]]]
    //CHECK:                                   [[OUT5_TILE_SIZE_W:%.+]] = affine.min #[[$MAP5_MIN_W]]([[W_LOOP5_ITER]])[[[OUT5_W]]]

    //CHECK:                                   [[START5_H_UNCLAMPED:%.+]] = affine.max #[[$MAP5_FLOOR_H]]([[H_LOOP5_ITER]])
    //CHECK:                                   [[START5_H:%.+]] = affine.min #[[$MAP5_CLAMP_MAX_H]]()[[[START5_H_UNCLAMPED]]]
    //CHECK:                                   [[START5_W_UNCLAMPED:%.+]] = affine.max #[[$MAP5_FLOOR_W]]([[W_LOOP5_ITER]])
    //CHECK:                                   [[START5_W:%.+]] = affine.min #[[$MAP5_CLAMP_MAX_W]]()[[[START5_W_UNCLAMPED]]]

    //CHECK:                                   [[END5_H_UNCLAMPED:%.+]] = affine.max #[[$MAP5_CEIL_H]]([[H_LOOP5_ITER]], [[OUT5_TILE_SIZE_H]])
    //CHECK:                                   [[END5_H:%.+]] = affine.min #[[$MAP5_CLAMP_MAX_H]]()[[[END5_H_UNCLAMPED]]]
    //CHECK:                                   [[END5_W_UNCLAMPED:%.+]] = affine.max #[[$MAP5_CEIL_W]]([[W_LOOP5_ITER]], [[OUT5_TILE_SIZE_W]])
    //CHECK:                                   [[END5_W:%.+]] = affine.min #[[$MAP5_CLAMP_MAX_W]]()[[[END5_W_UNCLAMPED]]]

    //CHECK:                                   [[SLICE5_SIZE_H:%.+]] = affine.apply #[[$MAP5_SIZE]]([[START5_H]], [[END5_H]])
    //CHECK:                                   [[SLICE5_SIZE_W:%.+]] = affine.apply #[[$MAP5_SIZE]]([[START5_W]], [[END5_W]])

    //CHECK:                                   [[SLICE5:%.+]] = tensor.extract_slice [[INPUT5]][0, 0, [[START5_H]], [[START5_W]]] [1, 32, [[SLICE5_SIZE_H]], [[SLICE5_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:                              : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 80]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:                              to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 40]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                                   [[INTERP5:%.+]] = VPU.Interpolate([[SLICE5]])
    //CHECK-SAME:                              attr = #IE.Interpolate<mode = <LINEAR>
    //CHECK-SAME:                              coord_mode = <ALIGN_CORNERS>
    //CHECK-SAME:                              scales_attr = [4.000000e+00, 4.000000e+00]
    //CHECK-SAME:                              -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 200, 160]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                                   [[INSERT5:%.+]] = tensor.insert_slice [[INTERP5]] into [[W_LOOP5_OUT]][0, 0, [[H_LOOP5_ITER]], [[W_LOOP5_ITER]]] [1, 32, [[OUT5_TILE_SIZE_H]], [[OUT5_TILE_SIZE_W]]]
    //CHECK-SAME:                              : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 200, 160]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK-SAME:                              into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:  scf.yield [[INSERT5]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:  scf.yield [[W_LOOP5]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:  return [[H_LOOP5]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 400, 320]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

// CHECK-LABEL: @LSTMGates
// CHECK:       ([[ARG0:%.+]]: tensor<1x1x1024x2048xf16>, [[ARG1:%.+]]: tensor<1x1x1024x512xf16>) -> (tensor<1024x512xf16>, tensor<1024x512xf16>)
module @LSTMGates {
  func.func @main(%arg0: tensor<1x1x1024x2048xf16>, %arg1: tensor<1x1x1024x512xf16>) -> (tensor<1024x512xf16>, tensor<1024x512xf16>) {

    %outputHiddenState, %outputCellState = VPU.LSTMGates(%arg0, %arg1) {tilingStrategy = [1, 1, 5, 1]} : tensor<1x1x1024x2048xf16>, tensor<1x1x1024x512xf16> -> tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>
    %0 = VPU.AffineReshape(%outputHiddenState) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1024, 512]} : tensor<1x1x1024x512xf16> -> tensor<1024x512xf16>
    %1 = VPU.AffineReshape(%outputCellState) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1024, 512]} : tensor<1x1x1024x512xf16> -> tensor<1024x512xf16>

    return %0, %1 : tensor<1024x512xf16>, tensor<1024x512xf16>

// CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
// CHECK-DAG:    [[C1024:%.+]] = arith.constant 1024 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1024x512xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C1024]] step [[C205]] iter_args([[ACC0:%.+]] = [[EMPTY]], [[ACC1:%.+]] = [[EMPTY]]) -> (tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>)
// CHECK-NEXT:      [[MIN:%.+]] = affine.min #map([[IDX]])
// CHECK-NEXT:      [[EXTRACT0:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 2048] [1, 1, 1, 1]
// CHECK-SAME:          to tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-NEXT:      [[EXTRACT1:%.+]] = tensor.extract_slice [[ARG1]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 512] [1, 1, 1, 1]
// CHECK-SAME:          to tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-NEXT:      %[[H:.*]], %[[C:.*]] = VPU.LSTMGates([[EXTRACT0]], [[EXTRACT1]]) {tiling_loop_index = 0 : i64}
// CHECK-SAME:          -> tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-NEXT:      [[INSERT0:%.+]] = tensor.insert_slice %[[H]] into [[ACC0]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 512] [1, 1, 1, 1]
// CHECK-SAME:          into tensor<1x1x1024x512xf16>
// CHECK-NEXT:      [[INSERT1:%.+]] = tensor.insert_slice %[[C]] into [[ACC1]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 512] [1, 1, 1, 1]
// CHECK-SAME:          into tensor<1x1x1024x512xf16>
// CHECK-NEXT:      scf.yield [[INSERT0]], [[INSERT1]] : tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>
// CHECK:         [[RESH0:%.+]] = VPU.AffineReshape([[SCF:%.+]]#0)
// CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1024, 512]} : tensor<1x1x1024x512xf16> -> tensor<1024x512xf16>
// CHECK-NEXT:    [[RESH1:%.+]] = VPU.AffineReshape([[SCF:%.+]]#1)
// CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1024, 512]} : tensor<1x1x1024x512xf16> -> tensor<1024x512xf16>
// CHECK-NEXT:    return [[RESH0]], [[RESH1]] : tensor<1024x512xf16>, tensor<1024x512xf16>

  }
}

// -----

// LSTMGates with NHWC layout on inputs and outputs. Tiled over dim 2 with factor 5.
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 1024, 205)>

// CHECK-LABEL: @LSTMGatesNHWC
// CHECK:       ([[ARG0:%.+]]: tensor<1x1x1024x2048xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x1x1024x512xf16, {order = #NHWC}>) -> (tensor<1x1x1024x512xf16, {order = #NHWC}>, tensor<1x1x1024x512xf16, {order = #NHWC}>)
module @LSTMGatesNHWC {
    func.func @LSTMGatesNHWC(
            %arg0: tensor<1x1x1024x2048xf16, {order = #NHWC}>,
            %arg1: tensor<1x1x1024x512xf16, {order = #NHWC}>
    ) -> (tensor<1x1x1024x512xf16, {order = #NHWC}>,
                tensor<1x1x1024x512xf16, {order = #NHWC}>) {

    %outputHiddenState, %outputCellState = VPU.LSTMGates(%arg0, %arg1) {
        tilingStrategy = [1, 1, 5, 1]
    } : tensor<1x1x1024x2048xf16, {order = #NHWC}>,
        tensor<1x1x1024x512xf16, {order = #NHWC}>
      -> tensor<1x1x1024x512xf16, {order = #NHWC}>,
         tensor<1x1x1024x512xf16, {order = #NHWC}>

    return %outputHiddenState, %outputCellState
        : tensor<1x1x1024x512xf16, {order = #NHWC}>,
          tensor<1x1x1024x512xf16, {order = #NHWC}>

// CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
// CHECK-DAG:    [[C1024:%.+]] = arith.constant 1024 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1024x512xf16, {order = #NHWC}>
// CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX:%.+]] = [[C0]] to [[C1024]] step [[C205]] iter_args([[ACC0:%.+]] = [[EMPTY]], [[ACC1:%.+]] = [[EMPTY]]) -> (tensor<1x1x1024x512xf16, {order = #NHWC}>, tensor<1x1x1024x512xf16, {order = #NHWC}>)
// CHECK-NEXT:      [[MIN:%.+]] = affine.min #map([[IDX]])
// CHECK-NEXT:      [[EXTRACT0:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 2048] [1, 1, 1, 1]
// CHECK-SAME:          to tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      [[EXTRACT1:%.+]] = tensor.extract_slice [[ARG1]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 512] [1, 1, 1, 1]
// CHECK-SAME:          to tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      %[[H:.*]], %[[C:.*]] = VPU.LSTMGates([[EXTRACT0]], [[EXTRACT1]]) {tiling_loop_index = 0 : i64}
// CHECK-SAME:          -> tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      [[INSERT0:%.+]] = tensor.insert_slice %[[H]] into [[ACC0]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 512] [1, 1, 1, 1]
// CHECK-SAME:          into tensor<1x1x1024x512xf16, {order = #NHWC}>
// CHECK-NEXT:      [[INSERT1:%.+]] = tensor.insert_slice %[[C]] into [[ACC1]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 512] [1, 1, 1, 1]
// CHECK-SAME:          into tensor<1x1x1024x512xf16, {order = #NHWC}>
// CHECK-NEXT:      scf.yield [[INSERT0]], [[INSERT1]] : tensor<1x1x1024x512xf16, {order = #NHWC}>, tensor<1x1x1024x512xf16, {order = #NHWC}>
// CHECK:        return [[SCF]]#0, [[SCF]]#1 : tensor<1x1x1024x512xf16, {order = #NHWC}>, tensor<1x1x1024x512xf16, {order = #NHWC}>
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 205)>

// LSTMGates with dynamic dim 2 (sequence length). Tiled over dim 2 with factor 5.
// CHECK-LABEL: @LSTMGatesDynamic
// CHECK:       ([[ARG0:%.+]]: tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>}>, [[ARG1:%.+]]: tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>)
// CHECK-SAME:   -> (tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>, tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>)
module @LSTMGatesDynamic {
    func.func @LSTMGatesDynamic(
      %arg0: tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>}>,
      %arg1: tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>
  ) -> (tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>,
        tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>) {

    %outputHiddenState, %outputCellState = VPU.LSTMGates(%arg0, %arg1) {
        tilingStrategy = [1, 1, 5, 1]
    } : tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>}>,
        tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>
      -> tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>,
         tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>

    return %outputHiddenState, %outputCellState
        : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>,
          tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>

// CHECK-DAG:    [[C205:%.+]] = arith.constant 205 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
// CHECK:        [[DIM:%.+]] = tensor.dim [[ARG1]], [[C2]] : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>
// CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C205]] iter_args([[ACC0:%.+]] = [[EMPTY]], [[ACC1:%.+]] = [[EMPTY]]) -> (tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>, tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>)
// CHECK-NEXT:      [[MIN:%.+]] = affine.min #map([[IDX]])[[[DIM]]]
// CHECK-NEXT:      [[SLICE0:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 2048] [1, 1, 1, 1] : tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>}> to tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 2048]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-NEXT:      [[SLICE1:%.+]] = tensor.extract_slice [[ARG1]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 512] [1, 1, 1, 1] : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}> to tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 512]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-NEXT:      [[H:%.+]], [[C:%.+]] = VPU.LSTMGates([[SLICE0]], [[SLICE1]]) {tiling_loop_index = 0 : i64}
// CHECK-SAME:          : tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 2048]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 512]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 512]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 512]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-NEXT:      [[INSERT0:%.+]] = tensor.insert_slice [[H]] into [[ACC0]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 512] [1, 1, 1, 1] : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 512]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>
// CHECK-NEXT:      [[INSERT1:%.+]] = tensor.insert_slice [[C]] into [[ACC1]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 512] [1, 1, 1, 1] : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 205, 512]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>
// CHECK-NEXT:      scf.yield [[INSERT0]], [[INSERT1]] : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>, tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>
// CHECK:        return [[SCF]]#0, [[SCF]]#1 : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>, tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>}>

  }
}

// -----

// TopK with NHWC layout on the input (values and indices follow the same order).
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TopKNHWC
// CHECK:       ([[ARG0:%.+]]: tensor<1x512x4096x4096xf32, {order = #NHWC}>) -> tensor<1x1x4096x4096xsi32, {order = #NHWC}>
module @TopKNHWC {
    func.func @TopKNHWC(%arg0: tensor<1x512x4096x4096xf32, {order = #NHWC}>)
            -> tensor<1x1x4096x4096xsi32, {order = #NHWC}> {
        %0 = VPU.Empty : tensor<1x1x1x8192xui8>
        %output_values, %target_shape = VPU.TopK(%arg0, %0) {
                axis = 1 : i64,
                element_type = si32,
                k_value = 1 : i64,
                mode = #IE.topk_mode<MAX>,
                sort = #IE.topk_sort_type<SORT_INDICES>,
                tilingStrategy = [1, 1, 4, 1]
        } : tensor<1x512x4096x4096xf32, {order = #NHWC}>,
                tensor<1x1x1x8192xui8>
            -> tensor<1x1x4096x4096xf32, {order = #NHWC}>,
                 tensor<1x1x4096x4096xsi32, {order = #NHWC}>
        return %target_shape : tensor<1x1x4096x4096xsi32, {order = #NHWC}>

// CHECK-DAG:    [[C1024:%.+]] = arith.constant 1024 : index
// CHECK-DAG:    [[C4096:%.+]] = arith.constant 4096 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY_VALUES:%.+]] = tensor.empty() : tensor<1x1x4096x4096xf32, {order = #NHWC}>
// CHECK-DAG:    [[EMPTY_INDICES:%.+]] = tensor.empty() : tensor<1x1x4096x4096xsi32, {order = #NHWC}>
// CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX:%.+]] = [[C0]] to [[C4096]] step [[C1024]] iter_args([[ACC_VALUES:%.+]] = [[EMPTY_VALUES]], [[ACC_INDICES:%.+]] = [[EMPTY_INDICES]]) -> (tensor<1x1x4096x4096xf32, {order = #NHWC}>, tensor<1x1x4096x4096xsi32, {order = #NHWC}>)
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 512, 1024, 4096] [1, 1, 1, 1]
// CHECK-SAME:          to tensor<1x512x1024x4096xf32, {order = #NHWC}>
// CHECK-NEXT:      [[V:%.+]], [[S:%.+]] = VPU.TopK([[SLICE]], %0) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>, tiling_loop_index = 0 : i64}
// CHECK-SAME:          -> tensor<1x1x1024x4096xf32, {order = #NHWC}>, tensor<1x1x1024x4096xsi32, {order = #NHWC}>
// CHECK-NEXT:      [[INSERT_VALUES:%.+]] = tensor.insert_slice [[V]] into [[ACC_VALUES]][0, 0, [[IDX]], 0] [1, 1, 1024, 4096] [1, 1, 1, 1]
// CHECK-SAME:          into tensor<1x1x4096x4096xf32, {order = #NHWC}>
// CHECK-NEXT:      [[INSERT_INDICES:%.+]] = tensor.insert_slice [[S]] into [[ACC_INDICES]][0, 0, [[IDX]], 0] [1, 1, 1024, 4096] [1, 1, 1, 1]
// CHECK-SAME:          into tensor<1x1x4096x4096xsi32, {order = #NHWC}>
// CHECK-NEXT:      scf.yield [[INSERT_VALUES]], [[INSERT_INDICES]] : tensor<1x1x4096x4096xf32, {order = #NHWC}>, tensor<1x1x4096x4096xsi32, {order = #NHWC}>
// CHECK:        return [[SCF]]#1 : tensor<1x1x4096x4096xsi32, {order = #NHWC}

    }
}

// -----

// CHECK-LABEL: @TopKWithKValue
// CHECK:       ([[ARG0:%.+]]: tensor<1x512x4096x4096xf32>) -> tensor<1x1x4096x4096xsi32>
module @TopKWithKValue {
  func.func @TopKWithKValue(%arg0: tensor<1x512x4096x4096xf32>) -> tensor<1x1x4096x4096xsi32> {
    %0 = VPU.Empty : tensor<1x1x1x8192xui8>
    %output_values, %target_shape = VPU.TopK(%arg0, %0) {
            axis = 1 : i64,
            element_type = si32,
            k_value = 1 : i64,
            mode = #IE.topk_mode<MAX>,
            sort = #IE.topk_sort_type<SORT_INDICES>,
            tilingStrategy = [1, 1, 4, 1]
    }: tensor<1x512x4096x4096xf32>, tensor<1x1x1x8192xui8> -> tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>
    return %target_shape : tensor<1x1x4096x4096xsi32>

// CHECK-DAG:    [[C1024:%.+]] = arith.constant 1024 : index
// CHECK-DAG:    [[C4096:%.+]] = arith.constant 4096 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[AUX:.*]] = VPU.Empty : tensor<1x1x1x8192xui8>
// CHECK:        [[EMPTY_VALUES:%.+]] = tensor.empty() : tensor<1x1x4096x4096xf32>
// CHECK:        [[EMPTY_INDICES:%.+]] = tensor.empty() : tensor<1x1x4096x4096xsi32>
// CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX:%.+]] = [[C0]] to [[C4096]] step [[C1024]] iter_args([[ACC_VALUES:%.+]] = [[EMPTY_VALUES]], [[ACC_INDICES:%.+]] = [[EMPTY_INDICES]]) -> (tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>)
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 512, 1024, 4096] [1, 1, 1, 1]
// CHECK-SAME:          to tensor<1x512x1024x4096xf32>
// CHECK-NEXT:      [[V:%.+]], [[S:%.+]] = VPU.TopK([[SLICE]], [[EMPTY:%.+]]) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>, tiling_loop_index = 0 : i64}
// CHECK-SAME:          -> tensor<1x1x1024x4096xf32>, tensor<1x1x1024x4096xsi32>
// CHECK-NEXT:      [[INSERT_VALUES:%.+]] = tensor.insert_slice [[V]] into [[ACC_VALUES]][0, 0, [[IDX]], 0] [1, 1, 1024, 4096] [1, 1, 1, 1]
// CHECK-SAME:          into tensor<1x1x4096x4096xf32>
// CHECK-NEXT:      [[INSERT_INDICES:%.+]] = tensor.insert_slice [[S]] into [[ACC_INDICES]][0, 0, [[IDX]], 0] [1, 1, 1024, 4096] [1, 1, 1, 1]
// CHECK-SAME:          into tensor<1x1x4096x4096xsi32>
// CHECK-NEXT:      scf.yield [[INSERT_VALUES]], [[INSERT_INDICES]] : tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>
// CHECK:        return [[SCF]]#1 : tensor<1x1x4096x4096xsi32>

  }
}

// -----

// TopK with axis=3 (reduce along W), k_value=2, tiled over dim 2 (H).
// CHECK-LABEL: @TopKAxis3KValue2
// CHECK:       ([[ARG0:%.+]]: tensor<1x16x256x512xf16>) -> (tensor<1x16x256x2xf16>, tensor<1x16x256x2xsi32>)
module @TopKAxis3KValue2 {
  func.func @TopKAxis3KValue2(%arg0: tensor<1x16x256x512xf16>) -> (tensor<1x16x256x2xf16>, tensor<1x16x256x2xsi32>) {
    %0 = VPU.Empty : tensor<1x1x1x8192xui8>
    %output_values, %target_shape = VPU.TopK(%arg0, %0) {
            axis = 3 : i64,
            element_type = si32,
            k_value = 2 : i64,
            mode = #IE.topk_mode<MAX>,
            sort = #IE.topk_sort_type<SORT_VALUES>,
            tilingStrategy = [1, 1, 4, 1]
    } : tensor<1x16x256x512xf16>, tensor<1x1x1x8192xui8> -> tensor<1x16x256x2xf16>, tensor<1x16x256x2xsi32>
    return %output_values, %target_shape : tensor<1x16x256x2xf16>, tensor<1x16x256x2xsi32>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C256:%.+]] = arith.constant 256 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY_VALUES:%.+]] = tensor.empty() : tensor<1x16x256x2xf16>
// CHECK-DAG:    [[EMPTY_INDICES:%.+]] = tensor.empty() : tensor<1x16x256x2xsi32>
// CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX:%.+]] = [[C0]] to [[C256]] step [[C64]] iter_args([[ACC_VALUES:%.+]] = [[EMPTY_VALUES]], [[ACC_INDICES:%.+]] = [[EMPTY_INDICES]]) -> (tensor<1x16x256x2xf16>, tensor<1x16x256x2xsi32>)
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 16, 64, 512] [1, 1, 1, 1]
// CHECK-SAME:          to tensor<1x16x64x512xf16>
// CHECK-NEXT:      [[V:%.+]], [[S:%.+]] = VPU.TopK([[SLICE]], [[EMPTY:%.+]]) {axis = 3 : i64, element_type = si32, k_value = 2 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_VALUES>, tiling_loop_index = 0 : i64}
// CHECK-SAME:          -> tensor<1x16x64x2xf16>, tensor<1x16x64x2xsi32>
// CHECK-NEXT:      [[INSERT_VALUES:%.+]] = tensor.insert_slice [[V]] into [[ACC_VALUES]][0, 0, [[IDX]], 0] [1, 16, 64, 2] [1, 1, 1, 1]
// CHECK-SAME:          into tensor<1x16x256x2xf16>
// CHECK-NEXT:      [[INSERT_INDICES:%.+]] = tensor.insert_slice [[S]] into [[ACC_INDICES]][0, 0, [[IDX]], 0] [1, 16, 64, 2] [1, 1, 1, 1]
// CHECK-SAME:          into tensor<1x16x256x2xsi32>
// CHECK-NEXT:      scf.yield [[INSERT_VALUES]], [[INSERT_INDICES]] : tensor<1x16x256x2xf16>, tensor<1x16x256x2xsi32>
// CHECK:        return [[SCF]]#0, [[SCF]]#1 : tensor<1x16x256x2xf16>, tensor<1x16x256x2xsi32>

  }
}

// -----

// TopK with axis=2 (reduce along H), k_value=4, tiled over dim 3 (W).
// CHECK-LABEL: @TopKAxis2TileOverW
// CHECK:       ([[ARG0:%.+]]: tensor<1x32x128x1024xf32>) -> (tensor<1x32x4x1024xf32>, tensor<1x32x4x1024xsi32>)
module @TopKAxis2TileOverW {
    func.func @TopKAxis2TileOverW(%arg0: tensor<1x32x128x1024xf32>) -> (tensor<1x32x4x1024xf32>, tensor<1x32x4x1024xsi32>) {
    %0 = VPU.Empty : tensor<1x1x1x2048xui8>
    %output_values, %target_shape = VPU.TopK(%arg0, %0) {
            axis = 2 : i64,
            element_type = si32,
            k_value = 4 : i64,
            mode = #IE.topk_mode<MIN>,
            sort = #IE.topk_sort_type<SORT_INDICES>,
            tilingStrategy = [1, 1, 1, 4]
    } : tensor<1x32x128x1024xf32>, tensor<1x1x1x2048xui8> -> tensor<1x32x4x1024xf32>, tensor<1x32x4x1024xsi32>
    return %output_values, %target_shape : tensor<1x32x4x1024xf32>, tensor<1x32x4x1024xsi32>

// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY_VALUES:%.+]] = tensor.empty() : tensor<1x32x4x1024xf32>
// CHECK-DAG:    [[EMPTY_INDICES:%.+]] = tensor.empty() : tensor<1x32x4x1024xsi32>
// CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX:%.+]] = [[C0]] to [[C1024:%.+]] step [[C256:%.+]] iter_args([[ACC_VALUES:%.+]] = [[EMPTY_VALUES]], [[ACC_INDICES:%.+]] = [[EMPTY_INDICES]]) -> (tensor<1x32x4x1024xf32>, tensor<1x32x4x1024xsi32>)
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[IDX]]] [1, 32, 128, 256] [1, 1, 1, 1] : tensor<1x32x128x1024xf32> to tensor<1x32x128x256xf32>
// CHECK-NEXT:      [[V:%.+]], [[S:%.+]] = VPU.TopK([[SLICE]], [[EMPTY:%.+]]) {axis = 2 : i64, element_type = si32, k_value = 4 : i64, mode = #IE.topk_mode<MIN>, sort = #IE.topk_sort_type<SORT_INDICES>, tiling_loop_index = 0 : i64}
// CHECK-SAME:          -> tensor<1x32x4x256xf32>, tensor<1x32x4x256xsi32>
// CHECK-NEXT:      [[INSERT_VALUES:%.+]] = tensor.insert_slice [[V]] into [[ACC_VALUES]][0, 0, 0, [[IDX]]] [1, 32, 4, 256] [1, 1, 1, 1] : tensor<1x32x4x256xf32> into tensor<1x32x4x1024xf32>
// CHECK-NEXT:      [[INSERT_INDICES:%.+]] = tensor.insert_slice [[S]] into [[ACC_INDICES]][0, 0, 0, [[IDX]]] [1, 32, 4, 256] [1, 1, 1, 1] : tensor<1x32x4x256xsi32> into tensor<1x32x4x1024xsi32>
// CHECK-NEXT:      scf.yield [[INSERT_VALUES]], [[INSERT_INDICES]] : tensor<1x32x4x1024xf32>, tensor<1x32x4x1024xsi32>
// CHECK:        return [[SCF]]#0, [[SCF]]#1 : tensor<1x32x4x1024xf32>, tensor<1x32x4x1024xsi32>

  }
}

// -----

// TopK with dynamic dim 2 (H). axis=1, k_value=1, tiled over dim 2 with factor 4.
// CHECK-LABEL: @TopKDynamic
// CHECK:       ([[ARG0:%.+]]: tensor<1x512x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 4096, 4096]> : tensor<4xsi64>}>) -> (tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>, tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>)
module @TopKDynamic {
    func.func @TopKDynamic(
      %arg0: tensor<1x512x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 4096, 4096]> : tensor<4xsi64>}>
  ) -> (tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>,
        tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>) {
    %0 = VPU.Empty : tensor<1x1x1x8192xui8>

    %output_values, %target_shape = VPU.TopK(%arg0, %0) {
        axis = 1 : i64,
        element_type = si32,
        k_value = 1 : i64,
        mode = #IE.topk_mode<MAX>,
        sort = #IE.topk_sort_type<SORT_INDICES>,
        tilingStrategy = [1, 1, 4, 1]
    } : tensor<1x512x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 4096, 4096]> : tensor<4xsi64>}>,
        tensor<1x1x1x8192xui8>
      -> tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>,
         tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>

    return %output_values, %target_shape
        : tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>,
          tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>

// CHECK-DAG:    [[C1024:%.+]] = arith.constant 1024 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
// CHECK:        [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x512x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 4096, 4096]> : tensor<4xsi64>}>
// CHECK-DAG:    [[EMPTY_VALUES:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>
// CHECK-DAG:    [[EMPTY_INDICES:%.+]] = tensor.empty([[DIM]]) : tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>
// CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX:%.+]] = [[C0]] to [[DIM]] step [[C1024]] iter_args([[ACC_VALUES:%.+]] = [[EMPTY_VALUES]], [[ACC_INDICES:%.+]] = [[EMPTY_INDICES]]) -> (tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>, tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>)
// CHECK-NEXT:      [[MIN:%.+]] = affine.min #map([[IDX]])[[[DIM]]]
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 512, [[MIN]], 4096] [1, 1, 1, 1] : tensor<1x512x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 4096, 4096]> : tensor<4xsi64>}> to tensor<1x512x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-NEXT:      [[V:%.+]], [[S:%.+]] = VPU.TopK([[SLICE]], [[EMPTY:%.+]]) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>, tiling_loop_index = 0 : i64}
// CHECK-SAME:          : tensor<1x512x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x8192xui8> -> tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}
// CHECK-NEXT:      [[INSERT_VALUES:%.+]] = tensor.insert_slice [[V]] into [[ACC_VALUES]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 4096] [1, 1, 1, 1] : tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>
// CHECK-NEXT:      [[INSERT_INDICES:%.+]] = tensor.insert_slice [[S]] into [[ACC_INDICES]][0, 0, [[IDX]], 0] [1, 1, [[MIN]], 4096] [1, 1, 1, 1] : tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>
// CHECK-NEXT:      scf.yield [[INSERT_VALUES]], [[INSERT_INDICES]] : tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>, tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>
// CHECK:        return [[SCF]]#0, [[SCF]]#1 : tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>, tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 4096, 4096]> : tensor<4xsi64>}>

  }
}

// -----

// CHECK-LABEL: @GeluTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @GeluTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Gelu(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[GELU:%.+]] = VPU.Gelu([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[GELU]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @ApplyTilingSCFSoftMax
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x256x140xf16>)
func.func @ApplyTilingSCFSoftMax(%arg0: tensor<1x16x256x140xf16>) -> tensor<1x16x256x140xf16> {
    %0 = VPU.SoftMax(%arg0) {
        axisInd = 3 : i64,
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x16x256x140xf16> -> tensor<1x16x256x140xf16>

    return %0 : tensor<1x16x256x140xf16>

    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 256 : index
    // CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 128 : index

    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x256x140xf16>
    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    // CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x256x140xf16>) {

    // CHECK:      [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[LOOP_ITER]], 0] [1, 16, 128, 140] [1, 1, 1, 1] : tensor<1x16x256x140xf16> to tensor<1x16x128x140xf16>
    // CHECK:      [[SOFTMAX:%.+]] = VPU.SoftMax([[SLICE]]) {axisInd = 3 : i64, tiling_loop_index = 0 : i64} : tensor<1x16x128x140xf16> -> tensor<1x16x128x140xf16>

    // CHECK:      [[INSERT:%.+]] = tensor.insert_slice [[SOFTMAX]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 16, 128, 140] [1, 1, 1, 1] : tensor<1x16x128x140xf16> into tensor<1x16x256x140xf16>
    // CHECK:      scf.yield [[INSERT]] : tensor<1x16x256x140xf16>

    // CHECK: return [[LOOP]] : tensor<1x16x256x140xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: @DequantizeTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256x!qElemType>
func.func @DequantizeTileOverH(%arg0: tensor<1x16x128x256x!qElemType>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Dequantize(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256x!qElemType> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[DQ:%.+]] = VPU.Dequantize([[SLICE]]) {dstElemType = f16, tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[DQ]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVN1NormalizeTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
// CHECK-SAME:  [[MEANVAR:%.+]]: tensor<1x16x1x2xf16>
func.func @MVN1NormalizeTileOverH(%arg0: tensor<1x16x128x256xf16>, %arg1: tensor<1x16x1x2xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = true, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16>, tensor<1x16x1x2xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1] : tensor<1x16x128x256xf16> to tensor<1x16x64x256xf16>
// CHECK:            [[MVN:%.+]] = VPU.MVN1Normalize([[SLICE]], [[MEANVAR]]) {across_channels = false, normalize_variance = true, tiling_loop_index = 0 : i64} : tensor<1x16x64x256xf16>, tensor<1x16x1x2xf16> -> tensor<1x16x64x256xf16>
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[MVN]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1] : tensor<1x16x64x256xf16> into tensor<1x16x128x256xf16>
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK: #[[$MAP_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
// CHECK: #[[$MAP_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 128)>

// CHECK-LABEL: @GeluTileDynHW
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
func.func @GeluTileDynHW(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}> {
    %0 = VPU.Gelu(%arg0) {tilingStrategy = [1, 1, 2, 2]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
    return %0 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>

// CHECK-DAG:    [[CST_128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[CST_64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[CST_0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[CST_3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[CST_2:%.+]] = arith.constant 2 : index
// CHECK:        [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[CST_2]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK:        [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[CST_3]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK:        [[EMPTY:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK:        [[SCF_H:%.+]] = scf.for [[IDX_H:%.+]] = [[CST_0]] to [[DIM_H]] step [[CST_64]] iter_args([[OUT_H:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>)
// CHECK:          [[SCF_W:%.+]] = scf.for [[IDX_W:%.+]] = [[CST_0]] to [[DIM_W]] step [[CST_128]] iter_args([[OUT_W:%.+]] = [[OUT_H]])
// CHECK:            [[SIZE_H:%.+]] = affine.min #[[$MAP_MIN_H]]([[IDX_H]])[[[DIM_H]]]
// CHECK:            [[SIZE_W:%.+]] = affine.min #[[$MAP_MIN_W]]([[IDX_W]])[[[DIM_W]]]
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
// CHECK:            [[GELU:%.+]] = VPU.Gelu([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[GELU]] into [[OUT_W]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK:          scf.yield [[SCF_W]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK:        return [[SCF_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
}

// -----

!qElemType_dyn = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK: #[[$MAP_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
// CHECK: #[[$MAP_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 128)>
// CHECK-LABEL: @DequantizeTileDynHW
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x?x?x!qElemType
func.func @DequantizeTileDynHW(%arg0: tensor<1x16x?x?x!qElemType_dyn, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}> {
    %0 = VPU.Dequantize(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 2]} : tensor<1x16x?x?x!qElemType_dyn, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
    return %0 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>

// CHECK-DAG:    [[CST_2:%.+]] = arith.constant 2 : index
// CHECK-DAG:    [[CST_3:%.+]] = arith.constant 3 : index
// CHECK:        [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[CST_2]]
// CHECK:        [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[CST_3]]
// CHECK:        [[EMPTY:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]])
// CHECK:        [[SCF_H:%.+]] = scf.for [[IDX_H:%.+]] = {{%.+}} to [[DIM_H]] step {{%.+}} iter_args([[ARG_H:%.+]] = [[EMPTY]])
// CHECK:            [[SCF_W:%.+]] = scf.for [[IDX_W:%.+]] = {{%.+}} to [[DIM_W]] step {{%.+}} iter_args([[ARG_W:%.+]] = [[ARG_H]])
// CHECK:                [[MIN_H:%.+]] = affine.min #[[$MAP_MIN_H]]([[IDX_H]])[[[DIM_H]]]
// CHECK:                [[MIN_W:%.+]] = affine.min #[[$MAP_MIN_W]]([[IDX_W]])[[[DIM_W]]]
// CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[MIN_H]], [[MIN_W]]]
// CHECK:                [[DQ:%.+]] = VPU.Dequantize([[SLICE]]) {dstElemType = f16, tiling_loop_index = 0 : i64}
// CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[DQ]] into [[ARG_W]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[MIN_H]], [[MIN_W]]]
// CHECK:                scf.yield [[INSERT]]
// CHECK:            scf.yield [[SCF_W]]
// CHECK:        return [[SCF_H]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: #[[$MAP_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
// CHECK: #[[$MAP_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 128)>
// CHECK-LABEL: @MVN1NormalizeTileDynHW
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK-SAME:  [[MEANVAR:%.+]]: tensor<1x16x1x2xf16>
func.func @MVN1NormalizeTileDynHW(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>, %arg1: tensor<1x16x1x2xf16>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}> {
    %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = true, tilingStrategy = [1, 1, 2, 2]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>, tensor<1x16x1x2xf16> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
    return %0 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>

// CHECK-DAG:    [[CST_2:%.+]] = arith.constant 2 : index
// CHECK-DAG:    [[CST_3:%.+]] = arith.constant 3 : index
// CHECK:        [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[CST_2]]
// CHECK:        [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[CST_3]]
// CHECK:        [[EMPTY:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]])
// CHECK:        [[SCF_H:%.+]] = scf.for [[IDX_H:%.+]] = {{%.+}} to [[DIM_H]] step {{%.+}} iter_args([[ARG_H:%.+]] = [[EMPTY]])
// CHECK:            [[SCF_W:%.+]] = scf.for [[IDX_W:%.+]] = {{%.+}} to [[DIM_W]] step {{%.+}} iter_args([[ARG_W:%.+]] = [[ARG_H]])
// CHECK:                [[MIN_H:%.+]] = affine.min #[[$MAP_MIN_H]]([[IDX_H]])[[[DIM_H]]]
// CHECK:                [[MIN_W:%.+]] = affine.min #[[$MAP_MIN_W]]([[IDX_W]])[[[DIM_W]]]
// CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[MIN_H]], [[MIN_W]]]
// CHECK:                [[MVN:%.+]] = VPU.MVN1Normalize([[SLICE]], [[MEANVAR]]) {across_channels = false, normalize_variance = true, tiling_loop_index = 0 : i64}
// CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[MVN]] into [[ARG_W]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[MIN_H]], [[MIN_W]]]
// CHECK:                scf.yield [[INSERT]]
// CHECK:            scf.yield [[SCF_W]]
// CHECK:        return [[SCF_H]]
}

// -----

// CHECK: #[[$MAP_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 128)>
// CHECK: #[[$MAP_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 70)>

// CHECK-LABEL: @SoftMaxTileDynHW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>)
func.func @SoftMaxTileDynHW(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}> {
    %0 = VPU.SoftMax(%arg0) {
        axisInd = 1 : i64,
        tilingStrategy = [1, 1, 2, 2]
    } : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
    return %0 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>

    // CHECK-DAG: [[CST_70:%.+]] = arith.constant 70 : index
    // CHECK-DAG: [[CST_128:%.+]] = arith.constant 128 : index
    // CHECK-DAG: [[CST_0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[CST_3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[CST_2:%.+]] = arith.constant 2 : index

    // CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[CST_2]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[CST_3]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
    // CHECK: [[EMPTY:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
    // CHECK: [[SCF_H:%.+]] = scf.for [[IDX_H:%.+]] = [[CST_0]] to [[DIM_H]] step [[CST_128]] iter_args([[OUT_H:%.+]] = [[EMPTY]])
    // CHECK-SAME:     -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>)
    // CHECK:   [[SCF_W:%.+]] = scf.for [[IDX_W:%.+]] = [[CST_0]] to [[DIM_W]] step [[CST_70]] iter_args([[OUT_W:%.+]] = [[OUT_H]])
    // CHECK:     [[SIZE_H:%.+]] = affine.min #[[$MAP_MIN_H]]([[IDX_H]])[[[DIM_H]]]
    // CHECK:     [[SIZE_W:%.+]] = affine.min #[[$MAP_MIN_W]]([[IDX_W]])[[[DIM_W]]]
    // CHECK:     [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
    // CHECK:     [[SOFTMAX:%.+]] = VPU.SoftMax([[SLICE]]) {axisInd = 1 : i64, tiling_loop_index = 0 : i64}
    // CHECK:     [[INSERT:%.+]] = tensor.insert_slice [[SOFTMAX]] into [[OUT_W]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
    // CHECK:     scf.yield [[INSERT]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
    // CHECK:   scf.yield [[SCF_W]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
    // CHECK: return [[SCF_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
}

// -----

// CHECK-LABEL: @AbsTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @AbsTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Abs(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[ABS:%.+]] = VPU.Abs([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ABS]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @SinTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @SinTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Sin(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[SIN:%.+]] = VPU.Sin([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SIN]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @SqrtTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @SqrtTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Sqrt(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[SQRT:%.+]] = VPU.Sqrt([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SQRT]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @TanhTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @TanhTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Tanh(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[TANH:%.+]] = VPU.Tanh([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[TANH]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MishTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @MishTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Mish(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[MISH:%.+]] = VPU.Mish([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[MISH]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @ReLUTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @ReLUTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.ReLU(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[RELU:%.+]] = VPU.ReLU([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[RELU]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @SigmoidTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @SigmoidTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Sigmoid(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[SIG:%.+]] = VPU.Sigmoid([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SIG]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @SoftPlusTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @SoftPlusTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.SoftPlus(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[SP:%.+]] = VPU.SoftPlus([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SP]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @SwishTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @SwishTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Swish(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[SWISH:%.+]] = VPU.Swish([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SWISH]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @AcosTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @AcosTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Acos(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[ACOS:%.+]] = VPU.Acos([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ACOS]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @AcoshTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @AcoshTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Acosh(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[ACOSH:%.+]] = VPU.Acosh([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ACOSH]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @AsinTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @AsinTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Asin(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[ASIN:%.+]] = VPU.Asin([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ASIN]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @AsinhTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @AsinhTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Asinh(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[ASINH:%.+]] = VPU.Asinh([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ASINH]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @AtanTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @AtanTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Atan(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[ATAN:%.+]] = VPU.Atan([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ATAN]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @AtanhTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @AtanhTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Atanh(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[ATANH:%.+]] = VPU.Atanh([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ATANH]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @CeilingTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @CeilingTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Ceiling(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[CEILING:%.+]] = VPU.Ceiling([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[CEILING]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @ClampTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @ClampTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[CLAMP:%.+]] = VPU.Clamp([[SLICE]]) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64, tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[CLAMP]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @CosTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @CosTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Cos(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[COS:%.+]] = VPU.Cos([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[COS]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @CoshTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @CoshTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Cosh(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[COSH:%.+]] = VPU.Cosh([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[COSH]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @CumSumTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @CumSumTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.CumSum(%arg0) {axis_value = 1 : i64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[CUMSUM:%.+]] = VPU.CumSum([[SLICE]]) {axis_value = 1 : i64, tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[CUMSUM]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @ErfTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @ErfTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Erf(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[ERF:%.+]] = VPU.Erf([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ERF]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @ExpTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @ExpTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Exp(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[EXP:%.+]] = VPU.Exp([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[EXP]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @FloorTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @FloorTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Floor(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[FLOOR:%.+]] = VPU.Floor([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[FLOOR]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @LogTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @LogTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Log(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[LOG:%.+]] = VPU.Log([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[LOG]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @NegativeTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @NegativeTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Negative(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[NEG:%.+]] = VPU.Negative([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[NEG]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @RoundTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @RoundTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[ROUND:%.+]] = VPU.Round([[SLICE]]) {mode = #IE.round_mode<HALF_TO_EVEN>, tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ROUND]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @SignTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @SignTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Sign(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[SIGN:%.+]] = VPU.Sign([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SIGN]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @SinhTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @SinhTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Sinh(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[SINH:%.+]] = VPU.Sinh([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SINH]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @EluTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @EluTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Elu(%arg0) {tilingStrategy = [1, 1, 2, 1], x = 1.000000e+00 : f64} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[ELU:%.+]] = VPU.Elu([[SLICE]]) {tiling_loop_index = 0 : i64, x = 1.000000e+00 : f64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ELU]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @HardSigmoidTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @HardSigmoidTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.HardSigmoid(%arg0) {alpha_value = 2.000000e-01 : f64, beta_value = 5.000000e-01 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[HARDSIGMOID:%.+]] = VPU.HardSigmoid([[SLICE]]) {alpha_value = 2.000000e-01 : f64, beta_value = 5.000000e-01 : f64, tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[HARDSIGMOID]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @HSigmoidTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @HSigmoidTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.HSigmoid(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[HSIGMOID:%.+]] = VPU.HSigmoid([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[HSIGMOID]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @HSwishTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @HSwishTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.HSwish(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[HSWISH:%.+]] = VPU.HSwish([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[HSWISH]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @LeakyReluTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @LeakyReluTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.LeakyRelu(%arg0) {negative_slope = 1.000000e-02 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[LEAKYRELU:%.+]] = VPU.LeakyRelu([[SLICE]]) {negative_slope = 1.000000e-02 : f64, tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[LEAKYRELU]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @PReluTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @PReluTileOverH(%arg0: tensor<1x16x128x256xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.PRelu(%arg0, %arg1) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[PRELU:%.+]] = VPU.PRelu([[SLICE]], %arg1) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[PRELU]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @SeluTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @SeluTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Selu(%arg0) {alpha_value = 1.6732000000000001 : f64, lambda_value = 1.0507000000000001 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[SELU:%.+]] = VPU.Selu([[SLICE]]) {alpha_value = {{.+}} : f64, lambda_value = {{.+}} : f64, tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SELU]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @TanTileOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf16>
func.func @TanTileOverH(%arg0: tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Tan(%arg0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %0 : tensor<1x16x128x256xf16>

// CHECK-DAG:    [[C64:%.+]] = arith.constant 64 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            [[TAN:%.+]] = VPU.Tan([[SLICE]]) {tiling_loop_index = 0 : i64}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[TAN]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 256] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: #[[$MAP_DYN:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 96)>
// CHECK: #[[$MAP_DYN1:.+]] = affine_map<(d0) -> (-d0 + 32, 5)>
// CHECK: #[[$MAP_DYN2:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$MAP_DYN3:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$MAP_DYN4:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$MAP_DYN5:.+]] = affine_map<(d0, d1) -> (0, d0 + d1 - 30)>
// CHECK: #[[$MAP_DYN6:.+]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>

// CHECK-LABEL:   @ApplyChannelUnevenTilingDynamicRawFilterShape
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x640x32x32xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<?x640x3x3xf16, {bounds = #const.OpaqueI64Elements<[640, 640, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
func.func @ApplyChannelUnevenTilingDynamicRawFilterShape(
    %arg0: tensor<1x640x32x32xf16, {order = #NHWC}>,
    %arg1: tensor<?x640x3x3xf16, {bounds = #const.OpaqueI64Elements<[640, 640, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x?x32x32xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 32, 32]> : tensor<4xsi64>, order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %dyn_oc = tensor.dim %arg1, %c0 : tensor<?x640x3x3xf16, {bounds = #const.OpaqueI64Elements<[640, 640, 3, 3]> : tensor<4xsi64>, order = #NHWC}>


    %0 = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape[%dyn_oc, 640, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 7, 1, 7]
    } : tensor<1x640x32x32xf16, {order = #NHWC}>, tensor<?x640x3x3xf16, {bounds = #const.OpaqueI64Elements<[640, 640, 3, 3]> : tensor<4xsi64>, order = #NHWC}>, index
      -> tensor<1x?x32x32xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 32, 32]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x?x32x32xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 32, 32]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[LOOP_END_W:%.+]] = arith.constant 32 : index
    // CHECK-DAG: [[LOOP_STEP_C:%.+]] = arith.constant 96 : index
    // CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant 5 : index
    // CHECK-DAG: [[LOOP_END_C:%.+]] = tensor.dim [[INPUT1]], [[LOOP_BEGIN]]

    // CHECK: [[LOOP_C:%.+]] = scf.for [[LOOP_ITER_C:%arg[0-9]+]] = [[LOOP_BEGIN]] to [[LOOP_END_C]] step [[LOOP_STEP_C]]
    // CHECK: [[LOOP_W:%.+]] = scf.for [[LOOP_ITER_W:%arg[0-9]+]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    // CHECK: [[SIZE_C:%.+]] = affine.min #[[$MAP_DYN]]([[LOOP_ITER_C]])[[[LOOP_END_C]]]

    // CHECK: [[SIZE_W_MAIN:%.+]] = affine.min #[[$MAP_DYN1]]([[LOOP_ITER_W]])
    // CHECK: [[OFF_W:%.+]] = affine.max #[[$MAP_DYN2]]([[LOOP_ITER_W]])
    // CHECK: [[V1:%.+]] = affine.max #[[$MAP_DYN3]]([[LOOP_ITER_W]])
    // CHECK: [[PAD_LOW:%.+]] = affine.min #[[$MAP_DYN4]]()[[[V1]]]
    // CHECK: [[V2:%.+]] = affine.max #[[$MAP_DYN5]]([[SIZE_W_MAIN]], [[OFF_W]])
    // CHECK: [[PAD_HIGH:%.+]] = affine.min #[[$MAP_DYN4]]()[[[V2]]]
    // CHECK: [[SIZE_W:%.+]] = affine.apply #[[$MAP_DYN6]]([[SIZE_W_MAIN]], [[PAD_LOW]], [[PAD_HIGH]])

    // CHECK: [[SLICE_INPUT:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, 0, [[OFF_W]]] [1, 640, 32, [[SIZE_W]]] [1, 1, 1, 1]
    // CHECK: [[SLICE_WEIGHTS:%.+]] = tensor.extract_slice [[INPUT1]][[[LOOP_ITER_C]], 0, 0, 0] [[[SIZE_C]], 640, 3, 3] [1, 1, 1, 1]
    // CHECK: [[PAD:%.+]] = tensor.pad [[SLICE_INPUT]] low[0, 0, 1, [[PAD_LOW]]] high[0, 0, 1, [[PAD_HIGH]]]

    // CHECK: [[INNER_CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[SLICE_WEIGHTS]]) rawFilterShape
    // CHECK-SAME: [[SIZE_C]], 640, 3, 3
}
