//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --scf-vertical-fusion="vf-merge-configuration=GREEDY enable-dynamic-dim-alignment=true" --resolve-shaped-type-result-dims --canonicalize --cse %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputConvDynamicType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
!inputD2SDynamicType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
!outputD2SDynamicType = tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>

module @test {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 640)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 128)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 2 - 1, 0)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0) -> (-(d0 floordiv 2) + 1, 0)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0, d1)[s0] -> (d0 - s0 + d1 floordiv 2 + 2, 0)>
// CHECK-DAG: #[[$MAP6:.+]] = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 floordiv 2 + 2)>

// CHECK: @Merge2DVFChainConvAddD2S
func.func @Merge2DVFChainConvAddD2S(%arg0: !inputConvDynamicType) -> !outputD2SDynamicType {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>

    %weights = const.Declare tensor<16x32x3x3xf16, {order = #NHWC}> = dense<1.0>
        : tensor<16x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [16, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
         strides = [1, 1],
        tilingStrategy = [1, 1, 1, 60]}
      : !inputConvDynamicType, tensor<16x32x3x3xf16, {order = #NHWC}> -> !inputD2SDynamicType

    %1 = VPU.DepthToSpace(%0) {
        block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 73, 1]}
          : !inputD2SDynamicType -> !outputD2SDynamicType

    return %1 : !outputD2SDynamicType

    // CHECK-DAG:   [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    // CHECK-DAG:   [[LOOP_STEP_W:%.+]] = arith.constant {{[0-9]+}} : index
    // CHECK-DAG:   [[LOOP_STEP_H:%.+]] = arith.constant {{[0-9]+}} : index
    // CHECK-DAG:   [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK-DAG:   [[CST_VAL_2:%.+]] = arith.constant 2 : index
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x32x3x3xf16, {order = #NHWC}>

    // CHECK:   [[DIM_H:%.+]] = tensor.dim [[ARG0]], [[CST_VAL_2]]
    // CHECK:   [[OUT_DIM_H:%.+]] = arith.muli [[DIM_H]], [[CST_VAL_2]]

    // CHECK:   [[DIM_W:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_W]]
    // CHECK:   [[OUT_DIM_W:%.+]] = arith.muli [[DIM_W]], [[CST_VAL_2]]

    // CHECK:   [[EMPTY:%.+]] = tensor.empty([[OUT_DIM_H]], [[OUT_DIM_W]]) : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>

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

    // CHECK:       [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IN_SLICE_OFFSET_H]], [[IN_SLICE_OFFSET_W]]] [1, 32, [[IN_SLICE_SIZE_H]], [[IN_SLICE_SIZE_W]]] [1, 1, 1, 1]

    // CHECK:       [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_BOTTOM]], [[PAD_LEFT]]] high[0, 0, [[PAD_TOP]], [[PAD_RIGHT]]]

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[WEIGHTS]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:    -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 320, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[CONV]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:    : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 320, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:    -> tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 640, 128]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[INSERT:%.+]] = tensor.insert_slice [[D2S]] into [[INNER_OUTPUT]]
    // CHECK:       scf.yield [[INSERT]]
    // CHECK:   scf.yield [[RESULT_W]]
    // CHECK:   return [[RESULT]] : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>
}
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 15)>

// CHECK-LABEL: @MergeVFPermuteCastAxis
func.func @MergeVFPermuteCastAxis(%arg0: tensor<1x?x1920x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>)
     -> tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}> {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x?x1920x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x?x1920x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64,
        lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1],
        tilingStrategy = [1, 1, 1, 80]}
        : tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
     -> tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    return %1: tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:   [[LOOP_STEP:%.+]] = arith.constant 15 : index
    // CHECK:   [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK:   [[DIM_INDEX:%.+]] = arith.constant 1 : index

    // CHECK:   [[DIM_C:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX]] : tensor<1x?x1920x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_C]]) : tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   [[LOOP:%.+]] = scf.for
    // CHECK-SAME:             [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[DIM_C]] step [[LOOP_STEP]]
    // CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                 [[INSERT_SIZE:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[DIM_C]]]
    // CHECK:                 [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, [[LOOP_ITER]], 0, 0] [1, [[INSERT_SIZE]], 1920, 16] [1, 1, 1, 1]
    // CHECK-SAME:            tensor<1x?x1920x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x?x1920x16xf16, {bounds = #const.OpaqueI64Elements<[1, 15, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:                 [[PERMUTECAST:%.+]] = VPU.PermuteCast([[SLICE]])
    // CHECK:                 [[CONV:%.+]] = VPU.NCE.Convolution([[PERMUTECAST]]
    // CHECK:                 [[INSERT:%.+]] = tensor.insert_slice [[CONV]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 16, [[INSERT_SIZE]], 1920] [1, 1, 1, 1]
    // CHECK-SAME:            tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 15, 1920]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   scf.yield [[INSERT]]

    // CHECK:   return [[LOOP]] : tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

config.Resources 6 of @NCE at 1.850000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 540)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 32)>

// CHECK-LABEL: @MergeVFPermuteCast2DAxis
func.func @MergeVFPermuteCast2DAxis(%arg0: tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>)
     -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}> {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64,
        lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1],
        tilingStrategy = [1, 1, 1, 80]}
        : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
     -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    return %1: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:   [[DIM_INDEX_H:%.+]] = arith.constant 2 : index
    // CHECK:   [[LOOP_STEP_H:%.+]] = arith.constant 32 : index
    // CHECK:   [[LOOP_STEP_C:%.+]] = arith.constant 540 : index
    // CHECK:   [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK:   [[DIM_INDEX_C:%.+]] = arith.constant 1 : index
    // CHECK:   [[DIM_C_0:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_C]] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[DIM_H_0:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_H]] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_C_0]], [[DIM_H_0]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:   [[LOOP_C:%.+]] = scf.for
    // CHECK-SAME:              [[LOOP_ITER_C:%arg[0-9]]] = [[LOOP_BEGIN]] to [[DIM_C_0]] step [[LOOP_STEP_C]]
    // CHECK-SAME:              iter_args([[LOOP_OUT_C:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>)
    // CHECK:                   [[LOOP_H:%.+]] = scf.for
    // CHECK-SAME:              [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[DIM_H_0]] step [[LOOP_STEP_H]]
    // CHECK-SAME:             iter_args([[LOOP_OUT_H:%arg[0-9]]] = [[LOOP_OUT_C]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>)

    // CHECK:                 [[SLICE_SIZE_C:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER_C]])[[[DIM_C_0]]]
    // CHECK:                 [[SLICE_SIZE_H:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_H]])[[[DIM_H_0]]]
    // CHECK:                 [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, [[LOOP_ITER_C]], [[LOOP_ITER_H]], 0] [1, [[SLICE_SIZE_C]], [[SLICE_SIZE_H]], 16] [1, 1, 1, 1] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 540, 32, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:                 [[PERMUTECAST:%.+]] = VPU.PermuteCast([[SLICE]])
    // CHECK:                 [[CONV:%.+]] = VPU.NCE.Convolution([[PERMUTECAST]]
    // CHECK:                 [[INSERT:%.+]] = tensor.insert_slice [[CONV]] into [[LOOP_OUT_H]][0, 0, [[LOOP_ITER_C]], [[LOOP_ITER_H]]] [1, 16, [[SLICE_SIZE_C]], [[SLICE_SIZE_H]]] [1, 1, 1, 1]
    // CHECK-SAME:            tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 32]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:   scf.yield [[INSERT]]
    // CHECK:   scf.yield [[LOOP_H]]
    // CHECK:   return [[LOOP_C]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 270)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
// CHECK-DAG: #[[$MAP6:.+]] = affine_map<(d0) -> (0, d0 * 2 - 1)>
// CHECK-DAG: #[[$MAP7:.+]] = affine_map<(d0) -> (d0 * -2 + 1, 0)>
// CHECK-DAG: #[[$MAP8:.+]] = affine_map<(d0, d1, d2, d3) -> (-d0 + d1 * 2 - d2 * 2 - d3 * 2 + 5)>
!inputDynamicType = tensor<1x16x?x?x!quant.uniform<ui8:f16, 0.0019697112195632038>, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
!outputDynamicType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
!qElemType1 = !quant.uniform<ui8:f16, 0.034925088695451328:128>
!qElemType3 = !quant.uniform<ui8:f16, 0.0091306873396331187:128>

module @test {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
  }
  config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.Resources 1 of @global {
    config.ExecutorResource 1 of @M2I
    config.ExecutorResource 2 of @DMA_NN
    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  }

  // CHECK-LABEL: @MergeVFQuantizedChain2TilesAligned
  // CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
  func.func @MergeVFQuantizedChain2TilesAligned(%arg0: !inputDynamicType)
      -> !outputDynamicType {
      %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1> : tensor<32x16x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<i8:f16, 0.034925088695451328>>, #const.ConvertElemType<!quant.uniform<ui8:f16, 0.034925088695451328:128>>, #const.Reorder<#NHWC>]
      %cst_1 = const.Declare tensor<32x32x3x3x!qElemType3, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<i8:f16, 0.0091306873396331187>>, #const.ConvertElemType<!quant.uniform<ui8:f16, 0.0091306873396331187:128>>, #const.Reorder<#NHWC>]
      %3 = VPU.NCE.Convolution(%arg0, %cst_0) rawFilterShape [32, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.100000e+02 : f64, clamp_high = 1.450000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.100000e+02 : f64>,  strides = [2, 2], tilingStrategy = [1, 1, 23, 4]} : !inputDynamicType, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x?x?x!quant.uniform<ui8:f16, 0.030033121856988646:110>, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %4 = VPU.NCE.Convolution(%3, %cst_1) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <LRELU>, clamp_low = 0.000000e+00 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [-0.000000e+00], adder = 0.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 23, 4]} : tensor<1x32x?x?x!quant.uniform<ui8:f16, 0.030033121856988646:110>, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3x!qElemType3, {order = #NHWC}> -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      return %4: !outputDynamicType

    // CHECK: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    // CHECK: [[PAD_VALUE:%.+]] = arith.constant 110 : i8
    // CHECK: [[ZERO_PAD_VALUE:%.+]] = arith.constant 0 : i8
    // CHECK: [[TILE_STEP_W:%.+]] = arith.constant 16 : index
    // CHECK: [[TILE_STEP_H:%.+]] = arith.constant 270 : index
    // CHECK: [[START_INDEX:%.+]] = arith.constant 0 : index
    // CHECK: [[CONV1_WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1> : tensor<32x16x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType2>, #const.ConvertElemType<!qElemType1>, #const.Reorder<#NHWC>]
    // CHECK: [[CONV2_WEIGHTS:%.+]] = const.Declare tensor<32x32x3x3x!qElemType3, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType4>, #const.ConvertElemType<!qElemType3>, #const.Reorder<#NHWC>]
    // CHECK: [[DIM_INDEX_H:%.+]] = arith.constant 2 : index
    // CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[HALF_HEIGHT:%.+]] = arith.divsi [[DIM_H]], [[DIM_INDEX_H]] : index
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[HALF_WIDTH:%.+]] = arith.divsi [[DIM_W]], [[DIM_INDEX_H]] : index
    // CHECK: [[OUTPUT_BUF:%.+]] = tensor.empty([[HALF_HEIGHT]], [[HALF_WIDTH]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_H:%.+]] = scf.for [[LOOP_ITER:%arg[0-9]]] = [[START_INDEX]] to [[HALF_HEIGHT]] step [[TILE_STEP_H]] iter_args([[LOOP_ITER_1:%arg[0-9]]] = [[OUTPUT_BUF]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK: [[LOOP_W:%.+]] = scf.for [[LOOP_ITER_2:%arg[0-9]]] = [[START_INDEX]] to [[HALF_WIDTH]] step [[TILE_STEP_W]] iter_args([[LOOP_ITER_3:%arg[0-9]]] = [[LOOP_ITER_1]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK:     [[MIN_HEIGHT:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[HALF_HEIGHT]]]
    // CHECK:     [[MIN_WIDTH:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_2]])[[[HALF_WIDTH]]]

    // CHECK:     [[MAX_HEIGHT_TEMP:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER]])
    // CHECK:     [[MAX_WIDTH_TEMP:%.+]] = affine.max #[[$MAP3]]([[LOOP_ITER]])
    // CHECK:     [[PAD_LOW_TEMP_H:%.+]] = affine.min #[[$MAP4]]()[[[MAX_WIDTH_TEMP]]]
    // CHECK:     [[MAX_HEIGHT_FINAL:%.+]] = affine.max #[[$MAP5]]([[MIN_HEIGHT]], [[MAX_HEIGHT_TEMP]])[[[HALF_HEIGHT]]]

    // CHECK:     [[PAD_HIGH_TEMP_H:%.+]] = affine.min #[[$MAP4]]()[[[MAX_HEIGHT_FINAL]]]
    // CHECK:     [[MAX_HEIGHT_TEMP_2:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER_2]])
    // CHECK:     [[MAX_WIDTH_TEMP_2:%.+]] = affine.max #[[$MAP3]]([[LOOP_ITER_2]])
    // CHECK:     [[PAD_LOW_TEMP_W:%.+]] = affine.min #[[$MAP4]]()[[[MAX_WIDTH_TEMP_2]]]
    // CHECK:     [[MAX_WIDTH_FINAL:%.+]] = affine.max #[[$MAP5]]([[MIN_WIDTH]], [[MAX_HEIGHT_TEMP_2]])[[[HALF_WIDTH]]]

    // CHECK:     [[PAD_HIGH_TEMP_W:%.+]] = affine.min #[[$MAP4]]()[[[MAX_WIDTH_FINAL]]]
    // CHECK:     [[MAX_HEIGHT_FULL:%.+]] = affine.max #[[$MAP6]]([[MAX_HEIGHT_TEMP]])
    // CHECK:     [[MAX_WIDTH_FULL:%.+]] = affine.max #[[$MAP7]]([[MAX_HEIGHT_TEMP]])
    // CHECK:     [[PAD_LOW_FULL_H:%.+]] = affine.min #[[$MAP4]]()[[[MAX_WIDTH_FULL]]]
    // CHECK:     [[APPLY_HEIGHT:%.+]] = affine.apply #[[$MAP8]]([[PAD_LOW_FULL_H]], [[MIN_HEIGHT]], [[PAD_LOW_TEMP_H]], [[PAD_HIGH_TEMP_H]])

    // CHECK:     [[MAX_HEIGHT_FULL_2:%.+]] = affine.max #[[$MAP6]]([[MAX_HEIGHT_TEMP_2]])
    // CHECK:     [[MAX_WIDTH_FULL_2:%.+]] = affine.max #[[$MAP7]]([[MAX_HEIGHT_TEMP_2]])
    // CHECK:     [[PAD_LOW_FULL_W:%.+]] = affine.min #[[$MAP4]]()[[[MAX_WIDTH_FULL_2]]]
    // CHECK:     [[APPLY_WIDTH:%.+]] = affine.apply #[[$MAP8]]([[PAD_LOW_FULL_W]], [[MIN_WIDTH]], [[PAD_LOW_TEMP_W]], [[PAD_HIGH_TEMP_W]])

    // CHECK:     [[EXTRACTED_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[MAX_HEIGHT_FULL]], [[MAX_HEIGHT_FULL_2]]] [1, 16, [[APPLY_HEIGHT]], [[APPLY_WIDTH]]] [1, 1, 1, 1] : tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 32]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:     [[PAD_CAST:%.+]] = builtin.unrealized_conversion_cast [[ZERO_PAD_VALUE]] : i8 to !qElemType
    // CHECK:     [[PADDED_TENSOR:%.+]] = tensor.pad [[EXTRACTED_SLICE]] low[0, 0, [[PAD_LOW_FULL_H]], [[PAD_LOW_FULL_W]]] high[0, 0, 0, 0] {
    // CHECK-NEXT:   ^bb0([[PAD_ARG_H:%.+]]: index, [[PAD_ARG_W:%.+]]: index, [[PAD_IXD_0:%.+]]: index, [[PAD_IXD_1:%.+]]: index):
    // CHECK-NEXT:   tensor.yield [[PAD_CAST]] : !qElemType
    // CHECK:     } : tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 540, 32]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 541, 33]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:     [[CONVOLUTION_0:%.+]] = VPU.NCE.Convolution([[PADDED_TENSOR]], [[CONV1_WEIGHTS]])
    // CHECK-SAME: -> tensor<1x32x?x?x!qElemType5, {bounds = #const.OpaqueI64Elements<[1, 32, 270, 16]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:     [[PAD_CAST_2:%.+]] = builtin.unrealized_conversion_cast [[PAD_VALUE:%.+]] : i8 to !qElemType5
    // CHECK:     [[PADDED_TILE:%.+]] = tensor.pad [[CONVOLUTION_0]] low[0, 0, [[PAD_LOW_TEMP_H]], [[PAD_LOW_TEMP_W]]] high[0, 0, [[PAD_HIGH_TEMP_H]], [[PAD_HIGH_TEMP_W]]] {
    // CHECK-NEXT: ^bb0([[PAD_ARG_H]]: index, [[PAD_ARG_W]]: index, [[PAD_IXD_0]]: index, [[PAD_IXD_1]]: index):
    // CHECK-NEXT: tensor.yield [[PAD_CAST_2]] : !qElemType5
    // CHECK: } : tensor<1x32x?x?x!qElemType5, {bounds = #const.OpaqueI64Elements<[1, 32, 270, 16]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?x!qElemType5, {bounds = #const.OpaqueI64Elements<[1, 32, 272, 18]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:     [[CONVOLUTION_1:%.+]] = VPU.NCE.Convolution([[PADDED_TILE]], [[CONV2_WEIGHTS]])
    // CHECK-SAME: -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 270, 16]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:     [[INSERTED_SLICE:%.+]] = tensor.insert_slice [[CONVOLUTION_1]] into [[LOOP_ITER_3]][0, 0, [[LOOP_ITER]], [[LOOP_ITER_2]]] [1, 32, [[MIN_HEIGHT]], [[MIN_WIDTH]]] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 270, 16]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:     scf.yield [[INSERTED_SLICE]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   scf.yield [[LOOP_W]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: return [[LOOP_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 62)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
//CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 2)>


module @test {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

config.PipelineOptions @Options {
        config.Option @VPU.AutoPaddingODU : true
        config.Option @VPU.AutoPaddingIDU : true
        config.Option @VPU.ReduceSupported : false
}

// CHECK: @AlignD2SOutput
func.func @AlignD2SOutput(%arg0: tensor<1x64x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> )
-> tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>  {
    // CHECK-SAME:  [[INPUT:%.+]]: tensor<1x64x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

   %0 = VPU.DepthToSpace(%arg0) {
        block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 67, 1]}
        : tensor<1x64x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
        -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>

   %1 = VPU.NCE.Eltwise(%0, %0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        tilingStrategy = [1, 1, 1, 72],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        -> tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

   return %1 : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

   // CHECK:   [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
   // CHECK:   [[LOOP_STEP_W:%.+]] = arith.constant 256 : index
   // CHECK:   [[LOOP_STEP_H:%.+]] = arith.constant 62 : index
   // CHECK:   [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

   // CHECK:   [[CONST_2:%.+]] = arith.constant 2 : index
   // CHECK:   [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[CONST_2]] : tensor<1x64x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
   // CHECK:   [[DIM_H_MUL2:%.+]] = arith.muli [[DIM_H]], [[CONST_2]] : index
   // CHECK:   [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x64x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
   // CHECK:   [[DIM_W_MUL2:%.+]] = arith.muli [[DIM_W]], [[CONST_2]] : index

   // CHECK:   [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_H_MUL2]], [[DIM_W_MUL2]]) : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

   // CHECK:   [[LOOP_H:%.+]] = scf.for
   // CHECK-SAME:              [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[DIM_H_MUL2]] step [[LOOP_STEP_H]]
   // CHECK-SAME:              iter_args([[LOOP_OUT_H:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>)

   // CHECK:                   [[LOOP_W:%.+]] = scf.for
   // CHECK-SAME:                               [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[DIM_W_MUL2]] step [[LOOP_STEP_W]]
   // CHECK-SAME:                               iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT_H]])

   // CHECK:                                    [[TMP_VALUE_H:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER_H]])[[[DIM_H_MUL2]]]
   // CHECK:                                    [[TMP_VALUE_W:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_W]])[[[DIM_W_MUL2]]]
   // CHECK:                                    [[SLICE_OFFSET_H:%.+]] = affine.apply #[[$MAP2]]([[LOOP_ITER_H]])
   // CHECK:                                    [[SLICE_SIZE_H:%.+]] = affine.apply #[[$MAP2]]([[TMP_VALUE_H]])
   // CHECK:                                    [[SLICE_OFFSET_W:%.+]] = affine.apply #[[$MAP2]]([[LOOP_ITER_W]])
   // CHECK:                                    [[SLICE_SIZE_W:%.+]] = affine.apply #[[$MAP2]]([[TMP_VALUE_W]])

    // CHECK:                                   [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET_H]], [[SLICE_OFFSET_W]]] [1, 64, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:                              tensor<1x64x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x64x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 31, 128]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                                   [[D2S:%.+]] = VPU.DepthToSpace([[SLICE]])
    // CHECK:                                   [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[D2S]], [[D2S]])
    // CHECK:                                   [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, [[TMP_VALUE_H]], [[TMP_VALUE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:                              tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 62, 256]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   scf.yield [[INSERT]]

    // CHECK:   scf.yield [[LOOP_W]]
    // CHECK:   return [[LOOP_H]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
}
}

// -----

// Test MC strategy adjustment with dynamic shapes:
// Producer Conv 1x1 has SplitOverKernel, consumer Conv 3x3 has SplitOverHeight.
// The adjustment logic should change the producer from SOK to a compatible strategy
// (HKSwitch or SOH) so that fusion can proceed even with dynamic spatial dimensions.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputDynType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
!outputDynType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

module @testDynMCStrategyAdjust {
config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FusionWithMCStrategyAdjustmentDynamic
func.func @FusionWithMCStrategyAdjustmentDynamic(%arg0: !inputDynType) -> !outputDynType {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %weights2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // Producer: Conv 1x1 with SOK — will be adjusted to enable fusion
    %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 22]}
      : !inputDynType, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> !outputDynType

    // Consumer: Conv 3x3 with SOH — requires OVERLAPPED input
    %1 = VPU.NCE.Convolution(%0, %weights2) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 22]}
      : !outputDynType, tensor<32x32x3x3xf16, {order = #NHWC}>
      -> !outputDynType

    return %1 : !outputDynType
}

// Strategy adjustment enables fusion with dynamic shapes — producer adjusted from SOK
// CHECK: scf.for
// CHECK: VPU.NCE.Convolution
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>
// CHECK: VPU.NCE.Convolution
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}
