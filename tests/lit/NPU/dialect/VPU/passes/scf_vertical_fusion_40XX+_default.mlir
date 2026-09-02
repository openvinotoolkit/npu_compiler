//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --scf-vertical-fusion="vf-merge-configuration=COST_BASED" --resolve-shaped-type-result-dims --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$OUT_OFFSET_AND_SIZE_MAP:.+]] = affine_map<(d0) -> ((d0 floordiv 26) * 26, (d0 floordiv 26) * 25 + 35)>
// CHECK: #[[$TILE_INDEX_CAP_AND_REMAINDER_MAP:.+]] = affine_map<(d0) -> (d0 floordiv 26, 35)>
// CHECK: #[[$OUT_SIZE_BY_CAPPED_TILE_INDEX_MAP:.+]] = affine_map<(d0) -> (-d0 + 60, 26)>
// CHECK: #[[$TO_SLICE_OFFSET_MAP:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$TO_PAD_CANDIDATE_MAP:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$PAD_CLAMP_MAP:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$PAD_HIGH_STAGE5_CANDIDATE_MAP:.+]] = affine_map<(d0, d1) -> (0, d0 + d1 - 958)>
// CHECK: #[[$PAD_HIGH_STAGE4_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (0, d0 + d1 - d2 - d3 - 956)>
// CHECK: #[[$PAD_HIGH_STAGE3_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d0 - d1 - d2 + d3 - d4 - d5 - 954)>
// CHECK: #[[$PAD_HIGH_STAGE2_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (0, d0 - d1 - d2 - d3 - d4 + d5 - d6 - d7 - 952)>
// CHECK: #[[$PAD_HIGH_STAGE1_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 + d7 - d8 - d9 - 950)>
// CHECK: #[[$PAD_HIGH_STAGE0_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 + d9 - d10 - d11 - 948)>
// CHECK: #[[$SLICE_SIZE_BY_OUT_AND_PAD_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12) -> (-d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - d9 + d10 - d11 - d12 + 12)>

// CHECK-LABEL: @MergeVFChain3Tiles
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x256x540x120xf16, {order = #NHWC}>)
func.func @MergeVFChain3Tiles(%arg0: tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>
 {
    %cst = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_9 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs(%arg0 : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_0) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_1) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %4 = VPU.NCE.DepthConvolution(%3, %cst_9) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %5 = VPU.NCE.Convolution(%4, %cst_2) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %6 = VPU.NCE.Convolution(%5, %cst_3) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %7 = VPU.NCE.Convolution(%6, %cst_4) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %8 = VPU.NCE.DepthConvolution(%7, %cst_5) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 20]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %9 = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs(%8 : tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>

    return %9: tensor<1x128x540x240xf16, {order = #NHWC}>

    // CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 26 : index
    // CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 960 : index
    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK: [[CAST_INPUT:%.+]] = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs([[INPUT]] : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    // CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>) {

    // CHECK:        [[OUT_OFFSET:%.+]] = affine.min #[[$OUT_OFFSET_AND_SIZE_MAP]]([[LOOP_ITER]])
    // CHECK:        [[OUT_TILE_INDEX_CAP:%.+]] = affine.min #[[$TILE_INDEX_CAP_AND_REMAINDER_MAP]]([[LOOP_ITER]])
    // CHECK:        [[OUT_SIZE:%.+]] = affine.min #[[$OUT_SIZE_BY_CAPPED_TILE_INDEX_MAP]]([[OUT_TILE_INDEX_CAP]])

    // CHECK:        [[TEMP_VALUE0:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[OUT_OFFSET]])
    // CHECK:        [[TEMP_VALUE1:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[OUT_OFFSET]])
    // CHECK:        [[PAD_LOW5:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE1]]]
    // CHECK:        [[TEMP_VALUE2:%.+]] = affine.max #[[$PAD_HIGH_STAGE5_CANDIDATE_MAP]]([[OUT_SIZE]], [[TEMP_VALUE0]])
    // CHECK:        [[PAD_HIGH5:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE2]]]
    // CHECK:        [[TEMP_VALUE3:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE0]])
    // CHECK:        [[TEMP_VALUE4:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE0]])
    // CHECK:        [[PAD_LOW4:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE4]]]
    // CHECK:        [[IN_SIZE0:%.+]] = affine.max #[[$PAD_HIGH_STAGE4_CANDIDATE_MAP]]([[TEMP_VALUE3]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH4:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[IN_SIZE0]]]
    // CHECK:        [[TEMP_VALUE5:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE3]])
    // CHECK:        [[TEMP_VALUE6:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE3]])
    // CHECK:        [[PAD_LOW3:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE6]]]
    // CHECK:        [[TEMP_VALUE7:%.+]] = affine.max #[[$PAD_HIGH_STAGE3_CANDIDATE_MAP]]([[TEMP_VALUE5]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH3:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE7]]]
    // CHECK:        [[TEMP_VALUE8:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE5]])
    // CHECK:        [[TEMP_VALUE9:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE5]])
    // CHECK:        [[PAD_LOW2:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE9]]]
    // CHECK:        [[TEMP_VALUE10:%.+]] = affine.max #[[$PAD_HIGH_STAGE2_CANDIDATE_MAP]]([[TEMP_VALUE8]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH2:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE10]]]
    // CHECK:        [[TEMP_VALUE11:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE8]])
    // CHECK:        [[TEMP_VALUE12:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE8]])
    // CHECK:        [[PAD_LOW1:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE12]]]
    // CHECK:        [[TEMP_VALUE13:%.+]] = affine.max #[[$PAD_HIGH_STAGE1_CANDIDATE_MAP]]([[TEMP_VALUE11]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH1:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE13]]]
    // CHECK:        [[SLICE_OFFSET:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE11]])
    // CHECK:        [[TEMP_VALUE14:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE11]])
    // CHECK:        [[PAD_LOW0:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE14]]]
    // CHECK:        [[TEMP_VALUE15:%.+]] = affine.max #[[$PAD_HIGH_STAGE0_CANDIDATE_MAP]]([[SLICE_OFFSET]], [[PAD_LOW1]], [[PAD_HIGH1]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH0:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE15]]]
    // CHECK:        [[SIZE_W:%.+]] = affine.apply #[[$SLICE_SIZE_BY_OUT_AND_PAD_MAP]]([[PAD_LOW0]], [[PAD_HIGH0]], [[PAD_LOW1]], [[PAD_HIGH1]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[CAST_INPUT]][0, 0, 0, [[SLICE_OFFSET]]] [1, 32, 540, [[SIZE_W]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:        [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, 1, [[PAD_LOW0]]] high[0, 0, 1, [[PAD_HIGH0]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:        tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[PAD1:%.+]] = tensor.pad [[CONV0]] low[0, 0, 1, [[PAD_LOW1]]] high[0, 0, 1, [[PAD_HIGH1]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:        [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, 1, [[PAD_LOW2]]] high[0, 0, 1, [[PAD_HIGH2]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:        [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[CONV2]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[PAD3:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, 1, [[PAD_LOW3]]] high[0, 0, 1, [[PAD_HIGH3]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:        [[CONV3:%.+]] = VPU.NCE.Convolution([[PAD3]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[PAD4:%.+]] = tensor.pad [[CONV3]] low[0, 0, 1, [[PAD_LOW4]]] high[0, 0, 1, [[PAD_HIGH4]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:        [[CONV4:%.+]] = VPU.NCE.Convolution([[PAD4]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[PAD5:%.+]] = tensor.pad [[CONV4]] low[0, 0, 1, [[PAD_LOW5]]] high[0, 0, 1, [[PAD_HIGH5]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:        [[CONV5:%.+]] = VPU.NCE.Convolution([[PAD5]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution([[CONV5]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[INSERT:%.+]] = tensor.insert_slice [[DWCONV1]] into [[LOOP_OUT]][0, 0, 0, [[OUT_OFFSET]]] [1, 32, 540, [[OUT_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK:    scf.yield [[INSERT]] : tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK:    [[CAST:%.+]] = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs([[LOOP]]
    // CHECK:    return [[CAST]] : tensor<1x128x540x240xf16, {order = #NHWC}>
}

// -----

config.Resources 6 of @NCE at 1.850000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$OUT_OFFSET_AND_SIZE_MAP:.+]] = affine_map<(d0) -> ((d0 floordiv 51) * 51, (d0 floordiv 51) * 50 + 10)>
// CHECK: #[[$TILE_INDEX_CAP_AND_REMAINDER_MAP:.+]] = affine_map<(d0) -> (d0 floordiv 51, 10)>
// CHECK: #[[$OUT_SIZE_BY_CAPPED_TILE_INDEX_MAP:.+]] = affine_map<(d0) -> (-d0 + 60, 51)>
// CHECK: #[[$TO_SLICE_OFFSET_MAP:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$TO_PAD_CANDIDATE_MAP:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$PAD_CLAMP_MAP:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$PAD_HIGH_STAGE5_CANDIDATE_MAP:.+]] = affine_map<(d0, d1) -> (0, d0 + d1 - 958)>
// CHECK: #[[$PAD_HIGH_STAGE4_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (0, d0 + d1 - d2 - d3 - 956)>
// CHECK: #[[$PAD_HIGH_STAGE3_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d0 - d1 - d2 + d3 - d4 - d5 - 954)>
// CHECK: #[[$PAD_HIGH_STAGE2_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (0, d0 - d1 - d2 - d3 - d4 + d5 - d6 - d7 - 952)>
// CHECK: #[[$PAD_HIGH_STAGE1_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 + d7 - d8 - d9 - 950)>
// CHECK: #[[$PAD_HIGH_STAGE0_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 + d9 - d10 - d11 - 948)>
// CHECK: #[[$SLICE_SIZE_BY_OUT_AND_PAD_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12) -> (-d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - d9 + d10 - d11 - d12 + 12)>

// CHECK-LABEL: @MergeVFChain6Tiles
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x256x540x120xf16, {order = #NHWC}>)
func.func @MergeVFChain6Tiles(%arg0: tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>
 {
    %cst = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_6 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs(%arg0 : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_0) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_1) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %4 = VPU.NCE.DepthConvolution(%3, %cst_6) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %5 = VPU.NCE.Convolution(%4, %cst_2) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %6 = VPU.NCE.Convolution(%5, %cst_3) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %7 = VPU.NCE.Convolution(%6, %cst_4) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %8 = VPU.NCE.DepthConvolution(%7, %cst_5) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 20]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %9 = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs(%8 : tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>

    return %9: tensor<1x128x540x240xf16, {order = #NHWC}>

    // CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 51 : index
    // CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 960 : index
    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK: [[CAST_INPUT:%.+]] = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs([[INPUT]] : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    // CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>) {

    // CHECK:        [[OUT_OFFSET:%.+]] = affine.min #[[$OUT_OFFSET_AND_SIZE_MAP]]([[LOOP_ITER]])
    // CHECK:        [[OUT_TILE_INDEX_CAP:%.+]] = affine.min #[[$TILE_INDEX_CAP_AND_REMAINDER_MAP]]([[LOOP_ITER]])
    // CHECK:        [[OUT_SIZE:%.+]] = affine.min #[[$OUT_SIZE_BY_CAPPED_TILE_INDEX_MAP]]([[OUT_TILE_INDEX_CAP]])

    // CHECK:        [[TEMP_VALUE0:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[OUT_OFFSET]])
    // CHECK:        [[TEMP_VALUE1:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[OUT_OFFSET]])
    // CHECK:        [[PAD_LOW5:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE1]]]
    // CHECK:        [[TEMP_VALUE2:%.+]] = affine.max #[[$PAD_HIGH_STAGE5_CANDIDATE_MAP]]([[OUT_SIZE]], [[TEMP_VALUE0]])
    // CHECK:        [[PAD_HIGH5:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE2]]]
    // CHECK:        [[TEMP_VALUE3:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE0]])
    // CHECK:        [[TEMP_VALUE4:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE0]])
    // CHECK:        [[PAD_LOW4:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE4]]]
    // CHECK:        [[IN_SIZE0:%.+]] = affine.max #[[$PAD_HIGH_STAGE4_CANDIDATE_MAP]]([[TEMP_VALUE3]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH4:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[IN_SIZE0]]]
    // CHECK:        [[TEMP_VALUE5:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE3]])
    // CHECK:        [[TEMP_VALUE6:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE3]])
    // CHECK:        [[PAD_LOW3:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE6]]]
    // CHECK:        [[TEMP_VALUE7:%.+]] = affine.max #[[$PAD_HIGH_STAGE3_CANDIDATE_MAP]]([[TEMP_VALUE5]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH3:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE7]]]
    // CHECK:        [[TEMP_VALUE8:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE5]])
    // CHECK:        [[TEMP_VALUE9:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE5]])
    // CHECK:        [[PAD_LOW2:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE9]]]
    // CHECK:        [[TEMP_VALUE10:%.+]] = affine.max #[[$PAD_HIGH_STAGE2_CANDIDATE_MAP]]([[TEMP_VALUE8]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH2:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE10]]]
    // CHECK:        [[TEMP_VALUE11:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE8]])
    // CHECK:        [[TEMP_VALUE12:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE8]])
    // CHECK:        [[PAD_LOW1:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE12]]]
    // CHECK:        [[TEMP_VALUE13:%.+]] = affine.max #[[$PAD_HIGH_STAGE1_CANDIDATE_MAP]]([[TEMP_VALUE11]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH1:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE13]]]
    // CHECK:        [[SLICE_OFFSET:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE11]])
    // CHECK:        [[TEMP_VALUE14:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE11]])
    // CHECK:        [[PAD_LOW0:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE14]]]
    // CHECK:        [[TEMP_VALUE15:%.+]] = affine.max #[[$PAD_HIGH_STAGE0_CANDIDATE_MAP]]([[SLICE_OFFSET]], [[PAD_LOW1]], [[PAD_HIGH1]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH0:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE15]]]
    // CHECK:        [[SIZE_W:%.+]] = affine.apply #[[$SLICE_SIZE_BY_OUT_AND_PAD_MAP]]([[PAD_LOW0]], [[PAD_HIGH0]], [[PAD_LOW1]], [[PAD_HIGH1]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[CAST_INPUT]][0, 0, 0, [[SLICE_OFFSET]]] [1, 32, 540, [[SIZE_W]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:        [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, 1, [[PAD_LOW0]]] high[0, 0, 1, [[PAD_HIGH0]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:        tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[PAD1:%.+]] = tensor.pad [[CONV0]] low[0, 0, 1, [[PAD_LOW1]]] high[0, 0, 1, [[PAD_HIGH1]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:        [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, 1, [[PAD_LOW2]]] high[0, 0, 1, [[PAD_HIGH2]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:        [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[CONV2]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[PAD3:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, 1, [[PAD_LOW3]]] high[0, 0, 1, [[PAD_HIGH3]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:        [[CONV3:%.+]] = VPU.NCE.Convolution([[PAD3]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[PAD4:%.+]] = tensor.pad [[CONV3]] low[0, 0, 1, [[PAD_LOW4]]] high[0, 0, 1, [[PAD_HIGH4]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:        [[CONV4:%.+]] = VPU.NCE.Convolution([[PAD4]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[PAD5:%.+]] = tensor.pad [[CONV4]] low[0, 0, 1, [[PAD_LOW5]]] high[0, 0, 1, [[PAD_HIGH5]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:        [[CONV5:%.+]] = VPU.NCE.Convolution([[PAD5]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution([[CONV5]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[INSERT:%.+]] = tensor.insert_slice [[DWCONV1]] into [[LOOP_OUT]][0, 0, 0, [[OUT_OFFSET]]] [1, 32, 540, [[OUT_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK:    scf.yield [[INSERT]] : tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK:    [[CAST:%.+]] = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs([[LOOP]]
    // CHECK:    return [[CAST]] : tensor<1x128x540x240xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

func.func @CastingDynamicPaddedOutputToStaticInVFChain(%arg0: tensor<1x4x1600x2560xf16>) -> tensor<1x16x1600x640xf16, {order = #NCHW}> {
  %cst = const.Declare tensor<16x32x3x3xf16, {order = #NHWC}> = dense<1.234380e+00> : tensor<16x32x3x3xf16>, [#const.Reorder<#NHWC>]
  %cst_2 = const.Declare tensor<64x16x3x2xf16, {order = #NHWC}> = dense<"0x1234"> : tensor<64x16x3x2xf16, {order = #NHWC}>
  %1 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 4 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 16]} -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
  %2 = VPU.ShapeCast {shape = [1, 16, 1600, 640]} inputs(%1 : tensor<1x4x1600x2560xf16, {order = #NHWC}>) -> tensor<1x16x1600x640xf16, {order = #NHWC}>
  %3 = VPU.NCE.Convolution(%2, %cst_2) rawFilterShape [64, 16, 3, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [2, 1], tilingStrategy = [1, 1, 1, 59]} : tensor<1x16x1600x640xf16, {order = #NHWC}>, tensor<64x16x3x2xf16, {order = #NHWC}> -> tensor<1x64x800x640xf16, {order = #NHWC}>
  %4 = VPU.ShapeCast {shape = [1, 32, 800, 1280]} inputs(%3 : tensor<1x64x800x640xf16, {order = #NHWC}>) -> tensor<1x32x800x1280xf16, {order = #NHWC}>
  %5 = VPU.NCE.Convolution(%4, %cst) rawFilterShape [16, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 59]} : tensor<1x32x800x1280xf16, {order = #NHWC}>, tensor<16x32x3x3xf16, {order = #NHWC}> -> tensor<1x16x800x1280xf16, {order = #NHWC}>
  %6 = VPU.DepthToSpace(%5) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 16, 1]} : tensor<1x16x800x1280xf16, {order = #NHWC}> -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
  %7 = VPU.ShapeCast {shape = [1, 16, 1600, 640]} inputs(%6 : tensor<1x4x1600x2560xf16, {order = #NHWC}>) -> tensor<1x16x1600x640xf16, {order = #NHWC}>
  %8 = VPU.NCE.MaxPool(%7) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], tilingStrategy = [1, 1, 1, 28]} -> tensor<1x16x1600x640xf16, {order = #NCHW}>
  return %8 : tensor<1x16x1600x640xf16, {order = #NCHW}>

// CHECK: func.func @CastingDynamicPaddedOutputToStaticInVFChain([[ARG0:%.+]]: tensor<1x4x1600x2560xf16>) -> tensor<1x16x1600x640xf16, {order = #NCHW}> {
// CHECK:    [[CST:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK:    [[C40:%.+]] = arith.constant 40 : index
// CHECK:    [[C2560:%.+]] = arith.constant 2560 : index
// CHECK:    [[C0:%.+]] = arith.constant 0 : index
// CHECK:    [[CST0:%.+]] = const.Declare tensor<16x32x3x3xf16, {order = #NHWC}> = dense<1.234380e+00> : tensor<16x32x3x3xf16>, [#const.Reorder<#NHWC>]
// CHECK:    [[CST1:%.+]] = const.Declare tensor<64x16x3x2xf16, {order = #NHWC}> = dense<2.543950e-01> : tensor<64x16x3x2xf16, {order = #NHWC}>
// CHECK:    [[PERMUTE:%.+]] = VPU.NCE.Permute([[ARG0]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 4 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 16]} -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
// CHECK:    [[SHAPECAST:%.+]] = VPU.ShapeCast {shape = [1, 16, 1600, 640]} inputs([[PERMUTE]] : tensor<1x4x1600x2560xf16, {order = #NHWC}>) -> tensor<1x16x1600x640xf16, {order = #NHWC}>
// CHECK:    [[CONV:%.+]] = VPU.NCE.Convolution([[SHAPECAST]], [[CST1]]) rawFilterShape [64, 16, 3, 2] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
// CHECK-SAME: strides = [2, 1], tilingStrategy = [1, 1, 1, 59]} : tensor<1x16x1600x640xf16, {order = #NHWC}>, tensor<64x16x3x2xf16, {order = #NHWC}> -> tensor<1x64x800x640xf16, {order = #NHWC}>
// CHECK:    [[SHAPECAST0:%.+]]  = VPU.ShapeCast {shape = [1, 32, 800, 1280]} inputs([[CONV]] : tensor<1x64x800x640xf16, {order = #NHWC}>) -> tensor<1x32x800x1280xf16, {order = #NHWC}>
// CHECK:    [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x4x1600x2560xf16, {order = #NHWC}>
// CHECK:    [[LOOP_RESULT:%.+]] = scf.for
// CHECK-SAME:              [[IDX:%arg[0-9]]] = [[C0]] to [[C2560]] step [[C40]]
// CHECK-SAME:              iter_args([[LOOP_OUT_H:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x4x1600x2560xf16, {order = #NHWC}>) {
// CHECK:        [[TILE_IDX:%.+]] = affine.max #map([[IDX]])
// CHECK:        [[TILE_IDX_OVERLAP:%.+]] = affine.max #map1([[IDX]])
// CHECK:        [[LOW_PADDING:%.+]] = affine.min #map2()[[[TILE_IDX_OVERLAP]]]
// CHECK:        [[HIGH_CANIDATE_PADDING:%.+]] = affine.max #map3([[TILE_IDX]])
// CHECK:        [[HIGH_PADDING:%.+]] = affine.min #map2()[[[HIGH_CANIDATE_PADDING]]]
// CHECK:        [[SLICE_WIDTH:%.+]] = affine.apply #map4([[LOW_PADDING]], [[HIGH_PADDING]])
// CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[SHAPECAST0]][0, 0, 0, [[TILE_IDX]]] [1, 32, 800, [[SLICE_WIDTH]]] [1, 1, 1, 1] : tensor<1x32x800x1280xf16, {order = #NHWC}> to tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[PADDING:%.+]] = tensor.pad [[SLICE]] low[0, 0, 1, [[LOW_PADDING]]] high[0, 0, 1, [[HIGH_PADDING]]] {
// CHECK:        ^bb0({{%[^:]+}}: index, {{%[^:]+}}: index, {{%[^:]+}}: index, {{%[^:]+}}: index):
// CHECK:           tensor.yield [[CST]] : f16
// CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[PADDING]], [[CST0]]) rawFilterShape [16, 32, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
// CHECK-SAME:      strides = [1, 1]} : tensor<1x32x802x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 802, 1282]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x32x3x3xf16, {order = #NHWC}> -> tensor<1x16x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[CAST:%.+]] = tensor.cast [[CONV0]] : tensor<1x16x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x800x20xf16, {order = #NHWC}>
// CHECK:        [[DTS:%.+]] = VPU.DepthToSpace([[CAST]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x800x20xf16, {order = #NHWC}> -> tensor<1x4x1600x40xf16, {order = #NHWC}>
// CHECK:        [[INSERT_SLICE:%.+]] = tensor.insert_slice [[DTS]] into [[LOOP_OUT_H]][0, 0, 0, [[IDX]]] [1, 4, 1600, 40] [1, 1, 1, 1] : tensor<1x4x1600x40xf16, {order = #NHWC}> into tensor<1x4x1600x2560xf16, {order = #NHWC}>
// CHECK:        scf.yield [[INSERT_SLICE]] : tensor<1x4x1600x2560xf16, {order = #NHWC}>
// CHECK:    {{.+}} = VPU.ShapeCast {shape = [1, 16, 1600, 640]} inputs([[LOOP_RESULT]] : tensor<1x4x1600x2560xf16, {order = #NHWC}>) -> tensor<1x16x1600x640xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @MergeConvertGelu
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertGelu(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Gelu(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[GELU:%.+]] = VPU.Gelu([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[GELU]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @MergeConvertDequantize
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertDequantize(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = !qElemType, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256x!qElemType>
    %1 = VPU.Dequantize(%0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256x!qElemType> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = !qElemType}
// CHECK:            [[DQ:%.+]] = VPU.Dequantize([[CONVERT]]) {dstElemType = f16}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[DQ]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @MergeConvertMVN1Normalize
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
// CHECK-SAME:  [[MEANVAR:%.+]]: tensor<1x16x1x2xf16>
func.func @MergeConvertMVN1Normalize(%arg0: tensor<1x16x128x256xf32>, %arg1: tensor<1x16x1x2xf16>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.MVN1Normalize(%0, %arg1) {across_channels = false, normalize_variance = true, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16>, tensor<1x16x1x2xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[MVN:%.+]] = VPU.MVN1Normalize([[CONVERT]], [[MEANVAR]]) {across_channels = false, normalize_variance = true}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[MVN]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @MergeConvertSoftMax
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x256x140xf32>)
func.func @MergeConvertSoftMax(%arg0: tensor<1x16x256x140xf32>) -> tensor<1x16x256x140xf16> {
    %0 = VPU.Convert(%arg0) {
        dstElemType = f16,
        tilingStrategy = [1, 1, 6, 1]
    } : tensor<1x16x256x140xf32> -> tensor<1x16x256x140xf16>

    %1 = VPU.SoftMax(%0) {
        axisInd = 3 : i64,
        tilingStrategy = [1, 1, 6, 1]
    } : tensor<1x16x256x140xf16> -> tensor<1x16x256x140xf16>

    return %1 : tensor<1x16x256x140xf16>

    // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x256x140xf16>
    // CHECK: [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]])
    // CHECK-SAME:     -> (tensor<1x16x256x140xf16>)
    // CHECK:     {{%.+}} = affine.min
    // CHECK:     {{%.+}} = affine.min
    // CHECK:     {{%.+}} = affine.min
    // CHECK:     [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
    // CHECK:     [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
    // CHECK:     [[SOFTMAX:%.+]] = VPU.SoftMax([[CONVERT]]) {axisInd = 3 : i64}
    // CHECK:     [[INSERT:%.+]] = tensor.insert_slice [[SOFTMAX]] into [[OUT]]
    // CHECK:     scf.yield [[INSERT]] : tensor<1x16x256x140xf16>
    // CHECK: return [[SCF]] : tensor<1x16x256x140xf16>
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// SoftMax(axisInd=3) tiles on H (axis 3 excluded from tiling); Convert tiles on W.
// VF fuses them using the SoftMax H-tiling — both ops run inside the same scf.for.
// CHECK-LABEL: @SoftMaxConvertDiffTilingAxes
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x256x140xf16>)
func.func @SoftMaxConvertDiffTilingAxes(%arg0: tensor<1x16x256x140xf16>) -> tensor<1x16x256x140xf32> {
    %0 = VPU.SoftMax(%arg0) {
        axisInd = 3 : i64,
        tilingStrategy = [1, 1, 6, 1]
    } : tensor<1x16x256x140xf16> -> tensor<1x16x256x140xf16>

    %1 = VPU.Convert(%0) {
        dstElemType = f32,
        tilingStrategy = [1, 1, 6, 1]
    } : tensor<1x16x256x140xf16> -> tensor<1x16x256x140xf32>

    return %1 : tensor<1x16x256x140xf32>

    // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x256x140xf32>
    // CHECK: [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]])
    // CHECK-SAME:     -> (tensor<1x16x256x140xf32>)
    // CHECK:     {{%.+}} = affine.min
    // CHECK:     {{%.+}} = affine.min
    // CHECK:     {{%.+}} = affine.min
    // CHECK:     [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
    // CHECK:     [[SOFTMAX:%.+]] = VPU.SoftMax([[SLICE]]) {axisInd = 3 : i64}
    // CHECK:     [[CONVERT:%.+]] = VPU.Convert([[SOFTMAX]]) {dstElemType = f32}
    // CHECK:     [[INSERT:%.+]] = tensor.insert_slice [[CONVERT]] into [[OUT]]
    // CHECK:     scf.yield [[INSERT]] : tensor<1x16x256x140xf32>
    // CHECK: return [[SCF]] : tensor<1x16x256x140xf32>
}

// -----

// CHECK-LABEL: @MergeConvertAbs
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertAbs(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Abs(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[ABS:%.+]] = VPU.Abs([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ABS]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertSin
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertSin(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Sin(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[SIN:%.+]] = VPU.Sin([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SIN]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertSqrt
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertSqrt(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Sqrt(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[SQRT:%.+]] = VPU.Sqrt([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SQRT]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertTanh
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertTanh(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Tanh(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[TANH:%.+]] = VPU.Tanh([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[TANH]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertMish
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertMish(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Mish(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[MISH:%.+]] = VPU.Mish([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[MISH]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertReLU
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertReLU(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.ReLU(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[RELU:%.+]] = VPU.ReLU([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[RELU]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertSigmoid
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertSigmoid(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Sigmoid(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[SIG:%.+]] = VPU.Sigmoid([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SIG]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertSoftPlus
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertSoftPlus(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.SoftPlus(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[SP:%.+]] = VPU.SoftPlus([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SP]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertSwish
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertSwish(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Swish(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[SWISH:%.+]] = VPU.Swish([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SWISH]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertAcos
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertAcos(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Acos(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[ACOS:%.+]] = VPU.Acos([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ACOS]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertAcosh
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertAcosh(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Acosh(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[ACOSH:%.+]] = VPU.Acosh([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ACOSH]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertAsin
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertAsin(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Asin(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[ASIN:%.+]] = VPU.Asin([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ASIN]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertAsinh
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertAsinh(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Asinh(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[ASINH:%.+]] = VPU.Asinh([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ASINH]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertAtan
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertAtan(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Atan(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[ATAN:%.+]] = VPU.Atan([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ATAN]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertAtanh
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertAtanh(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Atanh(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[ATANH:%.+]] = VPU.Atanh([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ATANH]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertCosh
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertCosh(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Cosh(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[COSH:%.+]] = VPU.Cosh([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[COSH]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertErf
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertErf(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Erf(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[ERF:%.+]] = VPU.Erf([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ERF]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertSinh
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertSinh(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Sinh(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[SINH:%.+]] = VPU.Sinh([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SINH]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertTan
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertTan(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Tan(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[TAN:%.+]] = VPU.Tan([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[TAN]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertElu
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertElu(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Elu(%0) {tilingStrategy = [1, 1, 8, 2], x = 1.000000e+00 : f64} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[ELU:%.+]] = VPU.Elu([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ELU]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertHardSigmoid
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertHardSigmoid(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.HardSigmoid(%0) {alpha_value = 2.000000e-01 : f64, beta_value = 5.000000e-01 : f64, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[HARDSIGMOID:%.+]] = VPU.HardSigmoid([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[HARDSIGMOID]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertHSwish
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertHSwish(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.HSwish(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[HSWISH:%.+]] = VPU.HSwish([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[HSWISH]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertLeakyRelu
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertLeakyRelu(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.LeakyRelu(%0) {negative_slope = 1.000000e-02 : f64, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[LEAKYRELU:%.+]] = VPU.LeakyRelu([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[LEAKYRELU]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertSelu
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertSelu(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Selu(%0) {alpha_value = 1.6732000000000001 : f64, lambda_value = 1.0507000000000001 : f64, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[SELU:%.+]] = VPU.Selu([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SELU]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1997824 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!outputStaticType = tensor<1x32x540x960x!quant.uniform<u8:f16, 0.030033121856988646:110>, {order = #NHWC}>

// CHECK: #[[$OUT_OFFSET_AND_SIZE_MAP:.+]] = affine_map<(d0) -> ((d0 floordiv 25) * 25, (d0 floordiv 25) * 24 + 24)>
// CHECK: #[[$TILE_INDEX_CAP_AND_REMAINDER_MAP:.+]] = affine_map<(d0) -> (d0 floordiv 25, 24)>
// CHECK: #[[$OUT_SIZE_BY_CAPPED_TILE_INDEX_MAP:.+]] = affine_map<(d0) -> (-d0 + 48, 25)>
// CHECK: #[[$TO_SLICE_OFFSET_MAP:.+]] = affine_map<(d0) -> (0, d0 * 2 - 1)>
// CHECK: #[[$TO_PAD_CANDIDATE_MAP:.+]] = affine_map<(d0) -> (d0 * -2 + 1, 0)>
// CHECK: #[[$PAD_CLAMP_MAP:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$SLICE_SIZE_MAP:.+]] = affine_map<(d0, d1) -> (d0 * 2 - d1 + 1)>

// CHECK-LABEL: @MergeSliceStaticInput
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x16x1080x1920xf16, {order = #NHWC}>
func.func @MergeSliceStaticInput(
    %arg0: tensor<1x16x1080x1920xf16, {order = #NHWC}>
  ) -> !outputStaticType {

  %weights_dw = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    = dense<1.0> : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

  %weights_cc = const.Declare tensor<32x1x1x32x!quant.uniform<u8:f16, 0.034925088695451328:128>, {order = #NHWC}>
    = dense<1> : tensor<32x3x3x3xsi8>,
      [#const.CastElemType<f16>,
       #const.CastElemType<!quant.uniform<i8:f16, 0.034925088695451328>>,
       #const.ConvertElemType<!quant.uniform<u8:f16, 0.034925088695451328:128>>,
       #const.Reorder<#NHWC>,
       #const.Reshape<[32, 1, 1, 27]>,
       #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 5]>]
  %bias_cc = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

  %depth = VPU.NCE.DepthConvolution(%arg0, %weights_dw) rawFilterShape [16, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 507.0 : f64>,

    strides = [1, 1],
    tilingStrategy = [1, 1, 2, 1]
  } -> tensor<1x16x1080x1920x!quant.uniform<u8:f16, 0.0019697112195632038>, {order = #NHWC}>

  %slice = VPU.Slice %depth [0, 0, 0, 0] [1, 4, 1080, 1920]
    : tensor<1x16x1080x1920x!quant.uniform<u8:f16, 0.0019697112195632038>, {order = #NHWC}>
    to tensor<1x4x1080x1920x!quant.uniform<u8:f16, 0.0019697112195632038>, {order = #NHWC}>

  %compress = VPU.NCE.CompressConvolution(%slice, %weights_cc, %bias_cc) rawFilterShape [32, 3, 3, 3] {
    cm_sp_pattern = 7 : i64,
    multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.0 : f64>,

    strides = [2, 2],
    tilingStrategy = [1, 1, 7, 3]
  } -> !outputStaticType

  return %compress : !outputStaticType

    // CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0 : i8
    // CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 25 : index
    // CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 960 : index
    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x540x960x!qElemType, {order = #NHWC}>
    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    // CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x540x960x!qElemType, {order = #NHWC}>) {

    // CHECK:        [[OUT_OFFSET:%.+]] = affine.min #[[$OUT_OFFSET_AND_SIZE_MAP]]([[LOOP_ITER]])
    // CHECK:        [[OUT_TILE_INDEX_CAP:%.+]] = affine.min #[[$TILE_INDEX_CAP_AND_REMAINDER_MAP]]([[LOOP_ITER]])
    // CHECK:        [[OUT_SIZE:%.+]] = affine.min #[[$OUT_SIZE_BY_CAPPED_TILE_INDEX_MAP]]([[OUT_TILE_INDEX_CAP]])

    // CHECK:        [[SLICE_OFFSET:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[OUT_OFFSET]])
    // CHECK:        [[TEMP_VALUE0:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[OUT_OFFSET]])
    // CHECK:        [[PAD_LOW:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE0]]]
    // CHECK:        [[SIZE_W:%.+]] = affine.apply #[[$SLICE_SIZE_MAP]]([[OUT_SIZE]], [[PAD_LOW]])
    // CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[SLICE_OFFSET]]] [1, 16, 1080, [[SIZE_W]]] [1, 1, 1, 1]
    // CHECK:        [[DEPTH:%.+]] = VPU.NCE.DepthConvolution([[SLICE]]
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[VPU_SLICE:%.+]] = VPU.Slice [[DEPTH]] [0, 0, 0, 0] [1, 4, 1080, -9223372036854775808]
    // CHECK:        [[CAST_PAD:%.+]] = builtin.unrealized_conversion_cast [[PAD_VALUE]] : i8 to !qElemType{{.*}}
    // CHECK:        [[PADDED:%.+]] = tensor.pad [[VPU_SLICE]] low[0, 0, 1, [[PAD_LOW]]] high[0, 0, 0, 0] {
    // CHECK:          tensor.yield [[CAST_PAD]]
    // CHECK:        [[COMPRESS:%.+]] = VPU.NCE.CompressConvolution([[PADDED]]
    // CHECK-SAME:       cm_sp_pattern = 7
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[INSERT:%.+]] = tensor.insert_slice [[COMPRESS]] into [[LOOP_OUT]][0, 0, 0, [[OUT_OFFSET]]] [1, 32, 540, [[OUT_SIZE]]]
    // CHECK:        scf.yield [[INSERT]] : tensor<1x32x540x960x!qElemType, {order = #NHWC}>
    // CHECK:    return [[LOOP]] : tensor<1x32x540x960x!qElemType, {order = #NHWC}>
}

// -----
config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DoubleLoopCastTest(%arg0: tensor<1x1600x2560x4xf32>) -> tensor<1x1600x2560x4xf32> {
  %cst = const.Declare tensor<32x1x1x144xf16, {order = #NHWC}> = dense<1.0> : tensor<32x1x1x144xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_11 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
  %cst_12 = const.Declare tensor<16x96x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x96x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_13 = const.Declare tensor<96x32x3x3x!quant.uniform<u8:f16, 5.7474947443195417E-4:128>, {order = #NHWC}> = dense<1> : tensor<96x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 5.7474947443195417E-4:128>>, #const.Reorder<#NHWC>]
  %cst_14 = const.Declare tensor<32x32x3x3x!quant.uniform<u8:f16, 5.4446090670192944E-4:128>, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 5.4446090670192944E-4:128>>, #const.Reorder<#NHWC>]
  %cst_15 = const.Declare tensor<32x32x3x3x!quant.uniform<u8:f16, 5.4758985837300616E-4:128>, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 5.4758985837300616E-4:128>>, #const.Reorder<#NHWC>]
  %cst_16 = const.Declare tensor<32x32x3x3x!quant.uniform<u8:f16, 5.8887451887130733E-4:128>, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 5.8887451887130733E-4:128>>, #const.Reorder<#NHWC>]
  %cst_17 = const.Declare tensor<32x32x3x3x!quant.uniform<u8:f16, 5.3848708961524213E-4:128>, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 5.3848708961524213E-4:128>>, #const.Reorder<#NHWC>]
  %cst_18 = const.Declare tensor<32x32x3x3x!quant.uniform<u8:f16, 5.3834269444147744E-4:128>, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 5.3834269444147744E-4:128>>, #const.Reorder<#NHWC>]
  %cst_19 = const.Declare tensor<32x32x3x3x!quant.uniform<u8:f16, 5.9490913853925821E-4:128>, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 5.9490913853925821E-4:128>>, #const.Reorder<#NHWC>]
  %cst_20 = const.Declare tensor<32x32x3x3x!quant.uniform<u8:f16, 5.792334675788879E-4:128>, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 5.792334675788879E-4:128>>, #const.Reorder<#NHWC>]
  %cst_21 = const.Declare tensor<32x32x3x3x!quant.uniform<u8:f16, 5.42596274731206E-4:128>, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 5.42596274731206E-4:128>>, #const.Reorder<#NHWC>]
  %cst_22 = const.Declare tensor<32x32x3x3x!quant.uniform<u8:f16, 6.1786350081948676E-4:128>, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 6.1786350081948676E-4:128>>, #const.Reorder<#NHWC>]
  %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x1600x2560x4xf32> -> tensor<1x4x1600x2560xf32, {order = #NHWC}>
  %1 = VPU.Convert(%0) {dstElemType = f16, tilingStrategy = [1, 1, 1, 68]} : tensor<1x4x1600x2560xf32, {order = #NHWC}> -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
  %output = VPU.NCE.Convolution(%1, %cst) rawFilterShape [32, 4, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.170000e+02 : f64, clamp_high = 1.380000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.170000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [2, 2], tilingStrategy = [1, 1, 1, 86]} : tensor<1x4x1600x2560xf16, {order = #NHWC}>, tensor<32x1x1x144xf16, {order = #NHWC}> -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0034980668741113998:117>, {order = #NHWC}>
  %output_23 = VPU.NCE.Convolution(%output, %cst_22) rawFilterShape [32, 32, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.170000e+02 : f64, clamp_high = 1.380000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.170000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 1, 92]} : tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0034980668741113998:117>, {order = #NHWC}>, tensor<32x32x3x3x!quant.uniform<u8:f16, 6.1786350081948676E-4:128>, {order = #NHWC}> -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0034980668741113998:117>, {order = #NHWC}>
  %output_24 = VPU.NCE.Eltwise(%output, %output_23) {is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.380000e+02 : f64, clamp_high = 1.170000e+02 : f64, scale = 3.5136938095092773E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.380000e+02 : f64, in1_mult = [2.934300e+04], in2_mult = [2.934300e+04]>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 46]} -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0033925405904358507:138>, {order = #NHWC}>
  %output_25 = VPU.NCE.Convolution(%output_24, %cst_21) rawFilterShape [32, 32, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.380000e+02 : f64, clamp_high = 1.170000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.380000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 1, 92]} : tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0033925405904358507:138>, {order = #NHWC}>, tensor<32x32x3x3x!quant.uniform<u8:f16, 5.42596274731206E-4:128>, {order = #NHWC}> -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0033925405904358507:138>, {order = #NHWC}>
  %output_26 = VPU.NCE.Eltwise(%output_24, %output_25) {is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.510000e+02 : f64, clamp_high = 1.040000e+02 : f64, scale = 2.7396716177463531E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.510000e+02 : f64, in1_mult = [2.845800e+04], in2_mult = [2.845800e+04]>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 46]} -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.004351120485978968:151>, {order = #NHWC}>
  %output_27 = VPU.NCE.Convolution(%output_26, %cst_20) rawFilterShape [32, 32, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.170000e+02 : f64, clamp_high = 1.380000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.170000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 1, 92]} : tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.004351120485978968:151>, {order = #NHWC}>, tensor<32x32x3x3x!quant.uniform<u8:f16, 5.792334675788879E-4:128>, {order = #NHWC}> -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0034980668741113998:117>, {order = #NHWC}>
  %output_28 = VPU.NCE.Eltwise(%output, %output_27) {is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.050000e+02 : f64, clamp_high = 1.500000e+02 : f64, scale = 3.0257739126682281E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.050000e+02 : f64, in1_mult = [2.934300e+04], in2_mult = [2.934300e+04]>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 46]} -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0039397156706043315:105>, {order = #NHWC}>
  %output_29 = VPU.NCE.Convolution(%output_28, %cst_19) rawFilterShape [32, 32, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.050000e+02 : f64, clamp_high = 1.500000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.050000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 1, 92]} : tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0039397156706043315:105>, {order = #NHWC}>, tensor<32x32x3x3x!quant.uniform<u8:f16, 5.9490913853925821E-4:128>, {order = #NHWC}> -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0039397156706043315:105>, {order = #NHWC}>
  %output_30 = VPU.NCE.Eltwise(%output_28, %output_29) {is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.380000e+02 : f64, clamp_high = 1.170000e+02 : f64, scale = 6.9852918386459351E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.380000e+02 : f64, in1_mult = [1.652400e+04], in2_mult = [1.652400e+04]>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 46]} -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0034130302130007278:138>, {order = #NHWC}>
  %output_31 = VPU.NCE.Convolution(%output_30, %cst_18) rawFilterShape [32, 32, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.380000e+02 : f64, clamp_high = 1.170000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.380000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 1, 92]} : tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0034130302130007278:138>, {order = #NHWC}>, tensor<32x32x3x3x!quant.uniform<u8:f16, 5.3834269444147744E-4:128>, {order = #NHWC}> -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0034130302130007278:138>, {order = #NHWC}>
  %output_32 = VPU.NCE.Eltwise(%output_30, %output_31) {is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.090000e+02 : f64, clamp_high = 1.460000e+02 : f64, scale = 2.3410655558109283E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.090000e+02 : f64, in1_mult = [2.863000e+04], in2_mult = [2.863000e+04]>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 46]} -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0050920921213486615:109>, {order = #NHWC}>
  %output_33 = VPU.NCE.Convolution(%output_32, %cst_17) rawFilterShape [32, 32, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.050000e+02 : f64, clamp_high = 1.500000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.050000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 1, 92]} : tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0050920921213486615:109>, {order = #NHWC}>, tensor<32x32x3x3x!quant.uniform<u8:f16, 5.3848708961524213E-4:128>, {order = #NHWC}> -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0039397156706043315:105>, {order = #NHWC}>
  %output_34 = VPU.NCE.Eltwise(%output_28, %output_33) {is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.090000e+02 : f64, clamp_high = 1.460000e+02 : f64, scale = 4.9212947487831116E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.090000e+02 : f64, in1_mult = [1.652400e+04], in2_mult = [1.652400e+04]>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 46]} -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0048445047116747091:109>, {order = #NHWC}>
  %output_35 = VPU.NCE.Convolution(%output_34, %cst_16) rawFilterShape [32, 32, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.090000e+02 : f64, clamp_high = 1.460000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.090000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 1, 92]} : tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0048445047116747091:109>, {order = #NHWC}>, tensor<32x32x3x3x!quant.uniform<u8:f16, 5.8887451887130733E-4:128>, {order = #NHWC}> -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0048445047116747091:109>, {order = #NHWC}>
  %output_36 = VPU.NCE.Eltwise(%output_34, %output_35) {is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.420000e+02 : f64, clamp_high = 1.130000e+02 : f64, scale = 5.3092837333679199E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.420000e+02 : f64, in1_mult = [2.031900e+04], in2_mult = [2.031900e+04]>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 46]} -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.004490463640175614:142>, {order = #NHWC}>
  %output_37 = VPU.NCE.Convolution(%output_36, %cst_15) rawFilterShape [32, 32, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.420000e+02 : f64, clamp_high = 1.130000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.420000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 1, 92]} : tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.004490463640175614:142>, {order = #NHWC}>, tensor<32x32x3x3x!quant.uniform<u8:f16, 5.4758985837300616E-4:128>, {order = #NHWC}> -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.004490463640175614:142>, {order = #NHWC}>
  %output_38 = VPU.NCE.Eltwise(%output_36, %output_37) {is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.170000e+02 : f64, clamp_high = 1.380000e+02 : f64, scale = 3.9102509617805481E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.170000e+02 : f64, in1_mult = [1.883400e+04], in2_mult = [1.883400e+04]>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 46]} -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0060971145536385333:117>, {order = #NHWC}>
  %output_39 = VPU.NCE.Convolution(%output_38, %cst_14) rawFilterShape [32, 32, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.090000e+02 : f64, clamp_high = 1.460000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.090000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 1, 92]} : tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0060971145536385333:117>, {order = #NHWC}>, tensor<32x32x3x3x!quant.uniform<u8:f16, 5.4446090670192944E-4:128>, {order = #NHWC}> -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0048445047116747091:109>, {order = #NHWC}>
  %output_40 = VPU.NCE.Eltwise(%output_34, %output_39) {is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.570000e+02 : f64, clamp_high = 9.800000e+01 : f64, scale = 3.0467286705970764E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.570000e+02 : f64, in1_mult = [2.031900e+04], in2_mult = [2.031900e+04]>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, tilingStrategy = [1, 1, 1, 46]} -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0078253755382463042:157>, {order = #NHWC}>
  %output_41 = VPU.NCE.Convolution(%output_40, %cst_13) rawFilterShape [96, 32, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 1, 256]} : tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0078253755382463042:157>, {order = #NHWC}>, tensor<96x32x3x3x!quant.uniform<u8:f16, 5.7474947443195417E-4:128>, {order = #NHWC}> -> tensor<1x96x800x1280xf16, {order = #NHWC}>
  %output_42 = VPU.NCE.Convolution(%output_41, %cst_12) rawFilterShape [16, 96, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tilingStrategy = [1, 1, 1, 640]} : tensor<1x96x800x1280xf16, {order = #NHWC}>, tensor<16x96x1x1xf16, {order = #NHWC}> -> tensor<1x16x800x1280xf16, {order = #NHWC}>
  %2 = VPU.DepthToSpace(%output_42) {block_size = 2 : i64, dstElemType = f32, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, tilingStrategy = [1, 1, 80, 1]} : tensor<1x16x800x1280xf16, {order = #NHWC}> -> tensor<1x4x1600x2560xf32, {order = #NHWC}>
  %3 = VPU.PermuteCast(%2) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x4x1600x2560xf32, {order = #NHWC}> -> tensor<1x1600x2560x4xf32>
  return %3 : tensor<1x1600x2560x4xf32>
}


// CHECK-LABEL: @DoubleLoopCastTest
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1600x2560x4xf32>

// CHECK:       [[LAST:%.+]] = scf.for {{.*}} -> (tensor<1x4x1600x2560xf32, {order = #NHWC}>)
// CHECK:         scf.for
// CHECK:           VPU.NCE.Eltwise
// CHECK:           [[CONV1:%.+]] = VPU.NCE.Convolution{{.*}}-> tensor<1x96x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           tensor.cast [[CONV1]] : tensor<1x96x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x96x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 16, 1280]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:           [[CONV2:%.+]] = VPU.NCE.Convolution
// CHECK-SAME:          -> tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 1280]>{{.*}}order = #NHWC}>
// CHECK:           [[D2S:%.+]] = VPU.DepthToSpace([[CONV2]])
// CHECK-SAME:          -> tensor<1x4x32x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 32, 2560]>{{.*}}order = #NHWC}>
// CHECK:           tensor.insert_slice
// CHECK:       [[OUT:%.+]] = VPU.PermuteCast([[LAST]])
// CHECK:       return [[OUT]]
