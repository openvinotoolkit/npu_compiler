//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --scf-vertical-fusion="vf-merge-configuration=GREEDY" --resolve-shaped-type-result-dims --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

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
    // CHECK:        [[SIZE_H:%.+]] = affine.apply #[[$SLICE_SIZE_BY_OUT_AND_PAD_MAP]]([[PAD_LOW0]], [[PAD_HIGH0]], [[PAD_LOW1]], [[PAD_HIGH1]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[CAST_INPUT]][0, 0, 0, [[SLICE_OFFSET]]] [1, 32, 540, [[SIZE_H]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

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
// CHECK: #[[$OUT_OFFSET_AND_SIZE_MAP:.+]] = affine_map<(d0) -> ((d0 floordiv 44) * 44, (d0 floordiv 44) * 43 + 14)>
// CHECK: #[[$TILE_INDEX_CAP_AND_REMAINDER_MAP:.+]] = affine_map<(d0) -> (d0 floordiv 44, 14)>
// CHECK: #[[$OUT_SIZE_BY_CAPPED_TILE_INDEX_MAP:.+]] = affine_map<(d0) -> (-d0 + 57, 44)>
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
    // CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 44 : index
    // CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 960 : index
    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK: [[CAST_INPUT:%.+]] = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs([[INPUT]] : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    // CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>) {

    // CHECK:        [[OUTPUT_OFFSET:%.+]] = affine.min #[[$OUT_OFFSET_AND_SIZE_MAP]]([[LOOP_ITER]])
    // CHECK:        [[OUTPUT_TILE_INDEX_CAP:%.+]] = affine.min #[[$TILE_INDEX_CAP_AND_REMAINDER_MAP]]([[LOOP_ITER]])
    // CHECK:        [[OUTPUT_SIZE:%.+]] = affine.min #[[$OUT_SIZE_BY_CAPPED_TILE_INDEX_MAP]]([[OUTPUT_TILE_INDEX_CAP]])

    // CHECK:        [[TEMP_VALUE0:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[OUTPUT_OFFSET]])
    // CHECK:        [[TEMP_VALUE1:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[OUTPUT_OFFSET]])
    // CHECK:        [[PAD_LOW5:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE1]]]
    // CHECK:        [[TEMP_VALUE2:%.+]] = affine.max #[[$PAD_HIGH_STAGE5_CANDIDATE_MAP]]([[OUTPUT_SIZE]], [[TEMP_VALUE0]])
    // CHECK:        [[PAD_HIGH5:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE2]]]
    // CHECK:        [[TEMP_VALUE3:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE0]])
    // CHECK:        [[TEMP_VALUE4:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE0]])
    // CHECK:        [[PAD_LOW4:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE4]]]
    // CHECK:        [[TEMP_VALUE5:%.+]] = affine.max #[[$PAD_HIGH_STAGE4_CANDIDATE_MAP]]([[TEMP_VALUE3]], [[OUTPUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH4:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE5]]]
    // CHECK:        [[TEMP_VALUE6:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE3]])
    // CHECK:        [[TEMP_VALUE7:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE3]])
    // CHECK:        [[PAD_LOW3:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE7]]]
    // CHECK:        [[TEMP_VALUE8:%.+]] = affine.max #[[$PAD_HIGH_STAGE3_CANDIDATE_MAP]]([[TEMP_VALUE6]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUTPUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH3:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE8]]]
    // CHECK:        [[TEMP_VALUE9:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE6]])
    // CHECK:        [[TEMP_VALUE10:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE6]])
    // CHECK:        [[PAD_LOW2:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE10]]]
    // CHECK:        [[TEMP_VALUE11:%.+]] = affine.max #[[$PAD_HIGH_STAGE2_CANDIDATE_MAP]]([[TEMP_VALUE9]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUTPUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH2:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE11]]]
    // CHECK:        [[TEMP_VALUE12:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE9]])
    // CHECK:        [[TEMP_VALUE13:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE9]])
    // CHECK:        [[PAD_LOW1:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE13]]]
    // CHECK:        [[TEMP_VALUE14:%.+]] = affine.max #[[$PAD_HIGH_STAGE1_CANDIDATE_MAP]]([[TEMP_VALUE12]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUTPUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH1:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE14]]]
    // CHECK:        [[SLICE_OFFSET:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE12]])
    // CHECK:        [[TEMP_VALUE15:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE12]])
    // CHECK:        [[PAD_LOW0:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE15]]]
    // CHECK:        [[SLICE_SIZE:%.+]] = affine.max #[[$PAD_HIGH_STAGE0_CANDIDATE_MAP]]([[SLICE_OFFSET]], [[PAD_LOW1]], [[PAD_HIGH1]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUTPUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[PAD_HIGH0:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[SLICE_SIZE]]]
    // CHECK:        [[SLICE_SIZE:%.+]] = affine.apply #[[$SLICE_SIZE_BY_OUT_AND_PAD_MAP]]([[PAD_LOW0]], [[PAD_HIGH0]], [[PAD_LOW1]], [[PAD_HIGH1]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUTPUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    // CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[CAST_INPUT]][0, 0, 0, [[SLICE_OFFSET]]] [1, 32, 540, [[SLICE_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

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
    // CHECK:        [[INSERT:%.+]] = tensor.insert_slice [[DWCONV1]] into [[LOOP_OUT]][0, 0, 0, [[OUTPUT_OFFSET]]] [1, 32, 540, [[OUTPUT_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
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
// CHECK: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 240)>

// CHECK-LABEL: @MergeDynamicEltwise
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
func.func @MergeDynamicEltwise(
         %arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
     %0 = VPU.NCE.Eltwise(%arg0, %arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
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

     %1 = VPU.NCE.Eltwise(%0, %0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
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

     return %1 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: [[LOOP_STEP:%.+]] = arith.constant 240 : index
    // CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK: [[DIM_INDEX:%.+]] = arith.constant 3 : index

    // CHECK: [[DIM:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM]]) : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    // CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                [[SLICE_SIZE:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER]])[[[LOOP_END]]]
    // CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[SLICE_SIZE]]] [1, 1, 1, 1]
    // CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 240]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[SLICE]], [[SLICE]])
    // CHECK:                [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[ELTWISE0]], [[ELTWISE0]])

    // CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE1]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[SLICE_SIZE]]] [1, 1, 1, 1]
    // CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 240]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                scf.yield [[INSERT]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:  return [[LOOP]] :  tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$CHAIN0_OFFSET_AND_SIZE_MAP:.+]] = affine_map<(d0) -> ((d0 floordiv 13) * 13, (d0 floordiv 13) * 12 + 72)>
// CHECK: #[[$CHAIN0_TILE_INDEX_CAP_AND_REMAINDER_MAP:.+]] = affine_map<(d0) -> (d0 floordiv 13, 72)>
// CHECK: #[[$CHAIN0_SIZE_BY_CAPPED_TILE_INDEX_MAP:.+]] = affine_map<(d0) -> (-d0 + 84, 13)>
// CHECK: #[[$TO_SLICE_OFFSET_MAP:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$TO_PAD_CANDIDATE_MAP:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$PAD_CLAMP_MAP:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$PAD_HIGH_CHAIN0_CANDIDATE_MAP:.+]] = affine_map<(d0, d1) -> (0, d0 + d1 - 958)>
// CHECK: #[[$CHAIN0_SLICE_SIZE_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
// CHECK: #[[$CHAIN1_OFFSET_AND_SIZE_MAP:.+]] = affine_map<(d0) -> ((d0 floordiv 27) * 27, (d0 floordiv 27) * 26 + 24)>
// CHECK: #[[$CHAIN1_TILE_INDEX_CAP_AND_REMAINDER_MAP:.+]] = affine_map<(d0) -> (d0 floordiv 27, 24)>
// CHECK: #[[$CHAIN1_SIZE_BY_CAPPED_TILE_INDEX_MAP:.+]] = affine_map<(d0) -> (-d0 + 50, 27)>
// CHECK: #[[$PAD_HIGH_CHAIN1_STAGE1_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (0, d0 + d1 - d2 - d3 - 956)>
// CHECK: #[[$CHAIN1_SLICE_SIZE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (-d0 - d1 + d2 - d3 - d4 + 4)>

// CHECK-LABEL: @MergeVF2Chains
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x540x960xf16, {order = #NHWC}>)
func.func @MergeVF2Chains(%arg0: tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
 {
    %cst = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_2) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %2 = VPU.Sign(%1) : tensor<1x32x540x960xf16, {order = #NHWC}>  -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_0) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %4 = VPU.NCE.Convolution(%3, %cst_1) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %4: tensor<1x32x540x960xf16, {order = #NHWC}>

    // CHECK-DAG: [[LOOP_STEP1:%.+]] = arith.constant 27 : index
    // CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK-DAG: [[LOOP_STEP0:%.+]] = arith.constant 13 : index
    // CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 960 : index
    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK: [[LOOP_OUTPUT0:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK: [[LOOP0:%.+]] = scf.for
    // CHECK-SAME:           [[LOOP_ITER0:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP0]]
    // CHECK-SAME:           iter_args([[LOOP_OUT0:%arg[0-9]]] = [[LOOP_OUTPUT0]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>)

    // CHECK:                [[INSERT_OFFSET0:%.+]] = affine.min #[[$CHAIN0_OFFSET_AND_SIZE_MAP]]([[LOOP_ITER0]])
    // CHECK:                [[INSERT_TILE_INDEX0_CAP:%.+]] = affine.min #[[$CHAIN0_TILE_INDEX_CAP_AND_REMAINDER_MAP]]([[LOOP_ITER0]])
    // CHECK:                [[INSERT_SIZE0:%.+]] = affine.min #[[$CHAIN0_SIZE_BY_CAPPED_TILE_INDEX_MAP]]([[INSERT_TILE_INDEX0_CAP]])


    // CHECK:                [[SLICE_OFFSET0:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[INSERT_OFFSET0]])
    // CHECK:                [[TEMP_VALUE0:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[INSERT_OFFSET0]])
    // CHECK:                [[PAD_LOW0:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE0]]]
    // CHECK:                [[TEMP_VALUE1:%.+]] = affine.max #[[$PAD_HIGH_CHAIN0_CANDIDATE_MAP]]([[INSERT_SIZE0]], [[SLICE_OFFSET0]])
    // CHECK:                [[PAD_HIGH0:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE1]]]
    // CHECK:                [[SLICE_SIZE0:%.+]] = affine.apply #[[$CHAIN0_SLICE_SIZE_MAP]]([[INSERT_SIZE0]], [[PAD_LOW0]], [[PAD_HIGH0]])

    // CHECK:                [[SLICE0:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[SLICE_OFFSET0]]] [1, 32, 540, [[SLICE_SIZE0]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[SLICE0]]

    // CHECK:                [[PAD0:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, 1, [[PAD_LOW0]]] high[0, 0, 1, [[PAD_HIGH0]]] {
    // CHECK:                tensor.yield [[PAD_VALUE]] : f16
    // CHECK:                tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]

    // CHECK:                [[INSERT0:%.+]] = tensor.insert_slice [[CONV0]] into [[LOOP_OUT0]][0, 0, 0, [[INSERT_OFFSET0]]] [1, 32, 540, [[INSERT_SIZE0]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK:                scf.yield [[INSERT0]] : tensor<1x32x540x960xf16, {order = #NHWC}>

    // CHECK: [[SIGN:%.+]] = VPU.Sign([[LOOP0]]) : tensor<1x32x540x960xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK: [[LOOP_OUTPUT1:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK: [[LOOP1:%.+]] = scf.for
    // CHECK-SAME:           [[LOOP_ITER1:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP1]]
    // CHECK-SAME:           iter_args([[LOOP_OUT1:%arg[0-9]]] = [[LOOP_OUTPUT1]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>)

    // CHECK:                [[INSERT_OFFSET1:%.+]] = affine.min #[[$CHAIN1_OFFSET_AND_SIZE_MAP]]([[LOOP_ITER1]])
    // CHECK:                [[INSERT_TILE_INDEX1_CAP:%.+]] = affine.min #[[$CHAIN1_TILE_INDEX_CAP_AND_REMAINDER_MAP]]([[LOOP_ITER1]])
    // CHECK:                [[INSERT_SIZE1:%.+]] = affine.min #[[$CHAIN1_SIZE_BY_CAPPED_TILE_INDEX_MAP]]([[INSERT_TILE_INDEX1_CAP]])

    // CHECK:                [[TEMP_VALUE2:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[INSERT_OFFSET1]])
    // CHECK:                [[TEMP_VALUE3:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[INSERT_OFFSET1]])
    // CHECK:                [[PAD_LOW2:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE3]]]
    // CHECK:                [[TEMP_VALUE4:%.+]] = affine.max #[[$PAD_HIGH_CHAIN0_CANDIDATE_MAP]]([[INSERT_SIZE1]], [[TEMP_VALUE2]])
    // CHECK:                [[PAD_HIGH2:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE4]]]
    // CHECK:                [[SLICE_OFFSET1:%.+]]  = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[TEMP_VALUE2]])
    // CHECK:                [[TEMP_VALUE5:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[TEMP_VALUE2]])
    // CHECK:                [[PAD_LOW1:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE5]]]
    // CHECK:                [[TEMP_VALUE6:%.+]] = affine.max #[[$PAD_HIGH_CHAIN1_STAGE1_CANDIDATE_MAP]]([[SLICE_OFFSET1]], [[INSERT_SIZE1]], [[PAD_LOW2]], [[PAD_HIGH2]])
    // CHECK:                [[PAD_HIGH1:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TEMP_VALUE6]]]
    // CHECK:                [[SLICE_SIZE1:%.+]] = affine.apply #[[$CHAIN1_SLICE_SIZE_MAP]]([[PAD_LOW1]], [[PAD_HIGH1]], [[INSERT_SIZE1]], [[PAD_LOW2]], [[PAD_HIGH2]])

    // CHECK:                [[SLICE1:%.+]] = tensor.extract_slice [[SIGN]][0, 0, 0, [[SLICE_OFFSET1]]] [1, 32, 540, [[SLICE_SIZE1]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                [[PAD1:%.+]] = tensor.pad [[SLICE1]] low[0, 0, 1, [[PAD_LOW1]]] high[0, 0, 1, [[PAD_HIGH1]]]
    // CHECK:                tensor.yield [[PAD_VALUE]] : f16
    // CHECK:                tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]

    // CHECK:                [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, 1, [[PAD_LOW2]]] high[0, 0, 1, [[PAD_HIGH2]]] {
    // CHECK:                tensor.yield [[PAD_VALUE]] : f16
    // CHECK:                tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]

    // CHECK:                [[INSERT1:%.+]] = tensor.insert_slice [[CONV2]] into [[LOOP_OUT1]][0, 0, 0, [[INSERT_OFFSET1]]] [1, 32, 540, [[INSERT_SIZE1]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK:                scf.yield [[INSERT1]] : tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK: return [[LOOP1]] : tensor<1x32x540x960xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

 // CHECK-LABEL: @MergeWithLayoutCast
 // CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x16x256x140xf16>,
 // CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x16x256x140xf16>)
 func.func @MergeWithLayoutCast(
         %arg0: tensor<1x16x256x140xf16>,
         %arg1: tensor<1x16x256x140xf16>
 ) -> tensor<1x16x256x140xf16> {
    %0 = VPU.LayoutCast(%arg0) {dst_order = #NHWC} : tensor<1x16x256x140xf16> -> tensor<1x16x256x140xf16, {order = #NHWC}>
    %1 = VPU.LayoutCast(%arg1) {dst_order = #NHWC} : tensor<1x16x256x140xf16> -> tensor<1x16x256x140xf16, {order = #NHWC}>
     %2 = VPU.NCE.Eltwise(%0, %1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
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
     } -> tensor<1x16x256x140xf16, {order = #NHWC}>

     %3 = VPU.LayoutCast(%2) {dst_order = #NCHW} : tensor<1x16x256x140xf16, {order = #NHWC}> -> tensor<1x16x256x140xf16>

     return %3 : tensor<1x16x256x140xf16>


    // CHECK: [[LOOP_STEP:%.+]] = arith.constant 70 : index
    // CHECK: [[LOOP_END:%.+]] = arith.constant 140 : index
    // CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x256x140xf16, {order = #NHWC}>

    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    // CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x256x140xf16, {order = #NHWC}>)

    // CHECK:                [[SLICE0:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, 70] [1, 1, 1, 1] : tensor<1x16x256x140xf16> to tensor<1x16x256x70xf16>
    // CHECK:                [[CAST0:%.+]] = VPU.LayoutCast([[SLICE0]]) {dst_order = #NHWC} : tensor<1x16x256x70xf16> -> tensor<1x16x256x70xf16, {order = #NHWC}>
    // CHECK:                [[SLICE1:%.+]] = tensor.extract_slice [[INPUT1]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, 70] [1, 1, 1, 1] : tensor<1x16x256x140xf16> to tensor<1x16x256x70xf16>
    // CHECK:                [[CAST1:%.+]] = VPU.LayoutCast([[SLICE1]]) {dst_order = #NHWC} : tensor<1x16x256x70xf16> -> tensor<1x16x256x70xf16, {order = #NHWC}>
    // CHECK:                [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CAST0]], [[CAST1]])
    // CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, 70] [1, 1, 1, 1] : tensor<1x16x256x70xf16, {order = #NHWC}> into tensor<1x16x256x140xf16, {order = #NHWC}>
    // CHECK:                scf.yield [[INSERT]] : tensor<1x16x256x140xf16, {order = #NHWC}>

    // pure view-like op doesn't have tilingStrategy, it cannot be tiled. it might be used to continue VF further, but we cannot start VF with that unfortunately
    // CHECK: [[CAST2:%.+]] = VPU.LayoutCast([[LOOP]]) {dst_order = #NCHW} : tensor<1x16x256x140xf16, {order = #NHWC}> -> tensor<1x16x256x140xf16>
    // CHECK: return [[CAST2]] : tensor<1x16x256x140xf16>
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 64)>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

 // CHECK-LABEL: @MergeConvertPermute
 // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
 func.func @MergeConvertPermute(
         %arg0: tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
 ) -> tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> {

    %0 = VPU.Convert(%arg0) {dstElemType = f16,
                            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                            tilingStrategy = [1, 1, 1, 17]}
                            : tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
                            -> tensor<1x3x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

    %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64,
                            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
                            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                            tilingStrategy = [1, 1, 36, 1]}
                            -> tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>

    return %1 : tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>


    // CHECK: [[LOOP_STEP:%.+]] = arith.constant 64 : index
    // CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK: [[DIM_INDEX:%.+]] = arith.constant 3 : index
    // CHECK: [[DIM0:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM0]]) : tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    // CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                [[INSERT_SIZE:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[LOOP_END]]]
    // CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 3, 1600, [[INSERT_SIZE]]] [1, 1, 1, 1] : tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:                [[CONVERT:%.+]] = VPU.Convert([[SLICE]])
    // CHECK:                [[PERMUTE:%.+]] = VPU.NCE.Permute([[CONVERT]])
    // CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[PERMUTE]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 1600, [[INSERT_SIZE]]] [1, 1, 1, 1] : tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 64]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                scf.yield [[INSERT]]

    // CHECK: return [[LOOP]] : tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 240)>

// CHECK-LABEL: @Merge2DDynamicEltwise
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
func.func @Merge2DDynamicEltwise(
         %arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
     %0 = VPU.NCE.Eltwise(%arg0, %arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
         is_inplace = true,
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
         tilingStrategy = [1, 1, 2, 4]
     } -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

     %1 = VPU.NCE.Eltwise(%0, %0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
         is_inplace = true,
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
         tilingStrategy = [1, 1, 2, 4]
     } -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

     return %1 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    // CHECK: [[LOOP_STEP_W:%.+]] = arith.constant 240 : index
    // CHECK: [[LOOP_STEP_H:%.+]] = arith.constant 16 : index
    // CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK: [[DIM_INDEX_H:%.+]] = arith.constant 2 : index

    // CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_END_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_END_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: [[LOOP_H:%.+]] = scf.for
    // CHECK-SAME:             [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    // CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                  [[LOOP_W:%.+]] = scf.for
    // CHECK-SAME:             [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    // CHECK-SAME:             iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                  [[SLICE_SIZE_H:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER_H]])[[[LOOP_END_H]]]
    // CHECK:                  [[SLICE_SIZE_W:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_W]])[[[LOOP_END_W]]]

    // CHECK:                  [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:             tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 240]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[SLICE]], [[SLICE]])
    // CHECK:                  [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[ELTWISE0]], [[ELTWISE0]])

    // CHECK:                  [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE1]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:             tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 240]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  scf.yield [[INSERT]]
    // CHECK:  scf.yield [[LOOP_W]]

    // CHECK: return [[LOOP_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 45)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 240)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$MAP4:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
// CHECK: #[[$MAP6:.+]] = affine_map<(d0, d1, d2, d3)[s0] -> (0, d0 + d1 - d2 - d3 - s0 + 4)>
// CHECK: #[[$MAP7:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (-d0 - d1 + d2 - d3 - d4 + 4)>


// CHECK-LABEL: @Merge2DVFChain3Tiles
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>)
func.func @Merge2DVFChain3Tiles(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
 {
    %cst = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                           pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                                           ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                           lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                            strides = [1, 1], tilingStrategy = [1, 1, 1, 21]}
        : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<32x32x3x3xf16, {order = #NHWC}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    %1 = VPU.NCE.DepthConvolution(%0, %cst_0) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                                       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                                       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                                       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                                        strides = [1, 1], tilingStrategy = [1, 1, 1, 20]}
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    %2 = VPU.NCE.Convolution(%1, %cst_1) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                                          ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                           strides = [1, 1], tilingStrategy = [1, 1, 1, 22]}
        : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<32x32x3x3xf16, {order = #NHWC}>
          -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    return %2: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-DAG: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK-DAG: [[DIM_INDEX_H:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant 240 : index
    // CHECK-DAG: [[LOOP_STEP_H:%.+]] = arith.constant 45 : index
    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_END_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_END_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: [[LOOP_H:%.+]] = scf.for
    // CHECK-SAME:             [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    // CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                  [[LOOP_W:%.+]] = scf.for
    // CHECK-SAME:             [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    // CHECK-SAME:             iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                  [[INSERT_SIZE_H:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER_H]])[[[LOOP_END_H]]]
    // CHECK:                  [[INSERT_SIZE_W:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_W]])[[[LOOP_END_W]]]

    // CHECK:                  [[DIM_H_1:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  [[DIM_W_1:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  [[TMP_VALUE7:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER_H]])
    // CHECK:                  [[TMP_VALUE6:%.+]] = affine.max #[[$MAP3]]([[LOOP_ITER_H]])
    // CHECK:                  [[PAD1_LOW_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE6]]]
    // CHECK:                  [[TMP_VALUE9:%.+]] = affine.max #[[$MAP5]]([[INSERT_SIZE_H]], [[TMP_VALUE7]])[[[DIM_H_1]]]
    // CHECK:                  [[PAD1_HIGH_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE9]]]
    // CHECK:                  [[TMP_VALUE5:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER_W]])
    // CHECK:                  [[TMP_VALUE8:%.+]] = affine.max #[[$MAP3]]([[LOOP_ITER_W]])
    // CHECK:                  [[PAD1_LOW_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE8]]]
    // CHECK:                  [[TMP_VALUE4:%.+]] = affine.max #[[$MAP5]]([[INSERT_SIZE_W]], [[TMP_VALUE5]])[[[DIM_W_1]]]
    // CHECK:                  [[PAD1_HIGH_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE4]]]

    // CHECK:                  [[DIM_H_2:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  [[DIM_W_2:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  [[SLICE_OFFSET_H:%.+]] = affine.max #[[$MAP2]]([[TMP_VALUE7]])
    // CHECK:                  [[TMP_VALUE3:%.+]] = affine.max #[[$MAP3]]([[TMP_VALUE7]])
    // CHECK:                  [[PAD0_LOW_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE3]]]
    // CHECK:                  [[TMP_VALUE2:%.+]] = affine.max #[[$MAP6]]([[SLICE_OFFSET_H]], [[INSERT_SIZE_H]], [[PAD1_LOW_H]], [[PAD1_HIGH_H]])[[[DIM_H_2]]]
    // CHECK:                  [[PAD0_HIGH_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE2]]]
    // CHECK:                  [[SLICE_SIZE_H:%.+]] = affine.apply #[[$MAP7]]([[PAD0_LOW_H]], [[PAD0_HIGH_H]], [[INSERT_SIZE_H]], [[PAD1_LOW_H]], [[PAD1_HIGH_H]])
    // CHECK:                  [[SLICE_OFFSET_W:%.+]] = affine.max #[[$MAP2]]([[TMP_VALUE5]])
    // CHECK:                  [[TMP_VALUE1:%.+]] = affine.max #[[$MAP3]]([[TMP_VALUE5]])
    // CHECK:                  [[PAD0_LOW_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE1]]]
    // CHECK:                  [[TMP_VALUE0:%.+]] = affine.max #[[$MAP6]]([[SLICE_OFFSET_W]], [[INSERT_SIZE_W]], [[PAD1_LOW_W]], [[PAD1_HIGH_W]])[[[DIM_W_2]]]
    // CHECK:                  [[PAD0_HIGH_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE0]]]
    // CHECK:                  [[SLICE_SIZE_W:%.+]] = affine.apply #[[$MAP7]]([[PAD0_LOW_W]], [[PAD0_HIGH_W]], [[INSERT_SIZE_W]], [[PAD1_LOW_W]], [[PAD1_HIGH_W]])

    // CHECK:                  [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET_H]], [[SLICE_OFFSET_W]]] [1, 32, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:             tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD0_LOW_H]], [[PAD0_LOW_W]]] high[0, 0, [[PAD0_HIGH_H]], [[PAD0_HIGH_W]]] {
    // CHECK:                  tensor.yield [[PAD_VALUE]] : f16
    // CHECK:                  tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 47, 242]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  [[CONV0:%.+]]  = VPU.NCE.Convolution([[PAD0]]
    // CHECK:                  [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[CONV0]]
    // CHECK:                  [[PAD1:%.+]] = tensor.pad [[DWCONV]] low[0, 0, [[PAD1_LOW_H]], [[PAD1_LOW_W]]] high[0, 0, [[PAD1_HIGH_H]], [[PAD1_HIGH_W]]] {
    // CHECK:                  tensor.yield [[PAD_VALUE]] : f16
    // CHECK:                  tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 47, 242]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    // CHECK:                  [[INSERT:%.+]] = tensor.insert_slice [[CONV1]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 32, [[INSERT_SIZE_H]], [[INSERT_SIZE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:             tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  scf.yield [[INSERT]]

    // CHECK:  scf.yield [[LOOP_W]]

    // CHECK: return [[LOOP_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


// CHECK-LABEL: @Merge2DVFChainConvertPermute
func.func @Merge2DVFChainConvertPermute(%arg0: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>)
     -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    %2 = VPU.Convert(%arg0) {
            dstElemType = f16,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            tilingStrategy = [1, 1, 1, 26]}
            : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
                -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    %3 = VPU.NCE.Permute(%2) {
            dstElemType = f16,
            dstOrder = #NHWC,
            expandedChannels = 16 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
            ppe = #VPU.PPEInt<mode = <NOOP>,
            clamp_low = -2147483648 : i64,
            clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64>,
            tilingStrategy = [1, 1, 1, 80]}
                -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
    return %3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:   [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    // CHECK:   [[LOOP_STEP_W:%.+]] = arith.constant {{[0-9]+}} : index
    // CHECK:   [[LOOP_STEP_H:%.+]] = arith.constant {{[0-9]+}} : index
    // CHECK:   [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK:   [[DIM_INDEX_H:%.+]] = arith.constant 2 : index
    // CHECK:   [[DIM_H:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_H]]
    // CHECK:   [[DIM_W:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_W]]
    // CHECK:   [[EMPTY:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   [[DIM_H2:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_H]]
    // CHECK:   [[DIM_W2:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_W]]
    // CHECK:   [[RESULT:%.+]] = scf.for [[SLICE_OFFSET_H:%.+]] = [[LOOP_BEGIN]] to [[DIM_H2]] step [[LOOP_STEP_H]] iter_args([[OUTER_OUTPUT:%.+]] = [[EMPTY]])
    // CHECK:       [[RESULT_W:%.+]] = scf.for [[SLICE_OFFSET_W:%.+]] = [[LOOP_BEGIN]] to [[DIM_W2]] step [[LOOP_STEP_W]] iter_args([[INNER_OUTPUT:%.+]] = [[OUTER_OUTPUT]])
    // CHECK:       [[SLICE_SIZE_H:%.+]] = affine.min #map([[SLICE_OFFSET_H]])[[[DIM_H2]]]
    // CHECK:       [[SLICE_SIZE_W:%.+]] = affine.min #map1([[SLICE_OFFSET_W]])[[[DIM_W2]]]
    // CHECK:       [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[SLICE_OFFSET_H]], [[SLICE_OFFSET_W]]] [1, 3, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    // CHECK:       [[PERMUTE:%.+]] = VPU.NCE.Permute([[CONVERT]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>
    // CHECK:       [[INSERT:%.+]] = tensor.insert_slice [[PERMUTE]] into [[INNER_OUTPUT]]
    // CHECK:       scf.yield [[INSERT]]
    // CHECK:   scf.yield [[RESULT_W]]
    // CHECK:   return [[RESULT]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeLoopWithEltwise
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
func.func @MergeLoopWithEltwise(
         %arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> {

    %cst = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_7 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 1, 20],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
         strides = [1, 1]
    }  : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    %1 = VPU.NCE.DepthConvolution(%0, %cst_1) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 1, 20],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
         strides = [1, 1]
    } -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    %2 = VPU.NCE.Convolution(%1, %cst_3) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 1, 20],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
         strides = [1, 1]
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    %3 = VPU.NCE.Convolution(%2, %cst_5) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 1, 20],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
         strides = [1, 1]
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%3, %cst_7) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 1, 20],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
         strides = [1, 1]
    } -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    %5 = VPU.NCE.Eltwise(%4, %1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        op_type = #VPU.eltwise_type<ADD>, tilingStrategy = [1, 1, 1, 20],
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00],
        fp_prelu_alpha = 1.000000e+00 : f64>
    } -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    return %5 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-DAG: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[DIM_INDEX_H:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant 256 : index
    // CHECK-DAG: [[LOOP_STEP_H:%.+]] = arith.constant 48 : index
    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_END_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_END_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: [[LOOP_H:%.+]] = scf.for
    // CHECK-SAME:             [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    // CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                  [[LOOP_W:%.+]] = scf.for
    // CHECK-SAME:             [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    // CHECK-SAME:             iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                  [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET_H:%.+]], [[SLICE_OFFSET_W:%.+]]] [1, 32, [[EXTRACT_SIZE_H:%.+]], [[EXTRACT_SIZE_W:%.+]]] [1, 1, 1, 1]

    // CHECK:                  [[PAD0:%.+]] = tensor.pad [[SLICE]]

    // CHECK:                  [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]
    // CHECK:                  [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[CONV0]]

    // CHECK:                  [[PAD1:%.+]] = tensor.pad [[DWCONV0]]

    // CHECK:                  [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    // CHECK:                  [[PAD2:%.+]] = tensor.pad [[CONV1]]

    // CHECK:                  [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]
    // CHECK:                  [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution([[CONV2]]

    // CHECK:                  [[EXTRACT_SLICE_SKIP_CONNECTION:%.+]] = tensor.extract_slice [[DWCONV0]]
    // CHECK:                  [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[DWCONV1]], [[EXTRACT_SLICE_SKIP_CONNECTION]])

    // CHECK:                  [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]]
    // CHECK-SAME:             tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 256]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  scf.yield [[INSERT]]
    // CHECK:  scf.yield [[LOOP_W]]

    // CHECK: return [[LOOP_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputF32DynamicType = tensor<1x12x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 12, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
!inputDynamicType = tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
!outputDynamicType = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>

module @test {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
// CHECK-DAG: #map = affine_map<(d0)[s0] -> (-d0 + s0, {{[0-9]+}})>
// CHECK-DAG: #map1 = affine_map<(d0)[s0] -> (-d0 + s0, {{[0-9]+}})>
// CHECK-DAG: #map2 = affine_map<(d0) -> (d0 floordiv 2)>

// CHECK: @Merge2DVFChainConvertD2S
func.func @Merge2DVFChainConvertD2S(%arg0: !inputF32DynamicType) -> !outputDynamicType {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x12x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 12, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
    %0 = VPU.Convert(%arg0) {
        dstElemType = f16,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 1, 26]}
        : !inputF32DynamicType -> !inputDynamicType

    %1 = VPU.DepthToSpace(%0) {
        block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 73, 1]}
          : !inputDynamicType -> !outputDynamicType

    return %1 : !outputDynamicType

    // CHECK:   [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    // CHECK:   [[LOOP_STEP_W:%.+]] = arith.constant {{[0-9]+}} : index
    // CHECK:   [[LOOP_STEP_H:%.+]] = arith.constant {{[0-9]+}} : index
    // CHECK:   [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK:   [[CST_VAL_2:%.+]] = arith.constant 2 : index

    // CHECK:   [[DIM_H:%.+]] = tensor.dim [[ARG0]], [[CST_VAL_2]]
    // CHECK:   [[OUT_DIM_H:%.+]] = arith.muli [[DIM_H]], [[CST_VAL_2]]

    // CHECK:   [[DIM_W:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_W]]
    // CHECK:   [[OUT_DIM_W:%.+]] = arith.muli [[DIM_W]], [[CST_VAL_2]]

    // CHECK:   [[EMPTY:%.+]] = tensor.empty([[OUT_DIM_H]], [[OUT_DIM_W]]) : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:   [[DIM_H2:%.+]] = tensor.dim [[ARG0]], [[CST_VAL_2]]
    // CHECK:   [[OUT_DIM_H2:%.+]] = arith.muli [[DIM_H2]], [[CST_VAL_2]]

    // CHECK:   [[DIM_W2:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_W]]
    // CHECK:   [[OUT_DIM_W2:%.+]] = arith.muli [[DIM_W2]], [[CST_VAL_2]]

    // CHECK:   [[RESULT:%.+]] = scf.for [[SLICE_OFFSET_H:%.+]] = [[LOOP_BEGIN]] to [[OUT_DIM_H2]] step [[LOOP_STEP_H]] iter_args([[OUTER_OUTPUT:%.+]] = [[EMPTY]])
    // CHECK:       [[RESULT_W:%.+]] = scf.for [[SLICE_OFFSET_W:%.+]] = [[LOOP_BEGIN]] to [[OUT_DIM_W2]] step [[LOOP_STEP_W]] iter_args([[INNER_OUTPUT:%.+]] = [[OUTER_OUTPUT]])

    // CHECK:       [[SLICE_SIZE_H:%.+]] = affine.min #map([[SLICE_OFFSET_H]])[[[OUT_DIM_H2]]]
    // CHECK:       [[SLICE_SIZE_W:%.+]] = affine.min #map1([[SLICE_OFFSET_W]])[[[OUT_DIM_W2]]]
    // CHECK:       [[IN_SLICE_OFFSET_H:%.+]] = affine.apply #map2([[SLICE_OFFSET_H]])
    // CHECK:       [[IN_SLICE_SIZE_H:%.+]] = affine.apply #map2([[SLICE_SIZE_H]])
    // CHECK:       [[IN_SLICE_OFFSET_W:%.+]] = affine.apply #map2([[SLICE_OFFSET_W]])
    // CHECK:       [[IN_SLICE_SIZE_W:%.+]] = affine.apply #map2([[SLICE_SIZE_W]])

    // CHECK:       [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IN_SLICE_OFFSET_H]], [[IN_SLICE_OFFSET_W]]] [1, 12, [[IN_SLICE_SIZE_H]], [[IN_SLICE_SIZE_W]]] [1, 1, 1, 1]

    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[CONVERT]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:    : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 48, 512]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:    -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 96, 1024]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[INSERT:%.+]] = tensor.insert_slice [[D2S]] into [[INNER_OUTPUT]]
    // CHECK:       scf.yield [[INSERT]]
    // CHECK:   scf.yield [[RESULT_W]]
    // CHECK:   return [[RESULT]] : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>
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

// CHECK-LABEL: @MergeVFPermuteCastStatic
func.func @MergeVFPermuteCastStatic(%arg0: tensor<1x1080x1920x16xf16>)
     -> tensor<1x16x1080x1920xf16, {order = #NHWC}> {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1080x1920x16xf16>
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1080x1920x16xf16> -> tensor<1x16x1080x1920xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64,
        lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1],
        tilingStrategy = [1, 1, 1, 80]}
        : tensor<1x16x1080x1920xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
     -> tensor<1x16x1080x1920xf16, {order = #NHWC}>

    return %1: tensor<1x16x1080x1920xf16, {order = #NHWC}>

    // CHECK:   [[LOOP_STEP:%.+]] = arith.constant 24 : index
    // CHECK:   [[LOOP_END:%.+]] = arith.constant 1920 : index
    // CHECK:   [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK:   [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x1080x1920xf16, {order = #NHWC}>
    // CHECK:   [[LOOP:%.+]] = scf.for
    // CHECK-SAME:             [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    // CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x1080x1920xf16, {order = #NHWC}>) {

    // CHECK:                 [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[LOOP_ITER]], 0] [1, 1080, 24, 16] [1, 1, 1, 1]
    // CHECK-SAME:            tensor<1x1080x1920x16xf16> to tensor<1x1080x24x16xf16>
    // CHECK:                 [[PERMUTECAST:%.+]] = VPU.PermuteCast([[SLICE]])
    // CHECK:                 [[CONV:%.+]] = VPU.NCE.Convolution([[PERMUTECAST]]
    // CHECK:                 [[INSERT:%.+]] = tensor.insert_slice [[CONV]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 1080, 24] [1, 1, 1, 1]
    // CHECK-SAME:            tensor<1x16x1080x24xf16, {order = #NHWC}> into tensor<1x16x1080x1920xf16, {order = #NHWC}>
    // CHECK:   scf.yield [[INSERT]]

    // CHECK:   return [[LOOP]] : tensor<1x16x1080x1920xf16, {order = #NHWC}>
}

// -----
// CHECK-LABEL: @QuantizeCastSCFVerticalFusionBlock

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0) -> (-d0 + 1280, 21)>
#map1 = affine_map<(d0) -> (0, d0 - 1)>
#map2 = affine_map<(d0) -> (-d0 + 1, 0)>
#map3 = affine_map<()[s0] -> (1, s0)>
#map4 = affine_map<(d0, d1) -> (0, d0 + d1 - 1278)>
#map5 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
module @QuantizeCastSCFVerticalFusionBlock {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    config.PipelineOptions @Options {
            config.Option @VPU.AutoPaddingODU : true
            config.Option @VPU.AutoPaddingIDU : true
            config.Option @VPU.ReduceSupported : false
    }
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" tensorNames = ["input"] : tensor<1x32x800x1280xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    } outputsInfo : {
        DataInfo "output" friendlyName = "output/sink_port_0" tensorNames = ["output"] : tensor<1x32x800x1280xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    }
    func.func @main(%arg0: tensor<1x32x800x1280xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>) -> tensor<1x32x800x1280xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>  {
        %cst_36 = const.Declare tensor<32x32x3x3x!quant.uniform<i8:f16, 5.1E-4>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<123> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<u8:f16, 5.1E-4:128>>, #const.ConvertElemType<!quant.uniform<i8:f16, 5.1E-4>>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
        %cst_24 = const.Declare tensor<32x16x1x1x!quant.uniform<u8<0:254>:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 32 : i64>, #const.CastElemType<!quant.uniform<u8<0:254>:f16, 1.000000e+00>>, #const.Reshape<[32, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]

        %13 = VPU.NCE.Convolution(%arg0, %cst_36) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 0.1 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 24]} : tensor<1x32x800x1280xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<32x32x3x3x!quant.uniform<i8:f16, 5.1E-4>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.123:106>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
        %14 = VPU.QuantizeCast(%13) {dstElemType = !quant.uniform<u8:f16, 1.000000e+00>} : tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.123:106>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
        %15 = VPU.NCE.DepthConvolution(%14, %cst_24) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 18]} -> tensor<1x32x800x1280xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
        return %15 : tensor<1x32x800x1280xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    }
    // CHECK:   func.func @main([[ARG0:%.+]]: tensor<1x32x800x1280xf16, {order = #NHWC}>
    // CHECK:   [[YIELD_VAL:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK:   [[LOOP_STEP_W:%.+]] = arith.constant 21 : index
    // CHECK:   [[DIM_END_W:%.+]] = arith.constant 1280 : index
    // CHECK:   [[LOOP_BEGIN_W:%.+]] = arith.constant 0 : index
    // CHECK:   [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x800x1280xf16, {order = #NHWC}>
    // CHECK:   [[LOOP_W:%.+]] = scf.for
    // CHECK-SAME:              [[DIM:%arg[0-9]]] = [[LOOP_BEGIN_W]] to [[DIM_END_W]] step [[LOOP_STEP_W]]
    // CHECK-SAME:              iter_args([[LOOP_OUTPUT_STEP:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x800x1280xf16, {order = #NHWC}>) {
    // CHECK:     [[OUT_SIZE:%.+]]       = affine.min #map([[DIM]])
    // CHECK:     [[SLICE_OFFSET:%.+]]   = affine.max #map1([[DIM]])
    // CHECK:     [[PAD_CAND_LOW:%.+]]   = affine.max #map2([[DIM]])
    // CHECK:     [[PAD_LOW:%.+]]        = affine.min #map3()[[[PAD_CAND_LOW]]]
    // CHECK:     [[PAD_CAND_HIGH:%.+]]  = affine.max #map4([[OUT_SIZE]], [[SLICE_OFFSET]])
    // CHECK:     [[PAD_HIGH:%.+]]       = affine.min #map3()[[[PAD_CAND_HIGH]]]
    // CHECK:     [[IN_SLICE_SIZE:%.+]]  = affine.apply #map5([[OUT_SIZE]], [[PAD_LOW]], [[PAD_HIGH]])
    // CHECK:     [[EXTRACTED_SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[SLICE_OFFSET]]] [1, 32, 800, [[IN_SLICE_SIZE]]] [1, 1, 1, 1]

    // CHECK:     [[PADDED:%.+]] = tensor.pad [[EXTRACTED_SLICE]] low[0, 0, 1, [[PAD_LOW]]] high[0, 0, 1, [[PAD_HIGH]]] {
    // CHECK-NEXT: ^bb0({{%[^:]+}}: index, {{%[^:]+}}: index, {{%[^:]+}}: index, {{%[^:]+}}: index):
    // CHECK-NEXT: tensor.yield [[YIELD_VAL]] : f16
    // CHECK:   [[CONVOLUTION:%.+]] = VPU.NCE.Convolution([[PADDED]]
    // CHECK-SAME: -> tensor<1x32x800x?x!qElemType3, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   [[QUANTIZE_CAST:%.+]] = VPU.QuantizeCast([[CONVOLUTION]]) {dstElemType = !qElemType4} : tensor<1x32x800x?x!qElemType3, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x32x800x?x!qElemType4, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   [[CONVOLUTION_0:%.+]] = VPU.NCE.DepthConvolution([[QUANTIZE_CAST]]
    // CHECK-SAME: -> tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   [[INSERTED_SLICE:%.+]] = tensor.insert_slice [[CONVOLUTION_0]] into [[LOOP_OUTPUT_STEP]][0, 0, 0, [[DIM]]] [1, 32, 800, [[OUT_SIZE]]] [1, 1, 1, 1] : tensor<1x32x800x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x800x1280xf16, {order = #NHWC}>
    // CHECK:   scf.yield [[INSERTED_SLICE]] : tensor<1x32x800x1280xf16, {order = #NHWC}>
    // CHECK:   return [[LOOP_W]] : tensor<1x32x800x1280xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, {{[0-9]+}})>
// CHECK: [[$MAP_1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, {{[0-9]+}})>

    // CHECK-LABEL: @FusionConvertPermute4Ops2DimDynamic
    // CHECK-SAME: ([[INPUT:%.+]]: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    func.func @FusionConvertPermute4Ops2DimDynamic(%arg0: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 39, 2]} : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, tilingStrategy = [1, 1, 13, 2]} -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 13, 2]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_0) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 13, 2]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    return %3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>

	// CHECK: [[C3:%.+]] = arith.constant 3 : index
    // CHECK: [[C_W_STEP:%.+]] = arith.constant [[W_STEP:[0-9]+]] : index
    // CHECK: [[C_H_STEP:%.+]] = arith.constant [[H_STEP:[0-9]+]] : index
    // CHECK: [[C0:%.+]] = arith.constant 0 : index
	// CHECK: [[CST:%.+]] = const.Declare
	// CHECK: [[CST_0:%.+]] = const.Declare
	// CHECK: [[C2:%.+]] = arith.constant 2 : index

    // CHECK: [[DIM:%.+]] = tensor.dim [[INPUT]], [[C2]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[DIM_1:%.+]] = tensor.dim [[INPUT]], [[C3]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[BUF:%.+]] = tensor.empty([[DIM]], [[DIM_1]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[DIM_2:%.+]] = tensor.dim [[INPUT]], [[C2]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[DIM_3:%.+]] = tensor.dim [[INPUT]], [[C3]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[LOOP_0:%.+]] = scf.for [[LOOP_ITER_0:%.+]] = [[C0]] to [[DIM_2]] step [[C_H_STEP]] iter_args([[OUTPUT_BUF_0:%.+]] = [[BUF]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK: [[LOOP_1:%.+]] = scf.for [[LOOP_ITER_1:%.+]] = [[C0]] to [[DIM_3]] step [[C_W_STEP]] iter_args([[OUTPUT_BUF_1:%.+]] = [[OUTPUT_BUF_0]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK: [[HEIGHT:%.+]] = affine.min [[$MAP]]([[LOOP_ITER_0]])[[[DIM_2]]]
    // CHECK: [[WIDTH:%.+]] = affine.min [[$MAP_1]]([[LOOP_ITER_1]])[[[DIM_3]]]
    // CHECK: [[EXTRACTED_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[LOOP_ITER_0]], [[LOOP_ITER_1]]] [1, 16, [[HEIGHT]], [[WIDTH]]] [1, 1, 1, 1] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], [[W_STEP]]]> : tensor<4xsi64>, order = #NCHW}>
	// CHECK:   [[CONVERT:%.+]] = VPU.Convert([[EXTRACTED_SLICE]])
	// CHECK: [[PERMUTE:%.+]] = VPU.NCE.Permute([[CONVERT]])
    // CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[PERMUTE]], [[CST]]) rawFilterShape [16, 16, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
 // CHECK-SAME: strides = [1, 1]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], [[W_STEP]]]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], [[W_STEP]]]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[CONV_0]], [[CST_0]]) rawFilterShape [16, 16, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME: strides = [1, 1]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], [[W_STEP]]]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], [[W_STEP]]]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[INSERTED_SLICE_1:%.+]] = tensor.insert_slice [[CONV_1]] into [[OUTPUT_BUF_1]][0, 0, [[LOOP_ITER_0]], [[LOOP_ITER_1]]] [1, 16, [[HEIGHT]], [[WIDTH]]] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], [[W_STEP]]]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: scf.yield [[INSERTED_SLICE_1]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: return [[LOOP_0]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>

  }

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, {{[0-9]+}})>

    // CHECK-LABEL: @FusionConvertPermute4Ops1DimDynamic
    // CHECK-SAME: ([[INPUT:%.+]]: tensor<1x16x?x1280xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    func.func @FusionConvertPermute4Ops1DimDynamic(%arg0: tensor<1x16x?x1280xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 39, 1]} : tensor<1x16x?x1280xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, tilingStrategy = [1, 1, 26, 1]} -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 26, 1]} : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_0) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 26, 1]} : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    return %3 : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: [[C_H_STEP:%.+]] = arith.constant [[H_STEP:[0-9]+]] : index
    // CHECK: [[C0:%.+]] = arith.constant 0 : index
	// CHECK: [[CST:%.+]] = const.Declare
	// CHECK: [[CST_0:%.+]] = const.Declare
	// CHECK: [[C2:%.+]] = arith.constant 2 : index

    // CHECK: [[DIM:%.+]] = tensor.dim [[INPUT]], [[C2]] : tensor<1x16x?x1280xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[BUF:%.+]] = tensor.empty([[DIM]]) : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[DIM_1:%.+]] = tensor.dim [[INPUT]], [[C2]] : tensor<1x16x?x1280xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[LOOP:%.+]] = scf.for [[LOOP_ITER:%.+]] = [[C0]] to [[DIM_1]] step [[C_H_STEP]] iter_args([[OUTPUT_BUF:%.+]] = [[BUF]]) -> (tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK: [[HEIGHT:%.+]] = affine.min [[$MAP]]([[LOOP_ITER]])[[[DIM_1]]]
    // CHECK: [[EXTRACTED_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[LOOP_ITER]], 0] [1, 16, [[HEIGHT]], 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x16x?x1280xf32, {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], 1280]> : tensor<4xsi64>, order = #NCHW}>
	// CHECK: [[CONVERT:%.+]] = VPU.Convert([[EXTRACTED_SLICE]])
	// CHECK: [[PERMUTE:%.+]] = VPU.NCE.Permute([[CONVERT]])
    // CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[PERMUTE]], [[CST]]) rawFilterShape [16, 16, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
 // CHECK-SAME: strides = [1, 1]} : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[CONV_0]], [[CST_0]]) rawFilterShape [16, 16, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME: strides = [1, 1]} : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[INSERTED_SLICE:%.+]] = tensor.insert_slice [[CONV_1]] into [[OUTPUT_BUF]][0, 0, [[LOOP_ITER]], 0] [1, 16, [[HEIGHT]], 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: scf.yield [[INSERTED_SLICE]] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: return [[LOOP]] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
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

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-DAG: [[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 432)>
// CHECK-DAG: [[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 48)>

// CHECK-LABEL: @PermuteEltwiseFusion
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
func.func @PermuteEltwiseFusion(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        tilingStrategy = [1, 1, 13, 2]
        } -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>

    %1 = VPU.NCE.Eltwise(%0, %0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        tilingStrategy = [1, 1, 13, 2],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        tilingStrategy = [1, 1, 13, 2],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    return %2 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-DAG:    [[DIM_W:%.+]] = arith.constant 3 : index
    // CHECK-DAG:    [[LOOP_W_STEP:%.+]] = arith.constant 432 : index
    // CHECK-DAG:    [[LOOP_H_STEP:%.+]] = arith.constant 48 : index
    // CHECK-DAG:    [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK-DAG:    [[DIM_H:%.+]] = arith.constant 2 : index
    // CHECK-DAG:    [[DIM_H_0:%.+]] = tensor.dim [[INPUT]], [[DIM_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-DAG:    [[DIM_W_0:%.+]] = tensor.dim [[INPUT]], [[DIM_W]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[LOOP_OUT:%.+]] = tensor.empty([[DIM_H_0]], [[DIM_W_0]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:    [[LOOP_H_END:%.+]] = tensor.dim [[INPUT]], [[DIM_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:    [[LOOP_W_END:%.+]] = tensor.dim [[INPUT]], [[DIM_W]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[LOOP_H:%.+]] = scf.for
    // CHECK-SAME:               [[LOOP_H_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_H_END]] step [[LOOP_H_STEP]]
    // CHECK-SAME:               iter_args([[LOOP_OUT_H:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>)

    // CHECK:                    [[LOOP_W:%.+]] = scf.for
    // CHECK-SAME:               [[LOOP_W_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_W_END]] step [[LOOP_W_STEP]]
    // CHECK-SAME:               iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT_H]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>)

    // CHECK:      [[SIZE_W:%.+]] = affine.min [[$MAP]]([[LOOP_W_ITER]])[[[LOOP_W_END]]]
    // CHECK:      [[SIZE_H:%.+]] = affine.min [[$MAP1]]([[LOOP_H_ITER]])[[[LOOP_H_END]]]

    // CHECK:      [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[LOOP_H_ITER]], [[LOOP_W_ITER]]] [1, 16, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:  tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 432]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:      [[PERMUTE:%.+]] = VPU.NCE.Permute([[SLICE]])
    // CHECK:      [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[PERMUTE]], [[PERMUTE]])
    // CHECK:      [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[PERMUTE]], [[ELTWISE0]])
    // CHECK:      [[INSERT_SLICE:%.+]] = tensor.insert_slice [[ELTWISE1]] into [[LOOP_OUT_W]][0, 0, [[LOOP_H_ITER]], [[LOOP_W_ITER]]] [1, 16, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
    // CHECK-SAME: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 432]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:      scf.yield [[INSERT_SLICE]]

    // CHECK:  scf.yield [[LOOP_W]]
    // CHECK: return [[LOOP_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 45)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 320)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$MAP4:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
// CHECK: #[[$MAP6:.+]] = affine_map<(d0, d1, d2, d3)[s0] -> (0, d0 + d1 - d2 - d3 - s0 + 4)>
// CHECK: #[[$MAP7:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (-d0 - d1 + d2 - d3 - d4 + 4)>


// CHECK-LABEL: @Merge2DVFChainCompressConv
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>)
func.func @Merge2DVFChainCompressConv(%arg0: tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
 {
    %cst = const.Declare tensor<32x4x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x4x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %0 = VPU.NCE.CompressConvolution(%arg0, %cst, %cst_3) rawFilterShape [32, 4, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                           pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                                           ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                           lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                            strides = [1, 1], tilingStrategy = [1, 1, 1, 21], cm_sp_pattern = 0}
        : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<32x4x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    %1 = VPU.NCE.DepthConvolution(%0, %cst_0) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                                       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                                       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                                       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                                        strides = [1, 1], tilingStrategy = [1, 1, 1, 20]}
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    %2 = VPU.NCE.Convolution(%1, %cst_1) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                                          ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                           strides = [1, 1], tilingStrategy = [1, 1, 1, 22]}
        : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<32x32x3x3xf16, {order = #NHWC}>
          -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    return %2: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-DAG: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK-DAG: [[DIM_INDEX_H:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant 320 : index
    // CHECK-DAG: [[LOOP_STEP_H:%.+]] = arith.constant 45 : index
    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_END_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_END_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: [[LOOP_H:%.+]] = scf.for
    // CHECK-SAME:             [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    // CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                  [[LOOP_W:%.+]] = scf.for
    // CHECK-SAME:             [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    // CHECK-SAME:             iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                  [[INSERT_SIZE_H:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER_H]])[[[LOOP_END_H]]]
    // CHECK:                  [[INSERT_SIZE_W:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_W]])[[[LOOP_END_W]]]

    // CHECK:                  [[DIM_H_1:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  [[DIM_W_1:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  [[TMP_VALUE7:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER_H]])
    // CHECK:                  [[TMP_VALUE6:%.+]] = affine.max #[[$MAP3]]([[LOOP_ITER_H]])
    // CHECK:                  [[PAD1_LOW_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE6]]]
    // CHECK:                  [[TMP_VALUE9:%.+]] = affine.max #[[$MAP5]]([[INSERT_SIZE_H]], [[TMP_VALUE7]])[[[DIM_H_1]]]
    // CHECK:                  [[PAD1_HIGH_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE9]]]
    // CHECK:                  [[TMP_VALUE5:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER_W]])
    // CHECK:                  [[TMP_VALUE8:%.+]] = affine.max #[[$MAP3]]([[LOOP_ITER_W]])
    // CHECK:                  [[PAD1_LOW_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE8]]]
    // CHECK:                  [[TMP_VALUE4:%.+]] = affine.max #[[$MAP5]]([[INSERT_SIZE_W]], [[TMP_VALUE5]])[[[DIM_W_1]]]
    // CHECK:                  [[PAD1_HIGH_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE4]]]

    // CHECK:                  [[DIM_H_2:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  [[DIM_W_2:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  [[SLICE_OFFSET_H:%.+]] = affine.max #[[$MAP2]]([[TMP_VALUE7]])
    // CHECK:                  [[TMP_VALUE3:%.+]] = affine.max #[[$MAP3]]([[TMP_VALUE7]])
    // CHECK:                  [[PAD0_LOW_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE3]]]
    // CHECK:                  [[TMP_VALUE2:%.+]] = affine.max #[[$MAP6]]([[SLICE_OFFSET_H]], [[INSERT_SIZE_H]], [[PAD1_LOW_H]], [[PAD1_HIGH_H]])[[[DIM_H_2]]]
    // CHECK:                  [[PAD0_HIGH_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE2]]]
    // CHECK:                  [[SLICE_SIZE_H:%.+]] = affine.apply #[[$MAP7]]([[PAD0_LOW_H]], [[PAD0_HIGH_H]], [[INSERT_SIZE_H]], [[PAD1_LOW_H]], [[PAD1_HIGH_H]])
    // CHECK:                  [[SLICE_OFFSET_W:%.+]] = affine.max #[[$MAP2]]([[TMP_VALUE5]])
    // CHECK:                  [[TMP_VALUE1:%.+]] = affine.max #[[$MAP3]]([[TMP_VALUE5]])
    // CHECK:                  [[PAD0_LOW_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE1]]]
    // CHECK:                  [[TMP_VALUE0:%.+]] = affine.max #[[$MAP6]]([[SLICE_OFFSET_W]], [[INSERT_SIZE_W]], [[PAD1_LOW_W]], [[PAD1_HIGH_W]])[[[DIM_W_2]]]
    // CHECK:                  [[PAD0_HIGH_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE0]]]
    // CHECK:                  [[SLICE_SIZE_W:%.+]] = affine.apply #[[$MAP7]]([[PAD0_LOW_W]], [[PAD0_HIGH_W]], [[INSERT_SIZE_W]], [[PAD1_LOW_W]], [[PAD1_HIGH_W]])

    // CHECK:                  [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET_H]], [[SLICE_OFFSET_W]]] [1, 4, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:             tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 45, 320]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD0_LOW_H]], [[PAD0_LOW_W]]] high[0, 0, [[PAD0_HIGH_H]], [[PAD0_HIGH_W]]] {
    // CHECK:                  tensor.yield [[PAD_VALUE]] : f16
    // CHECK:                  tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 45, 320]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 47, 322]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  [[CONV0:%.+]]  = VPU.NCE.CompressConvolution([[PAD0]]
    // CHECK:                  [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[CONV0]]
    // CHECK:                  [[PAD1:%.+]] = tensor.pad [[DWCONV]] low[0, 0, [[PAD1_LOW_H]], [[PAD1_LOW_W]]] high[0, 0, [[PAD1_HIGH_H]], [[PAD1_HIGH_W]]] {
    // CHECK:                  tensor.yield [[PAD_VALUE]] : f16
    // CHECK:                  tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 320]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 47, 322]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    // CHECK:                  [[INSERT:%.+]] = tensor.insert_slice [[CONV1]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 32, [[INSERT_SIZE_H]], [[INSERT_SIZE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:             tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 320]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  scf.yield [[INSERT]]

    // CHECK:  scf.yield [[LOOP_W]]

    // CHECK: return [[LOOP_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 35)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$MAP4:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
// CHECK: #[[$MAP6:.+]] = affine_map<(d0, d1, d2, d3)[s0] -> (0, d0 + d1 - d2 - d3 - s0 + 4)>
// CHECK: #[[$MAP7:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (-d0 - d1 + d2 - d3 - d4 + 4)>


// CHECK-LABEL: @Merge1DVFChainCompressConv
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x4x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>)
func.func @Merge1DVFChainCompressConv(%arg0: tensor<1x4x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
 {
    %cst = const.Declare tensor<32x4x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x4x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %0 = VPU.NCE.CompressConvolution(%arg0, %cst, %cst_3) rawFilterShape [32, 4, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                           pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                                           ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                           lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                            strides = [1, 1], tilingStrategy = [1, 1, 1, 21], cm_sp_pattern = 0}
        : tensor<1x4x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<32x4x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32>
        -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    %1 = VPU.NCE.DepthConvolution(%0, %cst_0) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                                       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                                       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                                       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                                        strides = [1, 1], tilingStrategy = [1, 1, 1, 20]}
        -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    %2 = VPU.NCE.Convolution(%1, %cst_1) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                                          ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                           strides = [1, 1], tilingStrategy = [1, 1, 1, 22]}
        : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<32x32x3x3xf16, {order = #NHWC}>
          -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    return %2: tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-DAG: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant 35 : index
    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x4x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_OUT:%.+]] = tensor.empty([[DIM_W]]) : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_END_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x4x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  [[LOOP_W:%.+]] = scf.for
    // CHECK-SAME:             [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    // CHECK-SAME:             iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                  [[INSERT_SIZE_W:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_W]])[[[LOOP_END_W]]]

    // CHECK:                  [[DIM_W_1:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x4x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  [[TMP_VALUE5:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER_W]])
    // CHECK:                  [[TMP_VALUE8:%.+]] = affine.max #[[$MAP3]]([[LOOP_ITER_W]])
    // CHECK:                  [[PAD1_LOW_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE8]]]
    // CHECK:                  [[TMP_VALUE4:%.+]] = affine.max #[[$MAP5]]([[INSERT_SIZE_W]], [[TMP_VALUE5]])[[[DIM_W_1]]]
    // CHECK:                  [[PAD1_HIGH_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE4]]]

    // CHECK:                  [[DIM_W_2:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x4x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  [[SLICE_OFFSET_W:%.+]] = affine.max #[[$MAP2]]([[TMP_VALUE5]])
    // CHECK:                  [[TMP_VALUE1:%.+]] = affine.max #[[$MAP3]]([[TMP_VALUE5]])
    // CHECK:                  [[PAD0_LOW_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE1]]]
    // CHECK:                  [[TMP_VALUE0:%.+]] = affine.max #[[$MAP6]]([[SLICE_OFFSET_W]], [[INSERT_SIZE_W]], [[PAD1_LOW_W]], [[PAD1_HIGH_W]])[[[DIM_W_2]]]
    // CHECK:                  [[PAD0_HIGH_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE0]]]
    // CHECK:                  [[SLICE_SIZE_W:%.+]] = affine.apply #[[$MAP7]]([[PAD0_LOW_W]], [[PAD0_HIGH_W]], [[INSERT_SIZE_W]], [[PAD1_LOW_W]], [[PAD1_HIGH_W]])

    // CHECK:                  [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[SLICE_OFFSET_W]]] [1, 4, 540, [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:             tensor<1x4x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x4x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 35]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, 1, [[PAD0_LOW_W]]] high[0, 0, 1, [[PAD0_HIGH_W]]] {
    // CHECK:                  tensor.yield [[PAD_VALUE]] : f16
    // CHECK:                  tensor<1x4x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 540, 35]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x4x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 542, 37]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  [[CONV0:%.+]]  = VPU.NCE.CompressConvolution([[PAD0]]
    // CHECK:                  [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[CONV0]]
    // CHECK:                  [[PAD1:%.+]] = tensor.pad [[DWCONV]] low[0, 0, 1, [[PAD1_LOW_W]]] high[0, 0, 1, [[PAD1_HIGH_W]]] {
    // CHECK:                  tensor.yield [[PAD_VALUE]] : f16
    // CHECK:                  tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 35]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 37]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    // CHECK:                  [[INSERT:%.+]] = tensor.insert_slice [[CONV1]] into [[LOOP_OUT_W]][0, 0, 0, [[LOOP_ITER_W]]] [1, 32, 540, [[INSERT_SIZE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:             tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 35]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  scf.yield [[INSERT]]

    // CHECK: return [[LOOP_W]] : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[$QELEMTYPE:.*]] = !quant.uniform<u8:f16, 0.0013198380376778396:130>
// CHECK: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, {{[0-9]+}})>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, {{[0-9]+}})>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$MAP3:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$MAP4:.*]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$MAP5:.*]] = affine_map<(d0, d1)[s0] -> (0, d0 + d1 - s0 + 2)>
// CHECK: #[[$MAP6:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>


// CHECK-LABEL: @Merge2DVFChainAvgPool
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x16x?x?x[[$QELEMTYPE]], {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
func.func @Merge2DVFChainAvgPool(%arg0: tensor<1x16x?x?x!quant.uniform<u8:f16, 0.0013198380376778396:130>, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}> {
  %cst = const.Declare tensor<16x16x3x3x!quant.uniform<u8:f16, 0.0060863536946913774:128>, {order = #NHWC}> = dense<1> : tensor<16x16x3x3xui8, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [16, 16, 3, 3]>, #const.CastElemType<!quant.uniform<u8:f16, 0.0060863536946913774:128>>]
  %19 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.290000e+02 : f64, clamp_high = 1.260000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.290000e+02 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 5, 2]} : tensor<1x16x?x?x!quant.uniform<u8:f16, 0.0013198380376778396:130>, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x3x3x!quant.uniform<u8:f16, 0.0060863536946913774:128>, {order = #NHWC}> -> tensor<1x16x?x?x!quant.uniform<u8:f16, 0.0029174812868529676:129>, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
  %21 = VPU.NCE.AveragePool(%19) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 0.0029174812868529676 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1], tilingStrategy = [1, 1, 63, 2]} -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
  %24 = VPU.NCE.Eltwise(%21, %21) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 32, 2]} -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
  return %24 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-DAG: [[I8_CAST:%.+]] = arith.constant -126 : i8
    // CHECK-DAG: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[DIM_INDEX_H:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant [[W_STEP:[0-9]+]] : index
    // CHECK-DAG: [[LOOP_STEP_H:%.+]] = arith.constant [[H_STEP:[0-9]+]] : index
    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x16x?x?x[[$QELEMTYPE]], {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x16x?x?x[[$QELEMTYPE]], {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_END_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x16x?x?x[[$QELEMTYPE]], {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[LOOP_END_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x16x?x?x[[$QELEMTYPE]], {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: [[LOOP_H:%.+]] = scf.for
    // CHECK-SAME:             [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    // CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                  [[LOOP_W:%.+]] = scf.for
    // CHECK-SAME:             [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    // CHECK-SAME:             iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                  [[INSERT_SIZE_H:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER_H]])[[[LOOP_END_H]]]
    // CHECK:                  [[INSERT_SIZE_W:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_W]])[[[LOOP_END_W]]]

    // CHECK:                  [[DIM_H_1:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x16x?x?x[[$QELEMTYPE]], {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  [[DIM_W_1:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x16x?x?x[[$QELEMTYPE]], {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  [[SLICE_OFFSET_H:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER_H]])
    // CHECK:                  [[TMP_VALUE6:%.+]] = affine.max #[[$MAP3]]([[LOOP_ITER_H]])
    // CHECK:                  [[PAD_LOW_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE6]]]
    // CHECK:                  [[TMP_VALUE9:%.+]] = affine.max #[[$MAP5]]([[INSERT_SIZE_H]], [[SLICE_OFFSET_H]])[[[DIM_H_1]]]
    // CHECK:                  [[PAD_HIGH_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE9]]]
    // CHECK:                  [[SLICE_SIZE_H:%.+]] = affine.apply #[[$MAP6]]([[INSERT_SIZE_H]], [[PAD_LOW_H]], [[PAD_HIGH_H]])

    // CHECK:                  [[SLICE_OFFSET_W:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER_W]])
    // CHECK:                  [[TMP_VALUE8:%.+]] = affine.max #[[$MAP3]]([[LOOP_ITER_W]])
    // CHECK:                  [[PAD_LOW_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE8]]]
    // CHECK:                  [[TMP_VALUE4:%.+]] = affine.max #[[$MAP5]]([[INSERT_SIZE_W]], [[SLICE_OFFSET_W]])[[[DIM_W_1]]]
    // CHECK:                  [[PAD_HIGH_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE4]]]
    // CHECK:                  [[SLICE_SIZE_W:%.+]] = affine.apply #[[$MAP6]]([[INSERT_SIZE_W]], [[PAD_LOW_W]], [[PAD_HIGH_W]])

    // CHECK:                  [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET_H]], [[SLICE_OFFSET_W]]] [1, 16, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:             tensor<1x16x?x?x[[$QELEMTYPE]], {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?x[[$QELEMTYPE]], {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], [[W_STEP]]]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:                  [[PAD_VALUE:%.+]] = builtin.unrealized_conversion_cast [[I8_CAST]] : i8 to [[$QELEMTYPE]]
    // CHECK:                  [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW_H]], [[PAD_LOW_W]]] high[0, 0, [[PAD_HIGH_H]], [[PAD_HIGH_W]]] {
    // CHECK:                  tensor.yield [[PAD_VALUE]] : [[$QELEMTYPE]]
    // CHECK:                  tensor<1x16x?x?x[[$QELEMTYPE]], {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], [[W_STEP]]]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?x[[$QELEMTYPE]], {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP_2:[0-9]+]], [[W_STEP_2:[0-9]+]]]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  [[CONV:%.+]]  = VPU.NCE.Convolution([[PAD0]]
    // CHECK:                  [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[CONV]]
    // CHECK:                  [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[AVGPOOL]]
    // CHECK:                  [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, [[INSERT_SIZE_H]], [[INSERT_SIZE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:             tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, [[H_STEP]], [[W_STEP]]]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                  scf.yield [[INSERT]]

    // CHECK:  scf.yield [[LOOP_W]]

    // CHECK: return [[LOOP_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
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

// CHECK-LABEL: @MergeL2Reduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x512xf16>
func.func @MergeL2Reduce(%arg0: tensor<1x32x175x512xf16>) -> tensor<1x1x175x512xf16, {order = #NHWC}> {
   %0 = VPU.NCE.Permute(%arg0) {
     dstElemType = f16, dstOrder = #NHWC, expandedChannels = 32 : i64,
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 2]
   } -> tensor<1x32x175x512xf16, {order = #NHWC}>
   %1 = VPU.ReduceL2(%0) {
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     axes_value = [1],
     keep_dims,
     tilingStrategy = [1, 1, 1, 2]
   } : tensor<1x32x175x512xf16, {order = #NHWC}> -> tensor<1x1x175x512xf16, {order = #NHWC}>
   return %1 : tensor<1x1x175x512xf16, {order = #NHWC}>

// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C512:%.+]] = arith.constant 512 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        [[SCF:%.+]] = scf.for [[ITER:%.+]] = [[C0]] to [[C512]] step [[C128]] iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x1x175x512xf16, {order = #NHWC}>) {
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[ITER]]] [1, 32, 175, 128] [1, 1, 1, 1] : tensor<1x32x175x512xf16> to tensor<1x32x175x128xf16>
// CHECK-NEXT:      [[PERM:%.+]] = VPU.NCE.Permute([[SLICE]])
// CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceL2([[PERM]]) {axes_value = [1], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>} : tensor<1x32x175x128xf16, {order = #NHWC}> -> tensor<1x1x175x128xf16, {order = #NHWC}>
// CHECK-NEXT:      [[INSERT_SLICE:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, 0, [[ITER]]] [1, 1, 175, 128] [1, 1, 1, 1] : tensor<1x1x175x128xf16, {order = #NHWC}> into tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK-NEXT:      scf.yield [[INSERT_SLICE]] : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        return [[SCF]] : tensor<1x1x175x512xf16, {order = #NHWC}>
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


// CHECK-LABEL: @MergeL1Reduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x512xf16>
func.func @MergeL1Reduce(%arg0: tensor<1x32x175x512xf16>) -> tensor<1x1x175x512xf16, {order = #NHWC}> {
   %0 = VPU.NCE.Permute(%arg0) {
     dstElemType = f16, dstOrder = #NHWC, expandedChannels = 32 : i64,
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 2]
   } -> tensor<1x32x175x512xf16, {order = #NHWC}>
   %1 = VPU.ReduceL1(%0) {
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     axes_value = [1],
     keep_dims,
     tilingStrategy = [1, 1, 1, 2]
   } : tensor<1x32x175x512xf16, {order = #NHWC}> -> tensor<1x1x175x512xf16, {order = #NHWC}>
   return %1 : tensor<1x1x175x512xf16, {order = #NHWC}>

// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C512:%.+]] = arith.constant 512 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        [[SCF:%.+]] = scf.for [[ITER:%.+]] = [[C0]] to [[C512]] step [[C128]] iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x1x175x512xf16, {order = #NHWC}>) {
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[ITER]]] [1, 32, 175, 128] [1, 1, 1, 1] : tensor<1x32x175x512xf16> to tensor<1x32x175x128xf16>
// CHECK-NEXT:      [[PERM:%.+]] = VPU.NCE.Permute([[SLICE]])
// CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceL1([[PERM]]) {axes_value = [1], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>} : tensor<1x32x175x128xf16, {order = #NHWC}> -> tensor<1x1x175x128xf16, {order = #NHWC}>
// CHECK-NEXT:      [[INSERT_SLICE:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, 0, [[ITER]]] [1, 1, 175, 128] [1, 1, 1, 1] : tensor<1x1x175x128xf16, {order = #NHWC}> into tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK-NEXT:      scf.yield [[INSERT_SLICE]] : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        return [[SCF]] : tensor<1x1x175x512xf16, {order = #NHWC}>
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

// CHECK-LABEL: @MergeMeanReduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x512xf16>
func.func @MergeMeanReduce(%arg0: tensor<1x32x175x512xf16>) -> tensor<1x1x175x512xf16, {order = #NHWC}> {
   %0 = VPU.NCE.Permute(%arg0) {
     dstElemType = f16, dstOrder = #NHWC, expandedChannels = 32 : i64,
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 2]
   } -> tensor<1x32x175x512xf16, {order = #NHWC}>
   %1 = VPU.ReduceMean(%0) {
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     axes_value = [1],
     keep_dims,
     tilingStrategy = [1, 1, 1, 2]
   } : tensor<1x32x175x512xf16, {order = #NHWC}> -> tensor<1x1x175x512xf16, {order = #NHWC}>
   return %1 : tensor<1x1x175x512xf16, {order = #NHWC}>

// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C512:%.+]] = arith.constant 512 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        [[SCF:%.+]] = scf.for [[ITER:%.+]] = [[C0]] to [[C512]] step [[C128]] iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x1x175x512xf16, {order = #NHWC}>) {
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[ITER]]] [1, 32, 175, 128] [1, 1, 1, 1] : tensor<1x32x175x512xf16> to tensor<1x32x175x128xf16>
// CHECK-NEXT:      [[PERM:%.+]] = VPU.NCE.Permute([[SLICE]])
// CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceMean([[PERM]]) {axes_value = [1], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>} : tensor<1x32x175x128xf16, {order = #NHWC}> -> tensor<1x1x175x128xf16, {order = #NHWC}>
// CHECK-NEXT:      [[INSERT_SLICE:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, 0, [[ITER]]] [1, 1, 175, 128] [1, 1, 1, 1] : tensor<1x1x175x128xf16, {order = #NHWC}> into tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK-NEXT:      scf.yield [[INSERT_SLICE]] : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        return [[SCF]] : tensor<1x1x175x512xf16, {order = #NHWC}>
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

// CHECK-LABEL: @MergeMaxReduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x512xf16>
func.func @MergeMaxReduce(%arg0: tensor<1x32x175x512xf16>) -> tensor<1x1x175x512xf16, {order = #NHWC}> {
   %0 = VPU.NCE.Permute(%arg0) {
     dstElemType = f16, dstOrder = #NHWC, expandedChannels = 32 : i64,
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 2]
   } -> tensor<1x32x175x512xf16, {order = #NHWC}>
   %1 = VPU.ReduceMax(%0) {
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     axes_value = [1],
     keep_dims,
     tilingStrategy = [1, 1, 1, 2]
   } : tensor<1x32x175x512xf16, {order = #NHWC}> -> tensor<1x1x175x512xf16, {order = #NHWC}>
   return %1 : tensor<1x1x175x512xf16, {order = #NHWC}>

// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C512:%.+]] = arith.constant 512 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        [[SCF:%.+]] = scf.for [[ITER:%.+]] = [[C0]] to [[C512]] step [[C128]] iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x1x175x512xf16, {order = #NHWC}>) {
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[ITER]]] [1, 32, 175, 128] [1, 1, 1, 1] : tensor<1x32x175x512xf16> to tensor<1x32x175x128xf16>
// CHECK-NEXT:      [[PERM:%.+]] = VPU.NCE.Permute([[SLICE]])
// CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceMax([[PERM]]) {axes_value = [1], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>} : tensor<1x32x175x128xf16, {order = #NHWC}> -> tensor<1x1x175x128xf16, {order = #NHWC}>
// CHECK-NEXT:      [[INSERT_SLICE:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, 0, [[ITER]]] [1, 1, 175, 128] [1, 1, 1, 1] : tensor<1x1x175x128xf16, {order = #NHWC}> into tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK-NEXT:      scf.yield [[INSERT_SLICE]] : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        return [[SCF]] : tensor<1x1x175x512xf16, {order = #NHWC}>
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

// CHECK-LABEL: @MergeSumReduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x512xf16>
func.func @MergeSumReduce(%arg0: tensor<1x32x175x512xf16>) -> tensor<1x1x175x512xf16, {order = #NHWC}> {
   %0 = VPU.NCE.Permute(%arg0) {
     dstElemType = f16, dstOrder = #NHWC, expandedChannels = 32 : i64,
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 2]
   } -> tensor<1x32x175x512xf16, {order = #NHWC}>
   %1 = VPU.ReduceSum(%0) {
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     axes_value = [1],
     keep_dims,
     tilingStrategy = [1, 1, 1, 2]
   } : tensor<1x32x175x512xf16, {order = #NHWC}> -> tensor<1x1x175x512xf16, {order = #NHWC}>
   return %1 : tensor<1x1x175x512xf16, {order = #NHWC}>

// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C512:%.+]] = arith.constant 512 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        [[SCF:%.+]] = scf.for [[ITER:%.+]] = [[C0]] to [[C512]] step [[C128]] iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x1x175x512xf16, {order = #NHWC}>) {
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[ITER]]] [1, 32, 175, 128] [1, 1, 1, 1] : tensor<1x32x175x512xf16> to tensor<1x32x175x128xf16>
// CHECK-NEXT:      [[PERM:%.+]] = VPU.NCE.Permute([[SLICE]])
// CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceSum([[PERM]]) {axes_value = [1], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>} : tensor<1x32x175x128xf16, {order = #NHWC}> -> tensor<1x1x175x128xf16, {order = #NHWC}>
// CHECK-NEXT:      [[INSERT_SLICE:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, 0, [[ITER]]] [1, 1, 175, 128] [1, 1, 1, 1] : tensor<1x1x175x128xf16, {order = #NHWC}> into tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK-NEXT:      scf.yield [[INSERT_SLICE]] : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        return [[SCF]] : tensor<1x1x175x512xf16, {order = #NHWC}>
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

// CHECK-LABEL: @MergeLogicalOrReduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x512xf16>
func.func @MergeLogicalOrReduce(%arg0: tensor<1x32x175x512xf16>) -> tensor<1x1x175x512xf16, {order = #NHWC}> {
   %0 = VPU.NCE.Permute(%arg0) {
     dstElemType = f16, dstOrder = #NHWC, expandedChannels = 32 : i64,
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 2]
   } -> tensor<1x32x175x512xf16, {order = #NHWC}>
   %1 = VPU.ReduceLogicalOr(%0) {
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     axes_value = [1],
     keep_dims,
     tilingStrategy = [1, 1, 1, 2]
   } : tensor<1x32x175x512xf16, {order = #NHWC}> -> tensor<1x1x175x512xf16, {order = #NHWC}>
   return %1 : tensor<1x1x175x512xf16, {order = #NHWC}>

// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C512:%.+]] = arith.constant 512 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        [[SCF:%.+]] = scf.for [[ITER:%.+]] = [[C0]] to [[C512]] step [[C128]] iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x1x175x512xf16, {order = #NHWC}>) {
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[ITER]]] [1, 32, 175, 128] [1, 1, 1, 1] : tensor<1x32x175x512xf16> to tensor<1x32x175x128xf16>
// CHECK-NEXT:      [[PERM:%.+]] = VPU.NCE.Permute([[SLICE]])
// CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceLogicalOr([[PERM]]) {axes_value = [1], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>} : tensor<1x32x175x128xf16, {order = #NHWC}> -> tensor<1x1x175x128xf16, {order = #NHWC}>
// CHECK-NEXT:      [[INSERT_SLICE:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, 0, [[ITER]]] [1, 1, 175, 128] [1, 1, 1, 1] : tensor<1x1x175x128xf16, {order = #NHWC}> into tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK-NEXT:      scf.yield [[INSERT_SLICE]] : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        return [[SCF]] : tensor<1x1x175x512xf16, {order = #NHWC}>
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

// CHECK-LABEL: @MergeLogicalAndReduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x512xf16>
func.func @MergeLogicalAndReduce(%arg0: tensor<1x32x175x512xf16>) -> tensor<1x1x175x512xf16, {order = #NHWC}> {
   %0 = VPU.NCE.Permute(%arg0) {
     dstElemType = f16, dstOrder = #NHWC, expandedChannels = 32 : i64,
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
     clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64,
     prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64,
     adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 2]
   } -> tensor<1x32x175x512xf16, {order = #NHWC}>
   %1 = VPU.ReduceLogicalAnd(%0) {
     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
     axes_value = [1],
     keep_dims,
     tilingStrategy = [1, 1, 1, 2]
   } : tensor<1x32x175x512xf16, {order = #NHWC}> -> tensor<1x1x175x512xf16, {order = #NHWC}>
   return %1 : tensor<1x1x175x512xf16, {order = #NHWC}>

// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C512:%.+]] = arith.constant 512 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        [[SCF:%.+]] = scf.for [[ITER:%.+]] = [[C0]] to [[C512]] step [[C128]] iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x1x175x512xf16, {order = #NHWC}>) {
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[ITER]]] [1, 32, 175, 128] [1, 1, 1, 1] : tensor<1x32x175x512xf16> to tensor<1x32x175x128xf16>
// CHECK-NEXT:      [[PERM:%.+]] = VPU.NCE.Permute([[SLICE]])
// CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceLogicalAnd([[PERM]]) {axes_value = [1], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>} : tensor<1x32x175x128xf16, {order = #NHWC}> -> tensor<1x1x175x128xf16, {order = #NHWC}>
// CHECK-NEXT:      [[INSERT_SLICE:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, 0, [[ITER]]] [1, 1, 175, 128] [1, 1, 1, 1] : tensor<1x1x175x128xf16, {order = #NHWC}> into tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK-NEXT:      scf.yield [[INSERT_SLICE]] : tensor<1x1x175x512xf16, {order = #NHWC}>
// CHECK:        return [[SCF]] : tensor<1x1x175x512xf16, {order = #NHWC}>
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

// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (-d0 + 512, 171)>

// CHECK-LABEL: @MergeSquareReduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x128x512xf16, {order = #NHWC}>
func.func @MergeSquareReduce(%arg0: tensor<1x64x128x512xf16, {order = #NHWC}>) -> tensor<1x1x128x512xf16> {
   %0 = VPU.LayoutCast(%arg0) {
       dst_order = #NCHW,
       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
       tilingStrategy = [1, 1, 1, 2]
   } : tensor<1x64x128x512xf16, {order = #NHWC}> -> tensor<1x64x128x512xf16>
   %1 = VPU.ReduceSquare(%0) {
       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
       axes_value = [1],
       keep_dims,
       tilingStrategy = [1, 1, 1, 2]
   } : tensor<1x64x128x512xf16> -> tensor<1x1x128x512xf16>
   return %1 : tensor<1x1x128x512xf16>

// CHECK-DAG:    [[C171:%.+]] = arith.constant 171 : index
// CHECK-DAG:    [[C512:%.+]] = arith.constant 512 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x128x512xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[ITER:%.+]] = [[C0]] to [[C512]] step [[C171]] iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x1x128x512xf16>) {
// CHECK-NEXT:      [[MIN:%.+]] = affine.min #[[$MAP1]]([[ITER]])
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[ITER]]] [1, 64, 128, [[MIN]]] [1, 1, 1, 1] : tensor<1x64x128x512xf16, {order = #NHWC}> to tensor<1x64x128x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 128, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      [[LC:%.+]] = VPU.LayoutCast([[SLICE]]) {dst_order = #NCHW, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>} : tensor<1x64x128x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 128, 512]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x64x128x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 128, 512]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceSquare([[LC]]) {axes_value = [1], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>} : tensor<1x64x128x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 128, 512]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x128x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 512]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-NEXT:      [[INSERT_SLICE:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, 0, [[ITER]]] [1, 1, 128, [[MIN]]] [1, 1, 1, 1] : tensor<1x1x128x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 512]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x128x512xf16>
// CHECK-NEXT:      scf.yield [[INSERT_SLICE]] : tensor<1x1x128x512xf16>
// CHECK:        return [[SCF]] : tensor<1x1x128x512xf16>
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

// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (-d0 + 512, 171)>

// CHECK-LABEL: @MergeProdReduce
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x64x128x512xf16, {order = #NHWC}>
func.func @MergeProdReduce(%arg0: tensor<1x64x128x512xf16, {order = #NHWC}>) -> tensor<1x1x128x512xf16> {
   %0 = VPU.LayoutCast(%arg0) {
       dst_order = #NCHW,
       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
       tilingStrategy = [1, 1, 1, 2]
   } : tensor<1x64x128x512xf16, {order = #NHWC}> -> tensor<1x64x128x512xf16>
   %1 = VPU.ReduceProd(%0) {
       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
       axes_value = [1],
       keep_dims,
       tilingStrategy = [1, 1, 1, 2]
   } : tensor<1x64x128x512xf16> -> tensor<1x1x128x512xf16>
   return %1 : tensor<1x1x128x512xf16>

// CHECK-DAG:    [[C171:%.+]] = arith.constant 171 : index
// CHECK-DAG:    [[C512:%.+]] = arith.constant 512 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x128x512xf16>
// CHECK:        [[SCF:%.+]] = scf.for [[ITER:%.+]] = [[C0]] to [[C512]] step [[C171]] iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x1x128x512xf16>) {
// CHECK-NEXT:      [[MIN:%.+]] = affine.min #[[$MAP1]]([[ITER]])
// CHECK-NEXT:      [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[ITER]]] [1, 64, 128, [[MIN]]] [1, 1, 1, 1] : tensor<1x64x128x512xf16, {order = #NHWC}> to tensor<1x64x128x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 128, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-NEXT:      [[LC:%.+]] = VPU.LayoutCast([[SLICE]]) {dst_order = #NCHW, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>} : tensor<1x64x128x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 128, 512]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x64x128x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 128, 512]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-NEXT:      [[REDUCE:%.+]] = VPU.ReduceProd([[LC]]) {axes_value = [1], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>} : tensor<1x64x128x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 128, 512]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x128x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 512]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-NEXT:      [[INSERT_SLICE:%.+]] = tensor.insert_slice [[REDUCE]] into [[OUT]][0, 0, 0, [[ITER]]] [1, 1, 128, [[MIN]]] [1, 1, 1, 1] : tensor<1x1x128x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 512]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x128x512xf16>
// CHECK-NEXT:      scf.yield [[INSERT_SLICE]] : tensor<1x1x128x512xf16>
// CHECK:        return [[SCF]] : tensor<1x1x128x512xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 34)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 floordiv 2)>

// CHECK-LABEL: @NCEPoolWithUnpaddedOutputChannels
module @NCEPoolWithUnpaddedOutputChannels {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
        config.Option @config.AutoPaddingIDU : true
    }
    func.func @ApplyTilingPaddedAvgPool(%arg0: tensor<1x12x1079x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 319]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 2158, 638]> : tensor<4xsi64>, order = #NHWC}> {
        %0 = VPU.Expand(%arg0) {
            pads_begin = [0, 0, 0, 0],
            pads_end = [0, 4, 0, 0]
        } : tensor<1x12x1079x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 319]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x1079x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1079, 319]> : tensor<4xsi64>, order = #NHWC}>
        %1 = VPU.NCE.AveragePool(%0) {
            input_padding = [0, 4, 0, 0],
            kernel_size = [1, 1],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.280000e+02 : f64, clamp_high = 1.270000e+02 : f64, scale = 1.275000e+02 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.280000e+02 : f64>,
            strides = [1, 1],
            tilingStrategy = [1, 1, 1, 4]
        } -> tensor<1x12x1079x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 319]> : tensor<4xsi64>, order = #NHWC}>
        %2 = VPU.DepthToSpace(%1) {
            block_size = 2 : i64,
            mode = #IE.depth_to_space_mode<DEPTH_FIRST>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            padded_channels = #IE.ChannelPadding<input = 0 : i64, output = 13 : i64>,
            tilingStrategy = [1, 1, 3, 2]
        } : tensor<1x12x1079x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 319]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 2158, 638]> : tensor<4xsi64>, order = #NHWC}>
        return %2 : tensor<1x16x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 2158, 638]> : tensor<4xsi64>, order = #NHWC}>

        // CHECK-DAG: [[DIM_IND:%.+]] = arith.constant 3 : index
        // CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant 34 : index
        // CHECK-DAG: [[LOOP_START_W:%.+]] = arith.constant 0 : index
        // CHECK-DAG: [[ELEM_BTW:%.+]] = arith.constant 2 : index

        // CHECK: [[W_RAW:%.+]] = tensor.dim [[INPUT:%.+]], [[DIM_IND]] : tensor<1x12x1079x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 319]> : tensor<4xsi64>, order = #NHWC}>
        // CHECK: [[W_DIM:%.+]] = arith.muli [[W_RAW]], [[ELEM_BTW]] : index
        // CHECK: [[OUTPUT:%.+]] = tensor.empty([[W_DIM]]) : tensor<1x16x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 2158, 638]> : tensor<4xsi64>, order = #NHWC}>
        // CHECK: [[W_RAW_LOOP:%.+]] = tensor.dim [[INPUT]], [[DIM_IND]] : tensor<1x12x1079x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 319]> : tensor<4xsi64>, order = #NHWC}>
        // CHECK: [[LOOP_END_W:%.+]] = arith.muli [[W_RAW_LOOP]], [[ELEM_BTW]] : index
        // CHECK: [[LOOP:%.+]] = scf.for [[W_ITER:%.+]] = [[LOOP_START_W]] to [[LOOP_END_W]] step [[LOOP_STEP_W]] iter_args([[LOOP_OUT:%.+]] = [[OUTPUT]]) -> (tensor<1x16x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 2158, 638]> : tensor<4xsi64>, order = #NHWC}>) {
        // CHECK:   [[TILE_W:%.+]] = affine.min #[[$MAP]]([[W_ITER]])[[[LOOP_END_W]]]
        // CHECK:   [[W_OFFSET:%.+]] = affine.apply #[[$MAP1]]([[W_ITER]])
        // CHECK:   [[TILE_W_DIV:%.+]] = affine.apply #[[$MAP1]]([[TILE_W]])
        // CHECK:   [[SLICE0:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[W_OFFSET]]] [1, 12, 1079, [[TILE_W_DIV]]] [1, 1, 1, 1] : tensor<1x12x1079x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 319]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x12x1079x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 17]> : tensor<4xsi64>, order = #NHWC}>
        // CHECK:   [[EXP_TILE:%.+]] = VPU.Expand([[SLICE0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]
        // CHECK:   [[POOL_TILE:%.+]] = VPU.NCE.AveragePool([[EXP_TILE]]) {input_padding = [0, 4, 0, 0], kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.280000e+02 : f64, clamp_high = 1.270000e+02 : f64, scale = 1.275000e+02 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.280000e+02 : f64>, strides = [1, 1]} -> tensor<1x12x1079x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 17]> : tensor<4xsi64>, order = #NHWC}>
        // CHECK:   [[D2S:%.+]] = VPU.DepthToSpace([[POOL_TILE]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, padded_channels = #IE.ChannelPadding<input = 0 : i64, output = 13 : i64>} : tensor<1x12x1079x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 17]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 2158, 34]> : tensor<4xsi64>, order = #NHWC}>
        // CHECK:   [[INSERT:%.+]] = tensor.insert_slice [[D2S]] into [[LOOP_OUT]][0, 0, 0, [[W_ITER]]] [1, 16, 2158, [[TILE_W]]] [1, 1, 1, 1] : tensor<1x16x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 2158, 34]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 2158, 638]> : tensor<4xsi64>, order = #NHWC}>
        // CHECK:   scf.yield [[INSERT]] : tensor<1x16x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 2158, 638]> : tensor<4xsi64>, order = #NHWC}>
        // CHECK: }
        // CHECK: return [[LOOP]] : tensor<1x16x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 2158, 638]> : tensor<4xsi64>, order = #NHWC}>

    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
    config.Resources 3 of @NCE at 2.100000e+03 MHz {
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }

    func.func @Merge2DVFSkipConnection(%arg0: tensor<1x32x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x32x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NCHW}> {
    %cst_2 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 32 : i64>, #const.Reshape<[32, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<128x128x3x3xf16, {order = #NHWC}> = dense<0> : tensor<128x128x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<i8:f16, -513.75686274509803:-1>>, #const.ConvertElemType<!quant.uniform<u8:f16, -513.75686274509803:127>>, #const.Dequantize, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<0.0> : tensor<128x32x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    %cst_5 = const.Declare tensor<128x1x1x384xi1> = dense<0.0> : tensor<128x32x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    %0 = VPU.GroupSparseTensor(%cst_4, %cst_5) {
            is_weights,
            sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64,
            numElems = dense<0> : tensor<128xi64>,
            alignment = 16 : i64>}
                -> !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<128xi64>, alignment = 16 : i64>>
    %1 = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, tilingStrategy = [1, 1, 2, 960]} :
            tensor<1x32x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NCHW}>
                -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NCHW}>
    %2 = VPU.NCE.Permute(%1) {
            dstElemType = f16,
            dstOrder = #NHWC,
            expandedChannels = 32 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 5.000000e-01 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>,
            tilingStrategy = [1, 1, 6, 3]}
                -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.NCE.DepthConvolution(%2, %cst_2) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            prelu_alpha = [1.000000e+00],
            adder = 0.000000e+00 : f64>,

            strides = [1, 1],
            tilingStrategy = [1, 1, 6, 3]}
                -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    %4 = VPU.NCE.Convolution(%3, %0) rawFilterShape [128, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>,

            strides = [2, 2],
            tilingStrategy = [1, 1, 6, 3]}
                : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>,
                !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<128xi64>, alignment = 16 : i64>>
                    -> tensor<1x128x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 270, 480]> : tensor<4xsi64>, order = #NHWC}>
    %5 = VPU.NCE.Convolution(%4, %cst_3) rawFilterShape [128, 128, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64,
            right = 1 : i64,
            top = 1 : i64,
            bottom = 1 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>,

            strides = [1, 1],
            tilingStrategy = [1, 1, 8, 3]}
                : tensor<1x128x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 270, 480]> : tensor<4xsi64>, order = #NHWC}>,
                tensor<128x128x3x3xf16, {order = #NHWC}>
                    -> tensor<1x128x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 270, 480]> : tensor<4xsi64>, order = #NHWC}>
    %6 = VPU.DepthToSpace(%5) {
            block_size = 2 : i64,
            mode = #IE.depth_to_space_mode<DEPTH_FIRST>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            tilingStrategy = [1, 1, 9, 2]}
                : tensor<1x128x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 270, 480]> : tensor<4xsi64>, order = #NHWC}>
                    -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    %7 = VPU.NCE.Eltwise(%6, %3) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>,
            tilingStrategy = [1, 1, 11, 3]}
                -> tensor<1x32x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NCHW}>
    return %7 : tensor<1x32x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-LABEL: @Merge2DVFSkipConnection
    // CHECK-SAME:    [[ARG0:%arg[0-9]+]]: tensor<1x32x?x?xf32

    // Reduced-fusion expectation is intentional and matches default-VF semantics.
    // The DepthConvolution is the producer of the main tiled chain (stridedConv -> conv ->
    // depthToSpace) AND the source of the skip connection consumed by the final Eltwise, so it has
    // multiple users. In collectTiledAndFusedOps the BFS only fuses a producer when every one of its
    // users is part of the fused set (checkProducersUsers). Because DepthConvolution has a user (the
    // skip Eltwise) outside the first tiled chain, it cannot be fully fused into a single tiled loop.
    // It is therefore kept in its own loop, feeding two separate scf.for regions, and the Eltwise is
    // applied after the second loop. The previous single-loop form fused a multi-user producer that
    // the aligned compatibility checks now correctly keep out.

    // CHECK:       [[DEPTH_CONVOLUTION:%.*]] = scf.for
    // CHECK:         scf.for
    // CHECK:           tensor.extract_slice
    // CHECK:           VPU.Convert
    // CHECK:           VPU.NCE.Permute
    // CHECK:           VPU.NCE.DepthConvolution
    // CHECK:           tensor.insert_slice
    // CHECK:         scf.yield
    // CHECK:       scf.yield

    // CHECK:       [[RESULT:%.*]] = scf.for
    // CHECK:         scf.for
    // CHECK:           [[DATA_SLICE:%.*]] = tensor.extract_slice
    // CHECK:           [[PADDED:%.*]] = tensor.pad [[DATA_SLICE]]
    // CHECK:           [[CONV_1:%.*]] = VPU.NCE.Convolution([[PADDED]]
    // CHECK:           [[PADDED_2:%.*]] = tensor.pad [[CONV_1]]
    // CHECK:           [[CONV_2:%.*]] = VPU.NCE.Convolution([[PADDED_2]]
    // CHECK:           [[DEPTH_TO_SPACE:%.*]] = VPU.DepthToSpace([[CONV_2]]
    // CHECK:           tensor.insert_slice [[DEPTH_TO_SPACE]]
    // CHECK:         scf.yield
    // CHECK:       scf.yield
    // CHECK:       [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[RESULT]], [[DEPTH_CONVOLUTION]])

    // CHECK:       return [[ELTWISE]] : tensor<1x32x?x?xf32
    }
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
  config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!outputDynamicType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @MergeSlice
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
func.func @MergeSlice(
    %arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
  ) -> !outputDynamicType  {

    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
      = dense<1.0> : tensor<16x16x1x1xf32>,
        [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %cst_0 = const.Declare tensor<32x32x3x3x!quant.uniform<u8:f16, 0.0091306873396331187:128>, {order = #NHWC}>
      = dense<1> : tensor<32x32x3x3xsi8>,
        [#const.CastElemType<f16>,
         #const.CastElemType<!quant.uniform<i8:f16, 0.0091306873396331187>>,
         #const.ConvertElemType<!quant.uniform<u8:f16, 0.0091306873396331187:128>>,
         #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x1x1x32x!quant.uniform<u8:f16, 0.034925088695451328:128>, {order = #NHWC}>
      = dense<1> : tensor<32x3x3x3xsi8>,
        [#const.CastElemType<f16>,
         #const.CastElemType<!quant.uniform<i8:f16, 0.034925088695451328>>,
         #const.ConvertElemType<!quant.uniform<u8:f16, 0.034925088695451328:128>>,
         #const.Reorder<#NHWC>,
         #const.Reshape<[32, 1, 1, 27]>,
         #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 5]>]

    %depth = VPU.NCE.DepthConvolution(%arg0, %weights) rawFilterShape [16, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEInt<
        mode = <NOOP>,
        clamp_low = 0 : i64,
        clamp_high = 255 : i64,
        lrelu_mult = 1 : i64,
        lrelu_shift = 0 : i64,
        fp_prelu_alpha = 507.68865966796875 : f64>,

      strides = [1, 1],
      tilingStrategy = [1, 1, 2, 1]
    } -> tensor<1x16x?x?x!quant.uniform<u8:f16, 0.0019697112195632038>, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    %slice = VPU.Slice %depth [0, 0, 0, 0] [1, 4, -9223372036854775808, -9223372036854775808]
      : tensor<1x16x?x?x!quant.uniform<u8:f16, 0.0019697112195632038>, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
      to tensor<1x4x?x?x!quant.uniform<u8:f16, 0.0019697112195632038>, {bounds = #const.OpaqueI64Elements<[1, 4, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    %compress = VPU.NCE.CompressConvolution(%slice, %cst_2, %cst_1) rawFilterShape [32, 3, 3, 3] {
      cm_sp_pattern = 7 : i64,
      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
      pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,

      strides = [2, 2],
      tilingStrategy = [1, 1, 7, 3]
    } -> tensor<1x32x?x?x!quant.uniform<u8:f16, 0.030033121856988646:110>, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    %conv = VPU.NCE.Convolution(%compress, %cst_0) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
      ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,

      strides = [1, 1],
      tilingStrategy = [1, 1, 7, 3]
    } : tensor<1x32x?x?x!quant.uniform<u8:f16, 0.030033121856988646:110>, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>,
        tensor<32x32x3x3x!quant.uniform<u8:f16, 0.0091306873396331187:128>, {order = #NHWC}>
      -> !outputDynamicType

    return %conv : !outputDynamicType
}

// CHECK:       [[C3:%.+]] = arith.constant 3 : index
// CHECK:       [[C110:%.+]] = arith.constant 110 : i8
// CHECK:       [[C0_I8:%.+]] = arith.constant 0 : i8
// CHECK:       [[C160:%.+]] = arith.constant 160 : index
// CHECK:       [[C45:%.+]] = arith.constant 45 : index
// CHECK:       [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
// CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<32x32x3x3x!qElemType{{.*}}, {order = #NHWC}> = dense<1>
// CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1>
// CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<32x1x1x32x!qElemType{{.*}}, {order = #NHWC}> = dense<1>
// CHECK:       [[C2:%.+]] = arith.constant 2 : index
// CHECK:       [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]]
// CHECK:       [[DIV0:%.+]] = arith.divsi [[DIM]], [[C2]]
// CHECK:       [[DIM_0:%.+]] = tensor.dim [[ARG0]], [[C3]]
// CHECK:       [[DIV1:%.+]] = arith.divsi [[DIM_0]], [[C2]]
// CHECK:       [[EMPTY:%.+]] = tensor.empty([[DIV0]], [[DIV1]])
// CHECK:       [[DIM_1:%.+]] = tensor.dim [[ARG0]], [[C2]]
// CHECK:       [[DIV2:%.+]] = arith.divsi [[DIM_1]], [[C2]]
// CHECK:       [[DIM_2:%.+]] = tensor.dim [[ARG0]], [[C3]]
// CHECK:       [[DIV3:%.+]] = arith.divsi [[DIM_2]], [[C2]]
// CHECK:       [[LOOP_OUTER:%.+]] = scf.for [[IV0:%.+]] = [[C0]] to [[DIV2]] step [[C45]] iter_args([[ARG_OUTER:%.+]] = [[EMPTY]])
// CHECK:         [[LOOP_INNER:%.+]] = scf.for [[IV1:%.+]] = [[C0]] to [[DIV3]] step [[C160]] iter_args([[ARG_INNER:%.+]] = [[ARG_OUTER]])
// CHECK:           [[MIN0:%.+]] = affine.min
// CHECK:           [[MIN1:%.+]] = affine.min
// CHECK:           [[DIM_3:%.+]] = tensor.dim [[ARG0]], [[C2]]
// CHECK:           [[DIV4:%.+]] = arith.divsi [[DIM_3]], [[C2]]
// CHECK:           [[DIM_4:%.+]] = tensor.dim [[ARG0]], [[C3]]
// CHECK:           [[DIV5:%.+]] = arith.divsi [[DIM_4]], [[C2]]
// CHECK:           [[MAX0:%.+]] = affine.max
// CHECK:           [[MAX1:%.+]] = affine.max
// CHECK:           [[MIN2:%.+]] = affine.min
// CHECK:           [[MAX2:%.+]] = affine.max
// CHECK:           [[MIN3:%.+]] = affine.min
// CHECK:           [[MAX3:%.+]] = affine.max
// CHECK:           [[MAX4:%.+]] = affine.max
// CHECK:           [[MIN4:%.+]] = affine.min
// CHECK:           [[MAX5:%.+]] = affine.max
// CHECK:           [[MIN5:%.+]] = affine.min
// CHECK:           [[MAX6:%.+]] = affine.max
// CHECK:           [[MAX7:%.+]] = affine.max
// CHECK:           [[MIN6:%.+]] = affine.min
// CHECK:           [[MAX8:%.+]] = affine.max
// CHECK:           [[MAX9:%.+]] = affine.max
// CHECK:           [[MIN7:%.+]] = affine.min
// CHECK:           [[APPLY0:%.+]] = affine.apply
// CHECK:           [[APPLY1:%.+]] = affine.apply
// CHECK:           [[EXTRACT:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[MAX6]], [[MAX8]]] [1, 16, [[APPLY0]], [[APPLY1]]]
// CHECK:           [[DEPTH:%.+]] = VPU.NCE.DepthConvolution([[EXTRACT]], [[CST]]) rawFilterShape [16, 1, 1, 1] {
    // CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
// CHECK-SAME:        strides = [1, 1]
// CHECK:           [[SLICE:%.+]] = VPU.Slice [[DEPTH]] [0, 0, 0, 0] [1, 4, -9223372036854775808, -9223372036854775808]
// CHECK:           [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[C0_I8]]
// CHECK:           [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[MIN6]], [[MIN7]]] high[0, 0, 0, 0]
// CHECK:           [[COMPRESS:%.+]] = VPU.NCE.CompressConvolution([[PAD0]], [[CST_2]], [[CST_1]]) rawFilterShape [32, 3, 3, 3] {
    // CHECK-SAME:        cm_sp_pattern = 7
// CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
// CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
// CHECK-SAME:        strides = [2, 2]
// CHECK:           [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[C110]]
// CHECK:           [[PAD1:%.+]] = tensor.pad [[COMPRESS]] low[0, 0, [[MIN2]], [[MIN4]]] high[0, 0, [[MIN3]], [[MIN5]]]
// CHECK:           [[CONV:%.+]] = VPU.NCE.Convolution([[PAD1]], [[CST_0]]) rawFilterShape [32, 32, 3, 3] {
    // CHECK-SAME:        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>
// CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
// CHECK-SAME:        strides = [1, 1]
// CHECK:           [[INSERT:%.+]] = tensor.insert_slice [[CONV]] into [[ARG_INNER]][0, 0, [[IV0]], [[IV1]]] [1, 32, [[MIN0]], [[MIN1]]]
// CHECK:           scf.yield [[INSERT]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         scf.yield [[LOOP_INNER]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:       return [[LOOP_OUTER]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeConvertWithInterpolate
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
func.func @MergeConvertWithInterpolate(%arg0: tensor<1x32x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x32x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 128, 128]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {
            dstElemType = f16,
            tilingStrategy = [1, 1, 2, 1]
        } : tensor<1x32x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    %1 = VPU.Interpolate(%0) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>,
            scales_attr = [2.000000e+00, 2.000000e+00],
            tilingStrategy = [1, 1, 2, 1]
        } : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x32x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 128, 128]> : tensor<4xsi64>, order = #NHWC}>

    return %1 : tensor<1x32x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 128, 128]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 43 : index
    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[DIM_INDEX:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[SCALE:%.+]] = arith.constant 2.000000e+00 : f64

    // CHECK: [[DIM:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]]
    // CHECK: [[DIM_I64:%.+]] = arith.index_cast [[DIM]] : index to i64
    // CHECK: [[DIM_F64:%.+]] = arith.sitofp [[DIM_I64]] : i64 to f64
    // CHECK: [[OUT_DIM_F64:%.+]] = arith.mulf [[DIM_F64]], [[SCALE]] : f64
    // CHECK: [[OUT_DIM_I64:%.+]] = arith.fptosi [[OUT_DIM_F64]] : f64 to i64
    // CHECK: [[OUT_DIM:%.+]] = arith.index_cast [[OUT_DIM_I64]] : i64 to index

    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[OUT_DIM]]) : tensor<1x32x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 128, 128]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: [[DIM2:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]]
    // CHECK: [[DIM2_I64:%.+]] = arith.index_cast [[DIM2]] : index to i64
    // CHECK: [[DIM2_F64:%.+]] = arith.sitofp [[DIM2_I64]] : i64 to f64
    // CHECK: [[OUT_DIM2_F64:%.+]] = arith.mulf [[DIM2_F64]], [[SCALE]] : f64
    // CHECK: [[OUT_DIM2_I64:%.+]] = arith.fptosi [[OUT_DIM2_F64]] : f64 to i64
    // CHECK: [[LOOP_BOUND:%.+]] = arith.index_cast [[OUT_DIM2_I64]] : i64 to index

    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_BOUND]] step [[LOOP_STEP]]
    // CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 128, 128]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                [[OUT_TILE:%.+]] = affine.min {{.+}}([[LOOP_ITER]]){{\[}}[[LOOP_BOUND]]{{\]}}
    // CHECK:                [[IN_START_RAW:%.+]] = affine.max {{.+}}([[LOOP_ITER]])
    // CHECK:                [[IN_START:%.+]] = affine.min {{.+}}(){{\[}}[[IN_START_RAW]]{{\]}}
    // CHECK:                [[IN_END_RAW:%.+]] = affine.max {{.+}}([[LOOP_ITER]], [[OUT_TILE]])
    // CHECK:                [[IN_END:%.+]] = affine.min {{.+}}(){{\[}}[[IN_END_RAW]]{{\]}}
    // CHECK:                [[IN_SIZE:%.+]] = affine.apply {{.+}}([[IN_START]], [[IN_END]])
    // CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]{{\[}}0, 0, [[IN_START]], 0] [1, 32, [[IN_SIZE]], 64]
    // CHECK:                [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
    // CHECK:                [[INTERP:%.+]] = VPU.Interpolate([[CONVERT]], {{%.+}}, {{%.+}})
    // CHECK-SAME:           attr = #IE.Interpolate<{{.*}}mode = <NEAREST>{{.*}}>
    // CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[INTERP]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 32, [[OUT_TILE]], 128]
    // CHECK:                scf.yield [[INSERT]]

    // CHECK: return [[LOOP]] : tensor<1x32x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 128, 128]> : tensor<4xsi64>, order = #NHWC}>
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Gelu(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = !qElemType, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256x!qElemType>
    %1 = VPU.Dequantize(%0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256x!qElemType> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.MVN1Normalize(%0, %arg1) {across_channels = false, normalize_variance = true, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16>, tensor<1x16x1x2xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[MVN:%.+]] = VPU.MVN1Normalize([[CONVERT]], [[MEANVAR]]) {across_channels = false, normalize_variance = true}
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[MVN]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @MergeEltwiseSoftMax
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x16x256x140xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x16x256x140xf16, {order = #NHWC}>)
func.func @MergeEltwiseSoftMax(
        %arg0: tensor<1x16x256x140xf16, {order = #NHWC}>,
        %arg1: tensor<1x16x256x140xf16, {order = #NHWC}>
) -> tensor<1x16x256x140xf16> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
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
        tilingStrategy = [1, 1, 2, 1]
    } -> tensor<1x16x256x140xf16, {order = #NHWC}>

    %1 = VPU.LayoutCast(%0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x16x256x140xf16, {order = #NHWC}> -> tensor<1x16x256x140xf16>

    %2 = VPU.SoftMax(%1) {
        axisInd = 3 : i64,
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x16x256x140xf16> -> tensor<1x16x256x140xf16>

    return %2 : tensor<1x16x256x140xf16>

    // CHECK-DAG: [[C64:%.+]] = arith.constant 64 : index
    // CHECK-DAG: [[C256:%.+]] = arith.constant 256 : index
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x256x140xf16>
    // CHECK: [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C256]] step [[C64]] iter_args([[OUT:%.+]] = [[EMPTY]])
    // CHECK-SAME:     -> (tensor<1x16x256x140xf16>)
    // CHECK:     [[SLICE0:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, [[IDX]], 0] [1, 16, 64, 140] [1, 1, 1, 1]
    // CHECK:     [[SLICE1:%.+]] = tensor.extract_slice [[INPUT1]][0, 0, [[IDX]], 0] [1, 16, 64, 140] [1, 1, 1, 1]
    // CHECK:     [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE0]], [[SLICE1]])
    // CHECK:     [[CAST:%.+]] = VPU.LayoutCast([[ELTWISE]]) {dst_order = #NCHW}
    // CHECK:     [[SOFTMAX:%.+]] = VPU.SoftMax([[CAST]]) {axisInd = 3 : i64}
    // CHECK:     [[INSERT:%.+]] = tensor.insert_slice [[SOFTMAX]] into [[OUT]][0, 0, [[IDX]], 0] [1, 16, 64, 140] [1, 1, 1, 1]
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

// CHECK-LABEL: @MergeConvertSoftMax
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x256x140xf32>)
func.func @MergeConvertSoftMax(%arg0: tensor<1x16x256x140xf32>) -> tensor<1x16x256x140xf16> {
    %0 = VPU.Convert(%arg0) {
        dstElemType = f16,
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x16x256x140xf32> -> tensor<1x16x256x140xf16>

    %1 = VPU.SoftMax(%0) {
        axisInd = 3 : i64,
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x16x256x140xf16> -> tensor<1x16x256x140xf16>

    return %1 : tensor<1x16x256x140xf16>

    // CHECK-DAG: [[C43:%.+]] = arith.constant 43 : index
    // CHECK-DAG: [[C256:%.+]] = arith.constant 256 : index
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x256x140xf16>
    // CHECK: [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C256]] step [[C43]] iter_args([[OUT:%.+]] = [[EMPTY]])
    // CHECK-SAME:     -> (tensor<1x16x256x140xf16>)
    // CHECK:     affine.min
    // CHECK:     affine.min
    // CHECK:     affine.min
    // CHECK:     [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, {{%.+}}, 0] [1, 16, {{%.+}}, 140]
    // CHECK:     [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
    // CHECK:     [[SOFTMAX:%.+]] = VPU.SoftMax([[CONVERT]]) {axisInd = 3 : i64}
    // CHECK:     [[INSERT:%.+]] = tensor.insert_slice [[SOFTMAX]] into [[OUT]][0, 0, {{%.+}}, 0] [1, 16, {{%.+}}, 140]
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
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x16x256x140xf16> -> tensor<1x16x256x140xf16>

    %1 = VPU.Convert(%0) {
        dstElemType = f32,
        tilingStrategy = [1, 1, 1, 2]
    } : tensor<1x16x256x140xf16> -> tensor<1x16x256x140xf32>

    return %1 : tensor<1x16x256x140xf32>

    // CHECK-DAG: [[C43:%.+]] = arith.constant 43 : index
    // CHECK-DAG: [[C256:%.+]] = arith.constant 256 : index
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x256x140xf32>
    // CHECK: [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[C0]] to [[C256]] step [[C43]] iter_args([[OUT:%.+]] = [[EMPTY]])
    // CHECK-SAME:     -> (tensor<1x16x256x140xf32>)
    // CHECK:     affine.min
    // CHECK:     affine.min
    // CHECK:     affine.min
    // CHECK:     [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, {{%.+}}, 0] [1, 16, {{%.+}}, 140]
    // CHECK:     [[SOFTMAX:%.+]] = VPU.SoftMax([[SLICE]]) {axisInd = 3 : i64}
    // CHECK:     [[CONVERT:%.+]] = VPU.Convert([[SOFTMAX]]) {dstElemType = f32}
    // CHECK:     [[INSERT:%.+]] = tensor.insert_slice [[CONVERT]] into [[OUT]][0, 0, {{%.+}}, 0] [1, 16, {{%.+}}, 140]
    // CHECK:     scf.yield [[INSERT]] : tensor<1x16x256x140xf32>
    // CHECK: return [[SCF]] : tensor<1x16x256x140xf32>
}

// -----

// CHECK: #[[$MAP_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
// CHECK: #[[$MAP_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 128)>

// CHECK-LABEL: @MergeConvertGeluDynHW
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
func.func @MergeConvertGeluDynHW(%arg0: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 2]} : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
    %1 = VPU.Gelu(%0) {tilingStrategy = [1, 1, 2, 2]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
    return %1 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>

// CHECK-DAG:    [[CST_3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[CST_2:%.+]] = arith.constant 2 : index
// CHECK-DAG:    [[CST_128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[CST_16:%.+]] = arith.constant 16 : index
// CHECK-DAG:    [[CST_0:%.+]] = arith.constant 0 : index
// CHECK:        [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[CST_2]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK:        [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[CST_3]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK:        [[EMPTY:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK:        [[DIM_H_END:%.+]] = tensor.dim [[INPUT]], [[CST_2]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK:        [[DIM_W_END:%.+]] = tensor.dim [[INPUT]], [[CST_3]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK:        [[SCF_H:%.+]] = scf.for [[IDX_H:%.+]] = [[CST_0]] to [[DIM_H_END]] step [[CST_16]] iter_args([[OUT_H:%.+]] = [[EMPTY]])
// CHECK-SAME:          -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>)
// CHECK:          [[SCF_W:%.+]] = scf.for [[IDX_W:%.+]] = [[CST_0]] to [[DIM_W_END]] step [[CST_128]] iter_args([[OUT_W:%.+]] = [[OUT_H]])
// CHECK:            [[SIZE_H:%.+]] = affine.min #[[$MAP_MIN_H]]([[IDX_H]])[[[DIM_H_END]]]
// CHECK:            [[SIZE_W:%.+]] = affine.min #[[$MAP_MIN_W]]([[IDX_W]])[[[DIM_W_END]]]
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[GELU:%.+]] = VPU.Gelu([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[GELU]] into [[OUT_W]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK:          scf.yield [[SCF_W]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK:        return [[SCF_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
}

// -----

!qElemType_dyn = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK: #[[$MAP_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
// CHECK: #[[$MAP_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 128)>
// CHECK-LABEL: @MergeConvertDequantizeDynHW
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x?x?xf32
func.func @MergeConvertDequantizeDynHW(%arg0: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}> {
    %0 = VPU.Convert(%arg0) {dstElemType = !qElemType_dyn, tilingStrategy = [1, 1, 2, 2]} : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}> -> tensor<1x16x?x?x!qElemType_dyn, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
    %1 = VPU.Dequantize(%0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 2]} : tensor<1x16x?x?x!qElemType_dyn, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
    return %1 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>

// CHECK-DAG:    [[CST_2:%.+]] = arith.constant 2 : index
// CHECK-DAG:    [[CST_3:%.+]] = arith.constant 3 : index
// CHECK:        [[DIM_H1:%.+]] = tensor.dim [[INPUT]], [[CST_2]]
// CHECK:        [[DIM_W1:%.+]] = tensor.dim [[INPUT]], [[CST_3]]
// CHECK:        [[EMPTY:%.+]] = tensor.empty([[DIM_H1]], [[DIM_W1]])
// CHECK:        [[DIM_H2:%.+]] = tensor.dim [[INPUT]], [[CST_2]]
// CHECK:        [[DIM_W2:%.+]] = tensor.dim [[INPUT]], [[CST_3]]
// CHECK:        [[SCF_H:%.+]] = scf.for [[IDX_H:%.+]] = {{%.+}} to [[DIM_H2]] step {{%.+}} iter_args([[ARG_H:%.+]] = [[EMPTY]])
// CHECK:            [[SCF_W:%.+]] = scf.for [[IDX_W:%.+]] = {{%.+}} to [[DIM_W2]] step {{%.+}} iter_args([[ARG_W:%.+]] = [[ARG_H]])
// CHECK:                [[MIN_H:%.+]] = affine.min #[[$MAP_MIN_H]]([[IDX_H]])[[[DIM_H2]]]
// CHECK:                [[MIN_W:%.+]] = affine.min #[[$MAP_MIN_W]]([[IDX_W]])[[[DIM_W2]]]
// CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[MIN_H]], [[MIN_W]]]
// CHECK:                [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = !qElemType}
// CHECK:                [[DQ:%.+]] = VPU.Dequantize([[CONVERT]]) {dstElemType = f16}
// CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[DQ]] into [[ARG_W]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[MIN_H]], [[MIN_W]]]
// CHECK:                scf.yield [[INSERT]]
// CHECK:            scf.yield [[SCF_W]]
// CHECK:        return [[SCF_H]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: #[[$MAP_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
// CHECK: #[[$MAP_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 128)>
// CHECK-LABEL: @MergeConvertMVN1NormalizeDynHW
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
// CHECK-SAME:  [[MEANVAR:%.+]]: tensor<1x16x1x2xf16>
func.func @MergeConvertMVN1NormalizeDynHW(%arg0: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>, %arg1: tensor<1x16x1x2xf16>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 2]} : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
    %1 = VPU.MVN1Normalize(%0, %arg1) {across_channels = false, normalize_variance = true, tilingStrategy = [1, 1, 2, 2]} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>, tensor<1x16x1x2xf16> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>
    return %1 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 256]> : tensor<4xsi64>}>

// CHECK-DAG:    [[CST_2:%.+]] = arith.constant 2 : index
// CHECK-DAG:    [[CST_3:%.+]] = arith.constant 3 : index
// CHECK:        [[DIM_H1:%.+]] = tensor.dim [[INPUT]], [[CST_2]]
// CHECK:        [[DIM_W1:%.+]] = tensor.dim [[INPUT]], [[CST_3]]
// CHECK:        [[EMPTY:%.+]] = tensor.empty([[DIM_H1]], [[DIM_W1]])
// CHECK:        [[DIM_H2:%.+]] = tensor.dim [[INPUT]], [[CST_2]]
// CHECK:        [[DIM_W2:%.+]] = tensor.dim [[INPUT]], [[CST_3]]
// CHECK:        [[SCF_H:%.+]] = scf.for [[IDX_H:%.+]] = {{%.+}} to [[DIM_H2]] step {{%.+}} iter_args([[ARG_H:%.+]] = [[EMPTY]])
// CHECK:            [[SCF_W:%.+]] = scf.for [[IDX_W:%.+]] = {{%.+}} to [[DIM_W2]] step {{%.+}} iter_args([[ARG_W:%.+]] = [[ARG_H]])
// CHECK:                [[MIN_H:%.+]] = affine.min #[[$MAP_MIN_H]]([[IDX_H]])[[[DIM_H2]]]
// CHECK:                [[MIN_W:%.+]] = affine.min #[[$MAP_MIN_W]]([[IDX_W]])[[[DIM_W2]]]
// CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[MIN_H]], [[MIN_W]]]
// CHECK:                [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:                [[MVN:%.+]] = VPU.MVN1Normalize([[CONVERT]], [[MEANVAR]]) {across_channels = false, normalize_variance = true}
// CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[MVN]] into [[ARG_W]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[MIN_H]], [[MIN_W]]]
// CHECK:                scf.yield [[INSERT]]
// CHECK:            scf.yield [[SCF_W]]
// CHECK:        return [[SCF_H]]
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MixedTilingFusionStatic
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x16x480x960xf16>
func.func @MixedTilingFusionStatic(%arg0: tensor<1x16x480x960xf16>)
        -> tensor<1x16x480x960xf16, {order = #NHWC}> {
    %cst_w = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
        clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64,
        prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        tilingStrategy = [1, 1, 20, 1]
    } -> tensor<1x16x480x960xf16, {order = #NHWC}>

    %1 = VPU.NCE.Convolution(%0, %cst_w) rawFilterShape [16, 16, 1, 1] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <LRELU>, clamp_low = 0.000000e+00 : f64,
        clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
        prelu_alpha = [-0.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 30]
    } : tensor<1x16x480x960xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
     -> tensor<1x16x480x960xf16, {order = #NHWC}>

    %2 = VPU.NCE.Convolution(%1, %cst_w) rawFilterShape [16, 16, 1, 1] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <LRELU>, clamp_low = 0.000000e+00 : f64,
        clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
        prelu_alpha = [-0.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 30]
    } : tensor<1x16x480x960xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
     -> tensor<1x16x480x960xf16, {order = #NHWC}>

    %3 = VPU.NCE.Convolution(%2, %cst_w) rawFilterShape [16, 16, 1, 1] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <LRELU>, clamp_low = 0.000000e+00 : f64,
        clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
        prelu_alpha = [-0.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 30]
    } : tensor<1x16x480x960xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
     -> tensor<1x16x480x960xf16, {order = #NHWC}>

    %4 = VPU.NCE.Eltwise(%3, %3) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
        clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
        prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        tilingStrategy = [1, 1, 20, 1]
    } -> tensor<1x16x480x960xf16, {order = #NHWC}>

    %5 = VPU.NCE.Convolution(%4, %cst_w) rawFilterShape [16, 16, 1, 1] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
        clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
        prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 30]
    } : tensor<1x16x480x960xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
     -> tensor<1x16x480x960xf16, {order = #NHWC}>

    return %5 : tensor<1x16x480x960xf16, {order = #NHWC}>

    // CHECK-DAG: [[C48:%.+]]  = arith.constant 48 : index
    // CHECK-DAG: [[C960:%.+]] = arith.constant 960 : index
    // CHECK-DAG: [[C480:%.+]] = arith.constant 480 : index
    // CHECK-DAG: [[C0:%.+]]   = arith.constant 0 : index
    // CHECK-DAG: [[CST_W:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x480x960xf16, {order = #NHWC}>

    // CHECK:     [[LOOP_H:%.+]] = scf.for [[IH:%.+]] = [[C0]] to [[C480]] step [[C48]]
    // CHECK-SAME:                 iter_args([[ARG_H:%.+]] = [[EMPTY]]) -> (tensor<1x16x480x960xf16, {order = #NHWC}>) {

    // CHECK:       [[LOOP_W:%.+]] = scf.for [[IW:%.+]] = [[C0]] to [[C960]] step [[C480]]
    // CHECK-SAME:                   iter_args([[ARG_W:%.+]] = [[ARG_H]]) -> (tensor<1x16x480x960xf16, {order = #NHWC}>) {

    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IH]], [[IW]]] [1, 16, 48, 480] [1, 1, 1, 1]
    // CHECK-SAME:    tensor<1x16x480x960xf16> to tensor<1x16x48x480xf16>

    // CHECK:         [[PERMUTE:%.+]] = VPU.NCE.Permute([[SLICE]])
    // CHECK:         [[CONV0:%.+]]   = VPU.NCE.Convolution([[PERMUTE]], [[CST_W]])
    // CHECK:         [[CONV1:%.+]]   = VPU.NCE.Convolution([[CONV0]], [[CST_W]])
    // CHECK:         [[CONV2:%.+]]   = VPU.NCE.Convolution([[CONV1]], [[CST_W]])
    // CHECK:         [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV2]], [[CONV2]])
    // CHECK:         [[CONV3:%.+]]   = VPU.NCE.Convolution([[ELTWISE]], [[CST_W]])

    // CHECK:         [[INSERT:%.+]] = tensor.insert_slice [[CONV3]] into [[ARG_W]][0, 0, [[IH]], [[IW]]] [1, 16, 48, 480] [1, 1, 1, 1]
    // CHECK-SAME:    tensor<1x16x48x480xf16, {order = #NHWC}> into tensor<1x16x480x960xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[INSERT]] : tensor<1x16x480x960xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       scf.yield [[LOOP_W]] : tensor<1x16x480x960xf16, {order = #NHWC}>
    // CHECK:     }
    // CHECK:     return [[LOOP_H]] : tensor<1x16x480x960xf16, {order = #NHWC}>
}

// -----

// CHECK: #[[$MAP_MIN_H:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
// CHECK: #[[$MAP_MIN_W:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 70)>

// CHECK-LABEL: @MergeConvertSoftMaxDynHW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>)
func.func @MergeConvertSoftMaxDynHW(%arg0: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}> {
    %0 = VPU.Convert(%arg0) {
        dstElemType = f16,
        tilingStrategy = [1, 1, 2, 2]
    } : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>

    %1 = VPU.SoftMax(%0) {
        axisInd = 1 : i64,
        tilingStrategy = [1, 1, 2, 2]
    } : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>

    return %1 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>

    // CHECK-DAG: [[CST_3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[CST_70:%.+]] = arith.constant 70 : index
    // CHECK-DAG: [[CST_16:%.+]] = arith.constant 16 : index
    // CHECK-DAG: [[CST_0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[CST_2:%.+]] = arith.constant 2 : index
    // CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[CST_2]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[CST_3]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
    // CHECK: [[EMPTY:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
    // CHECK: [[DIM_H_END:%.+]] = tensor.dim [[INPUT]], [[CST_2]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
    // CHECK: [[DIM_W_END:%.+]] = tensor.dim [[INPUT]], [[CST_3]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
    // CHECK: [[SCF_H:%.+]] = scf.for [[IDX_H:%.+]] = [[CST_0]] to [[DIM_H_END]] step [[CST_16]] iter_args([[OUT_H:%.+]] = [[EMPTY]])
    // CHECK-SAME:     -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>)
    // CHECK:   [[SCF_W:%.+]] = scf.for [[IDX_W:%.+]] = [[CST_0]] to [[DIM_W_END]] step [[CST_70]] iter_args([[OUT_W:%.+]] = [[OUT_H]])
    // CHECK:     [[SIZE_H:%.+]] = affine.min #[[$MAP_MIN_H]]([[IDX_H]])[[[DIM_H_END]]]
    // CHECK:     [[SIZE_W:%.+]] = affine.min #[[$MAP_MIN_W]]([[IDX_W]])[[[DIM_W_END]]]
    // CHECK:     [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
    // CHECK:     [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
    // CHECK:     [[SOFTMAX:%.+]] = VPU.SoftMax([[CONVERT]]) {axisInd = 1 : i64}
    // CHECK:     [[INSERT:%.+]] = tensor.insert_slice [[SOFTMAX]] into [[OUT_W]][0, 0, [[IDX_H]], [[IDX_W]]] [1, 16, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1]
    // CHECK:     scf.yield [[INSERT]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
    // CHECK:   scf.yield [[SCF_W]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
    // CHECK: return [[SCF_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 140]> : tensor<4xsi64>}>
}

// -----

// CHECK-LABEL: @MergeConvertAbs
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertAbs(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Abs(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Sin(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Sqrt(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Tanh(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Mish(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.ReLU(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Sigmoid(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.SoftPlus(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Swish(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Acos(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Acosh(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Asin(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Asinh(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Atan(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Atanh(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[ATANH:%.+]] = VPU.Atanh([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ATANH]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertCeiling
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertCeiling(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Ceiling(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[CEILING:%.+]] = VPU.Ceiling([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[CEILING]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertCos
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertCos(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Cos(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[COS:%.+]] = VPU.Cos([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[COS]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertCosh
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertCosh(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Cosh(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Erf(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[ERF:%.+]] = VPU.Erf([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[ERF]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertExp
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertExp(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Exp(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[EXP:%.+]] = VPU.Exp([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[EXP]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertFloor
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertFloor(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Floor(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[FLOOR:%.+]] = VPU.Floor([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[FLOOR]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertLog
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertLog(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Log(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[LOG:%.+]] = VPU.Log([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[LOG]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertNegative
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertNegative(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Negative(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[NEG:%.+]] = VPU.Negative([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[NEG]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertSign
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertSign(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Sign(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[SIGN:%.+]] = VPU.Sign([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SIGN]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

// CHECK-LABEL: @MergeConvertSinh
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertSinh(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Sinh(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Tan(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Elu(%0) {tilingStrategy = [1, 1, 2, 1], x = 1.000000e+00 : f64} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.HardSigmoid(%0) {alpha_value = 2.000000e-01 : f64, beta_value = 5.000000e-01 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.HSwish(%0) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.LeakyRelu(%0) {negative_slope = 1.000000e-02 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
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
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Selu(%0) {alpha_value = 1.6732000000000001 : f64, lambda_value = 1.0507000000000001 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
    return %1 : tensor<1x16x128x256xf16>

// CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x128x256xf16>
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[SELU:%.+]] = VPU.Selu([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SELU]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!outputStaticType = tensor<1x32x540x960x!quant.uniform<u8:f16, 0.030033121856988646:110>, {order = #NHWC}>

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
}

// CHECK:       [[C0_I8:%.+]] = arith.constant 0 : i8
// CHECK:       [[CSTEP:%.+]] = arith.constant {{[0-9]+}} : index
// CHECK:       [[C960:%.+]] = arith.constant 960 : index
// CHECK:       [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:   [[CST_W_DW:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
// CHECK-DAG:   [[CST_W_CC:%.+]] = const.Declare tensor<32x1x1x32x!qElemType{{.*}}, {order = #NHWC}>
// CHECK-DAG:   [[CST_B_CC:%.+]] = const.Declare tensor<32x1x1x4xsi32>
// CHECK:       [[EMPTY:%.+]] = tensor.empty() : tensor<1x32x540x960x!qElemType, {order = #NHWC}>
// CHECK:       [[LOOP:%.+]] = scf.for [[IV:%.+]] = [[C0]] to [[C960]] step [[CSTEP]] iter_args([[ARG_ITER:%.+]] = [[EMPTY]])
// CHECK:         [[EXTRACT:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, {{%.+}}] [1, 16, 1080, {{%.+}}] [1, 1, 1, 1]
// CHECK-SAME:      tensor<1x16x1080x1920xf16, {order = #NHWC}> to tensor<1x16x1080x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         [[DEPTH:%.+]] = VPU.NCE.DepthConvolution([[EXTRACT]], [[CST_W_DW]])
// CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK-SAME:      -> tensor<1x16x1080x?x!qElemType{{.*}}, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         [[SLICE:%.+]] = VPU.Slice [[DEPTH]] [0, 0, 0, 0] [1, 4, 1080, -9223372036854775808]
// CHECK-SAME:      tensor<1x16x1080x?x!qElemType{{.*}}, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x4x1080x?x!qElemType{{.*}}, {bounds = #const.OpaqueI64Elements<[1, 4, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         [[CAST:%.+]] = builtin.unrealized_conversion_cast [[C0_I8]] : i8 to !qElemType{{.*}}
// CHECK:         [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, 1, {{%.+}}] high[0, 0, 0, 0]
// CHECK:         [[COMPRESS:%.+]] = VPU.NCE.CompressConvolution([[PAD]], [[CST_W_CC]], [[CST_B_CC]])
// CHECK-SAME:      cm_sp_pattern = 7
// CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
// CHECK-SAME:      -> tensor<1x32x540x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:         [[INSERT:%.+]] = tensor.insert_slice {{%.+}} into [[ARG_ITER]][0, 0, 0, [[OUT_OFF_W:%.+]]] [1, 32, 540, [[TILE_W:.+]]] [1, 1, 1, 1]
// CHECK:         scf.yield [[INSERT]] : tensor<1x32x540x960x!qElemType, {order = #NHWC}>
// CHECK:       return [[LOOP]] : tensor<1x32x540x960x!qElemType, {order = #NHWC}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SCFVFSliceAxisConflict
// CHECK-SAME:    [[INPUT:%[^:]+]]: tensor<1x5120x1024x1xf16>
func.func @SCFVFSliceAxisConflict(%arg0: tensor<1x5120x1024x1xf16>) -> tensor<1x2560x1024x1xf16> {
    %0 = VPU.Slice %arg0 [0, 2560, 0, 0] [1, 2560, 1024, 1] : tensor<1x5120x1024x1xf16> to tensor<1x2560x1024x1xf16>
    %1 = VPU.SoftMax(%0) {axisInd = 1 : i64, tilingStrategy = [1, 8, 1, 1]} : tensor<1x2560x1024x1xf16> -> tensor<1x2560x1024x1xf16>
    return %1 : tensor<1x2560x1024x1xf16>
}


// CHECK-DAG:   [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:   [[C1024:%.+]] = arith.constant 1024 : index
// CHECK:       [[EMPTY:%.+]] = tensor.empty() : tensor<1x2560x1024x1xf16>
// CHECK:       [[LOOP:%.+]] = scf.for [[IV:%.+]] = [[C0]] to [[C1024]]
// CHECK-SAME:    iter_args([[ARG_ITER:%.+]] = [[EMPTY]])
// CHECK:         [[EXTRACT:%.+]] = tensor.extract_slice [[INPUT]][0, 0, {{%[^,]+}}, 0] [1, 5120, {{%[^,]+}}, 1]
// CHECK-SAME:      tensor<1x5120x1024x1xf16> to tensor<1x5120x?x1xf16, {{.+}}>
// CHECK:         [[SLICE:%.+]] = VPU.Slice [[EXTRACT]] [0, 2560, 0, 0] [1, 2560, -9223372036854775808, 1]
// CHECK-SAME:      to tensor<1x2560x?x1xf16, {{.+}}>
// CHECK:         [[SOFTMAX:%.+]] = VPU.SoftMax([[SLICE]]) {axisInd = 1 : i64}
// CHECK-NOT:     tilingStrategy
// CHECK:         [[INSERT:%.+]] = tensor.insert_slice [[SOFTMAX]] into [[ARG_ITER]]
// CHECK:         scf.yield [[INSERT]]
// CHECK:       return [[LOOP]] : tensor<1x2560x1024x1xf16>

// -----

// Tests for MC strategy alignment checks ported from VF default pipeline
// (isLegalFusion + isMCStrategyAligned):
//
// 1. @NoFusionMismatchedMCStrategy — one op has MC strategy, the other does not
// 2. @NoFusionSWOpWithTrueOverlapped (E#92130) — SW op without DMA lowering
//    paired with a true-overlapped consumer
// 3. @NoFusionSparseOperandTrueOverlapped (E#112803) — sparse activation operand
//    feeding a true-overlapped consumer
// 4. @FusionClusteringToSOHEltwise — Clustering → SOH eltwise (fuses in SCF;
//    eltwise segmented-like guard not applied in SCF path)
// 5. @FusionWithMCStrategyAdjustment / @NoMutationOnRejectedAdjustment /
//    @ChainedMCStrategyAdjustment / @MultiUserProducerNoAccumulation — producer MC
//    strategy adjustment (planFusion + applyFusionPlan): applied when fusion is
//    accepted, never applied when rejected, transitively consistent across chained
//    producers, and committed exactly once for multi-user producers

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @NoFusionMismatchedMCStrategy
func.func @NoFusionMismatchedMCStrategy(%arg0: tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // First op has MC strategy
    %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 22]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Second op does NOT have MC strategy — should not fuse with the first
    %1 = VPU.NCE.Convolution(%0, %weights) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 22]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %1 : tensor<1x32x540x960xf16, {order = #NHWC}>
}

// MC strategy mismatch prevents fusion — ops remain untiled
// CHECK-NOT: scf.for
// CHECK: VPU.NCE.Convolution
// CHECK: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK: VPU.NCE.Convolution
// CHECK-NOT: multiClusterStrategy
// CHECK: return

// -----

// E#92130: SW ops that do not support DMA lowering must not be fused with ops
// that produce true-overlapped distributions (memory shapes != compute shapes).
// Gelu does not support DMA lowering, and NCE.Convolution with kernel 3x3 and
// SplitOverHeight requires OVERLAPPED input distribution with halo, creating a
// true-overlapped consumer. This pair must remain unfused.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @NoFusionSWOpWithTrueOverlapped
func.func @NoFusionSWOpWithTrueOverlapped(%arg0: tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // Gelu - SW op that does not support lowering as DMA
    %0 = VPU.Gelu(%arg0) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1, 1, 6, 1]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // NCE Conv with kernel 3x3 and SOH — consumer input requires OVERLAPPED (true overlap)
    %1 = VPU.NCE.Convolution(%0, %weights) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 6, 1]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %1 : tensor<1x32x540x960xf16, {order = #NHWC}>
}

// SW op without DMA lowering + true-overlapped consumer prevents fusion
// CHECK-NOT: scf.for
// CHECK: VPU.Gelu
// CHECK: VPU.NCE.Convolution
// CHECK: return

// -----

// E#112803: True-overlapped consumer with sparse activation operand must not be
// fused. NCE.MaxPool produces a sparse activation tensor that flows into
// NCE.Convolution whose 3x3 kernel + SplitOverHeight creates a true-overlapped
// input distribution. The sparse operand type triggers the rejection path.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @NoFusionSparseOperandTrueOverlapped
func.func @NoFusionSparseOperandTrueOverlapped(
        %arg0: tensor<1x32x540x960xf16, {order = #NHWC}>,
        %arg1: tensor<1x32x540x960xi1, {order = #NHWC}>)
        -> tensor<1x32x540x960xf16, {order = #NHWC}> {
    %wt = const.Declare tensor<32x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<32x1x1x4xsi32>
    %weights = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %sparse_in = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x32x540x960xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x32x540x960xi1, {order = #NHWC}>>

    // Producer: NCE.MaxPool with SOH, activation sparsity in/out
    %pool = VPU.NCE.MaxPool(%sparse_in, %wt) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 6, 1]
      } -> !VPU.SparseTensor<data=tensor<1x32x540x960xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x32x540x960xi1, {order = #NHWC}>>

    // Consumer: NCE.Convolution with 3x3 kernel + SOH — true-overlapped input
    %conv = VPU.NCE.Convolution(%pool, %weights) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 6, 1]}
      : !VPU.SparseTensor<data=tensor<1x32x540x960xf16, {order = #NHWC}>,
                          sparsity_map=tensor<1x32x540x960xi1, {order = #NHWC}>>,
        tensor<32x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %conv : tensor<1x32x540x960xf16, {order = #NHWC}>
}

// Sparse operand + true-overlapped consumer prevents fusion
// CHECK-NOT: scf.for
// CHECK: VPU.GroupSparseTensor
// CHECK: VPU.NCE.MaxPool
// CHECK: VPU.NCE.Convolution
// CHECK: return

// -----

// Default VF: NCE eltwise consumer where input distribution is not segmented-like but
// output IS segmented-like is rejected (DMA cost accuracy guard).
// SCF VF: this guard is intentionally not applied, so this topology should fuse.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FusionClusteringToSOHEltwise
func.func @FusionClusteringToSOHEltwise(
        %arg0: tensor<1x32x540x960xf16, {order = #NHWC}>)
        -> tensor<1x32x540x960xf16, {order = #NHWC}> {
    %weights1 = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %weights2 = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<2.0>
        : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // Branch A: Clustering producer → DUPLICATED output
    %convA = VPU.NCE.Convolution(%arg0, %weights1) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 6, 1]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Branch B: Clustering producer → DUPLICATED output
    %convB = VPU.NCE.Convolution(%arg0, %weights2) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 6, 1]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Consumer: Eltwise with SOH — input becomes DUPLICATED, output is segmented-like
    %eltwise = VPU.NCE.Eltwise(%convA, %convB) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        tilingStrategy = [1, 1, 6, 1]}
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %eltwise : tensor<1x32x540x960xf16, {order = #NHWC}>
}

// Clustering → SOH eltwise: previously rejected by the eltwise segmented-like guard (which was
// inadvertently applied in the SCF path). After the fix aligning SCF with develop semantics
// (areDistributionAttrsCompatible only, no eltwise guard), this topology fuses. The eltwise
// segmented-like guard remains in the default VF isMCStrategyAligned path (cost-model concern)
// but is not applied in SCF's per-edge planFusion check.
// CHECK: scf.for
// CHECK: VPU.NCE.Convolution
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
// CHECK: VPU.NCE.Convolution
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
// CHECK: VPU.NCE.Eltwise
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK: return

// -----

// Test for MC strategy adjustment in SCF fusion (findProducerStrategyForFusion + applyFusionPlan):
// Producer Conv has SOK strategy, consumer Conv has SOH. The producer SOK output
// distribution is incompatible with the consumer SOH input. planFusion computes a
// compatible producer strategy via findProducerStrategyForFusion (without mutating IR),
// and applyFusionPlan commits it once fusion is accepted. Here the producer is adjusted
// from SOK to HKSwitch so that fusion can proceed.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FusionWithMCStrategyAdjustment
func.func @FusionWithMCStrategyAdjustment(%arg0: tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %weights2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // Producer: Conv 1x1 with SOK — will be adjusted to HKSwitch
    %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 22]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Consumer: Conv 3x3 with SOH — requires OVERLAPPED input
    %1 = VPU.NCE.Convolution(%0, %weights2) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 22]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %1 : tensor<1x32x540x960xf16, {order = #NHWC}>
}

// Strategy adjustment enables fusion — producer adjusted from SOK to HKSwitch.
// Exactly one tiled loop is emitted (single fused region).
// CHECK-COUNT-1: scf.for
// CHECK: VPU.NCE.Convolution
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>
// CHECK: VPU.NCE.Convolution
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK-NOT: scf.for

// -----

// CORE non-mutation regression test: planFusion plans a producer strategy adjustment,
// but fusion is then rejected by checkProducersUsers because the producer has a user
// outside the fusion set. The planned adjustment must NOT be applied.
//
// Producer Conv 1x1 has SOK; consumer/anchor Conv 3x3 has SOH. The producer SOK output is
// incompatible with the consumer SOH input, so planFusion computes a compatible producer
// strategy (HKSwitch) via findProducerStrategyForFusion — but returns it as a PLANNED
// adjustment only, without mutating IR. The producer result also escapes as a function
// output, so the producer has an external user. In collectTiledAndFusedOps the BFS rejects
// the fusion (checkProducersUsers) before applyFusionPlan runs, so no mutation is committed.
//
// This is the exact scenario the planFusion/applyFusionPlan split fixed. On the old eager
// implementation checkFusion mutated the producer to HKSwitch BEFORE the multi-user
// rejection, leaving the wrong strategy in the IR. The assertion below requires the producer
// to retain SplitOverKernel, so this test FAILS on the buggy code and PASSES on the fix.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @NoMutationOnRejectedAdjustment
func.func @NoMutationOnRejectedAdjustment(%arg0: tensor<1x32x540x960xf16, {order = #NHWC}>)
        -> (tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<1x32x540x960xf16, {order = #NHWC}>) {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %weights2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // Producer: Conv 1x1 with SOK — an adjustment to HKSwitch would be planned, but must NOT be applied
    %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 22]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Consumer/anchor: Conv 3x3 with SOH — requires OVERLAPPED input
    %1 = VPU.NCE.Convolution(%0, %weights2) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 22]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // %0 escapes as a second function output -> external (out-of-fusion-set) user of the producer
    return %1, %0 : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<1x32x540x960xf16, {order = #NHWC}>
}

// Fusion rejected (producer has an external user) AND producer strategy left unchanged.
// CHECK-NOT: scf.for
// CHECK: VPU.NCE.Convolution
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
// CHECK: VPU.NCE.Convolution
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK-NOT: scf.for
// CHECK: return

// -----

// Chained MC strategy adjustment with transitive visibility. Two stacked SOK Conv 1x1
// producers feed an SOH Conv 3x3 anchor. The BFS in collectTiledAndFusedOps walks up from
// the anchor: it first commits the lower producer (closest to the anchor) with an HKSwitch
// adjustment, then processes the upper producer. Because applyFusionPlan is committed inline
// before the next BFS step, planFusion for the upper producer observes the already-adjusted
// lower producer and plans a consistent HKSwitch strategy for it as well.
//
// This guards the transitive-visibility property of the planFusion/applyFusionPlan split: a
// committed producer adjustment must be visible to the planning of its own producers. Both
// producers end at HKSwitch, the anchor stays SplitOverHeight, and the whole chain fuses into
// a single scf.for. Each producer is adjusted exactly once — there is no double application.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @ChainedMCStrategyAdjustment
func.func @ChainedMCStrategyAdjustment(%arg0: tensor<1x32x540x960xf16, {order = #NHWC}>)
        -> tensor<1x32x540x960xf16, {order = #NHWC}> {
    %w1 = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %w2 = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %w3 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // Upper producer: Conv 1x1 with SOK — transitively adjusted to HKSwitch
    %0 = VPU.NCE.Convolution(%arg0, %w1) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 22]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Lower producer: Conv 1x1 with SOK — adjusted to HKSwitch first, then made visible upstream
    %1 = VPU.NCE.Convolution(%0, %w2) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 22]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Anchor: Conv 3x3 with SOH — stays SplitOverHeight
    %2 = VPU.NCE.Convolution(%1, %w3) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 22]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %2 : tensor<1x32x540x960xf16, {order = #NHWC}>
}

// Whole chain fuses into a single tiled loop; both SOK producers transitively adjusted to
// HKSwitch, anchor stays SOH.
// CHECK-COUNT-1: scf.for
// CHECK: VPU.NCE.Convolution
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>
// CHECK: VPU.NCE.Convolution
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>
// CHECK: VPU.NCE.Convolution
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK-NOT: scf.for

// -----

// Repeated-producer BFS test: a multi-user producer that is REJECTED on its first BFS
// encounter and COMMITTED on a later retry must end with exactly one strategy value — the
// re-evaluation must not accumulate or corrupt the producer's MC strategy.
//
// Topology forces the reject-then-retry path deterministically. The producer %p feeds the
// anchor Eltwise directly (skip input) AND a deeper branch %p -> %b -> %a -> Eltwise. The BFS
// in collectTiledAndFusedOps starts at the anchor and processes its operands: %a is fused
// first, then %p is encountered while %b (its other user) is NOT yet in the producers set, so
// checkProducersUsers REJECTS %p (planFusion is computed and discarded, no mutation). %b and
// %a are then walked in; on the second encounter all of %p's users (%b and the anchor) are in
// the set, so %p is committed. planFusion is side-effect-free, and applyFusionPlan runs only
// after acceptance, so the discarded first attempt leaves no residue.
//
// All ops are SOH (no adjustment needed), so the single committed strategy must remain exactly
// SplitOverHeight on every op. The whole graph fuses into ONE scf.for; the producer %p appears
// once and is read by both its deep consumer chain and the skip operand of the Eltwise.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @MultiUserProducerNoAccumulation
func.func @MultiUserProducerNoAccumulation(%arg0: tensor<1x32x540x960xf16, {order = #NHWC}>)
        -> tensor<1x32x540x960xf16, {order = #NHWC}> {
    %w = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // Producer %p: feeds the anchor Eltwise directly (skip) AND the deeper branch %b -> %a
    %p = VPU.NCE.Convolution(%arg0, %w) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1], tilingStrategy = [1, 1, 6, 1]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Deep branch op %b reading %p
    %b = VPU.NCE.Convolution(%p, %w) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1], tilingStrategy = [1, 1, 6, 1]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Deep branch op %a reading %b
    %a = VPU.NCE.Convolution(%b, %w) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1], tilingStrategy = [1, 1, 6, 1]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Anchor Eltwise reading %a (deep) and %p (skip) — the multi-user edge on %p
    %eltwise = VPU.NCE.Eltwise(%a, %p) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        tilingStrategy = [1, 1, 6, 1]}
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %eltwise : tensor<1x32x540x960xf16, {order = #NHWC}>
}

// Whole graph fuses into a single loop; %p committed exactly once with SOH unchanged, and the
// Eltwise skip operand is the same fused %p value (multi-user producer not duplicated/mutated).
// CHECK: [[LOOP:%.+]] = scf.for
// CHECK: [[P:%.+]] = VPU.NCE.Convolution
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK: [[B:%.+]] = VPU.NCE.Convolution([[P]]
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK: [[A:%.+]] = VPU.NCE.Convolution([[B]]
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK: VPU.NCE.Eltwise([[A]], [[P]])
// CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK: return
// Exactly one tiled loop — no second scf.for is produced by the multi-user producer.
// CHECK-NOT: scf.for

// -----

// Multi-user producer MC-strategy-adjustment SUPPRESSION guard. This validates the
// collectTiledAndFusedOps guard `plan.newProducerStrategy.has_value() && !producer->hasOneUse()`.
//
// Producer %p (Conv 1x1, SplitOverKernel) feeds a deep branch %p -> %b -> %a -> Eltwise AND the
// Eltwise skip input directly, so %p has two users. Both users (%b and the Eltwise) are pulled
// into the fused region, so checkProducersUsers PASSES for %p. When the BFS reaches %p through the
// %p->%b edge, planFusion plans an SplitOverKernel->HKSwitch adjustment (SOK producer is
// incompatible with the SOH consumer). Because %p has more than one use, the suppression guard
// rejects the adjustment: %p is left hoisted with its ORIGINAL SplitOverKernel strategy and is not
// mutated. Removing the guard would adjust %p to HKSwitch and pull it into the loop, failing the
// SplitOverKernel assertion below.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @NoAdjustmentForMultiUserProducer
func.func @NoAdjustmentForMultiUserProducer(%arg0: tensor<1x32x540x960xf16, {order = #NHWC}>)
        -> tensor<1x32x540x960xf16, {order = #NHWC}> {
    %w = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // Producer %p: Conv 1x1 with SOK — multi-user (feeds %b and the anchor Eltwise skip)
    %p = VPU.NCE.Convolution(%arg0, %w) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1], tilingStrategy = [1, 1, 6, 1]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Deep branch %b: Conv 1x1 SOH reading %p — SOK producer vs SOH consumer forces an adjustment plan
    %b = VPU.NCE.Convolution(%p, %w) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1], tilingStrategy = [1, 1, 6, 1]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Deep branch %a: Conv 1x1 SOH reading %b
    %a = VPU.NCE.Convolution(%b, %w) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1], tilingStrategy = [1, 1, 6, 1]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Anchor Eltwise SOH reading %a (deep) and %p (skip) — the multi-user edge on %p
    %eltwise = VPU.NCE.Eltwise(%a, %p) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        tilingStrategy = [1, 1, 6, 1]}
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %eltwise : tensor<1x32x540x960xf16, {order = #NHWC}>
}

// The multi-user producer keeps its original SplitOverKernel (adjustment suppressed, not mutated).
// CHECK:       [[P:%.+]] = VPU.NCE.Convolution(%arg0
// CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
// CHECK:       scf.for
// CHECK:       VPU.NCE.Convolution
// CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK:       VPU.NCE.Convolution
// CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK:       VPU.NCE.Eltwise
// CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK:       return
// The suppressed producer must never be mutated to the planned HKSwitch strategy.
// CHECK-NOT:   multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>

// -----

// Exhausted MC-strategy-adjustment search leaves IR unchanged. This validates that a failed
// findProducerStrategyForFusion (returning nullopt) does NOT mutate any strategy.
//
// Producer NCE.MaxPool (SplitOverKernel) consumes a sparse activation, so
// supportMultiClusterStrategyAdjustmentInVF returns false and findCompatibleMCStrategyForVF /
// findProducerStrategyForFusion cannot find any compatible producer strategy (nullopt). The
// consumer Conv 1x1 is SplitOverHeight, so the current strategies are incompatible and planFusion
// enters the adjustment search; with the search exhausted it returns SCFFusionPlan::rejected().
// applyFusionPlan is never reached, so the producer keeps its original SplitOverKernel strategy and
// no fusion is emitted. A regression that mutates IR during a failed search would change the
// MaxPool strategy and fail this test.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @NoMutationOnExhaustedAdjustmentSearch
func.func @NoMutationOnExhaustedAdjustmentSearch(
        %arg0: tensor<1x32x540x960xf16, {order = #NHWC}>,
        %arg1: tensor<1x32x540x960xi1, {order = #NHWC}>)
        -> tensor<1x32x540x960xf16, {order = #NHWC}> {
    %wt = const.Declare tensor<32x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<32x1x1x4xsi32>
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %sparse_in = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x32x540x960xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x32x540x960xi1, {order = #NHWC}>>

    // Producer: NCE.MaxPool with SOK and a sparse activation input -> not adjustable (search exhausts)
    %pool = VPU.NCE.MaxPool(%sparse_in, %wt) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 6, 1]
      } -> !VPU.SparseTensor<data=tensor<1x32x540x960xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x32x540x960xi1, {order = #NHWC}>>

    // Consumer: NCE.Convolution 1x1 with SOH — strategy incompatible with the SOK producer
    %conv = VPU.NCE.Convolution(%pool, %weights) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 6, 1]}
      : !VPU.SparseTensor<data=tensor<1x32x540x960xf16, {order = #NHWC}>,
                          sparsity_map=tensor<1x32x540x960xi1, {order = #NHWC}>>,
        tensor<32x32x1x1xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %conv : tensor<1x32x540x960xf16, {order = #NHWC}>
}

// Fusion rejected (adjustment search exhausted) — producer strategy left unchanged, no loop emitted.
// CHECK-NOT:   scf.for
// CHECK:       VPU.GroupSparseTensor
// CHECK:       VPU.NCE.MaxPool
// CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
// CHECK:       VPU.NCE.Convolution
// CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK:       return
// CHECK-NOT:   scf.for

// -----

// REGRESSION characterization: skip-connection topology where the eltwise segmented-like guard
// in checkCurrentMCStrategyCompatibility was rejecting Conv→Eltwise edges. The root cause was that
// SCF's planFusion and findProducerStrategyForFusion used the full checkCurrentMCStrategyCompatibility
// (which includes the eltwise guard) instead of the narrower areDistributionAttrsCompatible used by
// develop and default VF's alignMCStrategy. Fixed by aligning SCF's compatibility criterion to match
// develop: areDistributionAttrsCompatible for both the initial check and the adjustment search.
//
// Topology:
//     NCE.Permute (SOHO)
//         |
//     Conv1 (SOH, 3x3)   ← multi-user producer: feeds BOTH Conv2 AND Eltwise
//       |         \
//       |          \
//   Conv2 (SOH)    |  (skip connection)
//       |          |
//       +--- Eltwise(Add, SOH) ---+
//
// Expected behavior: Conv1 is the skip-connection source. Conv2 and Eltwise fuse into a tiled
// loop. Conv1 feeds the loop via a skip extract_slice with the {skip_connection_slice} marker.
// A regression that re-introduces the overly-strict eltwise guard in planFusion would reject the
// Conv→Eltwise edges and produce no scf.for, failing the CHECK below.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @SkipConnectionEltwiseSegmentedLikeRegression
func.func @SkipConnectionEltwiseSegmentedLikeRegression(%arg0: tensor<1x3x540x960xf16>) -> tensor<1x32x540x960xf16, {order = #NHWC}> {
    %w1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %w2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0>
        : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // NCE.Permute: layout change NCHW -> NHWC
    %permute = VPU.NCE.Permute(%arg0) {
        dstElemType = f16,
        dstOrder = #NHWC,
        expandedChannels = 32 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                         clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64,
                         prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        tilingStrategy = [1, 1, 6, 1]}
      : tensor<1x3x540x960xf16>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Conv1 (SOH, 3x3): multi-user producer — consumed by Conv2 AND Eltwise (skip connection source)
    %conv1 = VPU.NCE.Convolution(%permute, %w1) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1], tilingStrategy = [1, 1, 6, 1]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Conv2 (SOH, 3x3): deep branch consuming Conv1
    %conv2 = VPU.NCE.Convolution(%conv1, %w2) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1], tilingStrategy = [1, 1, 6, 1]}
      : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Eltwise(Add, SOH): skip connection — consumes Conv1 (skip) and Conv2 (deep)
    %add = VPU.NCE.Eltwise(%conv1, %conv2) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        tilingStrategy = [1, 1, 6, 1]}
      -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %add : tensor<1x32x540x960xf16, {order = #NHWC}>
}

// Fusion succeeds: the whole graph fuses into a single scf.for. NCE.Permute stays hoisted;
// Conv1, Conv2, and Eltwise are all inside the loop. Conv1's output feeds the Eltwise skip
// connection via a {skip_connection_slice} extract_slice within the loop body.
// CHECK:       VPU.NCE.Permute
// CHECK:       scf.for
// CHECK:         VPU.NCE.Convolution
// CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK:         VPU.NCE.Convolution
// CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK:         tensor.extract_slice
// CHECK-SAME:    skip_connection_slice
// CHECK:         VPU.NCE.Eltwise
// CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// CHECK:       return
