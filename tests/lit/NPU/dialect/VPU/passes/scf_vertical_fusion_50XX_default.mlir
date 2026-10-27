//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --scf-vertical-fusion="vf-merge-configuration=COST_BASED" --resolve-shaped-type-result-dims --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

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

// CHECK-LABEL: @MergeConvertSin
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x256xf32>
func.func @MergeConvertSin(%arg0: tensor<1x16x128x256xf32>) -> tensor<1x16x128x256xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf32> -> tensor<1x16x128x256xf16>
    %1 = VPU.Sin(%0) {tilingStrategy = [1, 1, 8, 2]} : tensor<1x16x128x256xf16> -> tensor<1x16x128x256xf16>
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
