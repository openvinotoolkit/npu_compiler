//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --scf-vertical-fusion="vf-merge-configuration=COST_BASED" --resolve-shaped-type-result-dims --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU6010

config.Resources 6 of @NCE at 1.850000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #[[$TO_SLICE_OFFSET_MAP:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$TO_PAD_CANDIDATE_MAP:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$PAD_CLAMP_MAP:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$PAD_HIGH_STAGE5_H_CANDIDATE_MAP:.+]] = affine_map<(d0) -> (0, d0 - 448)>
// CHECK: #[[$PAD_HIGH_STAGE5_W_CANDIDATE_MAP:.+]] = affine_map<(d0) -> (0, d0 - 718)>
// CHECK: #[[$PAD_HIGH_STAGE4_H_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2) -> (0, d0 - d1 - d2 - 446)>
// CHECK: #[[$PAD_HIGH_STAGE4_W_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2) -> (0, d0 - d1 - d2 - 716)>
// CHECK: #[[$PAD_HIGH_STAGE3_H_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (0, d0 - d1 - d2 - d3 - d4 - 444)>
// CHECK: #[[$PAD_HIGH_STAGE3_W_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (0, d0 - d1 - d2 - d3 - d4 - 714)>
// CHECK: #[[$PAD_HIGH_STAGE2_H_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 - 442)>
// CHECK: #[[$PAD_HIGH_STAGE2_W_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 - 712)>
// CHECK: #[[$PAD_HIGH_STAGE1_H_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - 440)>
// CHECK: #[[$PAD_HIGH_STAGE1_W_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - 710)>
// CHECK: #[[$PAD_HIGH_STAGE0_H_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - d9 - d10 - 438)>
// CHECK: #[[$SLICE_SIZE_H_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) -> (-d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - d9 - d10 - d11 + 102)>
// CHECK: #[[$PAD_HIGH_STAGE0_W_CANDIDATE_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - d9 - d10 - 708)>
// CHECK: #[[$SLICE_SIZE_W_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) -> (-d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - d9 - d10 - d11 + 252)>

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
    // CHECK-DAG: [[LOOP_STEP_W:%.+]] = arith.constant 240 : index
    // CHECK-DAG: [[LOOP_STEP_H:%.+]] = arith.constant 90 : index
    // CHECK-DAG: [[LOOP_END_W:%.+]] = arith.constant 960 : index
    // CHECK-DAG: [[LOOP_END_H:%.+]] = arith.constant 540 : index
    // CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK: [[CAST_INPUT:%.+]] = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs([[INPUT]] : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK: [[LOOP_H:%.+]] = scf.for
    // CHECK-SAME:       [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    // CHECK-SAME:       iter_args([[LOOP_OUT_H:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>) {
    // CHECK:      [[LOOP_W:%.+]] = scf.for
    // CHECK-SAME:       [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    // CHECK-SAME:       iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT_H]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>) {

    // CHECK:        [[IN_OFFSET5_H:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[LOOP_ITER_H]])
    // CHECK:        [[TMP_LOW5_H:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[LOOP_ITER_H]])
    // CHECK:        [[PAD_LOW5_H:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_LOW5_H]]]
    // CHECK:        [[TMP_HIGH5_H:%.+]] = affine.max #[[$PAD_HIGH_STAGE5_H_CANDIDATE_MAP]]([[IN_OFFSET5_H]])
    // CHECK:        [[PAD_HIGH5_H:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_HIGH5_H]]]
    // CHECK:        [[IN_OFFSET5_W:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[LOOP_ITER_W]])
    // CHECK:        [[TMP_LOW5_W:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[LOOP_ITER_W]])
    // CHECK:        [[PAD_LOW5_W:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_LOW5_W]]]
    // CHECK:        [[TMP_HIGH5_W:%.+]] = affine.max #[[$PAD_HIGH_STAGE5_W_CANDIDATE_MAP]]([[IN_OFFSET5_W]])
    // CHECK:        [[PAD_HIGH5_W:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_HIGH5_W]]]

    // CHECK:        [[IN_OFFSET4_H:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[IN_OFFSET5_H]])
    // CHECK:        [[TMP_LOW4_H:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[IN_OFFSET5_H]])
    // CHECK:        [[PAD_LOW4_H:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_LOW4_H]]]
    // CHECK:        [[TMP_HIGH4_H:%.+]] = affine.max #[[$PAD_HIGH_STAGE4_H_CANDIDATE_MAP]]([[IN_OFFSET4_H]], [[PAD_LOW5_H]], [[PAD_HIGH5_H]])
    // CHECK:        [[PAD_HIGH4_H:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_HIGH4_H]]]
    // CHECK:        [[IN_OFFSET4_W:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[IN_OFFSET5_W]])
    // CHECK:        [[TMP_LOW4_W:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[IN_OFFSET5_W]])
    // CHECK:        [[PAD_LOW4_W:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_LOW4_W]]]
    // CHECK:        [[TMP_HIGH4_W:%.+]] = affine.max #[[$PAD_HIGH_STAGE4_W_CANDIDATE_MAP]]([[IN_OFFSET4_W]], [[PAD_LOW5_W]], [[PAD_HIGH5_W]])
    // CHECK:        [[PAD_HIGH4_W:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_HIGH4_W]]]

    // CHECK:        [[IN_OFFSET3_H:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[IN_OFFSET4_H]])
    // CHECK:        [[TMP_LOW3_H:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[IN_OFFSET4_H]])
    // CHECK:        [[PAD_LOW3_H:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_LOW3_H]]]
    // CHECK:        [[TMP_HIGH3_H:%.+]] = affine.max #[[$PAD_HIGH_STAGE3_H_CANDIDATE_MAP]]([[IN_OFFSET3_H]], [[PAD_LOW4_H]], [[PAD_HIGH4_H]], [[PAD_LOW5_H]], [[PAD_HIGH5_H]])
    // CHECK:        [[PAD_HIGH3_H:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_HIGH3_H]]]
    // CHECK:        [[IN_OFFSET3_W:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[IN_OFFSET4_W]])
    // CHECK:        [[TMP_LOW3_W:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[IN_OFFSET4_W]])
    // CHECK:        [[PAD_LOW3_W:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_LOW3_W]]]
    // CHECK:        [[TMP_HIGH3_W:%.+]] = affine.max #[[$PAD_HIGH_STAGE3_W_CANDIDATE_MAP]]([[IN_OFFSET3_W]], [[PAD_LOW4_W]], [[PAD_HIGH4_W]], [[PAD_LOW5_W]], [[PAD_HIGH5_W]])
    // CHECK:        [[PAD_HIGH3_W:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_HIGH3_W]]]

    // CHECK:        [[IN_OFFSET2_H:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[IN_OFFSET3_H]])
    // CHECK:        [[TMP_LOW2_H:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[IN_OFFSET3_H]])
    // CHECK:        [[PAD_LOW2_H:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_LOW2_H]]]
    // CHECK:        [[TMP_HIGH2_H:%.+]] = affine.max #[[$PAD_HIGH_STAGE2_H_CANDIDATE_MAP]]([[IN_OFFSET2_H]], [[PAD_LOW3_H]], [[PAD_HIGH3_H]], [[PAD_LOW4_H]], [[PAD_HIGH4_H]], [[PAD_LOW5_H]], [[PAD_HIGH5_H]])
    // CHECK:        [[PAD_HIGH2_H:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_HIGH2_H]]]
    // CHECK:        [[IN_OFFSET2_W:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[IN_OFFSET3_W]])
    // CHECK:        [[TMP_LOW2_W:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[IN_OFFSET3_W]])
    // CHECK:        [[PAD_LOW2_W:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_LOW2_W]]]
    // CHECK:        [[TMP_HIGH2_W:%.+]] = affine.max #[[$PAD_HIGH_STAGE2_W_CANDIDATE_MAP]]([[IN_OFFSET2_W]], [[PAD_LOW3_W]], [[PAD_HIGH3_W]], [[PAD_LOW4_W]], [[PAD_HIGH4_W]], [[PAD_LOW5_W]], [[PAD_HIGH5_W]])
    // CHECK:        [[PAD_HIGH2_W:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_HIGH2_W]]]

    // CHECK:        [[IN_OFFSET1_H:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[IN_OFFSET2_H]])
    // CHECK:        [[TMP_LOW1_H:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[IN_OFFSET2_H]])
    // CHECK:        [[PAD_LOW1_H:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_LOW1_H]]]
    // CHECK:        [[TMP_HIGH1_H:%.+]] = affine.max #[[$PAD_HIGH_STAGE1_H_CANDIDATE_MAP]]([[IN_OFFSET1_H]], [[PAD_LOW2_H]], [[PAD_HIGH2_H]], [[PAD_LOW3_H]], [[PAD_HIGH3_H]], [[PAD_LOW4_H]], [[PAD_HIGH4_H]], [[PAD_LOW5_H]], [[PAD_HIGH5_H]])
    // CHECK:        [[PAD_HIGH1_H:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_HIGH1_H]]]
    // CHECK:        [[IN_OFFSET1_W:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[IN_OFFSET2_W]])
    // CHECK:        [[TMP_LOW1_W:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[IN_OFFSET2_W]])
    // CHECK:        [[PAD_LOW1_W:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_LOW1_W]]]
    // CHECK:        [[TMP_HIGH1_W:%.+]] = affine.max #[[$PAD_HIGH_STAGE1_W_CANDIDATE_MAP]]([[IN_OFFSET1_W]], [[PAD_LOW2_W]], [[PAD_HIGH2_W]], [[PAD_LOW3_W]], [[PAD_HIGH3_W]], [[PAD_LOW4_W]], [[PAD_HIGH4_W]], [[PAD_LOW5_W]], [[PAD_HIGH5_W]])
    // CHECK:        [[PAD_HIGH1_W:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_HIGH1_W]]]

    // CHECK:        [[SLICE_OFFSET_H:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[IN_OFFSET1_H]])
    // CHECK:        [[TMP_LOW0_H:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[IN_OFFSET1_H]])
    // CHECK:        [[PAD_LOW0_H:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_LOW0_H]]]
    // CHECK:        [[TMP_HIGH0_H:%.+]] = affine.max #[[$PAD_HIGH_STAGE0_H_CANDIDATE_MAP]]([[SLICE_OFFSET_H]], [[PAD_LOW1_H]], [[PAD_HIGH1_H]], [[PAD_LOW2_H]], [[PAD_HIGH2_H]], [[PAD_LOW3_H]], [[PAD_HIGH3_H]], [[PAD_LOW4_H]], [[PAD_HIGH4_H]], [[PAD_LOW5_H]], [[PAD_HIGH5_H]])
    // CHECK:        [[PAD_HIGH0_H:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_HIGH0_H]]]
    // CHECK:        [[SIZE_H:%.+]] = affine.apply #[[$SLICE_SIZE_H_MAP]]([[PAD_LOW0_H]], [[PAD_HIGH0_H]], [[PAD_LOW1_H]], [[PAD_HIGH1_H]], [[PAD_LOW2_H]], [[PAD_HIGH2_H]], [[PAD_LOW3_H]], [[PAD_HIGH3_H]], [[PAD_LOW4_H]], [[PAD_HIGH4_H]], [[PAD_LOW5_H]], [[PAD_HIGH5_H]])
    // CHECK:        [[SLICE_OFFSET_W:%.+]] = affine.max #[[$TO_SLICE_OFFSET_MAP]]([[IN_OFFSET1_W]])
    // CHECK:        [[TMP_LOW0_W:%.+]] = affine.max #[[$TO_PAD_CANDIDATE_MAP]]([[IN_OFFSET1_W]])
    // CHECK:        [[PAD_LOW0_W:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_LOW0_W]]]
    // CHECK:        [[TMP_HIGH0_W:%.+]] = affine.max #[[$PAD_HIGH_STAGE0_W_CANDIDATE_MAP]]([[SLICE_OFFSET_W]], [[PAD_LOW1_W]], [[PAD_HIGH1_W]], [[PAD_LOW2_W]], [[PAD_HIGH2_W]], [[PAD_LOW3_W]], [[PAD_HIGH3_W]], [[PAD_LOW4_W]], [[PAD_HIGH4_W]], [[PAD_LOW5_W]], [[PAD_HIGH5_W]])
    // CHECK:        [[PAD_HIGH0_W:%.+]] = affine.min #[[$PAD_CLAMP_MAP]]()[[[TMP_HIGH0_W]]]
    // CHECK:        [[SIZE_W:%.+]] = affine.apply #[[$SLICE_SIZE_W_MAP]]([[PAD_LOW0_W]], [[PAD_HIGH0_W]], [[PAD_LOW1_W]], [[PAD_HIGH1_W]], [[PAD_LOW2_W]], [[PAD_HIGH2_W]], [[PAD_LOW3_W]], [[PAD_HIGH3_W]], [[PAD_LOW4_W]], [[PAD_HIGH4_W]], [[PAD_LOW5_W]], [[PAD_HIGH5_W]])

    // CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[CAST_INPUT]][0, 0, [[SLICE_OFFSET_H]], [[SLICE_OFFSET_W]]] [1, 32, [[SIZE_H]], [[SIZE_W]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 102, 252]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:        [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_LOW0_H]], [[PAD_LOW0_W]]] high[0, 0, [[PAD_HIGH0_H]], [[PAD_HIGH0_W]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 102, 252]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 102, 252]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // CHECK:        [[PAD1:%.+]] = tensor.pad [[CONV0]] low[0, 0, [[PAD_LOW1_H]], [[PAD_LOW1_W]]] high[0, 0, [[PAD_HIGH1_H]], [[PAD_HIGH1_W]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 250]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 100, 250]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:        [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // CHECK:        [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, [[PAD_LOW2_H]], [[PAD_LOW2_W]]] high[0, 0, [[PAD_HIGH2_H]], [[PAD_HIGH2_W]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 98, 248]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 98, 248]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:        [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[CONV2]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // CHECK:        [[PAD3:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, [[PAD_LOW3_H]], [[PAD_LOW3_W]]] high[0, 0, [[PAD_HIGH3_H]], [[PAD_HIGH3_W]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 96, 246]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 96, 246]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:        [[CONV3:%.+]] = VPU.NCE.Convolution([[PAD3]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // CHECK:        [[PAD4:%.+]] = tensor.pad [[CONV3]] low[0, 0, [[PAD_LOW4_H]], [[PAD_LOW4_W]]] high[0, 0, [[PAD_HIGH4_H]], [[PAD_HIGH4_W]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 94, 244]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 94, 244]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:        [[CONV4:%.+]] = VPU.NCE.Convolution([[PAD4]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // CHECK:        [[PAD5:%.+]] = tensor.pad [[CONV4]] low[0, 0, [[PAD_LOW5_H]], [[PAD_LOW5_W]]] high[0, 0, [[PAD_HIGH5_H]], [[PAD_HIGH5_W]]] {
    // CHECK:          tensor.yield [[PAD_VALUE]] : f16
    // CHECK:          tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 92, 242]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 92, 242]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:        [[CONV5:%.+]] = VPU.NCE.Convolution([[PAD5]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // CHECK:        [[CAST_TILE:%.+]] = tensor.cast [[CONV5]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 90, 240]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x90x240xf16, {order = #NHWC}>
    // CHECK:        [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution([[CAST_TILE]]
    // CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:        [[INSERT:%.+]] = tensor.insert_slice [[DWCONV1]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 32, 90, 240] [1, 1, 1, 1] : tensor<1x32x90x240xf16, {order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK:        scf.yield [[INSERT]] : tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK:      scf.yield [[LOOP_W]] : tensor<1x32x540x960xf16, {order = #NHWC}>
    // CHECK:    [[CAST:%.+]] = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs([[LOOP_H]]
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
// CHECK:        [[SCF:%.+]] = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT1:%.+]] = [[EMPTY]]) -> (tensor<1x16x128x256xf16>)
// CHECK:          {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[OUT:%.+]] = [[OUT1]]) -> (tensor<1x16x128x256xf16>)
// CHECK:            [[SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK:            [[CONVERT:%.+]] = VPU.Convert([[SLICE]]) {dstElemType = f16}
// CHECK:            [[SIN:%.+]] = VPU.Sin([[CONVERT]])
// CHECK:            [[INSERT:%.+]] = tensor.insert_slice [[SIN]] into [[OUT]]
// CHECK:            scf.yield [[INSERT]] : tensor<1x16x128x256xf16>
// CHECK:        return [[SCF]] : tensor<1x16x128x256xf16>
}
